# "Solving" EuroSAT with 171 parameters

EuroSAT is not a hard dataset to solve with a modern pretrained vision model. The more interesting question is how little model we actually need.

We treated EuroSAT as a parameter-counting game: reach a target test accuracy while storing as few learned numbers as possible. The feature extractor could use fixed arithmetic, but every learned weight and bias in the classifier counted. We started with 94% as the target, reduced the model to 171 learned parameters, then raised the target to 95% and reached it with 279 parameters.

The result is not a new state of the art in accuracy. It is an exercise in model economy. The surprising part was that a linear classifier never became the limiting factor. Better hand-designed measurements of spatial structure consistently beat more complicated learned models at the same parameter budget.

## The short version

| Model | Test accuracy | Learned parameters |
|---|---:|---:|
| Per-band mean, standard deviation, minimum, and maximum | 90.96% | 477 |
| Smallest model above 94% | 94.33% | 171 |
| Source-backed model above 95% | 95.37% | 279 |
| Experimental richer-feature model above 95% | 95.57% | 252 |

The 171-parameter model uses 18 fixed spectral and spatial measurements followed by a 9-row linear classifier. The 279-parameter model uses 30 measurements. In both cases, the feature calculations contain no learned values.

For context, we also ran [`torchgeo/torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) on the exact same EuroSAT train, validation, and test lists. DOFA Large reached 98.33%, and OlmoEarth v1 Large reached 99.02%. Those systems are much more accurate, but they use 337 million and 668 million frozen backbone parameters plus 10,250-parameter linear probes. They answer a different modeling question on the same data: how good are the representations in a large pretrained model, rather than how few learned parameters are needed to classify EuroSAT.

## What we mean by "solving"

The quotation marks matter.

We did not prove that 171 parameters is the mathematical minimum. We found the smallest model in our search that passed a particular accuracy gate under our parameter-counting rules:

- The original target was greater than 94% test accuracy.
- Once that became easy, we raised the target to greater than 95%.
- Fixed image calculations counted as zero learned parameters.
- Learned classifier weights and biases counted.
- The feature-selection indices, source code, and hand-chosen formulas did not count as learned parameters.
- Compute time, memory used while calculating features, and implementation complexity were not the optimization target.

This is the normal machine-learning meaning of parameter count, but it is not the same as executable size or deployment cost. A Hough transform may contain no learned weights and still cost more CPU time than a tiny neural layer. The claim is therefore narrow: these are very small learned models, not necessarily the smallest or fastest possible programs.

## Dataset and modeling setup

We used the 13-band version of EuroSAT: 27,000 Sentinel-2 image patches, each 64 by 64 pixels, divided into 10 land-cover classes:

| Class | Class |
|---|---|
| AnnualCrop | Forest |
| HerbaceousVegetation | Highway |
| Industrial | Pasture |
| PermanentCrop | Residential |
| River | SeaLake |

The fixed split files contain 16,200 training images, 5,400 validation images, and 5,400 test images. We kept the native 64 by 64 resolution and used all 13 spectral bands.

Training and model selection followed three levels of evidence:

1. Fit model weights on the training split.
2. Use validation accuracy or repeated cross-validation inside the training split to choose regularization, feature families, and feature count.
3. Report test accuracy as the final held-out measurement.

As the search became more aggressive, we added a second layer of protection against feature-selection overfitting. One set of cross-validation splits guided backward feature elimination, while two disjoint sets of splits checked the selected subsets. The separate validation split was an additional gate.

There is an important caveat. The experiment scripts often printed test accuracy for diagnostic tables even when test did not control the selection rule. We did not choose the committed subsets by maximizing test accuracy, but we did repeatedly observe the test set during the research process. These numbers should be read as a careful experimental study, not as a pristine one-shot leaderboard submission.

## The obvious baseline: four numbers per band

The simplest reasonable model calculates four statistics for each spectral band:

```text
mean, standard deviation, minimum, maximum
```

With 13 bands, that produces 52 inputs. We fit multinomial logistic regression after standardizing those inputs. The regularization parameter `C` was swept from `0.0001` to `2000`, selected only by validation accuracy, and then evaluated on test.

The validation curve flattened at large `C`:

| C | Validation accuracy |
|---:|---:|
| 1 | 89.81% |
| 10 | 90.67% |
| 100 | 90.83% |
| 300 | **90.93%** |
| 400 | 90.91% |
| 1,000 | 90.91% |
| 2,000 | 90.89% |

The selected model reached **90.96% test accuracy**.

This baseline is already very good at spectrally distinctive classes. SeaLake reached 99.67% and Forest reached 98.19%. It struggled with classes where spatial arrangement matters: Highway reached only 72.78%, and PermanentCrop reached 84.20%.

That failure shaped the rest of the project. The model did not need a more expressive classifier yet. It needed measurements that could distinguish a road from a crop boundary, a regular field from mixed vegetation, and a street grid from parallel crop rows.

## How 52 inputs become 477 parameters

A conventional 10-class linear classifier on 52 inputs stores 10 weights per input and 10 biases:

```text
10 x (52 + 1) = 530 parameters
```

The latest models use a standard reference-class form of multinomial logistic regression. Adding or subtracting the same value from every class score changes neither the winning class nor the softmax probabilities. We can subtract one class's complete score from all 10 scores, make that class an implicit zero-score reference, and store only the remaining nine rows:

```text
9 x (52 + 1) = 477 parameters
```

This is not an approximation. It produces exactly the same predictions.

We apply the same idea to feature standardization. If the classifier is trained on standardized features,

```text
z = (x - mean) / scale
logits = W z + b
```

then the standardizer can be folded into the classifier:

```text
logits = (W / scale) x + (b - W mean / scale)
```

The deployed model is therefore one affine map on the raw fixed features. It does not need to store a separate standardizer.

## The feature extractor became the model

The full candidate pool grew to hundreds of measurements, but only the selected measurements entered the classifier and cost learned parameters. Each new family was intended to capture a physical or geometric property missing from the existing pool.

### Spectral summaries

We began with per-band means, standard deviations, and intensity percentiles. Means describe overall reflectance, standard deviations provide a rough measure of texture, and percentiles preserve more of the shape of each band's distribution than minimum and maximum alone.

These measurements are especially useful for water and vegetation because Sentinel-2 includes near-infrared and short-wave infrared bands. A model limited to RGB discards much of the information that makes EuroSAT easy.

### Multiscale gradient texture

For each band, we calculated gradient magnitude and summarized it at several pooling scales. This asks how much local change is present at fine, medium, and coarse scales.

This was the first decisive improvement. Earlier ablations reached 74.8% with band means, 87.8% with means and standard deviations, and about 92.6% after adding one scale of gradient statistics. Multiscale gradients pushed the fixed-feature linear model past 95%.

### Structure-tensor coherence

Gradient magnitude says how strong edges are, but not whether they point in a consistent direction. We added structure-tensor coherence:

```text
coherence = sqrt((Sxx - Syy)^2 + 4 Sxy^2) / (Sxx + Syy)
```

Coherence approaches one when local gradients share a dominant direction and approaches zero when the texture is isotropic. Roads and organized crop rows tend to be directional; forest and irregular fields are less so.

Adding coherence reduced the 94% feature-count floor from 65 selected inputs to 50.

### Orientation entropy and orientation histograms

Coherence is a second-moment summary and can be dominated by one strong edge. We therefore built a magnitude-weighted histogram of gradient directions and measured its entropy.

Low entropy means most edges point in one or two directions. High entropy means the directions are spread out. We also exposed a small orientation histogram directly and measured the strength of periodic peaks in the two-dimensional Fourier spectrum.

These features reduced the floor from 50 inputs to 40.

### Cross-band spatial correlation

Most features so far treated each spectral band independently. Cross-band correlation asks whether spatial patterns in two bands rise and fall together across the image.

Vegetation couples red and near-infrared structure differently from water or built surfaces. A few correlations between red, green, blue, near-infrared, and short-wave infrared bands gave the classifier information that no per-band statistic contained.

This reduced the floor from 40 inputs to 38.

### Texture inside spectral-index maps

A patch-level NDVI value throws away spatial arrangement. We instead calculated per-pixel maps for NDVI, NDWI, NDBI, NDMI, NBR, and BSI, then summarized their standard deviation, gradient magnitude, gradient variation, and percentile spread.

These features describe boundaries between materials rather than boundaries inside one raw band. A road cutting through vegetation creates a sharp NDVI edge. A uniform crop field and mixed herbaceous vegetation may have similar average NDVI but very different within-patch variation.

Index-map texture was the strongest single feature-family improvement in the early search and reduced the floor from 38 inputs to 34.

### Global straight lines

All of the direction measurements above are local. Many short parallel crop edges and one long road can have similar local orientation statistics.

We added Hough-transform summaries on panchromatic, NDVI, and NDBI edge maps. The most useful measurement was the fraction of edge evidence explained by the three strongest lines in the NDVI map. It distinguishes global collinearity from merely local direction.

One selected line feature reduced the floor from 25 inputs to 23.

### Corners and junctions

Lines are not enough to describe built environments. Residential and industrial areas contain intersections and grid-like junctions that do not appear in parallel crop rows.

We added Harris corner statistics on panchromatic, NDVI, and NDBI channels. A single corner-density measurement helped reduce the 94% model from 23 inputs to 18.

## Selecting a small subset without fooling ourselves

The first submissions ranked features once using an L1-regularized logistic regression and kept the top `k`. That was fast, but coefficient magnitude is only a proxy for the best subset. Two highly ranked features can carry nearly identical information.

We switched to backward-greedy elimination:

1. Start with a comfortably large candidate subset.
2. Temporarily remove each remaining feature.
3. Refit logistic regression and measure repeated training-set cross-validation.
4. Permanently remove the feature whose removal hurts least.
5. Repeat until the desired size is reached.

The cross-validation used to choose removals is optimistic because the algorithm is explicitly optimizing it. We therefore checked each low-parameter subset on separate cross-validation seeds that never guided selection. The final subset also had to pass validation.

This distinction mattered. On the same 305-feature pool, fixed L1 ranking needed 34 selected inputs to pass the 94% gate. Backward elimination found a 30-input model with higher test accuracy, and extending the same path eventually reached 25 inputs.

We also learned not to stop the search early. One submission declared 28 inputs as the floor simply because the elimination run stopped there. Resuming the deterministic path revealed that 25 inputs still passed.

## The parameter frontier

The table below reports the checkpoints as stored at the time. Submissions 1 through 11 stored all 10 class rows. Submission 12 introduced the smaller 9-row reference-class form.

| Submission | Parameters | Validation | Test | Main change |
|---:|---:|---:|---:|---|
| 01 | 1,010 | 94.26% | 95.02% | Spectral percentiles and multiscale gradients |
| 02 | 660 | 94.04% | 94.31% | Tune feature count and `C` |
| 03 | 510 | 93.98% | 94.13% | Add structure-tensor coherence |
| 04 | 410 | 94.06% | 94.57% | Add orientation entropy, histograms, and spectral peaks |
| 05 | 390 | 93.72% | 94.61% | Add cross-band spatial correlation |
| 06 | 350 | 94.35% | 94.37% | Add spectral-index map texture |
| 07 | 310 | 94.28% | 94.94% | Replace fixed ranking with backward elimination |
| 08 | 290 | 94.17% | 94.87% | Commit a smaller point on the same path |
| 09 | 260 | 94.04% | 94.43% | Extend elimination to 25 inputs |
| 10 | 240 | 94.04% | 94.72% | Add a global line measurement |
| 11 | 190 | 94.06% | 94.33% | Add a corner measurement and reach 18 inputs |
| 12 | **171** | **94.06%** | **94.33%** | Remove one redundant class row with no prediction change |
| 13 | **279** | **95.31%** | **95.37%** | Re-select 30 inputs for a robust 95% target |

Submissions 3 and 5 were selected by training-only cross-validation and passed the test threshold even though validation was slightly below 94%. Later submissions used the stricter rule that both independent cross-validation checks and validation had to pass.

## What is actually inside the 171-parameter model?

The 94% model uses 18 selected measurements:

| Family | Count | Examples |
|---|---:|---|
| Spectral percentiles | 5 | Median and upper or lower percentiles from red-edge, NIR, water-vapour, and SWIR bands |
| Raw-band gradient magnitude | 3 | Fine-scale blue, green gradients and a coarse green gradient |
| Orientation entropy | 1 | Directional complexity in a red-edge band |
| Cross-band spatial correlation | 2 | Red/NIR and blue/NIR spatial correlation |
| Spectral-index texture | 5 | NDVI, NDWI, NBR, and BSI gradient statistics |
| Global line structure | 1 | Top-three NDVI Hough-line strength |
| Corner structure | 1 | NDVI corner fraction |

The classifier stores nine rows, each with 18 weights and one bias:

```text
9 x (18 + 1) = 171 parameters
```

The model reaches 94.06% validation accuracy and 94.33% test accuracy. Its predictions are bit-identical to the earlier 190-parameter 10-row checkpoint.

The most important observation is the breadth of the selected set. The final 18 measurements are not 18 versions of the same texture statistic. They cover intensity, gradients, direction, band relationships, material-index boundaries, global lines, and junctions. Backward elimination spends the tiny feature budget across complementary axes.

## Raising the bar to 95%

The original prompt suggested raising the target once a simple method became too successful. We therefore changed the task from greater than 94% test accuracy to greater than 95%.

At this accuracy level, accepting the first validation result above 95% was unsafe. With 5,400 validation images and accuracy near 95%, the binomial standard deviation is approximately:

```text
sqrt(0.95 x 0.05 / 5400) = 0.003
```

That is about 0.3 percentage points. We observed the failure directly: a 28-input model reached exactly 95.00% validation accuracy but only 94.94% test accuracy.

The 95% submission therefore required:

- Validation accuracy of at least 95.3%.
- Two disjoint repeated-cross-validation checks above 95%.
- No test-driven subset choice.

The selected model uses 30 inputs and a 9-row reference-class head:

```text
9 x (30 + 1) = 279 parameters
```

It reached 95.31% validation and 95.37% test accuracy. The selected set contains 27 measurements from the main spectral and texture pool and three Harris corner measurements. The line family was available in the candidate pool but did not survive this higher-accuracy selection.

An experimental 27-input subset from a richer pool reached 95.57% test accuracy with 252 parameters. It includes one local-binary-pattern feature and one blob-size feature that still live in experiment code rather than the normal raw-image feature pipeline, so we do not treat it as a finished submission yet.

Moving from the robust 94% model to the robust 95% model costs only 108 learned parameters.

## What did not work

The negative results were as informative as the winning features.

### A small multilayer perceptron

We replaced logistic regression with a bottleneck neural network on the same fixed features. Even the wider configurations underperformed the linear model at similar parameter counts. The hand-designed features were already close to linearly separable, so compressing them through a small hidden layer destroyed useful information.

### Quadratic feature interactions

Squares and pairwise products raised validation accuracy but did not improve test accuracy. This was a straightforward case of using validation flexibility to fit noise.

### Reduced-rank class heads

The 10-class softmax has at most nine identifiable class-separation directions. We tested dropping below nine with matrix factorization and error-correcting output codes. Accuracy collapsed as soon as one of those directions was removed. The exact 9-row reference-class form is useful; a lower-rank approximation is not.

### Sequential floating feature selection

Allowing previously removed features to re-enter the subset changed the path but did not improve the honest accuracy floor. It cost two to three times as much compute as simple backward elimination.

### Tuning C for every subset size

Changing regularization slightly improved some fixed subsets, but it never made the next smaller subset pass validation on both selection partitions. Regularization was not the reason the feature floor existed.

### Element-wise weight sparsity

Instead of selecting whole features, we tried pruning individual class-feature weights. At the same 260-parameter budget, the dense model on 25 selected features reached 94.52% repeated cross-validation, while the best element-wise sparse model reached about 93.75%.

Feature sharing explains the result. A useful input such as NDVI gradient strength helps several classes at once. Paying for all class weights on a strong shared feature is more economical than spreading sparse weights across many weaker features.

### Tiny convolutional models

A single convolution followed by global average pooling was too weak. The best all-band configuration reached 89.2% with 2,058 parameters. The global average discarded the rich spatial summaries that the fixed features calculated explicitly.

### Knowledge distillation

A roughly 290,000-parameter teacher reached about 98%, but distilling its predictions into the tiny convolutional students did not help. Better supervision cannot compensate for a representation that cannot express the required structure.

### RGB-only models

RGB reduces convolutional input weights, but it removes NIR and SWIR bands that separate water, vegetation, crops, and built surfaces. For the fixed-feature linear model, dropping bands does not even reduce classifier parameters because the parameter count depends on selected feature count, not the number of source bands.

### Random convolutional features and MOSAIKS

Random convolutional features can reach good accuracy, but each individual feature is weak. Hundreds are needed.

In our experiments, a selected random-feature model needed roughly 2,500 to 5,000 linear-head parameters to pass 94%, and around 4,617 parameters to pass 95%. At the 190-parameter budget of submission 11, random features reached only about 82%.

Purpose-built measurements are dramatically more efficient in this regime because each one encodes a specific high-level property.

## How this compares with torchgeo-bench on the exact same EuroSAT split

[`torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) is a lightweight benchmark for evaluating frozen geospatial foundation models. Its [methodology](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/docs/user/methodology.rst) freezes a pretrained backbone, extracts one embedding per image, and evaluates either 5-nearest-neighbor classification or a linear logistic-regression probe.

The repository did not contain results for its plain `eurosat` wrapper, so we computed them at commit [`c953919`](https://github.com/torchgeo/torchgeo-bench/tree/c95391940d58358dd4c522c31703ee12ac030357). Every result below uses the same split files as our experiments:

| Split | Images | SHA-256 |
|---|---:|---|
| Train | 16,200 | `1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695` |
| Validation | 5,400 | `b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330` |
| Test | 5,400 | `cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592` |

We also set `eval.merge_val=false`, which changes the benchmark's default behavior so that validation selects `C` but the final linear probe is still fit on the 16,200 training images only. That matches our train/validation separation. The benchmark results use seed 0 and 200 bootstrap resamples for confidence intervals.

The computed CSVs and exact commands are stored in `experiments/torchgeo_bench_eurosat/`.

### The same 52-feature baseline agrees within 0.31 percentage points

The benchmark's [`ImageStatsBench`](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/src/torchgeo_bench/models/image_stats.py) calculates per-channel mean, population standard deviation, maximum, and minimum. This is the same 52-feature family as our baseline, apart from feature order.

| Implementation | Bands | Input size | Validation-selected C | Test accuracy | 95% bootstrap interval | Parameters |
|---|---:|---:|---:|---:|---:|---:|
| Our scikit-learn baseline | 13 | 64 | 300 after feature standardization | **90.96%** | 90.24-91.65% | 477 reference-class, 530 full-row |
| torchgeo-bench ImageStats | 13 | 64 | 0.000562 on raw features | **90.65%** | 89.91-91.43% | 530 full-row |

The confidence intervals overlap heavily. The 0.31-point difference is small enough to attribute to implementation details such as feature standardization, the logistic-regression solver, and the exact `C` grid. The absolute `C` values are not comparable because standardizing features changes the regularization scale.

The ImageStats benchmark used 13 log-spaced `C` values from `1e-5` to `1e-2` after an initial broad run located the useful raw-feature scale. The selected value was inside that range rather than on an edge. DOFA and OlmoEarth used the benchmark's default 40-value sweep from `1e-6` to `1e4`.

This exact comparison supports the original diagnosis: four global statistics per band are a strong 91% baseline, but they do not contain enough spatial information to reach 94%.

### Exact frozen-backbone results

We then ran two strong geospatial foundation models through the same benchmark harness and the same EuroSAT split.

| System | Bands used | Input size | Test accuracy | 95% bootstrap interval | Representation | Learned parameters used for prediction |
|---|---:|---:|---:|---:|---:|---:|
| Our 95% model | 13 | 64 | **95.37%** | 94.78-95.85% | 30 selected fixed features | **279** |
| DOFA Large + linear probe | 13 | 224 | **98.33%** | 98.02-98.65% | 1,024-dimensional embedding | 337.15M frozen backbone + 10,250 probe |
| OlmoEarth v1 Large + linear probe | 12 | 64 | **99.02%** | 98.78-99.26% | 1,024-dimensional embedding | 668.05M frozen backbone + 10,250 probe |

DOFA uses all 13 EuroSAT bands and resizes the patches to the 224-pixel input expected by the model. OlmoEarth runs at the native 64-pixel size, but its Sentinel-2 input layout has no B10 cirrus slot, so the torchgeo-bench wrapper skips B10 and uses the other 12 bands. These are model-specific input differences, not dataset or split differences.

Both frozen backbones produce 1,024 features. A conventional 10-class probe therefore stores:

```text
10 x (1024 + 1) = 10,250 parameters
```

The backbone sizes come from the benchmark's [`compute_cost.csv`](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/results/compute_cost.csv): 337,151,533 parameters for DOFA Large and 668,045,312 for OlmoEarth v1 Large.

On the exact test set, DOFA is 2.96 percentage points more accurate than our 279-parameter model, and OlmoEarth is 3.65 points more accurate. Their bootstrap intervals do not overlap ours, so this is a real accuracy gap rather than ordinary test-sample noise.

There are two useful parameter comparisons:

- Counting the complete prediction system, DOFA is about 1.21 million times larger and OlmoEarth is about 2.39 million times larger than our 279-parameter classifier.
- Counting only newly fit task-specific parameters, either 10,250-parameter probe is 36.7 times larger than our classifier.

The second comparison intentionally treats the pretrained backbone as given. The first counts every learned value used at prediction time. Neither captures the full engineering tradeoff because our fixed feature extractor contains human-designed code and the foundation models contain learned knowledge from external pretraining.

### KNN shows how much information is already in the frozen representations

The benchmark also evaluates 5-nearest-neighbor classification with no learned probe:

| Features | KNN-5 test accuracy |
|---|---:|
| Four image statistics per band | 84.67% |
| DOFA Large embedding | 95.74% |
| OlmoEarth v1 Large embedding | 98.24% |

KNN does not learn classifier weights, but it still requires the full frozen backbone and a searchable database of training embeddings. It is therefore a useful representation-quality result, not a zero-storage model.

### What is and is not apples-to-apples

The dataset comparison is now exact: all systems use the same 16,200 training images, 5,400 validation images, and 5,400 test images, and no system merges validation into final training.

The modeling setups intentionally remain different:

| Question | This project | torchgeo-bench frozen models |
|---|---|---|
| Main objective | Minimize learned parameters while passing an accuracy threshold | Measure the quality of a pretrained representation |
| Feature extractor | Fixed hand-designed arithmetic | Frozen DOFA or OlmoEarth backbone |
| Classifier | Logistic regression on a selected feature subset | Logistic regression on all 1,024 embedding dimensions |
| External pretraining | None | Large-scale external pretraining |
| Parameter accounting | Complete learned task model | Backbone and probe reported separately |

The fair conclusion is not that one approach universally wins. The foundation models buy another three to four accuracy points and reach approximately 99%. The hand-designed model gives up those points in exchange for reducing the learned task model from hundreds of millions of values to 279.

## What we learned

The project started with a question about tiny neural networks and ended with a lesson about representation.

A 52-input summary-statistics model was not limited by logistic regression. It was limited by the information discarded before logistic regression saw the image. Once we calculated physically meaningful spatial summaries, a linear decision boundary was enough.

The best features were not dozens of minor variations on intensity. They were measurements of distinct concepts:

- How strong is local texture?
- Do edges point in one direction or many?
- Do red and near-infrared patterns move together?
- Where do material-index boundaries occur?
- Is there one long line or many short edges?
- Do those edges meet in corners and junctions?

At very small parameter counts, one strong measurement is worth more than many generic projections.

The result also exposes the ambiguity in "small model." Our 171-parameter classifier is tiny in learned state, but it relies on a relatively elaborate fixed program. OlmoEarth is enormous in stored learned state, but its generic representation reaches substantially higher accuracy and transfers across many datasets. Neither dominates every deployment objective.

If the goal is maximum EuroSAT accuracy, use a strong pretrained model. If the goal is to understand how little learned state is needed once domain knowledge is allowed into the feature extractor, 171 parameters can take us surprisingly far.

## Next steps

The most immediate next step is to turn the experimental 252-parameter 95% result into a complete submission by promoting the selected local-binary-pattern and blob features into the reusable raw-image pipeline.

A stronger evaluation would also:

- Add bootstrap confidence intervals to our test results.
- Contribute the exact plain-EuroSAT benchmark rows upstream so this comparison does not need to be recomputed locally.
- Compare wall-clock feature extraction, FLOPs, memory, and executable size in addition to learned parameter count.
- Freeze the test set completely for a fresh rerun after all methodology choices are locked.
- Test whether the same 18- and 30-feature subsets transfer to a spatial EuroSAT split.

## Reproducing the local results

```bash
python experiments/image_statistics_baseline.py
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

The baseline sweep is stored in `experiments/image_statistics_baseline_result.txt`. Submission evaluation scripts rebuild their features from raw patches rather than using the precomputed feature matrices.

## Sources

- [`torchgeo/torchgeo-bench` README at commit c953919](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/README.md)
- [torchgeo-bench evaluation methodology](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/docs/user/methodology.rst)
- [torchgeo-bench EuroSAT and EuroSATSpatial definitions](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/src/torchgeo_bench/datasets/eurosat.py)
- [torchgeo-bench image-statistics implementation](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/src/torchgeo_bench/models/image_stats.py)
- [torchgeo-bench OlmoEarth v1 Large configuration](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/src/torchgeo_bench/conf/model/olmoearth_large.yaml)
- [torchgeo-bench DOFA Large configuration](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/src/torchgeo_bench/conf/model/torchgeo/dofa_large.yaml)
- [torchgeo-bench model compute and parameter profiles](https://github.com/torchgeo/torchgeo-bench/blob/c95391940d58358dd4c522c31703ee12ac030357/results/compute_cost.csv)
- Local exact-split benchmark outputs and commands: `experiments/torchgeo_bench_eurosat/`
