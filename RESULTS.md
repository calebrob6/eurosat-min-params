# EuroSAT minimum-parameter results

See [`FEATURES.html`](FEATURES.html) for an animated visual explanation of every core and experimental feature family.

## Executive summary

A logistic regression using only per-band image statistics does not reach the project's accuracy target. Using 13 bands × {mean, standard deviation, minimum, maximum} gives 52 fixed features; tuning `C` on the validation split selects `C=300` and produces **90.93% validation / 90.96% test accuracy**. The head stores 530 values in scikit-learn's 10-row form, or **477 parameters** in the equivalent reference-class form used by the latest submissions.

The current **>94% frontier** is submission 12: **94.33% test accuracy with 171 parameters**. It uses 18 selected parameter-free spectral and spatial features and a 9-row reference-class linear head. Relative to the image-statistics baseline, it is **+3.37 percentage points more accurate on test while using 64.2% fewer parameters**.

The current source-backed **>95% model** is submission 13: **95.37% test accuracy with 279 parameters**. Relative to the image-statistics baseline, it is **+4.41 percentage points more accurate while using 41.5% fewer parameters**. An experimental richer feature pool reaches **95.57% with 252 parameters**, but that result is not yet packaged as a raw-data-reproducible submission.

The current experimental **>96% frontier** uses 33 fixed features and **306 learned parameters**, reaching **96.17% validation / 96.04% test accuracy** on the default random split. The 33-feature subset is explicit and reproducible from the experiment caches, but it has not yet been promoted to a standalone submission checkpoint.

| Model | Fixed input features used by head | Deployable parameters | Validation | Test | Difference from image-statistics test |
|---|---:|---:|---:|---:|---:|
| Per-band mean/stdev/min/max + tuned logistic regression | 52 | 477 | 0.9093 | 0.9096 | — |
| Submission 12, current >94% frontier | 18 | 171 | 0.9406 | 0.9433 | +3.37 pp |
| Submission 13, current >95% model | 30 | 279 | 0.9531 | 0.9537 | +4.41 pp |
| Mega-pool experimental 95% result | 27 | 252 | 0.9520 | 0.9557 | +4.61 pp |
| Experimental >96% frontier | 33 | 306 | 0.9617 | 0.9604 | +5.08 pp |

## Evaluation protocol and parameter accounting

EuroSAT contains 27,000 13-band Sentinel-2 patches across 10 classes. The fixed project splits contain 16,200 train, 5,400 validation, and 5,400 test images. Models train only on the train split; validation or train-only cross-validation selects hyperparameters and feature subsets; test is the final held-out measurement.

All reported feature extractors are deterministic arithmetic on an input patch and therefore have zero learned parameters. The feature standardizer is folded into the logistic-regression weights, so it adds no deployed values. A 10-class affine softmax head is shift-invariant and can store one class as an implicit zero-logit reference, reducing the exact parameter count from `10 × (F + 1)` to `9 × (F + 1)` without changing any prediction. Historical submissions 01–11 report the 10-row checkpoints they actually stored; submissions 12–13 and the baseline comparison use the tighter reference-class count.

Validation accuracy on 5,400 samples has sampling variation of roughly 0.3 percentage points near these accuracies. Later experiments therefore use multi-seed train cross-validation, disjoint verification folds, and held-out validation together rather than trusting a single noisy threshold crossing.

## Training fractions on random and spatial splits

The fixed model in this comparison is the experimental 306-parameter frontier: a standardized multinomial logistic regression with `C=3`, folded into a 9-row reference-class affine head. It consumes 33 deterministic features: 32 selected columns from the 377-feature mega-pool plus `tail_aniso_low_ndvi`. The selected combination contains 23 core spectral, percentile, multiscale-gradient, orientation, and index-texture features; one Hough-line feature; three Harris-corner features; two local-binary-pattern features; one connected-component feature; one coarse index-texture feature; one coarse orientation-entropy feature; and the low-NDVI region-anisotropy feature. Feature extraction has no learned parameters, so every row below still deploys exactly `9 × (33 + 1) = 306` learned values.

The model was refit on stratified 1%, 2%, 5%, 10%, 20%, 50%, and 100% subsets of both the default random training split and TorchGeo's longitude-based `EuroSATSpatial` training split. Each protocol contains 16,200 train and 5,400 test images. Results are overall test accuracy, meaning the fraction of all 5,400 test images classified correctly. Each entry is the mean and sample standard deviation across subsample seeds 0–4. At 100%, all seeds use the same full training set and the deterministic fit is identical, so the standard deviation is zero.

| Training fraction | Images | Random-split test accuracy | Spatial-split test accuracy |
|---:|---:|---:|---:|
| 1% | 162 | 0.8674 ± 0.0103 | 0.8651 ± 0.0232 |
| 2% | 324 | 0.8936 ± 0.0086 | 0.8742 ± 0.0100 |
| 5% | 810 | 0.9288 ± 0.0031 | 0.8996 ± 0.0133 |
| 10% | 1,620 | 0.9424 ± 0.0026 | 0.9054 ± 0.0083 |
| 20% | 3,240 | 0.9491 ± 0.0017 | 0.9169 ± 0.0035 |
| 50% | 8,100 | 0.9570 ± 0.0011 | 0.9270 ± 0.0027 |
| 100% | 16,200 | **0.9604 ± 0.0000** | **0.9276 ± 0.0000** |

The random-split model exceeds 92% with only 5% of training data and exceeds 94% with 10%. The spatial curve is consistently harder above 1% and largely saturates between 50% and 100%; using all data leaves a 3.28-point gap relative to the random split.

These are post-hoc data-efficiency measurements of an already selected representation, not fresh model-selection experiments at each data budget. In particular, the feature subset and `C` were selected using the full default random train/validation partitions. For the spatial evaluation, the longitude split repartitions the same 27,000 images: 4,362 of its 5,400 test samples appeared in default random train or validation and could therefore have influenced feature selection, although no test labels are used when fitting the fraction-specific heads.

The same five-seed protocol was run for the exact ImageStats baseline: 52 fixed inputs formed by 13 bands × {mean, standard deviation, minimum, maximum}, followed by a standardized reference-class logistic regression with the baseline's fixed validation-selected `C=300`. Its deployed head has `9 × (52 + 1) = 477` learned parameters.

| Training fraction | Images | ImageStats random test accuracy | ImageStats spatial test accuracy |
|---:|---:|---:|---:|
| 1% | 162 | 0.7255 ± 0.0072 | 0.6818 ± 0.0512 |
| 2% | 324 | 0.7695 ± 0.0165 | 0.7180 ± 0.0387 |
| 5% | 810 | 0.8196 ± 0.0086 | 0.7269 ± 0.0316 |
| 10% | 1,620 | 0.8605 ± 0.0024 | 0.7741 ± 0.0282 |
| 20% | 3,240 | 0.8856 ± 0.0019 | 0.8110 ± 0.0175 |
| 50% | 8,100 | 0.9032 ± 0.0021 | 0.8359 ± 0.0122 |
| 100% | 16,200 | **0.9096 ± 0.0000** | **0.8543 ± 0.0000** |

ImageStats is consistently more data-hungry than the selected 33-feature model. At 10% of random-split training data it reaches 86.05%, compared with 94.24% for the 306-parameter model; at 100% the gap remains 5.07 points. The spatial repartition is particularly difficult for global per-band statistics, ending 5.54 points below the random split with all training data. As with the selected-feature curves, `C=300` is held fixed from the full random-split baseline rather than retuned at each fraction.

## Centroid-coordinate MLPs

A separate experiment tests how much of the random-split task can be solved from each patch's geographic location alone. The only source data are the WGS84 latitude and longitude of the GeoTIFF footprint centroid; no pixel values, class statistics, or image-derived features are used.

The sweep covers 220 combinations of fixed coordinate encodings and ReLU MLP shapes. Encodings include raw latitude/longitude, three-dimensional spherical coordinates, polynomial bases through degree four, separable and directional Fourier features with wavelengths from 0.005° to 20°, and fixed Europe-wide Gaussian RBF grids. Architectures include one-hidden-layer widths from 2 to 128 and two-hidden-layer shapes ranging from `4×2` to `128×64`. A standardizer is fit on train and can be folded into the first layer. The 10-class output is counted in lossless reference-class form with nine explicit rows, so every learned weight and bias is included.

The initial sweep uses seed 0 and validation only. Strong candidates across the parameter range are then rerun with seeds 0–4; the multi-seed validation Pareto frontier determines which models are evaluated on test. The table shows representative points from that frontier, with mean and sample standard deviation across the five seeds.

| Parameters | Coordinate encoding | Hidden shape | Validation accuracy | Test accuracy |
|---:|---|---:|---:|---:|
| 33 | Raw latitude/longitude | 2 | 0.2514 ± 0.0328 | 0.2581 ± 0.0336 |
| 85 | Degree-3 polynomial | 4 | 0.3317 ± 0.0085 | 0.3432 ± 0.0093 |
| 201 | Degree-2 polynomial | 8×8 | 0.4260 ± 0.0289 | 0.4215 ± 0.0268 |
| 457 | 11-scale axis Fourier | 8 | 0.4741 ± 0.0123 | 0.4654 ± 0.0133 |
| 969 | 11-scale axis Fourier | 16×8 | 0.5762 ± 0.0050 | 0.5669 ± 0.0044 |
| 1,801 | 11-scale axis Fourier | 32 | 0.6233 ± 0.0071 | 0.6148 ± 0.0053 |
| 3,593 | 4-direction Fourier | 32×16 | 0.6440 ± 0.0061 | 0.6384 ± 0.0062 |
| 5,385 | 11-scale axis Fourier | 64×32 | 0.6564 ± 0.0048 | 0.6558 ± 0.0079 |
| **7,753** | **11-scale axis Fourier** | **64×64** | **0.6616 ± 0.0058** | **0.6603 ± 0.0117** |

The strongest coordinate-only MLP reaches 66.03% test accuracy with 7,753 learned parameters. Wider 128-unit networks, denser directional Fourier bases, and fixed RBF grids do not improve validation accuracy. Geographic centroids therefore contain substantial random-split signal, but are far less accurate and less parameter-efficient than the 33-feature image model, which reaches 96.04% with 306 parameters.

## RESISC45 RGB transfer

The 33-feature handcrafted approach was adapted for a preliminary RESISC45 trial. RESISC45 contains 31,500 RGB images across 45 scene classes with TorchGeo's fixed 18,900/6,300/6,300 train/validation/test split. Images are resized bilinearly from 256×256 to 64×64 before fixed feature extraction.

Because RESISC45 has only RGB channels, the Sentinel-2 feature set cannot be transferred literally. The adapted 147-feature zero-parameter pool preserves the same concepts: per-channel distributions, multiscale gradients, structure-tensor coherence, orientation distributions, Fourier texture, RGB cross-channel correlation, fine and coarse texture of excess-green/color-contrast/saturation/lightness maps, and global Hough-line, Harris-corner, LBP, connected-component, radial-spectrum, and region-shape summaries. Train-only L1 coefficient ranking selects exactly 33 features from this pool.

A normalization audit confirmed that all 33 selected columns are standardized from training statistics to approximately zero mean and unit variance, with no constant features. The apparent preference for `C=3000` came from choosing the exact maximum on a flat validation plateau: `C=30` scores 61.27%, while `C=3000` scores 61.57%, a difference of 19 correct predictions among 6,300 validation images. Applying the repository's one-standard-error heuristic selects `C=30` as the smallest tested value within one standard error of the maximum.

With 45 classes, the 33-feature head stores `(45 - 1) × (33 + 1) = 1,496` learned parameters. The selected full-training model reaches **61.27% validation accuracy and 59.06% test accuracy**.

| Training fraction | Images | Seed | RESISC45 test accuracy |
|---:|---:|---:|---:|
| 1% | 189 | 0 | 0.2892 |
| 2% | 378 | 0 | 0.3443 |
| 5% | 945 | 0 | 0.4516 |
| 10% | 1,890 | 0 | 0.5024 |
| 20% | 3,780 | 0 | 0.5425 |
| 50% | 9,450 | 0 | 0.5700 |
| 100% | 18,900 | 0 | **0.5906** |

This is an initial transfer rather than a RESISC45-optimized frontier. The feature subset is selected once using the full training split, and the full-data validation-selected `C` is then held fixed for every fraction. The lower fractions use one stratified subsample seed as requested. The result shows that the fixed spatial-statistics approach transfers beyond EuroSAT, but RESISC45's 45 fine-grained RGB classes require substantially more discrimination than 33 global summary features provide.

## RESISC45 under a 1,024-parameter budget

The preliminary transfer above spends 1,496 parameters for 59.06% test accuracy. A follow-up experiment asks a harder question: what is the best RESISC45 test accuracy obtainable with at most **1,024 stored values**? With 45 classes a dense reference-class affine head costs `44 x (k + 1)`, so a conventional head can afford only 22 features. Two things change that.

**A larger, higher-resolution zero-parameter pool.** The original 147 columns were extracted from a bilinear 64x64 resize. `experiments/resisc45_cache_full.py` caches the native 256x256 images, and two GPU extractors add 1,432 further deterministic columns: `resisc45_gpu_features.py` (colour moments, fixed HSV/vegetation bin occupancies, multiscale gradient texture, structure-tensor coherence and orientation histograms, Haar subband energies, Fourier ring and sorted-wedge power, direction-averaged co-occurrence, box-counting lacunarity, coarse layout contrasts) and `resisc45_gpu_features2.py` (log-Gabor amplitude statistics, rotation-invariant uniform LBP histograms, normalised-autocorrelation periodicity, projection-profile regularity, percentile-thresholded morphology). The 1,579-column pool with an unconstrained 69,520-parameter head reaches **79.78% test accuracy**, against 70.71% for the original 147-column pool.

**Head structures that spend the budget differently.** Three families were compared at the same 1,024-value budget, all fitted on train with the operating point chosen on validation.

| Head | Features read | Parameters | Validation | Test |
|---|---:|---:|---:|---:|
| Dense affine, top-22 features | 22 | 1,012 | 0.5832 | 0.5700 |
| Dense one-hidden-layer ReLU, width 10 | 53 | 1,024 | 0.6303 | 0.6092 |
| Sparse reference-class head over the top 64 features | 63 | 1,024 | 0.6771 | 0.6579 |
| Sparse reference-class head over the top 128 features | 122 | 1,024 | 0.7116 | 0.6975 |
| Sparse reference-class head over the top 192 features | 171 | 1,024 | **0.7197** | **0.7002** |
| Sparse rank-16 projection over the whole pool | 211 | 1,024 | 0.6954 | 0.6751 |
| Sparse reference-class head over the whole pool | 516 | 1,024 | 0.7071 | 0.7016 |

The dense structured heads are a real improvement over the preliminary transfer: 60.92% test with 1,024 values beats 59.06% with 1,496. The larger jump comes from sparsity. A rank-*r* head fitted on the *whole* pool needs surprisingly little rank -- rank 8 over the 1,207 columns available before the oriented/periodic pool was added already reaches 66.43% -- so the binding constraint is the dense `k x r` projection, not the rank. Pruning that projection, or pruning the affine head directly, lets each class or each compound feature read its own subset of the pool.

**Parameter-accounting caveat.** The sparse rows count only their nonzero values, exactly as the project counts only learned values elsewhere, and the nonzero pattern is treated the same way as the feature-selection indices that every other result in this file already leaves uncounted. The pattern is nonetheless much larger here: 980 (class, feature) index pairs rather than a single 22-to-33 element feature list, so the sparse rows are not directly comparable to the dense EuroSAT frontier numbers. The dense rows in the table are the strictly conservative reading.

Reproduce the whole comparison with:

```bash
python experiments/resisc45_cache_full.py
python experiments/resisc45_gpu_features.py
python experiments/resisc45_gpu_features2.py
python experiments/resisc45_min_params.py
```

Results are written to `experiments/resisc45_min_params_result.csv`, and the pool columns used by the whole-pool sparse head are saved to `experiments/resisc45_sparse_pool_columns.npy`.

## Minimum parameters for 65% and 70% on RESISC45

The section above fixed the budget at 1,024 values and asked which head spends it best. `experiments/resisc45_param_frontier.py` fixes the *targets* instead and walks the budget down over the same 1,579-column pool. Candidate-list size, group-lasso strength, and `C` are chosen on validation; test is read once per row.

| Head | Parameters | Features read | Index pattern | Validation | Test |
|---|---:|---:|---:|---:|---:|
| Dense affine | 1,012 | 22 | 22 | 0.5587 | 0.5416 |
| Dense affine | 1,452 | 32 | 32 | 0.6149 | 0.5965 |
| Dense affine | 2,156 | 48 | 48 | 0.6708 | 0.6640 |
| Dense affine | 2,860 | 64 | 64 | 0.7175 | 0.7051 |
| Dense affine | 5,676 | 128 | 128 | 0.7868 | 0.7670 |
| Sparse reference-class | 384 | 126 | 340 | 0.5797 | 0.5686 |
| Sparse reference-class | 512 | 156 | 468 | 0.6287 | 0.6059 |
| Sparse reference-class | 640 | 88 | 596 | 0.6651 | 0.6435 |
| Sparse reference-class | **768** | 159 | 724 | 0.6881 | **0.6744** |
| Sparse reference-class | 896 | 163 | 852 | 0.7090 | 0.6900 |
| Sparse reference-class | **1,024** | 202 | 980 | 0.7246 | **0.7048** |

Gating on validation, a sparse head first clears 65% at **640** stored values (64.35% test) and 70% at **896** (69.00% test); the smallest budgets whose *test* accuracy also clears the two targets are **768** (67.44%) and **1,024** (70.48%). A dense affine head needs 2,156 values to reach 66.40% and 2,860 to reach 70.51%, so element-wise sparsity is worth roughly a **2.8x parameter reduction** at both targets. This is the opposite of the EuroSAT finding, where element-wise sparsity was about 0.8 points *worse* than feature selection at equal budget; with 45 classes instead of 10 the head dominates the budget, and letting each class pick its own columns is the only way to read a wide pool cheaply.

The 1,024-value row improves on the 70.02% reported in the previous section for two reasons. Once the support is fixed the problem is convex again, so the pruned mask is **refit by LBFGS at a validation-selected `C`** (`refit_masked_ref_logreg_gpu`) instead of being read off the pruning optimiser; iterative magnitude pruning finds a better support than a convex solver does, but a convex solver then places better weights on it. The candidate list is also selected on validation rather than fixed, and 256 columns wins over 192 at this budget.

**How much of the gain survives a compact index pattern?** A sparse head needs one column id per stored weight, which is a much larger deployment artefact than the single feature list every dense result in this file carries. `fit_block_sparse_logreg_gpu` interpolates: the 44 rows are clustered into `g` groups that share one feature list of 22 columns each, so the stored values stay at 1,012 while the pattern shrinks to `22g` ids plus 44 group labels (the labels are vacuous at `g=1`, whose real pattern is the 22 ids alone).

| Shared feature lists | Parameters | Features read | Index pattern | Validation | Test |
|---:|---:|---:|---:|---:|---:|
| 1 (ordinary feature selection) | 1,012 | 22 | 66 | 0.5549 | 0.5540 |
| 4 | 1,012 | 41 | 132 | 0.5973 | 0.5852 |
| 8 | 1,012 | 67 | 220 | 0.5967 | 0.5762 |
| 11 | 1,012 | 80 | 286 | 0.6357 | 0.6244 |
| 22 | 1,012 | 121 | 396 | 0.6367 | 0.6024 |
| 44 (one list per class) | 1,012 | 181 | 1,012 | 0.6983 | 0.6683 |

Sharing does not recover the sparse result: eight shared lists reach 57.62% test and twenty-two reach 60.24%, against 70.48% for the unstructured head at the same 1,012-1,024 values. Even one list per class, which already costs a full-size index pattern, only reaches 66.83%, because it forces every class to spend exactly 22 weights. The accuracy therefore comes from per-class freedom over *both* which columns and how many, not merely from a wider union of columns. There is no cheap middle ground: the dense rows remain the strictly conservative reading.

## Where the RESISC45 budget is lost

`experiments/resisc45_failure_analysis.py` compares the unconstrained 69,520-value head (79.78% test, what the pool can express) with the 1,024-value sparse head (70.43% test in that run) class by class, which separates pool failures from budget failures.

Classes the pool itself cannot separate, with the class each is most often confused for:

| Class | Ceiling accuracy | Budget accuracy | Top confusion |
|---|---:|---:|---|
| palace | 0.500 | 0.293 | church |
| basketball_court | 0.575 | 0.515 | tennis_court |
| tennis_court | 0.576 | 0.528 | medium_residential |
| railway_station | 0.649 | 0.435 | railway |
| church | 0.650 | 0.448 | palace |
| roundabout | 0.679 | 0.530 | intersection |

Classes the pool handles but the budget cannot afford:

| Class | Ceiling accuracy | Budget accuracy | Loss | Top confusion |
|---|---:|---:|---:|---|
| railway_station | 0.649 | 0.435 | 0.214 | railway |
| bridge | 0.850 | 0.643 | 0.207 | river |
| ship | 0.837 | 0.652 | 0.185 | harbor |
| railway | 0.829 | 0.650 | 0.179 | railway_station |
| river | 0.817 | 0.640 | 0.176 | wetland |
| commercial_area | 0.779 | 0.607 | 0.171 | palace |

Both lists are dominated by pairs that differ in *object layout* rather than in texture or colour statistics: a bridge is a river plus one elongated crossing structure, a harbour is water plus repeated docked hulls, a railway station is a railway plus platforms and buildings, and a palace is a church-like facade with different massing. The pool's 171 selected columns are correspondingly texture-heavy - 35 rotation-invariant LBP columns, 13 Fourier-ring columns, 10 multiscale gradient coefficients of variation - and contain nothing that counts or measures discrete elongated objects. Elongated-structure and repeated-object descriptors are therefore the highest-value additions to the pool, and they would help at every budget rather than only at the top.

Reproduce both experiments with:

```bash
python experiments/resisc45_param_frontier.py
python experiments/resisc45_failure_analysis.py
```

Results are written to `experiments/resisc45_param_frontier_result.csv` and `experiments/resisc45_failure_analysis_result.csv`.

## Image-statistics baseline

The baseline computes four statistics independently for each band:

```text
[mean(B01), stdev(B01), min(B01), max(B01), ..., mean(B12), stdev(B12), min(B12), max(B12)]
```

This produces 52 features. A `StandardScaler` is fit on train and folded into a multinomial logistic-regression head. `C` is selected strictly by validation accuracy over a broad logarithmic grid with a denser sweep around the optimum; test accuracy is evaluated only after that choice.

| `C` | Validation accuracy |
|---:|---:|
| 1 | 0.8981 |
| 10 | 0.9067 |
| 100 | 0.9083 |
| **300** | **0.9093** |
| 400 | 0.9091 |
| 1,000 | 0.9091 |
| 2,000 | 0.9089 |

The validation curve has effectively plateaued by `C≈100–300`, so the failure to reach 94% is not an untuned-regularization issue. The selected model scores **0.9096 test accuracy**. Its weakest test classes are Highway (0.7278) and PermanentCrop (0.8420), whereas Forest (0.9819) and SeaLake (0.9967) are already easy from patch-level spectral distributions. This is the central limitation of the baseline: global distribution summaries identify spectrally distinctive classes but discard the spatial structure needed for roads, crop patterns, boundaries, and built environments.

The conventional full 10-row head has `10 × (52 + 1) = 530` parameters. The prediction-identical reference-class form has `9 × (52 + 1) = 477`, which is the count used in fair comparisons with submissions 12 and 13.

## Results progression

The project improved parameter efficiency by first finding stronger parameter-free features, then improving feature selection, then removing redundant head parameters.

| Submission | Stored head | Parameters | Validation | Test | Main change |
|---:|---|---:|---:|---:|---|
| 01 | 10 rows | 1,010 | 0.9426 | 0.9502 | Spectral percentiles plus multiscale gradient texture |
| 02 | 10 rows | 660 | 0.9404 | 0.9431 | Tune feature count and `C` on validation |
| 03 | 10 rows | 510 | 0.9398 | 0.9413 | Structure-tensor coherence and train-only CV |
| 04 | 10 rows | 410 | 0.9406 | 0.9457 | Gradient-orientation entropy/histogram and spectral peaks |
| 05 | 10 rows | 390 | 0.9372 | 0.9461 | Cross-band spatial correlation |
| 06 | 10 rows | 350 | 0.9435 | 0.9437 | Spatial texture of NDVI/NDWI/NDBI/NDMI/NBR/BSI maps |
| 07 | 10 rows | 310 | 0.9428 | 0.9494 | Backward-greedy selection replaces fixed L1 top-k |
| 08 | 10 rows | 290 | 0.9417 | 0.9487 | Commit the lower backward-selection floor |
| 09 | 10 rows | 260 | 0.9404 | 0.9443 | Extend backward elimination to 25 features |
| 10 | 10 rows | 240 | 0.9404 | 0.9472 | Global Hough line feature |
| 11 | 10 rows | 190 | 0.9406 | 0.9433 | Harris corner/junction feature and 18-feature subset |
| 12 | 9 reference rows | **171** | **0.9406** | **0.9433** | Lossless removal of one redundant class row |
| 13 | 9 reference rows | **279** | **0.9531** | **0.9537** | Re-select 30 features for a robust >95% target |

Submissions 03 and 05 were selected by train-only cross-validation and clear the test threshold despite validation being slightly below 0.94. Later submissions require the selector's disjoint verification CV and validation to clear the gate, producing a stronger robustness standard.

Submission 01 already happened to score 95.02% on test, but its validation accuracy was 94.26%, it used 1,010 stored parameters, and it was selected for the original 94% task. Submission 13 is the relevant 95% result because it is selected with a 95.3% validation margin and two independent verification-CV blocks, avoiding the observed case where a model at exactly 95.00% validation scored only 94.94% test.

## What produced the gains

**Spatial information is essential.** Earlier ablations found band means at 74.8% test and mean plus standard deviation at 87.8%. Adding min/max raises that simple family to 90.96%, but it remains more than three points below the original target. Multiscale gradient features first crossed 95%, showing that the missing signal is predominantly spatial texture rather than additional tuning of the classifier.

**Orthogonal fixed features buy more than learned head capacity.** Structure-tensor coherence, orientation entropy, cross-band correlation, spectral-index map gradients, Hough lines, and Harris corners each expose a distinct spatial property. Because these transformations contain no learned values, adding a large candidate pool is free at deployment; only the features retained by the final linear head cost parameters.

**Feature selection became as important as feature design.** L1 coefficient ranking needed 34 selected features for a robust 94% model. Backward-greedy elimination repeatedly refits after each removal, recognizes redundant feature pairs, and reduced the same pool to 25 features. Adding one useful global-line feature then reduced the floor to 23, and one corner feature reduced it to 18.

**The reference-class head is a lossless 10% reduction.** Submission 12 subtracts one class row from all logits and drops the resulting zero row. Submission 11's 190 stored values become 171 with identical predictions on every validation and test image.

**Raising the target from 94% to 95% is relatively cheap.** The robust source-backed model grows from 18 to 30 selected features and from 171 to 279 parameters, an increase of 108 parameters, while test accuracy rises from 94.33% to 95.37%.

## Negative results and ruled-out directions

| Direction | Result |
|---|---|
| Bottleneck MLP or quadratic head features | Increased validation fit without improving held-out test; worse parameter efficiency than a linear head |
| Reduced-rank linear heads and ECOC | All nine discriminant dimensions are load-bearing; reducing below nine collapses accuracy |
| Sequential floating selection | Re-additions occur but do not improve the honest floor over backward elimination and cost substantially more compute |
| Per-subset `C` tuning | `C≈20` slightly improves some fixed subsets, but no `C` makes the next smaller subset pass the validation gate |
| Element-wise sparsity | About 0.8 percentage points worse than column/feature selection at the same parameter budget |
| Single-convolution GAP models | Best tested all-band model reached only 89.2% with 2,058 parameters |
| Distillation into tiny convolutional students | A 98% teacher does not overcome the student's representation bottleneck |
| RGB-only models | Remove the NIR/SWIR information that separates vegetation, crops, water, and built surfaces without reducing linear-head cost |
| MOSAIKS/random convolutional features | Can reach 95%+, but needs roughly 4,617 head parameters at the 95% floor, about 17× submission 13 |

The consistent conclusion is that the classifier is not the bottleneck. Purpose-built, parameter-free spatial summaries deliver far more accuracy per linear-head feature than additional learned capacity or generic random features.

## Development history from the original `main`

This work started from `main` at commit `7f85827`, where submission 11 was the latest packaged model: 190 parameters, 94.06% validation, and 94.33% test.

Commit `5adce7f` added reusable reference-class conversion/prediction helpers and packaged submission 12. This is a lossless 190 → 171 parameter reduction with identical validation and test predictions.

The follow-up work added the 95% frontier analysis, a configurable family list for the 95% backward-selection experiment, submission 13 at 279 parameters / 95.37% test, and the exact image-statistics baseline. These changes are now part of `main`. The 252-parameter mega-pool result remains experimental because its selected LBP and blob feature families have not been promoted into the submission's raw-data feature pipeline.

## Reproduction

```bash
python experiments/image_statistics_baseline.py
python experiments/eval_training_fractions.py --download-spatial-splits
python experiments/eval_imagestats_fractions.py --download-spatial-splits
python experiments/coordinate_mlp.py
python experiments/resisc45_33_feature.py --download
python experiments/resisc45_param_frontier.py
python experiments/resisc45_failure_analysis.py
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

The baseline's complete `C` sweep and per-class test results are stored in `experiments/image_statistics_baseline_result.txt`. The selected-feature and ImageStats five-seed fraction results are stored in `experiments/eval_training_fractions_result.csv` and `experiments/eval_imagestats_fractions_result.csv`, including model metadata, split protocol, individual seed accuracies, mean, and sample standard deviation. The coordinate-only MLP screen and multi-seed frontier are stored in `experiments/coordinate_mlp_screen.csv` and `experiments/coordinate_mlp_result.csv`. The preliminary RESISC45 transfer, including exact selected feature indices and names, is stored in `experiments/resisc45_33_feature_fractions.csv`. Dataset download and fraction evaluators verify the official TorchGeo checksums. Submission evaluation scripts recompute features from raw patches rather than relying on cached feature matrices.
