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

Gating on validation, a sparse head first clears 65% at **640** stored values (64.35% test) and 70% at **896** (69.00% test); the smallest budgets whose *test* accuracy also clears the two targets are **768** (67.44%) and **1,024** (70.48%). A dense affine head needs 2,156 values to reach 66.40% and 2,860 to reach 70.51%, so element-wise sparsity is worth roughly a **2.8x parameter reduction** at both targets. Three later sections lower the sparse numbers again: an object-layout quota takes 65% to 640 values, a prune-and-regrow support search takes 65% to 512 and 70% to 896, and generating the head from a fixed over-complete class dictionary takes 65% to 256 and 70% to 448. This is the opposite of the EuroSAT finding, where element-wise sparsity was about 0.8 points *worse* than feature selection at equal budget; with 45 classes instead of 10 the head dominates the budget, and letting each class pick its own columns is the only way to read a wide pool cheaply.

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

Both lists are dominated by pairs that differ in *object layout* rather than in texture or colour statistics: a bridge is a river plus one elongated crossing structure, a harbour is water plus repeated docked hulls, a railway station is a railway plus platforms and buildings, and a palace is a church-like facade with different massing. The pool's 171 selected columns are correspondingly texture-heavy - 35 rotation-invariant LBP columns, 13 Fourier-ring columns, 10 multiscale gradient coefficients of variation - and contain nothing that counts or measures discrete elongated objects. Elongated-structure and repeated-object descriptors are therefore the highest-value additions to the pool. The next section builds them and tests that prediction; it holds, but only under a quota, and for a different reason than expected.

Reproduce both experiments with:

```bash
python experiments/resisc45_param_frontier.py
python experiments/resisc45_failure_analysis.py
```

Results are written to `experiments/resisc45_param_frontier_result.csv` and `experiments/resisc45_failure_analysis_result.csv`.

## An object-layout pool, and why it only pays under a quota

The section above predicted that elongated-structure and repeated-object descriptors were the highest-value addition to the RESISC45 pool. `experiments/resisc45_gpu_features3.py` adds 505 such columns, extracted from the same native 256x256 cache in about seven GPU-minutes and still deterministic arithmetic on a single image:

* **radon** -- projection profiles at twelve fixed angles, so one long straight structure becomes a single sharp profile peak and a set of parallel structures becomes a periodic profile: angular energy shares, two-fold and four-fold angular harmonics, best-angle peak height, supra-threshold peak count, and profile periodicity;
* **ridge** -- Hessian ridge (vesselness) strength, linearity, sign, and orientation coherence at three fixed scales;
* **run** -- directional run lengths of thresholded ridge, edge, and dark masks in four directions, which measure how far a thin structure actually continues;
* **blob** -- difference-of-Gaussian local-maximum counts at three scales plus the second moments of the maximum cloud, which count repeated discrete objects and measure whether they lie along a line;
* **polar** -- rotational and mirror self-similarity, log-polar radial bands, and angular harmonics, which separate a ring layout from a radial-arm layout;
* **mask** -- second moments, border spanning, and profile breaks of percentile-thresholded regions.

**Merging the pools and re-ranking makes the frontier worse.** `experiments/resisc45_layout_gain.py` re-runs the sparse frontier on the 1,579-column pool and on the merged 2,084-column pool under identical settings. The group-lasso ranking likes the new family -- 113 of the merged top 256 columns come from it -- and the result is *worse on validation at every budget*, which is the criterion that governs selection.

| Parameters | Base validation | Merged validation | Base test | Merged test |
|---:|---:|---:|---:|---:|
| 512 | 0.6287 | 0.6160 | 0.6059 | 0.6008 |
| 640 | 0.6651 | 0.6613 | 0.6435 | 0.6508 |
| 768 | 0.6881 | 0.6784 | 0.6744 | 0.6576 |
| 896 | 0.7090 | 0.7000 | 0.6900 | 0.6886 |
| 1,024 | 0.7246 | 0.7140 | 0.7048 | 0.6940 |

Per class the intended effect is clearly present and clearly paid for. At 1,024 values the merged pool gains on exactly the classes the failure analysis named -- thermal_power_station +5.9, rectangular_farmland +5.2, ship +5.2, wetland +5.1, bridge +5.0 points -- and loses palace -12.9, storage_tank -8.8, harbor -7.7, desert -5.5, airplane -5.3. At a fixed budget the head reallocates weights to the new columns for the classes that want them and starves the rest.

**The diagnosis is redundancy, not weakness.** `experiments/resisc45_layout_diagnose.py` separates the two candidate explanations. The layout family is genuinely informative on its own: a 505-column unconstrained head reaches **65.21% test**, clearing the first target from the new family alone -- at 22,264 stored values, so this is a statement about information content rather than about the budget. But it adds almost nothing the old pool did not already have -- the unconstrained ceiling moves from 79.78% test on 1,579 columns to 79.37% on 2,084. The log-Gabor, autocorrelation, projection-profile, and LBP families already encode most of what oriented and periodic structure there is to encode. What the new family supplies is not new information but a *better-conditioned* small subset of it.

**A quota converts that into accuracy.** Instead of letting the two pools compete for one ranking, reserve `q` of the 256 candidate slots for the layout pool and give the rest to the base ranking. `experiments/resisc45_layout_frontier.py` sweeps `q` against the budget; `q` and `C` are chosen on validation and test is read once per row.

| Parameters | q=0 | q=32 | q=64 | q=128 | Selected q | Validation | Test |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.6059 | 0.6232 | 0.6290 | 0.6114 | 64 | 0.6387 | 0.6290 |
| 640 | 0.6421 | 0.6579 | 0.6554 | 0.6465 | 64 | 0.6749 | **0.6554** |
| 768 | 0.6740 | 0.6819 | 0.6794 | 0.6679 | 64 | 0.7008 | 0.6794 |
| 896 | 0.6860 | 0.6879 | 0.6967 | 0.6921 | 32 | 0.7127 | 0.6879 |
| 1,024 | 0.7048 | 0.7030 | **0.7167** | 0.7098 | 64 | 0.7305 | **0.7167** |

(the `q` columns are test accuracy at that quota; the last three columns are the validation-selected operating point. This experiment fixes the candidate list at 256 columns, so the `q=0` column is not identical to the frontier table above, which also swept the candidate-list size.)

Two frontier movements follow. The smallest budget whose test accuracy clears **65%** falls from 768 to **640 stored values** (65.54%), and the 1,024-value operating point improves from 70.48% to **71.67% test**. 70% still needs the full 1,024 values on test, although validation already clears it at 768. The next section moves both again by a margin this one cannot reach, and it does so by changing how the support is searched rather than what is in the pool. About a fifth of the columns the budgeted head ends up reading are layout columns (44 of 207 at 1,024 values).

**How much of that is the quota and how much is the family?** Widening a candidate list at all is worth something, so the same experiment runs a control that gives the same 64 displaced slots to the *next* 64 base columns by group-lasso rank instead of to layout columns.

| Parameters | q=0 | Control (64 more base columns) | 64 layout columns |
|---:|---:|---:|---:|
| 512 | 0.6059 | 0.6183 | 0.6290 |
| 640 | 0.6421 | 0.6483 | 0.6554 |
| 768 | 0.6740 | 0.6746 | 0.6794 |
| 896 | 0.6860 | 0.6875 | 0.6967 |
| 1,024 | 0.7048 | 0.7106 | 0.7167 |

About half the gain is candidate-list churn that any 64 extra columns would deliver, and the layout family beats the control on test at all five budgets by 0.5 to 1.1 points. The family-specific contribution is real but modest, and it is only reachable when the family is quota-limited rather than allowed to win the ranking on its own merits.

Reproduce the whole sequence with:

```bash
python experiments/resisc45_gpu_features3.py
python experiments/resisc45_layout_gain.py
python experiments/resisc45_layout_diagnose.py
python experiments/resisc45_layout_frontier.py
```

Results are written to `experiments/resisc45_layout_gain_result.csv` (with per-class deltas in `experiments/resisc45_layout_gain_classes.csv`), `experiments/resisc45_layout_diagnose_result.csv`, and `experiments/resisc45_layout_frontier_result.csv`. `resisc45_layout_frontier.py --summarise` re-prints the frontier summary from the existing CSV.

## Prune-and-regrow finds a much better support than prune-only

Every sparse head above chooses its support with `fit_sparse_logreg_gpu`, iterative magnitude pruning, which can only ever *remove* weights. A column the pruner drops early is gone for good, so the whole result depends on the candidate list handed to it up front -- and that list comes from an L2,1 group-lasso ranking that scores each column against the label and never against the columns already chosen. The quota of the previous section is a hand-tuned patch over exactly that weakness.

`fit_rigl_ref_logreg_gpu` removes the need for the patch. Following RigL, it holds the active-weight count at the budget for the whole run and periodically **drops the smallest active weights and regrows the same number of inactive entries with the largest dense loss gradient**, with the swapped fraction decayed on a cosine from its peak to zero over the first three quarters of training. The growth criterion is evaluated at the current fit, so an entry is grown only if it explains error the already-active weights leave behind: redundancy is scored where it actually matters, and the candidate list can be the entire pool. The support is refit convexly afterwards exactly as before.

`experiments/resisc45_rigl.py` crosses the two search methods with two quota candidate lists and adds a no-candidate-list arm, all on the merged 2,084-column pool. Search settings and `C` are chosen on validation; test is read once per cell.

| Parameters | Prune-only, top 256 | Prune-only, top 512 | Prune-and-regrow, top 256 | Prune-and-regrow, top 512 | Prune-and-regrow, whole pool |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.4740 | 0.4948 | 0.5513 | 0.5521 | **0.5710** |
| 384 | 0.5729 | 0.5597 | 0.6116 | 0.6092 | **0.6259** |
| 512 | 0.6290 | 0.6227 | **0.6624** | 0.6571 | 0.6476 |
| 640 | 0.6554 | 0.6583 | 0.6810 | 0.6813 | **0.6881** |
| 768 | 0.6794 | 0.6779 | 0.6948 | 0.6978 | **0.6983** |
| 896 | 0.6967 | 0.7019 | 0.7146 | **0.7162** | 0.7049 |
| 1,024 | 0.7167 | 0.7190 | 0.7214 | **0.7290** | 0.7205 |

(each cell is test accuracy at the validation-best search setting for that arm; the first column is the previous best protocol.)

**Regrowth beats pruning at every budget, and by far the most at small ones.** Holding the candidate list fixed, it gains +7.7 test points on the 256-column list and +5.7 on the 512-column list at 256 stored values, falling to +0.5 and +1.0 at 1,024; comparing each method's validation-selected arm instead gives +7.6 points at 256 values, +5.3 at 384, +2.8 at 512, and +1.0 at 1,024. This is the largest single improvement found for RESISC45 since element-wise sparsity itself, and it is purely a search-procedure change -- same pool, same head, same parameter accounting, same convex refit. Widening the candidate list without changing the search does almost nothing by comparison (prune-only gains 2.1 points at 256 values and loses accuracy at 384, 512, and 768), which is the control that separates the two explanations: the gain is the ability to *reconsider*, not the wider list.

Which candidate list wins depends on the budget, and the crossover is informative. Below 512 values the head has so few weights that the ranked list is the binding constraint and the whole 2,084-column pool wins outright; above 640 values a wider list starts to cost more in support overfitting than it returns, and the 512-column quota list wins. The whole-pool arm never collapses, though: it stays within 1.5 points of the best arm at every budget, so **the group-lasso candidate list, the quota, and the hand-tuning behind them are no longer load-bearing.**

| Parameters | Selected arm | Columns read | Validation | Test |
|---:|---|---:|---:|---:|
| 256 | whole pool | 176 | 0.5917 | 0.5710 |
| 384 | whole pool | 269 | 0.6419 | 0.6259 |
| **512** | top 512 | 255 | 0.6810 | **0.6571** |
| 640 | whole pool | 415 | 0.7043 | 0.6881 |
| 768 | top 512 | 342 | 0.7229 | 0.6978 |
| **896** | top 512 | 353 | 0.7378 | **0.7162** |
| 1,024 | top 512 | 383 | 0.7467 | **0.7290** |

Three frontier movements follow. The smallest budget whose test accuracy clears **65%** falls from 640 to **512 stored values** (65.71%); the smallest that clears **70%** falls from 1,024 to **896** (71.62%), with validation already clearing 70% at 640; and the 1,024-value operating point improves from 71.67% to **72.90% test**, which is 2.4 points above what a *dense* head reaches with 2,860 values. Against the dense frontier, element-wise sparsity is now worth a 4.2x parameter reduction at 65% (512 against 2,156) and a 3.2x reduction at 70% (896 against 2,860). Both numbers are superseded further below: dropping the element-wise restriction itself, by coding the same head in an over-complete class dictionary, takes 65% to 256 stored values and 70% to 448.

The one cost is breadth of feature extraction rather than stored values. A prune-only head at 1,024 values reads 207 pool columns; the regrown head reads 383, and the whole-pool arm reads 621. The stored-value count and the index pattern are unchanged -- one column id per stored weight either way -- but more of the 2,084-column pool has to be computed at inference time. Where extraction cost matters more than parameter count, the top-256 regrown arm is the compromise: 0.7214 test at 1,024 values from 228 columns, still 0.5 points above the prune-only head on a list half as wide. The section after next pushes that trade much further: measured against the whole-pool arm, restricting the pool to the top 128 ranked columns costs 1.75 points at 1,024 values and cuts extraction from 635 columns to 114, a 5.6x reduction.

Reproduce with:

```bash
python experiments/resisc45_rigl.py
```

Results are written to `experiments/resisc45_rigl_result.csv`; `resisc45_rigl.py --summarise` re-prints the comparison from the existing CSV.

## Degree-2 products: the information is real, the budget cannot buy it

Every RESISC45 head above is *linear* in the pool, so one stored value buys one
column.  A product of two pool columns is still deterministic arithmetic on a
single image -- no learned constants, the same zero-parameter status as every
other column -- and standardising it folds into the head exactly like any other
column, so a weight on `a*b` costs one stored value like any other weight.  At a
budget where each class can afford ten to twenty weights, a column that already
carries an interaction ought to be worth more per value than either factor
alone.  `experiments/resisc45_quadratic.py` tests that, and separately tests
whether products can buy back the one cost the previous section could not pay:
the regrown head reads 383 of 2,084 pool columns and the whole-pool arm reads
621, so a deployment has to compute most of the pool.  A degree-2 polynomial in
`K` base columns needs only those `K` columns extracted, however many products
it then reads.

Products are formed from the top `K` columns of the same train-only group-lasso
ranking the candidate lists use, giving `K(K+1)/2` extra columns (the diagonal
is the squares).  Every arm searches its support with the same prune-and-regrow
fitter over its whole column set -- no candidate list, no quota -- refits
convexly, and picks search settings and `C` on validation.

**Unconstrained, the interaction information is large and it is exactly where
the linear pool is narrow.**

| Pool | Columns extracted | Pool columns | Head values | Test |
|---|---:|---:|---:|---:|
| top 32, linear | 32 | 32 | 1,452 | 0.5363 |
| top 32, degree 2 | 32 | 560 | 24,684 | **0.6262** |
| top 64, linear | 64 | 64 | 2,860 | 0.6678 |
| top 64, degree 2 | 64 | 2,144 | 94,380 | **0.7314** |
| top 128, linear | 128 | 128 | 5,676 | 0.7446 |
| top 128, degree 2 | 128 | 8,384 | 368,940 | **0.7762** |
| top 256, linear | 256 | 256 | 11,308 | 0.7732 |
| top 256, degree 2 | 256 | 33,152 | 1,458,732 | **0.7814** |
| whole pool, linear | 2,084 | 2,084 | 91,740 | 0.7937 |
| whole pool + top-128 products | 2,084 | 10,340 | 455,004 | **0.8071** |

The gain decays monotonically with the width of the linear pool it is added to:
+9.0 points at 32 columns, +6.4 at 64, +3.2 at 128, +0.8 at 256, +1.3 on the
whole pool.  Degree 2 on 128 columns (77.62%) matches linear on 256 (77.32%), so
at the ceiling the expansion **halves the number of pool columns that have to be
extracted**.  It also lifts the pool ceiling itself for the first time since the
native-resolution rebuild: 79.37% on the merged 2,084-column pool, and 79.78%
on the best pool measured anywhere in this file, against 80.71% test here.

**Under a budget almost none of that survives.** Test accuracy at equal stored
values, with each arm at its validation-selected search setting and `C`:

| Parameters | top 128 linear | top 128 degree 2 | top 256 linear | top 256 degree 2 | whole linear | whole + products |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.5092 | **0.5330** | 0.5281 | **0.5659** | 0.5616 | **0.5722** |
| 384 | 0.5825 | **0.5981** | 0.6075 | **0.6219** | 0.6337 | 0.6181 |
| 512 | 0.6246 | 0.6210 | 0.6381 | 0.6460 | 0.6532 | 0.6587 |
| 640 | 0.6517 | 0.6578 | 0.6714 | 0.6619 | 0.6898 | 0.6744 |
| 768 | 0.6733 | 0.6668 | 0.6817 | 0.6859 | 0.6987 | 0.6930 |
| 896 | 0.6860 | 0.6795 | 0.7037 | 0.6894 | 0.7021 | 0.7083 |
| 1,024 | 0.6995 | 0.6940 | 0.7051 | 0.6959 | 0.7170 | 0.7195 |

At 256 and 384 stored values the expansion is worth 1.4 to 3.8 points on the
fixed-`K` pools, which is the regime the argument predicted: with 212 weights
for 45 classes each weight has to carry as much as possible.  From 512 values
upwards the effect vanishes into the +-0.6-point sampling noise of a 6,300-image
split and is as often negative as positive, and the frontier does not move --
this experiment clears 65% at 512 values and 70% at 896, exactly where the
previous section left them.  Validation is *not* neutral about this: it prefers
an expanded pool at four of the seven budgets, including 1,024 (0.7444 against
0.7348), while test does not follow.  The extra 8,000 to 33,000 candidate
columns give the support search more ways to fit 18,900 training images than
they give it real structure.

The mechanism is that above roughly 512 stored values the budgeted head is
**weight-limited, not column-limited**.  It already reads 348 distinct columns
at 512 values and 635 at 1,024 -- far more columns than any class can afford to
combine -- so a better-conditioned column does not relieve the binding
constraint, it just enlarges the search space.  This mirrors the EuroSAT entry
in the negative-results table below, where quadratic head features also raised
validation fit without improving test, and it reaches the same conclusion from
the opposite direction: on EuroSAT the pool was the constraint, here the head
is, and a quadratic expansion helps neither.

**What the experiment does buy is a cheaper extractor, and the linear pool
supplies most of that on its own.** The base pool columns a deployment must
actually compute, at 1,024 stored values:

| Arm | Base columns extracted | Test |
|---|---:|---:|
| top 32, linear | 32 | 0.5537 |
| top 64, linear | 55 | 0.6535 |
| top 128, linear | 114 | 0.6995 |
| top 128, degree 2 | 126 | 0.6940 |
| top 256, linear | 216 | 0.7051 |
| top 256, degree 2 | 241 | 0.6959 |
| whole pool + products | 395 | 0.7195 |
| whole pool, linear | 635 | 0.7170 |

Restricting the pool to the top 128 ranked columns costs 1.75 points and cuts
extraction from 635 columns to 114, a **5.6x reduction**; the top 256 costs 1.2
points for a 2.9x reduction.  Adding products on top of a narrow pool does not
improve that trade at this budget.  The extraction caveat the previous section
left open is therefore answerable, but by narrowing the pool rather than by
squaring it.

One accounting note for the product arms: stored values are unchanged at
`nonzeros + 44`, but a product weight has to name two base columns instead of
one, so the conservative index pattern for the 1,024-value `whole + products`
head is 1,640 ids rather than 980.

Reproduce with:

```bash
python experiments/resisc45_quadratic.py
```

Results are written to `experiments/resisc45_quadratic_result.csv` (the
`budget=0` rows are the unconstrained ceilings);
`resisc45_quadratic.py --summarise` re-prints the tables, including the
extraction Pareto front at every budget, from the existing CSV.

## Spending more on the support search: bagging and longer runs

The section above found that a wider pool only raises validation fit, and the
one before it found that the support search is where the accuracy is.  Together
those point at spending more compute on the search itself.
`experiments/resisc45_support_probe.py` spends it in the two obvious directions
and measures what comes back.  Both use the same convex refit and
validation-selected `C`, and both are judged the only way that matters: does
gating on validation move *test*?

**Bagging the support loses.** If a wider candidate set overfits the support
search, stability selection is the textbook fix -- fit prune-and-regrow on
several subsamples of train, keep the entries that recur most often, refit that
support convexly on the full split.

| Support | 512 validation | 512 test | 1,024 validation | 1,024 test |
|---|---:|---:|---:|---:|
| Single fit (current) | 0.6729 | **0.6532** | 0.7337 | **0.7211** |
| Bagged, 8 fits at 80% | 0.6527 | 0.6370 | 0.7246 | 0.7132 |
| Bagged, 16 fits at 80% | 0.6544 | 0.6406 | 0.7314 | 0.7202 |
| Bagged, 8 fits at 60% | 0.6568 | 0.6363 | 0.7305 | 0.7179 |

Every bagged variant is worse on both splits, by 1.3 to 1.7 points of test
accuracy at 512 values and 0.1 to 0.8 at 1,024, so the validation gate correctly
rejects all of them.  The reason is the same one that sank the group-lasso
ranking: **frequency voting scores each entry on how often it is chosen, never
on what it adds given the others.** Averaging over subsamples rewards entries
that are individually stable, which is exactly the redundant ones, and throws
away the conditional growth criterion that made prune-and-regrow work.

**Longer searches move validation but not the frontier.** `resisc45_rigl.py`
found 4,000 epochs with 100 mask updates beat 2,000 with 100, so the number of
regrow opportunities is a real hyperparameter; this sweeps it four-fold further
on all three candidate lists.  The reference for each budget is the validation
pick within the existing 4,000-epoch grid, which reproduces the arm and the test
accuracy `resisc45_rigl.py` selected at every budget from 512 upwards, so this
compares protocol against protocol rather than cell against cell.  (At 384 the
reference reads 0.6238 against that section's 0.6259, because its grid also
held a 4,000/200 setting this one does not.)

| Parameters | Reference grid, test | Extended grid, best validation | Extended grid, test | Change (points) |
|---:|---:|---:|---:|---:|
| 384 | 0.6238 | 0.6460 | 0.6279 | +0.41 |
| 512 | 0.6571 | 0.6817 | 0.6625 | +0.54 |
| 640 | 0.6881 | 0.7081 | 0.6859 | -0.22 |
| 768 | 0.6978 | 0.7310 | 0.7063 | +0.85 |
| 896 | 0.7162 | 0.7430 | 0.7210 | +0.48 |
| 1,024 | 0.7290 | 0.7490 | 0.7254 | -0.36 |

Validation rises at every budget, by up to 0.8 points at 768, and test moves by
between -0.36 and +0.85 -- four budgets up, two down, and no threshold moves:
65% is still first cleared at 512 stored values and 70% at 896.  The 1,024-value
operating point actually falls, because the extended grid's highest-validation
cell (top-512 list, 8,000 epochs, 0.7490) tests at 0.7254 against the 0.7290 the
unextended grid already reported.  Quadrupling the search is therefore
indistinguishable from noise under this protocol, and the conclusion is that
**the selection step, not the search, now consumes the difference**: at
these budgets a 6,300-image validation split resolves about +-0.6 points, and
every additional arm offered to the gate is another chance to spend that on
nothing.  Claims below a point need a repeated-split or bootstrap interval
before they should be believed.

Reproduce with:

```bash
python experiments/resisc45_support_probe.py
```

Results are written to `experiments/resisc45_support_probe_result.csv`;
`resisc45_support_probe.py --summarise` re-prints both comparisons from the
existing CSV.

## An over-complete class dictionary: one stored value for many classes

Every RESISC45 head above is *element-wise* sparse: one stored value buys one
`(class, column)` weight, so a column eight classes want costs eight values.
The previous section diagnosed why that now binds -- above about 512 stored
values the head already reads more columns than any class can afford to combine,
so it is **weight-limited, not column-limited**, and neither a wider pool nor a
better support search can relieve it. That leaves the head's parameterisation.

`experiments/resisc45_class_dict.py` generates the `44 x k` reference-class
weight matrix as `D @ P`. `D [44, atoms]` is a **fixed dictionary of unit-norm
class directions** and `P [atoms, k]` is a sparse code searched by the same
prune-and-regrow procedure and refit convexly at a validation-selected `C`. One
stored value now buys a whole *pattern across classes* rather than one class's
weight. Stored values are `nnz(P) + 44` biases, exactly as before, and the
dictionaries that win are drawn from a fixed seed, so `D` is as free as the
feature extractors themselves -- deterministic arithmetic with no learned
constants. The 44 identity atoms are one of the dictionaries tested, and that
arm reproduces the element-wise head *exactly*, support and all, which is what
makes the comparison a controlled one.

**Over-completeness is the whole effect; the choice of atoms is almost
irrelevant.** All arms below use the same 512-column candidate list, the same
search settings, and the same `C` grid.

| Dictionary | Atoms | 256 values | 512 values | 1,024 values |
|---|---:|---:|---:|---:|
| `singleton` (the element-wise head) | 44 | 0.5521 | 0.6571 | 0.7290 |
| `pca44`, the head's own principal class directions | 44 | 0.5503 | 0.6563 | 0.7376 |
| `dct44`, a fixed orthonormal rotation | 44 | 0.5422 | 0.6548 | 0.7297 |
| `mean-tree`, Ward class groups from the pool | 87 | 0.5857 | 0.6768 | 0.7381 |
| `weight-tree`, Ward class groups from the dense head | 87 | 0.5733 | 0.6648 | 0.7306 |
| `random87`, size-matched random class groups | 87 | 0.5768 | 0.6714 | 0.7378 |
| `gauss1024`, random directions | 1,024 | 0.6143 | 0.7016 | 0.7530 |
| `graded2048`, random class subsets | 2,092 | 0.6203 | 0.7035 | 0.7508 |
| `gauss4096` | 4,096 | 0.6416 | 0.7138 | 0.7552 |
| `gauss8192` | 8,192 | 0.6452 | 0.7233 | 0.7530 |
| `gauss16384` | 16,384 | 0.6519 | 0.7216 | 0.7576 |
| `gauss32768` | 32,768 | 0.6549 | 0.7222 | 0.7605 |

Three controls separate the explanations. A **complete** 44-atom basis is worth
nothing: rotating class space into the dense head's own principal directions
(`pca44`) or into a DCT basis moves test accuracy by between -1.0 and +0.9
points with no consistent sign, which is what should happen, because a complete
basis leaves the model class unchanged and only relabels the sparsity. **Which**
groups are used barely matters either: Ward class groups taken from the pool
beat size-matched random groups by 0.9 points at 256 values, 0.5 at 512 and
0.03 at 1,024 -- at or below the split's resolution -- and groups taken from the
dense head's own rows are the worst of the three. What does matter is **how
many** atoms there are -- 44 -> 1,024 -> 4,096 -> 16,384 buys +6.2, +2.7 and
+1.0 test points at 256
values -- and the curve flattens by about 16,384 atoms, where the code is
choosing 980 entries from 8.4 million. The 8,192-, 16,384- and 32,768-atom arms
are separated by less than the 0.6 points a 6,300-image split resolves, so the
top of that curve is one operating point, not three.

The paired bootstrap over test images (2,000 resamples, the same images for both
arms) puts the 16,384-atom gain at **+9.98 points [+8.9, +11.2] at 256 stored
values**, +6.4 [+5.4, +7.5] at 512, and +2.9 [+1.9, +3.8] at 1,024. Three
independent dictionary draws give 65.19/65.49/64.71% at 256 values,
72.16/72.44/71.67% at 512 and 75.76/76.21/76.27% at 1,024, so the effect does
not depend on the draw; the replication arms are reported but kept out of the
validation gate.

**The frontier moves by about a factor of two at both targets.**

| Parameters | Selected arm | Columns read | Deployed weights | Index bits | Validation | Test |
|---:|---|---:|---:|---:|---:|---:|
| 128 | `gauss16384` | 69 | 3,036 | 1,932 | 0.5522 | 0.5340 |
| 192 | `gauss16384` | 116 | 5,104 | 3,404 | 0.6335 | 0.6170 |
| **256** | `gauss32768` | 158 | 6,952 | 5,088 | 0.6694 | **0.6549** |
| 320 | `gauss16384` | 201 | 8,844 | 6,348 | 0.7000 | 0.6684 |
| 384 | `gauss16384` | 221 | 9,724 | 7,820 | 0.7171 | 0.6957 |
| **448** | `gauss16384` | 255 | 11,220 | 9,292 | 0.7302 | **0.7095** |
| 512 | `gauss32768` | 273 | 12,012 | 11,232 | 0.7470 | 0.7222 |
| 640 | `gauss16384` | 316 | 13,904 | 13,708 | 0.7552 | 0.7356 |
| 768 | `gauss16384` | 359 | 15,796 | 16,652 | 0.7657 | 0.7511 |
| 896 | `gauss16384` | 381 | 16,764 | 19,596 | 0.7727 | 0.7524 |
| 1,024 | `gauss16384` | 400 | 17,600 | 22,540 | 0.7817 | **0.7576** |

The smallest budget whose test accuracy clears **65%** falls from 512 to **256
stored values** (65.49%), and the smallest that clears **70%** falls from 896 to
**448** (70.95%). The 1,024-value operating point improves from 72.90% to
**75.76% test**, 5.3 points above what a dense affine head reaches with 2,860
values, and it closes 44% of the gap between the previous 1,024-value head and
the 79.37% ceiling of an unconstrained 91,740-value head on the same pool.
Against the dense frontier, the head is now worth an 8.4x parameter reduction at
65% (256 against 2,156) and a 6.4x reduction at 70% (448 against 2,860).

**What it costs.** Nothing in stored values or in feature extraction -- at 1,024
values the dictionary head reads 400 pool columns against 383 for the
element-wise head -- but two other things grow. The deployed weight matrix is
dense (17,600 nonzeros reconstructed from 980 stored values at load time), so
inference is a dense `45 x 512` matrix-vector product rather than a sparse one.
And each stored value needs a wider atom id: `log2(16384)` bits instead of
`log2(44)`, which takes the index pattern from 14.2 to 22.5 kbit at 1,024
values. Reading the whole frontier at **equal total bits** (32 bits per stored
value plus the index pattern) rather than at equal stored values, the dictionary
head is still ahead by +9.4 points at 0.74 KiB, +7.1 at 1.6 KiB, +5.0 at 2.9 KiB
and +3.4 at 5.0 KiB, so the gain survives paying for its own indices with room
to spare. The two rows above 5.7 KiB are extrapolations beyond where the
element-wise arms reach and are marked as such by `--summarise`.

The `C` grid's lower edge is load-bearing for these arms -- the 16,384-atom head
selects `C=0.1` at 512 and 1,024 values -- so it was checked against a wider
grid: `C=0.03` and `C=0.003` are worse on validation *and* on test at both
budgets, and `C=0.3` is worse on validation (its test accuracy at 1,024 values,
0.7583, is within noise of the selected cell's 0.7576), so the reported
operating points are interior optima rather than grid artefacts.

Reproduce with:

```bash
python experiments/resisc45_class_dict.py
```

Results are written to `experiments/resisc45_class_dict_result.csv` (about two
GPU-hours for the full grid); `resisc45_class_dict.py --summarise` re-prints the
tables, the bootstrap intervals, and the equal-bits comparison from the existing
CSV, and `--only-dicts` re-runs a single dictionary against its element-wise
control.

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
| Merging a new RESISC45 feature family into one group-lasso ranking | The 505-column object-layout pool wins 113 of the merged top 256 slots and lowers validation accuracy at every budget; it only pays when capped at a reserved quota of candidate slots |
| Iterative magnitude pruning as the support search for a sparse RESISC45 head | Prune-only cannot recover a column it drops, so it depends on a candidate list ranked against the label alone; prune-and-regrow beats it at every budget, by 7.6 points of test accuracy at 256 stored values and 1.0 at 1,024 |
| Degree-2 products of RESISC45 pool columns | Zero-parameter and genuinely informative -- they lift the pool ceiling from 79.37% to 80.71% test and are worth +9.0/+6.4/+3.2 points on 32/64/128-column pools -- but above 512 stored values the head is weight-limited, not column-limited, so the expansion only raises validation fit and leaves the frontier where it was |
| Bagged (stability-selection) supports for the sparse RESISC45 head | Voting over 8-16 prune-and-regrow supports fitted on 60-80% subsamples loses 1.3-1.7 points of test accuracy at 512 values and 0.1-0.8 at 1,024; frequency voting rewards individually stable entries and reintroduces exactly the redundancy the regrow criterion removes |
| Longer prune-and-regrow searches for RESISC45 | Quadrupling the search to 16,000 epochs and 400 mask updates raises validation at every budget, but the validation gate then moves test by between -0.36 and +0.85 and no target threshold moves; the 1,024-value operating point falls from 0.7290 to 0.7254 |
| Complete class bases for the RESISC45 sparse head | Coding the head in its own principal class directions or in a DCT basis instead of the identity moves test accuracy by between -1.0 and +0.9 points with no consistent sign; a complete basis leaves the model class unchanged, so only *over-complete* dictionaries help |
| Semantically chosen class groups for the RESISC45 dictionary head | Ward class groups from the pool or from the dense head's own rows beat size-matched random groups by 0.9 points at 256 stored values and by 0.03 at 1,024; the atom count is worth several times more than the atom content, and a random draw is as good as a designed one |

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
python experiments/resisc45_gpu_features3.py
python experiments/resisc45_layout_gain.py
python experiments/resisc45_layout_diagnose.py
python experiments/resisc45_layout_frontier.py
python experiments/resisc45_rigl.py
python experiments/resisc45_quadratic.py
python experiments/resisc45_support_probe.py
python experiments/resisc45_class_dict.py
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

The baseline's complete `C` sweep and per-class test results are stored in `experiments/image_statistics_baseline_result.txt`. The selected-feature and ImageStats five-seed fraction results are stored in `experiments/eval_training_fractions_result.csv` and `experiments/eval_imagestats_fractions_result.csv`, including model metadata, split protocol, individual seed accuracies, mean, and sample standard deviation. The coordinate-only MLP screen and multi-seed frontier are stored in `experiments/coordinate_mlp_screen.csv` and `experiments/coordinate_mlp_result.csv`. The preliminary RESISC45 transfer, including exact selected feature indices and names, is stored in `experiments/resisc45_33_feature_fractions.csv`. The RESISC45 object-layout experiments write `experiments/resisc45_layout_gain_result.csv`, `experiments/resisc45_layout_gain_classes.csv`, `experiments/resisc45_layout_diagnose_result.csv`, and `experiments/resisc45_layout_frontier_result.csv`, the prune-and-regrow comparison writes `experiments/resisc45_rigl_result.csv`, the degree-2 product experiment writes `experiments/resisc45_quadratic_result.csv`, the support-search probe writes `experiments/resisc45_support_probe_result.csv`, and the class-dictionary head writes `experiments/resisc45_class_dict_result.csv`. Dataset download and fraction evaluators verify the official TorchGeo checksums. Submission evaluation scripts recompute features from raw patches rather than relying on cached feature matrices.
