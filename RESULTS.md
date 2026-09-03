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

All reported feature extractors are deterministic arithmetic on an input patch and therefore have zero learned parameters. The feature standardizer is folded into the logistic-regression weights, so it adds no deployed values — for a head that *stores* its weights, which is every EuroSAT model here and every RESISC45 head up to the class-dictionary section. A head that *reconstructs* its weights from fixed dictionaries has to pay for the fold as well; "The standardiser was never free" prices that and re-derives the RESISC45 frontier under the tighter count. A 10-class affine softmax head is shift-invariant and can store one class as an implicit zero-logit reference, reducing the exact parameter count from `10 × (F + 1)` to `9 × (F + 1)` without changing any prediction. Historical submissions 01–11 report the 10-row checkpoints they actually stored; submissions 12–13 and the baseline comparison use the tighter reference-class count.

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

Gating on validation, a sparse head first clears 65% at **640** stored values (64.35% test) and 70% at **896** (69.00% test); the smallest budgets whose *test* accuracy also clears the two targets are **768** (67.44%) and **1,024** (70.48%). A dense affine head needs 2,156 values to reach 66.40% and 2,860 to reach 70.51%, so element-wise sparsity is worth roughly a **2.8x parameter reduction** at both targets. Four later sections lower the sparse numbers again: an object-layout quota takes 65% to 640 values, a prune-and-regrow support search takes 65% to 512 and 70% to 896, generating the head from a fixed over-complete class dictionary takes 65% to 256 and 70% to 448, adding a column dictionary of feature pairs takes 65% to 208 and 70% to 320, and dropping the 44 free intercepts takes 70% to 304. This is the opposite of the EuroSAT finding, where element-wise sparsity was about 0.8 points *worse* than feature selection at equal budget; with 45 classes instead of 10 the head dominates the budget, and letting each class pick its own columns is the only way to read a wide pool cheaply.

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

Three frontier movements follow. The smallest budget whose test accuracy clears **65%** falls from 640 to **512 stored values** (65.71%); the smallest that clears **70%** falls from 1,024 to **896** (71.62%), with validation already clearing 70% at 640; and the 1,024-value operating point improves from 71.67% to **72.90% test**, which is 2.4 points above what a *dense* head reaches with 2,860 values. Against the dense frontier, element-wise sparsity is now worth a 4.2x parameter reduction at 65% (512 against 2,156) and a 3.2x reduction at 70% (896 against 2,860). Both numbers are superseded further below: dropping the element-wise restriction itself, by coding the same head in an over-complete class dictionary, takes 65% to 256 stored values and 70% to 448, coding it in a dictionary of column pairs as well takes them to 208 and 320, and refusing to store a free intercept takes 70% to 304.

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
65% (256 against 2,156) and a 6.4x reduction at 70% (448 against 2,860).  The
next section applies the same idea to the *column* axis and takes the two
budgets to 208 and 320.

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

## A column dictionary too: what a stored value spends on the other axis

> **Repriced.** The width-2 atoms this section introduces make the deployed head
> need one stored `sigma_i / sigma_j` per column a pair touches, which the
> `nnz(P) + 44` count below does not include: the 208-value operating point is
> 415 deployed values and the 320-value one is 566.  "The standardiser was never
> free" prices it and recovers most of the difference by rounding `sigma` to a
> power of two.

The section above widened the alphabet a stored value may spell a *class*
pattern with and halved the budget at both targets.  Its lesson was mechanical
rather than semantic -- what paid was over-completeness, letting the support
search *choose* a pattern per stored value instead of spelling one out class by
class -- so the obvious question is whether the same trick works on the other
axis, where each stored value still buys exactly one pool column.

`experiments/resisc45_feature_dict.py` generates the head as `Dc @ P @ Df.T`:
`Dc [44, catoms]` of unit-norm class directions, `Df [k, fatoms]` of unit-norm
**column** directions, and `P` sparse.  One stored value buys the rank-1 outer
product `Dc[:, a] Df[:, f].T` -- a class pattern times a column pattern.
Stored values are `nnz(P) + 44` biases as before, both dictionaries are
generated from a fixed seed by a stated rule, and both controls are nested:
`Df = I` is exactly the class-dictionary head and `Dc = Df = I` is exactly the
element-wise head.  Those two arms reproduce the previous section's numbers to
four decimals wherever it reports them below 1,024 stored values, and to within
0.2 points at 1,024, where the reassociated matmul changes the last few support
entries.

Carrying the code as `nnz` live values indexed into `catoms x fatoms` rather
than as a dense `atoms x k` tensor is what makes the search affordable: a
512-value fit with a 4,096-atom class dictionary that took 41 s in the
class-dictionary code takes **2.5 s** here and returns a bit-identical support,
because an epoch is now `nnz` rank-1 updates and the dense `catoms x fatoms`
gradient is formed only at the ~100 mask updates that actually need it.  The
old form computes `(xs @ P.T) @ Dc.T`, which is 158 GFLOP per epoch at 16,384
atoms; forming `W = Dc @ P` first is about 200x fewer flops, and storing only
the live entries removes the optimiser's 0.5 GiB of dense state as well.

**Only width-2 atoms pay.**  All arms below use the same `gauss16384` class
dictionary, the same 512-column candidate list and the same search settings, so
they differ only in `Df`.  `pairsT` enumerates *every* signed pair
`(x_i +- x_j)/sqrt2` over the `T` highest-ranked columns; `randWxN` samples `N`
atoms of width `W` over all 512.

| Column dictionary | Atoms | 256 values | 512 values | 1,024 values |
|---|---:|---:|---:|---:|
| `identity` (the class-dictionary head) | 512 | 0.6519 | 0.7216 | 0.7595 |
| `pairs128`, all pairs of the top 128 columns | 16,768 | 0.6654 | 0.7397 | 0.7659 |
| `pairs181` | 33,092 | 0.6757 | 0.7373 | 0.7556 |
| `pairs256` | 65,792 | **0.6790** | 0.7403 | 0.7579 |
| `pairs320` | 102,592 | 0.6765 | 0.7317 | 0.7471 |
| `rand2x16384`, sampled pairs | 16,896 | 0.6681 | **0.7414** | 0.7587 |
| `rand2x65536`, sampled pairs | 66,048 | 0.6711 | 0.7317 | 0.7559 |
| `rand4x16384`, width 4 | 16,896 | 0.6560 | 0.7216 | 0.7584 |
| `rand8x16384`, width 8 | 16,896 | 0.6429 | 0.7248 | 0.7475 |
| `rand32x16384`, width 32 | 16,896 | 0.6417 | 0.7183 | 0.7511 |
| `gaussF16384`, dense random directions | 16,896 | 0.6557 | 0.7279 | 0.7560 |
| `graded`, pairs plus width-4 and width-8 atoms | 45,380 | 0.6798 | 0.7359 | 0.7575 |

This is the opposite of the class-side result and it is the point of the
section.  On the class axis *any* over-complete draw worked and the atom count
was all that mattered; on the column axis the atom count buys nothing by itself
-- 16,896 dense random directions are worth +0.4 points at 256 values
[-0.8, +1.4] and 16,896 width-32 atoms are worth **-1.0** -- and what pays is
specifically the **width-2** structure, +2.7 points at 256 values [+1.6, +3.8]
by paired bootstrap over test images.  The width sweep 2 -> 4 -> 8 -> 32 ->
dense at 256 values reads 0.6790, 0.6560, 0.6429, 0.6417, 0.6557 against
0.6519 for the identity, so the effect is already gone at width 4 and is a
*loss* by width 8.  The reason the two axes differ is that the head's class
pattern for a column is dense -- every class needs some weight -- while its
column pattern for a class is not, and a wide atom forces a class to pay for
columns it does not want.  A pair atom is the smallest structure that is still
a *choice*: it ties `|w_i| = |w_j|` and picks the relative sign, which is one
stored value for two weights whenever a class wants a sum or a contrast of two
features.

Enumerating pairs beats sampling them, but only just: `pairs181` reads 0.6757
at 256 values against 0.6681 for `rand2x16384` and 0.6711 for the four-times
larger `rand2x65536`, so being able to reach the *best* pair among the
top-ranked columns is worth about as much as quadrupling a random draw.  The
enumeration saturates by about 256 columns -- `pairs320` is no better at 256
values and clearly worse above 512, where 102,592 column atoms times 16,384
class atoms start to overfit 18,900 training images.

**The frontier moves again up to 640 values, and stops there.**

| Parameters | Selected column dictionary | Columns read | Deployed weights | Index bits | Validation | Test |
|---:|---|---:|---:|---:|---:|---:|
| 96 | `pairs256` | 75 | 3,300 | 1,560 | 0.5040 | 0.4887 |
| 128 | `pairs181` | 103 | 4,532 | 2,437 | 0.5803 | 0.5578 |
| 160 | `pairs256` | 133 | 5,852 | 3,481 | 0.6290 | 0.6079 |
| 176 | `pairs256` | 145 | 6,380 | 3,961 | 0.6481 | 0.6222 |
| 192 | `pairs256` | 157 | 6,908 | 4,441 | 0.6643 | 0.6360 |
| **208** | `pairs181` | 136 | 5,984 | 4,758 | 0.6760 | **0.6505** |
| 224 | `pairs256` | 163 | 7,172 | 5,401 | 0.6797 | 0.6619 |
| 240 | `pairs256` | 181 | 7,964 | 5,881 | 0.6992 | 0.6717 |
| 256 | `pairs256` | 181 | 7,964 | 6,361 | 0.7094 | 0.6790 |
| 272 | `pairs181` | 159 | 6,996 | 6,615 | 0.7065 | 0.6859 |
| 288 | `pairs256` | 194 | 8,536 | 7,321 | 0.7165 | 0.6968 |
| 304 | `pairs181` | 171 | 7,524 | 7,544 | 0.7184 | 0.6965 |
| **320** | `pairs256` | 211 | 9,284 | 8,282 | 0.7260 | **0.7010** |
| 352 | `pairs256` | 220 | 9,680 | 9,242 | 0.7303 | 0.7065 |
| 384 | `pairs256` | 228 | 10,032 | 10,202 | 0.7387 | 0.7243 |
| 448 | `pairs256` | 246 | 10,824 | 12,122 | 0.7563 | 0.7271 |
| 512 | `graded` | 456 | 20,064 | 13,792 | 0.7649 | 0.7359 |
| 640 | `pairs181` | 274 | 12,056 | 17,292 | 0.7695 | 0.7497 |
| 768 | `pairs256` | 306 | 13,464 | 21,724 | 0.7762 | 0.7551 |
| 896 | `rand2x16384` | 461 | 20,284 | 23,894 | 0.7792 | 0.7495 |
| 1,024 | `pairs128` | 355 | 15,620 | 27,473 | 0.7886 | 0.7659 |

The smallest budget whose test accuracy clears **65%** falls from 256 to **208
stored values** (65.05%) and the smallest that clears **70%** falls from 448 to
**320** (70.10%).  Against the dense affine head's 2,156 and 2,860 values that
is a **10.4x** and **8.9x** parameter reduction, and against the element-wise
sparse head of two sections ago (512 and 896) it is 2.5x and 2.8x.  At 208
values the head extracts 136 pool columns and deploys 5,984 dense weights
reconstructed from 164 stored values plus 44 biases.

**Above 640 values the column dictionary is worth nothing, and the table says
so.**  The paired bootstrap of `pairs256` against the class-dictionary head is
+1.9 points [+0.7, +3.0] at 192 values, +2.7 [+1.6, +3.8] at 256, +1.9
[+0.9, +2.8] at 512 and +1.7 [+0.8, +2.7] at 640 -- every interval excluding
zero -- but +0.3 [-0.6, +1.2] at 768, +0.8 [-0.1, +1.6] at 896 and **-0.2**
[-1.0, +0.7] at 1,024.  The 1,024-value
operating point in the table, 0.7659, comes from `pairs128` winning a
validation gate it wins by +0.6 points [-0.2, +1.5] on test -- inside the
+-0.6-point resolution of a 6,300-image split -- so it should be read as
unchanged from the 0.7595 the same code path gives with `Df = I`, not as an
improvement.  The 896-value row is the same effect with the opposite sign: the
gate picks `rand2x16384`, whose 0.7495 is below the 768-value row's 0.7551.
Three independent draws of `rand2x16384` give 0.6681/0.6752/0.6663 at 256
values and 0.7414/0.7330/0.7190 at 512, so the sampled arms carry about a point
of draw-to-draw variation of their own; the replication arms are reported but
kept out of the validation gate.

**What it costs.**  Extraction breadth is roughly unchanged, and falls where the
budget is largest: at 256 stored values the pair head reads 181 pool columns
against 158 for the class-dictionary head, but at 320 it reads 171 against 201
and at 512 it reads 230 against 287, because a pair atom pays for two columns
at once and the enumeration is restricted to the top-ranked ones.  The price is
again index bits -- a stored value needs `log2(65792)` bits for the column id
instead of `log2(512)`, which takes the index pattern from 4.9 to 6.4 kbit at
256 values and from 10.8 to 14.0 kbit at 512.  Read at **equal total bits**
rather than equal stored values the head is still ahead by +2.7 points at 0.57
KiB, +1.5 at 1.78 KiB, +0.9 at 3.68 KiB and +0.7 at 4.61 KiB, and the advantage
is gone by 5.7 KiB -- the same place the stored-value advantage goes.

Reproduce with:

```bash
python experiments/resisc45_feature_dict.py
```

Results are written to `experiments/resisc45_feature_dict_result.csv` (about
40 GPU-minutes for the full 21-budget grid);
`resisc45_feature_dict.py --summarise` re-prints the dictionary comparison, the
paired-bootstrap intervals, the frontier and the equal-bits comparison from the
existing CSV.

The `+ 44` in this section's parameter count -- 21% of the budget at the
208-value operating point -- is examined in the next section, which finds it
almost entirely wasted.

## The intercept was 21% of the budget

> **Repriced.** Centring at the training mean *is* an intercept, so a head with
> `b = 0` still deploys 44 numbers and the saving below is the second copy, not
> the first.  Once the fold is counted, the free intercept wins 77 of 84 paired
> cells at identical deployed cost -- but coding it against *uncentred* features
> does pay, at 16 values rather than 44, provided the convex refit is run to
> convergence.  See "The standardiser was never free".

Every RESISC45 head above has been budgeted as `nnz(P) + 44`: the sparse code
plus one free intercept per non-reference class.  That was a rounding error
when the budget was 1,024 stored values.  It is not one at the frontier the
section above reached -- at **208** values, 44 of them are intercept and only
164 are left to spell the weights, and at 96 values the intercept is 46% of
everything the model stores.

An intercept is a weight on a constant column, so nothing forces it to be free.
`experiments/resisc45_coded_bias.py` appends a constant column to the
standardised features and a matching identity atom to `Df`, which makes the
head a bias-free `Dc @ P @ Df.T` over `k + 1` columns: the prune-and-regrow
search then *decides* how many stored values an intercept is worth, against
every other entry it could grow instead.  The parameterisation is nested --
spending 44 values reproduces a free intercept exactly whenever `Dc` carries
the class singletons -- so it can only lose by search.  Three intercept rules
are compared at equal total stored values, everything else held at the previous
section's settings:

* `free` -- the incumbent, `nnz + 44` values;
* `coded` -- the intercept drawn from the same sparse code, `nnz` values;
* `none` -- no intercept in the standardised head at all, `nnz` values.

The `free` arms reproduce the previous section's numbers exactly (0.4887 at 96,
0.5578 at 128, 0.6505 at 208, 0.7010 at 320, 0.7403 at 512), and a re-run of
`resisc45_feature_dict.py` at 96 values reproduces all six of its arms to four
decimals, so the comparison below is like for like.

**Test accuracy at equal stored values, by intercept rule.**

| Parameters | `free` pairs256 | `free` pairs181 | `coded` pairs256 | `coded` pairs181 | `none` pairs256 |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.4887 | 0.4681 | **0.5641** | 0.5549 | **0.5641** |
| 112 | 0.5168 | 0.5203 | **0.5827** | 0.5798 | **0.5827** |
| 128 | 0.5594 | 0.5578 | 0.5946 | **0.6048** | 0.5946 |
| 144 | 0.5897 | 0.5789 | 0.6202 | 0.6141 | **0.6271** |
| 160 | 0.6079 | 0.6030 | 0.6290 | **0.6357** | 0.6287 |
| 176 | 0.6222 | 0.6119 | **0.6449** | 0.6387 | 0.6405 |
| 192 | 0.6360 | 0.6348 | 0.6463 | 0.6454 | **0.6465** |
| 208 | 0.6583 | 0.6505 | 0.6606 | 0.6567 | **0.6625** |
| 256 | 0.6790 | 0.6757 | 0.6863 | 0.6713 | **0.6886** |
| 320 | 0.7010 | 0.7021 | 0.7016 | 0.7032 | **0.7089** |
| 512 | 0.7403 | 0.7373 | 0.7370 | 0.7362 | **0.7417** |
| 1,024 | 0.7579 | 0.7556 | 0.7590 | 0.7543 | **0.7616** |

**The gain is large exactly where the head is starved and gone by 192 values.**
Paired bootstrap of the validation-selected `coded`/`none` arm against the
`free` incumbent at the same total stored values: **+6.6** points [+5.4, +7.8]
at 96, +6.3 [+5.1, +7.6] at 112, +4.5 [+3.4, +5.7] at 128, +3.8 [+2.6, +4.9]
at 144, +2.1 [+1.0, +3.2] at 160 and +2.3 [+1.2, +3.4] at 176 -- every interval
excluding zero -- then +1.0 [-0.1, +2.2] at 192 and +0.2 [-0.8, +1.3] at 208.
From 224 to 1,024 every interval straddles zero but one, +1.2 [+0.3, +2.1] at
448.  This is the same shape as the two sections above: the parameterisation
lever is worth a great deal while the head is starved and nothing once it is
not.

**The control is the whole story: the intercept is not worth buying cheaply,
it is worth not buying at all.** The `none` arm, which cannot spend anything on
an intercept, is indistinguishable from the `coded` one: over the 21 budgets it
wins 12, loses 6 and ties exactly 3, and no difference exceeds 1.2 points.  The
three exact ties are the tell -- at 96, 112 and 128 values the `coded` search
chose *zero* entries on the constant column, so the two arms fitted the same
support.  The `coded` search agrees everywhere else too: it puts **0 to 3** of
its stored values on the constant column at every budget up to 384 -- one at
the 208-value operating point -- and 2 to 10 from 448 to 1,024, where values
are cheap.  Given 16,384 class atoms times 65,792 column atoms to grow into,
the search wants about one stored value's worth of intercept, not 44.

The reason is the standardiser fold, and it is worth stating plainly because it
is what makes the accounting honest.  A deployed head is `w_eff = W / sigma`
and `b_eff = b - sum_j W_j mu_j / sigma_j`, so a head with `b = 0` in the
standardised space still has a *non-zero* deployed intercept -- the one implied
by centring the features at the training mean.  RESISC45's splits are exactly
class-balanced, so there is no prior to encode either, and 44 free values buy
almost nothing on top of the intercept the fold already supplies.  Nothing about
the accounting convention changes: both heads reconstruct `w_eff` and `b_eff`
from the stored code, the two fixed dictionaries and the same training
standardiser, and the coded head simply stores 44 fewer numbers to do it.

The effect is not an artefact of the pair dictionary.  With `Df = I` -- the
class-dictionary head of two sections ago, no pairs at all -- dropping the free
intercept is worth +4.2 points at 160 stored values (0.6156 against 0.5737).
Nor does making the nesting exact help: prepending the 44 class singletons to
the class dictionary, so that 44 code values could reproduce the free intercept
exactly, reads 0.6302 at 160 and 0.6743 at 256 against 0.6290 and 0.6863 for
the plain Gaussian dictionary -- the same conclusion the class-dictionary
section reached about designed atoms.

**The frontier.** Validation selects the arm at each budget; test is read once.

| Parameters | Selected arm | Columns read | Deployed weights | Index bits | Validation | Test |
|---:|---|---:|---:|---:|---:|---:|
| 96 | `coded` pairs181 | 105 | 4,620 | 2,785 | 0.5757 | 0.5549 |
| 112 | `coded` pairs181 | 109 | 4,796 | 3,250 | 0.6075 | 0.5798 |
| 128 | `coded` pairs181 | 119 | 5,236 | 3,714 | 0.6294 | 0.6048 |
| 144 | `none` pairs256 | 152 | 6,688 | 4,321 | 0.6410 | 0.6271 |
| 160 | `coded` pairs256 | 159 | 6,996 | 4,801 | 0.6589 | 0.6290 |
| 176 | `coded` pairs256 | 168 | 7,392 | 5,281 | 0.6652 | 0.6449 |
| 192 | `coded` pairs256 | 176 | 7,744 | 5,761 | 0.6710 | 0.6463 |
| **208** | `coded` pairs256 | 188 | 8,272 | 6,241 | 0.6808 | **0.6606** |
| 224 | `none` pairs256 | 189 | 8,316 | 6,721 | 0.6957 | 0.6713 |
| 240 | `free` pairs256 | 181 | 7,964 | 5,881 | 0.6992 | 0.6717 |
| 256 | `free` pairs256 | 181 | 7,964 | 6,361 | 0.7094 | 0.6790 |
| 272 | `none` pairs256 | 199 | 8,756 | 8,162 | 0.7141 | 0.6887 |
| 288 | `none` pairs256 | 214 | 9,416 | 8,642 | 0.7205 | 0.6994 |
| **304** | `coded` pairs256 | 215 | 9,460 | 9,122 | 0.7213 | **0.7038** |
| 320 | `free` pairs256 | 211 | 9,284 | 8,282 | 0.7260 | 0.7010 |
| 384 | `free` pairs256 | 228 | 10,032 | 10,202 | 0.7387 | 0.7243 |
| 448 | `free` pairs256 | 246 | 10,824 | 12,122 | 0.7563 | 0.7271 |
| 512 | `free` pairs256 | 256 | 11,264 | 14,043 | 0.7597 | 0.7403 |
| 640 | `free` pairs181 | 274 | 12,056 | 17,292 | 0.7695 | 0.7497 |
| 768 | `free` pairs256 | 306 | 13,464 | 21,724 | 0.7762 | 0.7551 |
| 1,024 | `free` pairs181 | 342 | 15,048 | 28,434 | 0.7830 | 0.7556 |

**65%** is still first cleared at **208 stored values**, but at 66.06% rather
than the 65.05% of the section above, and the 176- and 192-value rows close
most of the remaining gap (64.49% and 64.63% against 62.22% and 63.60%).
**70%** falls from 320 to **304 stored values** (70.38%).  Against a dense
affine head's 2,156 and 2,860 values that is a **10.4x** and **9.4x** parameter
reduction.  The honest summary of this section is therefore that it is worth
one point of accuracy at the 65% budget and one budget step at the 70% one; the
much larger numbers are all below 192 values, where the model is not clearing
either target.

Reproduce with:

```bash
python experiments/resisc45_coded_bias.py
```

Results are written to `experiments/resisc45_coded_bias_result.csv` (about
30 GPU-minutes for the 21-budget grid); `resisc45_coded_bias.py --summarise`
re-prints the intercept comparison, the paired-bootstrap intervals, the stored
values each coded arm spends on the constant column, and the frontier from the
existing CSV.

## Distillation buys nothing at these budgets

Every RESISC45 improvement in this file so far changed the head's
*parameterisation*.  Distillation is the obvious lever that changes neither the
head nor the pool: a teacher is a training-time object, so a student that
learns more from soft targets than from hard labels is accuracy for zero stored
values.  The budgeted head is starved -- 65% test against a 79.4% pool ceiling
-- which is the regime a teacher is supposed to help most, and with 45 classes
soft targets carry confusion structure the one-hot labels do not.

`experiments/resisc45_distill.py` fits the previous section's student with

```
(1 - alpha) * CE(hard) + alpha * T^2 * KL(teacher_T || student_T)
```

in *both* the prune-and-regrow search -- the loss it descends and the regrow
criterion, which is the same gradient -- and the convex refit, which stays
convex because a cross-entropy against a fixed target distribution is.  Four
teachers separate the possible mechanisms: the full-pool logistic teacher's own
train logits (`insample`, 95.0% train / 79.4% test), the same teacher 5-fold
cross-fitted so its train targets are honestly uncertain (`xfit`, 79.7% train),
a teacher fitted on the student's own 512 candidate columns (`candidate`, the
control for "the teacher sees columns the student cannot" -- and, at 79.8%
test, actually the *better* teacher of the two), and a *uniform* teacher at
`T = 1`, which is exactly label smoothing at `eps = alpha`.

**Not one arm helps.** Test accuracy against the hard-label head at the same
budget, with paired-bootstrap intervals:

| Arm | 208 values | 256 values |
|---|---:|---:|
| hard labels | 0.6583 | 0.6790 |
| `insample` T=1 a=0.5 | 0.6495 (-0.9 [-1.9, +0.2]) | 0.6752 (-0.4 [-1.4, +0.6]) |
| `insample` T=1 a=0.9 | 0.6441 (-1.4 [-2.5, -0.3]) | 0.6832 (+0.4 [-0.6, +1.3]) |
| `insample` T=2 a=0.5 | 0.6435 (-1.5 [-2.6, -0.4]) | 0.6741 (-0.5 [-1.5, +0.5]) |
| `insample` T=4 a=0.5 | 0.6211 (-3.7 [-4.8, -2.6]) | 0.6486 (-3.1 [-4.2, -1.9]) |
| `xfit` T=1 a=0.5 | 0.6473 (-1.1 [-2.1, -0.1]) | 0.6703 (-0.9 [-1.9, +0.1]) |
| `xfit` T=2 a=0.5 | 0.6440 (-1.4 [-2.5, -0.4]) | 0.6568 (-2.2 [-3.2, -1.2]) |
| `candidate` T=2 a=0.5 | 0.6449 (-1.3 [-2.4, -0.2]) | 0.6725 (-0.7 [-1.7, +0.3]) |
| `smooth` eps=0.1 | 0.6479 (-1.0 [-2.0, +0.1]) | 0.6789 (-0.0 [-1.0, +1.0]) |
| `smooth` eps=0.3 | 0.6346 (-2.4 [-3.4, -1.3]) | 0.6500 (-2.9 [-4.0, -1.8]) |

Of the 32 distilled arms exactly one is above its control (`insample` T=1
a=0.9 at 256 values, +0.4 points with an interval spanning zero), validation
picks a losing arm at 208 and the hard-label arm at 256, and the loss grows
monotonically with temperature for every teacher at both budgets.
Cross-fitting the teacher,
which is the standard fix for a teacher that is too confident on its own
training set, makes the result slightly *worse* rather than better, and label
smoothing -- the control for "any softening regularises" -- is neutral at
`eps = 0.1` and clearly negative at `eps = 0.3`.

The reading that fits all four arms is that this student is not
capacity-limited in the way a small network is.  It is a *linear* head on the
teacher's own features, sharing the teacher's hypothesis class and differing
only in how many values it may store; softening the targets adds no information
it can express and removes gradient signal on the hard decisions its few
weights have to get right.  This is the same conclusion the EuroSAT
convolutional-distillation row in the negative-results table reached from the
opposite direction, where the student's *representation* was the bottleneck.

Reproduce with:

```bash
python experiments/resisc45_distill.py
```

Results are written to `experiments/resisc45_distill_result.csv` (about
15 GPU-minutes); `resisc45_distill.py --summarise` re-prints the comparison
from the existing CSV.

## The standardiser was never free

Every RESISC45 section above rests on one line of the accounting protocol: "the
feature standardizer is folded into the logistic-regression weights, so it adds
no deployed values."  That is exactly true of a head whose weights are
**stored** -- folding `mu`/`sigma` into a stored matrix changes the numbers, not
how many there are -- and it is what makes the dense heads, the element-wise
sparse heads and the class-dictionary head honestly counted.  It stops being
true the moment the weights are **reconstructed**.  A deployment of
`Dc @ P @ Df.T` holds a sparse code and rebuilds
`w_eff = (Dc @ P @ Df.T) / sigma` and `b_eff = b - sum_j W_j mu_j / sigma_j`,
and those two divisions need numbers that are not in the code.
`experiments/resisc45_standardiser.py` prices them and then tries to stop
paying.

**The price list.** `sep_dict_deployed_values` in `experiments/resisc45_lib.py`
counts what a deployment must hold, always taking the cheapest way to pay:

* **The code**, `nnz` values, always.
* **The scaling**, only for column atoms of width >= 2.  A width-1 atom needs
  nothing: `w_eff[:, j] = Dc @ P[:, j] / sigma_j`, so dividing the code entries
  of column `j` by `sigma_j` reproduces it exactly and a per-column `sigma` is
  *absorbable* into values the head already stores.  A width-2 atom ties two
  columns to one value, so its deployed direction
  `(s_i / sigma_i, s_j / sigma_j)` needs their ratio; ratios compose, so the
  bill is one value per column a pair atom touches, minus one per connected
  component of the graph those atoms draw.
* **The intercept**, whenever the head stores one *or* the features are
  centred, because centring at the training mean *is* an intercept: a head with
  `b = 0` still deploys `b_eff = -sum_j W_j mu_j / sigma_j`.  A free intercept
  costs `K - 1 = 44` and covers the fold too, so the two never add up.

Under that price list the sections above split cleanly.  Everything up to and
including the class-dictionary head is counted correctly: those column atoms are
all width 1, and the 44 stored intercepts are the same 44 the fold needs.  The
two most recent sections are not.  The pair dictionary buys width-2 atoms and
never paid for their ratios, and the coded/`none` intercept saved the *second*
copy of `b_eff` rather than the first.

**What the pair dictionary actually costs.** The scaling bill grows with the
budget and does not saturate, because a wider support touches more columns.
The right-hand columns replace `sigma` with the nearest power of two, which is
the fix described below:

| Code values | `zscore` scaling | Deployed | Test | | Octave `sigma` scaling | Deployed | Test |
|---:|---:|---:|---:|---|---:|---:|---:|
| 64 | 59 | 167 | 0.5014 | | 10 | 118 | 0.4905 |
| 96 | 90 | 230 | 0.5810 | | 11 | 151 | 0.5710 |
| 128 | 116 | 288 | 0.6246 | | 11 | 183 | 0.5873 |
| 160 | 140 | 344 | 0.6525 | | 12 | 216 | 0.6357 |
| 208 | 163 | 415 | 0.6752 | | 11 | 263 | 0.6687 |
| 256 | 187 | 487 | 0.7021 | | 12 | 312 | 0.6765 |
| 320 | 202 | 566 | 0.7183 | | 12 | 376 | 0.6994 |
| 384 | 209 | 637 | 0.7265 | | 12 | 440 | 0.7083 |
| 512 | 217 | 773 | 0.7413 | | 13 | 569 | 0.7310 |
| 640 | 229 | 913 | 0.7524 | | 14 | 698 | 0.7292 |

So the operating points the two sections above reported are, priced honestly:
the **208** stored values that first cleared 65% are **424** deployed values --
the `none` arm at 208 code values reproduces the intercept section's 0.6625
exactly, and adds 44 for the fold's `b_eff` and 172 for the ratios -- and the
**304** that first cleared 70% are **542**.

**Rounding `sigma` to a power of two makes the pair dictionary nearly free.**
Two columns in the same octave are scaled by the *same* number, so their pair's
ratio is 1 and the shared factor is absorbable exactly like a width-1 atom's;
pairs across octaves then need one value per octave rather than one per atom.
The 512 candidate columns span 19 octaves, so the whole 65,792-atom pair
dictionary costs **10 to 14 values at every budget** instead of 59 to 229.
Conditioning is barely touched -- each column's scaled standard deviation lands
in `[1/sqrt(2), sqrt(2)]` -- and the price is 0.7 to 2.6 points of test accuracy
at equal *code* size against 128 to 215 fewer deployed values.  Restricting the
enumeration to within-octave pairs (`pairsbuck`, 6,478 atoms) removes the last
10 to 14 values and is a wash against the full enumeration.

**Once the fold is priced, a free intercept is the cheap one.** A centred head
deploys 44 intercept values whether or not it stores any, so `free` and `none`
cost the same and the only question is which fits better.  Over 84 paired cells
-- four column dictionaries times 21 budgets -- `free` wins **77** and loses 7,
by a mean of +1.16 points, at a mean deployed difference of +0.1 values.  The
intercept section's headline does not survive the repricing: it saved 44 stored
values that a deployment has to spend anyway.

**What does beat 44 is coding the intercept -- but only with a converged
refit.** Take the support found on centred features, approximate the deployed
`b_eff` by `q` class-dictionary atoms with matching pursuit, and refit convexly
on *uncentred* features, where the coded intercept is the only intercept there
is.  The result depends almost entirely on how long LBFGS runs, because the
uncentred problem is far worse conditioned than the centred one:

| Code values | Arm | 300 steps | 1,200 steps | 4,000 steps |
|---:|---|---:|---:|---:|
| 208 | `zscore/identity/none+mp16` | 0.5603 | 0.6156 | 0.6427 |
| 208 | `zscore/identity/none+mp44` | 0.5743 | 0.6213 | 0.6460 |
| 208 | `zbuck/pairsbuck/none+mp16` | 0.5613 | 0.6283 | 0.6476 |
| 256 | `zscore/identity/none+mp16` | 0.6011 | 0.6298 | 0.6537 |
| 256 | `zbuck/pairsbuck/none+mp44` | 0.6046 | 0.6446 | 0.6657 |

At 300 steps -- the setting every other refit in this file uses, and the one at
which every centred head is already converged -- the re-coding looks like a
4-to-6-point disaster.  At 4,000 it is a win: `q = 16` costs 16 values instead
of 44 and gives back 0.2 points or less.  The control that makes this safe is
that the *centred* rows do not move at all: 20 base rows measured at both 300
and 4,000 steps are identical to four decimals, so the extra steps buy nothing
except a converged uncentred fit.  Whether the constant column sits inside the
L2 penalty is worth at most 0.3 points either way, with no consistent sign.

**The honest frontier.** Validation picks the arm at each deployed-value
ceiling; test is read once.

| Deployed <= | Selected arm | Values | Bits | Columns | Steps | Validation | Test |
|---:|---|---:|---:|---:|---:|---:|---:|
| 128 | `zbuck` pairs256 `free` | 118 | 5,966 | 79 | 300 | 0.5060 | 0.4905 |
| 160 | `zbuck` pairs256 `free` | 151 | 8,060 | 103 | 300 | 0.5860 | 0.5710 |
| 192 | `zbuck` pairsbuck `free` | 172 | 9,292 | 123 | 300 | 0.6163 | 0.5971 |
| **224** | `zbuck` pairsbuck `free+mp16` | 224 | 13,656 | 169 | 4,000 | 0.6735 | **0.6540** |
| 256 | `zbuck` pairs256 `free+mp16` | 251 | 15,800 | 168 | 4,000 | 0.6935 | 0.6729 |
| 272 | `zbuck` pairs256 `free+mp16` | 251 | 15,800 | 168 | 4,000 | 0.6935 | 0.6729 |
| 288 | `zbuck` pairs256 `free` | 279 | 16,215 | 168 | 300 | 0.6954 | 0.6740 |
| 304 | `zbuck` pairsbuck `free` | 300 | 17,003 | 189 | 300 | 0.6975 | 0.6754 |
| 320 | `zbuck` pairs256 `free` | 312 | 18,269 | 180 | 300 | 0.7076 | 0.6765 |
| 352 | `zbuck` pairs256 `free+mp16` | 348 | 21,880 | 204 | 4,000 | 0.7165 | 0.6902 |
| **384** | `zbuck` pairsbuck `free+mp32` | 384 | 23,156 | 223 | 4,000 | 0.7275 | **0.7100** |
| 416 | `zbuck` pairsbuck `free` | 396 | 22,686 | 223 | 300 | 0.7332 | 0.7119 |
| 448 | `zbuck` pairs256 `free` | 440 | 26,301 | 211 | 300 | 0.7338 | 0.7083 |
| 480 | `zbuck` pairsbuck `free` | 460 | 26,546 | 239 | 300 | 0.7392 | 0.7143 |
| 512 | `zbuck` pairsbuck `free` | 492 | 28,380 | 243 | 300 | 0.7424 | 0.7202 |
| 640 | `zbuck` pairs256 `free` | 537 | 32,360 | 245 | 300 | 0.7529 | 0.7387 |
| 768 | `zscore` pairs256 `free` | 750 | 38,403 | 264 | 300 | 0.7600 | 0.7363 |
| 1,024 | `zscore` pairs256 `free` | 913 | 48,420 | 284 | 300 | 0.7735 | 0.7524 |

**65%** test is first cleared at **224 deployed values** and **70%** at **384**,
against a dense affine head's 2,156 and 2,860 -- which need no repricing,
because a dense head stores its weights.  That is **9.6x** and **7.4x**, and
both sit well inside the 1,024-value target this work was set.  Against the
repriced 424 and 542 above, the octave-rounded standardiser plus a coded
intercept is worth **1.9x** and **1.4x**; against the class-dictionary section,
which was correctly counted at 256 and 448, it is worth 1.1x at both.

**Two ways of not paying for the standardiser that do not work.**

* *Search on uncentred features.* With `mu = 0` the fold supplies no intercept,
  so a coded one is the only intercept there is and the centring bill is zero
  from the start.  The prune-and-regrow search then collapses to 2-4% test at
  every budget: with a median `|mu| / sigma` of 3.4 over the pool, the dense
  loss gradient at `w = 0` is dominated by the column means, which point in
  nearly the same class direction for every column, so the initial support and
  every regrow step spend themselves on rank-1 mean structure.  Centring is not
  a conditioning convenience for this search, it is what makes the regrow
  criterion informative -- which is why the re-coding above searches on centred
  features and only *refits* uncentred.
* *Share `sigma` per feature family.* Within-family standard deviations span up
  to 16.7 octaves in this pool, so a shared family `sigma` reproduces the
  uncentred failure exactly (3.98% test at 160 code values).  Octaves work
  because they bound the distortion to a factor of `sqrt(2)`; families do not,
  because a family is not a scale.

Reproduce with:

```bash
R=experiments/resisc45_standardiser
MAIN="zscore/pairs256/free zscore/pairs256/none zscore/identity/none \
      zbuck/pairsbuck/none zbuck/pairs256/none"
FREE="zscore/pairs256/free zbuck/pairs256/free zbuck/pairsbuck/free \
      zscore/identity/free"
BUCK="zbuck/pairs256/free zbuck/pairsbuck/free"
FINE="176 208 240 272 304 352 416 480"
python $R.py                                                       --out ${R}_a.csv
python $R.py --budgets $FINE --arms $MAIN                          --out ${R}_b.csv
python $R.py --budgets $FINE --arms $FREE                          --out ${R}_c.csv
python $R.py --budgets 208 256 352 448 --unpenalised-bias \
  --arms zscore/pairs256/free zscore/identity/none zbuck/pairsbuck/none \
                                                                   --out ${R}_d.csv
python $R.py --budgets 208 256 --steps 1200 --unpenalised-bias \
  --arms zscore/identity/none zbuck/pairsbuck/none                 --out ${R}_e.csv
python $R.py --budgets 208 256 --steps 4000 \
  --arms zscore/identity/none zbuck/pairsbuck/none                 --out ${R}_f.csv
python $R.py --budgets 192 208 224 240 --steps 4000 --mp-grid 8 16 32 \
  --arms $BUCK --mp-arms $BUCK                                     --out ${R}_g.csv
python $R.py --budgets 288 320 352 384 --steps 4000 --mp-grid 8 16 32 \
  --arms $BUCK --mp-arms $BUCK                                     --out ${R}_h.csv
python $R.py --merge ${R}_[a-h].csv
```

The passes are independent and were run two at a time across two GPUs; `--merge`
keys rows by `(budget, arm, steps)` and refuses to combine files that disagree,
which is also the cross-GPU determinism check.  The committed CSV is the union
of what was actually run: re-running the list above additionally fills in the
`q = 44` re-coding at every budget rather than only at four, because `MP_GRID`
gained that entry once the `q = 32` curve turned out to be flat.  Results are written to
`experiments/resisc45_standardiser_result.csv` (about four GPU-hours in total,
most of it the 4,000-step refits); `resisc45_standardiser.py --summarise`
re-prints the price list, the equal-code tables, the intercept re-coding, the
convergence table and the honest frontier from the existing CSV.


## Towards 80% under 4,096 values: the headroom map and where the budget goes

The sections above chase the *smallest* head that clears 65% and 70%. The next target is the opposite corner: **80% test under 4,096 deployed values**. The pool's own linear ceiling is 79.4% test, so no head on the existing columns can get there, and the first job is to find out what is missing -- information, nonlinearity, or head capacity -- and how much of each a budgeted head can buy. `experiments/resisc45_budget4096.py`, `resisc45_nonlinear_probe.py`, `resisc45_gpu_features4.py`, `resisc45_pool4_ceiling.py`, `resisc45_candidate_lists.py`, `resisc45_expanded_head.py` and `resisc45_head_regime.py` answer that in turn. Every arm is fitted on train with `C`, weight decay and the head structure chosen on validation and test read once; deployed values follow the accounting of "The standardiser was never free".

**Where the existing heads stand at 4,096.** Extending the earlier frontiers past 1,024 values, the incumbent pair-dictionary head reads 77.7% test at 4,385 deployed values (4,096 code values plus the intercept and 245 scaling ratios), and the class-dictionary head over the 512-column list reads 77.5% at exactly 4,096. Both are 2.5 points short of the target and within two points of the linear ceiling, so a better head on the same columns cannot close the gap.

| Head | 1,024 | 2,048 | 4,096 |
|---|---:|---:|---:|
| Element-wise prune-and-regrow, whole 2,084-column pool | 0.7211 | 0.7435 | 0.7652 |
| Class dictionary (16,384 atoms), whole pool | 0.7408 | 0.7471 | 0.7490 |
| Class dictionary (16,384 atoms), 512-column list | 0.7595 | 0.7673 | 0.7754 |
| Pair dictionary over the 512-column list, `zscore/pairs256/free` | 0.7600 (1,303) | 0.7700 (2,333) | 0.7768 (4,385) |
| Same with power-of-two `sigma`, `zbuck/pairs256/free` | 0.7497 (1,082) | 0.7641 (2,106) | 0.7640 (4,152) |
| Dense affine head, whole pool (91,740 values) | | | 0.7935 |

Test accuracy; deployed values in brackets where they differ from the column. The power-of-two `sigma` that was almost free below 640 values costs 0.6-1.3 points here, so the standardiser saving does not carry to large budgets.

**The pool is hiding nonlinear information from a linear head.** A one-hidden-layer ReLU head on the same 2,084 columns reads 82.6% test at width 1,024, three points above the linear ceiling. `resisc45_nonlinear_probe.py` asks which *zero-parameter* expansion of the columns recovers that, because an expansion is just more columns and the sparse machinery can then buy the ones it wants:

| Expansion of the pool | Columns | Validation | Test |
|---|---:|---:|---:|
| Linear (control) | 2,084 | 0.8125 | 0.7935 |
| Hinge at the column mean, `[z, relu(z)]` | 4,168 | 0.8230 | 0.8102 |
| Hinges at `z = -1, 0, 1` | 8,336 | 0.8244 | 0.8127 |
| ReLU of 4,096 random signed column pairs | 6,180 | 0.8335 | 0.8217 |
| ReLU of 16,384 random signed column pairs | 18,468 | 0.8406 | 0.8268 |
| ReLU MLP, width 16 / 32 / 64 / 128 / 512 | 2,084 | 0.7692 / 0.7959 / 0.8162 / 0.8263 / 0.8392 | 0.7525 / 0.7844 / 0.8016 / 0.8108 / 0.8267 |

Two things follow. A single hinge at each column's mean -- the response `|z|` in effect -- is worth 1.7 test points on its own and the extra knots almost nothing, so most of the additive nonlinearity is "distance from the typical value". And ReLUs of random *pairs* of standardised columns match the widest MLP with no learned first layer at all: the interaction information is width-2, exactly as it was on the linear head in "A column dictionary too". The same probe restricted to the 512-column candidate list reads 79.8% linear, 80.5% hinged, 83.0% with 16,384 pair ReLUs and 83.2% with a width-512 MLP, so the nonlinear headroom is inside the columns the budgeted heads already read.

**A fourth pool: random local features and colour-conditioned texture.** Every column so far is a global statistic of a hand-designed map. `resisc45_gpu_features4.py` adds 2,042 columns that summarise *which local patterns occur*: 1,536 MOSAIKS-style random convolutional features (128 seeded zero-mean `5 x 5 x 3` filters at three scales, ReLU on both signs, mean- and max-pooled), 384 random texton histograms (the sign pattern of six seeded `3 x 3` filters hashes each pixel into 64 codes), 66 colour-conditioned texture statistics (gradient energy, blob compactness, brightness and grid spread inside eleven fixed hue/grey masks) and 56 order statistics of `4 x 4` grid cells. All are deterministic given one seed, so the extractor still stores nothing; the whole pool extracts in about a GPU-minute.

| Pool | Columns | Validation | Test |
|---|---:|---:|---:|
| Merged pool (control) | 2,084 | 0.8125 | 0.7935 |
| + grid-cell order statistics | 2,140 | 0.8171 | 0.7960 |
| + colour-conditioned texture | 2,150 | 0.8214 | 0.8029 |
| + random convolutional features | 3,620 | 0.8176 | 0.8052 |
| + random texton histograms | 2,468 | 0.8148 | 0.7902 |
| + all four (merged + new) | 4,126 | 0.8254 | 0.8063 |
| Fourth pool alone | 2,042 | 0.6911 | 0.6803 |
| Hinge expansion of merged + new | 16,504 | 0.8359 | 0.8232 |
| Width-256 ReLU MLP on merged + new | 4,126 | 0.8422 | 0.8249 |

The new pool moves the linear ceiling from 79.4% to 80.6% test, with the random convolutional and colour-conditioned families carrying it and the texton histograms worth nothing. The colour-conditioned family is 66 columns for +0.9 points, the best per-column addition since the native-resolution rebuild; the random convolutional family is +1.2 points for 1,536 columns, each individually weak, which is the MOSAIKS shape recorded in the negative-results table.

**A single ranking over the widened pool is the wrong candidate list.** Every earlier dictionary head reads a 512-column list ranked by group lasso. Re-ranking the 4,126-column pool the same way gives 163 of 512 slots to the texton histograms and 163 to the object-layout pool, and the list's linear ceiling drops from 79.8% (the incumbent list) to 76.8%; the dictionary head on it reads 74.6-75.0% at every budget, *below* the incumbent on the old pool. Ranking the nonlinearly expanded pool is worse still: the pair-ReLU columns take 480 of the 504 columns the head ends up using and it reads 71-74%. This is iteration 3's lesson again, a family that is individually strong and collectively redundant floods a per-column ranking, and the fix is the same: `resisc45_candidate_lists.py` builds a **quota list** of 384 base, 128 layout and 128 fourth-pool columns from three separate rankings, whose 640 columns read **81.0% test** linearly -- above the whole old pool.

**What the head can buy.** `resisc45_expanded_head.py` fits the incumbent pair-dictionary head on the quota list and on three ways of admitting expanded columns, charging each hinge column its knot and each pair-ReLU column its ratio and offset on top of the base accounting:

| Candidate list | Columns | ~1,300 | ~2,300 | ~3,400 | ~3,900 |
|---|---:|---:|---:|---:|---:|
| Quota list, raw columns | 640 | 0.7662 (1,294) | 0.7789 (2,326) | 0.7711 (3,351) | 0.7748 (3,866) |
| Quota list plus hinge columns | 1,280 | 0.7595 (1,506) | 0.7662 (2,559) | 0.7756 (3,700) | 0.7790 (4,197) |
| Support of an element-wise 2,048-value search over raw + hinge + 8,192 pair ReLUs | 1,745 | 0.7589 (2,352) | 0.7776 (3,912) | 0.7748 (5,034) | 0.7725 (5,376) |
| Support of a 4,096-value search, same pool | 3,035 | 0.7613 (2,464) | 0.7722 (3,838) | 0.7803 (5,290) | 0.7775 (5,820) |
| Element-wise head over the whole expanded pool | 9,472 | 0.7327 (2,807) | 0.7619 (5,374) | 0.7671 (7,728) | 0.7690 (8,854) |

Test accuracy (deployed values). The hinge and pair-ReLU columns are real at the ceiling and useless under the budget: the head spends 221-341 knots and up to 1,900 pair constants for nothing, because it buys the expanded columns one at a time and their value is collective -- the dense head uses thousands of them at small weights, the same reason random convolutional features needed 4,617 head parameters on EuroSAT. The raw quota list is the only one that pays, and only up to about 2,300 values, after which the head plateaus at 77-78% while its own list has a linear ceiling of 81.0%. That plateau, not the pool, is now the binding constraint.

**The plateau is the search, and its weight decay is the lever.** `resisc45_head_regime.py` holds the 640-column quota list fixed and varies only the head, reporting train accuracy next to test:

| Head on the quota list | 1,068 values | 2,092 values | 3,628 values |
|---|---:|---:|---:|
| Dense affine (28,204 values) | | | 0.8102 (train 0.940) |
| Element-wise prune-and-regrow over the list | 0.7344 | 0.7657 | 0.7849 |
| Class dictionary, 256 atoms | 0.7489 | 0.7822 | 0.7938 |
| Class dictionary, 1,024 atoms | 0.7643 | 0.7790 | 0.7913 |
| Class dictionary, 4,096 atoms | 0.7659 | 0.7824 | 0.7778 |
| Class dictionary, 16,384 atoms | 0.7698 | 0.7775 | 0.7844 |
| Class dictionary, 16,384 atoms, search weight decay 1e-3 | 0.7667 | **0.7917** | **0.7957** |

Test accuracy; every row deploys exactly the code plus 44 intercepts, because identity column atoms absorb the standardiser. Two regime changes show up. Atom count, which bought 6 points at 256 values, is worth nothing at 3,628 (256 atoms match 16,384), so the over-complete class alphabet only matters while the head is starved. And a ten-fold stronger weight decay *during the prune-and-regrow search* -- the same 1e-4 has been used since iteration 4, tuned at 512 values -- is worth +1.4 points at 2,092 values and +1.1 at 3,628, with the highest validation accuracy of any arm at both budgets (0.8103 and 0.8141). The convex refit already selects `C` on validation, so this is a property of the *support* the search finds, not of the weights: at large budgets the search has enough freedom to overfit which entries it keeps.

**The search is chaotic, and a single run is inside the noise.** `resisc45_search_decay.py` sweeps the search decay from 3e-4 to 2e-3 and the drop fraction over 0.3/0.5/0.7 on the same list, and then repeats the best settings under decay perturbations of 0.2% per seed (the search draws no randomness, so a seed changes nothing on its own; a perturbation that small is a measurement of the search's own chaos, not of the decay):

| Single search, 3,628 deployed values | Runs | Mean test | SD | Min | Max |
|---|---:|---:|---:|---:|---:|
| Decay 1e-3, drop 0.5 (the frontier above) | 8 | 0.7895 | 0.0083 | 0.7741 | 0.7976 |
| Decay 1e-3, drop 0.3 | 8 | 0.7953 | 0.0030 | 0.7917 | 0.7990 |
| Decay 1.5e-3, drop 0.3 | 8 | 0.7960 | 0.0070 | 0.7829 | 0.8025 |
| Decay 1.5e-3, drop 0.5 | 8 | 0.7916 | 0.0046 | 0.7846 | 0.7984 |
| All 32 | 32 | 0.7931 | 0.0064 | 0.7741 | 0.8025 |

Two settings that differ in the fourth significant figure of the decay give supports whose test accuracy differs by up to 2.3 points. The 18-arm decay sweep reads 0.7687-0.8000 at 3,628 values with a standard deviation of 0.9 points, decays of 3e-3 and 1e-2 read 0.7759-0.7968 at 2,092-3,628 values, inside or below the single-search distribution and never above it, no wider quota list (512/128/128, 384/128/256, 384/256/128, 512/192/192) beats the 640-column incumbent, and validation (standard deviation 0.5 points over the same runs) picks the 79.6% arm from the sweep and the 80.2% arm from the repeats. Three of the 32 repeats read 80% or better; a run that does is a draw, not a method.

**A union of supports clears 80%.** The chaos is in *which* entries the search keeps, and different runs keep different good ones. The `union` arm of the same script runs the search several times under those 0.2% decay perturbations, takes the union of the supports (about 1.6x the budget for two searches and 3.6x for eight), refits it convexly, and magnitude-prunes it back to the budget in three geometric steps, each step a fresh convex refit at `C = 0.1`; the pruned support is then refitted at the validation-chosen `C` like every other head. It is iterative magnitude pruning from a union of prune-and-regrow supports rather than from a dense head:

| Union of searches, decay 1.5e-3, drop 0.3 | 3,116 values | 3,628 values |
|---|---:|---:|
| 2 searches (seeds 0-1) | 0.7984 | 0.8005 |
| 4 searches (seeds 0-3) | 0.8021 | 0.8021 |
| 8 searches (seeds 0-7) | 0.7992 | **0.8049** |
| 4 searches, disjoint seeds 16-19 | 0.7989 | 0.8041 |
| 8 searches, disjoint seeds 16-23 | 0.7946 | 0.7998 |
| 16 searches, disjoint seeds 16-31 | 0.7975 | 0.7986 |
| 8 searches, six prune steps | | 0.8021 |
| 8 searches, pruning refits at `C = 0.03` (highest validation, 0.8225) | | 0.8021 |
| Same at decay 1e-3, 2 / 4 / 8 searches | 0.7970 / 0.7978 / 0.7979 | 0.7995 / 0.7987 / 0.7995 |
| Control: one search at 2x the budget, pruned to the budget | | 0.7873 |
| Control: one search at 4x the budget, pruned to the budget | | 0.7895 |
| Control: four searches at 4x the budget, union pruned to the budget | | 0.7951 |

Test accuracy. At 3,628 deployed values the eleven union heads average **0.8011 with a standard deviation of 0.21 points** (0.7986-0.8049), against 0.7931 +/- 0.64 for a single search at the same settings; six of the eleven read 80% or better against three of 32 single runs, and the two disjoint search sets agree. The controls say where the gain comes from: pruning one over-provisioned search reads no better than a plain search, so it is the *union* of same-budget supports that matters, not the pruning. It is a variance reduction on the support -- stability selection by another route, but one that keeps the regrow criterion's decorrelation by pruning with a convex refit instead of voting, which is why it wins where the frequency vote in the negative-results table lost. Above eight searches the union grows past 5x the budget and the pruning has to discard more than it keeps, and the gain fades; four to eight is the range. The budget slack buys nothing: the same head at 3,884 and 4,076 deployed values reads 0.7992 and 0.8006, and at 2,092 and 2,604 it reads 0.790-0.791 and 0.794-0.798.

**The 4,096-value frontier after this section.** The union-of-eight head on the 640-column quota list at 3,584 code values reads **80.49% test at 3,628 deployed values** (validation 0.8179), and the validation pick among the union arms at that budget reads 80.21% (validation 0.8225); the method's expected test accuracy at 3,628 values is 80.1% +/- 0.2 over twelve settings and two disjoint search sets. That clears the 80% target under 4,096 values by a margin of about one standard deviation of the *method*, not of a single run, which is what the earlier 79.57% frontier was. What is left on the table is unchanged: the list's linear ceiling is 81.0% and its MLP ceiling 83%; the remaining lever is a head with a few learned hidden units, or a search that generalises better than a union of chaotic ones.

Reproduce with:

```bash
python experiments/resisc45_budget4096.py
python experiments/resisc45_standardiser.py --budgets 1024 2048 4096 \
  --arms zscore/pairs256/free zbuck/pairs256/free --mp-arms \
  --out experiments/resisc45_standardiser_4096.csv
python experiments/resisc45_nonlinear_probe.py
python experiments/resisc45_gpu_features4.py
python experiments/resisc45_pool4_ceiling.py
python experiments/resisc45_candidate_lists.py
python experiments/resisc45_expanded_head.py --ranking plain --out experiments/resisc45_expanded_head_plainrank.csv
python experiments/resisc45_expanded_head.py
python experiments/resisc45_head_regime.py
python experiments/resisc45_search_decay.py --arm decay --decays 3e-4 5e-4 7e-4 1e-3 1.5e-3 2e-3 --drops 0.3 0.5 0.7 --seeds 0
python experiments/resisc45_search_decay.py --arm quota --decays 5e-4 1e-3 1.5e-3 --budgets 3584 --seeds 0 \
  --out experiments/resisc45_search_decay_quota.csv
python experiments/resisc45_search_decay.py --arm decay --decays 3e-3 1e-2 --drops 0.5 --seeds 0 \
  --out experiments/resisc45_search_decay_coarse.csv
python experiments/resisc45_search_decay.py --arm decay --decays 1e-3 1.5e-3 --drops 0.3 0.5 --seeds 0 1 2 3 4 5 6 7 \
  --budgets 3584 --out experiments/resisc45_search_decay_noise.csv
python experiments/resisc45_search_decay.py --arm union --decays 1e-3 1.5e-3 --drops 0.3 --budgets 3072 3584 \
  --union-seeds 2 4 8 --out experiments/resisc45_search_decay_union.csv
```

The union table's replicate, low-budget, prune-variant, control and high-budget rows are the same command with `--seed-offset 16 --union-seeds 4 8 16`, `--budgets 2048 2560 --union-seeds 8`, `--prune-rounds 6` or `--prune-c 0.03`, `--union-seeds 1 4 --search-scale 2` or `4`, and `--budgets 3840 4032`; they are merged into the one CSV.

Results are written to `experiments/resisc45_budget4096_result.csv`, `experiments/resisc45_standardiser_4096.csv`, `experiments/resisc45_nonlinear_probe_result.csv`, `experiments/resisc45_pool4_ceiling_result.csv`, `experiments/resisc45_candidate_lists_result.txt`, `experiments/resisc45_expanded_head_plainrank.csv`, `experiments/resisc45_expanded_head_result.csv`, `experiments/resisc45_head_regime_result.csv`, `experiments/resisc45_search_decay_result.csv`, `experiments/resisc45_search_decay_quota.csv`, `experiments/resisc45_search_decay_coarse.csv`, `experiments/resisc45_search_decay_noise.csv` and `experiments/resisc45_search_decay_union.csv`; the whole section is about 75 GPU-minutes.

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
| Wide column atoms for the RESISC45 dictionary head | Widening the alphabet on the *column* axis only pays at width 2: 16,896 random atoms of width 4, 8 and 32 read 0.6560/0.6429/0.6417 test at 256 stored values and 16,896 dense random directions read 0.6557, against 0.6519 for the raw columns and 0.6790 for enumerated pairs; unlike the class axis, atom count alone buys nothing here, because a class's column pattern is sparse and a wide atom charges it for columns it does not want |
| Larger pair enumerations for the RESISC45 column dictionary | Enumerating every signed pair of the top 320 ranked columns (102,592 atoms) is no better than the top 256 at 256 stored values (0.6765 against 0.6790) and 0.9-1.1 points worse at 512 and 1,024, where the atom count starts to overfit 18,900 training images |
| Semantically chosen class groups for the RESISC45 dictionary head | Ward class groups from the pool or from the dense head's own rows beat size-matched random groups by 0.9 points at 256 stored values and by 0.03 at 1,024; the atom count is worth several times more than the atom content, and a random draw is as good as a designed one |
| Distillation into the budgeted RESISC45 head | A 79.4%-test full-pool logistic teacher, cross-fitted or not, plus a candidate-list teacher and label smoothing as controls, all lose: 31 of 32 distilled arms fall below the hard-label head at 208 and 256 stored values, by up to 5.8 points, and the loss grows monotonically with temperature. A linear student in the teacher's own hypothesis class gains nothing from softened targets and loses gradient signal on the decisions its few weights must get right |
| A free intercept for the RESISC45 dictionary head | The 44 intercepts are worth 0 to 3 stored values, not 44: a head with no intercept in the standardised space matches or beats a coded one at every budget, because the standardiser fold already supplies `b_eff = -sum_j W_j mu_j / sigma_j` and the splits are class-balanced. Above 192 stored values the saving is inside the split's resolution, so it only matters where the head is starved |
| Fitting the RESISC45 head on uncentred features | Dropping the centring removes the fold's 44-value deployed intercept at the source, but the prune-and-regrow search then collapses to 2-4% test at every budget: with a median `\|mu\| / sigma` of 3.4 the dense loss gradient at `w = 0` is dominated by the column means, which point in nearly the same class direction for every column, so the support fills with rank-1 mean structure. Centring is what makes the regrow criterion informative; the search has to see centred features even when the deployed head does not |
| Sharing one `sigma` per feature family | Within-family standard deviations span up to 16.7 octaves in this pool, so a family-shared `sigma` reproduces the uncentred collapse exactly (3.98% test at 160 code values). Sharing per power-of-two octave works instead, because it bounds the distortion to `sqrt(2)` while still making within-octave pair atoms cost nothing |
| Class singletons in the RESISC45 class dictionary | Prepending the 44 identity atoms to the Gaussian class dictionary, which makes a coded intercept able to reproduce a free one exactly, reads 0.6302 at 160 stored values and 0.6743 at 256 against 0.6290 and 0.6863 without them -- the same verdict on designed atoms the class-dictionary section reached |
| Buying nonlinear columns one at a time for the RESISC45 head | Hinge-at-mean and random pair-ReLU expansions lift the pool's linear ceiling from 79.4% to 81.0% and 82.7% test, but a budgeted dictionary head that admits them reads no better than on raw columns at any budget up to 5,800 deployed values and pays 220-1,900 extra constants for the privilege; their value is collective (the dense head spreads small weights over thousands of them), which is the MOSAIKS shape again |
| One group-lasso ranking over the widened 4,126-column RESISC45 pool | Floods the 512-column list with 163 random-texton and 163 object-layout columns, drops the list's linear ceiling from 79.8% to 76.8% test and the dictionary head to 74.6-75.0%; the quota list from three separate rankings reads 81.0% with 640 columns |
| Random texton histograms for RESISC45 | 384 columns from the sign pattern of six random `3 x 3` filters move the merged ceiling by -0.3 points; the rotation-invariant LBP family already carries what they measure |
| Power-of-two `sigma` above 1,024 RESISC45 values | The octave-rounded standardiser that cost 0.7-2.6 points below 640 values costs 0.6-1.3 points at 1,024-4,096, where the 230 scaling ratios it saves are under 6% of the budget |
| More class atoms for the RESISC45 head at large budgets | 256, 1,024, 4,096 and 16,384 atoms read 0.7938/0.7913/0.7778/0.7844 test at 3,628 values on the same list; the over-complete alphabet that bought +6 points at 256 values buys nothing once the head is not starved |
| Search weight decay above 2e-3 for the RESISC45 dictionary head | The ten-fold increase from 1e-4 to 1e-3 was worth a point, but 3e-3 and 1e-2 read 0.7759-0.7968 at 2,092-3,628 values, inside or below the single-search distribution (0.7931 +/- 0.0064) and never above it; the penalty is a sum over code entries, so it already scales with the budget, and 7e-4 to 2e-3 is flat inside the noise |
| Wider quota lists for the RESISC45 head at 3,628 values | 512/128/128, 384/128/256, 384/256/128 and 512/192/192 columns read 0.7721-0.7968 test against 0.7846-0.7976 for the 640-column 384/128/128 list at the same three decays; once the search is the constraint, more candidates only give it more to overfit |
| Pruning one over-provisioned prune-and-regrow search for RESISC45 | Searching at 2x or 4x the budget and magnitude-pruning back with convex refits reads 0.7873/0.7895 at 3,628 values, inside the single-search distribution (0.7931 +/- 0.0064); the union of several same-budget searches pruned the same way reads 0.8011 +/- 0.0021, so the gain is the union, not the pruning |
| Budget slack above 3,628 values for the RESISC45 union head | 3,884 and 4,076 deployed values read 0.7992 and 0.8006 against 0.8049 at 3,628; the head is not budget-limited between 3,000 and 4,100 values |

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
python experiments/resisc45_feature_dict.py
python experiments/resisc45_coded_bias.py
python experiments/resisc45_distill.py
python experiments/resisc45_standardiser.py
python experiments/resisc45_budget4096.py
python experiments/resisc45_nonlinear_probe.py
python experiments/resisc45_gpu_features4.py
python experiments/resisc45_pool4_ceiling.py
python experiments/resisc45_candidate_lists.py
python experiments/resisc45_expanded_head.py
python experiments/resisc45_head_regime.py
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

The baseline's complete `C` sweep and per-class test results are stored in `experiments/image_statistics_baseline_result.txt`. The selected-feature and ImageStats five-seed fraction results are stored in `experiments/eval_training_fractions_result.csv` and `experiments/eval_imagestats_fractions_result.csv`, including model metadata, split protocol, individual seed accuracies, mean, and sample standard deviation. The coordinate-only MLP screen and multi-seed frontier are stored in `experiments/coordinate_mlp_screen.csv` and `experiments/coordinate_mlp_result.csv`. The preliminary RESISC45 transfer, including exact selected feature indices and names, is stored in `experiments/resisc45_33_feature_fractions.csv`. The RESISC45 object-layout experiments write `experiments/resisc45_layout_gain_result.csv`, `experiments/resisc45_layout_gain_classes.csv`, `experiments/resisc45_layout_diagnose_result.csv`, and `experiments/resisc45_layout_frontier_result.csv`, the prune-and-regrow comparison writes `experiments/resisc45_rigl_result.csv`, the degree-2 product experiment writes `experiments/resisc45_quadratic_result.csv`, the support-search probe writes `experiments/resisc45_support_probe_result.csv`, the class-dictionary head writes `experiments/resisc45_class_dict_result.csv`, the column-dictionary head writes `experiments/resisc45_feature_dict_result.csv`, the intercept comparison writes `experiments/resisc45_coded_bias_result.csv`, the distillation comparison writes `experiments/resisc45_distill_result.csv`, the standardiser repricing writes `experiments/resisc45_standardiser_result.csv` (the single command above runs only the first of its nine passes; the full command list is in that section), and the 4,096-value headroom section writes `experiments/resisc45_budget4096_result.csv`, `experiments/resisc45_nonlinear_probe_result.csv`, `experiments/resisc45_pool4_ceiling_result.csv`, `experiments/resisc45_candidate_lists_result.txt`, `experiments/resisc45_expanded_head_result.csv` and `experiments/resisc45_head_regime_result.csv`. Dataset download and fraction evaluators verify the official TorchGeo checksums. Submission evaluation scripts recompute features from raw patches rather than relying on cached feature matrices.
