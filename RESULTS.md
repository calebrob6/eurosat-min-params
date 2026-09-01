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

## TorchGeo spatial split and training fractions

The 306-parameter model was also evaluated on TorchGeo's longitude-based `EuroSATSpatial` split. It contains the same 27,000 patches and the same 16,200/5,400/5,400 train/validation/test counts as the default split, but assigns geographically separated longitude regions to each partition. The spatial partitions are class-imbalanced, so the table reports both sample-weighted accuracy and balanced accuracy, the unweighted mean recall across the 10 classes.

The 33-feature subset and `C=3` regularization remain fixed. At each fraction, the standardizer and affine head are refit using a stratified subset of the spatial training partition. Fractions below 100% report the mean and standard deviation across seeds 0–9; spatial validation and test remain fixed and are never merged into spatial head training. The deployed model remains 306 parameters at every fraction.

**This is a post-hoc repartition stress test, not an independent model-selection benchmark.** The feature subset and `C` were previously selected using the default random train/validation partitions of the same 27,000 samples. Of the 5,400 spatial-test samples, 4,362 (80.8%) appeared in random train or validation and therefore could have influenced feature selection, although their spatial-test labels are not used when fitting any head reported below.

| Spatial train fraction | Images | Validation accuracy | Balanced validation | Test accuracy | Balanced test |
|---:|---:|---:|---:|---:|---:|
| 1% | 162 | 0.8584 ± 0.0237 | 0.8324 ± 0.0259 | 0.8543 ± 0.0228 | 0.8179 ± 0.0291 |
| 2% | 324 | 0.8801 ± 0.0173 | 0.8568 ± 0.0228 | 0.8733 ± 0.0085 | 0.8411 ± 0.0180 |
| 5% | 810 | 0.8980 ± 0.0122 | 0.8780 ± 0.0162 | 0.8985 ± 0.0112 | 0.8671 ± 0.0138 |
| 10% | 1,620 | 0.9177 ± 0.0070 | 0.9020 ± 0.0080 | 0.9055 ± 0.0066 | 0.8778 ± 0.0066 |
| 20% | 3,240 | 0.9283 ± 0.0047 | 0.9131 ± 0.0056 | 0.9169 ± 0.0058 | 0.8902 ± 0.0062 |
| 50% | 8,100 | 0.9377 ± 0.0034 | 0.9246 ± 0.0041 | 0.9247 ± 0.0038 | 0.8982 ± 0.0038 |
| 100% | 16,200 | 0.9459 | 0.9344 | **0.9276** | **0.9017** |

Under the spatial repartition, using all training data reduces test accuracy from 96.04% on the default random split to 92.76%, a 3.28-point gap. The full-data spatial model is strongest on SeaLake (99.59%), Forest (99.07%), and River (96.37%), while Pasture (71.30%), AnnualCrop (83.99%), Highway (85.59%), and PermanentCrop (86.64%) account for most of the performance loss. The learning curve remains data-limited at the top end: moving from 50% to 100% raises test accuracy from 92.47% to 92.76%.

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
python experiments/eval_spatial_fractions.py --download-splits
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

The baseline's complete `C` sweep and per-class test results are stored in `experiments/image_statistics_baseline_result.txt`. Spatial-fraction summaries, individual seed runs, and full-data per-class results are stored in `experiments/eval_spatial_fractions_result.txt`. The spatial evaluator verifies the official TorchGeo split checksums and remaps the existing 27,000-sample fixed-feature caches by filename. Submission evaluation scripts recompute features from raw patches rather than relying on cached feature matrices.
