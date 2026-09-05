# EuroSAT release results

This document covers the reproducible EuroSAT measurements in **Solving EuroSAT with as Few Parameters as Possible**. The complete RESISC45 research history is preserved on the [`resisc45` branch](https://github.com/calebrob6/eurosat-min-params/tree/resisc45). Earlier draft descriptions are historical; [REPRODUCIBILITY.md](REPRODUCIBILITY.md) lists the exact release scope and article discrepancies.

## Raw-data headline results

`python reproduce.py --download --check` recomputes every fixed feature from the original TIFFs. The three handwritten-feature heads are checked-in models; ImageStats is refit on train and C is selected only on validation.

| Model | Features | Parameters | Validation correct / 5,400 | Test correct / 5,400 | Test accuracy |
|---|---:|---:|---:|---:|---:|
| ImageStats | 52 | 477 | 4,910 | 4,912 | 90.96% |
| Submission 12 | 18 | 171 | 5,079 | 5,094 | 94.33% |
| Submission 13 | 30 | 279 | 5,147 | 5,150 | 95.37% |
| Submission 14 | 33 | 306 | 5,193 | 5,186 | 96.04% |

The fixed-feature 306-value head uses C=3. ImageStats uses C=300 selected from the sweep in `results/imagestats_c_sweep.csv`. Detailed counts and per-class scores are in `results/eurosat_models.csv` and `results/eurosat_per_class.csv`; the exact ordered feature list is `results/eurosat_306_features.csv`.

## Five-seed training fractions

Overall test accuracy, mean +/- sample standard deviation over stratified subsample seeds 0-4. Each fraction is relative to the 16,200-image training split; the 5,400 test images stay fixed. The 100% fits are deterministic. C and the selected representation are fixed from full-data experiments, not independently selected with each label budget.

| Train fraction | Train images | 306-value random | 306-value spatial | ImageStats random | ImageStats spatial |
|---|---:|---:|---:|---:|---:|
| 1% | 162 | 86.74 +/- 1.03 | 86.51 +/- 2.32 | 72.55 +/- 0.72 | 68.18 +/- 5.12 |
| 2% | 324 | 89.36 +/- 0.86 | 87.42 +/- 1.00 | 76.95 +/- 1.65 | 71.80 +/- 3.87 |
| 5% | 810 | 92.88 +/- 0.31 | 89.96 +/- 1.33 | 81.96 +/- 0.86 | 72.69 +/- 3.16 |
| 10% | 1,620 | 94.24 +/- 0.26 | 90.54 +/- 0.83 | 86.05 +/- 0.24 | 77.41 +/- 2.82 |
| 20% | 3,240 | 94.91 +/- 0.17 | 91.69 +/- 0.35 | 88.56 +/- 0.19 | 81.10 +/- 1.75 |
| 50% | 8,100 | 95.70 +/- 0.11 | 92.70 +/- 0.27 | 90.32 +/- 0.21 | 83.59 +/- 1.22 |
| 100% | 16,200 | 96.04 | 92.76 | 90.96 | 85.43 |

Reproduce with `python reproduce.py --download --fractions --check`. Expected per-seed measurements remain in `experiments/eval_training_fractions_result.csv` and `experiments/eval_imagestats_fractions_result.csv`; newly computed CSVs are written to `output/reproduce/`.

**Spatial caveat:** these are post-hoc repartition experiments, not independent geographic model-selection benchmarks. Of the 5,400 spatial-test images, 4,362 were in the original random train or validation sets used to choose the feature recipe. No spatial-test labels enter the fraction-specific head fits.

## Method and selection history

The fixed descriptor combines per-band distributions, multiscale gradients, orientation entropy, index-map texture, Hough lines, Harris corners, local binary patterns, connected components, and region-shape anisotropy. The numerical definition of each selected feature is frozen in code rather than inferred from a physical-sounding alias.

The original >96% candidate used 64 features and 585 reference-class parameters. Backward elimination, a regularization retune, a floating feature exchange, and a new region-shape feature reduced this to 33 features and 306 parameters. The final train CV scores were 96.20% for selection seeds 0-2, 96.15% for verification seeds 10-19, and 96.13% for seeds 30-39. These are overlapping repeated splits of the same training examples, not new held-out datasets.

`experiments/*ceiling96*_result.txt` contains historical search traces. The release refit freezes the selected subset and does not claim to re-run that entire adaptive search. The learned checkpoint, raw image evaluation, and fraction experiments are the supported reproducibility targets.

## Fresh frozen-backbone comparison

All 11 article backbone rows and 98 fraction points have been independently rerun with pinned TorchGeo-bench source and a separate GPU environment. Original supplied results are in `experiments/imported/`; fresh results, model-state hashes, and side-by-side comparisons are in `experiments/torchgeo_bench_eurosat/reproduced/`. Full-data accuracy differs by at most 0.11 percentage points (six images), and the largest curve-point difference is 0.44 points.

See [the benchmark reproduction guide](experiments/torchgeo_bench_eurosat/README.md) for the complete table, commands, and protocol caveats. The backbone curves use one nested unstratified sample sequence per model, not the five-seed stratified protocol of the local models; the fixed test-set bootstrap intervals do not cover training-subset variation.
