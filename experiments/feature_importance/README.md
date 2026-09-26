# EuroSAT full-pool feature importance

This experiment computes 389 handcrafted columns from the original 13-band EuroSAT TIFFs: the historical 377-feature pool plus all 12 spectral-tail region-shape measurements. The added family includes the final model's `tail_aniso_low_ndvi` feature at full-pool index 381, along with the other low/high pan, NDVI, and NDBI anisotropy and spread measurements.

## Protocol

Every classifier is L2-regularized multinomial logistic regression fitted on the official training split after per-feature standardization using training statistics. At every retained-feature count, a validation sweep selects a new `C`, with the strongest regularization winning exact ties. The selected fit supplies both that point's score and the coefficient importances used for the next removal.

Feature importance is the L2 norm of a feature's standardized coefficients across the ten classes. Starting with all 389 columns, the classifier is refitted and the five active columns with the smallest importance are removed. This repeats down to four features. Validation and test accuracy are recorded at every fixed step, but test scores never select `C`, features, or a stopping point.

The exact published 33-feature set is evaluated separately because it is not one of the five-at-a-time recursive subsets. Its `C` is independently validation-tuned over the same grid, and its validation/test scores are marked on the curve.

Coefficient magnitude is a model-specific conditional importance, not a causal or permutation importance. Correlated columns can divide or exchange weight, so use the elimination curve and removal order alongside the full-model ranking.

## Run

Install the environment described in the root README, then run from the repository root:

```bash
python -m experiments.feature_importance.run --download
```

Feature extraction uses `EuroSATFeatures('389')` from [`patch_features.py`](../../patch_features.py) (on the GPU if available; set `--device`) and is cached under `output/feature-importance-cache-v3/`; subsequent runs authenticate and reuse it. Results go to `output/feature-importance/`. Both paths must remain under `output/`, and a nonempty results directory is never overwritten.

The completed reference run is checked in under [`results/`](results/), including the full importance table and both plots. It used an earlier NumPy implementation of the same features and their earlier names; [`feature_names.csv`](../eurosat/results/feature_names.csv) maps those names to the current ones. A fresh run with the PyTorch module lands within a fraction of a point: the full 389-feature model scores 96.70% validation and 96.85% test, against the recorded 96.69% and 96.76%.

The one-feature-at-a-time follow-up is checked in separately under [`results_step1/`](results_step1/). It evaluates every retained-feature count, including a direct comparison between the recursive and published 33-feature sets.

## Outputs

| File | Contents |
|---|---|
| `c_sweep.csv` | Validation accuracy and selected `C` for every regularization candidate at every retained-feature count |
| `feature_importances.csv` | All 389 full-model feature ranks, aggregate importances, and per-class standardized coefficients |
| `elimination_order.csv` | Importance and model size when each feature was removed, plus the final retained features |
| `scores.csv` | Train, validation, and test accuracy for every retained-feature count |
| `frontier33.csv` | Separately tuned train, validation, and test score for the exact published 33-feature set |
| `frontier33_c_sweep.csv` | Validation sweep used to tune the published 33-feature set |
| `frontier33_features.csv` | Mapping from the published feature order into the 389-column pool |
| `score_by_features.png` | Test accuracy by retained-feature count with the exact 33-feature set starred |
| `top_feature_importances.png` | The 30 strongest full-model coefficient norms |
| `summary.json` | Regularization protocol and full, best-validation, and final curve points |
| `provenance.json` | Feature schema, source hashes, split hashes, package versions, and protocol settings |
