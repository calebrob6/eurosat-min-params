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

Feature extraction is cached under `output/feature-importance-cache-v2/`; subsequent runs authenticate and reuse it. Results go to `output/feature-importance/`. Both paths must remain under `output/`, and a nonempty results directory is never overwritten.

The completed reference run is checked in under [`results/`](results/), including the full importance table and both plots.

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
