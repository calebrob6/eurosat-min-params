# EuroSAT representation overlap results

New controlled analyses of frozen embeddings and raw handcrafted features. These do not replace the article scoreboard. H33 and H377 are not strictly nested: H33 includes one region-shape measurement outside H377.

| Backbone | H33 explains Z (%) | H377 explains Z (%) | H33 accuracy gain (pp) | H377 accuracy gain (pp) |
|---|---:|---:|---:|---:|
| resnet50 | 27.67 | 23.58 | 2.94 | 2.81 |
| convnext_tiny | 25.00 | 31.50 | 2.70 | 3.02 |
| dofa_large | 43.72 | 52.60 | 0.22 | 0.57 |
| olmoearth_nano | 78.78 | 88.66 | 0.89 | 0.94 |
| olmoearth_base | 59.49 | 73.18 | -0.04 | 0.15 |

The explanation percentages are native-coordinate held-out multivariate R-squared, not fractions of information or learned parameters. Reverse-decoding measurements and family summaries are in `features.csv` and `families.csv`; class-conditioned and shuffled decoding controls are in `decoding.csv`.

All validation candidate scores and scalar selection diagnostics are exported in `selection.csv`. Boundary choices describe the tested finite grids, not globally optimal regularization.

Accuracy gains compare a separately regularized, block-scaled concatenation with the controlled backbone-only probe. `complementarity.csv` includes paired intervals, corrected/error counts, exact McNemar p-values, Holm adjustment for the primary comparisons, shuffled and redundant-projection controls, and B10-disabled OlmoEarth input ablations. Log loss is secondary.

Intervals use 200 paired image bootstrap draws and are conditional on the fitted models. They do not measure feature-discovery, training-sample, or geographic uncertainty. EuroSAT was already extensively used in the earlier research; this is not an untouched external confirmation.

The PCA reconstruction reference sees the target embedding and is not a guaranteed held-out ceiling or an H-to-Z predictor. CKA describes geometry, not directional containment. Neither low linear decodability nor a probe gain proves information-theoretic absence from a backbone.
