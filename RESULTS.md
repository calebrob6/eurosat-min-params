# Results

| Features | Learned values | Validation | Test |
|---|---:|---:|---:|
| ImageStats (52) | 477 | 90.93% | 90.96% |
| Final fixed features (33) | 306 | 96.17% | 96.04% |

Run `.venv-reproduce/bin/python reproduce.py --download --check` to recompute the measurements from TIFFs and evaluate the saved final model. Add `--refit` to fit that head again and `--fractions` for smaller training sets.

The exact feature order, class scores, and learning curves are in `results/`. The [embedding comparison](experiments/representation_overlap/README.md) records combined-classifier accuracy and linear predictability in both directions.

The [RESISC45 baseline](experiments/resisc45/README.md) reaches 36.59% with ImageStats and 59.06% with its separate RGB 33-feature recipe. These are not EuroSAT features applied unchanged to RGB.
