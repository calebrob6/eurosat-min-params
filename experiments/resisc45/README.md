# RESISC45

Two baselines on 64x64 RGB patches: ImageStats and a fixed 33-feature subset.

```bash
python -m experiments.resisc45.run --download
```

This reads the official 256x256 JPEGs, resizes them with rasterio's bilinear resampling to uint8 64x64, fits on train, and evaluates on validation and test.

| Features | Learned values | Test accuracy |
|---|---:|---:|
| ImageStats | 572 | 36.59% |
| Fixed 33 | 1,496 | 59.06% |

ImageStats selects C on validation. The 33-feature model uses the saved subset and C=30, without repeating feature selection. These are RGB features chosen for RESISC45, not the EuroSAT features applied unchanged: the runner computes them with `PatchFeatures(['red', 'green', 'blue'], features=SELECTED_33)`.

The table above was recorded with an earlier NumPy implementation. The PyTorch version gives 36.48% and 58.79%; most of that difference comes from NumPy and scikit-learn versions, since the old code gives 36.51% and 58.92% in the same environment.

Outputs go to `output/resisc45/`; use a new `--output output/name` to repeat.
