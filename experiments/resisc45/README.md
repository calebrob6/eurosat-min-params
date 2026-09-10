# RESISC45

Two baselines on 64x64 RGB patches: ImageStats and a fixed 33-feature subset.

```bash
.venv-reproduce/bin/python -m experiments.resisc45.run --download --check
```

This reads the official 256x256 JPEGs, resizes them with rasterio's bilinear resampling to uint8 64x64, fits on train, and evaluates on validation and test. It does not need TorchGeo or historical caches. The download checks official file hashes and does not overwrite existing images.

| Features | Learned values | Test accuracy |
|---|---:|---:|
| ImageStats | 572 | 36.59% |
| Fixed 33 | 1,496 | 59.06% |

ImageStats selects C on validation. The 33-feature model uses the saved subset and C=30, without repeating feature selection. Outputs go to `output/resisc45/`; use a new `--output output/name` to repeat.

The article's older ImageStats number, 36.63%, has no saved fit. The 36.59% reference here is a fresh validation-tuned measurement, not a claim to recover that missing run.
