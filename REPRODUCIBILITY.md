# Reference runs

The public repo keeps the final experiment, not every trial that led to it. The blog lives in [GeoSpatial ML](https://geospatialml.com/posts/eurosat-min-params/).

## EuroSAT

```bash
uv venv --python 3.13 .venv-reproduce
uv pip install --python .venv-reproduce/bin/python -r requirements-reproduce.txt
.venv-reproduce/bin/python reproduce.py --download --check
```

The script streams the original TIFFs, evaluates `models/eurosat_33.npz`, and fits ImageStats with validation-selected regularization. The splits contain 16,200 training, 5,400 validation, and 5,400 test patches. Fitting never merges validation into training or selects on test.

`--refit` trains the same fixed 33-feature model at C=3 and writes a new checkpoint under the chosen `--output`. `--fractions` refits both heads with fewer labels on the random and spatial splits. Those curves use a feature set already chosen on the full random split, and the spatial split reuses the same images; they are not an independent test of feature discovery.

The reference environment uses Python 3.13.13 and the versions in `requirements-reproduce.txt`. Other numerical libraries can change fitted coefficients. The NumPy reference and PyTorch implementation are separate: PyTorch uses native operations, and binning or threshold results can differ near boundaries. It does not promise identical features or predictions.

## Band order

The order is `B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A`. It matches `EuroSAT.all_band_names`, a default TorchGeo 0.10 dataset object's `band_indices`, and its actual samples read with rasterio.

Both `src.data.BAND_NAMES` and `TIFF_BAND_NAMES` describe these physical channels, including when labeling regenerated article figures. Correcting display labels does not change the historical numeric feature aliases below.

The reference feature names retain two old aliases: `B_SWIR1=11` selects B12 and `B_SWIR2=12` selects B8A. Thus `nbr`, for example, uses `(B08-B8A)/(B08+B8A+epsilon)`, not the usual burn-ratio bands. Both implementations keep those numeric choices so the feature recipe stays consistent.

The 33 measurements are 32 columns from the 377-feature pool plus `tail_aniso_low_ndvi`. They are not a simple 33-column slice of that pool. `results/eurosat_306_features.csv` records the order.

## RESISC45

```bash
.venv-reproduce/bin/python -m experiments.resisc45.run --download --check
```

The command resizes original RGB JPEGs with rasterio's bilinear uint8 64x64 read, then fits ImageStats and the fixed 33-feature model. It reproduces 59.06% for the latter. ImageStats reaches 36.59%; the blog's earlier 36.63% quote has no saved fit. New outputs are separate from the reference table.

## Embeddings

The [backbone runner](experiments/torchgeo_bench_eurosat/README.md) extracts frozen embeddings, and the [comparison experiment](experiments/representation_overlap/README.md) fits combined classifiers and maps between representations. Both save filenames, settings, and hashes with their outputs. All numerical comparisons are selected on validation before test evaluation.

`python export_blog_results.py` exports the current saved tables without fitting anything. The earlier tiny-model checkpoints, exploratory searches, and project notes are intentionally not included.

## Historical article models

The 171- and 279-value checkpoints and their reference runner remain available at commit `924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c`. To evaluate them without changing the current checkout, first prepare EuroSAT and the CPU environment with the commands above, then run from the repository root:

```bash
git worktree add --detach output/historical \
  924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c
ln -s ../../data output/historical/data
.venv-reproduce/bin/python output/historical/reproduce.py \
  --check --output output/historical-models
```

Use a fresh worktree path when repeating setup. This shares only the original downloaded images, not historical feature caches. The archived command evaluates all three saved local heads, fits ImageStats, and checks its own pinned reference tables. It does not repeat feature discovery. The same archival revision includes supporting baseline scripts and the neural learning-curve source tables; their historical and newly measured results must not be conflated.
