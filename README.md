# Patch features

Fixed spectral and spatial features for satellite image patches, as a drop-in PyTorch module. Give it the names of your bands and it computes band statistics, gradient texture, spectral indices, lines, corners, local binary patterns, connected regions, and more, with nothing to train.

These are the features behind [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/), where 33 of them and a 306-parameter linear classifier reach 96% test accuracy on EuroSAT.

## Quick start

`patch_features.py` is a single file that only needs PyTorch. Copy it into your project, or clone this repo.

```python
import torch
from patch_features import PatchFeatures

extractor = PatchFeatures(bands=['B02', 'B03', 'B04', 'B08', 'B11', 'B12'])
images = torch.rand(8, 6, 64, 64) * 3000  # raw pixel values; no normalization needed
features = extractor(images)              # (8, 313)
extractor.feature_names[:3]               # ('mean_B02', 'mean_B03', 'mean_B04')

model = torch.nn.Sequential(extractor, torch.nn.Linear(extractor.num_features, 10))
```

The extractor has no learned parameters and runs on whatever device the input is on, so the classifier after it trains normally. Standardize the features (or use a model that doesn't care about scale) before fitting a linear head.

## Any band layout

List your bands in channel order. Blue, green, red, NIR, and SWIR are recognized from common names (`'red'`, `'R'`, Sentinel-2 `'B04'`, Landsat `'SR_B4'`, ...) and enable spectral indices such as NDVI; anything else still gets the per-band features.

| Input | Example | Default features |
|---|---|---:|
| RGB photos or aerial imagery | `PatchFeatures(['red', 'green', 'blue'])` | 162 |
| NAIP | `PatchFeatures(['R', 'G', 'B', 'NIR'])` | 218 |
| Landsat 8/9 | `PatchFeatures(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])` | 337 |
| Sentinel-2, all 13 bands | `PatchFeatures(['B01', ..., 'B12', 'B8A'])` | 481 |
| SAR or anything else | `PatchFeatures(['VV', 'VH'])` | 63 |

Use `roles` for other names, or to override a guess: `PatchFeatures(['b1', 'b2', 'b3', 'b4'], roles={'red': 'b3', 'nir': 'b4'})`. `extractor.roles` shows what was recognized.

Images can be any size of at least 16x16 pixels. Inputs must be finite; outputs are float32.

## Choosing features

By default you get every family for every available channel. Pass `features=` to compute only what you need, in the order you want:

```python
PatchFeatures(bands, features=['p90_ndvi', 'grad_mean_B08', 'corner_frac_s1_pan', 'corr_B04_B08'])
```

Names are `{measure}_{channel}`, or `{measure}_s{k}_{channel}` after `k` rounds of 2x2 average pooling. Channels are your bands, `pan` (the mean of all bands), and derived maps: `ndvi`, `ndwi`, `ndbi`, `ndmi`, `nbr`, `bsi` with NIR/SWIR, and `exg`, `ngrdi`, `ngbdi`, `nrbdi`, `saturation`, `lightness` with RGB. Any measure works on any channel:

| Measures | What they describe |
|---|---|
| `mean`, `std`, `min`, `max`, `p10` ... `p90`, `spread` | Pixel value distribution (`spread` is p90 - p10) |
| `grad_mean`, `grad_std` | Gradient magnitude: edges and fine texture |
| `coherence`, `orient_entropy`, `orient_hist0` ... `orient_hist3` | Whether gradients share a direction |
| `fft_peak`, `fft_slope` | Periodic structure and the balance of fine to coarse detail |
| `line_peak_frac`, `line_peak_len`, `line_top3` | Straight lines (Hough transform) |
| `corner_frac`, `corner_mag` | Corners and junctions (Harris) |
| `lbp_entropy`, `lbp_uniform` | Local binary patterns |
| `blob_largest`, `blob_count`, `blob_mean` | Connected regions above the median |
| `tail_aniso_low`, `tail_spread_low`, `tail_aniso_high`, `tail_spread_high` | Shape of the darkest and brightest quarter of pixels |

`corr_{a}_{b}` is the spatial correlation between two channels. `extractor.feature_families` groups the columns by family, which is handy for filtering the default list.

## The 306-parameter EuroSAT model

`EuroSATFeatures()` computes the article's 33 features from 13-band EuroSAT patches in TorchGeo's band order (`B01 ... B12 B8A`, raw values). The saved classifier head is a plain `nn.Linear`:

```python
from patch_features import EuroSATFeatures

model = torch.nn.Sequential(EuroSATFeatures(), torch.nn.Linear(33, 10))
model[1].load_state_dict(torch.load('models/eurosat306.pt'))
predictions = model(images).argmax(1)  # images: (N, 13, 64, 64)
```

Classes are in alphabetical order: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake. One row of the head is zero, so it stores 306 values. `EuroSATFeatures('52')` gives the ImageStats baseline, and `'377'` and `'389'` give the article's candidate pools. Like the trained model, these presets use B12 as `swir1` and B8A as `swir2`.

## Reproducing the article

```bash
pip install -r requirements.txt
python -m experiments.eurosat.run --download
```

This downloads EuroSAT, evaluates the saved model, and fits ImageStats. Add `--refit` to fit the 33-feature head again and `--fractions` for the learning curves. Tables go to `output/eurosat/`.

| Features | Learned values | Article (validation / test) | This code (validation / test) |
|---|---:|---:|---:|
| ImageStats (52) | 477 | 90.93% / 90.96% | 90.91% / 91.02% |
| Final fixed features (33) | 306 | 96.17% / 96.04% | 96.17% / 96.11% |

The article's numbers came from an earlier NumPy implementation of the same features, recorded in [`experiments/eurosat/results/`](experiments/eurosat/results/). This PyTorch version rounds slightly differently, so fitted baselines and a few predictions move by a fraction of a point.

| Experiment | |
|---|---|
| [`experiments/eurosat/`](experiments/eurosat/run.py) | The final model, ImageStats, and learning curves |
| [`experiments/feature_importance/`](experiments/feature_importance/README.md) | Ranking all 389 features and pruning the weakest |
| [`experiments/resisc45/`](experiments/resisc45/README.md) | RGB baselines on RESISC45 |
| [`experiments/torchgeo_bench_eurosat/`](experiments/torchgeo_bench_eurosat/README.md) | Frozen pretrained backbone embeddings |
| [`experiments/representation_overlap/`](experiments/representation_overlap/README.md) | Handcrafted features against those embeddings |

Recorded results use the features' earlier names; [`feature_names.csv`](experiments/eurosat/results/feature_names.csv) maps them to the current ones.

## License

[MIT](LICENSE) for the code. Datasets and downloaded model weights retain their own licenses.
