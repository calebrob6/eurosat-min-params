# EuroSAT with 306 parameters

Code for [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/). We measure spectral and spatial properties of each image, then fit a linear classifier. The final model uses **33 fixed features and 306 learned values**, reaching **96.04% test accuracy** on EuroSAT.

| Features | Learned values | Validation | Test |
|---|---:|---:|---:|
| ImageStats (52) | 477 | 90.93% | 90.96% |
| Final fixed features (33) | 306 | 96.17% | 96.04% |

## Install

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
git clone https://github.com/calebrob6/eurosat-min-params.git
cd eurosat-min-params
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install --torch-backend auto -e ".[experiments]"
```

Use `--torch-backend cpu` for a CPU-only install. The feature module itself only needs PyTorch; `experiments` adds the NumPy stack that the scripts below use.

## Use the features

```python
import torch
from eurosat_features import EuroSATFeatures

extractor = EuroSATFeatures("33")  # or "377", "389", or "52"
images = torch.rand(8, 13, 64, 64) * 10000
features = extractor(images)  # (8, 33)

model = torch.nn.Sequential(extractor, torch.nn.Linear(33, 10))
logits = model(images)
```

Move the module and input to CUDA to use a GPU. Outputs are float32 tensors on the input device. The extractor has no learned weights or image gradients; the classifier can be trained normally.

`"377"` is the historical feature pool and `"389"` adds all 12 region-shape measurements. `"52"` selects ImageStats.

Inputs must be 13-band 64x64 patches at the original pixel scale. The order matches `torchgeo.datasets.EuroSAT.all_band_names` and an untransformed dataset with default `bands`:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A
```

A few feature names keep older band aliases: `nbr`, for example, uses B08 and B8A rather than the usual burn-ratio bands. The PyTorch module and the NumPy reference in `src/` share those choices, but PyTorch uses its own numerical operations, so binning and threshold results can differ near boundaries.

To extract features from a directory of patches:

```bash
python extract_features.py path/to/patches/ --features 389 --output output/features.npz
```

The file contains `features`, `filenames`, and `feature_names`. Add `--device cuda:0` for GPU extraction.

## Compute the results

```bash
python reproduce.py --download
```

This downloads the original TIFFs, recomputes the features, evaluates `models/eurosat_33.npz`, fits ImageStats with validation-selected regularization, and writes tables to `output/reproduce/`. Add `--refit` to fit the 33-feature classifier again or `--fractions` for the learning curves. New files go under `output/`, never over the saved model.

```bash
python export_blog_results.py
```

That writes the article's tables from the saved results without fitting anything.

Scores from the saved model are exact; baselines that are fitted fresh can move by a fraction of a point across NumPy and scikit-learn versions.

## Repository layout

| Path | Purpose |
|---|---|
| `eurosat_features/` | PyTorch feature module |
| `src/` | NumPy reference features and linear classifier |
| `models/` | Final EuroSAT classifier |
| `results/` | Model scores, feature order, and learning curves |
| `experiments/feature_importance/` | [Rank all 389 features and prune the weakest](experiments/feature_importance/README.md) |
| `experiments/resisc45/` | [RGB baselines on RESISC45](experiments/resisc45/README.md) |
| `experiments/torchgeo_bench_eurosat/` | [Frozen pretrained backbone embeddings](experiments/torchgeo_bench_eurosat/README.md) |
| `experiments/representation_overlap/` | [Handcrafted features against those embeddings](experiments/representation_overlap/README.md) |

## License

[MIT](LICENSE) for the code. Datasets and downloaded model weights retain their own licenses.
