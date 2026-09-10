# EuroSAT with 306 parameters

Code for [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/). We measure spectral and spatial properties of each image, then fit a linear classifier. The final model uses **33 fixed features and 306 learned values**, reaching **96.04% test accuracy** on EuroSAT.

The repo keeps the final model, an ImageStats baseline, and the feature/embedding comparison—not the steps of the original search.

## Install

With [uv](https://docs.astral.sh/uv/getting-started/installation/) installed:

```bash
git clone https://github.com/calebrob6/eurosat-min-params.git
cd eurosat-min-params
uv venv --python 3.13 .venv
source .venv/bin/activate
uv pip install --torch-backend auto -e ".[io]"
```

Use `--torch-backend cpu` for a CPU-only install. The module itself only needs PyTorch; `io` adds TIFF reading.

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

`"377"` preserves the historical feature pool. `"389"` is the extended optimization pool used by the full-pool importance experiment: the same 377 columns followed by all 12 low/high pan, NDVI, and NDBI region-shape measurements. `"52"` selects ImageStats.

Inputs must be 13-band 64x64 patches at the original pixel scale. The order matches `torchgeo.datasets.EuroSAT.all_band_names` and an untransformed dataset with default `bands`:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A
```

TorchGeo returns those TIFF channels unchanged as float32. Explicit band selection or image transforms can change the order or values. PyTorch uses its own numerical operations; it does not emulate NumPy rounding. See [the reference notes](REPRODUCIBILITY.md) before using a saved NumPy-trained head.

To extract files:

```bash
python extract_features.py path/to/patches/ --features 389 --output output/features.npz
```

The file contains `features`, `filenames`, and `feature_names`. Add `--device cuda:0` for GPU extraction.

## Run the experiment

The reference classifier uses the original NumPy features:

```bash
uv venv --python 3.13 .venv-reproduce
uv pip install --python .venv-reproduce/bin/python -r requirements-reproduce.txt
.venv-reproduce/bin/python reproduce.py --download --check
```

This evaluates `models/eurosat_33.npz` and fits ImageStats. Add `--refit` to train the 33-feature classifier again or `--fractions` for the learning curves. New files go under `output/`, not over the saved model.

The basic RGB comparison is also on `main`:

```bash
.venv-reproduce/bin/python -m experiments.resisc45.run --download --check
```

That runs ImageStats and the fixed 33-feature RESISC45 model. For pretrained embeddings and combined classifiers, follow [the embedding comparison](experiments/representation_overlap/README.md). `python export_blog_results.py` exports the saved current-model tables without training.

To rank all 389 EuroSAT features—including every spectral-tail region-shape measurement—and trace regularized logistic-regression accuracy while recursively removing the five least-important features, run `.venv-reproduce/bin/python -m experiments.feature_importance.run --download`. See the [full-pool feature-importance experiment](experiments/feature_importance/README.md) for its protocol and outputs.

## Files

| Path | Purpose |
|---|---|
| `eurosat_features/` | PyTorch module |
| `models/` | Final EuroSAT classifier |
| `src/` | NumPy reference features and linear classifier |
| `results/` | Local-model results and learning curves |
| `experiments/` | RESISC45 and embedding comparisons |

## Development

```bash
uv pip install --torch-backend auto -e ".[io,style,tests]"
ruff check
ruff format --check
python -m unittest discover -s tests -p 'test_torch_features.py'
```

Ruff uses TorchGeo-style formatting and Google-style pydocstyle checks for the feature package, its CLI, and tests. There is no CI workflow.

## License

[MIT](LICENSE) for the code. Datasets and downloaded model weights retain their own licenses.
