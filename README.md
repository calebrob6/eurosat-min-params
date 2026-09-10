# EuroSAT with 306 parameters

**[Features](#use-the-features) | [Reproduce the results](#reproduce-the-results) | [Files](#files) | [License](#license)**

Code for [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/). We compute fixed spectral and spatial measurements from each satellite image, then fit a linear classifier. The 33-feature model reaches **96.04% test accuracy with 306 learned weights and biases** on EuroSAT.

This repository contains the feature extractors, saved models, experiment scripts, and results. The features are also available as a PyTorch module that runs on CPU or CUDA.

## Install

```bash
git clone https://github.com/calebrob6/eurosat-min-params.git
cd eurosat-min-params
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[io]"
```

The feature module only needs PyTorch. The `io` extra adds NumPy and rasterio for reading TIFFs and saving feature files.

## Use the features

```python
import torch
from eurosat_features import EuroSATFeatures

extractor = EuroSATFeatures(feature_set="33")

# A batch of eight 13-band, 64x64 patches in the original pixel-value scale.
images = torch.rand(8, 13, 64, 64) * 10000
features = extractor(images)  # (8, 33)

print(extractor.feature_names)
```

Use `feature_set="377"` for the full pool or `"52"` for per-band mean, standard deviation, minimum, and maximum. To run on a GPU, move both the module and images to CUDA:

```python
features = extractor.to("cuda")(images.to("cuda"))
```

The input band order is the physical order in TorchGeo's EuroSAT TIFFs:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A
```

Keep the original pixel values if you want to use the saved classifiers; do not normalize images to match an ImageNet model. The extractor returns one float32 vector per patch on the input device. It has no learned weights. You can train a head on these features, but the fixed extractor does not provide gradients with respect to the image.

Floating-point results can differ slightly between NumPy, CPU PyTorch, and CUDA. The reproduction commands below retain the original NumPy implementation for the paper's reference results.

```python
model = torch.nn.Sequential(
    EuroSATFeatures(feature_set="33"),
    torch.nn.Linear(33, 10),
)
logits = model(images)
```

This example has a new 10-row classifier. The published 306-value model stores only nine rows, with one class's score fixed at zero.

The 33-feature model uses 32 columns from the full pool plus one additional region-shape measurement. It is not simply a 33-column slice of the 377 features. The code keeps the article's numerical band choices; some older index names differ from their usual physical definitions. See [the band notes](REPRODUCIBILITY.md#band-order-and-feature-names).

### Extract features from TIFF files

```bash
python extract_features.py path/to/patches/ \
  --features 33 --device cuda:0 --output output/features.npz
```

Directories are searched recursively. Each TIFF must have 13 bands and 64x64 pixels. The output contains `features`, `filenames`, and `feature_names`. Existing files are not overwritten.

## Reproduce the results

Run these commands from the repository root. You do not need the separate blog checkout.

To export the article's tables from the saved results, without fitting any models:

```bash
python export_blog_results.py
```

The tables go to `output/blog-results/`: the 14-model comparison, learning curves, feature list, class accuracies, combined classifiers, and both directions of the embedding comparison.

To recompute the local EuroSAT results from the original TIFFs, use the separate CPU reference environment:

```bash
python3.13 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r requirements-reproduce.txt
python reproduce.py --download --fractions --check
```

This evaluates the saved 171-, 279-, and 306-value models, refits ImageStats, and reruns the random and spatial learning curves. Add `--refit-306` to fit a new copy of the published 33-feature classifier. New results go under `output/`; the saved models stay unchanged.

The pretrained backbones and embedding comparisons use a separate CUDA environment. After following the [environment setup](REPRODUCIBILITY.md#cuda-environment), run:

```bash
python reproduce_all.py --download
```

This runs the local models, backbone comparisons, representation comparisons, supporting baselines, and figure statistics. `--dry-run` prints the commands. `--steps overlap` runs just the representation comparisons, extracting the required embeddings first. Add `--exploratory` for the longer CNN and MOSAIKS runs. Results go to `output/reproduce-all/`, with original-versus-fresh comparisons kept alongside them.

The [reproduction guide](REPRODUCIBILITY.md) lists what each command covers, including the RESISC45 comparison. The original backbone scores can differ slightly on a fresh run. The old means-only, mean-plus-std, and RESISC45 ImageStats numbers lack saved fitting settings. Their new runs are labelled separately, not substituted for the article's numbers. The optional CNN and MOSAIKS reruns are described in the [supporting-baseline guide](experiments/article_baselines/README.md).

## Files

| Path | Contents |
|---|---|
| `eurosat_features/` | PyTorch feature extractors |
| `src/` | Original NumPy features, data loading, and small classifiers |
| `submissions/` | Saved models and their training/evaluation scripts |
| `results/` | Published local results and feature names |
| `experiments/torchgeo_bench_eurosat/` | Frozen-backbone reproduction |
| `experiments/representation_overlap/` | Combined classifiers, embedding comparisons, and saved results |
| `experiments/article_baselines/` | Supporting baselines and historical-result sources |
| `ideas/` | Notes from the feature search |

The full RESISC45 research history remains on the [`resisc45` branch](https://github.com/calebrob6/eurosat-min-params/tree/resisc45). Downloaded data, extracted embeddings, and new experiment outputs are ignored by Git.

## License

[MIT](LICENSE) for the code. EuroSAT, RESISC45, and downloaded model weights retain their own licenses.
