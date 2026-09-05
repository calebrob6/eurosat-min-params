# EuroSAT with 306 learned parameters

Reproduction code for **Solving EuroSAT with as Few Parameters as Possible**. A logistic regression on 33 fixed image measurements reaches **96.04% test accuracy** using **306 learned weights and biases**. No pretrained backbone is used by this model.

| Model | Fixed features | Learned values | Validation | Test |
|---|---:|---:|---:|---:|
| ImageStats (per-band mean/std/min/max) | 52 | 477 | 90.93% | 90.96% |
| Reference-class linear, submission 12 | 18 | 171 | 94.06% | 94.33% |
| Reference-class linear, submission 13 | 30 | 279 | 95.31% | 95.37% |
| **Reference-class linear, submission 14** | **33** | **306** | **96.17%** | **96.04%** |

## Reproduce the article's local results

Run from the repository root. The reference environment is **Python 3.13.13**, Linux, CPU only. No GPU, TorchGeo installation, pretrained weights, or pre-existing NumPy caches are needed for these results.

```bash
python3.13 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r requirements-reproduce.txt
python reproduce.py --download --fractions --check
```

The command downloads the checksum-pinned multispectral EuroSAT archive and TorchGeo random/spatial split files as needed, reads all 27,000 original TIFFs in bounded-memory batches, recomputes the image features, evaluates the checked-in 171/279/306-value heads, fits the ImageStats baseline with validation-selected C, and refits both five-seed learning curves. It fails if the local headline numbers or learning-curve measurements differ from the checked-in results. Data and generated results stay under ignored `data/` and `output/reproduce/`; no checked-in models or measurements are overwritten.

For only the model table, omit `--fractions`. To refit the fixed 306-value head on train instead of loading the checkpoint:

```bash
python reproduce.py --download --refit-306 --fractions --check --output output/refit
```

This writes a new `output/refit/model_306.npz`, **not** a new feature selection. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact scope, article corrections, external-backbone evidence, environment sensitivity, and interpretation of the spatial and low-data experiments.

**Article scope:** the local model table and both local learning curves reproduce exactly in the reference environment. Original pretrained-backbone outputs are now archived under `experiments/imported/`, and all 11 scoreboard rows plus 98 backbone curve points have been independently rerun. See [the backbone reproduction](experiments/torchgeo_bench_eurosat/README.md) for the pinned GPU environment and original-vs-fresh differences (at most 0.11 percentage points on the scoreboard). The article still needs the physical-band/percentile-label corrections documented in the reproduction notes.

## What is being counted?

The feature extractor is fixed arithmetic on each image. The training-set standardizer is folded into an affine classifier; subtracting one class's logits leaves nine stored class rows. For F selected features, the deployed count is `9 * (F + 1)`. For the best model, this is `9 * 34 = 306`. Feature selection, extraction compute, and the fixed feature-index list are not counted as learned scalar weights.

The release preserves the historical feature arithmetic, including legacy spectral-index aliases. Physical TIFF band names and percentile labels are documented explicitly in [REPRODUCIBILITY.md](REPRODUCIBILITY.md); changing them changes the model rather than reproducing it.

## Repository map

- `reproduce.py`: supported raw-TIFF reproduction entry point.
- `src/frontier.py`: frozen 33-feature recipe; `src/features.py`: core fixed features; `src/linmodel.py`: folded affine heads.
- `submissions/12_*`, `13_*`, `14_*`: the article's three handcrafted checkpoints.
- `results/`: raw-reproduced headline counts, per-class scores, feature names, and environment.
- `experiments/eval_*fractions_result.csv`: original five-seed random/spatial curves.
- `experiments/torchgeo_bench_eurosat/`: separate frozen-backbone comparison evidence.
- `experiments/`, `ideas/`, and older submissions: historical EuroSAT searches. Some historical scripts require derived caches; they are not the public reproduction entry point.
- `BLOG.md`, `RESULTS.html`, `FEATURES.html`: historical draft and interactive illustrations, not the release results manifest.

RESISC45 research and codebook experiments are preserved on the [`resisc45` branch](https://github.com/calebrob6/eurosat-min-params/tree/resisc45), separate from this EuroSAT release tree. The history has not been rewritten. The independently cloned `geospatialml/` blog checkout is ignored and is **not a submodule**.

## License

Code is provided under the [MIT license](LICENSE). EuroSAT images and externally downloaded model weights retain their respective owners' licenses; they are not included or relicensed here.
