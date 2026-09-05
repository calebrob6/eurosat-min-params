# Solving EuroSAT with as few parameters as possible

**The game: classify EuroSAT as accurately as possible with as few learned parameters as possible.** Instead of starting with a pretrained backbone, we ask how far fixed image measurements and a tiny classifier can go.

Our best handcrafted-feature model reaches **96.04% test accuracy with 306 learned weights and biases**. It computes 33 spectral and spatial measurements per image, then applies a single affine classifier. These are the original article results, not replacements with the slightly different numbers from later reruns.

## Rules of the game

1. **Use the same board.** EuroSAT has 27,000 Sentinel-2 patches, each 64x64 pixels with 13 bands and one of 10 labels. Use TorchGeo's fixed 16,200 training, 5,400 validation, and 5,400 test images.
2. **Count every learned weight and bias used for prediction.** A pretrained backbone's weights still count, even when frozen. Count its classifier too.
3. **Fixed arithmetic is allowed.** Means, percentiles, gradients, spectral ratios, line/corner statistics, and other deterministic image measurements have no learned weights. Feature engineering and subset selection do not count as deployed scalar parameters under this game's rules, but they are still part of the research effort.
4. **Select without test labels.** Fit weights on train; choose features and regularization with train-only cross-validation and/or validation. Do not merge validation into the final training set or use test accuracy to choose a configuration.
5. **Report the accuracy/parameter tradeoff.** We reduced the count needed to exceed 94%, then 95%, then 96%. These are the best models found by our search, not proofs of a global minimum.

The parameter-saving trick is simple: subtract one class's affine score from every class score, leaving one implicit zero-score reference class. A 10-class model with F features then stores `9 * (F + 1)` values instead of `10 * (F + 1)`, without changing the mathematical classifier. We also fold the training-set feature standardizer into the weights and bias, so it adds no deployed values.

| Model | Fixed features | Learned values | Validation | Test |
|---|---:|---:|---:|---:|
| ImageStats (per-band mean/std/min/max) | 52 | 477 | 90.93% | 90.96% |
| Reference-class linear, submission 12 | 18 | 171 | 94.06% | 94.33% |
| Reference-class linear, submission 13 | 30 | 279 | 95.31% | 95.37% |
| **Reference-class linear, submission 14** | **33** | **306** | **96.17%** | **96.04%** |

The 33 measurements combine spectral distributions, multiscale gradient texture, orientation entropy, spectral-index texture, Hough lines, Harris corners, local binary patterns, connected components, and low-NDVI region shape. The exact ordered recipe is in [`src/frontier.py`](src/frontier.py) and [`results/eurosat_306_features.csv`](results/eurosat_306_features.csv).

## Original results versus fresh reruns

**The uploaded CSVs contain the original blog numbers and are preserved on `main`.** For example, they report DOFA Large at **98.33%**, and OlmoEarth v1.2 Nano/Small/Base at **96.89% / 98.65% / 98.80%**. They have not been overwritten with the fresh reproduction scores.

| Evidence | Location |
|---|---|
| Original full-data backbone scoreboard | [`experiments/imported/eurosat-13band-merge-val-false-20260901/`](experiments/imported/eurosat-13band-merge-val-false-20260901/) |
| Original ResNet/ConvNeXt/DOFA training-fraction curves | [`experiments/imported/eurosat-train-fractions-20260901/`](experiments/imported/eurosat-train-fractions-20260901/) and [`eurosat-spatial-train-fractions-20260901/`](experiments/imported/eurosat-spatial-train-fractions-20260901/) |
| Original OlmoEarth curves and v1.2 full-data scores | [`experiments/imported/olmoearth-train-fractions-20260901/`](experiments/imported/olmoearth-train-fractions-20260901/) |
| Local handcrafted/ImageStats headline results | [`results/eurosat_models.csv`](results/eurosat_models.csv) |
| Original local five-seed curves | [`experiments/eval_training_fractions_result.csv`](experiments/eval_training_fractions_result.csv) and [`eval_imagestats_fractions_result.csv`](experiments/eval_imagestats_fractions_result.csv) |
| Fresh backbone runs and original-vs-fresh comparisons | [`experiments/torchgeo_bench_eurosat/reproduced/`](experiments/torchgeo_bench_eurosat/reproduced/) |

## Reproduce the blog results

Run all commands from the repository root. None requires the separately maintained blog checkout.

### Assemble the original tables and curve data

This uses only Python's standard library and the checked-in CSVs. It reads the original measurements; it does **not** train or evaluate a model.

```bash
python export_blog_results.py
```

It writes `output/blog-results/scoreboard.csv` (all 15 scoreboard rows) and `output/blog-results/training_fractions.csv` (all 126 points for nine methods, seven fractions, and two split protocols). Every row names its source CSV. The script requires the scoreboard accuracies to match the original article's rounded values. Local five-seed standard deviations and backbone bootstrap confidence intervals have separate fields so they are not confused.

### Rerun the local handcrafted models and ImageStats

The reference environment is **Python 3.13.13 on Linux, CPU only**. No GPU, TorchGeo installation, pretrained weights, or pre-existing NumPy caches are needed.

```bash
python3.13 -m venv .venv-reproduce
source .venv-reproduce/bin/activate
python -m pip install -r requirements-reproduce.txt
python reproduce.py --download --fractions --check
```

This downloads the checksum-pinned EuroSAT archive and random/spatial split files as needed, streams the original TIFFs, recomputes the features, evaluates the checked-in 171/279/306-value heads, fits ImageStats with validation-selected C, and refits both five-seed learning curves. `--check` fails if the headline counts, per-class scores, feature schema, ImageStats C sweep, or requested learning curves differ from the reference results.

For only the model table, omit `--fractions`. To also refit the frozen 33-feature model on train with C=3 instead of loading its checkpoint:

```bash
python reproduce.py --download --refit-306 --fractions --check --output output/refit
```

This writes `output/refit/model_306.npz`; it never overwrites the checked-in checkpoint or reruns the adaptive feature search. All downloaded data and generated results stay under ignored `data/` and `output/`.

### Rerun the frozen-backbone comparisons

This is a separate, pinned **CUDA GPU** environment. The 11 backbone scoreboard rows and 98 backbone learning-curve points use TorchGeo-bench's own model wrappers, feature extraction, and logistic-regression probe. Model weights remain frozen and every probe uses `merge_val=false`.

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
python3.13 -m venv output/benchmark-env
output/benchmark-env/bin/python -m pip install \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt

output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py --download \
  --models earthloc moco resnet18 vit_base \
  --output output/blog-backbones

output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py --download \
  --models resnet50 convnext_tiny dofa_base dofa_large olmoearth_nano olmoearth_small olmoearth_base \
  --datasets eurosat eurosat-spatial --fractions \
  --output output/blog-backbones

python experiments/torchgeo_bench_eurosat/compare.py --results-dir output/blog-backbones
```

The output includes fresh model/curve CSVs, resolved model configurations, weight and embedding hashes, and original-vs-fresh comparison CSVs. These commands leave the original uploads and checked-in reruns unchanged. See [the benchmark guide](experiments/torchgeo_bench_eurosat/README.md) for embedding reuse, batch sizes, and source revisions.

The independent backbone reruns reproduce the scores closely, not universally bit-for-bit: full-data differences are at most six test images (0.11 percentage points), and the maximum learning-curve difference is 0.44 points. The archived original values remain the source for the blog tables.

## Interpretation and scope

This is a parameter-count game, not a claim that feature engineering, extraction compute, or choosing a feature subset is free in practice. Pretrained models can transfer to many tasks without this dataset-specific search.

The local learning curves use five stratified training subsamples; the original backbone curves use one nested unstratified sample sequence per model. Spatial experiments are post-hoc repartitions of the same image universe used during feature selection, not untouched geographic model-selection benchmarks. The 74.8% means-only and 87.8% means-plus-std ablations remain historical notes, not exact reproduction targets.

The release preserves the original numerical feature arithmetic. Some historical physical-band aliases and percentile descriptions in the article need correction; changing their numeric indices would change the trained model. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete caveats and [RESULTS.md](RESULTS.md) for the local experiment summary.

## Research archive

`experiments/`, `ideas/`, and older submissions retain the EuroSAT search history. Some historical scripts require derived caches; the commands above are the supported reproduction paths. `BLOG.md`, `RESULTS.html`, and `FEATURES.html` are historical drafts and illustrations.

RESISC45 research is preserved on the [`resisc45` branch](https://github.com/calebrob6/eurosat-min-params/tree/resisc45), separate from the EuroSAT-focused `main` tree. The independently cloned `geospatialml/` blog checkout is ignored and is **not a submodule**.

## License

Code is provided under the [MIT license](LICENSE). EuroSAT images and externally downloaded model weights retain their respective owners' licenses; they are not included or relicensed here.
