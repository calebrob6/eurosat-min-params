# Copilot instructions

## Environment and commands

The public default branch is EuroSAT-only. RESISC45 experiments are preserved on the `resisc45` branch. Never stage the independent `geospatialml/` checkout, `blog_post.txt`, runtime `report.*.json` files, or credentials; `geospatialml` must not become a submodule.

For the release's CPU-only reference environment use Python 3.13.13 and `python -m pip install -r requirements-reproduce.txt`. The primary reproduction command is `python reproduce.py --download --fractions --check`; it streams original TIFFs and never depends on historical image/feature caches. `--refit-306` writes a new checkpoint under `output/`, never over a checked-in model. See `REPRODUCIBILITY.md` for article coverage and historical physical-band naming caveats.

Run commands from the repository root. The project is a collection of directly executable Python scripts rather than an installed package, and several scripts resolve checkpoints and result files relative to the root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cu128.txt
```

`requirements-cu128.txt` is the experiment-host configuration for its CUDA 12.8 driver. Use `requirements.txt` for a platform-neutral environment.

Download the official TorchGeo EuroSAT data and all three fixed splits:

```bash
python - <<'PY'
from torchgeo.datasets import EuroSAT

for split in ("train", "val", "test"):
    EuroSAT(root="data/EuroSAT", split=split, download=True)
PY
```

Build or refresh the raw NumPy caches used by the classical experiments:

```bash
python -m src.cache
```

Validation is script-based. Use a submission evaluation as the equivalent of a single targeted test; it loads the checked-in model and recomputes its configured features:

```bash
python submissions/12_reference_class_linear/eval.py
python submissions/13_reference_class_95/eval.py
```

Run a syntax check over the executable code:

```bash
python -m compileall -q src experiments submissions train.py
```

Smoke-test the separate TorchGeo/Lightning neural pipeline with one train, validation, and test batch:

```bash
python train.py --fast-dev-run --accelerator gpu --devices 1 --num-workers 0 --batch-size 2 --output-dir output/smoke
```

Full training and selection scripts can be expensive and overwrite checked-in `model.npz` or `*_result.txt` artifacts. Inspect their top-level constants and CLI options before running them, and check the resulting Git diff.

For fast GPU linear-probe C sweeps, install a sibling `torchgeo/torchgeo-bench` checkout editable and point it at the existing data:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git ../torchgeo-bench
python -m pip install -e '../torchgeo-bench[dev]'
ln -s EuroSAT data/eurosat
torchgeo-bench run -m imagestats -d eurosat --bands all --image-size 64 --device cuda:0 --batch-size 256 -o output/torchgeo-bench-imagestats.csv eval.merge_val=false eval.c_range='[-5,-2,13]'
```

## Architecture

`src/data.py` is the dataset contract. `TIFF_BAND_NAMES` gives the physical TorchGeo order; the older `BAND_NAMES`/SWIR aliases are frozen compatibility labels and must not be silently reinterpreted. `iter_images` streams original TIFFs. `src/cache.py` is the historical uint16 image-cache path.

`src/features.py` is the deployable zero-learned-parameter feature pipeline. `patch_features` appends spectral statistics, multiscale gradient summaries, coherence, orientation, cross-band, index-texture, Hough-line, and Harris-corner families according to explicit keyword configuration and returns both the matrix and ordered names.

`src/linmodel.py` is the primary tiny-head implementation. It standardizes selected features only for optimization, folds the scaler into the learned affine weights, and predicts directly from raw features. Its reference-class helpers subtract and remove one class row, giving the exact stored count `(K - 1) * (F + 1)` without changing predictions. `src/mlp.py` and `src/lowrank.py` contain alternative folded heads explored and largely ruled out for the frontier.

`src/select.py` implements train-only repeated stratified CV, backward-greedy elimination, and floating backward selection. Candidate removals are distributed through `ProcessPoolExecutor`; worker globals avoid repeatedly pickling the full feature matrix.

`experiments/` contains exploratory searches and feature-family prototypes. Successful fixed feature families are promoted into `src/features.py`. `submissions/<number>_<name>/` packages a reproducible result with its own `train.py`, `eval.py`, checked-in `model.npz`, and rationale. The root `train.py` is a separate conventional TorchGeo/Lightning neural baseline, not the minimal linear submission pipeline.

Use `RESULTS.md`, `REPRODUCIBILITY.md`, and `results/` for the 171/279/306-value release and publication caveats. `src/frontier.py` freezes the final 33-feature order. Timestamped files in `ideas/` and older experiment scripts are historical evidence, not the public reproduction entry point.

## Repository-specific conventions

- Do not use the test split to choose a feature family, subset, `C`, seed, or checkpoint. Selection uses train-only CV and/or validation; test is a final informational measurement.
- Greedy selection CV is optimistically biased. Confirm a candidate floor with disjoint verification seeds and held-out validation. Later searches conventionally use select seeds starting at `0`, verification block `10..19`, and a second verification block `30..39`.
- Preserve `patch_features` output order. Saved `feature_idx` arrays and the cached `o6 + line + corn2` layout depend on stable indices: 305 core features, 9 Hough-line features, then 6 Harris-corner features.
- Store every feature-extraction option needed for evaluation in `model.npz`. Evaluation scripts reconstruct the configuration from checkpoint fields and should bypass derived feature caches when claiming raw-data reproducibility.
- Count only learned values, but count all of them. Fixed arithmetic feature extraction is zero-parameter; a folded scaler adds no deployed values; full affine heads use `K * (F + 1)` and reference-class heads use `(K - 1) * (F + 1)`.
- Keep raw image caches as `uint16` on disk and convert to `float32` for feature computation. Derived cache names are split-prefixed under `data/cache/`; all data and caches remain Git-ignored.
- In multiprocessing experiment scripts, set `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, and `MKL_NUM_THREADS` to `1` before importing NumPy or scikit-learn to prevent BLAS oversubscription.
- Experiment scripts commonly expose configuration through module constants and environment variables such as `WORKERS`, `C_GRID`, `START`, `STOP`, and `PART`. Reuse these controls instead of creating near-duplicate scripts.
- Training scripts overwrite their local checkpoint. Exact historical coefficient refits can vary with scikit-learn/SciPy versions, so prefer evaluation scripts for smoke checks and do not keep an unintended regenerated binary.
