# EuroSAT Minimum-Parameter Experiments

This repository explores how few learned parameters are needed to exceed 94%
test accuracy on the 13-band EuroSAT land-use classification dataset. Reusable
data, feature, and model code lives in `src/`; exploratory runs live in
`experiments/`; and reproducible trained models live in `submissions/`.

## Setup

Create a virtual environment and install the classical and neural experiment
stacks:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cu128.txt
```

`requirements-cu128.txt` selects the CUDA 12.8 PyTorch wheels used by the experiment host. Use `requirements.txt` instead for a platform-neutral installation.

Download the multispectral EuroSAT archive and all three official split files
through TorchGeo:

```bash
python - <<'PY'
from torchgeo.datasets import EuroSAT

for split in ("train", "val", "test"):
    EuroSAT(root="data/EuroSAT", split=split, download=True)
PY
```

The repository's loaders expect the resulting structure to include:

```text
data/EuroSAT/
|-- eurosat-train.txt
|-- eurosat-val.txt
|-- eurosat-test.txt
`-- ds/images/remote_sensing/otherDatasets/sentinel_2/tif/
```

`data/`, generated caches, training output, and checkpoints are ignored by
Git.

## Run experiments

Reproduce the current 171-parameter linear submission:

```bash
python submissions/12_reference_class_linear/train.py
python submissions/12_reference_class_linear/eval.py
```

The first command automatically builds NumPy caches under `data/cache/` when
needed. See the submission's
[README](submissions/12_reference_class_linear/README.md) for its model,
parameter count, and evaluation protocol.

To run the TorchGeo neural baseline instead:

```bash
python train.py --data-dir data/EuroSAT --model resnet18 --epochs 10
```

Logs and checkpoints are written beneath `output/`. Use
`python train.py --help` to view band, optimizer, hardware, and smoke-test
options.

## Fast GPU C sweeps

The [`torchgeo/torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) CUDA `LogisticRegression` implementation can be installed editable into the same environment and pointed at the existing EuroSAT download:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git ../torchgeo-bench
python -m pip install -e '../torchgeo-bench[dev]'
ln -s EuroSAT data/eurosat
```

Run an ImageStats C sweep without re-downloading the dataset:

```bash
torchgeo-bench run -m imagestats -d eurosat --bands all --image-size 64 \
  --device cuda:0 --batch-size 256 --bootstrap 200 \
  -o output/torchgeo-bench-imagestats.csv \
  eval.merge_val=false eval.c_range='[-5,-2,13]'
```

See [`experiments/torchgeo_bench_eurosat/README.md`](experiments/torchgeo_bench_eurosat/README.md) for the exact DOFA, OlmoEarth, and ImageStats benchmark configurations previously run on these splits.
