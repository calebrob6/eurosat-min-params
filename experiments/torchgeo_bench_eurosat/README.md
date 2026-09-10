# Frozen backbone embeddings

This runner uses TorchGeo-bench to extract EuroSAT embeddings from ResNet-50, ConvNeXt-Tiny, DOFA Large, and OlmoEarth v1.2 Nano/Base. The backbones stay frozen.

From the repository root:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
uv venv --python 3.13 output/benchmark-env
uv pip install --python output/benchmark-env/bin/python \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt

output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --download --extract-only
```

Embeddings go to `output/benchmark-embeddings/`, with matching metadata in `output/backbones/`. Sample filenames are saved because the training loader shuffles. The metadata includes model settings, physical bands, weights and embedding hashes, and library versions.

Omit `--extract-only` to fit and evaluate a linear classifier too. C is selected on validation; final weights are fit on train only. `--models` selects fewer backbones and `--device` chooses a CUDA device.

OlmoEarth consumes 12 bands at 64x64, omitting B10. The other models consume 13 bands resized to 224x224. The wrappers use their pretrained normalization.

The [representation comparison](../representation_overlap/README.md) uses these embeddings to measure feature overlap and combined-classifier accuracy. Its defaults read the output paths above.

The article's compact backbone scores are in [blog_post/scoreboard.csv](blog_post/scoreboard.csv). They are recorded results, not promises of bit-identical refits. The saved package versions and weight revisions are under `reproduced/`.
