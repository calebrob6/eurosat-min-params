# Frozen backbone embeddings

This runner uses TorchGeo-bench to extract EuroSAT embeddings from ResNet-50, ConvNeXt-Tiny, DOFA Large, and OlmoEarth v1.2 Nano/Base by default. It also supports the six other backbones in the archived scoreboard. The backbones stay frozen.

From the repository root:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
uv venv --python 3.13 output/benchmark-env
uv pip install --python output/benchmark-env/bin/python \
  --no-deps --default-index https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
uv pip install --python output/benchmark-env/bin/python \
  --default-index https://pypi.org/simple \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt

output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --download --extract-only
```

The first install fetches only the two pinned CUDA wheels from the official PyTorch index. The second preserves those wheels and installs all remaining dependencies from PyPI with the historical constraints. Both steps must finish before running Python. This avoids exposing unrelated dependencies to the CUDA index: uv's first-index policy otherwise rejects pins such as `filelock` when that index contains only older versions. Neither the index policy nor any dependency pin is relaxed.

Keep this environment separate from the [standalone CPU reproduction](../../README.md). The locked backbone/overlap environment uses NumPy 2.5.2, whereas the CPU baseline pins NumPy 2.4.4. Use each workflow's declared environment for feature generation and fitting; installing plotting libraries does not make the two environments interchangeable. This also applies to figure scripts that refit the CPU baseline, rather than merely render already-computed tables.

Embeddings go to `output/benchmark-embeddings/`, with matching metadata in `output/backbones/`. Sample filenames are saved because the training loader shuffles. The metadata includes model settings, physical bands, weights and embedding hashes, and library versions.

Omit `--extract-only` to fit and evaluate a linear classifier too. C is selected on validation; final weights are fit on train only. `--models` selects specific backbones and `--device` chooses a CUDA device.

To rerun the full eleven-backbone archived scoreboard rather than only the five representation-study backbones (the article's displayed table omits EarthLoc):

```bash
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py --all-models --download
```

The extra model keys are `resnet18`, `vit_base`, `dofa_base`, `olmoearth_small`, `earthloc_s2_resnet50`, and `resnet50_s2all_moco`. `--fractions` additionally runs the seven nested training fractions; full-data scoreboard CSVs remain separate under `output/backbones/scoreboard/`.

## Article training-fraction figures

Figures 7 and 8 include seven neural backbones, not just the five overlap-study defaults. Run both random and spatial curves with:

```bash
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --models resnet50 convnext_tiny dofa_base dofa_large \
    olmoearth_nano olmoearth_small olmoearth_base \
  --datasets eurosat eurosat-spatial --fractions --download
```

Each curve CSV under `output/backbones/` contains seven nested seed-0 fractions. For Figure 7, the four non-Olmo 100% points instead use the unpermuted full-data rows under `output/backbones/scoreboard/`; do not substitute the permuted 100% curve rows. The original renderer copied those four points from the rounded scoreboard and drew degenerate intervals there. All spatial and Olmo points come from their fraction sweeps.

The original curve targets are retained under [`blog_post/training_fractions/`](blog_post/training_fractions/), restored byte-for-byte from commit `924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c` (`aed37c9^`). Their Git provenance can be checked directly, without a neighboring repository or a historical worktree:

```bash
git show 924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c:experiments/torchgeo_bench_eurosat/blog_post/training_fractions/random/resnet50.csv
```

Filter those archives to `method=linear`; their KNN rows were not plotted. The neural curves use one subsampling seed with conditional bootstrap intervals, whereas the handcrafted curves summarize five subsampling seeds.

[`reproduced/fraction_comparison.csv`](reproduced/fraction_comparison.csv) is a historical comparison of original results with an earlier independent rerun, not output from the current invocation. Its 100% raw curve rows use the nested permutation, including the four non-Olmo random rows; those are not the separate full-scoreboard endpoints used by Figure 7. New measurements always go under `output/backbones/`, leaving these reference tables unchanged.

## Inputs and compute profiles

OlmoEarth consumes 12 bands at 64x64, omitting B10. The other models consume 13 bands resized to 224x224; EarthLoc's wrapper further resizes to 320x320. The wrappers use their pretrained normalization.

The [representation comparison](../representation_overlap/README.md) uses these embeddings to measure feature overlap and combined-classifier accuracy. Its defaults read the output paths above.

The compact backbone scores, including the additional EarthLoc row, are in [blog_post/scoreboard.csv](blog_post/scoreboard.csv). They are recorded results, not promises of bit-identical refits. The saved package versions and weight revisions are under `reproduced/`.

`blog_post/compute_cost.csv` contains separate synthetic profiling measurements: its inputs have 12 channels at 224x224, not the extraction inputs above. In particular, its adapted-convolution parameter counts can differ from the 13-channel models recorded by this runner, and its OlmoEarth throughput does not describe native-64x64 extraction.

To recompute those separate profiles using the same pinned environment:

```bash
for config in \
  timm/resnet18 timm/resnet50 timm/vit/vit_base_patch16_224 timm/convnext_tiny \
  torchgeo/earthloc_s2_resnet50 torchgeo/resnet50_s2_all_moco \
  torchgeo/dofa_base torchgeo/dofa_large \
  olmoearth_v1_2_nano olmoearth_v1_2_small olmoearth_v1_2_base; do
  output/benchmark-env/bin/torchgeo-bench flops "model=$config" \
    'band_configs=[s2]' 'seg_band_configs=[]' image_size=224 \
    probe_num_classes=10 timing_batch_size=64 n_warmup=3 n_measure=20 \
    device=cuda:0 output=output/backbone-compute-cost.csv
done
```

FLOPs and parameter counts describe the configured graph; throughput, latency, and memory are hardware- and workload-dependent. Run timing measurements without other work on the selected device.
