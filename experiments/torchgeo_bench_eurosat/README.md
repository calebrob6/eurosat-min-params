# Frozen backbone embeddings

This runner uses TorchGeo-bench to extract EuroSAT embeddings from ResNet-50, ConvNeXt-Tiny, DOFA Large, and OlmoEarth v1.2 Nano/Base by default. It also supports six other backbones. The backbones stay frozen.

It needs CUDA and its own environment, because TorchGeo-bench is a separate project with its own pinned dependencies. From the repository root:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
uv venv --python 3.13 output/benchmark-env
uv pip install --python output/benchmark-env/bin/python \
  --no-deps --default-index https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
uv pip install --python output/benchmark-env/bin/python \
  --default-index https://pypi.org/simple \
  -r experiments/torchgeo_bench_eurosat/requirements.txt

output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --download --extract-only
```

The CUDA wheels install first and on their own, because that index does not carry the other dependencies.

Embeddings go to `output/benchmark-embeddings/`, with matching metadata in `output/backbones/`. Sample filenames are saved because the training loader shuffles.

Omit `--extract-only` to fit and evaluate a linear classifier too. C is selected on validation; final weights are fit on train only. `--models` selects specific backbones, `--all-models` runs all eleven, and `--device` chooses a CUDA device. The extra model keys are `resnet18`, `vit_base`, `dofa_base`, `olmoearth_small`, `earthloc_s2_resnet50`, and `resnet50_s2all_moco`.

`--fractions` additionally runs the seven nested training fractions, and `--datasets eurosat eurosat-spatial` covers both split protocols:

```bash
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --models resnet50 convnext_tiny dofa_base dofa_large \
    olmoearth_nano olmoearth_small olmoearth_base \
  --datasets eurosat eurosat-spatial --fractions --download
```

OlmoEarth consumes 12 bands at 64x64, omitting B10. The other models consume 13 bands resized to 224x224; EarthLoc's wrapper further resizes to 320x320. The wrappers use their pretrained normalization.

The [representation comparison](../representation_overlap/README.md) uses these embeddings to measure feature overlap and combined-classifier accuracy. Its defaults read the output paths above.

Recorded scores are in [blog_post/](blog_post/README.md).

To recompute the parameter and FLOP profiles:

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

Those profiles use 12-channel 224x224 inputs, so their adapted-convolution parameter counts differ from the 13-channel extraction above, and throughput is hardware-dependent.
