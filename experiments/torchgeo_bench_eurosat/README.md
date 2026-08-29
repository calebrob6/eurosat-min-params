# Exact EuroSAT runs with torchgeo-bench

These results were computed with [`torchgeo/torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) at commit `c95391940d58358dd4c522c31703ee12ac030357`.

Every run uses the plain `eurosat` dataset wrapper and the same fixed split files as this repository:

| Split | Images | SHA-256 |
|---|---:|---|
| train | 16,200 | `1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695` |
| val | 5,400 | `b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330` |
| test | 5,400 | `cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592` |

The benchmark was configured with `eval.merge_val=false`, so `C` is selected on validation and the final probe is fit on the 16,200 training images only. All confidence intervals use the benchmark's 200 bootstrap resamples with seed 0.

## Results

| Model | Bands used | Input size | Method | Test accuracy | 95% bootstrap interval |
|---|---:|---:|---|---:|---:|
| ImageStats | 13 | 64 | KNN-5 | 0.8467 | 0.8378-0.8552 |
| ImageStats | 13 | 64 | linear | 0.9065 | 0.8991-0.9143 |
| DOFA Large | 13 | 224 | KNN-5 | 0.9574 | 0.9524-0.9628 |
| DOFA Large | 13 | 224 | linear | 0.9833 | 0.9802-0.9865 |
| OlmoEarth v1 Large | 12 | 64 | KNN-5 | 0.9824 | 0.9791-0.9859 |
| OlmoEarth v1 Large | 12 | 64 | linear | 0.9902 | 0.9878-0.9926 |

`ImageStats` computes mean, population standard deviation, maximum, and minimum for each channel. DOFA consumes all 13 EuroSAT bands. OlmoEarth has no B10 cirrus slot, so the torchgeo-bench wrapper skips B10 and uses the other 12 bands even though the result row records the requested band configuration as `all`.

The linear probes use 1,024-dimensional embeddings, so each benchmark checkpoint stores `10 * (1024 + 1) = 10,250` task-specific probe parameters. The frozen backbones contain 337,151,533 parameters for DOFA Large and 668,045,312 parameters for OlmoEarth v1 Large according to `results/compute_cost.csv` in the benchmark repository.

## Local-model confidence intervals

For a direct uncertainty comparison, `local_models_bootstrap.csv` applies the same 200-resample, seed-0 percentile bootstrap used by torchgeo-bench to this repository's test predictions.

## Commands

```bash
torchgeo-bench run -m imagestats -d eurosat --bands all --image-size 64 \
  --device cuda:0 --batch-size 256 --bootstrap 200 \
  -o imagestats.csv eval.merge_val=false eval.c_range='[-5,-2,13]'

torchgeo-bench run -m torchgeo/dofa_large -d eurosat --bands all \
  --image-size 224 --device cuda:0 --batch-size 32 --bootstrap 200 \
  -o dofa_large.csv eval.merge_val=false

torchgeo-bench run -m olmoearth_large -d eurosat --bands all \
  --image-size 64 --device cuda:0 --batch-size 32 --bootstrap 200 \
  -o olmoearth_large.csv eval.merge_val=false
```

The available PyTorch build bundled a cuDNN version that does not support the machine's V100 GPUs. The DOFA and OlmoEarth commands were invoked through Python with `torch.backends.cudnn.enabled = False`, using PyTorch's native CUDA convolution kernels without changing the model architecture, weights, inputs, or evaluation procedure.
