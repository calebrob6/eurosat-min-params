# EuroSAT backbone reproduction with TorchGeo-bench

The article's original scoreboard and fraction CSVs have now been recovered under [`../imported/`](../imported/). Fresh runs in [`reproduced/`](reproduced/) independently evaluate all **11 backbone scoreboard rows** and **98 learning-curve points** using unmodified TorchGeo-bench APIs.

## Fresh reproduction

Run from this repository's root on a CUDA GPU. This is separate from the CPU-only `reproduce.py` environment; pretrained weights and the cloned source are never added to Git.

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
python3.13 -m venv output/benchmark-env
output/benchmark-env/bin/python -m pip install \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt

# Scoreboard-only models.
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py --download \
  --models earthloc moco resnet18 vit_base --output output/blog-backbones

# The seven models plotted in the article, both split protocols and all fractions.
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py --download \
  --models resnet50 convnext_tiny dofa_base dofa_large olmoearth_nano olmoearth_small olmoearth_base \
  --datasets eurosat eurosat-spatial --fractions --output output/blog-backbones

python experiments/torchgeo_bench_eurosat/compare.py --results-dir output/blog-backbones
```

The commands above generate new results under `output/blog-backbones/`, leaving all checked-in original and reproduction CSVs untouched. To assemble the article's original table and figure data without running models, use `python export_blog_results.py` from the repository root; it combines the imported backbones with the local model results under `output/blog-results/`.

Fresh measurements were made with Python 3.13.13, PyTorch 2.11.0+cu128, TorchGeo 0.10.0, timm 1.0.29, olmoearth-pretrain-minimal 0.0.6, and H100 NVL MIG GPUs. `reproduced/environment-freeze.txt` records all installed dependency versions. Per-model JSONs record resolved configs, physical bands, input size, GPU/runtime versions, frozen model-state hashes, extraction order, and embedding hashes.

`reproduced/hf_weight_revisions.json` records the Hugging Face snapshots resolved for timm and OlmoEarth. TorchGeo weight-enum identifiers are in the model JSONs and their download URLs are defined by the pinned TorchGeo package. The upstream wrappers resolve pretrained weights themselves; record and compare the backbone-state hash if a remote model repository changes.

`run.py` uses upstream `get_datasets`, model configuration/instantiation, `embed_split`, and `evaluate_logistic`. The only additions are source-index recording, checked embedding caches, nested fraction selection, and output/provenance handling. An untouched execution of the imported original fraction runner for ResNet-50 matched all six scores and selected Cs exactly; its output is `reproduced/original_runner_resnet50_control.csv`.

The model weights are frozen. Images use all 13 bands except OlmoEarth, which omits unsupported B10. OlmoEarth uses native 64x64 inputs and its pretrained normalization; other models use 224x224 bilinear inputs and the configured BandSpec normalization. No embedding StandardScaler or L2 normalization is added beyond the upstream implementation.

Fractions use **nested unstratified** seed-0 prefixes of a random permutation of the shuffled training embeddings, matching the original runner. This differs intentionally from our handcrafted models' five-seed stratified curves. Each backbone row selects C over the upstream 40-point `10^-6..10^4` grid on validation, fits on train only (`merge_val=false`), and reports test accuracy with 200 bootstrap resamples. The non-Olmo random full-data scoreboard used the training-loader order directly; OlmoEarth's full-data rows retain the permutation used by its fraction sweep, even in a full-only rerun. Full-data outputs for all models go under the selected output directory's `scoreboard/` subdirectory, separately from the curve CSVs.

Seed 0 does **not** imply the same subset across backbones: model initialization consumes Torch random state before the shuffled training loader runs. For example, the reproduced 162-image ResNet-50 subset shares only two images with the ConvNeXt-Tiny subset. This is inherited from the original runner; the new runner deliberately preserves it rather than silently changing the experiment. Subset hashes and cached sample filenames expose the difference. Bootstrap intervals cover test-image resampling, not uncertainty from these different training draws.

Embedding files, including actual sample filenames in extraction order, are stored under `output/benchmark-embeddings/`. By default they are reused only when their SHA-256, source/config, sampling protocol, batch size, and package versions match. Pass `--fresh` to re-extract. Use `run.py --output output/my-benchmark` and then `compare.py --results-dir output/my-benchmark` to keep a new result set and its comparisons separate from the checked-in CSVs; `compare.py --scoreboard-only` skips curve comparison when only full-data models were rerun. Batch sizes match the original scoreboard manifest (OlmoEarth 32, DOFA Base 32/Large 16, timm ResNets/ConvNeXt 64, ViT and EarthLoc 32).

## Original versus fresh scoreboard

| Model | Article/original | Fresh | Correct-image difference / 5,400 |
|---|---:|---:|---:|
| EarthLoc ResNet-50 | 86.43% | 86.37% | -3 |
| S2-all MoCo ResNet-50 | 92.78% | 92.80% | +1 |
| ResNet-18 | 93.93% | 93.91% | -1 |
| ResNet-50 | 94.96% | 95.07% | +6 |
| ViT Base | 95.15% | 95.19% | +2 |
| ConvNeXt-Tiny | 95.24% | 95.24% | 0 |
| DOFA Base | 97.50% | 97.39% | -6 |
| DOFA Large | 98.33% | 98.30% | -2 |
| OlmoEarth v1.2 Nano | 96.89% | 96.89% | 0 |
| OlmoEarth v1.2 Small | 98.65% | 98.67% | +1 |
| OlmoEarth v1.2 Base | 98.80% | 98.80% | 0 |

This is close numerical reproduction, **not universal bit-identical reproduction**. Every full-data difference is at most six images (0.1111 percentage points). The original environment was not completely locked; small numerical changes can also move the validation-selected C. No settings were chosen to minimize test differences.

`reproduced/scoreboard_comparison.csv` and `reproduced/fraction_comparison.csv` pair every fresh number with its original CSV, selected C, and bootstrap bounds. Across 98 curve points, 22 match exactly and the largest difference is 0.4444 percentage points (24 images) at 1% spatial data for ConvNeXt-Tiny. All 98 fresh point estimates fall within the original bootstrap intervals; that is a descriptive comparison, not a formal equivalence test.

Actual loaded backbone and probe counts are in every fresh row. In particular, the 13-band ViT has **87,764,736** backbone values rather than the article's rounded 12-band profiling count of 87.57M: one extra ViT patch-embedding channel adds 196,608 values, not merely a few thousand.

## Older archived benchmark runs

The top-level `dofa_large.csv`, `imagestats.csv`, and `olmoearth_large.csv` below predate the recovered article runs. The OlmoEarth row is **v1 Large**, not v1.2 Nano/Small/Base. The benchmark ImageStats row (90.65%) is not our independently standardized, 90.96% ImageStats baseline.

These archived results were computed with [`torchgeo/torchgeo-bench`](https://github.com/torchgeo/torchgeo-bench) at commit `c95391940d58358dd4c522c31703ee12ac030357`.

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

The DOFA and OlmoEarth probes use 1,024-dimensional embeddings, so their heads store `10 * (1024 + 1) = 10,250` task-specific probe parameters. ImageStats uses 52 features and a 530-value full head. The frozen backbones contain 337,151,533 parameters for DOFA Large and 668,045,312 parameters for OlmoEarth v1 Large according to `results/compute_cost.csv` in the benchmark repository.

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
