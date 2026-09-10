# Reproducibility findings

## Scope and provenance

This audit started on September 10, 2026, from clean commit `aed37c9cfb76d761797bc68132b8e349d4fdec19` of this repository. The target is [Solving EuroSAT with as Few Parameters as Possible](https://geospatialml.com/posts/eurosat-min-params/). The website could not be fetched in this environment: HTTPS failed during TLS negotiation and HTTP returned status 470. The author then supplied its raw source under `blog_post/`; that supplied source, rather than an inferred version of the article, defines the claims covered here. Those untracked source files are not modified or included in this PR.

There were initially no repository-local environments, datasets, downloaded backbone weights, image/feature caches, or generated results. All measured results below use newly downloaded original images and fresh environments. Package downloads may use package-manager caches; that does not reuse fitted models or extracted representations. Evaluating a committed checkpoint, refitting a classifier, reproducing feature discovery, and exporting an existing result table are different operations and are identified separately.

The supplied `blog_post/index.qmd` has SHA-256 `cc280f909d73a1dc1d2fd867d28d9c9c36d0d0904bb6ddfe58d07fdefa9a8473`; its original `figures-data.js` has SHA-256 `5e2dc5abaded3b4901d99da5bdf3a849ca277201d458faa94e7ccbd0865a80d8`. These identify the audited source without assuming that a front-matter draft flag establishes the live website's publication state.

**Not every article result can be reproduced exactly. The central EuroSAT result does reproduce, including a byte-identical refit, and the complete representation-overlap study reproduces byte-for-byte.** The article also includes historical models, ablations, neural learning curves, exploratory approaches, and interactive figures that are not all exposed by the original basic commands. Some original fit metadata is genuinely missing; other material is recoverable from Git history.

| Article component | Audit outcome |
|---|---|
| 306-value model and ImageStats | Exact saved-checkpoint evaluation and refits |
| Historical 171-/279-value models | Exact evaluations and bit-identical weight refits |
| Local random/spatial learning curves | All 28 rows / 140 fits match existing checks |
| Representation-overlap study | All 9,610 CSV rows, three figures, and README byte-identical |
| Main neural scoreboard | All models rerun; 3 of 10 article rows match reported rounding |
| Neural learning curves | All 98 points rerun; 23 match original reported accuracy, all raw curves match the archived independent rerun |
| RESISC45 | Fixed33 matches; ImageStats 36.59%, not historical 36.63% |
| Means / means+std ablations | New-reference results match; historical 74.8% / 87.8% not recovered |
| CNN / MOSAIKS | Full-data measurements obtained, with missing deployment artifacts and a preserved MOSAIKS assertion failure |
| Article illustrations | 38 image assets byte-identical; JavaScript/model/rendering discrepancies documented |
| Full adaptive search and qualitative failed-method claims | Not independently reproduced as exhaustive searches |

The important archival revision is `924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c`, the parent of the repository-simplification commit. It preserves the 18- and 30-feature checkpoints, supporting baseline runners, exploratory scripts, and neural learning-curve tables. Their absence from the current tree must not be confused with absence from the public repository's history.

## Fresh environments and commands

The host is Linux x86-64 with an AMD EPYC 9V84 CPU, 80 visible CPU cores, and ample memory/disk. Two NVIDIA H100 NVL MIG `3g.47gb` devices are available to PyTorch. The initial `nvidia-smi` query could not report memory/utilization, but actual CUDA matrix multiplication subsequently worked. None of the host's preexisting Python packages were used as a substitute for installing the documented environments.

`uv` was initially absent. The audit installed it in an isolated tooling environment, then followed the CPU instructions:

```bash
python -m venv output/tooling-env
output/tooling-env/bin/python -m pip install uv
output/tooling-env/bin/uv venv --python 3.13 .venv-reproduce
output/tooling-env/bin/uv pip install --python .venv-reproduce/bin/python \
  -r requirements-reproduce.txt

.venv-reproduce/bin/python reproduce.py --download --check
.venv-reproduce/bin/python reproduce.py --download --refit --fractions --check \
  --output output/reproduction-audit/eurosat-refit
.venv-reproduce/bin/python -m experiments.resisc45.run --download --check
```

The CPU environment used Python **3.13.13**, NumPy **2.4.4**, SciPy **1.18.1**, scikit-learn **1.9.0**, rasterio **1.5.1**, and GDAL **3.12.4**. The runners set OpenMP/OpenBLAS/MKL thread counts to one before numerical imports. The main package pins installed without changes. The complete CPU freeze is retained locally at `output/reproduction-audit/cpu-freeze.txt`.

The initial EuroSAT command completed in **7m 13.757s**, including download and extraction. The second command, which independently reread the TIFFs, refitted the final head, and ran all local learning curves, completed in **6m 33.793s**. RESISC45 completed in **206.347s**. These are observations on this host, not portable performance promises. An initial timing wrapper failed because `/usr/bin/time` was absent; rerunning the actual commands with Bash's `time` succeeded. That wrapper failure was not a repository defect.

## Dataset integrity and protocol

EuroSAT was downloaded from the pinned `torchgeo/eurosat` Hugging Face revision `1ce6f1bfb56db63fd91b6ecc466ea67f2509774c`. All 27,000 TIFFs were read in each complete core run. The random splits contain **16,200 / 5,400 / 5,400** distinct images across ten classes. RESISC45 was downloaded from `isaaccorley/resisc45` revision `883edc0eee77b2c84225472f10f126e3ed83fa6e`; its splits contain **18,900 / 6,300 / 6,300** images across 45 classes. All 31,500 extracted RESISC45 JPEGs were also compared byte-for-byte with their archive members.

| Artifact | Verified SHA-256 |
|---|---|
| EuroSAT archive | `751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59` |
| EuroSAT random train split | `1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695` |
| EuroSAT random validation split | `b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330` |
| EuroSAT random test split | `cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592` |
| EuroSAT spatial train split | `2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f` |
| EuroSAT spatial validation split | `6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b` |
| EuroSAT spatial test split | `de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452` |
| RESISC45 archive | `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53` |
| RESISC45 train split | `ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b` |
| RESISC45 validation split | `08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342` |
| RESISC45 test split | `e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805` |

EuroSAT inputs are original-scale, float32, 13-band 64x64 patches, not normalized display RGB images or GEO-Bench's modified dataset. The physical channel order is `B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B8A`. Historical feature aliases deliberately retain their original numerical indices: `B_SWIR1=11` is physical B12 and `B_SWIR2=12` is physical B8A. Consequently, the saved recipe's `nbr` is not the conventional burn-ratio band pairing. Renaming or correcting those indices would change the experiment rather than reproduce it.

RESISC45 uses rasterio's bilinear uint8 read from original 256x256 JPEGs to 64x64, followed by conversion to float32. This is not interchangeable with an arbitrary Pillow/torchvision resize.

All current reference heads fit training images only. ImageStats regularization is selected on validation, without merging validation into training. The 33-feature EuroSAT refit uses the already selected recipe and fixed `C=3`.

## Core EuroSAT models: reproduced

| Model / operation | Learned values | Validation correct / 5,400 | Test correct / 5,400 | Test accuracy |
|---|---:|---:|---:|---:|
| Saved 33-feature head evaluated on fresh features | 306 | 5,193 | 5,186 | 96.037037% |
| Fresh 33-feature head, fixed recipe and C=3 | 306 | 5,193 | 5,186 | 96.037037% |
| Fresh ImageStats, validation-selected C=300 | 477 | 4,910 | 4,912 | 90.962963% |

Both commands passed the unmodified `--check` comparisons for aggregate results, all 40 validation/test per-class rows, the exact ordered 33-feature schema, and all 22 entries of the ImageStats regularization sweep. Validation accuracies round to the article's **96.17%** and **90.93%**, respectively.

The newly trained checkpoint equals the saved checkpoint in every array and metadata field, and also as an entire file:

```text
models/eurosat_33.npz
output/reproduction-audit/eurosat-refit/eurosat_33.npz
SHA-256: e1bb5faf06f8bfed6d7c8bcde86e8448869cd58ac468ce2641cb2a044291ccec
```

The parameter arithmetic is correct: a 10-class affine head can represent one class with a constant zero logit and store only nine rows. Therefore, 33 features require `9*(33+1)=306` learned values; 52 ImageStats features require `9*(52+1)=477`. The standardizer is folded into that affine map. The article's comparison convolution has `13*3*3*3+3=354` parameters. None of this proves that 306 is a global minimum over all possible feature recipes.

The final recipe has **32 columns from the 377-feature pool plus `tail_aniso_low_ndvi`**, not a literal 33-column slice of that pool. Its family counts are 2 per-band statistics, 8 percentiles, 4 multiscale gradients, 4 orientation entropies, 7 spectral-index texture measurements, 1 Hough measurement, 3 Harris measurements, 2 local binary patterns, 1 connected-component measurement, and 1 region-shape measurement, totaling 33.

The reproduced test-class comparisons are:

| Class | Images | ImageStats (%) | Final33 (%) |
|---|---:|---:|---:|
| AnnualCrop | 596 | 91.61 | 96.48 |
| Forest | 608 | 98.19 | 98.68 |
| HerbaceousVegetation | 573 | 90.05 | 95.29 |
| Highway | 496 | 72.78 | 91.53 |
| Industrial | 501 | 92.02 | 95.81 |
| Pasture | 396 | 90.66 | 94.19 |
| PermanentCrop | 538 | 84.20 | 91.26 |
| Residential | 554 | 93.68 | 99.46 |
| River | 529 | 93.19 | 96.22 |
| SeaLake | 609 | 99.67 | 99.67 |

## Local learning curves: reproduced

All **28** rows were regenerated: two feature sets, two split protocols, seven fractions, and five seeds per fraction, for **140 classifier fits**. Every individual seed accuracy, mean, and sample standard deviation passed the existing absolute tolerance of `1e-6` in accuracy units against the rounded saved CSVs. No tolerance was changed for these results. The 100% row repeats the same full training set five times, so its zero standard deviation is not an estimate of uncertainty over independent datasets.

The following entries are fresh test accuracy percentages, `mean +/- sample standard deviation`:

| Fraction | Training images | Final33 random | Final33 spatial | ImageStats random | ImageStats spatial |
|---|---:|---:|---:|---:|---:|
| 1% | 162 | 86.7444 +/- 1.0330 | 86.5111 +/- 2.3216 | 72.5519 +/- 0.7222 | 68.1815 +/- 5.1162 |
| 2% | 324 | 89.3630 +/- 0.8574 | 87.4222 +/- 1.0040 | 76.9481 +/- 1.6457 | 71.8037 +/- 3.8651 |
| 5% | 810 | 92.8815 +/- 0.3139 | 89.9630 +/- 1.3302 | 81.9593 +/- 0.8641 | 72.6926 +/- 3.1581 |
| 10% | 1,620 | 94.2444 +/- 0.2584 | 90.5370 +/- 0.8266 | 86.0519 +/- 0.2437 | 77.4148 +/- 2.8237 |
| 20% | 3,240 | 94.9148 +/- 0.1710 | 91.6889 +/- 0.3473 | 88.5630 +/- 0.1945 | 81.1037 +/- 1.7549 |
| 50% | 8,100 | 95.7000 +/- 0.1122 | 92.6963 +/- 0.2663 | 90.3222 +/- 0.2110 | 83.5926 +/- 1.2193 |
| 100% | 16,200 | 96.0370 +/- 0.0000 | 92.7593 +/- 0.0000 | 90.9630 +/- 0.0000 | 85.4259 +/- 0.0000 |

These confirm the article's rounded Final33 table, the roughly **3.28 percentage-point** random/spatial gap at 100%, accuracy above 92% with 810 random-split training images, accuracy above 94% with 1,620 images, and roughly **14.2 / 18.3 percentage-point** gains over ImageStats at 1% on random/spatial splits.

The article's exact spatial-overlap count also checks: **4,362 of 5,400** spatial-test images occur in the random training/validation sets; only 1,038 are in the random test set. These curves reuse a recipe discovered on the full random split. They reproduce sample efficiency conditional on that frozen recipe, not feature discovery from 162 labels or an untouched spatial generalization experiment.

## RESISC45: one article result reproduced, one historical mismatch

| Model | Features | Learned values | Selected C | Validation correct / 6,300 | Test correct / 6,300 | Fresh test (%) | Article (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ImageStats | 12 | 572 | 30 | 2,289 | 2,305 | 36.587302 | 36.63 |
| Fixed RGB subset | 33 | 1,496 | 30 | 3,860 | 3,721 | 59.063492 | 59.06 |

All four current reference CSV rows matched exactly. An independent ImageStats implementation also produced identical fitted weights, biases, regularization, and predictions. The 33-feature result agrees with the article after rounding. Its features are an independently selected RGB recipe, not the EuroSAT recipe applied unchanged to three channels.

The historical **36.63%** ImageStats claim was **not reproduced**. The difference from the fresh result is about **-0.0427 percentage points**. The repository already acknowledges that the original fit is missing; the fresh validation-selected **36.59%** is its current reference. That disclosure is useful, but it does not recover the older result or establish why it differed. No C or preprocessing was changed to chase the article's test percentage.

The original RESISC45 feature-discovery search was not rerun by the basic command; it uses the fixed subset. Detailed independent-control and archive-integrity evidence is retained locally in `output/resisc45-reproduction-evidence/reproduction-summary.json`.

## Historical article rows and feature-search provenance

The earlier 18- and 30-feature models were recovered from the pinned archival revision and evaluated on freshly extracted TIFF features. The archived runner passed its exact checks in **5m 38.172s**:

```bash
git worktree add --detach output/reproduction-audit/historical \
  924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c
ln -s ../../../data output/reproduction-audit/historical/data
.venv-reproduce/bin/python output/reproduction-audit/historical/reproduce.py \
  --check --output output/reproduction-audit/historical-models
```

| Frozen subset | Learned values | Fixed C | Validation correct / 5,400 | Test correct / 5,400 | Fresh test (%) | Article (%) |
|---|---:|---:|---:|---:|---:|---:|
| 18 features | 171 | 10 | 5,079 | 5,094 | 94.333333 | 94.33 |
| 30 features | 279 | 10 | 5,147 | 5,150 | 95.370370 | 95.37 |

An additional independent run reread all 27,000 TIFFs and refitted both fixed subsets on training images with `fit_folded_logreg(..., feature_idx=saved_indices, C=10)`, followed by reference-class conversion. Both new weight and bias arrays are **bit-identical** to the archived checkpoints, and every validation/test prediction agrees. That run took **5m 11.951s**; its script and measurements are retained at `output/reproduction-audit/refit_historical_heads.py` and `output/reproduction-audit/historical-refits/metrics.csv`. These are genuine frozen-recipe refits, not a replay of L1 ranking, greedy pruning, or feature exchanges. No archived checkpoint was overwritten.

Thus the two smaller article rows are reproducible, but only after restoring access to historical code. The PR documents the immutable worktree workflow rather than restoring the entire removed research tree into the streamlined package.

Numerical reproducibility must also be distinguished from the history of model selection. At archival revision `924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c`, `submissions/13_reference_class_95/README.md` explicitly discusses a 28-feature candidate failing the test threshold and motivates the stricter validation gate in that context. Its `train.py` prints test accuracy for multiple candidate feature counts, although the programmatic acceptance predicate itself uses validation and cross-validation scores. Those records show that test results were visible during historical exploration; rerunning a frozen model today cannot establish that the entire research process was blind to test feedback. This audit does not rerun or certify the full adaptive feature-discovery history.

## ImageStats text ablations: historical percentages not recovered

The article quotes **74.8%** for per-band means and **87.8%** for means plus standard deviations. The archived supporting-baseline documentation explicitly says that their original C values, fitted coefficients, and sweeps were not retained. It provides separately labeled new reference measurements, not the original fits.

Fresh TIFF statistics and the archived validation-selection recipe reproduced those new references exactly:

| Features | Learned values | Selected C | Validation correct / 5,400 | Test correct / 5,400 | Fresh test (%) | Article (%) |
|---|---:|---:|---:|---:|---:|---:|
| 13 per-band means | 126 | 100 | 4,115 | 4,123 | 76.351852 | 74.8 |
| 26 interleaved means/stds | 243 | 200 | 4,790 | 4,820 | 89.259259 | 87.8 |

The differences from the rounded article numbers are about **+1.5519** and **+1.4593 percentage points**. These are not rounding discrepancies. No test-set score was used to pick a replacement C that would imitate the old quotes.

An initial independently declared control grouped all means before all standard deviations rather than interleaving them, and selected C using reference-class rather than full-head float32 scores. It selected **C=150**, with **4,789** validation-correct and the same **4,820** test-correct images for mean/std. Its predictions nevertheless differed from the archived-order replay on 11 validation and 17 test images. Equal aggregate accuracy is not proof of identical fitted models. Both controls remain recorded rather than discarding the first once the archived ordering was recovered.

The initial fresh run took **81.284s**; replaying the archived recipe on those newly derived, hash-checked statistics took **35.948s**. The canonical addendum and exact reference comparison are under `output/reproduction-audit/text-ablations/archived-reference-run/`. The archived `experiments/article_baselines/run.py` documents the same separately labeled measurements and can rerun the supporting baselines without old image caches.

## Other approaches: fresh measurements and unresolved archival defects

The archived `experiments/article_baselines/other.py` adapters were also exercised on full, freshly decoded EuroSAT data using the neural environment and GPU1. They predeclare configurations rather than searching for an article-matching test score. Original historical checkpoints are unavailable for these article notes, so these runs are not exact historical recovery.

| Method | Declared run | Fresh test result | Article | Execution status |
|---|---|---:|---:|---|
| Single-convolution CNN | 13 bands, 16 filters, seed 0, 60 epochs | 89.33% as printed | 89.2% | Training completed |
| MOSAIKS | Gaussian seed 1, 2,048 filters / 4,096 signed-ReLU features, train-only L1 top 512, C=1 | 5,155/5,400 = 95.462963% | About 95% | Fit measured, original export assertion failed |

The CNN ran for **130.206s** and selected zero-based epoch **53** by validation performance, printing validation accuracy **0.8891** and test accuracy **0.8933**. With exactly 5,400 examples, those rounded figures uniquely imply 4,801 and 4,824 correct predictions, but the trainer does not actually export those predictions or its selected checkpoint. They are inferred counts, not independently inspected saved predictions. Its 89.33% is not a recovery of the missing 89.2% run.

The CNN also trains **2,090 parameters** and reports **2,058 deployed values**, assuming batch-normalization and input-standardization folding. Folding the input centering through its zero-padded convolution needs special boundary treatment: a diagnostic constant input equal to its training mean gives zero normalized input, but ordinary raw zero padding after affine folding introduces nonzero border responses. The measured synthetic border error was **3.333333**, versus below **6e-7** in the interior; padding with the training means restores equivalence. The archived trainer exports neither a folded deployment artifact nor a verified boundary implementation. Thus **2,058 remains a reported deployment count, not an independently verified equivalent saved model**. This does not establish that the reported training accuracy is wrong.

The relevant archived implementation is [`experiments/conv_gap.py`](https://github.com/calebrob6/eurosat-min-params/blob/924f0a80b27f4f2d8fdab4cdfcbd13515ee4989c/experiments/conv_gap.py): lines 75-80 normalize with 13 training means/stds, lines 90-100 define the convolution/BN/classifier tensors, and lines 164-170 optionally export logits rather than a folded model. The companion `experiments/article_baselines/other.py` records the reported deployed count without validating that export.

MOSAIKS completed fitting but failed its unchanged assertion that full-head and reference-class float32 predictions must be exactly equal. The failure repeated. On test entry 4,411, `Highway_1358` (a TIFF; the official split names it `.jpg`), the full float32 head predicted Industrial and the reference head predicted Residential; **both are incorrect**, so both still score **5,155/5,400**. The top-two margin under float64 evaluation was approximately **2.16e-6**. Float64 full/reference evaluation agreed, supporting a near-tie arithmetic explanation, but no assertion was suppressed and no precision, C, seed, or subset was changed to claim a passing historical run.

An explicitly labeled diagnostic captured the same-setting fitted head, fresh features, and predictions before re-raising the original failure. Validation was **5,140/5,400 = 95.185185%** for both forms, with no prediction disagreements. The primary command took **116.541s** and the diagnostic **81.601s**. There were no convergence warnings or out-of-memory failures. scikit-learn emitted deprecated-penalty/default-`l1_ratio` warnings; the installed implementation still honored the explicitly requested L1 penalty.

The reference-class MOSAIKS head has **4,617** learned values, but the nonlinear input pipeline also retains 13 learned means and 13 learned standard deviations: **4,643 learned values in total**. Those cannot be folded into the final affine head across the nonlinear random features. The article's approximate head-only count is therefore distinct from a total learned-state count.

Complete settings, exact failures, diagnostic model/input hashes, and limitations are in `output/reproduction-audit/optional-baselines/summary.json`. These timings are observational and do not claim exclusive GPU use. The retired adapters' assertion and deployment-export limitations are documented, not silently patched in the archival checkout or presented as passing current-package workflows.

The qualitative claims about small MLPs, reduced-rank heads, knowledge distillation, RGB-only variants, and quadratic interactions were **not independently replayed as an exhaustive search**. Historical scripts and notes do not specify one complete, uniquely recoverable experiment for every such conclusion. Repeating a selected configuration would not establish that an entire model family fails. The same limitation applies to claiming a globally minimal parameter count.

## Neural comparisons

### Clean installation failure: fixed and independently rechecked

The original documented installation failed before any neural model ran. `requirements.txt` supplied a blanket CUDA extra index, while the historical freeze pinned `filelock==3.32.5`. With uv's default first-index behavior, the CUDA index's incomplete `filelock` listing prevented resolving that pinned version from PyPI.

The first recovery used `--index-strategy unsafe-best-match` over the two known indexes and installed all historical pins. The final PR fix **does not require relaxing index policy**. It removes the blanket extra index and explicitly installs only the two CUDA-specific packages from the PyTorch repository, followed by all dependencies from PyPI:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
uv venv --python 3.13 output/benchmark-env
uv pip install --python output/benchmark-env/bin/python --no-deps \
  --default-index https://download.pytorch.org/whl/cu128 \
  'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
uv pip install --python output/benchmark-env/bin/python \
  --default-index https://pypi.org/simple \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt
```

A **second initially empty environment**, `output/benchmark-env-index-check`, validated the final procedure. All **121 pinned package versions** matched the saved freeze, with the pinned editable TorchGeo-bench source as package 122; dependency consistency passed. The final suite of 77 backbone/representation software tests passed, and a real ResNet-50 refit reproduced the first measured accuracy, selected C, confidence interval, training-subset hash, and embedding hash. No dependency-version pin was loosened to make installation succeed.

This neural environment uses Python **3.13.13**, PyTorch **2.11.0+cu128**, CUDA **12.8**, and NumPy **2.5.2**. It is deliberately separate from the NumPy-2.4.4 CPU reference environment. The final installation validation is recorded in `output/reproduction-audit/explicit-index-validation.json`.

### Fresh backbone extraction and main scoreboard

The audit downloaded fresh weights, extracted fresh TIFF-derived embeddings, and refitted **all eleven** archived scoreboard configurations. Ten appear in the article; EarthLoc is an additional archived row. The original runner exposed only five aliases, so the PR adds the remaining supported aliases and an explicit `--all-models` option while preserving the original five-model default.

The complete sequence used the existing runner, initially split across devices and model groups. Its equivalent all-model invocation after the fix is:

```bash
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --download --all-models
```

All five saved full backbone-state hashes and all five saved embedding-archive hashes matched their fresh counterparts exactly. All seven recorded Hugging Face revisions resolved to the recorded revisions in this run. Those revision records are provenance, not an unconditional guarantee that every upstream loader enforces the revision on future downloads; future reproduction should compare the generated hashes too.

The refitted scoreboard is not an exact recovery of all historical scores:

| Backbone | Article / archived test (%) | Fresh correct / 5,400 | Fresh test (%) | Same two-decimal percentage? |
|---|---:|---:|---:|---|
| SSL4EO-S12 MoCo ResNet-50 | 92.78 | 5,011 | 92.796296 | No |
| ResNet-18 | 93.93 | 5,071 | 93.907407 | No |
| ResNet-50 | 94.96 | 5,134 | 95.074074 | No |
| ViT-Base/16 | 95.15 | 5,140 | 95.185185 | No |
| ConvNeXt-Tiny | 95.24 | 5,143 | 95.240741 | Yes |
| OlmoEarth v1.2 Nano | 96.89 | 5,232 | 96.888889 | Yes |
| DOFA Base | 97.50 | 5,259 | 97.388889 | No |
| DOFA Large | 98.33 | 5,308 | 98.296296 | No |
| OlmoEarth v1.2 Small | 98.65 | 5,328 | 98.666667 | No |
| OlmoEarth v1.2 Base | 98.80 | 5,335 | 98.796296 | Yes |
| EarthLoc S2 ResNet-50, archived extra | 86.43 | 4,698 | 87.000000 | No |

Three rows match the saved rounding and eight do not. Among the ten article rows, the largest discrepancy from a rounded published percentage is about **0.1141 percentage points** for ResNet-50. The extra EarthLoc row differs by **0.57 points**. Repeating the ResNet-50, EarthLoc, DOFA Base, and DOFA Large fits retained their measured accuracies and selected C values; this is not a single unexplained transient run. ResNet-50 and DOFA Large disagree in accuracy despite exact saved backbone/embedding hashes. The audit therefore does not attribute those differences to changed inputs or tune parameters on test to remove them.

[The committed fresh scoreboard comparison](results/reproduction_audit/neural_scoreboard.csv) records exact accuracies, correct counts, selected C, bootstrap intervals, and hash comparisons for every row. The first standalone fits use their original model-specific training-order protocol, which should not be silently substituted with the later overlap experiment's block normalization and selection grid.

### Parameter counts and compute profiles

All eleven recreated **12-channel, 224x224** synthetic profiles exactly matched the stored structural measurements: backbone/probe parameter counts, feature dimensions, input shape, and backbone/probe/total GFLOPs. Their timing did not reproduce exactly and should not be expected to: for example, OlmoEarth Nano throughput was about **199.7** rather than the recorded **336.8** patches/second on this MIG allocation.

These profiles are not the actual EuroSAT extraction workload. Most extraction runs use **13 channels at 224x224**; OlmoEarth uses **12 channels at 64x64**, omitting physical B10, and the extra archived EarthLoc wrapper further resizes its input to 320x320. The extra input channel changes the adapted input layer's parameter count:

| Backbone | Recorded 12-channel parameters | Actual 13-channel extraction parameters |
|---|---:|---:|
| ResNet-18 | 11,204,736 | 11,207,872 |
| ResNet-50 / MoCo | 23,536,256 | 23,539,392 |
| ConvNeXt-Tiny | 27,833,952 | 27,835,488 |
| ViT-Base/16 | 87,568,128 | 87,764,736 |
| EarthLoc S2 ResNet-50 | 27,690,436 | 27,693,572 |

DOFA and OlmoEarth parameter counts matched their recorded counts without these adapted-layer changes. The conventional neural probes store `10*(embedding_dimension+1)` parameters, whereas the small handcrafted heads use the nine-row reference-class form. The article notes its profiling convention, but the current runner documentation needed to make the profile/extraction distinction explicit; this PR adds that clarification.

### Full representation-overlap study: reproduced exactly

The complete pipeline ran from fresh original and B10-ablated handcrafted features and the fresh five-backbone embeddings:

```bash
output/benchmark-env/bin/python -m experiments.representation_overlap.run prepare
output/benchmark-env/bin/python -m experiments.representation_overlap.run all
```

All six handcrafted feature-archive hashes matched their saved counterparts. The pipeline recomputed **5,148 validation candidates**, locked and evaluated **249 jobs**, and regenerated its reports. **All ten CSV files, totaling 9,610 rows, all three PNG figures, and the generated results README were byte-for-byte identical to the checked-in outputs.** This includes controls and unsuccessful comparisons, not just headline rows:

| Export | Reproduced rows |
|---|---:|
| Selection | 5,148 |
| Classification | 59 |
| Complementarity | 49 |
| Decoding | 190 |
| Feature families | 201 |
| Named features | 3,130 |
| Geometry | 133 |
| Paired reconstruction | 7 |
| PCA references | 203 |
| Per-class comparisons | 490 |

The combined-classifier table therefore reproduces exactly. Asterisks below mark the seven Holm-significant gains across the ten declared primary comparisons:

| Backbone | Embedding alone (%) | +33 features (%) | +377 features (%) |
|---|---:|---:|---:|
| ResNet-50 | 95.07 | 98.02* | 97.89* |
| ConvNeXt-Tiny | 95.15 | 97.85* | 98.17* |
| DOFA Large | 98.20 | 98.43 | 98.78* |
| OlmoEarth v1.2 Nano | 96.81 | 97.70* | 97.76* |
| OlmoEarth v1.2 Base | 98.80 | 98.76 | 98.94 |

The standalone H33/H377 probes reproduce **96.04% / 97.15%**. ResNet-50 +H33 improves by **2.9444 percentage points**, with the article's **[2.33, 3.48]** interval after rounding; ConvNeXt-Tiny +H377 improves by **3.0185 points**, with **[2.44, 3.52]**. DOFA Large +H377's exact gain is **0.574074 points**: the article's **0.58** comes from subtracting the two rounded display accuracies, not rounding the unrounded gain itself. Neither addition establishes a Holm-significant improvement for OlmoEarth Base.

The directional linear-decoding results also reproduce:

| Backbone | H33 -> embedding variance (%) | H377 -> embedding variance (%) | Embedding -> H33 mean R-squared | Largest NDVI region R-squared | Low-NDVI anisotropy R-squared |
|---|---:|---:|---:|---:|---:|
| ResNet-50 | 27.7 | 23.6 | 0.757 | 0.210 | 0.355 |
| ConvNeXt-Tiny | 25.0 | 31.5 | 0.748 | 0.228 | 0.367 |
| DOFA Large | 43.7 | 52.6 | 0.867 | 0.339 | 0.441 |
| OlmoEarth v1.2 Nano | 78.8 | 88.7 | 0.855 | 0.276 | 0.415 |
| OlmoEarth v1.2 Base | 59.5 | 73.2 | 0.901 | 0.385 | 0.452 |

The class-centered Nano values of **47.8% / 69.5%**, null/shuffled controls, B10 ablations, paired comparisons, PCA references, and feature-family outputs are included in the exact CSV match. The unstable ResNet-50 H377 reconstruction remains unstable, including its approximately **[-1.1%, 36.7%]** bootstrap interval; no outlier was removed to improve the result. These are linear predictability and controlled-classification measurements, not mutual-information estimates or evidence that poorly linearly decoded information is absent from a backbone. Their conditional intervals do not include uncertainty from the historical feature-discovery process.

Fresh provenance records correctly refer to this run's inputs/source/environment rather than being copied from the old run. The [fourteen saved/fresh file-hash comparisons](results/reproduction_audit/overlap_hashes.json) are committed with this audit. The complete study finished before the display-band-label fix described below; its source guard correctly rejects reusing that old cache after the source changes. Neither source provenance nor the guard was rewritten to pretend otherwise. A new study must use fresh cache/study/export directories.

To verify the final code rather than merely infer that label changes are harmless, a second `prepare` run after the fix reread all 27,000 original TIFFs into new cache/study/export directories. It completed in **12m 12.486s**, and **all six original/B10-zeroed train/validation/test NPZ archives were byte-identical** to the first run. The [committed comparison](results/reproduction_audit/band_label_feature_check.json) records the command and hashes. This verifies unchanged numerical inputs without pretending that the old source fingerprint is current.

### Neural learning curves

All **98 points** were rerun: seven backbones, two split protocols, and seven fractions. The command covering the complete article figure set is:

```bash
output/benchmark-env/bin/python experiments/torchgeo_bench_eurosat/run.py \
  --models resnet50 convnext_tiny dofa_base dofa_large \
    olmoearth_nano olmoearth_small olmoearth_base \
  --datasets eurosat eurosat-spatial --fractions --download
```

The recovered protocol uses a nested seed-0 permutation of shuffled training embeddings, a 40-value C grid from `10**-6` through `10**4`, train-only fitting, validation selection, and 200 bootstrap resamples. No settings were changed using test outcomes. All **14** backbone-state/embedding-archive pairs, covering seven models on both split protocols, matched the recovered reference hashes exactly.

There are two distinct comparison targets. **All 98 raw curve accuracy/C/confidence-interval tuples exactly match the earlier independent rerun** stored at archival revision `924f0a8`. That does not mean the original article curves match: **23 of 98 plotted accuracies agree at their original reported precision**. The largest discrepancy is **0.444444 percentage points**, or 24 images, for ConvNeXt-Tiny on the 1% spatial split: **84.018519%** freshly measured versus **83.574074%** originally recorded.

| Backbone | Random points matching original / 7 | Maximum random difference (points) | Spatial points matching original / 7 | Maximum spatial difference (points) |
|---|---:|---:|---:|---:|
| ResNet-50 | 3 | 0.129630 | 1 | 0.148148 |
| ConvNeXt-Tiny | 3 | 0.092593 | 0 | 0.444444 |
| DOFA Base | 0 | 0.296296 | 0 | 0.277778 |
| DOFA Large | 2 | 0.203704 | 3 | 0.240741 |
| OlmoEarth v1.2 Nano | 3 | 0.148148 | 1 | 0.259259 |
| OlmoEarth v1.2 Small | 3 | 0.111111 | 1 | 0.111111 |
| OlmoEarth v1.2 Base | 1 | 0.111111 | 2 | 0.148148 |

The article's four random-split, non-Olmo 100% endpoints are special: its renderer appended the rounded, unpermuted full-data scoreboard values with degenerate plotted intervals. They are not the same rows as the fully permuted 100% fraction fits. The audit compares those four article endpoints with the corresponding fresh full-data scoreboard fits and retains the raw curve rows separately. It does not silently substitute whichever endpoint looks closer to the article.

[The committed 98-row comparison](results/reproduction_audit/neural_fractions.csv) preserves both forms, the original targets, selected C values, intervals, sample counts, and input/subset hashes. The PR also restores the 14 compact original curve CSVs and the historical independent-rerun comparison **byte-for-byte** from Git, explicitly labeled as historical references rather than freshly generated results. The original files contain 94 linear rows plus unplotted KNN rows; the four full-scoreboard endpoints supply the other plotted points.

The fresh measurements support the actual Figure 7 alt text: Final33 beats ResNet-50 and ConvNeXt-Tiny at every random fraction and beats DOFA Base at 1%, 2%, 5%, and 10%, while DOFA Large and all three OlmoEarth variants remain above it throughout. The supplied article already states the DOFA Base exception; it is not a newly discovered error in its wording.

At 100% spatial training:

| Model | Fresh test accuracy (%) |
|---|---:|
| ResNet-50 | 88.537037 |
| DOFA Base | 91.777778 |
| Final33 | 92.759259 |
| OlmoEarth v1.2 Nano | 92.907407 |
| ConvNeXt-Tiny | 93.351852 |
| DOFA Large | 94.518519 |
| OlmoEarth v1.2 Base | 97.537037 |
| OlmoEarth v1.2 Small | 97.592593 |

The fresh Final33/Nano gap is **0.148148 points**, satisfying the stated **0.2-point** bound. The original unrounded data's gap was **0.203704**, so the original prose was a rounded approximation to that bound. The fresh DOFA Large/Final33 gap is **1.759259 points**, which rounds to **1.8**, not **1.7**; the original gap was **1.666667**, which does round to 1.7. The other Figure 8 point comparisons also hold: Final33 beats DOFA Base and ResNet-50 at every spatial fraction and trails ConvNeXt-Tiny at 50% and 100%.

Finally, all four light/dark random/spatial figures were regenerated again from **these 98 fresh neural measurements and the 28 fresh handcrafted curve rows**, not merely the archived tables. They are retained locally under `output/reproduction-audit/fresh-neural-curves/`, with a source/data/image-hash manifest. Their expected numerical differences mean they are not byte-identical to the article's original figures.

## Interactive figures and plot regeneration

The supplied `generate_figures.py` and `generate_fraction_plot.py` were copied to `output/reproduction-audit/figures/` before running, because they write next to themselves and would otherwise overwrite the author's supplied assets. The first generator accepts `--experiment-root`; its default expects an absent nested `eurosat-min-params/` checkout. The static-plot script has no argument parser: even `--help` attempts plotting and initially failed on its missing nested historical CSV path.

The interactive generator needs Pillow in addition to the pinned CPU dependencies. Running it in the neural benchmark environment, whose NumPy is **2.5.2**, failed its existing validation-accuracy assertion: ImageStats reached **4,905/5,400 = 90.833333%**, instead of **4,910/5,400**. Running the pinned CPU environment initially failed because Pillow was absent. A separate environment containing `requirements-reproduce.txt` plus **Pillow 12.3.0** resolved both issues:

```bash
output/tooling-env/bin/uv venv --python 3.13 output/reproduction-audit/figure-env
output/tooling-env/bin/uv pip install \
  --python output/reproduction-audit/figure-env/bin/python \
  -r requirements-reproduce.txt pillow==12.3.0
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  output/reproduction-audit/figure-env/bin/python \
  output/reproduction-audit/figures/generate_figures.py \
  --experiment-root "$PWD" --workers 4
```

Before the successful run, the three statistics caches generated by the failed NumPy-2.5.2 run were moved aside so the inputs were freshly derived again. Comparing both sets afterward showed **bit-identical statistics arrays** in all three splits. Thus the differing fitted outcome was not caused by different raw-image statistics in this experiment. The generator's existence-only `.npy` cache has no provenance check; future runs should not reuse it after changing input files or split order.

The pinned run completed in **16.983s**, reproduced ImageStats validation/test counts, and generated 30 gallery pictures plus eight feature-demo pictures. **All 38 WebP image files were byte-identical to the supplied article images.** It did not reproduce the entire supplied JavaScript dataset byte-for-byte:

| Interactive quantity | Supplied article data | Fresh pinned run |
|---|---:|---:|
| ImageStats validation accuracy (%) | 90.925926 | 90.925926 |
| ImageStats `generated_test_accuracy` (%) | 91.018519 | 90.962963 |
| ImageStats `reported_test_accuracy` (%) | 90.962963 | 90.962963 |
| AnnualCrop example orientation entropy | 1.6651 | 1.6669 |
| AnnualCrop example coherence | 0.7641 | 0.7641 |
| Forest example orientation entropy | 2.0155 | 2.0157 |
| AnnualCrop / PermanentCrop example mean NDVI | 0.3748 / 0.3622 | 0.3748 / 0.3622 |

The supplied interactive head therefore has a different recorded generated accuracy from the displayed article baseline: **three additional correct test images**. Its 477 stored coefficient/bias entries and some example logits/probabilities differ from the fresh pinned head. The script permits up to `0.001` absolute test-accuracy drift, so its success alone cannot establish that the embedded classifier equals the reported baseline. The existing figure-data metadata does distinguish generated and reported accuracy; this audit does not erase that distinction or rewrite the author's original source.

Before correcting band labels, there were 478 unequal scalar values under `imagestats`, 160 under `examples`, and 795 scalar/string values under orientation features; the other feature groups agreed exactly. Orientation differences include four encoded bin maps and rounded distribution values, with maximum numeric difference **0.0171**. The quoted example entropies still round to **1.67 / 2.02**, and coherence to **0.76**. These are reproduced rounded caption claims, not bit-identical recovery of every original visualization value.

The freshly generated coarse-scale B03 gradient medians are **368.706** for Industrial, **6.7395** for SeaLake, and **30.713** for Forest. Their ratios are **54.7082** and **12.0049**, supporting the article's approximate "50x" and "10x" descriptions. The similar-mean-NDVI example values also reproduce.

### Incorrect band labels: fixed in this PR

The interactive generator imports `src.data.BAND_NAMES`. That public list still described the old, incorrect channel ordering even though `TIFF_BAND_NAMES` correctly described actual TIFFs. Consequently, the supplied and initially regenerated figures mislabeled channels 8 through 12 as `B08A B09 B10 B11 B12`, rather than physical `B09 B10 B11 B12 B8A`.

The fix makes both public band-name lists describe the physical TIFF order. **The trained numeric feature aliases and every feature calculation are unchanged.** A regression test compares the labels and legacy SWIR indices with the saved checkpoint metadata. Regenerating the isolated article assets after the fix exports the correct labels and retains the reproduced classification counts. Its JavaScript SHA-256 is `3b43aeb71f7a829ce8d5827ed50cc3217675d0d8232c7ea56d362751ae313477`.

The live website's existing assets are not updated by this repository PR. They need regeneration in the website source using the pinned numerical environment and corrected labels.

### Static learning-curve figures

`figures-data.js` contains no neural training curves. Figures 7 and 8 are static plots whose numerical inputs are separate CSVs. Linking the isolated plotting directory to the pinned archival checkout and its three `experiments/imported/` sweep directories allowed the supplied plotting script to run without modifying it:

```bash
ln -s ../historical output/reproduction-audit/figures/eurosat-min-params
ln -s ../historical/experiments/imported/eurosat-train-fractions-20260901 \
  output/reproduction-audit/figures/eurosat-train-fractions-20260901
ln -s ../historical/experiments/imported/eurosat-spatial-train-fractions-20260901 \
  output/reproduction-audit/figures/eurosat-spatial-train-fractions-20260901
ln -s ../historical/experiments/imported/olmoearth-train-fractions-20260901 \
  output/reproduction-audit/figures/olmoearth-train-fractions-20260901
output/benchmark-env/bin/python \
  output/reproduction-audit/figures/generate_fraction_plot.py
```

All four random/spatial light/dark plots rendered at **1548x936** using Matplotlib **3.11.1** and Pillow **12.3.0**. They are not byte- or pixel-identical to the supplied WebPs; a visual comparison shows matching plotted curves with rendering/layout differences. This successfully recovers plotting from the original recorded tables, **not independent neural curve measurements**. The original plot also uses different uncertainty definitions: sample standard deviations across five subsamples for local models, versus conditional test-bootstrap confidence intervals for neural probes.

## Fresh feature-package install: a test portability defect

The documented CPU development install was also exercised in a separate fresh environment:

```bash
output/tooling-env/bin/uv venv --python 3.13 .venv
output/tooling-env/bin/uv pip install --python .venv/bin/python \
  --torch-backend cpu -e '.[io,style,tests]'
.venv/bin/ruff check --no-fix
.venv/bin/ruff format --check
.venv/bin/python -m unittest tests/test_torch_features.py \
  tests/test_feature_cli.py tests/test_torchgeo_bands.py
```

Its unpinned development dependencies selected **PyTorch 2.14.0+cpu**, **NumPy 2.5.3**, and **Ruff 0.16.6**. Installation and formatting checks succeeded, but the original percentile comparison failed on `p90_b10` for a negative-valued synthetic image: PyTorch returned **-991.2716674804688**, versus NumPy's **-991.2703857421875**, an absolute difference of **0.00128173828125 DN**.

The cause is native float32 quantile-probability arithmetic, not the saved classifier or a feature-order error. PyTorch's float32 representation of `0.9` is `0.8999999761581421`; over 4,096 samples its interpolation rank differs from the double-precision mathematical rank. The neighboring pixels in this fixture are -997.73876953125 and -984.802001953125. The probability/rank difference predicts about 0.001263 DN of the discrepancy before output rounding. A relative budget based only on the percentile output's magnitude underestimates this error for values closer to zero.

The fix changes **only** the percentile test's absolute budget to **0.01 DN**, one part per million of this fixture's 10,000-DN input scale, while retaining a `1e-6` relative budget. Feature extraction remains native PyTorch; no NumPy-emulation arithmetic or classifier/result tolerance was introduced. Exact native `torch.quantile` semantics remain covered by separate tests. Afterward, the same test command discovered **27 tests: 24 passed and 3 skipped**; the skips require TorchGeo, which is intentionally absent from that CPU feature-package environment. The raw-TIFF feature checks did run because the fresh dataset was present.

The final targeted CPU command passed all **30** reproduction-contract, table-export, and RESISC45 tests, including the new physical-band-label regression. The fresh development environment passed all **24** native-feature and extraction-CLI tests. The pinned neural environment also passed **25** native-feature/TorchGeo tests, exercising both CPU and CUDA paths with actual downloaded TIFFs and no skips, plus all **7** backbone protocol tests. These are supporting software checks, not substitutes for the full-data numerical runs above.

## Existing-table export is not reproduction

```bash
python -S export_blog_results.py --output output/reproduction-audit/blog-export
```

This succeeded without third-party imports and emitted nine tables, including the current scoreboard, local curves, per-class results, feature order, representation tables, and RESISC45 references. It checks internal consistency of the stored CSVs; it does not download data, train anything, or reproduce the interactive article. In particular, the current local-model export intentionally omits the historical 171- and 279-value rows. Passing this command alone cannot support an "all article results reproduced" claim.

## Changes and remaining limitations

The PR fixes the demonstrated clean-install index-resolution failure, exposes all archived scoreboard model configurations without expanding the default workload, restores the compact neural curve reference tables, corrects public physical-band labels without changing trained feature arithmetic, and repairs the overly strict native-percentile test budget. It also documents the historical-worktree route and the separation between CPU and neural numerical environments. Existing model checkpoints and result values are not changed to make discrepancies disappear.

Fresh compact measurements are committed under [`results/reproduction_audit/`](results/reproduction_audit/): the neural scoreboard and full curve comparison, both local curve tables, historical tiny-head refit results, ablation/optional-baseline evidence, and overlap file hashes. They are separate from the original references. Full logs, environments, downloaded weights, fitted checkpoints, and large arrays remain under ignored `output/`; original images remain under ignored `data/`.

The remaining limitations are real: missing original ablation/CNN/MOSAIKS fits, neural scores that differ despite matched inputs, the archived MOSAIKS export assertion, unverified CNN deployment folding, stale/mislabeled website figure data until regenerated, and the lack of an independent replay or validation of the complete adaptive search. Pretrained backbones were downloaded and frozen as the article prescribes; their external pretraining experiments and contextual claims about unrelated foundation-model sizes were not reproduced here.

**Conclusion:** the core small-model and representation-comparison claims have strong clean-start reproduction evidence. Claiming that every number, embedded classifier, historical experiment, and figure reproduces exactly would be incorrect.
