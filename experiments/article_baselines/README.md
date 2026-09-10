# Supporting baselines

These scripts cover the smaller baseline comparisons and the “Other approaches” numbers in the final article. `archive_manifest.csv` records the quoted values and their sources. New measurements are kept separately in `measured/`.

## Run

Use the CPU environment from `requirements-reproduce.txt`, from the repository root:

```bash
.venv-reproduce/bin/python -m experiments.article_baselines.run \
  --output output/article-baselines/new-run --download --check
```

This runs EuroSAT and RESISC45. It writes `metrics.csv`, the historical claim list, and separate `eurosat/` and `resisc45/` directories containing predictions, fitted heads, feature names, validation sweeps, and input/environment hashes. Use a new directory for each run; nonempty output directories are rejected.

To run just one dataset:

```bash
.venv-reproduce/bin/python -m experiments.article_baselines.run \
  --dataset eurosat --output output/article-baselines/eurosat-new --check
.venv-reproduce/bin/python -m experiments.article_baselines.run \
  --dataset resisc45 --output output/article-baselines/resisc45-new --check
```

The commands need NumPy, SciPy, scikit-learn, and rasterio. They do not need PyTorch, TorchGeo, Pillow, or old image/feature caches. `--workers 4` enables parallel image reads; numerical fitting uses one BLAS thread.

## What the reruns reproduce

| Dataset / features | C | Correct / test images | New accuracy | Article |
|---|---:|---:|---:|---:|
| EuroSAT means | 100 | 4,123 / 5,400 | 76.35% | 74.8% |
| EuroSAT mean + std | 200 | 4,820 / 5,400 | 89.26% | 87.8% |
| EuroSAT full statistics | 300 | 4,912 / 5,400 | 90.96% | 90.96% |
| RESISC45 full statistics | 30 | 2,305 / 6,300 | 36.59% | 36.63% |
| RESISC45 archived 33 features | 30 | 3,721 / 6,300 | 59.06% | 59.06% |

The first two article numbers survive only as rounded research notes, without the original C or coefficients. The RESISC45 ImageStats quote likewise has no saved C sweep or checkpoint. Their new runs select C on validation and are labelled `new_reference_missing_historical_fit_metadata`, not exact reproductions.

The full EuroSAT statistics grid was saved, and selecting its first validation maximum again gives C=300 and 90.96%. The RESISC45 33-feature result was also recovered: its 147-column RGB pool, ordered subset, C=30, and preprocessing are preserved at commit `68480cb7aca0e73289da2ac36ec95faeab020b2d`. `archived/resisc45_33_full.csv` copies that commit's full-training result. `rgb.py` contains the required RGB feature recipe without the later research searches.

`--check` means agreement with the **new, labelled reference measurements** in `measured/metrics.csv`. It does not mean the missing historical fits were recovered. No C, subset, or seed was changed to make a test score match the article.

The measured environment is Python 3.13.13, NumPy 2.4.4, SciPy 1.18.1, scikit-learn 1.9.0, and rasterio 1.5.1/GDAL 3.12.4. The raw-image runs took about 80 seconds for EuroSAT and 169 seconds for RESISC45 with four readers.

## Data and preprocessing

EuroSAT uses the same original TIFFs under `data/EuroSAT/` as `reproduce.py`. RESISC45 uses:

```text
data/RESISC45/NWPU-RESISC45/<class>/<class>_<id>.jpg
data/RESISC45/resisc45-train.txt
data/RESISC45/resisc45-val.txt
data/RESISC45/resisc45-test.txt
```

`--download` obtains missing official files from pinned dataset revisions and checks their SHA-256 hashes. Existing files are never replaced; an existing file with a wrong checksum is an error. Archive extraction rejects unsafe paths, symlinks, and duplicate members.

Every run checks split sizes, disjoint membership, labels, image shapes and types, and records a hash of the decoded pixels in split order. That hash identifies the inputs used; it does not assert that a locally modified image was compared with the original archive.

RESISC45 JPEGs start at 256x256. We read them with `rasterio.read(out_shape=(3, 64, 64), resampling=Resampling.bilinear)`, retaining uint8 values before converting to float32. Pillow and torchvision resize operations are not interchangeable with that saved recipe.

The RESISC45 archive and split files are pinned to Hugging Face revision `883edc0eee77b2c84225472f10f126e3ed83fa6e` of `isaaccorley/resisc45`. `download.py` records the URLs and checksums. Downloads require an explicit flag, and output is restricted to new directories under the repository's ignored `output/`.

## Repeat the RESISC45 feature selection

The default replays the saved subset and C. To repeat the selection instead:

```bash
.venv-reproduce/bin/python -m experiments.article_baselines.run \
  --dataset resisc45 --reselect-resisc33 \
  --output output/article-baselines/resisc45-reselection
```

This ranks the 147 columns on train using `l1_rank(C=.05, random_state=0)`, keeps 33, and selects the smallest C within one binomial standard error of the best validation accuracy. The historical grid is `.01, .03, .1, .3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000`. These runs are labelled `new_selection`; a different L1 solver can change the feature order. Test is not used for selection.

## Optional CNN and MOSAIKS runs

These need the torch experiment environment and can use several GB of RAM and disk space:

```bash
output/benchmark-env/bin/python experiments/article_baselines/other.py \
  --method tiny-cnn --epochs 60 --gpu 0 \
  --output-dir output/article-baselines/tiny-cnn-rerun
output/benchmark-env/bin/python experiments/article_baselines/other.py \
  --method mosaiks --device cuda:0 \
  --output-dir output/article-baselines/mosaiks-rerun
```

The inputs are new named arrays decoded from original TIFFs into the run directory, not the old `data/cache` files.

The CNN adapter uses `experiments/conv_gap.py` with all 13 bands, 16 filters, seed 0, and an explicit 60-epoch budget. Its reported deployed count is `16*(9*13+11)+10=2058`. The original 89.2% note says 60–80 epochs but does not identify the exact run, weights, or seed. The source reports test accuracy when a new best validation epoch is found; validation, not test, chooses the epoch. The new training log is not an exact replay of a saved checkpoint.

The MOSAIKS adapter uses TorchGeo Gaussian random convolutional features: seed 1, 2,048 filters producing 4,096 positive/negative ReLU features, train-only L1 top 512, C=1, and a `9*(512+1)=4617`-value head. That is the article's head-only count. The new checkpoint also stores 13 input means and 13 input standard deviations before the nonlinear random features, for **4,643 learned values in total**; those cannot be folded into the final linear head.

The old roughly-95% note has no saved filter subset, fitted head, or exact score at 512 features. The original floor-search script included test accuracy in its gate; this adapter deliberately does not repeat that practice. It fixes 512 features and C=1 before scoring test once, and records a new result.

These optional configurations were exercised on small synthetic inputs; the measured CSVs do not claim full-data CNN or MOSAIKS reruns.

## Tests

```bash
.venv-reproduce/bin/python -m unittest discover -s tests -p 'test_article_baselines.py'
```

The tests cover feature order, resize arithmetic, split checks, protected outputs, validation-only selection, and the historical/new-result distinction.
