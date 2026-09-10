# Reproducing the article

The reference is the [final GeoSpatial ML draft](https://github.com/isaaccorley/geospatialml/blob/251513f694d4cd27a067aaed7af386212ede812d/posts/eurosat-min-params/index.qmd). Its source is kept in a separate repository; none of the commands below requires that checkout.

There are two different tasks here: exporting the original measurements and running the experiments again. The original uploaded backbone scores are still the source for the article's first comparison table. Fresh scores and their differences are saved separately.

## Saved tables

```bash
python export_blog_results.py
```

This command uses only the Python standard library. It writes:

| File in `output/blog-results/` | Article content |
|---|---|
| `scoreboard.csv` | The 14-model comparison |
| `training_fractions.csv` | All 126 points behind the random and spatial learning curves |
| `features_33.csv` | The ordered list of 33 measurements |
| `class_accuracy.csv` | Validation and test accuracy by class for the local models |
| `combined_features.csv` | Backbone-only and combined-feature classifiers, gains, intervals, and adjusted p-values |
| `explained_variance.csv` | Embedding variance predicted by 33 and 377 features |
| `decoded_features.csv` | Mean reverse-decoding R-squared and the two NDVI measurements |
| `class_centered_variance.csv` | Decoding after removing training class means |
| `figure_examples.csv`, `figure_gradient_medians.csv` | Quoted orientation examples and class gradient summaries |
| `historical_claims.csv` | Supporting numerical claims, their sources, and missing original settings |
| `supporting_results.csv`, `archived_resisc33.csv` | New raw-image baseline runs, kept separate from the recovered RESISC45 row |

`--include-archive` includes the EarthLoc row removed from the final draft. The original imported CSVs remain in `experiments/imported/`; exporting a table never trains a model.

## CPU environment

The local reference results use Python 3.13.13, NumPy 2.4.4, SciPy 1.18.1, scikit-learn 1.9.0, and rasterio 1.5.1.

```bash
python3.13 -m venv .venv-reproduce
.venv-reproduce/bin/python -m pip install -r requirements-reproduce.txt
.venv-reproduce/bin/python reproduce.py --download --fractions --check
```

This streams all 27,000 original TIFFs in split-file order, recomputes the features, evaluates the saved 171/279/306-value heads, refits ImageStats, and reruns both five-seed learning curves. It does not read historical image or feature caches. `--check` requires the local counts and requested curves to match the saved results.

Add `--refit-306 --output output/refit` to refit the fixed 33-feature classifier at C=3. The new checkpoint is `output/refit/model_306.npz`, not the checked-in submission. This does not repeat the adaptive feature search.

Numerical libraries and solver stopping behavior matter when refitting coefficients. The supported reproduction path retains the original NumPy/SciPy feature arithmetic. The PyTorch package is for reusing the measurements; it is not silently substituted into the historical experiments.

## CUDA environment

The frozen backbones and combined classifiers use a separate environment:

```bash
git clone https://github.com/torchgeo/torchgeo-bench.git output/torchgeo-bench
git -C output/torchgeo-bench checkout 9c8e4afab46675d7279c88828dfcbf0ca99b3a07
python3.13 -m venv output/benchmark-env
output/benchmark-env/bin/python -m pip install \
  -r experiments/torchgeo_bench_eurosat/requirements.txt \
  -c experiments/torchgeo_bench_eurosat/reproduced/environment-freeze.txt
```

Use an unmodified checkout at that revision. The scripts record resolved model settings, weights, package versions, physical bands, sample filenames, and file hashes. They download pretrained weights through the pinned model wrappers. The original resolved weight revisions are in `experiments/torchgeo_bench_eurosat/reproduced/hf_weight_revisions.json`; if upstream weights change, their recorded hashes expose the change.

## Run the experiments

With both environments installed:

```bash
python reproduce_all.py --download
```

The driver runs each job with the right Python executable and saves new files under `output/reproduce-all/`. Run `python reproduce_all.py --dry-run` to see the commands first. Supply `--cpu-python` or `--benchmark-python` to use environments at other paths.

Choose only the work you need:

```bash
python reproduce_all.py --steps local
python reproduce_all.py --steps backbones --download
python reproduce_all.py --steps overlap --download
python reproduce_all.py --steps baselines figures --download
```

Add `--exploratory` to also train the recovered CNN and MOSAIKS configurations. They are new experiments, not exact reproductions of the old notes, and need several additional GB of disk/RAM and a CUDA GPU.

Use a different `--output output/name` for a different command set or changed source files. Repeating a command skips completed stages after checking their saved file hashes. Some interrupted stages, particularly the supporting baselines, reject partial output directories and require a new output location rather than overwriting files. New results never replace the saved article tables or models.

### Frozen backbones

The backbone jobs run the archived scoreboard and all learning-curve configurations. Their output includes one extra historical EarthLoc row; the final article export omits it. The original runs were not completely environment-locked, so a fresh run need not be bit-identical. The existing independent reproduction differs by at most six test images in full-data accuracy and 24 images at an individual learning-curve point.

The original backbone curves use seed-0 nested, unstratified subsets of shuffled embeddings. Those subsets differ between backbones because model initialization consumes random state before the training loader shuffles. This protocol is preserved for reproduction, not recommended for a new paired comparison. The handcrafted curves instead use five stratified draws.

`experiments/torchgeo_bench_eurosat/compare.py` writes every original-versus-fresh difference. The [benchmark guide](experiments/torchgeo_bench_eurosat/README.md) gives the individual commands and batch sizes.

### Combining and comparing representations

The overlap job starts from newly extracted embeddings and their matching metadata, not a pre-existing local cache. With `--steps overlap`, extraction uses `--extract-only`, so it does not fit or score a classifier before the comparison study.

The later steps prepare the handcrafted features from raw TIFFs, select settings using train and validation, lock them, evaluate test, and export the report. Each job uses the same filenames in the same order. Both original and B10-disabled OlmoEarth comparisons are included.

`output/reproduce-all/overlap-results/` contains the new tables and figures. `overlap-comparison.csv` compares them with `experiments/representation_overlap/results/`. The default run reports differences rather than hiding them. Add `--check` to require local results and the compared overlap measurements to match; `compare_overlap.py --atol` exposes the numerical tolerance for the latter.

All maps and classifiers are fit on train only. Validation chooses ridge strength, logistic-regression C, and the relative weight of the handcrafted block. Test is not used to choose a model. The method definitions and individual commands are in the [representation study guide](experiments/representation_overlap/README.md).

### Supporting results

`.venv-reproduce/bin/python -m experiments.article_baselines.run` reruns the supporting baselines from TIFFs and JPEGs, including RESISC45. Its `--check` compares against the explicitly labelled new measurements, not an unrecovered original fit. The CPU environment is used for these fits too.

`python experiments/figure_statistics.py` recomputes the 120 training patches per class used for the distribution examples and the orientation examples quoted in Figure 4. It reads TIFFs, not the blog's cached JavaScript. The browser rendering and interactive controls remain in the pinned blog source.

The means-only 74.8% and means-plus-standard-deviation 87.8% figures were originally rounded research notes without saved fitting settings. The new validation-tuned fits reach 76.35% and 89.26%, respectively. The RESISC45 ImageStats quote of 36.63% likewise lacks an original sweep or checkpoint; the new run reaches 36.59%. These three historical quotes cannot currently be reproduced exactly from the available records.

The archived RESISC45 33-feature recipe and C=30 were recovered, and its raw-JPEG refit reproduces 59.06% with 1,496 stored values. JPEGs are resized to uint8 64x64 using rasterio's bilinear resampling, as in the original experiment, not Pillow or torchvision.

The tiny-CNN and MOSAIKS results also lack complete original fitted artifacts. Safe, opt-in commands rerun their recovered configurations into new output directories; they do not assert exact recovery of 89.2% or a particular approximately-95% score. See [the supporting result records](experiments/article_baselines/README.md) for the original notes, pinned RESISC45 source, and the separate fresh measurements.

The article's MOSAIKS count is for the head alone: 4,617 values. The new adapter also stores 26 input-normalization values before the nonlinear features and reports 4,643 learned values in total.

## Band order and feature names

The physical TIFF order is:

```text
0:B01 1:B02 2:B03 3:B04 4:B05 5:B06 6:B07
7:B08 8:B09 9:B10 10:B11 11:B12 12:B8A
```

The historical code's `B_SWIR1=11` and `B_SWIR2=12` select B12 and B8A, not B11 and B12. Both the reference implementation and PyTorch package preserve those numeric choices. NDVI uses B08/B04 and NDWI uses B03/B08 as usual. The older `ndbi`, `ndmi`, `nbr`, and `bsi` names must be read with the historical definitions; in particular, `nbr` is `(B08-B8A)/(B08+B8A+epsilon)`, not the usual normalized burn ratio.

Early percentile names were also generated in the wrong order. The current names were corrected without changing values or columns. The final eight percentile features are:

```text
p75_b10  p10_b11  p90_b8  p10_b3
p50_b11  p75_b3   p75_b0  p10_b6
```

These use B11/B12/B09/B04/B01/B07, not the band list in the article's feature-family examples. `results/eurosat_306_features.csv` is the exact ordered recipe. Changing band indices to match a name would change the measurements and invalidate the saved models.

## Limits and article discrepancies

The 33 features use 32 columns from the 377-feature pool plus a separately added low-NDVI region-shape measurement. The larger pool is not a strict superset.

The spatial split reuses the same 27,000 images used during feature design; 4,362 spatial-test patches were in random train or validation. The curves test a frozen feature choice, not an untouched feature-search procedure. All label-budget curves likewise keep the full-data feature choice fixed.

The article's ViT parameter footnote understates the cost of the thirteenth input channel: it adds 196,608 values, and the loaded 13-band backbone has 87,764,736 values. Original rounded parameter descriptions are preserved in the table export rather than silently rewritten.

The text's 0.58-point DOFA gain subtracts the displayed rounded accuracies, 98.78 minus 98.20. The exact paired difference is 31/5,400, or 0.574074 points, which rounds to 0.57. The exported paired results retain the exact calculation.

Fixed feature extraction has no learned weights, but it still takes code, compute, and feature-selection work. A failed linear decoder does not prove that information is absent from an embedding, and high explained variance does not establish preservation of every useful decision.
