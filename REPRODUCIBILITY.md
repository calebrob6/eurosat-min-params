# Reproduction scope and article audit

## Reference article and environment

The target is `posts/eurosat-min-params/index.qmd` in the separate GeoSpatial ML blog repository at revision `dd70b6ac0815abc825bd22d558611ca924cc4d0d`, titled **Solving EuroSAT with as Few Parameters as Possible**. The local `geospatialml/` checkout is intentionally ignored; a public clone of this repository does not need it. `blog_post.txt` contained a different article and is not used.

The CPU reference environment is Python 3.13.13, NumPy 2.4.4, SciPy 1.18.1, scikit-learn 1.9.0, and rasterio 1.5.1. Install `requirements-reproduce.txt` in a separate environment. `reproduce.py` fixes OpenMP/OpenBLAS/MKL thread counts to one before importing numerical libraries. Unpinned versions or different BLAS stopping behavior can move a small number of predictions in refits; saved model evaluation is distinct from reproducing a coefficient fit.

`reproduce.py --download --fractions --check` bypasses all historical NumPy image and feature caches. It reads the 27,000 TIFFs in fixed split-file order and recomputes all measurements in bounded batches. It preserves checked-in artifacts and emits:

| Output | Meaning |
|---|---|
| `models.csv` | Validation/test integer counts for the three saved heads and refitted ImageStats |
| `per_class.csv` | Per-class validation/test accuracy |
| `features_306.csv` | Exact 33-column order, historical pool indices, corrected feature names |
| `imagestats_c_sweep.csv` | Train-only fits, validation-selected C |
| `frontier_fractions.csv`, `imagestats_fractions.csv` | Both protocols, all fractions, all five seed scores, mean and sample SD |
| `environment.json` | Runtime versions, fixed data revision, split checksums |
| `model_306.npz` (with `--refit-306`) | Train-only refit of the frozen 33-feature model |

The 171/279/306 checkpoints are not picked using new test results. The new 306 checkpoint is a fixed-subset, C=3 refit of the historical experiment, not a rerun of the feature search. All headline counts and the five-seed curves are reproduced in the reference environment. The curves retain the original float32 full-logit scoring order; reparameterization is algebraically identical but finite-precision subtraction can move an individual near-tie.

## Physical band names versus historical feature aliases

**The historical code used incorrect physical labels for the last five TIFF channels. We preserve the numerical indices to reproduce the article's scores, not the incorrect labels.**

TorchGeo's EuroSAT TIFF order is:

```text
0:B01 1:B02 2:B03 3:B04 4:B05 5:B06 6:B07
7:B08 8:B09 9:B10 10:B11 11:B12 12:B8A
```

Historical `B_SWIR1=11` and `B_SWIR2=12` therefore select **B12 and B8A**, not B11 and B12. NDVI (B08/B04) and NDWI (B03/B08) have the stated interpretation. The feature aliases `ndbi`, `ndmi`, `nbr`, and `bsi` must be interpreted by their code: they use channels 11/12 as above. In particular, historical `nbr` is `(B08-B8A)/(B08+B8A+epsilon)`, not the conventional normalized burn ratio. Silently changing these indices invalidates the existing weights and changes the experimental question.

Percentile values were always emitted in percentile-major order, but early feature-name strings were generated in band-major order. The names are now corrected **without reordering any values**. The final eight percentile features are:

```text
p75_b10  p10_b11  p90_b8  p10_b3
p50_b11  p75_b3   p75_b0  p10_b6
```

These are B11/B12/B09/B04/B01/B07 measurements, not the blue/green/NIR selection described in the article's family-example column. The counts (eight percentiles, 33 features, 306 learned values) and accuracies remain correct. The release leaves the external blog checkout unchanged; these physical-label descriptions should be corrected in the article before publication.

## Limits of the claims

- Feature extraction costs compute and feature-selection work even though it has no learned scalar weights under the game's accounting.
- The spatial split reuses the same image universe as the feature-design experiments. 4,362 spatial-test examples previously occurred in random train/validation. Treat the spatial curve as a stress test of a frozen design, not untouched geographic generalization.
- Full-training feature selection and regularization selection precede all fraction fits. These are label-efficiency curves for a preselected representation, not end-to-end feature discovery under each label budget.
- The blog's historical 74.8% means-only and 87.8% means-plus-std figures are recorded in `ideas/2026-07-01_0100_initial_landscape.md`. Their original coefficient artifacts and full hyperparameter sweep were not saved. They are not part of the exact `--check` target and should not be described as fresh reproduced ablations.
- Frozen-backbone scores require separately downloaded pretrained weights and TorchGeo-bench. They are not recomputed by the CPU-only handcrafted-model command. See `experiments/torchgeo_bench_eurosat/` for their distinct provenance and protocol.

## Public-release boundaries

The complete RESISC45 state at commit `68480cb` is preserved on the remote `resisc45` branch. RESISC45 files are removed from the default release tree, not erased from Git history. Repository visibility is not changed by this preparation. MIT covers this code, not EuroSAT data, pretrained weights, or the separately maintained blog.

## External benchmark evidence still required

`results/article_claims.csv` distinguishes fresh raw-data reproduction from archived external evidence and article-only claims. It is **not** a new benchmark result table. Parameter strings on external rows are article quotations, not measurements made by the release runner.

The available `experiments/torchgeo_bench_eurosat/` directory contains only:

- `dofa_large.csv`: 98.33% on the full random EuroSAT split; matches the current article.
- `imagestats.csv`: 90.65% from the separate TorchGeo-bench ImageStats implementation, **not** the locally reproduced 90.96% baseline.
- `olmoearth_large.csv`: 99.02% for **OlmoEarth v1 Large**, **not** the article's v1.2 Nano/Small/Base rows.
- `local_models_bootstrap.csv` and the older benchmark README.

Original run outputs for EarthLoc, MoCo, timm ResNet/ViT/ConvNeXt, DOFA Base, and OlmoEarth v1.2 Nano/Small/Base are not in this checkout. Nor are the backbone learning-curve directories read by the article's `generate_fraction_plot.py`: `eurosat-train-fractions-20260901/`, `eurosat-spatial-train-fractions-20260901/`, and `olmoearth-train-fractions-20260901/`. The four DOFA/ResNet/ConvNeXt 100% random points are hardcoded by that plotting script.

Consequently, the local 306-value/ImageStats curves in figures 7 and 8 reproduce, but the complete backbone-comparison figures and the other scoreboard rows cannot yet be claimed reproducible from this repository. Recover the original CSVs plus benchmark/config/weight revisions, or rerun and revise those external comparisons before asserting full-article reproducibility. Do not substitute the older OlmoEarth v1 Large CSV for a v1.2 result.
