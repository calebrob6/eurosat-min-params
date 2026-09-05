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

## External benchmark reproduction

The original backbone experiment outputs were subsequently supplied under `experiments/imported/`. All 11 full-data article backbone rows and 98 learning-curve points have now been rerun using a fresh, unmodified TorchGeo-bench checkout at `9c8e4afab46675d7279c88828dfcbf0ca99b3a07`. See [`experiments/torchgeo_bench_eurosat/README.md`](experiments/torchgeo_bench_eurosat/README.md) for installation and run commands.

The fresh GPU environment is separate from the pinned CPU-only environment above. JSON manifests record resolved model configs, bands, backbone-state hashes, sample order, runtime versions, and embedding hashes. The original and fresh scores remain in separate CSVs; `scoreboard_comparison.csv` and `fraction_comparison.csv` expose every difference rather than replacing the article's results.

The historical backbone fraction runner used a seed-0 random permutation of **shuffled training embeddings**, taking progressively larger prefixes. These are nested **unstratified** subsets, unlike the five-seed stratified handcrafted/ImageStats curves. The new runner matches this protocol; an execution of the untouched imported runner with ResNet-50 matched all six new linear scores and selected Cs exactly.

The backbone subsets are also model-dependent: initialization consumes Torch RNG state before the training loader shuffles, so the same nominal seed does not give each backbone identical training examples. The reproduced 1% ResNet-50 and ConvNeXt-Tiny subsets share two of 162 examples. This affects interpretation of low-data comparisons and is explicitly retained for historical reproduction.

Scores reproduce closely, not all bit-for-bit: maximum full-data difference is six of 5,400 images (0.1111 percentage points). ConvNeXt-Tiny, OlmoEarth v1.2 Nano, and OlmoEarth v1.2 Base match the original full-data scores exactly. The maximum curve difference is 0.4444 percentage points at 1% spatial data for ConvNeXt-Tiny. All 98 new curve scores fall within the original bootstrap intervals, which is descriptive agreement rather than proof of statistical equivalence.

`results/article_claims.csv` points to the recovered source evidence and comparison. The older top-level `olmoearth_large.csv` remains a different v1 Large experiment (99.02%), not a substitute for the v1.2 models. The top-level TorchGeo-bench ImageStats result (90.65%) is likewise distinct from the locally reproduced 90.96% baseline.

Actual loaded 13-band backbone counts are saved in the fresh CSVs. The article's ViT parameter footnote needs a correction: adding the thirteenth patch-embedding channel costs **196,608 values**, not a few thousand. The fresh ViT count is **87,764,736**, compared with the rounded 12-band profiling figure of 87.57M.
