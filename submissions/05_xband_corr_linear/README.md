# Submission 05 — 390-parameter linear model (adds cross-band correlation)

**Result: 390 parameters, test accuracy 0.9461 (val 0.9372, train-CV 0.9417).**
A **5% parameter cut** from submission 04 (410 params, test 0.9457), achieved by
adding one parameter-free feature family — the **cross-band spatial
correlation** — that moves the honest, de-noised feature-count floor from k=40
down to k=38. Independently re-verified end-to-end from the raw GeoTIFFs by
`eval.py` (val/test match `train.py` exactly).

| submission | params | test | note |
|-----------:|-------:|-----:|------|
| 01 | 1010 | 0.9502 | spectral + gradient texture |
| 02 |  660 | 0.9431 | val-selected |
| 03 |  510 | 0.9413 | + structure-tensor coherence |
| 04 |  410 | 0.9457 | + orientation entropy / histogram / FFT peak |
| **05** | **390** | **0.9461** | **+ cross-band spatial correlation** |

## What changed vs submission 04

Submission 04 used a 273-feature pool (spectral + gradient-magnitude texture +
coherence + orientation entropy/histogram + FFT peak) and selected k=40 (410
params). This submission adds **one more zero-parameter feature family** to the
pool. Because feature extraction is fixed arithmetic on the raw bands (no learned
weights) and only the L1-**selected** k features enter the deployed model, adding
a family to the pool costs nothing in the parameter count (still `10·(k+1)`); it
only gives the selector better features, so it reaches the 0.940 bar at smaller k.

### Cross-band spatial correlation (`xband=True`, 8 features) — the lever
For 8 informative Sentinel-2 band pairs (RED/NIR, GREEN/NIR, BLUE/NIR, SWIR1/NIR,
RED/SWIR1, GREEN/RED, NIR/SWIR2, SWIR1/SWIR2) compute the **Pearson correlation
of the two bands over the 64×64 patch pixels**,
`corr = mean((a−ā)(b−b̄)) / (std_a · std_b)`.

This is the **only** family in the pool that looks at the *joint* spatial
structure of two bands — every other feature (intensity stats, gradient texture,
coherence, orientation) is computed one band at a time. Vegetation couples
RED/NIR spatially very differently from built-up or water, so a handful of these
correlations is a discriminative axis **orthogonal** to everything already in the
pool. The L1 selector duly pulls **3 of the 8** into the top-38 —
(BLUE,NIR), (GREEN,RED), (SWIR1,NIR) — which is exactly why the base 273-pool
cannot reach k=38 (its CV there is ~0.938) but this pool can (0.9417).

### Why this and not the families that failed
The iteration-4 note flagged the next lever as an *orthogonal* family (not more
orientation). Four candidates were built and swept (`experiments/`):

| pool (base273 + …) | floor k | params | CV | val | test |
|--------------------|--------:|-------:|-----:|-----:|-----:|
| base273 (submission 04)      | 40 | 410 | 0.9412 | 0.9406 | 0.9457 |
| + variogram (coarseness)     | — | — | <0.940 at k≤40 | | |
| + GLCM homogeneity           | — | — | <0.940 at k≤40 | | |
| + 2nd-scale orient. entropy  | 38 | 390 | 0.9404 | 0.9396 | 0.9426 |
| **+ cross-band correlation** | **38** | **390** | **0.9417** | 0.9372 | **0.9461** |

Variogram and homogeneity *hurt* (they add noise at the selected k). A 2nd
orientation scale also reaches 390 but weaker on CV and test. Cross-band
correlation wins on the honest CV selector **and** gives the best test.

### Selection (unchanged methodology)
`k` is chosen by **5-fold cross-validation on the train split, averaged over 10
shuffles** (never touches val or test): the smallest k with mean CV ≥ 0.940.
That rule selects **k=38** because CV(k=37)=0.9395 < 0.940 ≤ CV(k=38)=0.9417.
The crossing is not shuffle-luck — re-confirmed at **20 shuffles**
(CV 0.9414, standard error 1×10⁻⁴, so k=37 stays below and k=38 stays above).

## Frontier on the 281-feature pool (selection uses train CV only)

| k  | params | train-CV | val    | test   |
|----|--------|----------|--------|--------|
| 37 | 380    | 0.9395   | 0.9361 | 0.9459 |
| 38 | 390    | 0.9417   | 0.9372 | 0.9461 | ← selected (smallest k with CV ≥ 0.940) |
| 39 | 400    | 0.9424   | 0.9385 | 0.9491 |
| 40 | 410    | 0.9427   | 0.9378 | 0.9480 |

## Honest caveat

Unlike submissions 03–04, the held-out **val** accuracy at the CV-selected k is
**below 0.940** (0.9372), even though both the train-CV selector (0.9417) and the
final test (0.9461) clear it comfortably. This is a mild robustness flag from
selecting the winning pool among several candidates: the cross-band family lifts
train-CV a touch more than it lifts this particular val split. The submission is
still valid — the honest CV rule (which never touches val or test) selects k=38,
and the deployed 390-parameter model genuinely scores **test 0.9461 > 0.94**,
independently recomputed from the raw patches. But the val gap means 390 is a
*thinner* margin than 410; a future iteration wanting a wider cushion could ship
xcorr at k=40 (410 params, test 0.9480, the best test on the frontier) instead.

## Weakest classes (test, from `eval.py`)
Highway 0.857 and PermanentCrop 0.901 still cap accuracy — unchanged from
submission 04. The extra directional/joint features help the average but not
these two confusable crop/road classes, so the next accuracy lever must target
them specifically.

## Reproduce

```bash
# train (selects k by 10-shuffle train-CV, saves model.npz)
python submissions/05_xband_corr_linear/train.py
# independent end-to-end verification from raw GeoTIFFs
python submissions/05_xband_corr_linear/eval.py
```

## Files
- `train.py` — builds the 281-feature parameter-free pool, L1-ranks, selects k by
  10-shuffle 5-fold train-CV (≥0.940), folds the standardiser into the linear
  weights, saves `model.npz`.
- `eval.py` — reloads `model.npz`, recomputes every feature from the raw patches
  (bypassing all caches) using the stored config, reports val/test + per-class.
- `model.npz` — `W` (10×38), `b` (10), `feature_idx` (38 selected raw columns),
  and the feature config. Deployed parameter count = `W.size + b.size = 390`.
