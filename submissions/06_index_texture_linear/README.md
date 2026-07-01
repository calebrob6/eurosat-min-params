# Submission 06 — 350-parameter linear model (index-map spatial texture)

**Result:** test accuracy **0.9437**, val **0.9435**, honest train-CV **0.9445**,
**350 parameters** (k=34 features, 10 classes → 10×(34+1)=350). Independently
verified end-to-end from raw GeoTIFF patches by `eval.py`.

This is a **10% parameter cut from submission 05** (390 params, test 0.9461)
while staying comfortably above the 94% test bar — and, unlike submission 05,
val *and* test *and* CV all clear 0.940 (a wider, honest margin).

## What changed vs submission 05

Submission 05 reached the honest CV≥0.940 floor at k=38 (390 params) on a
281-feature parameter-free pool. This submission adds **one more zero-parameter
feature family** that moves that floor down to **k=34 (350 params)**:

**Index-map spatial texture** (`index_texture=True`, 24 features). Every *other*
index feature in the pool is a per-band **mean** ratio — one scalar NDVI/NDWI/…
per patch — which throws away all within-patch spatial structure. This family
instead computes the **per-pixel index map** (e.g. `NDVI(x, y)`) for six indices
(NDVI, NDWI, NDBI, NDMI, NBR, BSI) and summarises how uniform vs mixed it is
across the 64×64 patch:

- spatial **std** of the map,
- gradient-magnitude **mean** and **std** of the map,
- robust **spread** (p90 − p10).

6 indices × 4 statistics = **24 features, zero learned parameters**.

### Why it works (physical intuition)

A managed **PermanentCrop** patch is spatially near-uniform in NDVI (one crop),
whereas **HerbaceousVegetation** / mixed fields vary pixel-to-pixel; a **Highway**
is a sharp low-NDVI paved streak crossing a vegetated background — a large NDVI
*gradient* at modest std. None of that is visible to a *mean* index. The axis is
orthogonal to every per-band intensity/gradient/coherence/orientation statistic
and to the cross-band correlation family (which measures whether two bands
co-vary, not how heterogeneous one index is).

It is the **single strongest parameter-free family found so far**: on its own it
drops the honest floor 390 → 350, and the L1 selector pulls **9 of its 24
features into the top-34** (the gradient-magnitude and gradient-std of the
NBR/NDWI/NDVI/BSI/NDMI maps dominate; no `spread` feature is selected). That is
why the base 281-feature pool cannot reach k=34 but this 305-feature pool can.

## Selection rule (honest, never touches val/test)

Smallest k whose mean **10-shuffle 5-fold train-CV** accuracy ≥ 0.940:

| k | params | CV(10) | CV(20)±SE | val | test |
|---|-------:|-------:|----------:|----:|----:|
| 32 | 330 | 0.9381 | 0.9381 ± 0.0002 | 0.9394 | 0.9411 |
| 33 | 340 | 0.9395 | 0.9394 ± 0.0001 | 0.9398 | 0.9426 |
| 34 | 350 | 0.9445 | 0.9443 ± 0.0001 | 0.9435 | 0.9437 | ← floor |
| 35 | 360 | 0.9454 | 0.9450 ± 0.0001 | 0.9463 | 0.9467 |
| 36 | 370 | 0.9460 | — | 0.9474 | 0.9474 |

The k=33→34 jump is a sharp, seed-stable crossing: the 20-shuffle standard error
is ~1e-4, so k=33 is genuinely below 0.940 and k=34 genuinely above (by ~40 SE).
This is not seed-selection luck.

## Per-class test accuracy (from `eval.py`)

```
Highway               0.861   <- still the cap
PermanentCrop         0.892
Pasture               0.904
Industrial            0.930
HerbaceousVegetation  0.935
AnnualCrop            0.953
River                 0.958
Residential           0.986
Forest                0.992
SeaLake               0.993
```

Highway and PermanentCrop remain the weakest classes (unchanged identity since
submission 03). Every recent gain is on the average, not on these two.

## Files

- `train.py` — builds the 305-feature pool (submission-05 pool + index-texture),
  L1-ranks it, selects k by the 10-shuffle train-CV ≥ 0.940 rule, folds the
  standardiser into the weights, saves `model.npz`. Run:
  `OMP_NUM_THREADS=1 python submissions/06_index_texture_linear/train.py`
- `eval.py` — loads `model.npz`, **recomputes all features from raw patches**
  (bypassing every cache) with the stored config, reports val/test + per-class.
- `model.npz` — `W` (10×34), `b` (10), `feature_idx` (34), plus the feature
  config and recorded metrics. Deployed model is a single affine map
  `logits = x[:, feature_idx] @ W.T + b`; parameter count = `W.size + b.size = 350`.

## Reproduce

```bash
OMP_NUM_THREADS=1 python submissions/06_index_texture_linear/train.py
OMP_NUM_THREADS=4 python submissions/06_index_texture_linear/eval.py
```
