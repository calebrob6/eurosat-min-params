# Submission 04 — 410-parameter linear model (adds orientation entropy)

**Result: 410 parameters, test accuracy 0.9457 (val 0.9406, train-CV 0.9411).**
A **20% parameter cut** from submission 03 (510 params, test 0.9413), achieved by
adding parameter-free *orientation* features that move the honest, de-noised
feature-count selection down from k=50 to k=40. Independently re-verified
end-to-end from the raw GeoTIFFs by `eval.py`.

| submission | params | test | note |
|-----------:|-------:|-----:|------|
| 01 | 1010 | 0.9502 | spectral + gradient texture |
| 02 |  660 | 0.9431 | val-selected |
| 03 |  510 | 0.9413 | + structure-tensor coherence |
| **04** | **410** | **0.9457** | **+ orientation entropy / histogram / FFT peak** |

## What changed vs submission 03

Submission 03 used a 195-feature pool (spectral + gradient-magnitude texture +
structure-tensor coherence) and selected k=50 (510 params). This submission adds
**three more zero-parameter feature families** to the pool. Because feature
extraction is fixed arithmetic on the raw bands (no learned weights) and only the
L1-**selected** k features enter the deployed model, adding families to the pool
costs nothing in the parameter count (still `10·(k+1)`); it only gives the
selector better features to choose from, so it reaches the 0.940 bar at a
smaller k.

### 1. Gradient-orientation entropy (`orient_entropy_bins=8`, 13 features) — the lever
For each band, build the magnitude-weighted, unsigned (mod-π) gradient
orientation histogram and take its **Shannon entropy**. Low entropy = one
dominant edge direction (roads, crop rows); high entropy = isotropic texture
(forest, water).

This is a *different* statistic from coherence, even though both measure
directionality. Coherence `√((Sxx−Syy)²+4·Sxy²)/(Sxx+Syy)` is a second-moment
scalar that a single strong edge saturates toward 1; the entropy sees the whole
orientation *distribution* and separates one-direction from two-direction from
uniform. On its own this one 13-feature family already drops the floor from k=50
(510) to k=45 (460) and gives the highest test accuracy of any single family
tried (0.9487).

### 2. Orientation histogram (`orient_hist_bins=4`, 52 features) + FFT peak (`spectral_peak=True`, 13)
The per-band orientation histogram (of which the entropy above is a scalar
summary) exposes *which* direction dominates; the FFT mid-frequency peak/mean
flags periodic crop-row texture. Adding both to the pool alongside the entropy
lets L1 selection reach 0.940 at **k=40 (410 params)** instead of k=45.

### Selection (unchanged methodology)
`k` is chosen by **5-fold cross-validation on the train split, averaged over 5
shuffles** (never touches val or test): the smallest k with mean CV ≥ 0.940.
That rule selects k=40. The pick is not shuffle-luck — confirmed at **10
shuffles** (CV 0.9411→0.9412), and it independently clears the bar on **val**
(0.9406) as well.

## Frontier on the 273-feature pool (selection uses train CV only)

| k  | params | train-CV | val    | test   |
|----|--------|----------|--------|--------|
| 35 | 360    | 0.9340   | 0.9331 | 0.9380 |
| 40 | 410    | 0.9411   | 0.9406 | 0.9457 | ← selected (smallest k with CV ≥ 0.940) |

At k=35 the CV (0.9340) is clearly short of 0.940, so 410 is the honest floor for
this pool. See `ideas/2026-07-01_0228_orientation_entropy_410.md` for the full
per-family ablation (five families tried; only orientation-based ones help — a
3rd coherence scale adds nothing) and the combo table.

## Files
- `train.py` — builds the 273-feature pool via
  `patch_features(coherence_scales=2, orient_entropy_bins=8, orient_hist_bins=4,
  spectral_peak=True)`, ranks by L1, selects k by de-noised train-CV, folds the
  StandardScaler into the linear weights, saves `model.npz`.
- `eval.py` — **independent** check: recomputes every feature from the raw
  patches (bypassing all caches) using the config stored in the checkpoint, then
  reports val/test accuracy and per-class test accuracy.
- `model.npz` — `W` (10×40), `b` (10), `feature_idx` (40), plus k, C and the
  feature config. Deployed model: `logits = x[:, feature_idx] @ W.T + b`.

## Reproduce
```bash
python submissions/04_orientation_entropy_linear/train.py   # writes model.npz
python submissions/04_orientation_entropy_linear/eval.py    # independent verify
```

## Parameter count
The deployed model is a single affine map on 40 raw parameter-free features:
`W` is 10×40 and `b` is 10 → **410 parameters**. The StandardScaler is folded
into `W`/`b` (standardisation is linear), and the feature extractor has **zero**
learned parameters, so 410 is the honest total.
