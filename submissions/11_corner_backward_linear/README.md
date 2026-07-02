# Submission 11 — 190-parameter linear model (o6 + line + corner pool, k=18)

**Test accuracy 0.9433 · 190 parameters (k=18) · independently verified from raw patches**

A **50-parameter (20.8%) cut** from submission 10 (240) — the largest single-
iteration param drop of the project — from adding one new orthogonal parameter-
free family (Harris corners) on top of the submission-10 pool and re-selecting
with a **fast signal-first** method rather than the ~1h full descent.

## What changed: the Harris corner/junction family

`harris_corner_features` (`src/features.py`, `patch_features(harris_corners=True)`):
a 6-feature zero-parameter family measuring **corner / junction density** (Harris
response) on the panchromatic, NDVI and NDBI channels — `frac` (corner-pixel
fraction) and `mag` (mean response) per channel. Pool order is
`o6[0:305] + line[305:314] + corn2[314:320]` (320-dim).

**Why it is orthogonal.** Corners occur where edges *meet*. Every prior directional
feature is either local (coherence, orientation entropy/histogram, index-map
variants — aggregates of isolated per-pixel gradient directions) or a straight-line
statistic (the submission-10 Hough family). None separates a **junction grid**
(Residential / Industrial blocks, where lines cross) from **parallel rows that
never cross** (AnnualCrop) — a global-layout axis the corner response captures.

**The pre-screen that qualified it (iteration 15).** Added to submission 10's
frozen k=23 subset, every one of the 6 corner features lifts held-out val; the
best, `corn2mag_ndbi`, by **+0.0046** (`corn2frac_ndbi` +0.0044, `corn2frac_ndvi`
+0.0041) — more than the +0.0042 the line family showed before it cut 260→240.
This ~seconds-long screen is the cheap qualifier that separates a real orthogonal
family from selection noise, run before any descent.

## Method: fast signal-first selection (not the ~1h full descent)

Instead of the two-partition L1-top-42 descent (`experiments/backward_select.py`,
~1h), this submission uses `experiments/fast_probe.py` + a warm-started descent:

1. **Warm start from the validated frontier.** Begin the backward-greedy descent
   from submission 10's 23 features UNION only the 6 NEW corner features
   (|init|=29) — *not* the L1 top-42 (~54) and *not* re-adding the 9 line features
   sub10 already settled. If the family can't improve the validated frontier it
   won't help, so this loses no real signal while cutting most of the descent.
2. **Greedy drop** the feature whose removal most helps mean 5-fold train-CV over
   SELECT seeds {0..9}, re-fitting each step, recording every intermediate subset.
3. **Confirm each k on TWO disjoint verify blocks + held-out val.** verify-A = CV
   seeds {10..19}, verify-B = CV seeds {30..39} (neither used to select), plus the
   held-out val split. Commit the smallest k clearing 0.940 on all three.

The deployed model is a single affine map `logits = x[:, idx] @ W.T + b` on the
k=18 selected raw features (StandardScaler folded in), so **params = 10·(18+1) = 190**.

## Honest floor: k=18 (190 params)

| k | params | verify-A | verify-B | val | test |
|--:|-------:|---------:|---------:|----:|-----:|
| 20 | 210 | 0.9475 | 0.9469 | 0.9404 | 0.9478 |
| 19 | 200 | 0.9459 | 0.9456 | 0.9417 | 0.9448 |
| **18** | **190** | **0.9442** | **0.9442** | **0.9406** | **0.9433** |
| 17 | 170 | — | — | <0.94 | (val gate fails on 3-seed probe) |

Every k down to 18 clears 0.940 on **both** denoised verify blocks *and* val, so
k=18 is committed. The two verify-CV blocks (10-seed × 5-fold each) are the
reliable floor indicator and sit at **+0.0042** above the bar; test is **+0.0033**.
val (0.9406) is the thinnest margin (+0.0006), but val on 5,400 patches has
σ≈0.003, so val is the noisiest of the three held-out estimates — the two verify
blocks and the independent test are what make k=18 defensible. `eval.py` recomputes
every feature from the raw `.tif` patches (bypassing all caches) and reproduces
val 0.9406 / test 0.9433 exactly.

## Selected features (k=18)

16 from the `o6` spectral/texture pool + **1 line** (`linet3_ndvi`, idx 310, the
sole line feature that survived to submission 10) + **1 corner**
(`corn2frac_ndvi`, idx 316). Indices:
`[58, 65, 68, 75, 86, 92, 93, 145, 203, 273, 275, 287, 291, 294, 297, 298, 310, 316]`.
One feature from each genuinely-new orthogonal axis (line, corner) again moves the
whole floor — the recurring pattern of the project.

## Reproduce

```bash
python submissions/11_corner_backward_linear/train.py \
    --warm-from submissions/10_line_backward_linear/model.npz --k-min 18
python submissions/11_corner_backward_linear/eval.py     # independent, from raw patches
```

## Caveats / next

- val margin at k=18 is thin (+0.0006); the denoised verify blocks (+0.0042) and
  independent test (+0.0033) carry the claim. A stricter robustness re-check
  (more held-out val-like folds) before pushing to k=17 is warranted.
- k=17 fails the val gate on the 3-seed probe (val 0.937/0.939) while verify-CV is
  still >0.94 — val is the binding constraint below 190, as it has been since 260.
- Next lever (iteration 16, pre-screened): cross-family stacking of
  corn2 + lbp2 + blob2lrg is nearly additive (frozen-23 + two features → val
  0.9493), suggesting headroom below 190 if a second new family is pooled and the
  same fast probe qualifies it.
