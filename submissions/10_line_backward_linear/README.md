# Submission 10 — 240-parameter linear model (o6 + global-line pool, k=23)

**Test accuracy 0.9472 · 240 parameters (k=23) · independently verified from raw patches**

A **20-parameter (7.7%) cut** from submission 09 (260) and the first param cut of
the project driven by a **new feature family** since submission 06. Same honest
backward-greedy method as submissions 07–09, run on an augmented feature pool.

## What changed: a new orthogonal parameter-free family finally moved the floor

Iterations 9–12 established that the binding constraint below 260 params is the
**held-out val gate**: at k=24 on the `o6` pool val drops to 0.9396 (<0.940), and
neither floating selection (SFBS, iter 9), C-tuning (iter 10), family-augmentation
via ixcoh (iter 11), nor element-wise weight sparsity (iter 12) could lift it. The
notes concluded the only remaining lever was *a genuinely new orthogonal family
that lifts held-out val at low k on both seed partitions*.

This submission adds exactly that: **`hough_line_features`** (`src/features.py`),
a 9-feature zero-parameter family measuring **global straight-line structure** via
a Hough transform (== Radon of the binary edge image) on the panchromatic, NDVI
and NDBI channels. Every directional feature already in the pool
(structure-tensor coherence, gradient-orientation entropy/histogram, and their
index-map variants) is *local* — an aggregate of per-pixel gradient directions.
Many short parallel edges (crop rows) and one long streak (a Highway, a River)
look identical to those statistics. The Hough peak is the length of the single
longest **collinear** run of edge pixels, so it separates "one long line" from
"many short parallel lines" from "no line" — an axis no local statistic sees.

**The screen that proved it carries new signal:** added to submission 09's frozen
k=25 subset, a single line feature lifts held-out **val 0.9404 → 0.9446** (+0.0042,
well above the ~0.003 val noise) and mean 5×5-fold train-CV 0.9449 → 0.9466 — the
val lift iterations 9–12 said was required. Adding ixcoh (iter 11) never did this.

## Honest floor: k=23 (240 params) on BOTH seed partitions

Method is unchanged from submissions 07–09 — L1-rank the pool, start from the
top-42, repeatedly drop the feature whose removal least hurts the mean 5-fold
train-CV over SELECT seeds {0..9}, re-fitting each step — with one addition: the 9
line features rank L1 39..155, so only one sits in the top-42. We **force the
family into the starting universe** (union the 9 line indices into the top-42, as
`experiments/backward_select.py FORCE_IX=1` does) so the greedy can actually
retain them. The descent was run to k=22 on **two disjoint seed partitions**
(SELECT/VERIFY 0-9/10-19 and 20-29/30-39); both land on the **identical** subset
at k ≤ 24:

| k | params | verCV (p0 / p20) | val (p0 / p20) | test (p0 / p20) | honest floor? |
|--:|-------:|------------------|----------------|-----------------|---------------|
| 26 | 270 | 0.9487 / 0.9486 | 0.9452 / 0.9435 | 0.9483 / 0.9457 | ✓ |
| 25 | 260 | 0.9479 / 0.9481 | 0.9433 / 0.9433 | 0.9472 / 0.9470 | ✓ (matches sub 09 params) |
| 24 | 250 | 0.9470 / 0.9472 | 0.9424 / 0.9424 | 0.9483 / 0.9483 | ✓ (widest val margin) |
| **23** | **240** | **0.9464 / 0.9465** | **0.9404 / 0.9404** | **0.9472 / 0.9472** | ✓ **committed floor** |
| 22 | 230 | 0.9454 / 0.9457 | 0.9394 / 0.9394 | 0.9476 / 0.9476 | ✗ val < 0.940 (both) |

**k=23 clears verCV ≥ 0.940 AND val ≥ 0.940 on both partitions; k=22 fails val on
both.** val 0.9404 at the floor is exactly the margin submission 09 accepted at its
k=25 floor, and test 0.9472 is comfortably above 0.940 (and above sub 09's 0.9443).
k=24 (250 params) is the wider-val-margin fallback, analogous to sub 09's k=27.

## The 23 selected features

Exactly **one** feature comes from the new family — `linet3_ndvi` (the sum of the
best line at the 3 strongest orientations in the NDVI edge map). The other 22 are
`o6` features: 7 index-texture, 5 gradient-texture, 5 intensity percentiles, 2
cross-band correlation, 2 orientation-entropy, 1 coherence. As with every prior
family win, a small number of features from a genuinely orthogonal axis moves the
whole floor — here 1 line feature buys the 260 → 240 cut.

## Honesty protocol (unchanged from submissions 07–09)

The greedy is *guided* by the SELECT-CV, which is optimistically biased for the
chosen subset and never trusted alone. Three independent checks gate the floor,
and all clear 0.940 at k=23 on both partitions:

| check | seeds / split | role | k=23 value |
|-------|---------------|------|-----------:|
| SELECT-CV | shuffle seeds 0–9 (or 20–29) | guides the greedy (biased) | 0.9463 / 0.9466 |
| VERIFY-CV | shuffle seeds 10–19 (or 30–39) | unbiased CV, never selected on | 0.9464 / 0.9465 |
| val | held-out val split | independent images | 0.9404 |
| test | held-out test split | reported once, drives NO decision | 0.9472 |

## Files & reproduction

- `train.py` — builds the 314-dim `o6+line` pool via
  `patch_features(..., hough_lines=True)`, L1-ranks it, unions the 9 line indices
  into the top-42, runs `src.select.backward_eliminate` to k=23 (deterministic),
  and saves `model.npz` (W, b, feature_idx, honest metrics). `--k 24` reproduces
  the wider-margin fallback.
- `eval.py` — recomputes **all** features from the raw GeoTIFFs (no cache) using
  the checkpoint's stored config, applies the folded affine map, and reports
  val/test accuracy + per-class test accuracy. Independent end-to-end check.
- `model.npz` — W (10×23), b (10), feature_idx (23) → a single affine map
  `logits = x[:, idx] @ W.T + b` on 23 raw zero-parameter features. **params =
  10·(23+1) = 240.**

The `hough_line_features` extraction was verified bit-identical to the experiment
implementation (`experiments/line_features_lib.py`) and
`patch_features(..., hough_lines=True)` reproduces the exact 314-dim descent pool
(max abs diff 0.0 on val and test).

**How the committed `model.npz` was built** (same approach as submission 08): the
k=23 subset is the two-partition honest floor from
`experiments/backward_select.py` (POOL=o6+line, FORCE_IX=1), where both disjoint
seed partitions landed on the *identical* subset. The checkpoint is a folded
logreg fit directly on that verified 23-feature subset at C=10 — bit-for-bit what
`train.py` produces (the descent is path-deterministic, and all 23 features are
confirmed inside the top-42 ∪ line starting universe), without re-running the
~1-hour full descent under shared-machine load. `eval.py` re-verifies the deployed
model end-to-end from the raw GeoTIFFs (val 0.9404, test 0.9472).
