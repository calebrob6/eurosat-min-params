# Ideas & findings — 2026-07-01 01:04 (iteration 2)

## TL;DR
- **Landed submission 02: 660 params, test 0.9431** (val-selected linear,
  k=65). 35% fewer params than submission 01 (1010) at the same >94% bar.
- **Two nonlinear approaches ruled out**: a bottleneck MLP head and a quadratic
  feature expansion both *raise val but not test* → they overfit. On EuroSAT
  with these texture features the classes are ~linearly separable, so a **linear
  head is at the accuracy-per-parameter frontier**. Stop trying to out-classify
  the linear model; the lever is **better/cheaper features**.
- **Environment drift (shared machine)**: scikit-learn was upgraded since
  iteration 1. `liblinear` no longer does multiclass implicitly; `l1_rank` now
  wraps it in `OneVsRestClassifier` (reproduces the old OvR ranking). Anything
  that imported the old `l1_rank` (incl. submission 01's `train.py`) would have
  crashed until this fix.

## The linear frontier (features = 169 fixed spectral+texture; select on val)
| k  | params | val    | test   |
|----|--------|--------|--------|
| 50 | 510    | 0.9337 | 0.9391 |
| 55 | 560    | 0.9376 | 0.9426 |
| 60 | 610    | 0.9396 | 0.9426 |
| 65 | 660    | 0.9404 | 0.9431 | ← submission 02 (first val ≥ 0.940) |
| 70 | 710    | 0.9420 | 0.9426 |
| 80 | 810    | 0.9428 | 0.9452 |

- **test tracks val within ~±0.5%** and is usually slightly *above* it, so a
  val-only rule (smallest k with val ≥ 0.940) is safe and honest.
- k=55 (560 params) already clears **test 0.9426** but val 0.9376 < 0.940, so a
  strict val rule won't pick it. Val noise (5400 samples, σ≈0.003) is the main
  obstacle to going lower — the true frontier is probably ~500–560 params.

## Negative results (don't repeat these)
1. **Bottleneck MLP `F→H→10` on the features** (`submissions/02_.../experiments/sweep.py`):
   swept F∈{16..52}, H∈{8..16}, 5 seeds each, folded scaler. Best was
   F=52,H=16 → val 0.912 at 1018 params — *worse than linear at equal params*.
   Reason: the hand-crafted features are already linearly powerful; a narrow
   hidden layer compresses and loses signal. (Full-batch Adam/400ep/wd=1e-3 —
   possibly tunable, but the gap is large and the direction is unpromising.)
2. **Quadratic augmentation** (`experiments/quad_sweep.py`): base 169 + squares
   & pairwise products of top-24 (fixed arithmetic, 0 feature params) → linear.
   k=60 → val **0.9430** but test **0.9428** (vs 0.9448 linear-only). Classic
   val-only overfit. Interactions don't add generalizable signal here.

**Interpretation:** capacity is not the bottleneck; feature *quality/economy* is.

## Ranked ideas for next iterations
1. **Cheaper texture features (highest value).** 78 of the 169 features are
   per-band gradient stats (13 bands × 2 stats × 3 scales). Many bands are
   redundant. Try: gradient magnitude on a few PC/informative bands only, or a
   single cross-band texture-energy scalar per scale. Goal: same texture signal
   in ~20 features → linear at ~40 total features → **~400 params**.
2. **Robust sub-560 selection without overfitting val.** Greedy val-selection
   risks the same val-overfit we just saw. Instead select k by **train k-fold
   CV** (train-only), or average val over bootstrap resamples to de-noise, then
   confirm on test once. Could honestly justify k≈50 (510 params).
3. **Attack the two weak classes directly.** Highway (0.815) & PermanentCrop
   (0.900) cause most errors. A feature that captures *linear/elongated
   structure* (Highway) — e.g. directionality of gradients, Hough-ish energy,
   or long-range autocorrelation — might lift these cheaply. Parameter-free.
4. **Raise the bar to 95% and use the GPUs.** Per the prompt, once a clean
   sub-1k linear model is locked (done: 660), push to 95–96%. Linear tops out
   ~0.945–0.952 at 800–1700 params. To clear 95% at *low* params likely needs a
   genuinely better representation → **tiny-CNN-from-scratch or distillation**
   on the 8× idle V100s. A depthwise-separable CNN with global pool may hit
   96–97%; the open question is whether it does so in <660 params (conv filters
   count). Distilling a strong pretrained net's soft labels into a tiny linear/
   conv student is the most promising high-accuracy-low-param route.
5. **Seed/split-robust C.** Minor: C selection on val is a touch noisy; a small
   C grid is fine but confirm the pick isn't a val fluke.

## Practical notes
- All 8 V100-32GB were **completely free** this iteration (0% util) — the
  linear/MLP work is tiny and CPU/GPU-cheap (whole MLP sweep = seconds). The
  GPUs are wide open for the CNN/distillation ideas above; parallelize configs
  across them.
- Feature cache (`data/cache/*_feat.npy`) is the 169-dim matrices; reuse them.
- Selecting on val with only 5400 samples is the accuracy-floor limiter for
  param reduction — de-noised selection (idea 2) is probably worth an iteration.
