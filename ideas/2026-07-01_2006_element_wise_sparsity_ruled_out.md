# Iteration 12 — Element-wise weight sparsity ruled out (frontier stays 260 params)

**Date:** 2026-07-01 20:06
**Outcome:** NEGATIVE (no parameter reduction). Submission 09's 260-param
dense-backward linear model remains the frontier. A genuinely untried lever —
*element-wise* weight sparsity — is now closed with honest, airtight evidence.

## The idea

Every prior submission (01–09) selects whole **features** = whole *columns* of
the 10×F weight matrix. Dropping a feature zeros all 10 class-weights for it at
once, so a feature useful for only a few classes still pays for 10 weights.
An **element-wise** L1 / pruned model lets individual `W[c,f]` go to zero
independently and — counting the honest deployed params as `nnz(W)+10` (the
standard neural-net pruning convention; the sparsity *pattern* is structural,
exactly as prior submissions treat *which* k features are chosen as free) —
could in principle beat the 260-param dense-backward frontier if features were
useful for few classes each. This lever was never tried before this iteration.

## What was measured

Three independent element-wise-sparse approaches, all on the same 305-dim `o6`
pool, all counting `nnz(W)+10`:

1. **OvR-L1** (per-class liblinear L1, `experiments/sparse_l1_sweep.py`): sweep C.
2. **Masked-multinomial IMP** (`experiments/masked_imp.py`): joint multinomial
   head with a fixed 0/1 mask, pruned by global magnitude and **retrained** at
   each sparsity level (removes OvR's per-class handicap AND the L1 shrinkage
   bias — the *strongest* form of the idea). GPU/torch LBFGS.
3. **Honest CV of the whole IMP pipeline** (`experiments/masked_imp_cv.py`):
   run IMP inside every fold (standardise on fold-train, prune by fold-train
   magnitude, retrain, score held-out fold), 5 seeds × 5 folds, at 4 weight
   decays. IMP prunes by magnitude, never by held-out accuracy, so this CV is an
   estimate it never optimised.

## Results

**OvR-L1** needs ~553 non-zero weights (**563 params**) to first clear val≥0.940
— more than 2× the 260-param dense-backward frontier. At ~250–290 weights
(matched params) it gives only val ~0.90–0.92.

**Masked-IMP, single split, low WD (5e-4)** *looked* like a win — val 0.9407 /
test 0.9437 at 210 params, holding ≥0.940 down to ~200 params, and even beating
dense-backward at 260 params (val 0.9415 / test 0.9480). **This was a mirage.**

**Honest multi-seed CV kills it.** The 25-fold CV of the IMP pipeline **never
reaches 0.940 at any sparsity or any WD**:

| WD | best CV | CV @ 260 params |
|----|---------|-----------------|
| 1e-4 | 0.9369 | 0.9365 |
| 2e-4 | 0.9351 | 0.9350 |
| **5e-4** | **0.9375** | 0.9374 |
| 1e-3 | 0.9335 | 0.9330 |

**Anchor on the identical CV harness** (`/tmp/anchor.py`, sklearn multinomial):

| model (260 params) | honest 5×5 CV |
|---|---|
| **dense-backward-25** (column selection) | **0.9452 ± 0.0006** ✓ |
| element-wise IMP (best, WD=5e-4) | 0.9375 ✗ |
| (dense-full-305, 3060 params, for scale) | 0.9501 |

At **identical 260 params**, dense-backward scores 0.9452 vs IMP 0.9375 — an
**0.008 gap (>10 SE)**. Element-wise sparsity is decisively worse.

## Why

The frontier's parameter efficiency comes precisely from the **column
(feature-sharing) structure**: each of the 25 backward-selected features is a
strong, generic texture/spectral axis used by *all 10 classes jointly*, so its
10 weights are all load-bearing and the "cost per useful feature" is amortised
across classes. Scattering the same non-zero-weight budget element-wise forces
each feature to serve only a few classes, which needs *more* distinct features
to cover all 10 classes — the opposite of economical. Both the learned-L1
pattern (OvR, 563 params) and magnitude-IMP (600+ params to match) land ~2.3×
worse, and every WD confirms it.

## Methodological lesson (the important one)

The single-split val/test at these margins is **dangerously noisy** and produced
a false positive: val 0.9407 (5400 samples, σ≈0.003) read as a win when the
25-fold CV (σ≈0.0006, ~5× tighter) showed the truth was 0.9375. Two IMP scripts
with different prune schedules even disagreed by ~0.007 at 260 params — pruning-
schedule sensitivity is itself an instability red flag. **Always read the floor
off the multi-seed CV the selector never optimised, never a single val/test
split** — the same rule that has governed every backward-select submission.

## Bar for the next cut (unchanged, now with one more dead lever)

To go below 260 params honestly, a change must lift **multi-seed CV** at <260
params ≥ 0.940 (then confirm on val + test on BOTH seed partitions). Levers now
ruled out: SFBS/floating (iter 9), C-per-k (iter 10), family-through-backward
(iter 11), **element-wise weight sparsity (iter 12)**. Remaining untried:
- A genuinely NEW orthogonal parameter-free family (e.g. line/Hough length stats
  for the persistent-weakest Highway class) that lifts CV at k=24 on both
  partitions.
- Raise the difficulty bar to 95/96% test and re-optimise params there.
- GPU distillation of a strong pretrained net into a few conv layers (the linear
  frontier is CPU-only; 8×V100/32GB sit idle).
