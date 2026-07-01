# Submission 09 — 260-parameter linear model (backward-greedy, k=25)

**Test accuracy 0.9443 · 260 parameters (k=25) · independently verified from raw patches**

A **30-parameter (10.3%) cut** from submission 08 (290), still comfortably above
the 0.940 bar. Same method and same zero-parameter `o6` feature pool as
submissions 07/08 — this submission **lands a lower honest floor that the earlier
runs never measured** because they stopped the backward descent at k=28.

## What changed: measure below k=28 for the first time

Submissions 07 and 08 ran the deterministic backward-greedy descent only down to
`STOP=28`, so they *reported* k=28 as the floor without ever evaluating k<28.
Iteration 9 resumed the identical, deterministic descent from the verified k=28
subset and continued to k=24 on **both** independent seed partitions
(`experiments/extend_lowk.py`). The resume reproduces the k=28 row bit-for-bit
(verCV 0.9462 / val 0.9417 / test 0.9487), confirming it is the same path; the
new low-k rows reveal the real floor is **k=25**, not k=28:

| k | params | verCV (p0 / p20) | val (p0 / p20) | test (p0 / p20) | honest floor? |
|--:|-------:|------------------|----------------|-----------------|---------------|
| 28 | 290 | 0.9462 / 0.9462 | 0.9417 / 0.9415 | 0.9487 / 0.9478 | ✓ (submission 08) |
| 27 | 280 | 0.9459 / 0.9457 | 0.9420 / 0.9417 | 0.9472 / 0.9474 | ✓ (widest val margin) |
| 26 | 270 | 0.9455 / 0.9457 | 0.9407 / 0.9404 | 0.9474 / 0.9470 | ✓ |
| **25** | **260** | **0.9451 / 0.9450** | **0.9404 / 0.9407** | **0.9443 / 0.9459** | ✓ **committed floor** |
| 24 | 250 | 0.9440 / 0.9441 | 0.9396 / 0.9387 | 0.9452 / 0.9456 | ✗ val < 0.940 (both) |

**k=25 clears verCV ≥ 0.940 AND val ≥ 0.940 on both partitions; k=24 fails val on
both.** So k=25 (260 params) is the honest backward-greedy floor on this pool.
Nothing about the method changed — L1-rank the 305-dim `o6` pool, start from the
top-42, then repeatedly drop the feature whose removal least hurts the mean 5-fold
train-CV (re-fitting each step) down to 25 features.

## Honesty protocol (unchanged from submissions 07/08)

The greedy is *guided* by the SELECT-CV, so that CV is optimistically biased for
the chosen subset and is never trusted alone. Three independent checks gate the
floor, and all three clear 0.940 at k=25:

| check | seeds / split | role | k=25 value (partition 0) |
|-------|---------------|------|-------------------------:|
| SELECT CV | shuffle seeds 0–9 | drives the greedy removals (biased) | 0.9449 |
| **VERIFY CV** | shuffle seeds 10–19 (never used to choose a feature) | unbiased CV estimate | **0.9451** |
| **val** | held-out val split (different images) | independent generalisation | **0.9404** |
| test | held-out test split | reported once, used for NO decision | 0.9443 |

The **independent-seed partition** (SELECT 20–29 / VERIFY 30–39) reaches the same
k=25 floor — verCV 0.9450 / val 0.9407 / test 0.9459 — so k=25 clearing the bar is
not a seed-partition artifact.

**Margin note (why 260 and not lower).** verCV (0.9451, +0.0051) and test (0.9443,
+0.0043) both clear 0.940 with room; the tightest of the three checks is val
(0.9404, only +0.0004). k=24 pushes val to 0.9396/0.9387 — below 0.940 on both
partitions — so 260 is as low as this pool + selector go while keeping every
honest check above the bar. Callers wanting a wider *val* margin can step back to
**k=27 (280 params)**, whose val (0.9420 / 0.9417) is actually the widest of the
low-k rows and exceeds submission 08's k=28 — a robust 10-param cut — via
`train.py --k 27`.

## Deployed model

A single affine map on the 25 selected **raw** (parameter-free) features:

```
logits = x[:, feature_idx] @ W.T + b        # W: (10, 25), b: (10,)
pred   = argmax(logits)
```

Parameter count = `10 * (25 + 1) = 260` exactly (the StandardScaler is folded into
`W`, `b`; feature extraction has zero learned parameters). `model.npz` stores
`W`, `b`, `feature_idx`, and the `o6` feature config.

## Reproduce

```bash
python submissions/09_backward_260_linear/train.py     # re-derive k=25, save model.npz
python submissions/09_backward_260_linear/eval.py      # independent check from raw patches
```

`train.py` re-runs the deterministic backward-greedy selection (L1 top-42 → k=25,
SELECT seeds 0–9); `eval.py` recomputes **all** features from the raw GeoTIFF
patches (bypassing every cache) and applies the folded linear model, an end-to-end
check of extraction + the deployed affine map.

## Frontier

| sub | params | test | key change |
|----:|-------:|-----:|------------|
| 01 | 1010 | 0.9502 | spectral + multi-scale gradient |
| 06 |  350 | 0.9437 | index-map spatial texture |
| 07 |  310 | 0.9494 | backward-greedy selection (o6 pool) |
| 08 |  290 | 0.9487 | commit the k=28 backward-greedy floor |
| **09** | **260** | **0.9443** | **extend the descent below k=28 → k=25 floor** |
