# Submission 07 — 310-parameter linear model via backward-greedy selection

**Test accuracy 0.9494 · 310 parameters (k=30) · independently verified from raw patches**

A **40-parameter (11%) cut** from submission 06 (350) — and the first win of the
project that comes from **a better feature selector, not a new feature family**.
Same 305-feature zero-parameter `o6` pool as submission 06; the only change is how
30 of those features are chosen.

## What changed: backward-greedy elimination instead of L1-rank top-k

Every prior submission selected features by **L1-rank `top-k`**: fit one L1
logreg, rank features by coefficient magnitude, keep the first `k`. That order is
fixed and never reconsidered — it is only a *proxy* for the best `k`-feature
subset. The honest multi-seed train-CV≥0.940 floor on the `o6` pool under L1-rank
is **k=34 (350 params)**; below that the L1 top-k drops below 0.940 (k=33 CV
0.9395).

Submission 07 replaces the selector with **backward-greedy elimination**
(`src/select.py::backward_eliminate`): start from the L1 **top-42** (a set well
above the bar), then repeatedly drop the single feature whose removal *least*
hurts the mean 5-fold train-CV, re-fitting after every drop. Re-evaluating the
whole remaining set at each step finds far smaller subsets that still clear
0.940. The floor falls to **k=30 (310 params)** — and test *rises* to 0.9494,
because the retained 30 features generalise better than L1's top-30.

## Honesty protocol (backward-greedy on CV can overfit the CV)

Because the greedy is *guided* by CV, that selection CV is optimistically biased
for the chosen subset. We therefore never trust it alone; three independent
checks gate the floor:

| check | seeds / split | role | k=30 value |
|-------|---------------|------|-----------:|
| SELECT CV | shuffle seeds 0–9 | drives the greedy removals (biased) | 0.9474 |
| **VERIFY CV** | shuffle seeds 10–19 (never used to choose a feature) | unbiased CV estimate | **0.9468** |
| **val** | held-out val split (different images) | independent generalisation | **0.9428** |
| test | held-out test split | reported once, used for NO decision | 0.9494 |

All of VERIFY-CV, val and test clear 0.940 with margin at k=30. An **independent
seed re-run** (SELECT seeds 20–29, VERIFY 30–39) reproduced the same floor,
confirming it is not an artifact of the seed partition (see the ideas note).

The chosen 30-feature subset draws from every family in the pool (index-map
texture, multi-scale gradient, percentiles, orientation entropy, coherence,
cross-band correlation) — backward elimination spreads the budget across
orthogonal axes rather than piling onto the highest-magnitude L1 features.

## Deployed model

A single affine map on the 30 selected **raw** (parameter-free) features:

```
logits = x[:, feature_idx] @ W.T + b        # W: (10, 30), b: (10,)
pred   = argmax(logits)
```

Parameter count = `10 * (30 + 1) = 310` exactly (the StandardScaler is folded into
`W`, `b`; feature extraction has zero learned parameters). `model.npz` stores
`W`, `b`, `feature_idx`, and the `o6` feature config.

## Reproduce

```bash
# derive the model (backward-greedy from the o6 pool; ~minutes, parallel)
python submissions/07_backward_select_linear/train.py --k 30 --workers 16
# independently verify end-to-end from raw GeoTIFFs (recomputes all features)
python submissions/07_backward_select_linear/eval.py
```

`train.py` re-derives the exact same 30 features (the greedy is deterministic
given the pool, select seeds, and C); `eval.py` recomputes the whole feature pool
from the raw patches — bypassing every cache — and applies the folded weights, a
genuine end-to-end check.

## Frontier (all independently verified from raw patches)

| sub | params | test | key change |
|----:|-------:|-----:|------------|
| 01 | 1010 | 0.9502 | spectral + multi-scale gradient |
| 03 |  510 | 0.9413 | structure-tensor coherence |
| 04 |  410 | 0.9457 | gradient-orientation entropy |
| 05 |  390 | 0.9461 | cross-band spatial correlation |
| 06 |  350 | 0.9437 | index-map spatial texture |
| **07** | **310** | **0.9494** | **backward-greedy selection (same pool)** |
