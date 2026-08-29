# Raising the bar to >95%: hand-crafted wins again at ~252–279 params

*2026-07-02 16:00 — iteration 21 (new threshold, brief's escalation clause)*

Objective escalated to **>95% TEST**. Re-ran the pipeline. Hand-crafted
parameter-free features + folded **reference-class** linear head (9·(k+1), iter-19)
clear it cheaply; MOSAIKS needs ~17× the params; conv/distillation (iter 17) are
hopeless and were skipped.

## Hand-crafted mega-pool reaches 95% easily

Richest cached pool = o6 + line + corn2 + lbp2 + blob2 + sslope2 + ixcoh + ixtex2 +
xcorr + oent2 = **377 feats** (`experiments/ceiling95.py`). Full-pool logreg (C=1):
**val 0.9669 / test 0.9674**. The extra families that were REDUNDANT at the 94%
floor (lbp2/blob2/…) now help — higher accuracy needs more features.

Honest backward-greedy floor at the 95% gate (verify-A seeds 10-19 AND verify-B
seeds 30-39 AND val, all ≥0.95; `experiments/floor95_backward.py`):

| pool | k | params(9·) | verA | verB | val | test | notes |
|------|--:|-----------:|-----:|-----:|----:|-----:|-------|
| mega | 27 | **252** | 0.9573 | 0.9573 | 0.9520 | **0.9557** | robust floor |
| mega | 25 | 234 | 0.9564 | 0.9563 | 0.9504 | 0.9522 | val-noise island (k24,26 fail) |
| o6+line+corn2 | 30 | **279** | 0.9573 | 0.9575 | 0.9531 | **0.9537** | reproducible NOW (all src) |
| o6+line+corn2 | 28 | 261 | 0.9561 | 0.9559 | 0.9500 | 0.9494 | GATE PASSES but test<0.95! |

Family breakdown at the mega floor (k=27): **o6:22, corn2:2, line:1, lbp2:1,
blob2:1** — o6 dominant plus one each of the orthogonal families.

## Two honesty notes at the 95% level

1. **Thin-gate trap.** val σ≈0.003, so committing the smallest k where val JUST
   clears 0.95 can yield test<0.95 (o6+line+corn2 k=28: val 0.9500 but test 0.9494).
   A VALID >95% submission must gate with margin (**val ≳ 0.953 = 0.95+1σ**), which
   lands k=30 (279) reproducibly / k=27 (252) on the mega pool.
2. **val non-monotonicity** (mega k=25 passes but k=24/26 fail) is the same noise —
   read the floor off the contiguous passing region, not an isolated squeak.

## MOSAIKS at 95% (seed-free gaussian, `mosaiks_floor95.py`)

Needs ~512 features to clear the gate → **4617 params** (9·513); the full 4096
feats give test 0.956. ~17× the hand-crafted params. Empirical is more accurate
(0.965) but its filters cost 60k–240k stored params. **MOSAIKS still loses on
parameters**, even at 95%. (Its value: it's the accuracy CEILING for cheap fixed
features, ~0.965–0.967 — the same as the hand-crafted full pool, interestingly.)

## Frontier @ >95%

- **Reproducible now: 279 params** (o6+line+corn2, k=30, test 0.9537) — all
  features already in `src/`; landable immediately with the reference-class head.
- **252 params** (mega pool, k=27, test 0.9557) if lbp2 + blob2 are promoted to
  `src/features.py` (each contributes 1 feature at the floor) — a ~27-param cut for
  the cost of promoting 2 clean-room families from `experiments/gstruct2_features_lib.py`.
- Context: 94% frontier is 171 params (sub 12). **94%→95% costs only ~80–110 params.**

## Next levers below 252

- Pre-screen more orthogonal families (junction-angle, morphology) vs the frozen
  95% subset via `fast_probe.py` — same method that drove the 94% ladder.
- Two-partition confirm (select 20-29) for the committed k before a final submission.
