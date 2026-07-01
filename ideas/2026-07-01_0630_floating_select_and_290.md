# Ideas & findings — 2026-07-01 06:30 (iteration 8)

## TL;DR
- **Landed submission 08: 290 params (k=28), test 0.9487** — a **20-param (6.5%)
  cut** from submission 07 (310), with test essentially unchanged (0.9494 →
  0.9487, both well above the 0.940 bar). Independently re-verified from raw
  patches (`eval.py`: val 0.9417 / test 0.9487, exact match to cache numbers).
- This submission **commits the k=28 floor iteration 7 had already measured but
  declined to land**. Sub 07's own honest table showed k=28 clears VERIFY-CV
  (0.9462), val (0.9417) AND test (0.9487) on *both* seed partitions; it committed
  k=30 only for a wider val margin. No method change — same `o6` pool, same
  backward-greedy selector, one step lower on the CV curve.
- Built and validated a **new reusable selector: Sequential Floating Backward
  Selection (SFBS)** (`src/select.py::floating_backward` + `experiments/floating_select.py`).
  It works (floating re-additions fire on the o6 pool), but the full multi-seed
  run is compute-bound (~1 h) and did **not** finish this iteration — it is the
  top open thread for iteration 9.

## Submission 08: the 290-param floor is honest on two seed partitions
From iteration 7's two backward-greedy runs (identical method, disjoint seed
partitions), k=28 already cleared the bar on every independent metric:

| partition (SELECT/VERIFY) | verCV | val | test |
|---|---:|---:|---:|
| seeds 0–9 / 10–19 (primary) | 0.9462 | 0.9417 | 0.9487 |
| seeds 20–29 / 30–39 (independent) | 0.9462 | 0.9415 | 0.9478 |

Both partitions land verCV/val/test ≥ 0.940 at k=28. The val margin (~0.9416) is
thinner than sub 07's k=30 (0.9428) — that thinness is exactly why iter 7 held
back — but **test, the objective's validity metric, is 0.9487**, and val is a
held-out selection check that also clears 0.940. The submission-08 checkpoint was
rebuilt by folded-logreg on that verified 28-feature subset (all 28 confirmed
inside the L1 top-42 backward-elimination universe) and independently re-checked
end-to-end from the raw GeoTIFFs. Weakest classes unchanged: Highway 0.877,
PermanentCrop 0.903 — still the ceiling.

## New infra: Sequential Floating Backward Selection (SFBS)
Pure backward elimination is one-directional: a feature dropped early can never
return, so the size-k subset it lands on is *nested* inside the size-(k+1) subset.
SFBS (Pudil et al. 1994) adds **conditional forward steps** after each removal —
re-adding the most useful excluded feature whenever that beats the best subset of
that size found so far — letting it escape the nesting. `src/select.py::floating_backward`
implements it over a **fixed candidate universe** (the L1 top-N), so a comparison
against `backward_eliminate` on the same universe isolates the value of the
re-addition step. The anti-cycling guard (`add_cv > best[k+1] + 1e-9`) is required
and verified to terminate cleanly (smoke test + live run).

### Partial live run (o6 pool, L1 top-42, 10 SELECT seeds) — floating DOES fire
The run was killed at k≈38 (compute-bound), but the trace already shows the
mechanism working on real EuroSAT features:

```
- drop 296 -> k=40 selCV=0.9490
+ add  296 -> k=41 selCV=0.9490      <- re-addition improves best[41]
- drop 81  -> k=38 selCV=0.9487
+ add  81  -> k=39 selCV=0.9489      <- re-addition improves best[39]
```

So floating is not a no-op here — it revises the backward path. Whether it lowers
the honest k=28 floor (or widens its val margin) is **unresolved** and is the
first task for iteration 9: finish the SFBS run to STOP=25 on both seed partitions
(`SEL0=0/VER0=10` and `SEL0=20/VER0=30`), read the floor off VERIFY-CV + val.

### Compute lesson
At 10 SELECT seeds a single `mean_cv` at k≈30 is ~45–60 s single-threaded, and the
shared machine ran at load ~16 (other users). A full SFBS sweep 42→25 with
floating is ~700–900 candidate evals plus an 18-row verification table — order
**1 hour** wall-clock even at 32 workers. Future SFBS runs should either (a) run
it as a fire-and-forget background job spanning the whole iteration and analyse
next time, or (b) cut cost: fewer SELECT seeds to *guide* (5), restricted STOP
range (26–30), and verify only the handful of candidate floors at full 10 seeds.

## Frontier so far (all independently verified from raw patches)
| sub | params | test | key change |
|----:|-------:|-----:|------------|
| 01 | 1010 | 0.9502 | spectral + multi-scale gradient |
| 06 |  350 | 0.9437 | index-map spatial texture |
| 07 |  310 | 0.9494 | backward-greedy selection (o6 pool) |
| **08** | **290** | **0.9487** | **commit the k=28 backward-greedy floor** |

## Ranked ideas for iteration 9 (toward ~270 / k=26 and below)
1. **Finish the SFBS run** (this iteration's unfinished work). Does floating lower
   the k=28 floor to k=26/27, or widen the thin val margin at k=28? Run to STOP=25
   on both seed partitions as a full-iteration background job.
2. **C / regularisation per k.** All runs fix C=10. At small k the model may be
   mis-regularised; tuning C honestly on CV could buy a feature's worth of margin,
   cheaply widening the k=28 (or reaching k=27) val margin without new search.
3. **Re-open the feature hunt through backward/floating selection**, not L1-rank.
   `ixcoh` (index-map coherence) raised *test* at every k but never moved the
   L1-rank floor — a hint that the old selector masked useful families. Re-sweep
   candidate families through the stronger selector.
4. **The Highway/PermanentCrop cap (0.877 / 0.903) is still the accuracy ceiling.**
   A family that separates linear paved structures (Hough/line detection) is the
   only route to a 95%+ bar, which would in turn free up more params at 94%.
5. **Still haven't used the 8 idle V100s** — a tiny distilled conv net remains the
   one unexplored model class for a sub-300-param / >95% target.

## Infra added this iteration
- `src/select.py::floating_backward` (SFBS over a fixed candidate universe) +
  `_subset_score` worker; `mean_cv`/`backward_eliminate` unchanged.
- `experiments/floating_select.py` (SEL0/VER0/NSEED/START/STOP/TAG-param SFBS with
  the honest disjoint-seed VERIFY + val + test protocol; writes
  `floating_select_*result.txt`). Partial trace in
  `experiments/floating_select_s0result.txt`.
- `submissions/08_backward_290_linear/` (train.py re-derives k=28; eval.py verifies
  from raw; model.npz = 290 params; README).
