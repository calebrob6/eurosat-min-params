# Re-sweeping families through backward-greedy (ixcoh) — RULED OUT as a param lever

**Iteration 11 — 2026-07-01 19:11**

## Question

Iterations 7–10 established that on the `o6` 305-feature pool, **backward-greedy
feature selection** reaches an honest floor of **k=25 (260 params, submission 09,
test 0.9443)**; k=24 fails held-out val on both seed partitions, and C-tuning
(iteration 10) does not move it. The notes' **#1 untried lever** was:

> re-sweep candidate families through backward-greedy (NOT L1-rank). `ixcoh`
> (index-map structure-tensor coherence, 6 features) raised *test* at every k in
> the iteration-7 floor sweep but never moved the *L1* floor — a hint it might
> move the *backward* floor.

This iteration tested exactly that: does adding `ixcoh` to the pool and running
backward-greedy lower the floor below k=25 (or at least widen the razor-thin
k=24 val margin enough to legitimize a 250-param model)?

## What had to be fixed first

The existing `backward_select.py POOL=o6+ixcoh` was a **no-op**: the descent
starts from the L1 **top-42**, but the 6 `ixcoh` features rank at L1 positions
**62, 101, 155, 178, 307, 308** — *all below 42*. So the descent never even
sees `ixcoh` and reproduces the pure-o6 floor exactly (confirmed:
`ixcoh_in_set=0` at k=42 on both partitions).

Fix (this iteration, all in `experiments/backward_select.py`):
- **`FORCE_IX=1`** unions the added-family indices `[o6_dim, dim)` into the
  starting universe, so the descent begins from **L1 top-42 ∪ all 6 ixcoh = 48
  features** and backward-greedy is free to *keep* ixcoh if it helps. This is the
  maximal-power test (greedy gets free choice over the best o6 features + every
  ixcoh feature regardless of L1 rank).
- **`C` env var** (was hardcoded 10) so the CV-optimal C≈20 from iteration 10 can
  be swept later.
- **Per-step drop logging + `ixcoh_in_set` counter + low-k subset dumps**, so a
  null result reveals *why* (family dropped early vs retained-but-no-gain).

## Result — forced-union descent, C=10, both partitions (48 → k=23)

Honest floor = smallest k with **verCV≥0.940 AND val≥0.940**. Low-k rows:

| k | params | P0 verCV | P0 val | P0 ixcoh | P20 verCV | P20 val | P20 ixcoh |
|---|---|---|---|---|---|---|---|
| 28 | 290 | 0.9459 | 0.9407 | 1 | 0.9451 | 0.9406 | 3 |
| 27 | 280 | 0.9459 | **0.9420** | 0 | 0.9441 | **0.9415** | 3 |
| 26 | 270 | 0.9457 | 0.9402 | 0 | 0.9440 | **0.9398 ✗** | 3 |
| 25 | 260 | 0.9448 | **0.9393 ✗** | 0 | 0.9431 | 0.9413 | 2 |
| 24 | 250 | 0.9441 | 0.9380 ✗ | 0 | 0.9425 | 0.9407 | 2 |
| 23 | 240 | 0.9431 | 0.9381 ✗ | 0 | 0.9414 | 0.9406 | 2 |

Per-partition floors: **P0 = k=26**, **P20 = k=23** — a 3-step disagreement.

### The strict two-partition robust floor (both must pass at that k)

| pool | robust floor | note |
|---|---|---|
| **o6 (submission 09)** | **k=25 (260)** | P0 0.9404 ✓, P20 0.9407 ✓ |
| **o6+ixcoh forced** | **k=27 (280)** | P0 breaks at k=25 (0.9393), P20 breaks at k=26 (0.9398); both recover only at k=27 |

**Forcing `ixcoh` in REGRESSES the robust floor 260 → 280 params (−20).** It is
strictly worse, not better. No 250-param model, and it *loses* the k=25 that pure
o6 holds.

## Why it fails

1. **Seed-partition-dependent path.** The greedy is guided by SELECT-CV over its
   seed set. With SELECT 0–9 (P0) it drops all ixcoh by k=27 and lands on a
   *different, worse* pure-o6 subset than the real o6 descent (the k=25 subsets
   differ: forced-path has feats 94,144,278,58 where o6 has 145,33,300,304),
   giving val 0.9393 vs o6's 0.9404. With SELECT 20–29 (P20) it *keeps* ixcoh
   305 (ndvi) + 310 (bsi) to k=23. Opposite outcomes from the same pool = classic
   selection luck, not signal.
2. **ixcoh inflates SELECT-CV but not val.** Where ixcoh is retained (P20), verCV
   and val are consistently *lower* than pure-o6 at matching k (e.g. P0 k=34: o6
   val 0.9452 vs forced 0.9437). ixcoh nudges the greedy's own CV up enough to be
   kept, but it generalizes worse to held-out val — the exact SELECT-CV-overfit
   the honesty protocol was built to catch.
3. **Perturbation cost.** Even on P0 where ixcoh is fully dropped by k=27, the
   early "detour" of carrying ixcoh changed which o6 features were dropped, so the
   deterministic low-k subset is worse than the clean o6 descent.

## Verdict

**The #1 untried lever is closed for `ixcoh`.** Re-sweeping the index-coherence
family through backward-greedy does **not** lower the honest floor — it regresses
it by 20 params under the strict two-partition rule. This is consistent with the
project's recurring finding that **only a family that helps *alone* is worth
pooling** (subs 05/06) and that **selCV gains that don't reproduce on disjoint
verify/val are noise** (subs 07–10). Submission 09 (k=25, 260 params) stays the
frontier.

## Still untried (ranked, for next iterations)

1. **Other families through the same forced-union path** — `ixoent`/`ixdir`
   already looked neutral/negative under L1; `xband`-style *pairwise* families or
   a genuinely new axis (line/Hough length statistics targeting Highway 0.873,
   the persistent weakest class) are the only plausible movers. But the bar is
   now higher: a family must lift **held-out val at k=24 on BOTH partitions**, not
   just its own selCV. `FORCE_IX` + a new POOL branch makes this a one-line test.
2. **Distill a tiny conv net on the idle 8×V100 GPUs** — the linear+handcrafted
   route is at a hard floor (260 params, val-gated); a learned 2–3-layer conv with
   a few hundred params is the only path likely to break well below 260 while
   clearing 94%. Every iteration so far has left all 8 GPUs idle.
3. **Raise the bar to 95%/96%** (per the objective's fallback) and optimize params
   there — the 260-param model already clears 94% comfortably on test (0.9443–
   0.9469 at C=20).

## Reproduce

```
# forced-union backward descent (writes backward_select_o6_ixcoh_*force_result.txt)
FORCE_IX=1 POOL=o6+ixcoh START=42 STOP=23 SEL0=0  VER0=10 NSEED=10 C=10 TAG=p0C10force  WORKERS=16 python experiments/backward_select.py
FORCE_IX=1 POOL=o6+ixcoh START=42 STOP=23 SEL0=20 VER0=30 NSEED=10 C=10 TAG=p20C10force WORKERS=16 python experiments/backward_select.py
```
~50 min/partition on the shared machine (48→23, 10 SELECT + 10 VERIFY seeds).
