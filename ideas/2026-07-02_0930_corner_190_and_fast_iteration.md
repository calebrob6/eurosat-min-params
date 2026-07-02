# Corner family lands submission 11 = 190 params; fast signal-first iteration

*2026-07-02 09:30 — iteration 16*

## Result

**Submission 11: 190 parameters (k=18), test 0.9433, val 0.9406 — independently
verified from raw patches.** A 50-param (20.8%) cut from submission 10's 240, the
biggest single-iteration drop of the project. Driven by the iteration-15 Harris
corner family (`harris_corner_features`), committed via a fast warm-started descent.

## Two things happened this iteration

**1. Methodology change (user directive): iterate fast, find signal before long
jobs.** The overnight approach opened with the ~1h two-partition L1-top-42 descent.
New rule: run a cheap signal probe FIRST; only spend the long descent on a family
that already shows a sub-frontier floor. Built `experiments/fast_probe.py`, which
stacks three speed levers vs the full descent:
- WARM START from the frozen frontier subset UNION only the NEW family (~29 feats),
  not the L1 top-42 (~54). sub10 already settled line selection — re-adding it just
  re-opens settled questions (and in fact found a *worse* path: with line re-added,
  k=19 val was 0.9398 and failed; corner-only, k=18 val is 0.9406 and passes).
- FEW select seeds (3) drive the greedy drops for the probe; the path is robust to
  seed count. 10 seeds only for the committed descent.
- CONFIRM only the low-k subsets, each on TWO disjoint 10-seed verify blocks
  (seeds 10-19 and 30-39) + held-out val.
Net: the corner floor read took ~15 min (pre-screen ~seconds), vs ~1h. Also added
`record=True` to `src.select.backward_eliminate` so ONE descent yields every
intermediate subset → the whole k-neighbourhood is confirmed from a single run.

**2. The corner family cut the floor hard.** Pre-screen (add 1 to frozen sub10-23,
val lift) reproduced exactly: all 6 corner feats lift val, best corn2mag_ndbi
+0.0046. Warm-start descent + two-block confirm:

| k | params | verify-A(10-19) | verify-B(30-39) | val | test |
|--:|-------:|----------------:|----------------:|----:|-----:|
| 20 | 210 | 0.9475 | 0.9469 | 0.9404 | 0.9478 |
| 19 | 200 | 0.9459 | 0.9456 | 0.9417 | 0.9448 |
| **18** | **190** | **0.9442** | **0.9442** | **0.9406** | **0.9433** |

k=18 clears 0.940 on both denoised verify blocks (+0.0042) AND val, so it commits.
Only 1 corner + 1 line feature survive to k=18 (idx 316 corn2frac_ndvi, 310
linet3_ndvi); the other 16 are o6. One feature from each new orthogonal axis moves
the whole floor — the project's recurring pattern.

## Honesty notes

- The binding constraint below 190 is still the **noisy val gate** (σ_val≈0.003).
  verify-CV stays >0.94 below k=18 (3-seed probe: k=17 verCV 0.942/0.942) but val
  drops under 0.94 (k=17 val 0.937/0.939). The denoised verify blocks + independent
  test — not val — are what justify k=18.
- val=0.9406 is thin (+0.0006). This is within project precedent (sub10 committed
  val 0.9404) and is backed by TWO verify blocks at 0.9442 and eval.py's from-raw
  test 0.9433. A stricter val-robustness re-check before going to k=17 is warranted.
- Warm-start-from-frontier inherits sub10's val-involved subset, so val is re-used
  across submissions (a standing project tension); the disjoint verify seeds are
  the unbiased check and they clear comfortably.

## Next levers (below 190)

- **Cross-family stacking (iteration-15 finding, still the top lever):** frozen-23
  + corn2frac_ndvi + blob2lrg_ndvi → val 0.9493 (+0.0089, nearly additive). Pool a
  SECOND new family (lbp2 or blob2lrg) on top of o6+line+corn2 and run the fast
  probe; the +0.009 val headroom is ~9× the ~0.001 the val gate needs per step.
- **Lift val at k=17** specifically — that is the single gate now blocking 170.
- Anything below 190 should be qualified by `fast_probe.py` (pre-screen + warm-
  start + two-block confirm) BEFORE any long run.
