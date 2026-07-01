# Ideas & findings — 2026-07-01 08:10 (iteration 10)

## TL;DR
- **No parameter cut this iteration.** I tested iteration-9's ranked idea #2 —
  **tune the logreg inverse-L2 strength `C` per k** — and it is **ruled out as a
  parameter-reduction lever**: the honest floor is **C-invariant at k=25 (260
  params)**. Every submission had fixed `C=10` arbitrarily; sweeping C does not
  legitimise k=24.
- **BUT C≈20 is the CV-optimal regularisation and a FREE robustness upgrade to the
  frontier.** At the SAME k=25 (260 params), C=20 gives strictly better val AND
  test than C=10 on both seed partitions — chosen honestly by the greedy's own
  SELECT-CV, which prefers C≈20–50 over 10. Submission 09 should deploy at C=20.
- **The binding constraint at the floor remains val, and it is C-immovable.** At
  k=24 the backward-greedy drops the selCV-maximising feature, which happens to
  tank val (C=20 → val 0.9378/0.9391, even *worse* than C=10's 0.9396/0.9387).
  No C makes val≥0.940 at k=24 on *both* partitions under the honest selector.

## What I ran
1. `experiments/c_sweep_lowk.py` — probe C∈{0.3,1,2,3,5,10,20,50,100} on the
   *fixed* C=10 backward-greedy subsets at k=25/24, both partitions, reporting
   selCV / verCV / val / test. Cheap first look: does regularisation strength
   alone move the checks?
2. `experiments/c_descent_lowk.py` + `experiments/extend_lowk.py` (now takes a `C`
   env var) — the honest test: **re-run the whole backward descent guided by the
   CV-picked larger C**, so each low-k subset is the one C=20 actually prefers,
   then read the floor off the independent checks. Resumed from the verified k=28
   subset (fast) and descended to k=22 at C=20 on both partitions.

## The C=20 low-k table (resume from k=28, honest protocol)
Honest floor rule = smallest k with verCV≥0.940 AND val≥0.940, on BOTH partitions.

| k | params | p0 verCV/val/test | p20 verCV/val/test | honest? |
|--:|-------:|-------------------|--------------------|---------|
| 25 | **260** | 0.9451 / **0.9409** / 0.9446 | 0.9448 / **0.9411** / 0.9469 | ✓ floor |
| 24 | 250 | 0.9441 / 0.9378 / 0.9450 | 0.9445 / 0.9391 / 0.9448 | ✗ val (both) |
| 23 | 240 | 0.9427 / 0.9391 / 0.9424 | 0.9435 / 0.9385 / 0.9448 | ✗ val (both) |
| 22 | 230 | 0.9411 / 0.9370 / 0.9391 | 0.9421 / 0.9378 / 0.9407 | ✗ |

Compare the C=10 floor (submission 09): k=25 val 0.9404/0.9407, test 0.9443/0.9459.
**C=20 is strictly better at k=25** (val +0.0005/+0.0004, test +0.0003/+0.0010,
verCV tied) and still fails k=24. selCV (the greedy's guide, training-only) rises
monotonically to C≈20–50 then plateaus, so C=20 is the *honest* pick — it is not
chosen on val/test.

## Why C can't cut params here
The floor is bound by **val**, and at the k=25→24 drop the greedy's selCV-optimal
feature to remove is *not* the val-optimal one. C changes which feature that is
(C=10 drops feat 300 on p0; C=20 drops a different one) but not the fact that the
selCV-honest drop hurts val below 0.940. The one subset that clears p0 k=24 (the
C=10 drop-300 set re-fit at C=20, val 0.9411 in the probe) still **fails p20**
(val 0.9391), so there is no single honest procedure giving k=24 on both
partitions. `verCV` (+0.004) and `test` (+0.005) clear k=24 comfortably — only
val gates it, and val at k=24 is ~0.28σ under 0.940 (val σ≈0.003), i.e. within
noise but failing the project's strict gate.

## Frontier (unchanged; all independently verified from raw patches)
| sub | params | test | key change |
|----:|-------:|-----:|------------|
| 08 | 290 | 0.9487 | commit k=28 backward-greedy floor |
| **09** | **260** | **0.9443** | extend descent to k=25 floor (C=10) |
| — | 260 | 0.9446/0.9469 | *same model at C=20 — strictly better, still 260* |

## Ranked ideas for iteration 11
1. **Re-sweep feature families through backward-greedy (NOT L1-rank).** Still the
   #1 untried lever. `ixcoh` (index-map coherence) raised test at every k but never
   moved the *L1* floor — a hint it may move the *backward* floor. `backward_select.py`
   already supports `POOL=o6+ixcoh`. Cost: a full 42→22 descent is ~1.5 h under
   current shared-machine load — run it as a fire-and-forget full-iteration
   background job on both partitions, or resume-from-k28 on the augmented pool
   after one slow 42→28 descent. This is the most likely route below 260 params.
2. **A genuinely new orthogonal parameter-free feature** that lifts the whole
   low-k val curve. Every existing family was vetted under L1-rank; some may help
   only under backward-greedy (see #1). Highway (0.873) and PermanentCrop (0.887)
   are still the accuracy cap — line/Hough features for paved linear structures
   are the untried orthogonal axis and the only plausible route to a 95% bar.
3. **The 8 idle V100s remain unused.** A tiny distilled conv net (ZCA-whitened
   init, distill a strong pretrained EuroSAT model into 1–2 conv layers) is the
   one unexplored model class for a sub-250-param / >95% target.
4. **Deploy submission 09 at C=20** (free robustness upgrade, same 260 params) if
   a more defensible frontier point is wanted before pushing lower.

## Infra added
- `experiments/c_sweep_lowk.py` — fixed-subset C probe (selCV/verCV/val/test grid).
- `experiments/c_descent_lowk.py` — honest per-C full backward descent from the
  L1-top-N (PART/C_GRID/START/STOP env-parametrised), both partitions.
- `experiments/extend_lowk.py` — now takes a `C` env var (default 10.0, so old
  behaviour is unchanged) to re-run the low-k drops from the verified k=28 subset
  under a different C; writes `extend_lowk_s{SEL0}_C{C}_result.txt`.
- Practical note: **higher C ⇒ slower descents** (weaker L2 → more lbfgs iters,
  ~28 s/mean_cv at C=20). The per-row serial SELECT+VERIFY CV is the ~2 min/step
  wall-clock bottleneck; resume from a saved subset rather than re-running 42→k.
