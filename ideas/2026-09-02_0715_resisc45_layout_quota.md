# RESISC45 object-layout pool: informative, redundant, and only useful under a quota

*2026-09-02 07:15 — RESISC45 iteration 3*

## The idea

Iteration 2's failure analysis said the residual RESISC45 errors are *object-layout*
distinctions — bridge/river, ship/harbor, railway_station/railway,
roundabout/intersection — and that the 1,579-column pool "contains nothing that
counts or measures discrete elongated objects". So: build that family and expect
the frontier to move at every budget.

`resisc45_gpu_features3.py` adds 505 columns — twelve-angle Radon projection
profiles, Hessian ridge statistics, directional run lengths, DoG local-maximum
counts with the second moments of the maximum cloud, log-polar symmetry, and
thresholded-mask moments. Seven GPU-minutes for all 31,500 images.

## What actually happened

**Merging the pools and re-ranking made it worse.** One group lasso over the
merged 2,084 columns puts 113 layout columns in the top 256, and validation
accuracy drops at *every* budget (0.7140 vs 0.7246 at 1,024 values). The per-class
deltas show exactly the predicted effect and its price: ship +5.2, bridge +5.0,
wetland +5.1 — against palace −12.9, storage_tank −8.8, harbor −7.7. At a fixed
budget the head reallocates weights to the new columns for the classes that want
them and starves everything else.

**The reason is redundancy, not weakness.** The layout family alone reaches
**65.21% test** with a 505-column unconstrained head — it clears the first target
by itself. But the *merged* unconstrained ceiling is 79.37% against 79.78% for the
base pool: it adds essentially no information the log-Gabor, autocorrelation,
projection-profile, and LBP families did not already carry. What it supplies is a
better-conditioned small subset of information the pool already had.

## The fix: reserve slots instead of holding a contest

Give the layout pool a fixed quota `q` of the 256 candidate slots rather than
letting it compete for them. Selecting `q` on validation:

| Parameters | q=0 test | selected q | test |
|---:|---:|---:|---:|
| 512 | 0.6059 | 64 | 0.6290 |
| 640 | 0.6421 | 64 | **0.6554** |
| 768 | 0.6740 | 64 | 0.6794 |
| 896 | 0.6860 | 32 | 0.6879 |
| 1,024 | 0.7048 | 64 | **0.7167** |

**65% test now first clears at 640 stored values instead of 768**, and the
1,024-value point improves 70.48% → 71.67%. A control that hands the same 64
displaced slots to the *next* 64 base columns gets about half the gain, so roughly
half is candidate-list churn and half is the layout family itself (it beats the
control at all five budgets by 0.5–1.1 points).

## Lesson worth keeping

A group-lasso ranking scores each column against the *label*, not against the
columns already selected. When a new family is individually strong but collectively
redundant, that criterion floods the candidate list with near-duplicates of
information the head can already get more cheaply. The cost is invisible in the
unconstrained ceiling and only appears under a parameter budget. Quota-limiting a
new family is a one-line intervention that turns a −1.1-point regression into a
+1.2-point gain; a redundancy-aware ranking (orthogonal matching pursuit against
the already-selected set) is the obvious next thing to try.
