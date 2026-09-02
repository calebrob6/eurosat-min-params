# The RESISC45 budget was limited by the support search, not the pool

*2026-09-02 07:40 — RESISC45 iteration 4*

## The idea

Iteration 3 ended with a hand-tuned patch. Adding a new feature family made the
frontier *worse* until 64 of the 256 candidate slots were reserved for it,
because an L2,1 group-lasso ranking scores each column against the label and
never against the columns already selected. The obvious next move was a
redundancy-aware ranking. The better move was to notice why a ranking is needed
at all.

`fit_sparse_logreg_gpu` prunes: it starts dense on a candidate list and removes
weights. A column it drops in round one can never come back, so the candidate
list has to be right up front, and the only tool for building it is a ranking.
Every candidate-list experiment in iterations 1-3 — subset sizes, group-lasso
`lam`, the quota, the churn control — is downstream of that one limitation.

RigL removes it. Hold the active-weight count at the budget, and every so often
drop the smallest active weights and regrow the same number of *inactive* ones
with the largest dense loss gradient. The dense gradient is cheap here (one
`resid.T @ xs`, the same cost as the forward pass) and it is evaluated at the
current fit, so an entry is grown only if it explains error the active weights
leave behind. Redundancy gets scored against the fit rather than against the
label, which is what the group lasso could never do.

## What happened

Test accuracy at equal stored values, each method at its validation-selected
candidate list:

| Parameters | Prune-only | Prune-and-regrow | Delta |
|---:|---:|---:|---:|
| 256 | 0.4948 | 0.5710 | +7.6 |
| 384 | 0.5729 | 0.6259 | +5.3 |
| 512 | 0.6290 | 0.6571 | +2.8 |
| 768 | 0.6794 | 0.6978 | +1.8 |
| 1,024 | 0.7190 | 0.7290 | +1.0 |

65% test falls from 640 to **512** stored values, 70% from 1,024 to **896**, and
the 1,024-value point goes from 71.67% to **72.90%** — 1.5 points better than a
*dense* head with 2,860 values.

Three things are worth remembering beyond the numbers.

**The wider list is not the mechanism.** Giving prune-only the same 512-column
list gains 2.1 points at 256 values and *loses* accuracy at 384, 512 and 768.
The gain comes from being able to reconsider a dropped column, not from having
more columns to drop.

**The candidate list stopped being load-bearing.** Prune-and-regrow on the whole
2,084-column pool, with no ranking and no quota at all, stays within 1.2 points
of the best arm at every budget and wins outright below 512 values. Everything
iteration 3 built to manage the candidate list is now optional; the quota only
survives because the ranked list is a mild regulariser on the support search at
larger budgets, where 44 x 2,084 candidate entries start to overfit 18,900
training images.

**The gain is largest where the budget is tightest.** +7.6 points at 256 values
against +1.0 at 1,024. A prune-only search wastes its first rounds discovering
which columns are redundant; when only 212 weights survive, that discovery is
most of the job.

## What this suggests next

The support search was the binding constraint, and one change to it was worth
more than the entire object-layout feature pool. Two follow-ups look cheap:
re-run the per-class failure analysis on the regrown head (the previous analysis
described a support that no longer exists), and try the same regrowth idea on
the *rank* structure — a sparse reduced-rank projection whose nonzeros are
regrown the same way — since the reduced-rank head was only ever beaten because
its dense `k x r` projection could not be searched this way.
