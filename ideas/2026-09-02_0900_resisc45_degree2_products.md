# Degree-2 products are informative and unaffordable

*2026-09-02 09:00 — RESISC45 iteration 5*

## The idea

Every RESISC45 head so far is linear in the pool, so one stored value buys one
column. A product of two pool columns is still deterministic arithmetic on one
image — zero learned constants, and its standardiser folds into the head like
any other column's — so a weight on `a*b` costs exactly one stored value. With
45 classes and 1,024 values each class can afford about twenty-two weights, and
under that pressure a column that already carries an interaction should be worth
more per value than either factor alone. That is the whole hypothesis.

There was a second motive. Iteration 4 closed on the one cost prune-and-regrow
does not pay in stored values: the head reads 383 of 2,084 pool columns and the
whole-pool arm reads 621, so a deployment computes most of the pool. A degree-2
polynomial in `K` base columns needs only those `K` columns extracted, however
many products it then reads — a way to buy back extraction breadth for free.

## What happened

**The information is real and it is exactly where the pool is narrow.** At the
unconstrained ceiling, degree 2 is worth +9.0 test points on a 32-column pool,
+6.4 on 64, +3.2 on 128, +0.8 on 256, and +1.3 on the whole 2,084-column pool
(79.37% → 80.71%, the first ceiling movement since the native-resolution
rebuild). Degree 2 on 128 columns matches linear on 256, so at the ceiling the
expansion halves the extraction breadth.

**Under a budget almost none of it survives.** At 256 and 384 stored values the
expansion is worth 1.5–3.8 points at fixed extraction breadth, which is the
predicted regime. From 512 upwards it vanishes into the ±0.6-point sampling
noise of a 6,300-image split and is as often negative as positive. The frontier
does not move: 65% at 512 values, 70% at 896, exactly where iteration 4 left
them.

The instructive part is that **validation prefers an expanded pool at four of
seven budgets** — 0.7444 against 0.7348 at 1,024 — and test does not follow. The
8,000 to 33,000 extra candidate columns give the support search more ways to fit
18,900 training images than they give it structure.

The mechanism: above roughly 512 stored values the budgeted head is
*weight-limited, not column-limited*. It already reads 348 distinct columns at
512 values and 635 at 1,024 — far more columns than any class can afford to
combine — so a better-conditioned column does not relieve the binding
constraint, it only enlarges the search space. Iteration 4's "the head, not the
pool, is the constraint" holds one level deeper than it was stated: it is not
just that the pool is wide enough, it is that *making the pool better cannot
help at all* while weights are the scarce resource.

## Two negatives worth not repeating

Both are scripted in `experiments/resisc45_support_probe.py`.

**Bagged supports lose.** If wider pools overfit the support search, stability
selection is the textbook answer: fit prune-and-regrow on 8–16 subsamples of
train, keep the entries that recur most, refit convexly. It loses 1.3–1.7 points
at 512 values and 0.1–0.8 at 1,024. Frequency voting scores each entry on how
often it is chosen, never on what it adds given the others — the same
label-only criterion the group-lasso ranking used, and the same failure. The
regrow criterion is valuable precisely because it is conditional.

**Longer searches buy validation, not the frontier.** Raising the search from
4,000 epochs / 100 mask updates to as much as 16,000 / 400 lifts validation at
every budget, by up to 0.8 points, and the validation gate then moves test by
between −0.36 and +0.85 — four budgets up, two down, and neither the 65% nor the
70% threshold moves. The 1,024-value operating point actually *falls*: the
extended grid's highest-validation cell (top-512 list, 8,000 epochs, val 0.7490)
tests at 0.7254 against the 0.7290 the unextended grid already reported.
Iteration 4's finding that 4,000 beat 2,000 does not extrapolate, and every
additional arm offered to the gate is one more chance to spend the split's
resolution on nothing.

## What this suggests next

Two of the three things tried this iteration raised validation without raising
test, and the third lowered both. That is the signature of being at the noise
floor of a 6,300-image split, not of three bad ideas — differences below about
0.6 points cannot be resolved at all, and the selection step itself is now the
thing consuming them. Two consequences:

* any future RESISC45 claim at these budgets needs either a repeated-split or
  bootstrap confidence interval, or an effect larger than a point, before it
  should be believed;
* the remaining headroom is the 7-point gap between the 1,024-value head
  (72.90%) and the pool ceiling (79.37% linear, 80.71% with products), and
  closing it needs something that changes how many weights a class effectively
  has — a genuinely different head parameterisation — rather than another pool
  or another search.
