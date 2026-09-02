# The intercept was 21% of the budget

*2026-09-02 16:00 — RESISC45 iteration 8*

## The idea

Every head in this run has been budgeted as `nnz(P) + 44`: the sparse code
plus one free intercept per non-reference class. That was a rounding error when
the budget was 1,024 stored values. It is not one now — iteration 7 put the 65%
target at **208** values, where 44 of them are intercept and only 164 are left
to spell the weights, and at 96 values the intercept is 46% of everything the
model stores.

An intercept is a weight on a constant column, so nothing forces it to be free.
Append a constant column to the standardised features and a matching identity
atom to `Df`, and the head becomes a bias-free `Dc @ P @ Df.T` over `k + 1`
columns: the prune-and-regrow search then *decides* how many stored values an
intercept is worth, against every other entry it could grow instead. The
parameterisation is nested — 44 values reproduce the free intercept exactly
whenever `Dc` carries the class singletons — so it can only lose by search.

## What happened

It pays, and it pays exactly where the head is starved: **+6.6** points
[+5.4, +7.8] at 96 stored values, +4.5 at 128, +2.1 at 160, +2.3 at 176, then
+1.0 [-0.1, +2.2] at 192 and nothing distinguishable from zero from 224 to
1,024. The frontier moves modestly: 65% is still first cleared at **208**
values but at 66.06% instead of 65.05%, and 70% falls from 320 to **304**
(70.38%).

The headline number and the frontier number are far apart because the large
gains are all below 192 values, where the model clears neither target. That is
the third consecutive iteration with the same shape — [[2026-09-02_1230_resisc45_class_dictionary]]
and [[2026-09-02_1420_resisc45_column_pairs]] both found large effects below
~512 values and nothing above — and it is now a reliable enough pattern to plan
around: a parameterisation change is worth testing only if the target budget is
in the starved regime, and the starved regime keeps getting smaller as the
frontier moves down.

## The control is the whole story

The `none` arm — no intercept in the standardised head at all — matches or
beats the `coded` arm at every single budget, and the coded search agrees with
it: given 16,384 class atoms times 65,792 column atoms to grow into, it puts
**0 to 3** stored values on the constant column below 384 (one at the
208-value operating point, zero at 96 and 112). So the gain is not "buy a
cheaper intercept", it is "stop buying one".

The reason is the standardiser fold, and it is worth being explicit because it
is what makes the accounting honest. A deployed head is `w_eff = W / sigma` and
`b_eff = b - sum_j W_j mu_j / sigma_j`, so `b = 0` in the standardised space
still leaves a *non-zero* deployed intercept — the one implied by centring at
the training mean. RESISC45's splits are exactly class-balanced, so there is no
prior to encode either. The 44 free values were buying a correction to an
intercept that was already there.

This generalises past this head. Any model in this repo that folds a
standardiser and then also stores a free intercept is paying twice for the same
thing, and the second payment is `K - 1` values.

## What did not work

* **Making the nesting exact.** Prepending the 44 class singletons to the
  Gaussian class dictionary, so 44 code values could reproduce a free intercept
  exactly, reads 0.6302 at 160 and 0.6743 at 256 against 0.6290 and 0.6863
  without them. Same verdict on designed atoms as iteration 6.
* **Distillation**, in a separate experiment. A 79.4%-test full-pool logistic
  teacher, the same teacher 5-fold cross-fitted, a teacher fitted on the
  student's own candidate columns (which is the *better* teacher, 79.8%), and
  label smoothing as a softening control: 31 of 32 arms fall below the
  hard-label head at 208 and 256 stored values, monotonically in temperature.
  A linear student in the teacher's own hypothesis class has nothing to gain
  from softened targets and loses gradient signal on the decisions its few
  weights have to get right — which is the opposite failure to the EuroSAT
  convolutional-distillation row, where the student's *representation* was the
  bottleneck.

## What this suggests next

* The two remaining `+ K` costs in the accounting are now the candidate-list
  indices and the standardiser itself. `mu` and `sigma` are 2 x 512 values that
  the fold hides; every head in this file leans on them, and the coded-intercept
  result shows the head leans on them *harder* than anyone counted. A section
  that prices them honestly — or replaces them with a fixed, data-independent
  scaling — would either confirm the frontier or move it a long way in the
  wrong direction, and it is better to know.
* The per-class picture still has not been looked at since iteration 2, and the
  head has changed parameterisation three times since. bridge, ship and
  railway_station were the starved classes.
* At 208 values the head now reads 188 pool columns. The extractor is a far
  larger deployment artefact than the head and nothing in this run has
  optimised it.
