# The standardiser was never free

*2026-09-02 17:00 — RESISC45 iteration 9*

## The idea

[[2026-09-02_1600_resisc45_coded_intercept]] ended by pointing at the two
remaining uncounted `+ K` costs, and the larger of them was the standardiser
itself: "`mu` and `sigma` are 2 x 512 values that the fold hides; every head in
this file leans on them, and the coded-intercept result shows the head leans on
them *harder* than anyone counted."

The convention says folding `mu`/`sigma` into the weights costs nothing. That
is true when the weights are **stored** — folding changes the numbers, not how
many there are — and it silently stopped being true when the head started
**reconstructing** its weights from `Dc @ P @ Df.T`. A deployment then has to
rebuild `w_eff = W / sigma` and `b_eff = b - sum_j W_j mu_j / sigma_j` out of
numbers the code does not contain.

## The price, exactly

Two of the three costs are avoidable and one is not:

* **Width-1 column atoms are free.** `w_eff[:, j] = Dc @ P[:, j] / sigma_j`, so
  dividing the code entries of column `j` by `sigma_j` reproduces the deployed
  head exactly. Per-column `sigma` is *absorbable* into values the head already
  stores. This is why every head up to [[2026-09-02_1230_resisc45_class_dictionary]]
  was correctly counted.
* **Width-2 atoms are not.** One value spans two columns, so the deployed
  direction `(s_i / sigma_i, s_j / sigma_j)` needs their ratio. Ratios compose,
  so the bill is one value per column touched minus one per connected
  component — 59 to 229 values, growing with the budget, entirely uncounted by
  [[2026-09-02_1420_resisc45_column_pairs]].
* **Centring is an intercept.** `b = 0` still deploys
  `b_eff = -sum_j W_j mu_j / sigma_j`. So the coded/`none` intercept of the
  previous iteration saved the *second* copy of 44 numbers, not the first.

Priced honestly, the 208-value head that first cleared 65% deploys **424**
values and the 304-value one that cleared 70% deploys **542**.

## The fix that pays: round sigma to a power of two

Two columns in the same octave are scaled by the *same* number, so their pair's
ratio is 1 and the common factor is absorbable exactly like a width-1 atom's.
Nineteen octaves cover the 512 candidate columns, so the whole 65,792-atom pair
dictionary costs **10 to 14 values** instead of 59 to 229, and conditioning is
barely touched because every scaled column lands in `[1/sqrt(2), sqrt(2)]`.
The honest frontier becomes **224** deployed values at 65% and **384** at 70%,
against 2,156 and 2,860 for a dense affine head.

Sharing `sigma` per *feature family* instead — the first thing I tried — is a
total failure (3.98% test), because within-family standard deviations span up to
16.7 octaves. The grouping has to be about scale, and a family is not a scale.

## The lesson that nearly cost the iteration

The intercept re-coding — approximate `b_eff` with `q` class atoms, refit on
uncentred features — read **0.5603** at 300 LBFGS steps and **0.6427** at 4,000,
for the identical model at the identical budget. I had already drafted it as a
negative result on the 300-step numbers, with the `q = 44` control (which spans
the whole intercept space and still lost) as the clinching argument that the 44
numbers were irreducible. Both were artefacts of an unconverged optimiser: the
uncentred problem is far worse conditioned than the centred one, and 300 steps
is enough for one and not the other.

The control that makes the corrected version safe is that the *centred* rows do
not move: 20 base rows measured at both step counts are identical to four
decimals. **A shared optimiser setting is not a shared amount of optimisation.**
Any comparison that changes the conditioning of the problem — dropping the
centring, changing the parameterisation, swapping the basis — has to re-check
convergence per arm, not per experiment.

## What this suggests next

* The candidate list is the last uncounted thing. The 512 columns come from a
  group-lasso ranking on train, and the pair enumeration from a ranking inside
  that; both are data-derived model structure priced only in index bits. The
  same argument that caught the standardiser applies to them.
* The frontier arm now reads 169 pool columns at 224 deployed values. The
  extractor is still the larger deployment artefact and nothing has optimised it.
* `sep_dict_deployed_values` should be the *only* place a stored-value count is
  computed. Three sections were reported against a count that lived in a
  one-line helper nobody re-derived when the head changed shape.
