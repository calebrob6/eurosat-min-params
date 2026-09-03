# RESISC45: the search is chaotic, and a union of its supports clears 80%

*2026-09-03. Companion to "The search is chaotic" and "A union of supports
clears 80%" in RESULTS.md.*

## The question

The headroom note left the 4,096-value frontier at 79.57% test (3,628 deployed
values, 16,384-atom class-dictionary head on the 640-column quota list,
search weight decay 1e-3) with the prune-and-regrow search's own overfitting
named as the binding constraint and its decay as the lever. This iteration was
meant to tune that decay. It found something more basic first.

## Findings

**1. The search is deterministic and chaotic.** It starts from the top-k of
the dense gradient and never draws a random number, so its seed argument does
nothing. Perturbing the decay by 0.2% per "seed" instead gives supports whose
test accuracy differs by up to 2.3 points: 32 near-identical searches at 3,628
values read 0.7931 +/- 0.0064 (0.7741-0.8025). Every earlier single-run
comparison at this budget -- including the +1.1 for decay 1e-3 over 1e-4, and
the 79.57% frontier itself -- was read against that noise without knowing it.
Validation has a 0.5-point spread over the same runs and cannot pick the good
draws reliably.

**2. The decay and the list were not the lever.** Decays above 2e-3 lose
1-2 points, the 7e-4 to 2e-3 range is flat inside the noise, drop fraction 0.3
is marginally better than 0.5, and no wider quota list beats the 640-column
incumbent. Once the search is the constraint, more candidates are more to
overfit.

**3. A union of supports is the lever.** Run the search four to eight times
under the tiny perturbations, take the union of the supports (1.6-3.6x the
budget), refit convexly, magnitude-prune back to the budget in three geometric
steps with a fresh convex refit each step, then refit at the validation `C`.
Eleven such heads at 3,628 values read 0.8011 +/- 0.0021 (0.7986-0.8049),
with two disjoint search sets agreeing; the best reads 80.49% and the
validation pick 80.21%. The single-search noise is variance in *which* good
entries are kept, and the union keeps them all before a convex objective
decides.

**4. The controls locate the gain.** Pruning one search run at 2x or 4x the
budget reads 0.787-0.790, inside the single-search distribution, so it is the
union of *same-budget* searches that matters and not iterative magnitude
pruning from a larger support. Beyond eight searches the union passes 5x the
budget and the gain fades. Budget slack above 3,628 buys nothing (0.799 at
3,884, 0.801 at 4,076), and below it the union head reads 0.798 at 3,116 and
0.794-0.798 at 2,604.

## Why this is not the stability selection that lost before

The frequency vote over subsampled supports (negative-results table) rewards
individually stable entries and reintroduces the redundancy the regrow
criterion removes. The union keeps every entry any search wanted and lets a
convex refit plus magnitude pruning decide which survive, so correlated
entries compete on the refit's terms rather than being counted twice.

## What is left

The list's linear ceiling is 81.0% and its MLP ceiling 83%. The next lever is
a head with a few learned hidden units (a sparse first layer plus a dense
second), or a search whose single run generalises as well as the union does.
Any comparison at this budget now needs at least four perturbed repeats, or
the union arm, to be read at all.
