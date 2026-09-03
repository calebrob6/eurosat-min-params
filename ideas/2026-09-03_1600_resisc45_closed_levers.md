# Two non-pool levers closed on the averaged RESISC45 list (2026-09-03 16:00)

## Where this started

Iteration 5 left the union-of-supports head at 81.4% +/- 0.2 test at 3,628
deployed values on the dihedral-averaged pools (validation pick 81.56%), 1.2-1.7
points under its list's 83.1-83.3% dense ceiling, and named two levers that do
not touch the pools: a fixed nonlinearity per column and a head with a few
learned hidden units.  This iteration read both at the ceiling of the averaged
768-column 384/128/128 + 128 list, and the first under the union head too.

## Per-column monotone transforms (null)

The standardiser folds into the sparse code, so a scale-free map
`sign(x)|x|^p` of every column costs no deployed value; any map with a
per-column constant inside the nonlinearity (shift, scale, knot) costs one
value per column the support touches.  `resisc45_transform_ceiling.py`:
powers 1/2, 1/3, 1/4 read 0.8303 / 0.8278 / 0.8211 against 0.8325 raw, a
per-column pick by skewness 0.8313, a relative log 0.8330, and the train
quantile map to a normal, the bound on any monotone per-column function,
0.8302.  Raw plus a square-root copy reads 0.8335 where raw plus a hinge copy
reads 0.8398.  Under the union head the square root reads 0.8132 / 0.8105 and
the pick 0.8117 / 0.8122 at 3,628 values against 0.8122 / 0.8175 raw.  The
columns are missing a bend, not a change of scale.

## The nonlinear headroom survived averaging

`--probe` on the averaged list: hinge at the mean 0.8398, three knots 0.8430,
4,096 pair-ReLUs 0.8473, MLP width 64 / 128 / 512 / 1,024 0.8398 / 0.8459 /
0.8595 / 0.8627 against 0.8325 linear.  Three points of MLP headroom, as on
the old pool, but the hinge share has fallen from 1.7 to 0.7.

## A few learned hidden units (null)

`resisc45_hidden_units.py`: a dense linear head plus `H` ReLU units trained
jointly, each unit's inputs magnitude-pruned to `s` columns and retrained.
Up to 32 units with 8-64 inputs buy 0.1-0.6 points for 200-3,500 values; 64
units with dense inputs buy 1.5 for 52,000.  The budget table says 3,000 code
values buy 2.5 points, so no split of 3,628 values towards a hidden layer can
pay.  The hinge read from the other side (`resisc45_union_variants.py --d8
--hinge`, one constant charged per hinge column touched) has the union head
spread over 524-548 of the 768 hinges and read 0.8140 / 0.8143 at 4,160-4,176
values and 0.8092 / 0.8163 at 3,640-3,655: the raw list's numbers at 3,628.
Iteration 1's finding (the expansion's value is collective) holds under a head
whose reads are stable to 0.2 points.

## What to take from it

* Both non-pool levers are closed.  The frontier is unchanged at 81.56% /
  3,628 values (method 81.4% +/- 0.2), 80% at 2,092 values.
* The nonlinear headroom is real (3 points) but is spread over hundreds of
  small hidden weights or hinge columns; nothing a sparse head or a handful of
  units can hold.  Stop trying to buy it under the budget.
* The only lever that has moved the frontier in six iterations is the list's
  dense ceiling, and it moves through the pools: new information (pool 5) or
  nuisance removal (dihedral averaging).  Candidates left: part-configuration
  columns for church/palace and basketball/tennis, or other nuisance
  averagings (translation, scale) read first at the dense ceiling.
