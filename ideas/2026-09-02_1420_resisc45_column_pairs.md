# The other axis is not like the class axis

*2026-09-02 14:20 — RESISC45 iteration 7*

## The idea

Iteration 6 generated the head as `Dc @ P` with `Dc` a fixed over-complete
dictionary of class directions, and halved the budget at both targets. Its
controls said the mechanism was *over-completeness*, not semantics: a complete
44-atom basis change was worth nothing, designed class groups were worth almost
nothing over random ones, and only the atom count mattered. That is a purely
mechanical claim, so it should transfer: put a dictionary on the *column* axis
too and write `W = Dc @ P @ Df.T`, where one stored value buys the rank-1
outer product `Dc[:, a] Df[:, f].T` — a class pattern times a column pattern.
Both controls are nested (`Df = I` is iteration 6, `Dc = Df = I` is the
element-wise head), so neither can lose by expressiveness.

## What happened

It works, and the frontier moves again: 65% test now clears at **208 stored
values** (was 256) and 70% at **320** (was 448), with paired-bootstrap gains of
+1.9 points [+0.7, +3.0] at 192 values, +2.7 [+1.6, +3.8] at 256 and +1.9
[+0.9, +2.8] at 512.

But the mechanism is *not* the one iteration 6 found, and the width sweep is
what says so. At 256 stored values, 16,896 column atoms of width 4, 8 and 32
read 0.6560, 0.6429 and 0.6417 test, and 16,896 dense random directions read
0.6557, against **0.6519 for the raw columns**. Every one of those is a
wash or a loss. Only width 2 pays — 0.6790 for an exhaustive enumeration of
signed pairs over the top 256 ranked columns.

So on the class axis any over-complete draw worked and only the count mattered;
on the column axis the count buys nothing and only one specific structure does.
The asymmetry has an obvious explanation once stated: the head's class pattern
for a column is *dense* — every class needs some weight on a column that
matters — while its column pattern for a class is *sparse*, which is why
feature selection works at all. A wide atom is a bargain when the target is
dense and a tax when it is sparse, because it charges a class for columns it
does not want. A pair is the smallest atom that is still a choice: it ties
`|w_i| = |w_j|` and picks the relative sign, which is one stored value for two
weights whenever a class wants a sum or a contrast of two features.

## What did not work

* **Width 4 and up**, monotonically decaying to the dense limit — above.
* **Sampling instead of enumerating.** 16,384 sampled pairs read 0.6681 at 256
  values and 65,536 sampled pairs read 0.6711, against 0.6757 for a 33,092-atom
  exhaustive enumeration over the top 181 columns. Being able to reach the best
  pair is worth about as much as quadrupling a random draw.
* **Bigger enumerations.** `pairs320` (102,592 atoms) matches `pairs256` at 256
  values and is 0.9-1.1 points worse at 512 and 1,024. The support search is
  over `catoms x fatoms`, and 16,384 x 102,592 candidate entries overfit 18,900
  training images.
* **Anything at all above 640 stored values.** The bootstrap against the
  class-dictionary head is +1.7 points [+0.8, +2.7] at 640 but +0.3
  [-0.6, +1.2] at 768 and -0.2 [-1.0, +0.7] at 1,024. The 0.7659 in the
  frontier table is a validation gate picking `pairs128` by +0.6 points
  [-0.2, +1.5] on test — inside the split's resolution — and the 896-value row
  is the same effect with the opposite sign, landing *below* the 768-value row.
  Both are reported as such.

## A free 16x on every future run

Carrying the code as `nnz` live values indexed into `catoms x fatoms`, instead
of as a dense `atoms x k` tensor, makes a fit that took 41 s in iteration 6's
code take **2.5 s**, with a bit-identical support. The old form computes
`(xs @ P.T) @ Dc.T`, which is 158 GFLOP per epoch at 16,384 atoms; forming
`W = Dc @ P` first is 200x fewer flops, and only materialising the `nnz` live
entries removes the optimiser's 0.5 GiB of dense state as well. Every remaining
RESISC45 experiment on this head is now cheap enough to run the whole grid
rather than a sample of it.

## What this suggests next

* Both dictionaries are now *fixed* and the search chooses among them. The
  thing that has never been tried is letting the pair enumeration be
  **conditional**: `pairsT` pairs the top `T` columns by a label-only
  group-lasso rank, which is the same criterion that sank iteration 3's merged
  ranking. Ranking pairs by what they add *given* the support already chosen is
  the natural next step, and iteration 6's idea file asked for the same thing on
  the column list.
* The per-class picture still has not been looked at since iteration 2, and the
  head has roughly tripled its effective weights since. bridge, ship and
  railway_station were the classes the budget starved; if they are still
  starved, the object-layout pool deserves a second look, and if they are not,
  the remaining 4-point gap to the pool ceiling is somewhere new.
* At 208 stored values the head reads 136 pool columns. The *extractor* is now
  a much larger deployment artefact than the head, and nothing in this run has
  optimised it.
