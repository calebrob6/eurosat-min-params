# Dihedral averaging of the RESISC45 pools, and a null sixth pool (2026-09-03 14:00)

## Where this started

Iteration 4 left the union-of-supports head at 80.9% +/- 0.3 test at 3,628
deployed values on the 736-column list (best 81.30%), the head recovering its
list's dense ceiling (82.3-82.5%) minus 1.2-1.4 points, and named the next
lever: part-configuration features or learned hidden units for the two big
confusions (church/palace, basketball/tennis court).

## The sixth pool (null)

`experiments/resisc45_gpu_features6.py` is the cheapest zero-parameter
part-configuration extractor: seeded He-scaled random ReLU networks three
layers deep (5x5x3 -> 32, 3x3 -> 64, 3x3 -> 128, max-pool 2 between layers)
on the image average-pooled by 2 and by 4, global mean and max of layers 2 and
3, 768 columns in ten GPU-seconds.  Random deep features carry much of the
value of trained ones in the literature (Saxe et al. 2011), and the pool alone
reads 58.5% test against 55.1% for the fourth pool's single-layer random
convolutions with twice the columns.  On the 736-column list it is worth
nothing (`resisc45_pool6_ceiling.py`: 0.8224-0.8265 against 0.8230 for every
family, scale, layer and top-N subset), and under the union head with a fifth
quota block four heads read 0.8104 +/- 0.19 against 0.8091 +/- 0.29.  The two
big confusions do not move.  Random compositions of random patterns are not
the missing information.

## Dihedral averaging (real)

Aerial imagery has no canonical orientation, yet many columns are not
rotation-invariant: gradient-orientation histograms relative to the axes, line
orientation, sorted grid cells, Radon and ridge layout, and every random
convolution.  `experiments/resisc45_dihedral_pools.py` re-runs each GPU pool's
`extract` on the eight rotations and flips of the image (numpy views over the
cached uint8 array, so no memory cost) and stores the mean as
`resisc45_{split}_gpu{,2,3,4,5}d8_pool.npy`; the identity view reproduces the
cached pool to 3e-5; about 25 GPU-minutes for all five pools.  The 147-column
CPU pool is left alone.  Nothing in the deployed head changes: this is
test-time augmentation of a fixed extractor, at eight times the extraction cost.

`experiments/resisc45_dihedral_ceiling.py`, dense heads with `C` on validation:

* the same 736 columns read 82.30% -> 83.13% test (validation 84.1 -> 84.7),
  with *lower* train accuracy (0.955 -> 0.951): a nuisance was removed;
* re-ranking on the averaged pools gives the same answer (83.06%) with 90-100%
  of the top columns shared per block;
* appending the averaged copies to the originals reads 83.14%, no better than
  replacing them, so the orientation-dependent part carried no signal;
* per block: fourth pool (random convolutions, cells) +0.9 alone, base +0.5,
  layout and fifth pool flat;
* merged 4,268 columns 81.62% -> 83.29%.

Under the union head (`resisc45_union_variants.py --d8`), eight heads on the
+96 and +128 lists with four disjoint search sets read 0.8144 +/- 0.17 points
at 3,628 deployed values (0.8122-0.8175) against 0.8091 +/- 0.29 on the
original pools, validation 0.830-0.839 against 0.824-0.830.  Validation pick:
+128 list, seeds 32-39, 81.56% test; best 81.75%.  Smaller budgets gain
0.5-1.0 points each: 80% cleared by both seed sets at 2,092 values and by one
of two at 1,580; 81% cleared by both at 2,604.

## What to take from it

* Any pool column that depends on image orientation should be averaged over
  the dihedral group before it is ranked or searched; the gain is readable on
  validation and costs no deployed value.  Future pools should be extracted
  averaged from the start (`resisc45_dihedral_pools.py --pools <name>` after
  adding the module to `MODULES`).
* The gap between the union head and its list's dense ceiling is still
  1.2-1.7 points, so the ceiling is still the lever.
* Church/palace reads 33 errors under the averaged same-column dense head,
  the first arm below 35, but the dense pair counts move by a few errors
  between arms; the pair is still 0.5-0.7 points on its own.
* Random deep features are exhausted as a pool idea.  What remains for the two
  big pairs is hand-designed part features (a steeple and its shadow, a
  cross-shaped roof, the key and centre circle of a court) or a head with a
  few learned hidden units, both open.
