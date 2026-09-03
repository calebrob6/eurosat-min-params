# A fifth RESISC45 pool: line geometry and region shape (2026-09-03 12:00)

## Where this started

The union-of-supports head sat at 80.0% +/- 0.2 test at 3,628 deployed values
against its 640-column list's 81.0% linear ceiling, and every search-side knob
had been read under the head itself and found inside the noise
(`2026-09-03_1000_resisc45_union_plateau.md`).  The head recovers the ceiling
minus about a point, so the only lever left was the ceiling: a list with more
information in it.

`experiments/resisc45_union_failures_pairs.csv` lists the pairs the head still
confuses, and the striking thing is that the dense 4,126-column head and a
width-1,024 MLP on the same columns confuse the *same* pairs by about the same
counts (church/palace 35-45, basketball/tennis court 24-35, lake/river 16-17,
farmland/terrace 16-18, railway/railway station 16-21).  That says the
information is absent from the pool, not hidden from the head by the budget.

## What the pool measures

Every family in pools 1-4 is a global statistic of intensity, texture,
oriented energy, Radon/ridge/blob layout, random convolutions or colour-masked
texture.  None measures whether lines are *curved*, what colour lies *beside*
bright markings, or the *shape* of a colour region.  Those are the distinctions
the top pairs turn on, so `experiments/resisc45_gpu_features5.py` reads:

* `line` (42 columns): white and black top-hat of the panchromatic map
  thresholded into thin bright and thin dark line pixels; fraction, a
  rotation-normalised 8-bin orientation histogram, its entropy, rectilinearity
  (dominant plus orthogonal bins), structure-tensor coherence, and curvature
  as the doubled-angle orientation change 4 and 8 pixels along the local
  tangent (zero on straight and on crossing lines, positive on arcs); for the
  bright lines also the hue/grey composition and map means of a 4-pixel ring
  around them.
* `curv` (30 columns): the same entropy, rectilinearity, coherence and
  curvature on the top-20% edges of the panchromatic and excess-green maps at
  scales 1, 2 and 4.
* `cc` (70 columns): scipy connected components of seven fixed masks (water,
  vegetation, tan soil, bright, dark, saturated red, bright lines) at 128 x
  128; count, largest area, elongation, box fill, border sides touched,
  boundary/area, centroid offset, second-largest area, area entropy.

142 columns; about 40 GPU-seconds plus a CPU minute for the components.

## What happened

* Ceiling (`resisc45_pool5_ceiling.py`): the 640-column quota list's dense
  ceiling moves from 81.02% to 82.46% test (validation 83.4 to 84.5) with the
  pool appended, the largest per-column gain of any pool; the merged
  4,126-column ceiling moves 80.6 to 81.6.  `line` +0.5 alone, `curv` +0.6
  alone, `cc` 0.0 alone, all three +1.4; the top 96 by group lasso +1.3.
* Attribution (`resisc45_pool5_ceiling_pairs.csv`): the gain is *not* on the
  target pairs.  Church/palace, basketball/tennis, lake/river and
  farmland/terrace move by at most three errors.  Baseball diamond/basketball
  court (14 to 8), industrial area/storage tank (12 to 6), freeway/railway (10
  to 4), circular/rectangular farmland (10 to 5), airport/river (11 to 6) and
  medium residential/tennis court (24 to 19) carry it.
* Union head (`resisc45_union_variants.py --quotas q384/128/128/N`, a fourth
  quota block): at 3,628 deployed values eight heads on four lists and two
  disjoint search sets read 0.8091 +/- 0.29 points (0.8044-0.8130), every one
  above the incumbent's 0.8011 mean, seven above its 0.8049 best draw, and
  validation moves with test (0.824-0.830 against 0.818-0.823).  Validation
  pick: 384/128/128 + 96, seeds 16-23, **81.30% test at 3,628 values**.  At
  2,604 values all four rows read 0.8032-0.8049; at 2,092 three of four read
  above 0.80.  The 640-column 320/128/128 + 64 list also beats every incumbent
  row, so the gain is the columns, not the width.

## What this says

* A new pool is worth trying when the dense head, the MLP and the budgeted
  head all fail on the same pairs; when only the budgeted head fails, it is a
  search problem.  This was the first time in the RESISC45 work that the three
  agreed, and the pool paid.
* Design features for specific pairs, but read the gain over all pairs.  The
  curvature and ring-colour columns separate their pairs at the class-mean
  level (0.5-1.2 standard deviations) and still do not fix them under a
  linear head; the same columns quietly give back a third to a half of the
  errors on six pairs nobody designed for.
* The union head keeps recovering the ceiling minus 1.2-1.4 points whatever
  the list, so ceiling gains transfer almost whole and the ceiling remains
  the lever.  Church/palace and basketball/tennis court are 70 test errors
  (1.1 points) and have resisted every zero-parameter column; the next pool
  should probably look at something that is not a global statistic at all
  (part configuration, e.g. a spire or dome next to a roof, or the count and
  aspect of marking rectangles), or the head should get a few learned hidden
  units for those two pairs specifically.
