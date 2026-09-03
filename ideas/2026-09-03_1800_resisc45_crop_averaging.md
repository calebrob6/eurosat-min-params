# Crop averaging lifts the RESISC45 frontier; scale averaging is a null (2026-09-03 18:00)

## Where this started

Iteration 6 closed the two non-pool levers (per-column transforms, a few
learned hidden units, hinge columns under the head) and left the union head at
81.4% +/- 0.2 test at 3,628 deployed values on the dihedral-averaged pools, with
the list's dense ceiling as the only lever that had ever moved the frontier.
Iteration 5's dihedral averaging was the cheapest ceiling lever so far (+0.8 on
the same columns, zero deployed cost), so the next candidates named were the
other nuisances an aerial scene carries: the field of view (the scene
continues past the frame) and the pixel scale.

## Three jitter families at the dihedral pools' cost

`experiments/resisc45_jitter_pools.py` averages each GPU pool over eight views
that pair the eight dihedral elements with a second nuisance, so every family
costs the same eight extractions as `d8`:

* `c224`: each dihedral view over a distinct 224 x 224 crop of the 256 x 256
  image (four corners, four mid-edges);
* `s192` / `s320`: the eight dihedral views of the whole image resized to
  192 or 320 pixels (area down, bicubic up).

The extractors take any image size (pooling factors and cell grids are
relative to the image); the layout pool works at a fixed 128-pixel resolution,
so its views are resized back to 256 first.  The lacunarity columns of the
ratio maps are the only ones that do not survive a change of view (near-zero
correlation with the cache), and they are not in any ranked list.

## At the ceiling (`resisc45_jitter_ceiling.py`, dense heads on the 768-column list)

| Views | Test | Validation |
|---|---:|---:|
| d8 (iteration 5) | 0.8325 | 0.8487 |
| c224 alone | **0.8381** | 0.8559 |
| s192 / s320 alone | 0.8254 / 0.8341 | 0.8386 / 0.8486 |
| d8 + c224 (16 views) | 0.8365 | **0.8589** |
| d8 + s192 + s320 (24 views) | 0.8340 | 0.8510 |
| d8 + c224 + s192 + s320 (32 views), same columns / re-ranked | 0.8371 / 0.8387 | 0.8583 / 0.8554 |
| d8 columns + 32-view copies (1,536) | 0.8340 | 0.8537 |
| merged 4,268 columns, d8 / 32 views | 0.8329 / 0.8386 | 0.8478 / 0.8527 |

Crop averaging is worth +0.4 to +0.6 test and +0.7 to +1.0 validation on the
same columns; scale averaging is worth nothing (192 px loses information, 320
is flat, and the 24-view scale average is inside the noise).  Per block the
gain is spread: base +0.3, fifth pool +0.3, layout and fourth pool flat.  In
the pair file the gain comes from dense/medium residential (20 -> 14),
intersection/roundabout (20 -> 15), freeway/overpass (9 -> 4) and
bridge/river; church/palace stays at 37-38 and basketball/tennis at 21-23.

## Under the union head (`resisc45_union_variants.py --pool-suffix`)

| Pools, 3,628 deployed values | Eight heads (two lists x four seed sets) | Validation |
|---|---:|---:|
| d8 (iteration 5) | 0.8144 +/- 0.16 (0.8122-0.8175) | 0.830-0.839 |
| c224 | 0.8198 +/- 0.29 (0.8159-0.8254) | 0.834-0.838 |
| j16 = mean(d8, c224) | **0.8213 +/- 0.16 (0.8187-0.8244)** | 0.836-0.842 |

The head moved by the ceiling gain, as it has after every pool-side change,
and validation moved with it.  Validation pick on j16: the 736-column
384/128/128 + 96 list, seeds 16-23, 82.02% test (validation 0.8422); best draw
82.44%.  Smaller budgets on the + 96 list (seeds 0-7 / 16-23, j16 against
d8): 1,068 values 0.792 / 0.799 against 0.786 / 0.783; 1,580 0.799 / 0.807
against 0.803 / 0.799; 2,092 0.807 / 0.810 against 0.809 / 0.813; 2,604
0.814 / 0.819 against 0.810 / 0.814; 3,116 0.825 / 0.818 against 0.816 /
0.814.  The gain is readable at 1,068 and from 2,604 up and inside the seed
spread between; the + 128 list on the crop pools reads 0.805 / 0.807 at 1,580
and 0.817 / 0.812 at 2,092, so 80% is now cleared by both seed sets at 1,580
deployed values.

## What to take from it

* Nuisance averaging generalises: the field of view is a second nuisance
  worth about half a point at zero deployed cost, and it stacks with the
  dihedral average (j16 reads at or above either alone).  Pixel scale is
  *not* a nuisance on RESISC45 -- the ground sample distance is class
  information -- so averaging over it does nothing.
* The recipe for reading a new view family is now fixed: extract it at the
  dihedral pools' cost by pairing the eight dihedral elements with the new
  nuisance, read the same 768 columns at the dense ceiling (validation
  moves with test for ceiling levers), then confirm under the union head.
* The head still recovers its list's ceiling minus 1.2-1.7 points, and the two
  big confusions (church/palace, basketball/tennis) did not move.  More crops
  (a 16-view crop family at 8 more extractions), smaller crops (192 px), or
  the crop average applied to a re-ranked list are the cheap follow-ups;
  part features for the two big pairs remain the expensive one.
