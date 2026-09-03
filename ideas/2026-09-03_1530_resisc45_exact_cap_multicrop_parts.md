# Exact-cap, multicrop, and part-configuration RESISC45 follow-up

## Starting point

The crop-plus-dihedral `j16` union head averaged 82.19% test at 3,628 deployed values on its validation-preferred `384/128/128 + 96` list. The next objective was maximum accuracy under 4,096 honestly counted deployed values, with unrestricted deterministic views.

## Spending the remaining deployment budget

Four disjoint search-offset groups were run at 3,884 and exactly 4,096 deployed values on both current quota lists. The exact cap is not optimal:

| Deployed values | Validation-selected list/view | Mean validation | Mean test | Test standard deviation |
|---:|---|---:|---:|---:|
| 3,628 | `j16`, own-ranked `+96` | 0.83960 | 0.82188 | 0.00182 |
| 3,884 | `j32crop`, d8-ranked `+128` | **0.84525** | **0.82438** | 0.00227 |
| 4,096 | `j32crop`, d8-ranked `+128` | 0.84270 | 0.82390 | 0.00170 |

The best individual test draw was 82.87% from the j16 `+128` list at 3,884 values, but that list loses on mean validation and is not the selected model. Adding the final 212 values lowers both mean validation and mean test, so the operating point is 3,884 rather than the hard cap.

## Progressive crop averaging

`resisc45_jitter_pools.py` now accepts a cyclic crop-offset shift, allowing complementary eight-view crop/dihedral pairings to be added progressively. Each pairing covers the same eight 224-pixel crop locations with a different assignment to the eight dihedral transforms. Their cached means were combined with the full-frame d8 mean:

| Cached family | Effective views | Same-column validation | Same-column test |
|---|---:|---:|---:|
| `d8` | 8 | 0.8487 | 0.8325 |
| `j16` | 16 | 0.8589 | 0.8365 |
| `crop16` | 16 crops | 0.8592 | 0.8421 |
| `j24` | 24 | 0.8597 | 0.8402 |
| `j32crop` | 32 | **0.8643** | 0.8410 |
| `j40crop` | 40 | 0.8627 | 0.8406 |
| `j48crop` | 48 | 0.8629 | 0.8419 |

Validation rises through 32 views and declines for both larger means, satisfying the stopping rule. Re-ranking on the averaged values is consistently worse on validation; the winning configuration uses j32crop feature values with the original d8 column rankings.

Under the repeated union head, the j32crop ceiling gain mostly becomes validation margin rather than test gain. At 3,884 values the d8-ranked j32crop `+128` list reads 84.53% +/- 0.14 validation and 82.44% +/- 0.23 test, essentially tied on test with j16 at the same budget. At 4,096 values it falls to 84.27% validation and 82.39% test. More view averaging is therefore closed as a head-side accuracy lever even though it improves the dense ceiling.

## Cross-part configuration pool

`resisc45_gpu_features7.py` adds 966 fixed columns describing relationships among eleven color masks plus edge, bright-marking, and dark-line masks. It measures individual mask radius/spread/anisotropy and every unordered pair's containment, IoU, centroid distance, radial order, principal-axis agreement, coarse layout correlation, and adjacency at three radii. The pool is extracted under the winning j32crop view average.

| Dense arm | Columns | Validation | Test |
|---|---:|---:|---:|
| j32crop baseline, d8-ranked | 768 | 0.8643 | 0.8410 |
| + top 64 part columns | 832 | 0.8633 | 0.8437 |
| + top 96 part columns | 864 | 0.8644 | 0.8452 |
| + top 128 part columns | 896 | **0.8649** | **0.8454** |
| + top 192 part columns | 960 | 0.8624 | 0.8424 |
| + all part columns | 1,734 | 0.8557 | 0.8378 |
| part pool alone | 966 | 0.7024 | 0.6922 |

The top 128 columns improve validation by only 0.06 points, below the repeated-head noise floor, and the full pool overfits. Basketball/tennis errors fall from 24 to 21, but church/palace remains at 33. The pool therefore fails the predeclared validation gate and is not propagated into the expensive union head.

## Result

The validation-selected frontier under 4,096 deployed values is now 82.44% +/- 0.23 test at 3,884 values, up about 0.25 points from the 3,628-value j16 baseline. The exact 4,096-value model is slightly worse. Progressive translation averaging improves the dense ceiling but stops transferring through the sparse head, and the targeted part-configuration pool does not provide a readable validation gain. Previously closed residual hidden-unit, hinge, transform, and search variants remain closed.

Raw repeated rows and the validation selections are consolidated by `experiments/resisc45_exact_cap.py` into `experiments/resisc45_exact_cap_result.csv`.
