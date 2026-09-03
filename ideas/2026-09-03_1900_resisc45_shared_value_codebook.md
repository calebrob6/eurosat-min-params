# Shared-value codebooks collapse the RESISC45 learned-value frontier

## Question

The 3,884-value sparse dictionary head stores one independent float for every active class-atom/feature entry. Its learned support pattern already costs more bits than its values. If those entries naturally reuse a small set of magnitudes, a learned scalar codebook can separate the number of learned floating-point values from the amount of learned discrete structure.

## Model

The experiment uses the validation-winning 768-column j32crop list with d8-derived column rankings and the fixed 16,384-atom Gaussian class dictionary. Eight perturbed RigL searches are unioned and pruned to a fixed support. Feature standard deviations are rounded to powers of two, so their data-derived exponents are discrete ids rather than learned floating-point scales. Every support entry stores a class-atom id, candidate-column id, and codebook-level id. The 768-column candidate mapping, signed sigma-exponent origin/deltas, and any bias-atom ids are charged in structure bits.

A Q-level feature codebook restricts all support entries to Q learned scalar values. The raw 44-class reference intercept is either stored directly or reconstructed from q learned coefficients over the same fixed Gaussian class dictionary. Codebook assignments and bias atoms are initialized from a continuous convex refit, then the shared values are jointly refit with LBFGS. Reconstruction uses deterministic one-hot matrix products; repeated runs reproduce identical supports, assignments, atoms, and accuracies.

## Compact learned-value frontier

Four disjoint support-search offsets were run with an 8,192-entry support and a validation grid `C={0.001,0.003,0.01,0.03,0.1}`. The table reports the validation-selected C at each learned-value count:

| Learned values | Feature levels | Bias atoms | Mean validation | Mean test | Test standard deviation | Test range | Total model bits |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 2 | 16 | 0.8024 | 0.7852 | 0.0024 | 0.7825-0.7878 | 216,955 |
| 19 | 3 | 16 | 0.8187 | **0.8021** | 0.0035 | 0.7971-0.8052 | 225,179 |
| 20 | 3 | 17 | 0.8235 | **0.8066** | 0.0070 | 0.8003-0.8160 | 225,225 |
| 21 | 3 | 18 | 0.8279 | 0.8119 | 0.0049 | 0.8071-0.8173 | 225,271 |
| 22 | 4 | 18 | 0.8368 | 0.8200 | 0.0022 | 0.8183-0.8227 | 225,303 |
| 24 | 4 | 20 | 0.8420 | 0.8225 | 0.0030 | 0.8190-0.8254 | 225,395 |
| 28 | 4 | 24 | 0.8435 | 0.8256 | 0.0013 | 0.8237-0.8267 | 225,579 |
| 30 | 6 | 24 | 0.8459 | 0.8282 | 0.0023 | 0.8251-0.8306 | 233,835 |
| 32 | 8 | 24 | 0.8471 | **0.8292** | 0.0009 | 0.8281-0.8303 | 233,899 |

Nineteen learned values are the first validation-selected mean above 80%, but one support reads 79.71%. Twenty values are the first point at which all four support-search offsets clear 80%. The floor is sharp: 18 values average 78.52%.

## Maximum-accuracy point

Increasing only the discrete support improves accuracy without increasing Q. A 32,768-entry support beats 8,192 on validation; 65,536 entries do not. With a direct 44-value raw intercept:

| Learned values | Feature levels | Support | Mean validation | Mean test | Test standard deviation | Total model bits |
|---:|---:|---:|---:|---:|---:|---:|
| 76 | 32 | 32,768 | 0.8530 | 0.8365 | 0.0013 | 964,216 |
| **108** | **64** | **32,768** | **0.8536** | **0.8369** | 0.0015 | 998,008 |
| 172 | 128 | 32,768 | 0.8530 | 0.8365 | 0.0012 | 1,032,824 |

The 108-value model is within approximately 0.4 points of the 84.1% dense ceiling on the same list. More levels and a 65,536-entry support do not improve validation.

## Values versus bits

This is a learned-value frontier, not a storage-free model. The previous 3,884-value head occupies approximately 224,256 logical bits under the same fixed-width accounting: 124,288 value bits, 92,160 support-index bits, and 7,808 candidate-mapping bits. The 19-value codebook model occupies about 225,179 bits, so it uses roughly the same total storage while replacing thousands of independent floats with discrete assignments. The 32-value model occupies about 233,899 bits and gains roughly half a test point. The 108-value maximum-accuracy model occupies about 998,008 bits, around 122 KiB, because its 32,768 assignments dominate storage.

The result is nevertheless meaningful for the repository's stated metric: learned scalar values fall from 3,884 to 20 at the repeatable 80% threshold, and from 3,884 to 108 while accuracy rises from 82.44% to 83.69%. Assignment bits, physical artifact sizes, and complete decode metadata are reported beside every result.

## Reproducibility

`experiments/resisc45_codebook.py` performs support search, continuous refit, feature-level quantization, class-dictionary bias coding, strict four-offset aggregation, complete logical bit accounting, and artifact persistence. `experiments/resisc45_codebook_eval.py` reconstructs candidate mappings, power-of-two scales, support assignments, class atoms, codebook levels, and raw intercepts from each checked-in artifact and reproduces validation/test predictions. The selected artifacts live under `experiments/resisc45_codebook_models/`.
