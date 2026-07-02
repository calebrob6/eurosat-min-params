# MOSAIKS: strong ACCURACY (95.8%) but NOT parameter-competitive (needs ~2-5k params for 94%)

*2026-07-02 14:10 — iteration 18*

Tested MOSAIKS (Rolf 2021 random convolutional features) — empirical & random
filters + a linear head and a tiny MLP — on EuroSAT. `experiments/mosaiks.py`
(featurizer + heads), `experiments/mosaiks_floor.py` (accuracy-vs-params curve),
8×V100. Filters are p=3, K of them, ±ReLU after conv + GAP → 2K features; input
band standardisation folds into the fixed conv (0 params).

## Answer to "can we get over 94% / 95%?" — YES (accuracy), NO (params)

**Accuracy ceiling (all features, linear head) — CORRECTED via torchgeo RCF.**
Two implementations were run: a hand-rolled one (`mosaiks.py`) and torchgeo's
canonical `torchgeo.models.RCF` (`mosaiks_torchgeo.py`, k=3, 13-band standardised
input). torchgeo is authoritative; use its numbers:

| filters | K | features | extractor params | head params | test (torchgeo) |
|---------|--:|---------:|------------------:|------------:|-----:|
| gaussian (random) | 512 | 1024 | ~0 (seed) | 10250 | 0.9491 |
| gaussian (random) | 2048 | 4096 | ~0 (seed) | 40970 | 0.9563 |
| empirical (ZCA) | 512 | 1024 | 59904 stored | 10250 | **0.9637** |
| empirical (ZCA) | 2048 | 4096 | 239616 stored | 40970 | **0.9652** |

- **EMPIRICAL (ZCA-whitened) BEATS gaussian** (0.965 vs 0.956). A first hand-rolled
  pass WRONGLY found the reverse (random > empirical, empirical only 0.89-0.93) —
  that was a BUG: it skipped the **ZCA whitening** of empirical patches that the
  MOSAIKS paper / torchgeo `RCF._normalize` do. With ZCA, real patches are the
  better filters, and empirical saturates faster (K=512 already 0.964). The
  hand-rolled *gaussian* numbers were ~correct (torchgeo's raw-N(0,1)+bias=-1 vs the
  hand-rolled unit-norm+bias=0 made little difference). **Lesson: use the reference
  implementation for a known method before drawing conclusions.**
- Gaussian is seed-free (~0 extractor params); **empirical must STORE K·C·k·k filter
  values** (59904 @ K=512, 239616 @ K=2048) — so empirical wins accuracy but is
  hopeless on parameters.
- **The tiny MLP does NOT help** (hand-rolled test): at every K the linear head wins
  (K=2048 gaussian: linear 0.958 vs MLP-H32 0.951). MOSAIKS' power is the many
  random projections, not head capacity — a deeper head just overfits.

## Why it is NOT parameter-competitive (the project's actual objective)

Head params = 10·(k+1) for k KEPT features. L1-top-k accuracy-vs-params curve
(random, K=1024 → 2048 feats):

| features k | params | test |
|--:|--:|--:|
| **18** | **190** | **0.820** |
| 64 | 650 | 0.909 |
| 96 | 970 | 0.930 |
| 128 | 1290 | 0.934 |
| 192 | 1930 | 0.940 (verCV 0.937 — not yet honest-94%) |
| ~2048 | ~20k | 0.958 |

**At the frontier's own 190-param budget, random MOSAIKS = 0.820 test vs the hand-
crafted linear model's 0.9433.** MOSAIKS needs ~256–512 features (~2500–5000 head
params) to honestly clear 94%.

**Direct small-K check (torchgeo RCF, "count only learned params" convention).** A
K-filter RCF gives 2K features (±ReLU), so the LEARNED head is 10·(2K+1) — NOT K+1
(EuroSAT is 10-class). K=50 → 100 feats → **1010** learned params, best (empirical)
**test 0.9254**; K=25 → 50 feats → 510 params, best 0.8919 — both below 94% and
already >2.7× the 190-param frontier's params at LOWER accuracy (0.9433). Confirms
RCF loses on parameters even when the random filters are counted as free: the
10-class head scales with feature count and RCF needs ~200 feats for 94%. Each random projection is individually weak; you
need hundreds to aggregate the signal that ~18 purpose-built features (coherence,
Hough line, Harris corner, index-map texture) already carry. Hand-crafted features
are ~10–25× more parameter-efficient here.

## Takeaway

- **Do NOT pursue MOSAIKS for the min-parameter @94% objective** — it is an order
  of magnitude off the 190-param frontier. Gaussian is seed-free but needs hundreds
  of head features; ZCA-empirical is more accurate but its filters cost tens of
  thousands of stored params. The MLP head doesn't help either.
- **MOSAIKS is the accuracy ceiling for cheap fixed features:** ZCA-empirical
  **0.965** (K≥512), gaussian **0.956** (seed-free). If the objective ever escalates
  to "min params @ 95% or 96%" (the brief's clause), gaussian RCF is the seed-free
  featurizer to revisit for the high-accuracy regime (still param-heavy: ~4k feats
  → ~40k head params for 0.956; a selected subset for 95% is untested).
- Possible (low-priority) follow-up: pre-screen a few top random-MOSAIKS features
  as a NEW family against the frozen 190-param subset (the standard `fast_probe.py`
  qualifier). Unlikely to lift val — random feats at k=18 are 12 pts weaker than
  the hand-crafted set — but it is the only MOSAIKS angle that could touch the
  frontier. Frontier stays submission 11 = 190 params, test 0.9433.
