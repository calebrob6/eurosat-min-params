# Ideas & findings — 2026-07-01 01:38 (iteration 3)

## TL;DR
- **Landed submission 03: 510 params, test 0.9446** (linear on 195-feature pool,
  k=50 selected by train-only CV). A 23% param cut from submission 02 (660) AND
  higher test accuracy. The win came from **one new parameter-free feature
  family: structure-tensor coherence** (directional-texture), which specifically
  helps the weakest class (Highway).
- **Two paths ruled out this iteration** (both are dead ends — don't repeat):
  1. **Reduced-rank linear head** (factor the k×K weight as (k×r)(r×K)).
  2. **De-noised CV selection *on the old features*** — it confirms 660 is the
     honest floor there, it does NOT get you lower by itself.

## WIN: structure-tensor coherence features
Magnitude-only texture (gradient mean/std) says how *strong* local texture is,
not its *orientation*. Coherence = √((Sxx−Syy)²+4Sxy²)/(Sxx+Syy) from the
per-band structure tensor is 0 for isotropic texture, →1 for one dominant
direction — i.e. it flags **linear structure** (roads). Added at 2 scales × 13
bands = 26 zero-parameter features (`src/features.coherence_features`, exposed
via `patch_features(..., coherence_scales=2)`).

Frontier on the augmented 195-pool (selection by 5×5-fold **train** CV only):

| k  | params | train-CV | val    | test   |
|----|--------|----------|--------|--------|
| 45 | 460    | 0.9383   | 0.9354 | 0.9415 |
| 50 | 510    | 0.9404   | 0.9404 | 0.9446 | ← submission 03 (smallest k, CV≥0.940) |
| 55 | 560    | 0.9432   | 0.9420 | 0.9480 |
| 60 | 610    | 0.9443   | 0.9431 | 0.9485 |
| 65 | 660    | 0.9453   | 0.9430 | 0.9506 |

- Coherence features rank into the top-50; the gain is **spread across classes**
  (test/CV up at every k), not a big single-class jump. At the shipped k=50 the
  weakest classes are still Highway (0.819) and PermanentCrop (0.896) — coherence
  helps Highway more at higher k, but at k=50 the net gain is distributed.
- Note k=65 augmented reaches **test ~0.949–0.951 at 660 params** — matching/
  beating submission 01 (0.9502 at 1010). Coherence helps at every k, we just
  spend the gain on fewer params.

## NEGATIVE 1: reduced-rank linear head does not help
Idea: the k×K logit weight matrix has effective rank ≤ K−1 = 9 (softmax is
shift-invariant across classes), so factor it as (k×r)(r×K) → params
`k·r + r·K + K`, hoping r < 9 saves parameters. Implemented two ways in
`src/lowrank.py`:
- **Torch (CE-optimal-ish, Adam+wd)**: underoptimised, rank-9 only hit val 0.921
  (should equal full linear 0.949). Not trustworthy without more tuning.
- **SVD-truncation of a well-optimised full linear solution** (the clean,
  deterministic reference): at r=9 it **exactly recovers full linear** (as
  theory predicts), but dropping even ONE dimension collapses it —
  r=8→test 0.80, r=7→0.69, r=6→0.55. **All 9 discriminant dimensions are
  needed**; the class structure on these features is essentially full-rank.

Consequence: rank-9 (the only loss-less rank) costs `k·9 + 100`, which only
beats full linear's `k·10 + 10` when k > 90 — i.e. never in the low-param regime
we care about (k≈50). **Reduced-rank buys nothing here.** (The torch-vs-SVD gap
at r=8, 0.93 vs 0.80, shows SVD-truncation is not accuracy-optimal, but even a
perfectly-optimised rank-8 can't beat 660 in our param budget.)

## NEGATIVE 2: CV selection alone doesn't beat 660 on the *old* features
A single-seed 5-fold CV once showed k=55 at CV 0.941 (looked like a 560-param
win), but that was **fold luck**. Averaging 5 shuffles:
`k=50→0.9370, k=55→0.9393, k=60→0.9399, k=65→~0.942` — the de-noised CV plateaus
*just under* 0.940 at k=55–60, so "smallest k with CV≥0.940" lands at **k=65
(660)**, matching submission 02. So val-selection wasn't merely unlucky; the old
169 features genuinely need ~65 to generalise at 0.940. The methodological
lesson (use multi-seed CV, not single-seed and not raw val) is real and now
baked into submission 03's `train.py`, but it only pays off *combined with the
better features*.

## Why test runs ~0.5% above val/CV (still true)
val (5400) and train-CV both sit ~0.5% below test at every k. That gap is a real
split-difficulty difference, but it's unusable for selection (measuring it needs
test). So the honest floor is wherever CV/val crosses 0.940, and better features
are the only lever that moves it.

## Ranked ideas for next iterations
1. **More/better orientation features (cheapest next win).** Coherence at 2
   scales already helped; try (a) a 3rd scale, (b) HOG-style oriented-gradient
   energy in a few bins on 1–2 informative channels, (c) coherence only on
   informative bands to avoid diluting the L1 ranking. Target: k=45 clearing
   CV≥0.940 → **460 params**.
2. **Attack PermanentCrop (now the weakest, ~0.90).** It overlaps AnnualCrop/
   Herbaceous spectrally. A feature capturing crop-row periodicity (local FFT
   peak energy, parameter-free) might separate managed crops from grassland.
3. **Cheaper spectral half.** 65 of 169 base features are percentiles; many are
   redundant with mean/std. Prune the feature *generators* (not just L1-select)
   so the pool is smaller and selection less noisy.
4. **Raise the bar to 95–96% and use the idle GPUs.** k=65 augmented already
   hits 0.9506 at 660 params. For >95% at *low* params, tiny-CNN-from-scratch or
   distillation on the 8× free V100s is the route (conv filters count as params;
   a depthwise-separable global-pool net may still be competitive). This is the
   big open direction — all 8 GPUs were idle again this iteration.

## Infra added this iteration
- `src/features.coherence_features` + `patch_features(coherence_scales=…)`.
- `src/lowrank.py` — reduced-rank linear (torch + SVD-truncation) with folding
  and param counting. **Kept for reference even though reduced-rank lost**, since
  the SVD-truncation is a clean tool and the fold math is reusable.
- `experiments/`: `lowrank_sweep.py`, `lowrank_svd_sweep.py`, `cv_select.py`,
  `coherence_test.py` (all reproduce the numbers above).
- Augmented feature cache: `data/cache/*_feat_coh.npy` (195-dim).
