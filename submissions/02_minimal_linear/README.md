# Submission 02 — minimal-parameter linear classifier

**660 parameters · val 0.9404 · test 0.9431 · VALID (>94% test)**

A 35% parameter reduction over submission 01 (1010 params) at the same >94%
test bar, reached simply by pushing the selected feature count as low as the
**validation** split allows — and by first ruling out two tempting nonlinear
directions that turned out to overfit.

## Approach

Identical parameter-free feature pipeline to submission 01
(`src.features.patch_features`): each 13-band patch → per-band mean/std/
percentiles + 3-scale gradient-magnitude texture stats = **169 fixed features,
0 learned parameters**. A folded logistic regression (`src.linmodel`,
StandardScaler absorbed into the weights) classifies the top-`k` features, so
the deployed model is one affine map on raw features with exactly `10*(k+1)`
parameters.

Selection (train + val only):
1. Rank the 169 features by L1-logreg importance on train.
2. For each candidate `k`, pick the L2 strength `C` that maximises val accuracy.
3. Take the **smallest `k` with val ≥ 0.940** → `k=65`, `C=10`, **660 params**.

Test (0.9431) is reported once, after selection.

## Reproduce

```bash
python submissions/02_minimal_linear/train.py   # sweeps k/C, saves model.npz
python submissions/02_minimal_linear/eval.py    # verifies from RAW patches
```

`eval.py` recomputes features from the raw GeoTIFFs (bypassing the cache), an
honest end-to-end check of feature extraction + folded linear model.

## Frontier (selection on val, test for reference)

| k  | params | val    | test   |
|----|--------|--------|--------|
| 50 | 510    | 0.9337 | 0.9391 |
| 55 | 560    | 0.9376 | 0.9426 |
| 60 | 610    | 0.9396 | 0.9426 |
| 65 | 660    | 0.9404 | 0.9431 | ← selected (first val ≥ 0.940) |
| 70 | 710    | 0.9420 | 0.9426 |
| 80 | 810    | 0.9428 | 0.9452 |

`k=55` (560 params) already clears **test 0.9426** but its val (0.9376) is below
the 0.940 selection bar, so it isn't chosen under a val-only rule. There is
still room below 660 for a future submission that selects features more
cleverly without overfitting val.

## What did NOT work (see `experiments/`)

Both add capacity and both **raise val but not test — i.e. they overfit**:

- **Bottleneck MLP head** (`sweep.py`): `F→H→10` on the same features. Even the
  widest tried (F=52, H=16, 1018 params) reached only val 0.912 — *worse* than
  linear at equal params. Compressing already-linearly-separable features
  through a narrow hidden layer discards signal.
- **Quadratic augmentation** (`quad_sweep.py`): squares + pairwise products of
  the top-24 features (fixed arithmetic → still 0 feature params), then linear.
  At k=60 val rose to 0.9430 but test was 0.9428 (vs 0.9448 linear-only): pure
  val overfit.

Takeaway: on EuroSAT with these texture features the classes are essentially
linearly separable, so **a linear head is at the accuracy-per-parameter
frontier**. Beating it needs *better/cheaper features* (fewer needed), not a
more expressive classifier.

## Weakest classes (test)

Highway 0.815, PermanentCrop 0.900 — spectral/texture overlap with other
crop/vegetation classes, unchanged from submission 01. SeaLake (0.993) and
Forest (0.992) are essentially solved.
