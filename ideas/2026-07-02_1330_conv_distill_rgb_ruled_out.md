# Tiny conv+GAP, distillation, and RGB-only all RULED OUT (none beat 190 params)

*2026-07-02 13:30 — iteration 17 (negative results)*

Tested last night's queued ideas (conv+GAP+logreg, distillation) plus an RGB-only
variant to save parameters. **All lose decisively to the 190-param linear frontier
(submission 11, test 0.9433).** Harness: `experiments/conv_gap.py` (torch, 8×V100),
60–80 epochs, Adam+cosine, flip/rot90 aug, epoch chosen by val / test reported once.
BN + fixed per-band input standardisation fold into conv1 at deploy (affine, like
the StandardScaler folding), so deploy params = `F*(9*C_in + 11) + 10`.

## Single conv -> GAP -> logreg (idea C): architecturally too weak

| bands | F | deploy params | test |
|-------|--:|--------------:|-----:|
| RGB | 4 | 162 | 0.639 |
| RGB | 8 | 314 | 0.711 |
| RGB | 16 | 618 | 0.738 |
| RGB | 24 | 922 | 0.754 |
| all13 | 8 | 1034 | 0.835 |
| all13 | 16 | 2058 | **0.892** (best conv; 10.8× the params, still <94%) |

**Why:** one conv + global-average-pool collapses the whole patch to exactly F
numbers (per-filter response averages). That is the entire representation the linear
head sees — no spatial/texture structure survives GAP. To rival our 18 hand-crafted
multi-band features you'd need ~18+ filters *and* each as informative as a
coherence/Hough/corner statistic; params blow past 190 long before accuracy arrives.
The only sub-190 conv (RGB F=4, 162) gets 0.639.

## Distillation (idea from prompt): does NOT rescue the student

Teachers (3-block CNN, ~290k params, don't ship) are strong: **all13 0.984, RGB
0.980**. Distilling into the tiny students (T=4, α=0.7) changed nothing:

| student | no KD | with KD |
|---------|------:|--------:|
| all13 F=16 (2058) | 0.892 | 0.878 |
| all13 F=8 (1034) | 0.835 | 0.834 |
| RGB F=24 (922) | 0.754 | 0.730 |
| RGB F=16 (618) | 0.738 | 0.724 |

Soft labels add no *capacity*. The bottleneck is the student architecture (GAP after
one conv), not the supervision, so KD (even from a 0.98 teacher) can't close a
>0.05 gap. A bigger student could learn from the teacher, but bigger = more deploy
params = off the frontier.

## RGB-only to save parameters: wrong lever for BOTH models

- **Linear model:** params = 10·(k+1), independent of band count — RGB saves ZERO
  params, and dropping NIR/SWIR (the bands that separate crop/veg/water in EuroSAT)
  needs *more* features to hold 94%. Strictly worse. (RGB-linear floor probe was
  killed mid-run under machine load; the param-accounting makes the outcome moot.)
- **Conv model:** RGB *does* cut conv input channels (F·9·3 vs F·9·13) but is far
  weaker per filter (RGB F=24 0.754 vs all13 F=16 0.892) because it discards the
  most discriminative bands — and the conv is uncompetitive regardless.

## Takeaway / don't-retry

The parameter-free-features + folded-linear approach is dramatically more
parameter-efficient than any tiny learned conv at these budgets: each feature costs
10 shared weights yet encodes rich multi-band texture a single conv layer can't
reconstruct. **Do not re-explore single-conv+GAP, distillation-into-tiny-conv, or
RGB-only for a param win.** If learned features are ever revisited, the open
(expensive, likely still off-frontier) question is a *multi-layer* student small
enough to ship — but the frontier lever remains: new orthogonal parameter-free
feature families pre-qualified by `fast_probe.py`, next being a 2nd global-structure
family (lbp2 / blob2lrg) stacked on corner (frozen23+corn2+blob2lrg → val 0.9493).
