# Reference-class head: the 10x multiplier is really 9x → frontier 190 → 171 (lossless)

*2026-07-02 15:00 — iteration 19*

## The idea

Every submission counts the linear head as `10*(F+1)` (10 classes × F weights +
10 biases). But a **softmax is shift-invariant**: adding the same constant to all
10 class logits changes neither argmax nor the probabilities. So exactly one class
is redundant — fix it as a **reference class with a constant 0 logit** and store
weights for only the other 9. A 10-class linear head therefore needs
`(K-1)*(F+1) = 9*(F+1)` parameters, with **identical predictions**. We were
over-counting every head by exactly `(F+1)`.

This is standard multinomial-logit practice (K-1 free categories); it is a
legitimate reparameterization, not a metric hack — the deployed model computes the
same argmax. `experiments/fewer_class_params.py` derives it post-hoc from a trained
head: `W' = W - W[ref]; b' = b - b[ref]` makes row `ref` identically 0, so drop it.

## Result (submission 11 features, F=18)

| head | params | val | test | identical? |
|------|-------:|----:|-----:|:----------:|
| full 10-class | 190 | 0.9406 | 0.9433 | — |
| **reference-class (9)** | **171** | 0.9406 | 0.9433 | **yes (exact)** |

**New frontier = 171 params, test 0.9433**, a free 19-param (10%) cut, zero
accuracy change. Applies to EVERY submission (each drops by F+1): e.g. sub 10
(F=23) 240→216, sub 09 (F=25) 260→234, etc. — but sub 11 stays the frontier.

## Can we go BELOW 9 output dims? No (ECOC cliff)

Tested error-correcting output codes = b binary base-learners + a FIXED seeded
code matrix (a rank-b head with a FREE codebook, params `b*(F+1)`), best of 40
code seeds picked on val:

| b | params | test |
|--:|-------:|-----:|
| 4 | 76 | 0.697 |
| 6 | 114 | 0.833 |
| 8 | 152 | 0.849 |
| 9 | 171 | 0.864 |
| 10 | 190 | 0.858 |

All far below the exact softmax (0.9433). Two lessons: (1) the 10 classes span 9
discriminant dims — dropping below 9 outputs loses 8–25% (confirms iter-3's SVD
result that all 9 dims are load-bearing); (2) even at b=9/10, independent binary
learners + a random code + linear decoding is a much worse rank-9 head than the
jointly-fit softmax. **ECOC/low-rank is a dead end for param reduction here.**

## What's left to try below 9*(F+1)

The only remaining lever on the head is **per-discriminant feature sparsity** — let
different class-discriminants use different feature subsets (a hierarchical / tree
head where each binary split uses only the few features it needs, params =
Σ_nodes (f_node+1)). Caveats: iter-12 already found element-wise weight sparsity
LOSES to shared column selection (features are shared across classes), and a flat
9-way with all 18 feats is exactly 171, so a tree only wins if the natural class
hierarchy has genuinely cheap splits (e.g. SeaLake vs rest via one water index)
that outweigh cascading errors. Untested; speculative; probably marginal. The
reference-class 171 is the solid, lossless win to bank now.

## Action

Adopt `(K-1)*(F+1)` as the honest head count project-wide; deploy submission 11 in
reference-class form → **171 params**. Update its eval/README to store/report the
9-row head.
