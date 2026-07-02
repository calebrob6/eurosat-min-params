# Submission 12 — 171-parameter reference-class linear model

**Test accuracy: 0.9433 (>94%). Deployed parameters: 171.**
A lossless reparameterization of submission 11 (190 params): same 18 features,
same predictions on every val and test image, 10% fewer stored parameters.

## What the model is

A single affine map on 18 parameter-free features, with a **9-row
reference-class head**:

```
logits_rest = x[:, feature_idx] @ W.T + b        # W: (9, 18), b: (9,)
logits      = insert 0 at class ref_class        # class 0 = AnnualCrop
pred        = argmax(logits)
```

Params = 9×18 weights + 9 biases = **171**. The features are the
submission-11 backward-greedy subset of the 320-dim `o6+line+corn2` pool
(16 o6 features + `linet3_ndvi` + `corn2frac_ndvi`), all computed by
`src.features.patch_features` with zero learned parameters.

## Why this is legitimate, not a metric hack

A softmax/argmax head is **shift-invariant**: adding any constant to all K
logits changes neither the argmax nor the softmax probabilities. Subtracting
class `ref`'s row (`W' = W − W[ref]`, `b' = b − b[ref]`) makes row `ref`
identically zero, so it need not be stored. This is standard multinomial-logit
practice (K−1 free categories): the honest parameter count of a K-class linear
head is `(K−1)(F+1)`, and every prior submission over-counted by exactly
`F+1`. `train.py` asserts the 9-row head's predictions are **bit-identical**
to the full 10-row head on both val and test before saving.

## Why 9 rows is the head's floor (measured)

`experiments/fewer_class_params.py` probed below 9 output dimensions with
error-correcting output codes (b binary base-learners + a fixed seeded code
matrix, params `b(F+1)`, best of 40 code seeds chosen on val):

| b (output dims) | params | test |
|--:|--:|--:|
| 4 | 76 | 0.697 |
| 8 | 152 | 0.849 |
| 9 | 171 | 0.864 |
| exact softmax (this model) | **171** | **0.9433** |

The 10 EuroSAT classes span 9 discriminant dimensions (consistent with the
iteration-3 SVD finding that all 9 are load-bearing), and independent binary
learners with linear decoding are a far worse rank-9 head than the jointly-fit
softmax. Rank/ECOC reduction below 9 is a dead end.

## Results

| model | params | val | test |
|---|--:|--:|--:|
| submission 11 (full 10-row head) | 190 | 0.9406 | 0.9433 |
| **submission 12 (reference-class)** | **171** | **0.9406** | **0.9433** |

Predictions are identical by construction; accuracies verified independently
by `eval.py`, which recomputes all features **from the raw patches** (no
caches) and applies only the stored 171 numbers.

## Provenance / honesty protocol

The 18-feature subset comes from submission 11's backward-greedy descent on
the 320-dim pool: select seeds {0..9} drive the greedy; two disjoint verify
blocks ({10..19}: 0.9442, {30..39}: 0.9442) plus held-out val (0.9406) gate
the commit; test drives no decision. `train.py` refits deterministically on
that fixed subset (folded logreg, C=10), checks the refit reproduces the
submission-11 checkpoint, converts to reference-class form, and asserts
prediction identity.

## Reproduce

```bash
python submissions/12_reference_class_linear/train.py   # fit + convert + save model.npz
python submissions/12_reference_class_linear/eval.py    # verify from raw patches
```
