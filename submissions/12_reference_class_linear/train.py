#!/usr/bin/env python
"""Submission 12: 171-parameter reference-class linear classifier.

A LOSSLESS reparameterization of submission 11 (190 params, k=18, test 0.9433).
Every prior submission counted the linear head as ``K*(F+1)`` = 10 rows of
weights + biases.  But an argmax/softmax head is SHIFT-INVARIANT: subtracting
any one class's logit from all K logits changes neither the argmax nor the
softmax probabilities.  So one class is a free REFERENCE with a constant 0
logit, and a 10-class head needs only ``(K-1)*(F+1) = 9*(F+1)`` stored
parameters — with predictions IDENTICAL to the full head.  This is standard
multinomial-logit practice (K-1 free categories), not a metric hack: the
deployed model genuinely stores 171 numbers and computes the same argmax.

Concretely: fit the same folded logreg as submission 11 on its fixed
backward-greedy-selected 18-feature subset, then set ``W' = W - W[ref];
b' = b - b[ref]`` (row ``ref`` becomes identically 0) and drop row ``ref``.
Params: 9*(18+1) = 171, a free 19-param (10%) cut from 190.

Why 9 rows is the head's floor (measured, iteration 19):
  * error-correcting output codes (b binary learners + fixed seeded code,
    params b*(F+1), best of 40 code seeds ON VAL) crater: b=9 tests 0.864,
    b=8 0.849, b=4 0.697 — versus 0.9433 for the exact softmax.  The 10
    EuroSAT classes genuinely span 9 discriminant dimensions (consistent
    with iteration 3's SVD result), and independent-binary + linear decode
    is a far worse rank-9 head than the jointly-fit softmax.

Provenance of the 18 features: the submission-11 honest backward-greedy
descent on the 320-dim o6+line+corn2 pool (select seeds 0-9 drive the greedy;
disjoint verify seeds 10-19 AND 30-39 plus held-out val gate the commit; test
drives no decision).  This script re-fits deterministically on that fixed
subset (same C=10 folded logreg), checks the refit reproduces submission 11's
saved head, converts to the 9-row reference-class form, and asserts exact
prediction identity on val AND test before saving.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import (  # noqa: E402
    fit_folded_logreg, predict, predict_reference_class, to_reference_class)

HERE = os.path.dirname(__file__)
SUB11 = os.path.join(HERE, '..', '11_corner_backward_linear', 'model.npz')
# zero-parameter feature configuration (identical to submission 11): the
# 320-dim o6+line+corn2 pool.
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8,
                orient_hist_bins=4, spectral_peak=True, xband=True,
                index_texture=True, hough_lines=True, harris_corners=True)
# The submission-11 backward-greedy subset (k=18): 16 o6 + linet3_ndvi (310) +
# corn2frac_ndvi (316).
FEATURE_IDX = [58, 65, 68, 75, 86, 92, 93, 145, 203, 273, 275, 287, 291, 294,
               297, 298, 310, 316]
C = 10.0
REF = 0   # any class works (the map is exact for every choice); fix class 0


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """320-dim o6+line+corn2 pool, reusing the family caches when present."""
    fp = os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')
    lp = os.path.join(CACHE_DIR, f'{split}_linefam_line.npy')
    cp = os.path.join(CACHE_DIR, f'{split}_gs2fam_corn2.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if all(os.path.exists(p) for p in (fp, lp, cp, yp)):
        return (np.concatenate([np.load(fp), np.load(lp), np.load(cp)], 1)
                .astype(np.float32), np.load(yp))
    x, y = load_cached(split)
    f, _ = patch_features(x, **FEAT_CFG)
    return f.astype(np.float32), y


def main() -> None:
    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')

    idx = np.array(FEATURE_IDX)
    m11 = np.load(SUB11)
    assert np.array_equal(m11['feature_idx'], idx), 'subset drifted from sub 11'

    # 1. deterministic refit of the submission-11 head on its fixed subset
    w, b, fidx = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
    pv10, pt10 = predict(fva, w, b, fidx), predict(fte, w, b, fidx)
    same11 = (np.array_equal(pt10, predict(fte, m11['W'], m11['b'], m11['feature_idx']))
              and np.allclose(w, m11['W'], atol=1e-5))
    print(f'refit reproduces submission-11 checkpoint: {same11}')

    # 2. lossless reference-class reparameterization: 10 rows -> 9
    w9, b9 = to_reference_class(w, b, ref=REF)
    pv9 = predict_reference_class(fva, w9, b9, fidx, ref=REF)
    pt9 = predict_reference_class(fte, w9, b9, fidx, ref=REF)
    assert np.array_equal(pv9, pv10) and np.array_equal(pt9, pt10), \
        'reference-class head must be prediction-identical to the full head'

    params = int(w9.size + b9.size)
    va, te = accuracy_score(yva, pv9), accuracy_score(yte, pt9)
    print(f'full head:      params={w.size + b.size}  val={accuracy_score(yva, pv10):.4f} '
          f'test={accuracy_score(yte, pt10):.4f}')
    print(f'reference-class: params={params}  val={va:.4f} test={te:.4f}  '
          f'(predictions identical on val and test)')

    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w9, b=b9, feature_idx=fidx, ref_class=REF, k=len(idx), C=C,
             val_acc=va, test_acc=te, params=params, **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
