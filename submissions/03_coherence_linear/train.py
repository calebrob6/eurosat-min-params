#!/usr/bin/env python
"""Submission 03: 510-parameter linear classifier on spectral+texture+coherence.

Two changes vs submission 02 (660 params) let us honestly go lower:

1.  **A new parameter-free feature family: structure-tensor coherence.**
    ``patch_features(..., coherence_scales=2)`` appends 26 coherence features
    (13 bands x 2 scales) to the 169 spectral+texture features.  Coherence
    measures how *directional* the local gradient field is, capturing linear
    structure (e.g. Highway roads) that magnitude-only texture misses.  It has
    zero learned parameters.

2.  **De-noised, train-only feature-count selection.**  Submission 02 picked k
    on the 5400-sample *val* split (noisy).  Here k is chosen by 5-fold
    cross-validation on the *train* split, averaged over 5 shuffles -- a
    lower-variance generalisation estimate that never touches val or test.
    Rule: the smallest k whose mean CV accuracy >= ``--cv-threshold`` (0.940).

With coherence added, that rule selects k=50 (510 params); the same rule on the
non-coherence features lands at k=65 (660).  The deployed model is a single
affine map ``logits = x[:, idx] @ W.T + b`` on the selected raw features, so
params == 10*(k+1).  Test is reported once at the end.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402

HERE = os.path.dirname(__file__)
COH_SCALES = 2  # structure-tensor coherence octaves appended to the 169 base feats


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Augmented (169 + 26 coherence) features, cached to *_feat_coh.npy."""
    fp = os.path.join(CACHE_DIR, f'{split}_feat_coh.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if os.path.exists(fp) and os.path.exists(yp):
        return np.load(fp), np.load(yp)
    x, y = load_cached(split)
    f, _ = patch_features(x, coherence_scales=COH_SCALES)
    np.save(fp, f)
    np.save(yp, y)
    return f, y


def cv_accuracy(x, y, idx, C, seeds, folds):
    """Mean k-fold train-CV accuracy over several shuffles (train only)."""
    vals = []
    for sd in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv-threshold', type=float, default=0.940)
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[40, 45, 50, 55, 60, 65])
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--folds', type=int, default=5)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (169 base + {13*COH_SCALES} coherence)')
    order = l1_rank(ftr, ytr)

    print(f'{"k":>4} {"params":>7} {"cv":>7} {"val":>7} {"test":>7}')
    chosen = None
    for k in args.ks:
        idx = order[:k]
        cv = cv_accuracy(ftr, ytr, idx, args.C, args.seeds, args.folds)
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        flag = ''
        if chosen is None and cv >= args.cv_threshold:
            chosen = (k, args.C, w, b, fi, cv, va, te, num_params(w, b))
            flag = '  <- selected'
        print(f'{k:>4} {num_params(w, b):>7} {cv:>7.4f} {va:>7.4f} {te:>7.4f}{flag}')

    if chosen is None:  # fall back to the largest k
        k = args.ks[-1]
        idx = order[:k]
        cv = cv_accuracy(ftr, ytr, idx, args.C, args.seeds, args.folds)
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        chosen = (k, args.C, w, b, fi, cv, va, te, num_params(w, b))

    k, C, w, b, fi, cv, va, te, p = chosen
    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=fi, k=k, C=C, coherence_scales=COH_SCALES,
             cv_acc=cv, val_acc=va, test_acc=te, params=p)
    print(f'\nSAVED k={k} C={C:.0f} params={p} cv={cv:.4f} val={va:.4f} '
          f'test={te:.4f} -> {out}')


if __name__ == '__main__':
    main()
