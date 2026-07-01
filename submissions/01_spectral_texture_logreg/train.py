#!/usr/bin/env python
"""Submission 01: linear classifier on fixed spectral + multi-scale texture features.

Pipeline (all feature extraction is *parameter-free* fixed arithmetic):
  raw 13-band patch -> per-band mean/std/percentiles + multi-scale gradient
  texture stats (169 features) -> select top-k by an L1 ranking -> logistic
  regression -> fold the standardiser into the weights.

The deployed model is a single affine map ``logits = x[:, idx] @ W.T + b`` on
the selected raw features, so the parameter count is exactly ``10 * (k + 1)``.

Model selection uses ONLY the val split; the test split is reported once at the
end for the final accuracy.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402

HERE = os.path.dirname(__file__)


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Return cached (features, labels) for a split, computing if absent."""
    fp = os.path.join(CACHE_DIR, f'{split}_feat.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if os.path.exists(fp) and os.path.exists(yp):
        return np.load(fp), np.load(yp)
    x, y = load_cached(split)
    f, _ = patch_features(x)
    np.save(fp, f)
    np.save(yp, y)
    return f, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--val-threshold', type=float, default=0.942,
                    help='Smallest k whose val accuracy exceeds this is saved.')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[40, 50, 60, 70, 80, 90, 100, 120, 169])
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')

    order = l1_rank(ftr, ytr)

    print(f'{"k":>4} {"params":>7} {"val":>7} {"test":>7}')
    chosen = None
    for k in args.ks:
        idx = order[:k]
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        p = num_params(w, b)
        flag = ''
        if chosen is None and va >= args.val_threshold:
            chosen = (k, w, b, fi, va, te, p)
            flag = '  <- selected'
        print(f'{k:>4} {p:>7} {va:>7.4f} {te:>7.4f}{flag}')

    if chosen is None:  # fall back to the largest k
        k = args.ks[-1]
        idx = order[:k]
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        chosen = (k, w, b, fi, va, te, num_params(w, b))

    k, w, b, fi, va, te, p = chosen
    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=fi, k=k, val_acc=va, test_acc=te, params=p)
    print(f'\nSAVED k={k} params={p} val={va:.4f} test={te:.4f} -> {out}')


if __name__ == '__main__':
    main()
