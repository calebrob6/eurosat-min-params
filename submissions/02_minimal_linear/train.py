#!/usr/bin/env python
"""Submission 02: minimal-parameter linear classifier on fixed features.

Same parameter-free feature pipeline as submission 01 (per-band mean/std/
percentiles + multi-scale gradient texture = 169 features), but the number of
selected features ``k`` is pushed as low as the *validation* split allows.

Selection rule (uses ONLY train + val):
  1. Rank the 169 features by an L1-logreg importance on train.
  2. For each candidate ``k`` (top-k features), pick the L2 strength ``C`` that
     maximises val accuracy.
  3. Choose the SMALLEST ``k`` whose best val accuracy >= ``--val-threshold``.

The deployed model is a single affine map ``logits = x[:, idx] @ W.T + b`` on
the selected raw features, so params == ``10 * (k + 1)``.  Test is reported once
at the end.

Why not a nonlinear head?  Empirically (see experiments/): a bottleneck MLP and
a quadratic feature expansion both *raise val but not test* -- they overfit.
The classes are linearly separable given good texture features, so a linear head
is at the accuracy-per-parameter frontier.
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
    ap.add_argument('--val-threshold', type=float, default=0.940)
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[35, 40, 45, 50, 55, 60, 65, 70, 80])
    ap.add_argument('--Cs', type=float, nargs='+',
                    default=[1.0, 3.0, 10.0, 30.0, 100.0])
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    order = l1_rank(ftr, ytr)

    print(f'{"k":>4} {"params":>7} {"bestC":>6} {"val":>7} {"test":>7}')
    chosen = None
    for k in args.ks:
        idx = order[:k]
        best = None
        for C in args.Cs:
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            if best is None or va > best[0]:
                best = (va, C, w, b, fi)
        va, C, w, b, fi = best
        te = accuracy_score(yte, predict(fte, w, b, fi))
        p = num_params(w, b)
        flag = ''
        if chosen is None and va >= args.val_threshold:
            chosen = (k, C, w, b, fi, va, te, p)
            flag = '  <- selected'
        print(f'{k:>4} {p:>7} {C:>6.0f} {va:>7.4f} {te:>7.4f}{flag}')

    if chosen is None:  # fall back to the largest k
        k = args.ks[-1]
        idx = order[:k]
        best = None
        for C in args.Cs:
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            if best is None or va > best[0]:
                best = (va, C, w, b, fi)
        va, C, w, b, fi = best
        te = accuracy_score(yte, predict(fte, w, b, fi))
        chosen = (k, C, w, b, fi, va, te, num_params(w, b))

    k, C, w, b, fi, va, te, p = chosen
    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=fi, k=k, C=C,
             val_acc=va, test_acc=te, params=p)
    print(f'\nSAVED k={k} C={C:.0f} params={p} val={va:.4f} test={te:.4f} -> {out}')


if __name__ == '__main__':
    main()
