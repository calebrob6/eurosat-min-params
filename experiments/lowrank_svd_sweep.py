#!/usr/bin/env python
"""Optimal reduced-rank (SVD-truncation) sweep over (feature-count k, rank r).

Unlike the torch sweep, this uses the *optimal* rank-r approximation of a
well-optimised full linear model, so rank==9 recovers full-linear accuracy and
the accuracy-vs-rank curve is honest.  Params = k*r + r*K + K.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402
from src.lowrank import (  # noqa: E402
    fit_lowrank_svd, num_params_lowrank, predict_lowrank,
)


def _load(split):
    return (np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy')),
            np.load(os.path.join(CACHE_DIR, f'{split}_y.npy')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', type=int, nargs='+', default=[50, 60, 80, 100, 130, 169])
    ap.add_argument('--ranks', type=int, nargs='+', default=[3, 4, 5, 6, 7, 8, 9])
    ap.add_argument('--C', type=float, default=10.0)
    args = ap.parse_args()

    ftr, ytr = _load('train')
    fva, yva = _load('val')
    fte, yte = _load('test')
    order = l1_rank(ftr, ytr)

    print(f'{"k":>4} {"r":>4} {"params":>7} {"val":>7} {"test":>7}')
    rows = []
    for k in args.ks:
        idx = order[:k]
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        print(f'{k:>4} {"lin":>4} {num_params(w, b):>7} {va:>7.4f} {te:>7.4f}')
        for r in args.ranks:
            if r >= min(k, 10):
                continue
            a, c, d, fi = fit_lowrank_svd(ftr, ytr, rank=r, feature_idx=idx, C=args.C)
            va = accuracy_score(yva, predict_lowrank(fva, a, c, d, fi))
            te = accuracy_score(yte, predict_lowrank(fte, a, c, d, fi))
            p = num_params_lowrank(a, c, d)
            rows.append((p, k, r, va, te))
            print(f'{k:>4} {r:>4} {p:>7} {va:>7.4f} {te:>7.4f}')
        sys.stdout.flush()

    # Best configs by smallest params among those with val >= 0.940
    ok = [row for row in rows if row[3] >= 0.940]
    ok.sort()
    print('\nconfigs with val>=0.940, sorted by params:')
    for p, k, r, va, te in ok[:12]:
        print(f'  params={p:>5} k={k:>3} r={r} val={va:.4f} test={te:.4f}')


if __name__ == '__main__':
    main()
