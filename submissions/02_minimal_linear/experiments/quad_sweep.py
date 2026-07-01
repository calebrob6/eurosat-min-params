#!/usr/bin/env python
"""Does quadratic augmentation (parameter-free products of features) let a
linear model clear 94% with FEWER selected features than linear-only?

Augment the 169 base features with squares and pairwise products of the top-M
(by L1 importance).  Products are fixed arithmetic -> zero extra feature params.
Then L1-rank the augmented set and sweep the selected count k.
Selection on val; test printed for reference.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402


def feats(split):
    return (np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy')),
            np.load(os.path.join(CACHE_DIR, f'{split}_y.npy')))


def augment(x, top_idx):
    """Append squares and pairwise products of columns in top_idx."""
    cols = [x]
    t = x[:, top_idx]
    M = t.shape[1]
    prods = []
    for i in range(M):
        for j in range(i, M):  # includes squares (i==j)
            prods.append(t[:, i] * t[:, j])
    cols.append(np.stack(prods, axis=1).astype(np.float32))
    return np.concatenate(cols, axis=1)


def main():
    ftr, ytr = feats('train')
    fva, yva = feats('val')
    fte, yte = feats('test')

    base_order = l1_rank(ftr, ytr)
    M = 24
    top = base_order[:M]

    atr = augment(ftr, top)
    ava = augment(fva, top)
    ate = augment(fte, top)
    print(f'augmented feature dim: {atr.shape[1]} (base 169 + {atr.shape[1]-169} quad)')

    order = l1_rank(atr, ytr)

    ks = [30, 35, 40, 45, 50, 55, 60, 70]
    Cs = [3.0, 10.0, 30.0, 100.0]
    print(f'{"k":>4} {"params":>7} {"bestC":>6} {"val":>7} {"test":>7}')
    rows = []
    for k in ks:
        idx = order[:k]
        best = None
        for C in Cs:
            w, b, fi = fit_folded_logreg(atr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(ava, w, b, fi))
            if best is None or va > best[0]:
                best = (va, C, w, b, fi)
        va, C, w, b, fi = best
        te = accuracy_score(yte, predict(ate, w, b, fi))
        p = num_params(w, b)
        rows.append((k, p, C, va, te))
        print(f'{k:>4} {p:>7} {C:>6.0f} {va:>7.4f} {te:>7.4f}')

    print('\n--- min params meeting val threshold ---')
    for thr in (0.940, 0.943, 0.945, 0.950):
        ok = [r for r in rows if r[3] >= thr]
        if ok:
            b = min(ok, key=lambda r: r[1])
            print(f'val>={thr}: k={b[0]} params={b[1]} C={b[2]:.0f} val={b[3]:.4f} test={b[4]:.4f}')
        else:
            print(f'val>={thr}: none')


if __name__ == '__main__':
    main()
