#!/usr/bin/env python
"""Map the linear (folded-logreg) params-vs-accuracy frontier at fine k.

Selection is on val only; test printed for reference.
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


def main():
    ftr, ytr = feats('train')
    fva, yva = feats('val')
    fte, yte = feats('test')
    order = l1_rank(ftr, ytr)

    ks = [35, 40, 45, 50, 55, 60, 70, 80]
    Cs = [1.0, 3.0, 10.0, 30.0, 100.0]

    print(f'{"k":>4} {"params":>7} {"bestC":>6} {"val":>7} {"test":>7}')
    rows = []
    for k in ks:
        idx = order[:k]
        best = None
        for C in Cs:
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            if best is None or va > best[0]:
                best = (va, C, w, b, fi)
        va, C, w, b, fi = best
        te = accuracy_score(yte, predict(fte, w, b, fi))
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
