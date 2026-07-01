#!/usr/bin/env python
"""Sweep tiny-MLP-on-features configs to map the params-vs-accuracy frontier.

Selects everything on val; prints test only for reference (the final submission
picks its config on val alone).
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import l1_rank  # noqa: E402
from src.mlp import fit_folded_mlp  # noqa: E402


def feats(split):
    return (np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy')),
            np.load(os.path.join(CACHE_DIR, f'{split}_y.npy')))


def main():
    ftr, ytr = feats('train')
    fva, yva = feats('val')
    fte, yte = feats('test')

    order = l1_rank(ftr, ytr)

    Fs = [16, 20, 26, 32, 40, 52]
    Hs = [8, 10, 12, 16]
    seeds = [0, 1, 2, 3, 4]

    # Verify the fold once: FoldedMLP on raw features must match training path.
    m0, v0, _ = fit_folded_mlp(ftr, ytr, fva, yva, order[:20], hidden=12, seed=0)
    va_folded = accuracy_score(yva, m0.predict(fva))
    print(f'[fold check] best_val={v0:.4f} folded_val={va_folded:.4f} '
          f'(should match) params={m0.num_params}')

    print(f'\n{"F":>3} {"H":>3} {"params":>7} {"val*":>7} {"seed*":>5} {"test":>7}')
    rows = []
    for F in Fs:
        idx = order[:F]
        for H in Hs:
            best = None
            for s in seeds:
                m, va, _ = fit_folded_mlp(ftr, ytr, fva, yva, idx, hidden=H, seed=s)
                if best is None or va > best[0]:
                    best = (va, s, m)
            va, s, m = best
            te = accuracy_score(yte, m.predict(fte))
            p = m.num_params
            rows.append((F, H, p, va, s, te))
            print(f'{F:>3} {H:>3} {p:>7} {va:>7.4f} {s:>5} {te:>7.4f}')

    # Pareto frontier on (params, val).
    print('\n--- Pareto (min params at each val>=threshold) ---')
    for thr in (0.940, 0.945, 0.950):
        ok = [r for r in rows if r[3] >= thr]
        if ok:
            best = min(ok, key=lambda r: r[2])
            print(f'val>={thr}: F={best[0]} H={best[1]} params={best[2]} '
                  f'val={best[3]:.4f} test={best[5]:.4f}')
        else:
            print(f'val>={thr}: none')


if __name__ == '__main__':
    main()
