#!/usr/bin/env python
"""Sweep reduced-rank linear classifiers over (feature-count k, rank r).

Goal: beat the plain-linear frontier (submission 02 = 660 params, test 0.9431)
by factoring the k x K weight matrix as (k x r)(r x K).  Params = k*r + r*K + K.

Prints, for each (k, r), the deployed param count and val/test accuracy, plus the
plain full-rank linear baseline at each k for reference.  Selection is by val
only; test is shown for the whole grid so the val-vs-test gap is visible.
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
    fit_lowrank, num_params_lowrank, predict_lowrank,
)


def _load(split):
    return (np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy')),
            np.load(os.path.join(CACHE_DIR, f'{split}_y.npy')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', type=int, nargs='+', default=[40, 60, 80, 100, 169])
    ap.add_argument('--ranks', type=int, nargs='+', default=[4, 5, 6, 7, 8, 9])
    ap.add_argument('--wd', type=float, nargs='+', default=[3e-3])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0])
    ap.add_argument('--epochs', type=int, default=3000)
    args = ap.parse_args()

    ftr, ytr = _load('train')
    fva, yva = _load('val')
    fte, yte = _load('test')
    order = l1_rank(ftr, ytr)

    print(f'{"k":>4} {"r":>3} {"wd":>7} {"seed":>4} {"params":>7} '
          f'{"val":>7} {"test":>7}')
    for k in args.ks:
        idx = order[:k]
        # plain full-rank linear baseline at this k (best of a small C grid on val)
        best = None
        for C in (3.0, 10.0, 30.0):
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            if best is None or va > best[0]:
                best = (va, C, w, b, fi)
        va, C, w, b, fi = best
        te = accuracy_score(yte, predict(fte, w, b, fi))
        print(f'{k:>4} {"lin":>3} {C:>7.0f} {"-":>4} {num_params(w, b):>7} '
              f'{va:>7.4f} {te:>7.4f}')
        for r in args.ranks:
            if r >= min(k, 10):
                continue  # rank >= K makes the factorisation pointless
            for wd in args.wd:
                for seed in args.seeds:
                    a, c, d, fi = fit_lowrank(
                        ftr, ytr, rank=r, feature_idx=idx,
                        weight_decay=wd, epochs=args.epochs, seed=seed)
                    va = accuracy_score(yva, predict_lowrank(fva, a, c, d, fi))
                    te = accuracy_score(yte, predict_lowrank(fte, a, c, d, fi))
                    p = num_params_lowrank(a, c, d)
                    print(f'{k:>4} {r:>3} {wd:>7.4f} {seed:>4} {p:>7} '
                          f'{va:>7.4f} {te:>7.4f}')
        sys.stdout.flush()


if __name__ == '__main__':
    main()
