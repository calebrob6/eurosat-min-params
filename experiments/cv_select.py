#!/usr/bin/env python
"""De-noised feature-count selection via train-only K-fold cross-validation.

Submission 02 uses k=65 (660 params) only because the *val* accuracy first
crosses 0.940 there -- but plain linear at k=50 already gets test 0.9407 (510
params).  The gap is val noise (5400 samples).  Here we pick k using 5-fold CV
on the TRAIN split only (no val, no test), which is a lower-variance estimate of
generalisation, then confirm on val and report test ONCE.

Feature ranking (L1) is done once on full train (parameter-free selection,
matching submissions 01/02); C is fixed at 10 a priori.  Selection rule: the
smallest k whose mean 5-fold CV accuracy >= --cv-threshold.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402


def _load(split):
    return (np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy')),
            np.load(os.path.join(CACHE_DIR, f'{split}_y.npy')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ks', type=int, nargs='+',
                    default=[35, 40, 45, 50, 55, 60, 65, 70, 80])
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--cv-threshold', type=float, default=0.940)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    ftr, ytr = _load('train')
    fva, yva = _load('val')
    fte, yte = _load('test')
    order = l1_rank(ftr, ytr)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    print(f'{"k":>4} {"params":>7} {"cv_mean":>8} {"cv_std":>7} '
          f'{"val":>7} {"test":>7}')
    results = []
    for k in args.ks:
        idx = order[:k]
        accs = []
        for tr, te in skf.split(ftr, ytr):
            w, b, fi = fit_folded_logreg(ftr[tr], ytr[tr], feature_idx=idx, C=args.C)
            accs.append(accuracy_score(ytr[te], predict(ftr[te], w, b, fi)))
        cv_mean, cv_std = float(np.mean(accs)), float(np.std(accs))
        # full-train fit for val/test reference
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te_ = accuracy_score(yte, predict(fte, w, b, fi))
        p = num_params(w, b)
        results.append((k, p, cv_mean, cv_std, va, te_))
        print(f'{k:>4} {p:>7} {cv_mean:>8.4f} {cv_std:>7.4f} {va:>7.4f} {te_:>7.4f}')

    chosen = next((r for r in results if r[2] >= args.cv_threshold), None)
    if chosen is None:
        chosen = results[-1]
    k, p, cvm, cvs, va, te_ = chosen
    print(f'\nSELECTED (smallest k with CV>={args.cv_threshold}): '
          f'k={k} params={p} cv={cvm:.4f} val={va:.4f} test={te_:.4f}')


if __name__ == '__main__':
    main()
