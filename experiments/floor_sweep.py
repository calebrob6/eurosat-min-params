#!/usr/bin/env python
"""Measure the honest CV>=0.940 param floor for different feature pools.

Baseline pool = the 195 coherence-augmented features (submission 03).  We compute
several new zero-parameter families, cache them, and for each candidate pool
report the smallest k whose multi-seed train-CV accuracy >= threshold, together
with its val/test accuracy.  The goal: a family that moves the floor below k=50
(510 params).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from experiments import new_features_lib as nf  # noqa: E402

FAMILIES = {
    'hogpan': lambda x: nf.hog_pan(x, nbins=8, scales=2),
    'hog4': lambda x: nf.hog_features(x, nbins=4, scales=1),
    'coh3': lambda x: nf.coherence_scale3(x),
    'oent': lambda x: nf.orient_entropy(x, nbins=8),
    'fft': lambda x: nf.fft_periodicity(x),
}


def base_pool(split):
    return np.load(os.path.join(CACHE_DIR, f'{split}_feat_coh.npy'))


def family(split, name):
    fp = os.path.join(CACHE_DIR, f'{split}_fam_{name}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    x, _ = load_cached(split)
    feats, _ = FAMILIES[name](x)
    np.save(fp, feats)
    return feats


def build_pool(split, extra):
    parts = [base_pool(split)]
    for name in extra:
        parts.append(family(split, name))
    return np.concatenate(parts, 1)


def cv_acc(x, y, idx, C, seeds, folds):
    vals = []
    for sd in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def evaluate(extra, ks, C, seeds, folds, thresh):
    ftr, ytr = build_pool('train', extra), np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    fva, yva = build_pool('val', extra), np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    fte, yte = build_pool('test', extra), np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    order = l1_rank(ftr, ytr)
    label = '+'.join(extra) if extra else 'base195'
    print(f'\n=== pool: base195+[{label}]  dim={ftr.shape[1]} ===')
    print(f'{"k":>4} {"params":>7} {"cv":>7} {"val":>7} {"test":>7}')
    picked = None
    for k in ks:
        idx = order[:k]
        cv = cv_acc(ftr, ytr, idx, C, seeds, folds)
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        flag = ''
        if picked is None and cv >= thresh:
            picked = (k, cv, va, te)
            flag = '  <- selected'
        print(f'{k:>4} {10*(k+1):>7} {cv:>7.4f} {va:>7.4f} {te:>7.4f}{flag}')
    if picked:
        k, cv, va, te = picked
        print(f'FLOOR [{label}]: k={k} params={10*(k+1)} cv={cv:.4f} val={va:.4f} test={te:.4f}')
    else:
        print(f'FLOOR [{label}]: none reached cv>={thresh}')
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extra', nargs='*', default=[],
                    help='family names to append to the base pool')
    ap.add_argument('--ks', type=int, nargs='+', default=[35, 40, 45, 50, 55])
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--thresh', type=float, default=0.940)
    args = ap.parse_args()
    evaluate(args.extra, args.ks, args.C, args.seeds, args.folds, args.thresh)


if __name__ == '__main__':
    main()
