#!/usr/bin/env python
"""Honest backward-greedy floor at the 95% gate on the mega hand-crafted pool.

Fast signal-first: warm-start from the L1 top-START of the 377-feat mega pool,
backward-greedy descend with few select seeds (path is seed-robust), record every
subset, then confirm each k on TWO disjoint 10-seed verify blocks + held-out val,
gating at 0.95. Reports the smallest k clearing all three -> params 9*(k+1) with
the reference-class head.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402

FAMILIES = ['feat_o6', 'linefam_line', 'gs2fam_corn2', 'gs2fam_lbp2',
            'gs2fam_blob2', 'gs2fam_sslope2', 'difam_ixcoh', 'ixfam_ixtex2',
            'ofam_xcorr', 'ofam_oent2']
GATE = 0.95


def pool(split):
    parts = [np.load(os.path.join(CACHE_DIR, f'{split}_{f}.npy')) for f in FAMILIES
             if os.path.exists(os.path.join(CACHE_DIR, f'{split}_{f}.npy'))]
    return np.concatenate(parts, 1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=60)
    ap.add_argument('--kmin', type=int, default=28)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--families', type=str, default=None,
                    help='comma-separated family cache stems (default: full mega pool)')
    args = ap.parse_args()
    global FAMILIES
    if args.families:
        FAMILIES = args.families.split(',')

    ftr, fva, fte = pool('train'), pool('val'), pool('test')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    print(f'mega pool {ftr.shape[1]} feats; backward-greedy top-{args.start} -> {args.kmin} '
          f'(select {args.select_seeds}); gate {GATE}', flush=True)

    order = l1_rank(ftr, ytr)
    init = order[:args.start]
    _, _, subs = backward_eliminate(ftr, ytr, init, args.kmin,
                                    select_seeds=args.select_seeds, C=args.C,
                                    workers=args.workers, record=True)
    print(f'\n{"k":>4} {"p(9x)":>6} {"verA":>7} {"verB":>7} {"val":>7} {"test":>7}  gate?')
    floor = None
    for k in sorted(subs, reverse=True):
        s = subs[k]
        w, b, fi = fit_folded_logreg(ftr, ytr, s, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        vera = mean_cv(ftr, ytr, s, range(10, 20), C=args.C)
        verb = mean_cv(ftr, ytr, s, range(30, 40), C=args.C)
        ok = vera >= GATE and verb >= GATE and va >= GATE
        if ok:
            floor = k
        print(f'{k:>4} {9*(k+1):>6} {vera:>7.4f} {verb:>7.4f} {va:>7.4f} {te:>7.4f}'
              f'{"  PASS" if ok else ""}', flush=True)
    print(f'\n95% honest floor: k={floor} -> params={9*(floor+1) if floor else None} '
          f'(reference-class head)', flush=True)


if __name__ == '__main__':
    main()
