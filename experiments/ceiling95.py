#!/usr/bin/env python
"""Can the hand-crafted parameter-free pool honestly reach 95%? At what params?

Assembles the richest available cached feature families, fits a folded logreg, and
reads the L1-top-k accuracy curve (3-seed verify-CV + val + test). The 95% gate is
honest only if verify-CV/val >= 0.95, not just test. params = (K-1)*(k+1) = 9*(k+1)
with the reference-class head (iter-19).
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402

# richest useful pool: o6 + every orthogonal family that ever helped a floor
FAMILIES = ['feat_o6', 'linefam_line', 'gs2fam_corn2', 'gs2fam_lbp2',
            'gs2fam_blob2', 'gs2fam_sslope2', 'difam_ixcoh', 'ixfam_ixtex2',
            'ofam_xcorr', 'ofam_oent2']


def pool(split):
    parts = []
    for f in FAMILIES:
        p = os.path.join(CACHE_DIR, f'{split}_{f}.npy')
        if os.path.exists(p):
            parts.append(np.load(p))
    return np.concatenate(parts, 1).astype(np.float32)


def main():
    ftr, fva, fte = pool('train'), pool('val'), pool('test')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    print(f'mega hand-crafted pool: {ftr.shape[1]} features from {len(FAMILIES)} families')

    for C in (1.0, 10.0):
        w, b, fi = fit_folded_logreg(ftr, ytr, np.arange(ftr.shape[1]), C=C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        print(f'  full pool C={C:<5} val={va:.4f} test={te:.4f}')

    order = l1_rank(ftr, ytr)
    print(f'\n{"k":>4} {"p(9x)":>6} {"verCV3":>7} {"val":>7} {"test":>7}  gate?')
    for k in [18, 24, 32, 48, 64, 96, 128, 160, 200, 256, ftr.shape[1]]:
        if k > ftr.shape[1]:
            continue
        idx = order[:k]
        w, b, fi = fit_folded_logreg(ftr, ytr, idx, C=10.0)
        ver = mean_cv(ftr, ytr, idx, range(10, 13), C=10.0)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        g = '  >=95!' if (ver >= .95 and va >= .95 and te >= .95) else ('  test>=95' if te >= .95 else '')
        print(f'{k:>4} {9*(k+1):>6} {ver:>7.4f} {va:>7.4f} {te:>7.4f}{g}', flush=True)


if __name__ == '__main__':
    main()
