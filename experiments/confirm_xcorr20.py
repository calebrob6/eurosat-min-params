#!/usr/bin/env python
"""20-seed confirmation of the xcorr pool CV floor (guards against pool-selection luck)."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from src.cache import CACHE_DIR
from src.linmodel import fit_folded_logreg, l1_rank, predict


def pool(split):
    a = np.load(os.path.join(CACHE_DIR, f'{split}_feat_o4.npy'))
    b = np.load(os.path.join(CACHE_DIR, f'{split}_ofam_xcorr.npy'))
    return np.concatenate([a, b], 1).astype(np.float32)


ftr, fva, fte = pool('train'), pool('val'), pool('test')
ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
order = l1_rank(ftr, ytr)
print('xcorr cols (>=273) in top-38:', [int(i) for i in order[:38] if i >= 273])
for k in (37, 38, 39):
    idx = order[:k]
    seedvals = []
    for sd in range(20):
        skf = StratifiedKFold(5, shuffle=True, random_state=sd)
        accs = [accuracy_score(ytr[te], predict(ftr[te], *fit_folded_logreg(ftr[tr], ytr[tr], feature_idx=idx)))
                for tr, te in skf.split(ftr, ytr)]
        seedvals.append(np.mean(accs))
    cv = np.mean(seedvals); se = np.std(seedvals) / np.sqrt(len(seedvals))
    w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx)
    va = accuracy_score(yva, predict(fva, w, b, fi))
    te = accuracy_score(yte, predict(fte, w, b, fi))
    print(f'k={k} params={10*(k+1)} cv20={cv:.4f}+-{se:.4f} val={va:.4f} test={te:.4f}')
print('DONE')
