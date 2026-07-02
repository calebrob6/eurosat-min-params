"""Reproduce the 95% descent and report the family breakdown at the floor."""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.cache import CACHE_DIR
from src.linmodel import l1_rank
from src.select import backward_eliminate

FAMILIES = ['feat_o6', 'linefam_line', 'gs2fam_corn2', 'gs2fam_lbp2',
            'gs2fam_blob2', 'gs2fam_sslope2', 'difam_ixcoh', 'ixfam_ixtex2',
            'ofam_xcorr', 'ofam_oent2']

bounds, lo = [], 0
parts = []
for f in FAMILIES:
    a = np.load(os.path.join(CACHE_DIR, f'train_{f}.npy'))
    parts.append(a); bounds.append((lo, lo + a.shape[1], f)); lo += a.shape[1]
ftr = np.concatenate(parts, 1).astype(np.float32)
ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))

def fam_of(i):
    for a, b, name in bounds:
        if a <= i < b:
            return name.replace('feat_', '').replace('fam_', '').replace('gs2', '').replace('di', '').replace('ix', 'ix_').replace('of', '')
    return '?'

order = l1_rank(ftr, ytr)
_, _, subs = backward_eliminate(ftr, ytr, order[:60], 24, select_seeds=[0, 1, 2],
                                C=10.0, workers=12, record=True)
for k in (30, 27, 25):
    s = subs[k]
    from collections import Counter
    c = Counter(fam_of(int(i)) for i in s)
    print(f'k={k} ({9*(k+1)} params): ' + ', '.join(f'{n}:{v}' for n, v in c.most_common()))
