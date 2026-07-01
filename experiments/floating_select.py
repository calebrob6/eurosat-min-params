#!/usr/bin/env python
"""Sequential Floating Backward Selection (SFBS) with honest disjoint-seed checks.

Submission 07 used pure backward-greedy elimination, which is one-directional: a
feature dropped early can never return, so its size-k subset is *nested* inside
its size-(k+1) subset.  SFBS (``src.select.floating_backward``) adds conditional
forward steps -- re-adding the most useful excluded feature whenever that beats
the best subset of that size found so far -- so it can escape the nesting and
land on a better subset at every size.

Both searches run over the SAME fixed candidate universe (the L1 top-``START``),
so any improvement is attributable to the floating (re-addition) capability, not
to a larger pool.

Honesty protocol (identical to ``backward_select.py``):
  * SELECT seeds {SEL0..SEL0+NSEED-1} drive every SFBS decision.
  * VERIFY seeds {VER0..VER0+NSEED-1} -- never used to choose a feature -- give an
    unbiased CV estimate of each resulting subset.
  * val (fully held out) is the third independent check.
The honest floor is the smallest k where BOTH VERIFY-CV and val clear 0.940.
test is reported alongside but drives NO decision.

Env: START (universe size / L1 top-N, default 42), STOP (smallest k, default 24),
SEL0/VER0/NSEED (seed partition), TAG (output suffix).  Writes
``floating_select_<tag>_result.txt``.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import floating_backward, mean_cv  # noqa: E402

START = int(os.environ.get('START', '42'))
STOP = int(os.environ.get('STOP', '24'))
_SEL0 = int(os.environ.get('SEL0', '0'))
_VER0 = int(os.environ.get('VER0', '10'))
_NSEED = int(os.environ.get('NSEED', '10'))
SELECT_SEEDS = list(range(_SEL0, _SEL0 + _NSEED))
VERIFY_SEEDS = list(range(_VER0, _VER0 + _NSEED))
TAG = os.environ.get('TAG', '')
FOLDS = 5
C = 10.0
THRESH = 0.940
WORKERS = int(os.environ.get('WORKERS', '32'))


def load_pool(split):
    return np.load(os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')).astype(np.float32)


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr = load_pool('train')
    fva = load_pool('val')
    fte = load_pool('test')

    order = l1_rank(ftr, ytr)
    universe = order[:START].copy()

    tag = (TAG + '_') if TAG else ''
    outpath = os.path.join(os.path.dirname(__file__),
                           f'floating_select_{tag}s{_SEL0}result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    emit(f'SFBS o6 dim={ftr.shape[1]} universe=L1top{START} STOP={STOP} '
         f'SELECT={SELECT_SEEDS[0]}..{SELECT_SEEDS[-1]} '
         f'VERIFY={VERIFY_SEEDS[0]}..{VERIFY_SEEDS[-1]}')

    best, trace = floating_backward(
        ftr, ytr, universe, STOP, select_seeds=SELECT_SEEDS,
        folds=FOLDS, C=C, workers=WORKERS, log=lambda m: emit('  ' + m))

    emit(f'\n{"k":>4} {"params":>7} {"selCV":>7} {"verCV":>7} {"val":>7} {"test":>7}')
    rows = []
    for k in sorted(best):
        sel, idx = best[k]
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
        ver = mean_cv(ftr, ytr, idx, VERIFY_SEEDS, FOLDS, C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        rows.append((k, 10 * (k + 1), sel, ver, va, te, list(idx)))
        emit(f'{k:>4} {10*(k+1):>7} {sel:>7.4f} {ver:>7.4f} {va:>7.4f} {te:>7.4f}')

    honest = [r for r in rows if r[3] >= THRESH and r[4] >= THRESH]
    emit('\n########## HONEST FLOOR (verCV>=0.940 AND val>=0.940) ##########')
    if honest:
        b0 = min(honest, key=lambda r: r[0])
        emit(f'k={b0[0]} params={b0[1]} selCV={b0[2]:.4f} verCV={b0[3]:.4f} '
             f'val={b0[4]:.4f} test={b0[5]:.4f}')
        emit('feature_idx=' + ','.join(map(str, b0[6])))
    else:
        emit('none cleared both verCV>=0.940 and val>=0.940')
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
