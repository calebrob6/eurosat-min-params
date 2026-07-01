#!/usr/bin/env python
"""Backward-greedy feature elimination with honest disjoint-seed verification.

The L1-rank top-k order is only a proxy for the best k-feature subset.  This
script instead does backward-greedy elimination: start from the L1 top-``START``
features (a known-good set well above the 0.940 bar), and repeatedly drop the one
feature whose removal *least* hurts (or most helps) the mean train-CV, tracing a
CV-vs-k curve down to ``STOP``.  It explores a different, potentially smaller
subset than the fixed L1 order.

Honesty protocol (avoids the greedy-on-CV overfit the notes warn about):
  * SELECT seeds {0..9} drive the greedy removal decisions.
  * VERIFY seeds {10..19} -- never used to choose a feature -- give an unbiased
    CV estimate of the resulting subset.
  * val (fully held out) is the third independent check.
The honest floor is the smallest k where BOTH the VERIFY-CV and val clear 0.940.
test is reported alongside but never used for any decision.

Pool is selected by ``POOL`` env var: ``o6`` (the 305-feature submission-06 pool)
or ``o6+ixcoh`` (adds the 6-feature index-map coherence family, which raised test
at every k in the iteration-7 floor sweep).  BLAS threads pinned to 1; candidate
removals fan out over a ProcessPoolExecutor.  Writes ``backward_select_result.txt``.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from experiments import directional_index_lib as di  # noqa: E402

START = int(os.environ.get('START', '42'))
STOP = int(os.environ.get('STOP', '28'))
_SEL0 = int(os.environ.get('SEL0', '0'))
_VER0 = int(os.environ.get('VER0', '10'))
_NSEED = int(os.environ.get('NSEED', '10'))
SELECT_SEEDS = list(range(_SEL0, _SEL0 + _NSEED))
VERIFY_SEEDS = list(range(_VER0, _VER0 + _NSEED))
TAG = os.environ.get('TAG', '')
FOLDS = 5
C = 10.0
THRESH = 0.940
POOL = os.environ.get('POOL', 'o6')
WORKERS = int(os.environ.get('WORKERS', '16'))

# module globals populated in each worker via initializer
_XTR = _YTR = None


def _family(split, name):
    fp = os.path.join(CACHE_DIR, f'{split}_difam_{name}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    x, _ = load_cached(split)
    feats, _ = di.FAMILIES[name](x)
    np.save(fp, feats)
    return feats


def build_pool(split):
    parts = [np.load(os.path.join(CACHE_DIR, f'{split}_feat_o6.npy'))]
    if POOL == 'o6+ixcoh':
        parts.append(_family(split, 'ixcoh'))
    elif POOL != 'o6':
        raise ValueError(POOL)
    return np.concatenate(parts, 1).astype(np.float32)


def _cv_seeds(x, y, idx, seeds):
    vals = []
    for sd in seeds:
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def _init(xtr, ytr):
    global _XTR, _YTR
    _XTR, _YTR = xtr, ytr


def _select_cv_drop(args):
    """Return SELECT-CV of the current feature set with feature at pos j removed."""
    cur_idx, j = args
    sub = np.delete(cur_idx, j)
    return j, _cv_seeds(_XTR, _YTR, sub, SELECT_SEEDS)


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr = build_pool('train')
    fva = build_pool('val')
    fte = build_pool('test')
    order = l1_rank(ftr, ytr)
    cur = order[:START].copy()

    tag = POOL.replace('+', '_') + (('_' + TAG) if TAG else '')
    outpath = os.path.join(os.path.dirname(__file__), f'backward_select_{tag}_result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    emit(f'POOL={POOL} dim={ftr.shape[1]} START={START} STOP={STOP} '
         f'SELECT={SELECT_SEEDS[0]}..{SELECT_SEEDS[-1]} '
         f'VERIFY={VERIFY_SEEDS[0]}..{VERIFY_SEEDS[-1]}')
    emit(f'{"k":>4} {"params":>7} {"selCV":>7} {"verCV":>7} {"val":>7} {"test":>7}'
         f'  {"dropped":>10}')

    trace = []
    ex = ProcessPoolExecutor(max_workers=WORKERS, initializer=_init, initargs=(ftr, ytr))
    # evaluate the starting set
    k = len(cur)
    while True:
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=cur, C=C)
        ver = _cv_seeds(ftr, ytr, cur, VERIFY_SEEDS)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        # SELECT-CV of the full current set
        sel_full = _cv_seeds(ftr, ytr, cur, SELECT_SEEDS)
        trace.append((len(cur), 10 * (len(cur) + 1), sel_full, ver, va, te, list(cur)))
        emit(f'{len(cur):>4} {10*(len(cur)+1):>7} {sel_full:>7.4f} {ver:>7.4f} '
             f'{va:>7.4f} {te:>7.4f}')
        if len(cur) <= STOP:
            break
        # find the feature whose removal maximises SELECT-CV
        results = list(ex.map(_select_cv_drop, [(cur, j) for j in range(len(cur))]))
        best_j, best_cv = max(results, key=lambda t: t[1])
        dropped = cur[best_j]
        cur = np.delete(cur, best_j)
    ex.shutdown()

    # honest floor: smallest k with verCV>=0.940 AND val>=0.940
    honest = [row for row in trace if row[3] >= THRESH and row[4] >= THRESH]
    emit('\n########## HONEST FLOOR (verCV>=0.940 AND val>=0.940) ##########')
    if honest:
        best = min(honest, key=lambda r: r[0])
        emit(f'k={best[0]} params={best[1]} selCV={best[2]:.4f} verCV={best[3]:.4f} '
             f'val={best[4]:.4f} test={best[5]:.4f}')
        emit('feature_idx=' + ','.join(map(str, best[6])))
    else:
        emit('none: no k in trace cleared both verCV>=0.940 and val>=0.940')
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
