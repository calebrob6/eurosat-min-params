#!/usr/bin/env python
"""C-per-k probe: does the logreg inverse-L2 strength move the low-k floor?

Every submission so far fixes ``C=10`` in both the CV that guides the greedy and
the final fold.  At the k=25 honest floor the val margin is razor-thin (+0.0004)
and k=24 fails val by ~0.0005-0.0013 on both partitions -- so if a *different* C
generalises better at low k it could widen the k=25 margin or legitimise k=24
(250 params).  This script is the cheap first probe: take the *fixed* subsets the
C=10 backward-greedy descent already landed on (from ``extend_lowk_*``) and, for a
grid of C, measure the four honest metrics.

Honesty protocol (unchanged): SELECT seeds are the greedy's guide CV; VERIFY seeds
(disjoint) + val are the independent checks; test drives no decision.  C is a
hyperparameter, so the honest way to *pick* it is by SELECT-CV, then confirm
verCV/val/test>=0.940 -- and confirm the SAME C on the other seed partition.

Caveat: these subsets were chosen by the C=10 greedy, so this probe only shows
whether the *regularisation strength alone* helps on the C=10 subset.  If it does,
``c_descent_lowk.py`` re-runs the whole descent per C for a fully honest floor.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402

# Fixed subsets from the C=10 pure-backward descent (experiments/extend_lowk_*).
# k=24 = k=25 minus the feature the C=10 greedy drops next (300 on p0, 91 on p20).
SUBSETS = {
    0: {
        25: [291, 285, 145, 297, 181, 93, 92, 111, 294, 195, 275, 33, 287, 68,
             196, 203, 188, 75, 86, 95, 273, 300, 304, 65, 295],
        24: [291, 285, 145, 297, 181, 93, 92, 111, 294, 195, 275, 33, 287, 68,
             196, 203, 188, 75, 86, 95, 273, 304, 65, 295],
    },
    20: {
        25: [291, 285, 297, 181, 93, 144, 92, 111, 294, 195, 275, 278, 287, 58,
             68, 196, 203, 188, 75, 86, 91, 95, 273, 65, 295],
        24: [291, 285, 297, 181, 93, 144, 92, 111, 294, 195, 275, 278, 287, 58,
             68, 196, 203, 188, 75, 86, 95, 273, 65, 295],
    },
}

PARTITIONS = {0: (list(range(0, 10)), list(range(10, 20))),
              20: (list(range(20, 30)), list(range(30, 40)))}

C_GRID = [0.3, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
FOLDS = 5
THRESH = 0.940
WORKERS = int(os.environ.get('WORKERS', '18'))

_D = {}


def _init(ftr, ytr, fva, yva, fte, yte):
    _D['ftr'], _D['ytr'] = ftr, ytr
    _D['fva'], _D['yva'] = fva, yva
    _D['fte'], _D['yte'] = fte, yte


def _eval(args):
    part, k, idx, C = args
    sel_seeds, ver_seeds = PARTITIONS[part]
    ftr, ytr = _D['ftr'], _D['ytr']
    idx = np.asarray(idx)
    sel = mean_cv(ftr, ytr, idx, sel_seeds, FOLDS, C)
    ver = mean_cv(ftr, ytr, idx, ver_seeds, FOLDS, C)
    w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
    va = accuracy_score(_D['yva'], predict(_D['fva'], w, b, fi))
    te = accuracy_score(_D['yte'], predict(_D['fte'], w, b, fi))
    return (part, k, C, sel, ver, va, te)


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr = np.load(os.path.join(CACHE_DIR, 'train_feat_o6.npy')).astype(np.float32)
    fva = np.load(os.path.join(CACHE_DIR, 'val_feat_o6.npy')).astype(np.float32)
    fte = np.load(os.path.join(CACHE_DIR, 'test_feat_o6.npy')).astype(np.float32)

    tasks = []
    for part, ks in SUBSETS.items():
        for k, idx in ks.items():
            for C in C_GRID:
                tasks.append((part, k, idx, C))

    outpath = os.path.join(os.path.dirname(__file__), 'c_sweep_lowk_result.txt')
    with open(outpath, 'w') as out:
        def emit(m):
            print(m, flush=True)
            out.write(m + '\n')
            out.flush()

        emit('C-PER-K PROBE on fixed C=10 backward-greedy subsets (o6 pool)')
        emit(f'{"part":>4} {"k":>3} {"params":>6} {"C":>6} {"selCV":>7} '
             f'{"verCV":>7} {"val":>7} {"test":>7}  honest?')

        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init,
                                 initargs=(ftr, ytr, fva, yva, fte, yte)) as ex:
            rows = list(ex.map(_eval, tasks))

        rows.sort(key=lambda r: (r[0], -r[1], r[2]))
        for part, k, C, sel, ver, va, te in rows:
            honest = 'YES' if (ver >= THRESH and va >= THRESH) else ''
            emit(f'{part:>4} {k:>3} {10*(k+1):>6} {C:>6g} {sel:>7.4f} '
                 f'{ver:>7.4f} {va:>7.4f} {te:>7.4f}  {honest}')

        # For each (part, k), the C picked honestly by SELECT-CV, and its checks.
        emit('\n### C chosen by SELECT-CV (honest pick), per (part,k) ###')
        by_pk = {}
        for r in rows:
            by_pk.setdefault((r[0], r[1]), []).append(r)
        for (part, k), rs in sorted(by_pk.items(), key=lambda t: (t[0][0], -t[0][1])):
            best = max(rs, key=lambda r: r[3])  # max selCV
            p, kk, C, sel, ver, va, te = best
            ok = 'HONEST' if (ver >= THRESH and va >= THRESH) else 'fails'
            emit(f'part={part} k={k} params={10*(k+1)} -> C*={C:g} '
                 f'selCV={sel:.4f} verCV={ver:.4f} val={va:.4f} test={te:.4f} [{ok}]')
        emit('ALL_DONE')


if __name__ == '__main__':
    main()
