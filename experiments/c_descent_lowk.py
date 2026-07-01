#!/usr/bin/env python
"""Honest per-C backward-greedy descent: does a CV-picked larger C lower the floor?

The C-per-k probe (``c_sweep_lowk.py``) showed the greedy's own guide CV (SELECT)
prefers a *larger* inverse-L2 strength than the C=10 every submission fixes:
selCV rises monotonically to ~C=20-50 and plateaus, and that larger C also widens
the k=25 val margin.  But the probe held the *subset* fixed at the one the C=10
greedy landed on.  The honest test is to re-run the WHOLE descent guided by the
larger C, so each low-k subset is the one that C actually prefers -- then read the
floor off the independent checks.

C is a single global hyperparameter.  It is chosen honestly by SELECT-CV (the
guide seeds), which is why we also print selCV per k so the reader can confirm the
larger C dominates C=10 along the whole curve.  The honest floor is, as always,
the smallest k with BOTH verCV>=0.940 (disjoint seeds) AND val>=0.940; test drives
no decision.  We run both seed partitions -- a floor is only trusted if the SAME C
lowers it on both.

Runs C in ``C_GRID`` x partitions {0,20}, START=42 -> STOP, one result file.
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
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import _init, _subset_score, mean_cv  # noqa: E402

START = int(os.environ.get('START', '42'))
STOP = int(os.environ.get('STOP', '22'))
FOLDS = 5
THRESH = 0.940
WORKERS = int(os.environ.get('WORKERS', '32'))
C_GRID = [float(c) for c in os.environ.get('C_GRID', '20,50').split(',')]
_ALL_PARTS = {0: (list(range(0, 10)), list(range(10, 20))),
              20: (list(range(20, 30)), list(range(30, 40)))}
# PART restricts to one seed partition so two descents can run concurrently
# (one per partition) without CPU oversubscription; default runs both.
_PART = os.environ.get('PART', '')
PARTITIONS = {int(_PART): _ALL_PARTS[int(_PART)]} if _PART else _ALL_PARTS
_SUFFIX = f'_p{_PART}' if _PART else ''


def descend(ftr, ytr, fva, yva, fte, yte, init_idx, sel_seeds, ver_seeds, C, emit):
    """Full backward-greedy descent at fixed C; return list of per-k rows."""
    cur = list(init_idx)
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init,
                             initargs=(ftr, ytr)) as ex:
        while True:
            idx = np.asarray(cur)
            sel = mean_cv(ftr, ytr, idx, sel_seeds, FOLDS, C)
            ver = mean_cv(ftr, ytr, idx, ver_seeds, FOLDS, C)
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            te = accuracy_score(yte, predict(fte, w, b, fi))
            rows.append((len(cur), 10 * (len(cur) + 1), sel, ver, va, te, list(cur)))
            emit(f'{len(cur):>4} {10*(len(cur)+1):>7} {sel:>7.4f} {ver:>7.4f} '
                 f'{va:>7.4f} {te:>7.4f}')
            if len(cur) <= STOP:
                break
            args = [(f, [c for c in cur if c != f], sel_seeds, FOLDS, C)
                    for f in cur]
            res = dict(ex.map(_subset_score, args))
            drop_f = max(res, key=res.get)
            cur = [c for c in cur if c != drop_f]
    return rows


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr = np.load(os.path.join(CACHE_DIR, 'train_feat_o6.npy')).astype(np.float32)
    fva = np.load(os.path.join(CACHE_DIR, 'val_feat_o6.npy')).astype(np.float32)
    fte = np.load(os.path.join(CACHE_DIR, 'test_feat_o6.npy')).astype(np.float32)

    order = l1_rank(ftr, ytr)
    init_idx = order[:START].copy()

    outpath = os.path.join(os.path.dirname(__file__),
                           f'c_descent_lowk{_SUFFIX}_result.txt')
    out = open(outpath, 'w')

    def emit(m):
        print(m, flush=True)
        out.write(m + '\n')
        out.flush()

    summary = []
    for part, (sel_seeds, ver_seeds) in PARTITIONS.items():
        for C in C_GRID:
            emit(f'\n===== partition {part} (SEL {sel_seeds[0]}..{sel_seeds[-1]} '
                 f'VER {ver_seeds[0]}..{ver_seeds[-1]})  C={C:g}  START={START} '
                 f'STOP={STOP} =====')
            emit(f'{"k":>4} {"params":>7} {"selCV":>7} {"verCV":>7} {"val":>7} '
                 f'{"test":>7}')
            rows = descend(ftr, ytr, fva, yva, fte, yte, init_idx,
                           sel_seeds, ver_seeds, C, emit)
            honest = [r for r in rows if r[3] >= THRESH and r[4] >= THRESH]
            if honest:
                b0 = min(honest, key=lambda r: r[0])
                emit(f'  honest floor: k={b0[0]} params={b0[1]} selCV={b0[2]:.4f} '
                     f'verCV={b0[3]:.4f} val={b0[4]:.4f} test={b0[5]:.4f}')
                summary.append((part, C, b0))
            else:
                emit('  honest floor: none cleared verCV>=0.940 AND val>=0.940')

    emit('\n########## SUMMARY: honest floor per (partition, C) ##########')
    for part, C, b0 in summary:
        emit(f'part={part} C={C:g} -> k={b0[0]} params={b0[1]} verCV={b0[3]:.4f} '
             f'val={b0[4]:.4f} test={b0[5]:.4f}')
        emit('   feature_idx=' + ','.join(map(str, b0[6])))
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
