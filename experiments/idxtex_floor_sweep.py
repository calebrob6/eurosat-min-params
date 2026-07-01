#!/usr/bin/env python
"""Floor sweep for index-texture families on top of the submission-05 pool.

Base pool = the 281-feature o5 pool (spectral + gradient texture + coherence +
orientation entropy/histogram + FFT peak + cross-band correlation, cached as
``{split}_feat_o5.npy``).  For each candidate spec we append one or more
index-texture families (index_texture, index_entropy, index_texture_scale2),
L1-rank the combined pool, and report the honest multi-seed train-CV floor: the
smallest k in ``KS`` whose mean 10x5-fold CV accuracy >= 0.940.  The goal is to
reach 0.940 below submission 05's k=38 (390 params).

Each spec runs in its own process with BLAS threads pinned to 1.  Results also
written to ``idxtex_floors_result.txt``.
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
from experiments import index_texture_lib as it  # noqa: E402

KS = [int(v) for v in os.environ.get('SWEEP_KS', '34,35,36,37,38,40').split(',')]
SEEDS = list(range(10))
FOLDS = 5
C = 10.0
THRESH = 0.940


def _family(split, name):
    fp = os.path.join(CACHE_DIR, f'{split}_ixfam_{name}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    x, _ = load_cached(split)
    feats, _ = it.FAMILIES[name](x)
    np.save(fp, feats)
    return feats


def _pool(split, extra):
    parts = [np.load(os.path.join(CACHE_DIR, f'{split}_feat_o5.npy'))]
    for name in extra:
        parts.append(_family(split, name))
    return np.concatenate(parts, 1).astype(np.float32)


def _cv(x, y, idx):
    vals = []
    for sd in SEEDS:
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def run_spec(spec):
    extra = [] if spec == 'base' else spec.split('+')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr = _pool('train', extra)
    fva = _pool('val', extra)
    fte = _pool('test', extra)
    order = l1_rank(ftr, ytr)
    rows = []
    picked = None
    for k in KS:
        idx = order[:k]
        cv = _cv(ftr, ytr, idx)
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        rows.append((k, 10 * (k + 1), cv, va, te))
        if picked is None and cv >= THRESH:
            picked = (k, 10 * (k + 1), cv, va, te)
    return spec, ftr.shape[1], rows, picked


def main():
    specs = sys.argv[1:] or [
        'base', 'ixtex', 'ixent', 'ixtex2',
        'ixtex+ixent', 'ixtex+ixtex2', 'ixtex+ixent+ixtex2',
    ]
    outpath = os.path.join(os.path.dirname(__file__), 'idxtex_floors_result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    results = []
    with ProcessPoolExecutor(max_workers=min(len(specs), 8)) as ex:
        for res in ex.map(run_spec, specs):
            spec, dim, rows, picked = res
            results.append(res)
            emit(f'\n=== {spec}  (dim={dim}) ===')
            emit(f'{"k":>4} {"params":>7} {"cv":>7} {"val":>7} {"test":>7}')
            for k, p, cv, va, te in rows:
                mark = '  <-' if picked and k == picked[0] else ''
                emit(f'{k:>4} {p:>7} {cv:>7.4f} {va:>7.4f} {te:>7.4f}{mark}')
            if picked:
                emit(f'FLOOR {spec}: k={picked[0]} params={picked[1]} '
                     f'cv={picked[2]:.4f} val={picked[3]:.4f} test={picked[4]:.4f}')
            else:
                emit(f'FLOOR {spec}: none reached cv>={THRESH}')

    emit('\n\n########## SUMMARY (sorted by params) ##########')
    ranked = []
    for spec, dim, rows, picked in results:
        if picked:
            ranked.append((picked[1], spec, picked))
    for p, spec, pk in sorted(ranked):
        emit(f'{spec:24s} floor params={pk[1]} k={pk[0]} '
             f'cv={pk[2]:.4f} val={pk[3]:.4f} test={pk[4]:.4f}')
    nofloor = [s for s, d, r, pk in results if pk is None]
    if nofloor:
        emit('no-floor (cv<0.940 at all k): ' + ', '.join(nofloor))
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
