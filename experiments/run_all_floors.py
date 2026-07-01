#!/usr/bin/env python
"""Run the CV>=0.940 floor sweep for several feature pools and print a summary.

Runs each candidate pool in a separate process (BLAS threads pinned to 1 to avoid
oversubscription) and prints one FLOOR line per pool plus a final ranked table.
Pass pool specs as CLI args: each spec is a '+'-joined list of family names (or
'base' for the bare 195-pool).  Example:

    python run_all_floors.py base hogpan hog4 coh3 oent fft hogpan+coh3
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
from experiments import new_features_lib as nf  # noqa: E402

FAMILIES = {
    'hogpan': lambda x: nf.hog_pan(x, nbins=8, scales=2),
    'hog4': lambda x: nf.hog_features(x, nbins=4, scales=1),
    'coh3': lambda x: nf.coherence_scale3(x),
    'oent': lambda x: nf.orient_entropy(x, nbins=8),
    'fft': lambda x: nf.fft_periodicity(x),
}

KS = [40, 45]
SEEDS = list(range(10))
FOLDS = 5
C = 10.0
THRESH = 0.940


def _family(split, name):
    fp = os.path.join(CACHE_DIR, f'{split}_fam_{name}.npy')
    if os.path.exists(fp):
        return np.load(fp)
    x, _ = load_cached(split)
    feats, _ = FAMILIES[name](x)
    np.save(fp, feats)
    return feats


def _pool(split, extra):
    parts = [np.load(os.path.join(CACHE_DIR, f'{split}_feat_coh.npy'))]
    for name in extra:
        parts.append(_family(split, name))
    return np.concatenate(parts, 1)


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
    specs = sys.argv[1:] or ['base', 'hogpan', 'hog4', 'coh3', 'oent', 'fft']
    outpath = os.path.join(os.path.dirname(__file__), 'floors_result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    results = []
    with ProcessPoolExecutor(max_workers=min(len(specs), 6)) as ex:
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
        emit(f'{spec:20s} floor params={pk[1]} k={pk[0]} '
             f'cv={pk[2]:.4f} val={pk[3]:.4f} test={pk[4]:.4f}')
    nofloor = [s for s, d, r, pk in results if pk is None]
    if nofloor:
        emit('no-floor (cv<0.940 at all k): ' + ', '.join(nofloor))
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
