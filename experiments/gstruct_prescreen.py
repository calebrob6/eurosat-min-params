#!/usr/bin/env python
"""Frozen-subset pre-screen for the global-structure (gstruct) candidate families.

The iteration-13 qualifier: add ONE candidate feature to the FROZEN frontier
subset (submission 10, k=23 on the o6+line pool, val 0.9404) and read the
HELD-OUT VAL lift.  The line family lifted val 0.9404 -> 0.9446 and went on to
cut the floor; ixcoh never lifted val and regressed the floor.  A candidate
family is only worth the ~90-min forced two-partition descent if at least one
of its features lifts val here.

Candidates are the 24 cached gstruct features (4 families x pan/NDVI/NDBI):
  blob      bloblrg/blobnc/blobmsz   -- connected-component region granularity
  specslope specslope_*              -- radial spectral (FFT) slope
  corner    cornfrac/cornmag_*       -- Harris-style corner/junction density
  lbp       lbpent/lbpuni_*          -- local binary pattern entropy/uniformity

For each candidate: fit folded logreg (C=10, matching submission 10) on
frozen-23 + candidate, report val accuracy and a 5-seed 5x5-fold train-CV.
Writes gstruct_prescreen_result.txt.
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

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402

C = 10.0
CV_SEEDS = list(range(5))
FOLDS = 5

_XTR = _YTR = None


# GS=old screens the lost-cache iteration-14 features; GS=gs2 (default) screens
# the reproducible reimplementations in gstruct2_features_lib.
GS = os.environ.get('GS', 'gs2')
GS2_FAMS = ['blob2', 'corn2', 'lbp2', 'sslope2']


def gs_block(split):
    if GS == 'old':
        names = open(os.path.join(CACHE_DIR, 'gstruct_names.txt')).read().split()
        return np.load(os.path.join(CACHE_DIR, f'{split}_gstruct.npy')), names
    import experiments.gstruct2_features_lib as g2
    feats, names = [], []
    for fam in GS2_FAMS:
        feats.append(np.load(os.path.join(CACHE_DIR, f'{split}_gs2fam_{fam}.npy')))
        names += g2.FAMILY_NAMES[fam]
    return np.concatenate(feats, 1), names


def build_pool(split):
    o6 = np.load(os.path.join(CACHE_DIR, f'{split}_feat_o6.npy'))
    line = np.load(os.path.join(CACHE_DIR, f'{split}_linefam_line.npy'))
    gs, _ = gs_block(split)
    return np.concatenate([o6, line, gs], 1).astype(np.float32)


def _cv(x, y, idx):
    vals = []
    for sd in CV_SEEDS:
        skf = StratifiedKFold(FOLDS, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def _init(xtr, ytr):
    global _XTR, _YTR
    _XTR, _YTR = xtr, ytr


def _cv_job(idx):
    return _cv(_XTR, _YTR, np.asarray(idx))


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    ftr, fva = build_pool('train'), build_pool('val')
    _, gnames = gs_block('val')
    o6line_dim = ftr.shape[1] - len(gnames)  # 314: candidates occupy [314, dim)

    frozen = np.load(os.path.join(os.path.dirname(__file__), '..', 'submissions',
                                  '10_line_backward_linear', 'model.npz'))['feature_idx']
    assert len(frozen) == 23 and frozen.max() < o6line_dim

    cands = [frozen] + [np.concatenate([frozen, [o6line_dim + g]])
                        for g in range(len(gnames))]
    labels = ['frozen23 (baseline)'] + gnames

    outpath = os.path.join(os.path.dirname(__file__),
                           f'gstruct_prescreen_{GS}_result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    # held-out val first (the decisive check, one fit each)
    vals = []
    for idx in cands:
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
        vals.append(accuracy_score(yva, predict(fva, w, b, fi)))

    with ProcessPoolExecutor(max_workers=13, initializer=_init,
                             initargs=(ftr, ytr)) as ex:
        cvs = list(ex.map(_cv_job, [list(c) for c in cands]))

    base_val, base_cv = vals[0], cvs[0]
    emit(f'{"candidate":>18} {"val":>7} {"dval":>8} {"cv5":>7} {"dcv":>8}')
    for lab, v, cv in zip(labels, vals, cvs):
        emit(f'{lab:>18} {v:>7.4f} {v-base_val:>+8.4f} {cv:>7.4f} {cv-base_cv:>+8.4f}')
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
