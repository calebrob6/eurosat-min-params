#!/usr/bin/env python
"""Submission 13: minimum-parameter linear model at the >95% test bar.

Same o6+line+corn2 parameter-free pool as submission 11 (patch_features with
coherence/orient/xband/index-texture/Hough/Harris), same honest backward-greedy
selection, same reference-class (K-1 row) head as submission 12 -- but selected
for the RAISED 95% threshold. Because 95% must hold on TEST and val has sigma
~0.003, the selection gate is held at val >= 0.953 (= 0.95 + 1 sigma) AND both
disjoint verify-CV blocks >= 0.95, so the committed model clears test 0.95 with
margin rather than squeaking past a noisy gate.

Deployed head: 9 rows (class ref has an implicit 0 logit), params = 9*(k+1).
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.data import NUM_CLASSES  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import (fit_folded_logreg, l1_rank, predict,  # noqa: E402
                          predict_reference_class, to_reference_class)
from src.select import backward_eliminate, mean_cv  # noqa: E402

HERE = os.path.dirname(__file__)
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8, orient_hist_bins=4,
                spectral_peak=True, xband=True, index_texture=True,
                hough_lines=True, harris_corners=True)
VAL_GATE, VER_GATE = 0.953, 0.950   # val margin = 0.95 + ~1 sigma_val


def feats(split):
    fp = os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')
    lp = os.path.join(CACHE_DIR, f'{split}_linefam_line.npy')
    cp = os.path.join(CACHE_DIR, f'{split}_gs2fam_corn2.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if all(os.path.exists(p) for p in (fp, lp, cp, yp)):
        return (np.concatenate([np.load(fp), np.load(lp), np.load(cp)], 1)
                .astype(np.float32), np.load(yp))
    x, y = load_cached(split)
    f, _ = patch_features(x, **FEAT_CFG)
    return f.astype(np.float32), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', type=int, default=60)
    ap.add_argument('--kmin', type=int, default=26)
    ap.add_argument('--confirm-hi', type=int, default=34)
    ap.add_argument('--commit', type=int, default=None)
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--workers', type=int, default=12)
    args = ap.parse_args()

    ftr, ytr = feats('train'); fva, yva = feats('val'); fte, yte = feats('test')
    print(f'o6+line+corn2 pool: {ftr.shape[1]} feats; backward-greedy top-{args.start}'
          f' -> {args.kmin} (select {args.select_seeds}); gate val>={VAL_GATE} '
          f'verCVx2>={VER_GATE}', flush=True)

    order = l1_rank(ftr, ytr)
    _, _, subs = backward_eliminate(ftr, ytr, order[:args.start], args.kmin,
                                    select_seeds=args.select_seeds, C=args.C,
                                    workers=args.workers, record=True)
    print(f'\n{"k":>3} {"params":>6} {"verA":>7} {"verB":>7} {"val":>7} {"test":>7}  commit?')
    passing, info = [], {}
    for k in range(args.confirm_hi, args.kmin - 1, -1):
        if k not in subs:
            continue
        s = subs[k]
        w, b, fi = fit_folded_logreg(ftr, ytr, s, C=args.C)
        wr, br = to_reference_class(w, b, ref=0)
        va = accuracy_score(yva, predict_reference_class(fva, wr, br, fi))
        te = accuracy_score(yte, predict_reference_class(fte, wr, br, fi))
        vera = mean_cv(ftr, ytr, s, range(10, 20), C=args.C)
        verb = mean_cv(ftr, ytr, s, range(30, 40), C=args.C)
        ok = vera >= VER_GATE and verb >= VER_GATE and va >= VAL_GATE
        if ok:
            passing.append(k)
        info[k] = dict(wr=wr, br=br, fi=fi, va=va, te=te, vera=vera, verb=verb)
        print(f'{k:>3} {wr.size+br.size:>6} {vera:>7.4f} {verb:>7.4f} {va:>7.4f} '
              f'{te:>7.4f}{"  PASS" if ok else ""}', flush=True)

    commit_k = args.commit if args.commit is not None else (min(passing) if passing else None)
    if commit_k is None:
        print('\nNO k cleared the gate'); return
    d = info[commit_k]
    params = int(d['wr'].size + d['br'].size)
    assert params == (NUM_CLASSES - 1) * (commit_k + 1)
    print(f'\nCOMMIT k={commit_k} params={params} verA={d["vera"]:.4f} '
          f'verB={d["verb"]:.4f} val={d["va"]:.4f} test={d["te"]:.4f}')
    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=d['wr'], b=d['br'], feature_idx=d['fi'], ref_class=0, k=commit_k,
             C=args.C, verify_cv=d['vera'], verify_cv_b=d['verb'], val_acc=d['va'],
             test_acc=d['te'], params=params, **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
