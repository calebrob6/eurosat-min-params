#!/usr/bin/env python
"""Submission 08: 290-parameter linear classifier (backward-greedy, k=28).

Identical method and feature pool to submission 07 -- the zero-parameter 305-dim
``o6`` pool and ``src.select.backward_eliminate`` backward-greedy selection -- but
pushed one honest step further down the CV curve, from k=30 (310 params) to
**k=28 (290 params)**.

Submission 07 committed to k=30 only for a *comfortable* val margin; its own
honest-floor table already showed k=28 clears the bar (VERIFY-CV 0.9462, val
0.9417, test 0.9487) on BOTH independent seed partitions.  Submission 08 lands
that floor: a 20-parameter (6.5%) cut with test unchanged at 0.9487.

Selection is deterministic: L1-rank the o6 pool, take the top-42 as the starting
set, then repeatedly drop the single feature whose removal least hurts the mean
5-fold train-CV over SELECT seeds {0..9}, re-fitting after every drop, down to
k=28.  Honesty protocol (never trust the greedy's own CV):
  * SELECT seeds {0..9}  drive the greedy removals.
  * VERIFY seeds {10..19}  -- never used to choose a feature -- give an unbiased CV.
  * val (held out, different images) is a third independent check.
  * test is reported once and drives NO decision.
The deployed model is a single affine map ``logits = x[:, idx] @ W.T + b`` on the
28 selected raw features, so params == 10*(28+1) == 290.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, num_params, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402

HERE = os.path.dirname(__file__)
# identical zero-parameter feature configuration to submissions 06 and 07 (o6 pool)
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8,
                orient_hist_bins=4, spectral_peak=True, xband=True,
                index_texture=True)


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """305-dim o6 pool (reuses the submission-06/07 cache when present)."""
    fp = os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if os.path.exists(fp) and os.path.exists(yp):
        return np.load(fp), np.load(yp)
    x, y = load_cached(split)
    f, _ = patch_features(x, **FEAT_CFG)
    np.save(fp, f)
    np.save(yp, y)
    return f, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=28, help='target feature count')
    ap.add_argument('--start', type=int, default=42, help='L1 top-N to start from')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--verify-seeds', type=int, nargs='+', default=list(range(10, 20)))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (o6 pool, same as submissions 06/07)')

    order = l1_rank(ftr, ytr)
    init = order[:args.start]
    print(f'backward-greedy from L1 top-{args.start} down to k={args.k} '
          f'(select seeds {args.select_seeds[0]}..{args.select_seeds[-1]})')
    fi, trace = backward_eliminate(
        ftr, ytr, init, args.k, select_seeds=args.select_seeds,
        C=args.C, workers=args.workers)
    for kk, cv in trace:
        print(f'  k={kk:>3} selCV={cv:.4f}')

    # honest checks on the final subset
    w, b, feat_idx = fit_folded_logreg(ftr, ytr, feature_idx=fi, C=args.C)
    ver = mean_cv(ftr, ytr, fi, args.verify_seeds, C=args.C)
    va = accuracy_score(yva, predict(fva, w, b, feat_idx))
    te = accuracy_score(yte, predict(fte, w, b, feat_idx))
    p = num_params(w, b)
    sel = trace[-1][1]
    print(f'\nk={args.k} params={p} selCV={sel:.4f} verifyCV={ver:.4f} '
          f'val={va:.4f} test={te:.4f}')

    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=feat_idx, k=args.k, C=args.C,
             sel_cv=sel, verify_cv=ver, val_acc=va, test_acc=te, params=p,
             **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
