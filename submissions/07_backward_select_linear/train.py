#!/usr/bin/env python
"""Submission 07: sub-350-parameter linear classifier via backward-greedy selection.

Same zero-parameter feature pool as submission 06 (the 305-feature ``o6`` pool:
spectral + multi-scale gradient texture + coherence + orientation entropy/hist +
FFT peak + cross-band correlation + index-map texture).  **No new features** --
the entire win comes from a better feature *selector*.

Submission 06 selected features by L1-rank ``top-k`` order, a one-shot proxy that
never reconsiders a ranking.  Here we use **backward-greedy elimination**
(``src.select.backward_eliminate``): start from the L1 top-42 and repeatedly drop
the single feature whose removal least hurts the mean 5-fold train-CV, re-fitting
after every drop.  Re-evaluating the whole remaining set at each step finds much
smaller subsets that still clear 0.940 -- the honest floor falls from k=34 (350
params) to k=30 (310 params), an 11%% parameter cut, with no accuracy loss (test
actually *rises* to 0.9494 because the retained features generalise better).

Honesty protocol (backward-greedy on CV can overfit the CV, so we never trust the
selection CV alone):
  * SELECT seeds {0..9} drive the greedy removals.
  * VERIFY seeds {10..19} -- never used to choose a feature -- give an unbiased
    CV estimate of the resulting subset.
  * val (fully held out, different images) is a third independent check.
  * test is reported once at the end and used for NO decision.
The floor k is the target where the VERIFY-CV and val both clear 0.940 with margin
(an independent-seed re-run, SELECT {20..29}/VERIFY {30..39}, confirmed the same
floor -- see the README).  The deployed model is a single affine map
``logits = x[:, idx] @ W.T + b`` on the selected raw features, so params == 10*(k+1).
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
# identical zero-parameter feature configuration to submission 06 (the o6 pool)
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8,
                orient_hist_bins=4, spectral_peak=True, xband=True,
                index_texture=True)


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """305-dim o6 pool (reuses submission-06's cache when present)."""
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
    ap.add_argument('--k', type=int, default=30, help='target feature count')
    ap.add_argument('--start', type=int, default=42, help='L1 top-N to start from')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--verify-seeds', type=int, nargs='+', default=list(range(10, 20)))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (o6 pool, same as submission 06)')

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
