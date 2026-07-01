#!/usr/bin/env python
"""Submission 10: sub-260-parameter linear classifier (o6 + global-line pool).

Extends the submission-06..09 zero-parameter pool with a NEW orthogonal
parameter-free family -- ``hough_line_features`` (global straight-line / Hough
statistics on the pan, NDVI and NDBI channels) -- and re-runs the same honest
backward-greedy elimination on the augmented 314-dim ``o6+line`` pool.

Why a new family finally moved the floor.  Every directional feature already in
the pool (structure-tensor coherence, gradient-orientation entropy/histogram and
their index-map variants) is *local* -- an aggregate of per-pixel gradient
directions.  Crop rows (many short parallel edges) and a single long streak (a
Highway, a River) look identical to those.  The Hough family measures *global
collinearity*: a run of L collinear edge pixels makes an accumulator peak of value
L, so the peak is the length of the single longest straight line in the patch.
Added to the frontier k=25 subset it lifts held-out val 0.9404 -> 0.9446 (+0.004,
well above val noise) -- the val lift iterations 9-12 established was the binding
constraint below 260 params -- so the honest floor drops below k=25.

Selection is deterministic and replicates experiments/backward_select.py with
POOL=o6+line, FORCE_IX=1:
  * L1-rank the 314-dim o6+line pool; take the top-42 as the starting set, then
    UNION in the 9 line-family indices (they rank 39..155 in L1, so only 1 sits in
    the top-42 -- forcing the family in is required for the greedy to consider it).
  * Repeatedly drop the single feature whose removal least hurts the mean 5-fold
    train-CV over SELECT seeds {0..9}, re-fitting after every drop, down to k.
Honesty protocol (never trust the greedy's own CV):
  * SELECT seeds {0..9}   drive the greedy removals (optimistically biased).
  * VERIFY seeds {10..19}  -- never used to choose a feature -- give an unbiased CV.
  * val (held out, different images) is a third independent check.
  * test is reported once and drives NO decision.
The deployed model is a single affine map ``logits = x[:, idx] @ W.T + b`` on the
k selected raw features, so params == 10*(k+1).
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
# zero-parameter feature configuration: the submission-06..09 o6 pool (305 feats)
# PLUS the iteration-13 global-line family (9 feats) -> 314-dim o6+line pool.
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8,
                orient_hist_bins=4, spectral_peak=True, xband=True,
                index_texture=True, hough_lines=True)
O6_DIM = 305          # first 305 columns are the o6 pool; [305:314] are the line family


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """314-dim o6+line pool, reusing the o6 and line caches when present."""
    fp = os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')
    lp = os.path.join(CACHE_DIR, f'{split}_linefam_line.npy')
    yp = os.path.join(CACHE_DIR, f'{split}_y.npy')
    if os.path.exists(fp) and os.path.exists(lp) and os.path.exists(yp):
        return np.concatenate([np.load(fp), np.load(lp)], 1).astype(np.float32), np.load(yp)
    x, y = load_cached(split)
    f, _ = patch_features(x, **FEAT_CFG)
    return f.astype(np.float32), y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=23, help='target feature count')
    ap.add_argument('--start', type=int, default=42, help='L1 top-N to start from')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--verify-seeds', type=int, nargs='+', default=list(range(10, 20)))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (o6+line pool: 305 o6 + 9 line)')

    # replicate FORCE_IX: top-`start` UNION the 9 line-family indices
    order = l1_rank(ftr, ytr)
    init = order[:args.start].tolist()
    line_idx = list(range(O6_DIM, ftr.shape[1]))
    init = np.array(init + [f for f in line_idx if f not in set(init)])
    print(f'backward-greedy from L1 top-{args.start} UNION 9 line feats '
          f'(|init|={len(init)}) down to k={args.k} '
          f'(select seeds {args.select_seeds[0]}..{args.select_seeds[-1]})')
    fi, trace = backward_eliminate(
        ftr, ytr, init, args.k, select_seeds=args.select_seeds,
        C=args.C, workers=args.workers)
    n_line = int(np.sum(np.asarray(fi) >= O6_DIM))
    for kk, cv in trace:
        print(f'  k={kk:>3} selCV={cv:.4f}')

    # honest checks on the final subset
    w, b, feat_idx = fit_folded_logreg(ftr, ytr, feature_idx=fi, C=args.C)
    ver = mean_cv(ftr, ytr, fi, args.verify_seeds, C=args.C)
    va = accuracy_score(yva, predict(fva, w, b, feat_idx))
    te = accuracy_score(yte, predict(fte, w, b, feat_idx))
    p = num_params(w, b)
    sel = trace[-1][1]
    print(f'\nk={args.k} params={p} n_line={n_line} selCV={sel:.4f} '
          f'verifyCV={ver:.4f} val={va:.4f} test={te:.4f}')

    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=feat_idx, k=args.k, C=args.C,
             sel_cv=sel, verify_cv=ver, val_acc=va, test_acc=te, params=p,
             n_line=n_line, **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
