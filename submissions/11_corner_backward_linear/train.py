#!/usr/bin/env python
"""Submission 11: sub-240-parameter linear classifier (o6 + line + corner pool).

Extends the submission-10 ``o6+line`` pool with a NEW orthogonal parameter-free
family -- ``harris_corner_features`` (Harris corner/junction density on the pan,
NDVI and NDBI channels) -- and re-runs the same honest backward-greedy
elimination on the augmented 320-dim ``o6+line+corn2`` pool.

Why corners move the floor.  Corners occur where edges MEET -- a global-layout
axis nothing else in the pool sees.  Local directional statistics (coherence,
orientation entropy/histogram, their index-map variants) aggregate isolated
per-pixel gradient directions, and the Hough family sees straight lines; neither
separates a junction grid (Residential / Industrial blocks) from parallel rows
that never cross (AnnualCrop).  The frozen-subset pre-screen (the iteration-13
qualifier that predicted the line family's win) is even stronger here: every one
of the 6 corner features lifts held-out val on the frozen submission-10 subset,
the best (corn2mag_ndbi) by +0.0046 -- more than the +0.0042 the line family
showed before cutting the floor 260 -> 240.

Selection is deterministic and replicates experiments/backward_select.py with
POOL=o6+line+corn2, FORCE_IX=1:
  * L1-rank the 320-dim pool; take the top-42 as the starting set, then UNION in
    the 15 line+corner family indices [305, 320) (most rank below the top-42 --
    forcing the families in is required for the greedy to consider them).
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
# zero-parameter feature configuration: the submission-10 o6+line pool (314
# feats) PLUS the iteration-15 Harris corner family (6 feats) -> 320-dim pool.
FEAT_CFG = dict(coherence_scales=2, orient_entropy_bins=8,
                orient_hist_bins=4, spectral_peak=True, xband=True,
                index_texture=True, hough_lines=True, harris_corners=True)
O6_DIM = 305          # [0:305] o6 pool; [305:314] line family; [314:320] corner


def _feats(split: str) -> tuple[np.ndarray, np.ndarray]:
    """320-dim o6+line+corn2 pool, reusing the family caches when present."""
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=21, help='target feature count')
    ap.add_argument('--start', type=int, default=42, help='L1 top-N to start from')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--verify-seeds', type=int, nargs='+', default=list(range(10, 20)))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (o6+line+corn2 pool: 305 o6 + 9 line + 6 corner)')

    # replicate FORCE_IX: top-`start` UNION the 15 line+corner family indices
    order = l1_rank(ftr, ytr)
    init = order[:args.start].tolist()
    fam_idx = list(range(O6_DIM, ftr.shape[1]))
    init = np.array(init + [f for f in fam_idx if f not in set(init)])
    print(f'backward-greedy from L1 top-{args.start} UNION 15 line+corner feats '
          f'(|init|={len(init)}) down to k={args.k} '
          f'(select seeds {args.select_seeds[0]}..{args.select_seeds[-1]})')
    fi, trace = backward_eliminate(
        ftr, ytr, init, args.k, select_seeds=args.select_seeds,
        C=args.C, workers=args.workers)
    n_line = int(np.sum((np.asarray(fi) >= O6_DIM) & (np.asarray(fi) < 314)))
    n_corn = int(np.sum(np.asarray(fi) >= 314))
    for kk, cv in trace:
        print(f'  k={kk:>3} selCV={cv:.4f}')

    # honest checks on the final subset
    w, b, feat_idx = fit_folded_logreg(ftr, ytr, feature_idx=fi, C=args.C)
    ver = mean_cv(ftr, ytr, fi, args.verify_seeds, C=args.C)
    va = accuracy_score(yva, predict(fva, w, b, feat_idx))
    te = accuracy_score(yte, predict(fte, w, b, feat_idx))
    p = num_params(w, b)
    sel = trace[-1][1]
    print(f'\nk={args.k} params={p} n_line={n_line} n_corn={n_corn} '
          f'selCV={sel:.4f} verifyCV={ver:.4f} val={va:.4f} test={te:.4f}')

    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=w, b=b, feature_idx=feat_idx, k=args.k, C=args.C,
             sel_cv=sel, verify_cv=ver, val_acc=va, test_acc=te, params=p,
             n_line=n_line, n_corn=n_corn, **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
