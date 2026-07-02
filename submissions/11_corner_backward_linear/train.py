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


CORN_LO = 314         # corner family occupies [314:320]
THRESH = 0.940


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--k-min', type=int, default=18, help='descend to this k')
    ap.add_argument('--confirm-hi', type=int, default=24,
                    help='confirm every k in [k-min .. confirm-hi]')
    ap.add_argument('--commit', type=int, default=None,
                    help='force-commit this k instead of the auto-detected floor')
    ap.add_argument('--start', type=int, default=42, help='L1 top-N to start from')
    ap.add_argument('--warm-from', type=str, default=None,
                    help='model.npz whose feature_idx seeds a WARM START: the frozen '
                         'frontier subset UNION ONLY the new corner family. sub10 '
                         'already settled the line selection, so re-adding line just '
                         're-opens settled questions; adding only the new family is '
                         'the clean incremental "does corner let the frontier go '
                         'lower?" question -- and far less descent than the top-N start.')
    ap.add_argument('--C', type=float, default=10.0)
    ap.add_argument('--select-seeds', type=int, nargs='+', default=list(range(10)))
    ap.add_argument('--verify-a', type=int, nargs='+', default=list(range(10, 20)))
    ap.add_argument('--verify-b', type=int, nargs='+', default=list(range(30, 40)))
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()

    ftr, ytr = _feats('train')
    fva, yva = _feats('val')
    fte, yte = _feats('test')
    print(f'features: {ftr.shape[1]} (o6+line+corn2 pool: 305 o6 + 9 line + 6 corner)')

    fam_idx = list(range(O6_DIM, ftr.shape[1]))
    if args.warm_from:
        base = np.load(args.warm_from)['feature_idx'].astype(int).tolist()
        init = np.array(sorted(set(base + list(range(CORN_LO, ftr.shape[1])))))
        print(f'WARM START from {os.path.relpath(args.warm_from)} ({len(base)} frozen '
              f'feats) UNION {ftr.shape[1]-CORN_LO} NEW corner feats (|init|={len(init)}) '
              f'down to k={args.k_min} (select seeds '
              f'{args.select_seeds[0]}..{args.select_seeds[-1]})')
    else:
        # replicate FORCE_IX: L1 top-`start` UNION the 15 line+corner family indices
        order = l1_rank(ftr, ytr)
        init = order[:args.start].tolist()
        init = np.array(init + [f for f in fam_idx if f not in set(init)])
        print(f'backward-greedy from L1 top-{args.start} UNION {len(fam_idx)} '
              f'line+corner feats (|init|={len(init)}) down to k={args.k_min} '
              f'(select seeds {args.select_seeds[0]}..{args.select_seeds[-1]})')

    _, trace, subsets = backward_eliminate(
        ftr, ytr, init, args.k_min, select_seeds=args.select_seeds,
        C=args.C, workers=args.workers, record=True)
    sel_by_k = dict((kk, cv) for kk, cv in trace)

    # Confirm each k in the neighbourhood on TWO disjoint verify blocks (neither
    # used to select) + held-out val. The honest floor is the smallest k that
    # clears THRESH on verify-A, verify-B AND val -- committing a subset validated
    # two ways beyond the greedy's own seeds.
    print(f'\n{"k":>3} {"params":>6} {"nCorn":>5} {"selCV":>7} {"verA":>7} {"verB":>7}'
          f' {"val":>7} {"test":>7}  floor?')
    passing = []
    info = {}
    for k in range(args.confirm_hi, args.k_min - 1, -1):
        if k not in subsets:
            continue
        s = subsets[k]
        w, b, fidx = fit_folded_logreg(ftr, ytr, feature_idx=s, C=args.C)
        va = accuracy_score(yva, predict(fva, w, b, fidx))
        te = accuracy_score(yte, predict(fte, w, b, fidx))
        vera = mean_cv(ftr, ytr, s, args.verify_a, C=args.C)
        verb = mean_cv(ftr, ytr, s, args.verify_b, C=args.C)
        ncorn = int(np.sum(np.asarray(s) >= CORN_LO))
        ok = vera >= THRESH and verb >= THRESH and va >= THRESH
        if ok:
            passing.append(k)
        info[k] = dict(w=w, b=b, fidx=fidx, va=va, te=te, vera=vera, verb=verb,
                       ncorn=ncorn, sel=sel_by_k.get(k, -1.0), params=num_params(w, b))
        print(f'{k:>3} {num_params(w,b):>6} {ncorn:>5} {sel_by_k.get(k,-1):>7.4f} '
              f'{vera:>7.4f} {verb:>7.4f} {va:>7.4f} {te:>7.4f}  {"PASS" if ok else "fail"}')

    commit_k = args.commit if args.commit is not None else (min(passing) if passing else None)
    if commit_k is None:
        print('\nNO k in the neighbourhood cleared verify-A, verify-B AND val >= 0.940')
        return
    d = info[commit_k]
    n_line = int(np.sum((d['fidx'] >= O6_DIM) & (d['fidx'] < CORN_LO)))
    print(f'\nCOMMIT k={commit_k} params={d["params"]} n_line={n_line} n_corn={d["ncorn"]} '
          f'selCV={d["sel"]:.4f} verifyA={d["vera"]:.4f} verifyB={d["verb"]:.4f} '
          f'val={d["va"]:.4f} test={d["te"]:.4f}')
    out = os.path.join(HERE, 'model.npz')
    np.savez(out, W=d['w'], b=d['b'], feature_idx=d['fidx'], k=commit_k, C=args.C,
             sel_cv=d['sel'], verify_cv=d['vera'], verify_cv_b=d['verb'],
             val_acc=d['va'], test_acc=d['te'], params=d['params'],
             n_line=n_line, n_corn=d['ncorn'], **FEAT_CFG)
    print(f'SAVED -> {out}')


if __name__ == '__main__':
    main()
