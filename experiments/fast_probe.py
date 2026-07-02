#!/usr/bin/env python
"""Fast signal-first probe: does a candidate family beat the current frontier floor?

Run this FIRST, before spending the ~1h two-partition honest descent
(experiments/backward_select.py). It answers "is there signal?" in minutes by
attacking the three costs of the full descent at once:

  1. WARM START from the frozen frontier subset UNION the candidate family
     (~29 feats), not the L1 top-42 (~54 feats): ~10 descent steps, not ~35.
     If the candidate can't improve on the validated frontier, it won't help --
     so starting there loses no real signal and cuts most of the work.
  2. FEW select seeds (default 3) drive the greedy drops. The greedy PATH is
     robust to seed count; 10 seeds only matters for the final unbiased estimate.
  3. CONFIRM only the low-k subsets, each on a DISJOINT 10-seed verify block plus
     held-out val -- and run two select-seed partitions to check the low-k subset
     is partition-stable (the fast analog of the overnight two-descent protocol).

Net: minutes, not ~an hour. Only a family that shows a sub-frontier honest floor
HERE is worth a full independent descent / a committed submission.

Configured below for the iteration-15 Harris corner family on o6+line+corn2,
warm-started from submission 10 (k=23, 240 params). Change BASE_MODEL / POOL_FAMS
/ CAND_RANGE to probe another family.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import _drop_score, _init, mean_cv  # noqa: E402

# ---- configuration (edit to probe a different family) ---------------------
BASE_MODEL = 'submissions/10_line_backward_linear/model.npz'   # frozen frontier
POOL_PARTS = ['feat_o6', 'linefam_line', 'gs2fam_corn2']        # cache stems, in order
CAND_LO, CAND_HI = 314, 320    # candidate family index range in the built pool
CAND_NAMES = ['corn2frac_pan', 'corn2mag_pan', 'corn2frac_ndvi',
              'corn2mag_ndvi', 'corn2frac_ndbi', 'corn2mag_ndbi']
C = 10.0
PRESCREEN_SEEDS = [0, 1, 2]
SEARCH_SEEDS_A = [0, 1, 2]      # partition A select seeds (drive greedy)
SEARCH_SEEDS_B = [20, 21, 22]   # partition B select seeds
VERIFY_A = list(range(10, 20))  # disjoint from A
VERIFY_B = list(range(30, 40))  # disjoint from B
MIN_K = 17                      # descend to here
CONFIRM_HI = 24                 # confirm k in [MIN_K .. CONFIRM_HI]
WORKERS = 12
THRESH = 0.940


def _pool(split):
    return np.concatenate(
        [np.load(os.path.join(CACHE_DIR, f'{split}_{p}.npy')) for p in POOL_PARTS],
        axis=1).astype(np.float32)


def descend(ftr, ytr, init, sel_seeds, min_k):
    """Warm-start backward-greedy; return {k: subset} for every k visited."""
    idx = np.asarray(init).copy()
    subsets = {}
    with ProcessPoolExecutor(WORKERS, initializer=_init, initargs=(ftr, ytr)) as ex:
        while True:
            subsets[len(idx)] = idx.copy()
            if len(idx) <= min_k:
                break
            args = [(idx, j, sel_seeds, 5, C) for j in range(len(idx))]
            res = list(ex.map(_drop_score, args))
            best_j, _ = max(res, key=lambda t: t[1])
            idx = np.delete(idx, best_j)
    return subsets


def main():
    t0 = time.time()
    ftr, fva, fte = _pool('train'), _pool('val'), _pool('test')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    base = np.asarray(np.load(BASE_MODEL)['feature_idx'], dtype=int)
    print(f'pool={ftr.shape[1]}d  base(frozen frontier)={len(base)} feats  '
          f'cand=[{CAND_LO}:{CAND_HI}]  ({time.time()-t0:.0f}s)', flush=True)

    # ---- (1) PRE-SCREEN: add each candidate to the frozen base, read val lift --
    base_val = accuracy_score(yva, predict(fva, *fit_folded_logreg(ftr, ytr, base, C=C)))
    base_cv = mean_cv(ftr, ytr, base, PRESCREEN_SEEDS, C=C)
    print(f'\n== PRE-SCREEN (add 1 to frozen {len(base)} : base val={base_val:.4f} '
          f'cv3={base_cv:.4f}) ==')
    for j, ci in enumerate(range(CAND_LO, CAND_HI)):
        if ci in set(base.tolist()):
            continue
        sub = np.append(base, ci)
        v = accuracy_score(yva, predict(fva, *fit_folded_logreg(ftr, ytr, sub, C=C)))
        cv = mean_cv(ftr, ytr, sub, PRESCREEN_SEEDS, C=C)
        print(f'  +{CAND_NAMES[j]:16s} val={v:.4f} ({v-base_val:+.4f})  '
              f'cv3={cv:.4f} ({cv-base_cv:+.4f})')

    # ---- (2) WARM-START mini-descents (two select-seed partitions) -------------
    init = np.concatenate([base, np.arange(CAND_LO, CAND_HI)])
    init = np.array(sorted(set(init.tolist())))
    print(f'\n== WARM-START DESCENTS from |init|={len(init)} '
          f'(frozen {len(base)} UNION {CAND_HI-CAND_LO} candidates) down to k={MIN_K} ==')
    ta = time.time()
    subA = descend(ftr, ytr, init, SEARCH_SEEDS_A, MIN_K)
    print(f'  partition A (select {SEARCH_SEEDS_A}) done ({time.time()-ta:.0f}s)', flush=True)
    tb = time.time()
    subB = descend(ftr, ytr, init, SEARCH_SEEDS_B, MIN_K)
    print(f'  partition B (select {SEARCH_SEEDS_B}) done ({time.time()-tb:.0f}s)', flush=True)

    # ---- (3) CONFIRM low-k subsets on DISJOINT verify blocks + held-out val ----
    print(f'\n== CONFIRM (verCV on disjoint 10-seed blocks + val; test is FYI) ==')
    hdr = (f'{"k":>3} {"par":>4} | {"nCorn":>5} {"verCV":>7} {"val":>7} {"test":>7}'
           f' {"same?":>6} | floor?')
    print(hdr)
    floor = None
    for k in range(CONFIRM_HI, MIN_K - 1, -1):
        rowok = True
        same = set(subA[k].tolist()) == set(subB[k].tolist())
        for tag, subs, ver in (('A', subA, VERIFY_A), ('B', subB, VERIFY_B)):
            s = subs[k]
            w, b, fi = fit_folded_logreg(ftr, ytr, s, C=C)
            vc = mean_cv(ftr, ytr, s, ver, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            te = accuracy_score(yte, predict(fte, w, b, fi))
            ncorn = int(np.sum(np.asarray(s) >= CAND_LO))
            ok = vc >= THRESH and va >= THRESH
            rowok = rowok and ok
            print(f'{k:>3} {tag:>4} | {ncorn:>5} {vc:>7.4f} {va:>7.4f} {te:>7.4f}'
                  f' {str(same):>6} | {"PASS" if ok else "fail"}')
        if rowok:
            floor = k   # keep descending; smallest passing k wins
    print(f'\nFAST HONEST FLOOR (both partitions verCV>=0.940 AND val>=0.940): '
          f'k={floor} -> params={10*(floor+1) if floor else None} '
          f'(frontier is 240 @ k=23)')
    print(f'total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
