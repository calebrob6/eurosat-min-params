#!/usr/bin/env python
"""What the sixth (multi-layer random convolution) pool is worth at the top.

Same protocol as ``resisc45_pool5_ceiling.py``: the union head recovers its
list's dense ceiling minus 1.2-1.4 points, so a pool is read first through the
ceiling of the incumbent quota list, here the 736-column
384 base / 128 layout / 128 pool-4 / 96 pool-5 list of iteration 4.  Dense
multinomial heads on train, ``C`` on validation, test read once:

* the 736-column list alone, with the whole sixth pool, with each scale's
  network and with each recorded layer;
* the list with the top ``--tops`` sixth-pool columns by group-lasso ranking;
* the sixth pool alone, and the merged 4,268-column pool with and without it.

Writes the test confusion counts of the 30 most confused pairs for every arm.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from collections import Counter

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import class_names
from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import accuracy, fit_logreg_gpu, group_lasso_rank
from resisc45_pool5_ceiling import dense, pair_counts, write_rows

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_pool6_ceiling_result.csv')
PAIRS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_pool6_ceiling_pairs.csv')
NEW_POOL = 'gpu6_pool'
QUOTA = (384, 128, 128, 96)
FAMILIES = (('scale2', r'^rdeep2'), ('scale4', r'^rdeep4'),
            ('layer1', r'^rdeep\dl1'), ('layer2', r'^rdeep\dl2'),
            ('mean', r'mean_'), ('max', r'max_'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-pool', default=NEW_POOL)
    parser.add_argument('--tops', type=int, nargs='*', default=[64, 128, 192, 256])
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--pairs', default=PAIRS_PATH)
    parser.add_argument('--skip-merged', action='store_true')
    args = parser.parse_args()

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    pool4, _, _ = load_pools(('gpu4_pool',))
    pool5, _, _ = load_pools(('gpu5_pool',))
    new, _, new_names = load_pools((args.new_pool,))
    new_names = np.array(new_names)
    blocks = (base, layout, pool4, pool5)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in blocks]
    new_order = group_lasso_rank(new['train'], y['train'], lam=LAM, epochs=1500)[0]
    print('top new columns:', list(new_names[new_order[:24]]), flush=True)
    quota = {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, orders, QUOTA)],
                               axis=1) for s in base}
    merged = {s: np.concatenate([b[s] for b in blocks], axis=1) for s in base}

    def cat(a, b, cols=None):
        return {s: np.concatenate([a[s], b[s] if cols is None else b[s][:, cols]], axis=1)
                for s in a}

    arms = [('list736', quota), ('list736+pool6', cat(quota, new))]
    for fam, pattern in FAMILIES:
        cols = np.array([i for i, n in enumerate(new_names) if re.search(pattern, n)])
        arms.append((f'list736+{fam}', cat(quota, new, cols)))
    for top in args.tops:
        arms.append((f'list736+top{top}', cat(quota, new, new_order[:top])))
    arms.append(('pool6', new))
    if not args.skip_merged:
        arms += [('merged4268', merged), ('merged4268+pool6', cat(merged, new))]

    rows, preds = [], {}
    for arm, x in arms:
        t0 = time.time()
        best = dense(x, y)
        preds[arm] = best.pop('pred')
        rows.append({'arm': arm, 'columns': x['train'].shape[1],
                     **{k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)

    names = class_names()
    counts = {arm: pair_counts(preds[arm], y['test']) for arm in preds}
    top_pairs = [p for p, _ in counts['list736'].most_common(30)]
    pair_rows = [{'pair': f'{names[p[0]]} / {names[p[1]]}',
                  **{arm: counts[arm][p] for arm in preds}} for p in top_pairs]
    write_rows(pair_rows, args.pairs)
    for row in pair_rows[:15]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
