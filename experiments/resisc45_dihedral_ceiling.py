#!/usr/bin/env python
"""What dihedral averaging of the pools is worth at the top.

Reads the 736-column 384/128/128/96 quota list (the iteration-4 frontier list)
as a dense head, then the same list built from the dihedral-averaged pools
(``resisc45_dihedral_pools.py``), first with the *same columns* (indices taken
from the original ranking) and then re-ranked on the averaged pools; then the
list with the averaged copy of every column appended; then each block swapped
to its averaged version on its own; finally the merged pool both ways.  Dense
multinomial heads on train, ``C`` on validation, test read once, with the
test confusion counts of the 30 most confused pairs written for every arm.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import class_names
from resisc45_layout_gain import LAM, load_pools
from resisc45_lib import group_lasso_rank
from resisc45_pool5_ceiling import dense, pair_counts, write_rows

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_dihedral_ceiling_result.csv')
PAIRS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_dihedral_ceiling_pairs.csv')
BLOCKS = (('base', ('gpu_pool', 'gpu2_pool', 'rgb_pool'), ('gpud8_pool', 'gpu2d8_pool', 'rgb_pool')),
          ('layout', ('gpu3_pool',), ('gpu3d8_pool',)),
          ('pool4', ('gpu4_pool',), ('gpu4d8_pool',)),
          ('pool5', ('gpu5_pool',), ('gpu5d8_pool',)))
QUOTA = (384, 128, 128, 96)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--pairs', default=PAIRS_PATH)
    parser.add_argument('--skip-merged', action='store_true')
    args = parser.parse_args()

    orig, avg, y = [], [], None
    for _, pools, pools_d8 in BLOCKS:
        b, y, _ = load_pools(pools)
        a, _, _ = load_pools(pools_d8)
        orig.append(b)
        avg.append(a)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in orig]
    orders_d8 = [group_lasso_rank(a['train'], y['train'], lam=LAM, epochs=1500)[0] for a in avg]
    for (name, _, _), o, od in zip(BLOCKS, orders, orders_d8):
        q = QUOTA[[b[0] for b in BLOCKS].index(name)]
        print(f'{name}: {len(set(o[:q]) & set(od[:q]))}/{q} top columns shared', flush=True)

    def build(blocks, ords):
        return {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, ords, QUOTA)],
                                  axis=1) for s in y}

    def cat(a, b):
        return {s: np.concatenate([a[s], b[s]], axis=1) for s in a}

    list_orig = build(orig, orders)
    list_same = build(avg, orders)
    arms = [('list736', list_orig),
            ('list736-d8-samecols', list_same),
            ('list736-d8-reranked', build(avg, orders_d8)),
            ('list736+d8-samecols', cat(list_orig, list_same))]
    for i, (name, _, _) in enumerate(BLOCKS):
        mixed = [avg[j] if j == i else orig[j] for j in range(len(BLOCKS))]
        arms.append((f'list736-d8-{name}-only', build(mixed, orders)))
    if not args.skip_merged:
        merged = {s: np.concatenate([b[s] for b in orig], axis=1) for s in y}
        merged_d8 = {s: np.concatenate([b[s] for b in avg], axis=1) for s in y}
        arms += [('merged4268', merged), ('merged4268-d8', merged_d8),
                 ('merged4268+d8', cat(merged, merged_d8))]

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
    for row in pair_rows[:12]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
