#!/usr/bin/env python
"""What the fifth (line-geometry and region-shape) pool is worth at the top.

The union-of-supports head recovers its list's linear ceiling minus about one
point, so a new pool can only move the budgeted frontier if it moves the
*ceiling* of the 640-column quota list first.  Every arm here is a dense
multinomial head fitted on train with ``C`` chosen on validation and test read
once:

* the 384/128/128 quota list alone (the incumbent's list), then with the whole
  fifth pool and with each of its three families appended;
* the list with the top ``--tops`` fifth-pool columns by a group-lasso ranking
  (what a fourth quota would admit);
* the merged 4,126-column pool with and without the fifth pool, and the fifth
  pool alone.

For the list with and without the fifth pool it also writes the test
confusion counts of the pairs the union head confuses most, so a gain can be
attributed to the pairs the pool was built for.
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

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_pool5_ceiling_result.csv')
PAIRS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_pool5_ceiling_pairs.csv')
NEW_POOL = 'gpu5_pool'
DENSE_C = (0.003, 0.01, 0.03, 0.1)
FAMILY = re.compile(r'^(line|curv|cc)')
QUOTA = (384, 128, 128)


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dense(x, y):
    best = None
    for C in DENSE_C:
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        pred = {s: (x[s] @ w.T + b).argmax(1) for s in x}
        val = accuracy(pred['val'], y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'train_accuracy': accuracy(pred['train'], y['train']),
                    'val_accuracy': val, 'test_accuracy': accuracy(pred['test'], y['test']),
                    'pred': pred['test']}
    return best


def pair_counts(pred: np.ndarray, y: np.ndarray) -> Counter:
    counts: Counter = Counter()
    for p, t in zip(pred, y):
        if p != t:
            counts[tuple(sorted((int(p), int(t))))] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--new-pool', default=NEW_POOL)
    parser.add_argument('--tops', type=int, nargs='*', default=[32, 64, 96])
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    pool4, _, _ = load_pools(('gpu4_pool',))
    new, _, new_names = load_pools((args.new_pool,))
    new_names = np.array(new_names)
    blocks = (base, layout, pool4)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in blocks]
    new_order = group_lasso_rank(new['train'], y['train'], lam=LAM, epochs=1500)[0]
    print('top new columns:', list(new_names[new_order[:24]]), flush=True)
    quota = {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, orders, QUOTA)],
                               axis=1) for s in base}
    merged = {s: np.concatenate([b[s] for b in blocks], axis=1) for s in base}

    def cat(a, b, cols=None):
        return {s: np.concatenate([a[s], b[s] if cols is None else b[s][:, cols]], axis=1)
                for s in a}

    arms = [('list640', quota), ('list640+pool5', cat(quota, new))]
    for fam in ('line', 'curv', 'cc'):
        cols = np.array([i for i, n in enumerate(new_names) if n.startswith(fam)])
        arms.append((f'list640+{fam}', cat(quota, new, cols)))
    for top in args.tops:
        arms.append((f'list640+top{top}', cat(quota, new, new_order[:top])))
    arms += [('pool5', new), ('merged4126', merged), ('merged4126+pool5', cat(merged, new))]

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
    top_pairs = [p for p, _ in counts['list640'].most_common(30)]
    pair_rows = []
    for p in top_pairs:
        pair_rows.append({'pair': f'{names[p[0]]} / {names[p[1]]}',
                          **{arm: counts[arm][p] for arm in preds}})
    write_rows(pair_rows, PAIRS_PATH)
    for row in pair_rows[:15]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
