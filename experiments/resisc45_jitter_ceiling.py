#!/usr/bin/env python
"""What translation and scale averaging of the pools is worth at the top.

Reads the dihedral-averaged 384/128/128/128 quota list (the iteration-5
frontier list, ``d8``) as a dense head, then the same columns (indices from
the ``d8`` ranking) built from each jitter family of
``resisc45_jitter_pools.py`` -- ``c224`` (eight crops paired with the eight
dihedral views), ``s192`` and ``s320`` (the eight dihedral views at 0.75x and
1.25x pixel scale) -- alone and averaged with ``d8`` (16-, 24- and 32-view
averages), then the widest average re-ranked, the ``d8`` list with the widest
average appended (does the jitter-dependent part carry signal?), each block
swapped to the widest average on its own, and the merged pools both ways.
Dense multinomial heads on train, ``C`` on validation, test read once; the
test confusion counts of the 30 most confused pairs are written per arm.
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

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_jitter_ceiling_result.csv')
PAIRS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_jitter_ceiling_pairs.csv')
BLOCKS = (('base', ('gpu_pool', 'gpu2_pool', 'rgb_pool')),
          ('layout', ('gpu3_pool',)),
          ('pool4', ('gpu4_pool',)),
          ('pool5', ('gpu5_pool',)))
QUOTA = (384, 128, 128, 128)


def family_pool(pool: str, family: str) -> str:
    if pool == 'rgb_pool':
        return pool
    stem, _, suffix = pool.partition('_')
    return f'{stem}{family}_{suffix}'


def load_family(family: str):
    blocks, y = [], None
    for _, pools in BLOCKS:
        b, y, _ = load_pools(tuple(family_pool(p, family) for p in pools))
        blocks.append(b)
    return blocks, y


def mean_of(families: list[list[dict]]) -> list[dict]:
    return [{s: np.mean([f[i][s] for f in families], axis=0).astype(np.float32) for s in families[0][i]}
            for i in range(len(families[0]))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--families', nargs='*', default=['c224', 's192', 's320'])
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--pairs', default=PAIRS_PATH)
    parser.add_argument('--skip-merged', action='store_true')
    args = parser.parse_args()

    loaded = {'d8': load_family('d8')[0]}
    y = load_family('d8')[1]
    for family in args.families:
        loaded[family] = load_family(family)[0]
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0]
              for b in loaded['d8']]

    def build(blocks, ords=orders):
        return {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, ords, QUOTA)],
                                  axis=1) for s in y}

    def cat(a, b):
        return {s: np.concatenate([a[s], b[s]], axis=1) for s in a}

    averages = {}
    for family in args.families:
        averages[f'd8+{family}'] = mean_of([loaded['d8'], loaded[family]])
    scales = [f for f in args.families if f.startswith('s')]
    if len(scales) > 1:
        averages['d8+' + '+'.join(scales)] = mean_of([loaded['d8']] + [loaded[f] for f in scales])
    if len(args.families) > 1:
        averages['d8+' + '+'.join(args.families)] = mean_of([loaded['d8']]
                                                            + [loaded[f] for f in args.families])
    widest_name = list(averages)[-1]
    widest = averages[widest_name]

    arms = [('list768-d8', build(loaded['d8']))]
    arms += [(f'list768-{f}-samecols', build(loaded[f])) for f in args.families]
    arms += [(f'list768-{name}-samecols', build(blocks)) for name, blocks in averages.items()]
    orders_w = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in widest]
    for (name, _), o, ow in zip(BLOCKS, orders, orders_w):
        q = QUOTA[[b[0] for b in BLOCKS].index(name)]
        print(f'{name}: {len(set(o[:q]) & set(ow[:q]))}/{q} top columns shared', flush=True)
    arms.append((f'list768-{widest_name}-reranked', build(widest, orders_w)))
    arms.append((f'list768-d8+{widest_name}-appended', cat(build(loaded['d8']), build(widest))))
    for i, (name, _) in enumerate(BLOCKS):
        mixed = [widest[j] if j == i else loaded['d8'][j] for j in range(len(BLOCKS))]
        arms.append((f'list768-{widest_name}-{name}-only', build(mixed)))
    if not args.skip_merged:
        arms.append(('merged-d8', {s: np.concatenate([b[s] for b in loaded['d8']], axis=1) for s in y}))
        arms.append((f'merged-{widest_name}', {s: np.concatenate([b[s] for b in widest], axis=1)
                                               for s in y}))

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
    top_pairs = [p for p, _ in counts['list768-d8'].most_common(30)]
    pair_rows = [{'pair': f'{names[p[0]]} / {names[p[1]]}',
                  **{arm: counts[arm][p] for arm in preds}} for p in top_pairs]
    write_rows(pair_rows, args.pairs)
    for row in pair_rows[:12]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
