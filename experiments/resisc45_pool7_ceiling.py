#!/usr/bin/env python
"""Screen the cross-part configuration pool on the j32crop dense ceiling."""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import class_names  # noqa: E402
from resisc45_layout_gain import LAM, load_pools  # noqa: E402
from resisc45_lib import group_lasso_rank  # noqa: E402
from resisc45_pool5_ceiling import dense, pair_counts, write_rows  # noqa: E402

RESULT_PATH = os.path.join(
    EXPERIMENTS_DIR, 'resisc45_pool7_ceiling_result.csv'
)
PAIRS_PATH = os.path.join(
    EXPERIMENTS_DIR, 'resisc45_pool7_ceiling_pairs.csv'
)
VALUE_SUFFIX = 'j32crop'
RANK_SUFFIX = 'd8'
BLOCK_POOLS = (
    ('gpu_pool', 'gpu2_pool', 'rgb_pool'),
    ('gpu3_pool',),
    ('gpu4_pool',),
    ('gpu5_pool',),
)
QUOTA = (384, 128, 128, 128)
NEW_POOL = 'gpu7j32crop_pool'


def suffixed(pool: str, suffix: str) -> str:
    """Insert a cached view suffix into a pool stem."""
    if pool == 'rgb_pool':
        return pool
    return pool.replace('_pool', f'{suffix}_pool')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tops', type=int, nargs='*', default=[32, 64, 96, 128, 192])
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--pairs', default=PAIRS_PATH)
    args = parser.parse_args()

    value_blocks = []
    rank_blocks = []
    labels = None
    for pools in BLOCK_POOLS:
        values, labels, _ = load_pools(
            tuple(suffixed(pool, VALUE_SUFFIX) for pool in pools)
        )
        ranking, _, _ = load_pools(
            tuple(suffixed(pool, RANK_SUFFIX) for pool in pools)
        )
        value_blocks.append(values)
        rank_blocks.append(ranking)

    orders = [
        group_lasso_rank(
            block['train'], labels['train'], lam=LAM, epochs=1500
        )[0]
        for block in rank_blocks
    ]
    base = {
        split: np.concatenate(
            [
                block[split][:, order[:quota]]
                for block, order, quota in zip(
                    value_blocks, orders, QUOTA, strict=True
                )
            ],
            axis=1,
        )
        for split in labels
    }

    new, _, new_names = load_pools((NEW_POOL,))
    new_names = np.asarray(new_names)
    new_order = group_lasso_rank(
        new['train'], labels['train'], lam=LAM, epochs=1500
    )[0]
    print('top part columns:', list(new_names[new_order[:24]]), flush=True)

    def append(columns: np.ndarray | None = None) -> dict[str, np.ndarray]:
        return {
            split: np.concatenate(
                (
                    base[split],
                    new[split] if columns is None else new[split][:, columns],
                ),
                axis=1,
            )
            for split in base
        }

    arms = [('list768-j32-d8rank', base)]
    arms.extend(
        (f'list768+part{top}', append(new_order[:top]))
        for top in args.tops
    )
    arms.extend((('list768+part-all', append()), ('part-only', new)))

    rows = []
    predictions = {}
    for arm, features in arms:
        start = time.time()
        best = dense(features, labels)
        predictions[arm] = best.pop('pred')
        rows.append({
            'arm': arm,
            'columns': features['train'].shape[1],
            **{
                key: round(value, 4) if isinstance(value, float) else value
                for key, value in best.items()
            },
            'seconds': round(time.time() - start, 1),
        })
        print(rows[-1], flush=True)
        write_rows(rows, args.out)

    names = class_names()
    counts = {
        arm: pair_counts(prediction, labels['test'])
        for arm, prediction in predictions.items()
    }
    top_pairs = [
        pair for pair, _ in counts['list768-j32-d8rank'].most_common(30)
    ]
    pair_rows = [
        {
            'pair': f'{names[pair[0]]} / {names[pair[1]]}',
            **{arm: counts[arm][pair] for arm in predictions},
        }
        for pair in top_pairs
    ]
    write_rows(pair_rows, args.pairs)


if __name__ == '__main__':
    main()
