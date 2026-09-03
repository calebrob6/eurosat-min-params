#!/usr/bin/env python
"""Linear ceilings of RESISC45 candidate lists built over the widened pool.

A budgeted dictionary head only ever reads its candidate list, so the list's
own unconstrained linear ceiling bounds what the head can reach.  This compares
one group-lasso ranking over the whole 4,126-column pool (the three earlier
pools plus ``resisc45_gpu_features4.py``) against quota lists that take a fixed
number of columns from each pool's *separate* ranking, and prints the family
composition of the plain list to show where its slots go.
"""
from __future__ import annotations

import collections
import os
import re
import sys

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import accuracy, fit_logreg_gpu, group_lasso_rank

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_candidate_lists_result.txt')
C_GRID = (0.01, 0.03, 0.1)


def main() -> None:
    base, y, base_names = load_pools(BASE_POOLS)
    layout, _, layout_names = load_pools((LAYOUT_POOL,))
    new, _, new_names = load_pools(('gpu4_pool',))
    pool = {s: np.concatenate([base[s], layout[s], new[s]], axis=1) for s in base}
    names = base_names + layout_names + new_names
    bw, lw = base['train'].shape[1], layout['train'].shape[1]

    def family(j: int) -> str:
        if j < bw:
            return 'base'
        if j < bw + lw:
            return 'layout'
        return 'new:' + re.match(r'[a-z]+', names[j]).group(0)

    lines: list[str] = []

    def ceiling(idx: np.ndarray, tag: str) -> None:
        best = None
        for C in C_GRID:
            w, b = fit_logreg_gpu(pool['train'][:, idx], y['train'], C=C, steps=400)
            val = accuracy((pool['val'][:, idx] @ w.T + b).argmax(1), y['val'])
            test = accuracy((pool['test'][:, idx] @ w.T + b).argmax(1), y['test'])
            if best is None or val > best[0]:
                best = (val, test, C)
        lines.append(f'{tag}: {len(idx)} columns val {best[0]:.4f} test {best[1]:.4f} C={best[2]:g}')
        print(lines[-1], flush=True)

    plain = group_lasso_rank(pool['train'], y['train'], lam=LAM, epochs=1500)[0]
    counts = collections.Counter(family(j) for j in plain[:512])
    lines.append('plain-ranked top 512 by family: ' + ', '.join(
        f'{k} {v}' for k, v in counts.most_common()))
    print(lines[-1], flush=True)
    orders = [group_lasso_rank(block['train'], y['train'], lam=LAM, epochs=1500)[0]
              for block in (base, layout, new)]
    offsets = (0, bw, bw + lw)

    def quota(*sizes: int) -> np.ndarray:
        return np.concatenate([o[:q] + off for o, q, off in zip(orders, sizes, offsets)])

    ceiling(plain[:512], 'plain rank, top 512')
    ceiling(plain[:1024], 'plain rank, top 1024')
    ceiling(quota(384, 128, 0), 'quota 384 base + 128 layout (incumbent list)')
    ceiling(quota(384, 128, 128), 'quota 384 base + 128 layout + 128 new')
    ceiling(quota(448, 128, 64), 'quota 448 base + 128 layout + 64 new')
    ceiling(np.arange(pool['train'].shape[1]), 'whole 4,126-column pool')
    with open(RESULT_PATH, 'w') as handle:
        handle.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
