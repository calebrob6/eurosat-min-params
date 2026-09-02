#!/usr/bin/env python
"""RESISC45 parameter frontier with a quota-limited object-layout pool.

`resisc45_layout_diagnose.py` showed that the 505 object-layout columns of
`resisc45_gpu_features3.py` are informative on their own (65.21% test as a whole
505-column head) but almost entirely redundant with the 1,579-column pool when
the head is unconstrained (79.37% merged against 79.78% base).  Letting them
compete freely in one group-lasso ranking therefore *loses* accuracy at a fixed
budget, because 113 of the merged top 256 are layout columns and they displace
base columns the head needs.  Reserving a fixed quota of candidate slots instead
of letting them compete recovers a gain.

This experiment walks the budget down with the quota as the only new degree of
freedom.  The base ranking and the layout ranking are computed separately on
train, the candidate list is the top ``256 - q`` base columns plus the top ``q``
layout columns, the sparsity pattern comes from iterative magnitude pruning on
train, and the support is refit convexly.  The quota and ``C`` are chosen on
validation; test is read once per reported row.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import (  # noqa: E402
    BASE_POOLS,
    LAM,
    LAYOUT_POOL,
    load_pools,
    sweep_c,
)
from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_layout_frontier_result.csv')
BUDGETS = (512, 640, 768, 896, 1024)
CANDIDATES = 256
QUOTAS = (0, 32, 64, 128)
CONTROL_QUOTA = 64
TARGETS = (0.65, 0.70)


def summarise(rows: list[dict]) -> None:
    """Print the validation-selected frontier and the control comparison."""
    print('\nvalidation-selected quota per budget:')
    selected = []
    for budget in BUDGETS:
        best = max((r for r in rows if r['budget'] == budget and r['quota'] != 'control'),
                   key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {budget:>5d} values  quota={best["quota"]!s:<4s} '
              f'val={best["val_accuracy"]:.4f} test={best["test_accuracy"]:.4f} '
              f'({best["layout_features"]} of {best["features"]} columns from the layout pool)')
    for target in TARGETS:
        hit = next((r for r in selected if r['val_accuracy'] >= target), None)
        miss = next((r for r in selected if r['test_accuracy'] >= target), None)
        print(f'{target:.0%}: first met on validation at '
              f'{hit["parameters"] if hit else "no"} parameters, on test at '
              f'{miss["parameters"] if miss else "no"} parameters')
    print('\ncontrol (the same 64 slots given to lower-ranked base columns):')
    for budget in BUDGETS:
        row = {str(r['quota']): r for r in rows if r['budget'] == budget}
        print(f'  {budget:>5d} values  quota0={row["0"]["test_accuracy"]:.4f} '
              f'control={row["control"]["test_accuracy"]:.4f} '
              f'layout64={row["64"]["test_accuracy"]:.4f}')


def load_rows() -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(RESULT_PATH, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row['budget'] = int(row['budget'])
        row['parameters'] = int(row['parameters'])
        row['features'] = int(row['features'])
        row['layout_features'] = int(row['layout_features'])
        row['val_accuracy'] = float(row['val_accuracy'])
        row['test_accuracy'] = float(row['test_accuracy'])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    if args.summarise:
        summarise(load_rows())
        return

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]

    # 'control' displaces the same 64 candidate slots with lower-ranked *base*
    # columns, so a gain that survives it cannot be candidate-list churn.
    variants = [(str(q), q) for q in QUOTAS] + [('control', CONTROL_QUOTA)]
    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        nonzeros = budget - (NUM_CLASSES - 1)
        for quota, size in variants:
            if quota == 'control':
                idx = np.concatenate((base_order[:CANDIDATES - size],
                                      base_order[CANDIDATES:CANDIDATES + size])).astype(np.int64)
            else:
                idx = np.concatenate((base_order[:CANDIDATES - size],
                                      layout_order[:size] + base_width)).astype(np.int64)
            w, _ = fit_sparse_logreg_gpu(merged['train'][:, idx], y['train'], nonzeros=nonzeros,
                                         epochs=args.epochs, rounds=6)
            mask = (w[1:] != 0).astype(np.float32)
            best = sweep_c(merged['train'][:, idx], y['train'], merged, y, idx, mask, args.steps)
            rows.append({
                'budget': budget, 'quota': quota,
                'parameters': sparse_logreg_params(best['nonzeros']),
                'features': best['features'],
                'layout_features': int((best['columns'] >= base_width).sum()),
                'structure': f'top{CANDIDATES}/C={best["C"]}',
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
            })
            print(' '.join(f'{k}={v}' for k, v in rows[-1].items()), flush=True)

    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows)

if __name__ == '__main__':
    main()
