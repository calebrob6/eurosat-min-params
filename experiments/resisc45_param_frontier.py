#!/usr/bin/env python
"""How few stored values does RESISC45 need for 65% and 70% test accuracy?

`resisc45_min_params.py` fixed the budget at 1,024 values and asked which head
structure spends it best.  This experiment fixes the *targets* instead and walks
the budget down, over the same 1,579-column zero-parameter pool, for three head
families:

* ``dense`` -- a reference-class affine head on the top-``k`` pool columns, the
  conservative reading in which the deployment artefact is one feature list;
* ``sparse`` -- the same head pruned so each class reads its own subset of a
  top-``k`` candidate list, which needs one index per stored weight;
* ``block`` -- an intermediate in which the 44 rows share only ``g`` feature
  lists, so the index pattern is ``g * k`` column ids instead of one per weight.

Rankings, sparsity patterns, and weights come from train only.  The candidate
list size, the group count, the group-lasso strength, and ``C`` are chosen on
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

from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    accuracy,
    fit_block_sparse_logreg_gpu,
    fit_logreg_gpu,
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    logreg_params,
    refit_masked_ref_logreg_gpu,
    sparse_logreg_params,
)
from resisc45_min_params import load_all  # noqa: E402

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_param_frontier_result.csv')
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DENSE_K = (8, 12, 16, 22, 32, 48, 64, 96, 128)
BUDGETS = (128, 192, 256, 320, 384, 512, 640, 768, 896, 1024)
SUBSET_SIZES = (32, 48, 64, 96, 128, 192, 256)
LAMBDAS = (3.0, 10.0)
GROUP_COUNTS = (1, 2, 4, 8, 11, 22, 44)
TARGETS = (0.65, 0.70)


def best_by_val(candidates):
    return max(candidates, key=lambda row: row['val_accuracy'])


def sweep_c(xtr, ytr, x, y, idx, mask, steps):
    """Refit a fixed support at several ``C`` and keep the best validation fit."""
    out = []
    for C in C_GRID:
        w, b = refit_masked_ref_logreg_gpu(xtr, ytr, mask, C=C, steps=steps)
        out.append({
            'C': C,
            'val_accuracy': accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val']),
            'test_accuracy': accuracy((x['test'][:, idx] @ w.T + b).argmax(1), y['test']),
            'nonzeros': int((w != 0).sum()),
            'features': int((np.abs(w).sum(0) > 0).sum()),
        })
    return best_by_val(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()

    x, y, _ = load_all()
    print(f'pool {x["train"].shape[1]} features', flush=True)
    orders = {lam: group_lasso_rank(x['train'], y['train'], lam=lam, epochs=1500)[0]
              for lam in LAMBDAS}
    rows: list[dict[str, object]] = []

    def record(**row) -> None:
        rows.append(row)
        print(' '.join(f'{k}={v}' for k, v in row.items()), flush=True)

    for k in DENSE_K:  # conservative reading: one feature list, no per-class pattern
        cands = []
        for lam, order in orders.items():
            idx = order[:k]
            for C in C_GRID:
                w, b = fit_logreg_gpu(x['train'][:, idx], y['train'], C=C, steps=args.steps)
                cands.append({
                    'val_accuracy': accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val']),
                    'test_accuracy': accuracy((x['test'][:, idx] @ w.T + b).argmax(1), y['test']),
                    'structure': f'lam{lam}/C={C}',
                })
        best = best_by_val(cands)
        record(head='dense', parameters=logreg_params(k), features=k, index_pattern=k,
               structure=best['structure'], val_accuracy=round(best['val_accuracy'], 4),
               test_accuracy=round(best['test_accuracy'], 4))

    for budget in BUDGETS:  # each class reads its own subset of a shared candidate list
        nonzeros = budget - (NUM_CLASSES - 1)
        cands = []
        for lam, order in orders.items():
            for k in SUBSET_SIZES:
                if (NUM_CLASSES - 1) * k < nonzeros:
                    continue
                idx = order[:k]
                w, _ = fit_sparse_logreg_gpu(x['train'][:, idx], y['train'], nonzeros=nonzeros,
                                             epochs=args.epochs, rounds=6)
                mask = (w[1:] != 0).astype(np.float32)
                best = sweep_c(x['train'][:, idx], y['train'], x, y, idx, mask, args.steps)
                best['structure'] = f'lam{lam}/top{k}/C={best["C"]}'
                cands.append(best)
        best = best_by_val(cands)
        record(head='sparse', parameters=sparse_logreg_params(best['nonzeros']),
               features=best['features'], index_pattern=best['nonzeros'],
               structure=best['structure'], val_accuracy=round(best['val_accuracy'], 4),
               test_accuracy=round(best['test_accuracy'], 4))

    per_group_k = (1024 - (NUM_CLASSES - 1)) // (NUM_CLASSES - 1)
    for groups in GROUP_COUNTS:  # index pattern shrinks to groups * per_group_k ids
        cands = []
        for lam, order in orders.items():
            idx = order[:256]
            w, _, assign, mask = fit_block_sparse_logreg_gpu(
                x['train'][:, idx], y['train'], n_groups=groups,
                per_group_k=per_group_k, steps=args.steps, rounds=6)
            best = sweep_c(x['train'][:, idx], y['train'], x, y, idx, mask, args.steps)
            best['structure'] = f'lam{lam}/g{groups}/C={best["C"]}'
            best['distinct'] = int(len(np.unique(assign)))
            cands.append(best)
        best = best_by_val(cands)
        record(head='block', parameters=(NUM_CLASSES - 1) * (per_group_k + 1),
               features=best['features'],
               index_pattern=best['distinct'] * per_group_k + (NUM_CLASSES - 1),
               structure=best['structure'], val_accuracy=round(best['val_accuracy'], 4),
               test_accuracy=round(best['test_accuracy'], 4))

    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for head in ('dense', 'sparse'):
        family = sorted((r for r in rows if r['head'] == head), key=lambda r: r['parameters'])
        for target in TARGETS:
            hit = next((r for r in family if r['val_accuracy'] >= target), None)
            if hit is None:
                print(f'{head}: no budget reaches {target:.0%} validation')
            else:
                print(f'{head}: {target:.0%} validation first met at {hit["parameters"]} '
                      f'parameters (test {hit["test_accuracy"]:.4f})')


if __name__ == '__main__':
    main()
