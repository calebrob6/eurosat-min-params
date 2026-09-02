#!/usr/bin/env python
"""Minimum-parameter RESISC45 frontier under a 1,024-value budget.

RESISC45's 45 classes make the head, not the features, the binding constraint: a
dense reference-class affine head costs ``44 * (k + 1)``, so only 22 pool
columns fit inside 1,024 stored values.  This experiment measures three ways to
spend that budget on a 1,579-column zero-parameter pool:

* a dense reduced-rank or one-hidden-layer head on the top-``k`` features;
* a sparse reference-class head over the top-``k`` features, which keeps a small
  feature set but lets each class read its own subset of it;
* a sparse head over the whole pool, which maximises accuracy but no longer
  corresponds to a small feature set.

Feature ranking, sparsity patterns, and head weights come from train only; the
operating point is chosen on validation; test is read once per reported row.
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
    SPLITS,
    accuracy,
    fit_logreg_gpu,
    fit_lowrank_gpu,
    fit_mlp_gpu,
    fit_sparse_logreg_gpu,
    fit_sparse_lowrank_gpu,
    group_lasso_rank,
    load_pool,
    lowrank_params,
    mlp_params,
    predict_mlp,
    sparse_logreg_params,
    sparse_lowrank_params,
)

BUDGET = 1024
POOLS = ('gpu_pool', 'gpu2_pool', 'rgb_pool')
RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_min_params_result.csv')
SUBSET_SIZES = (32, 48, 64, 96, 128, 192)
DENSE_RANKS = (8, 10, 12, 14, 16)
SPARSE_RANKS = (10, 12, 14, 16, 18)
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def load_all() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Concatenate the three cached RESISC45 pools into one feature matrix."""
    loaded = [load_pool(name) for name in POOLS]
    labels = loaded[0][1]
    features = {
        split: np.concatenate([pool[0][split] for pool in loaded], axis=1).astype(np.float32)
        for split in SPLITS
    }
    names = [name for pool in loaded for name in pool[2]]
    if features['train'].shape[1] != len(names):
        raise ValueError('pool width does not match the concatenated names')
    return features, labels, names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1200)
    args = parser.parse_args()

    x, y, names = load_all()
    print(f'pool {x["train"].shape[1]} features', flush=True)
    rows: list[dict[str, object]] = []

    def record(model: str, features: int, structure: str, params: int,
               val: float, test: float) -> None:
        rows.append({
            'model': model, 'features': features, 'structure': structure,
            'parameters': params, 'val_accuracy': round(val, 4),
            'test_accuracy': round(test, 4),
        })
        print(f'{model:<14s} k={features:<5d} {structure:<12s} p={params:<5d} '
              f'val={val:.4f} test={test:.4f}', flush=True)

    for C in (0.03, 0.1):
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        record('dense-full', x['train'].shape[1], f'C={C}',
               44 * (x['train'].shape[1] + 1),
               accuracy((x['val'] @ w.T + b).argmax(1), y['val']),
               accuracy((x['test'] @ w.T + b).argmax(1), y['test']))

    ranks: dict[str, np.ndarray] = {}
    ranks['global'] = group_lasso_rank(x['train'], y['train'], lam=3.0,
                                       epochs=1500)[0]
    for rank in (10, 12):
        ranks[f'rank{rank}'] = group_lasso_rank(x['train'], y['train'], lam=1.0,
                                                epochs=1500, rank=rank)[0]

    for tag, order in ranks.items():
        idx = order[:22]  # the widest dense affine head that fits the budget
        best = max(
            (fit_logreg_gpu(x['train'][:, idx], y['train'], C=C) + (C,) for C in C_GRID),
            key=lambda fit: accuracy((x['val'][:, idx] @ fit[0].T + fit[1]).argmax(1), y['val']),
        )
        record('dense-affine', 22, f'{tag}/C={best[2]}', 44 * 23,
               accuracy((x['val'][:, idx] @ best[0].T + best[1]).argmax(1), y['val']),
               accuracy((x['test'][:, idx] @ best[0].T + best[1]).argmax(1), y['test']))
        for rank in DENSE_RANKS:
            k = (BUDGET - 44 * (rank + 1)) // rank
            if k < 4:
                continue
            idx = order[:k]
            a, c, d = fit_lowrank_gpu(x['train'][:, idx], y['train'], rank=rank,
                                      weight_decay=1e-4, epochs=2500)
            record('dense-lowrank', k, f'{tag}/r{rank}', lowrank_params(k, rank),
                   accuracy(((x['val'][:, idx] @ a) @ c + d).argmax(1), y['val']),
                   accuracy(((x['test'][:, idx] @ a) @ c + d).argmax(1), y['test']))
            k2 = (BUDGET - rank - 44 * (rank + 1)) // rank
            idx2 = order[:k2]
            w1, b1, w2, b2 = fit_mlp_gpu(x['train'][:, idx2], y['train'],
                                         hidden=rank, weight_decay=1e-4, epochs=2500)
            record('dense-mlp', k2, f'{tag}/h{rank}', mlp_params(k2, rank),
                   accuracy(predict_mlp(x['val'][:, idx2], w1, b1, w2, b2), y['val']),
                   accuracy(predict_mlp(x['test'][:, idx2], w1, b1, w2, b2), y['test']))

    order = ranks['global']
    for k in SUBSET_SIZES:
        idx = order[:k]
        nonzeros = min(BUDGET - 44, 44 * k)
        w, b = fit_sparse_logreg_gpu(x['train'][:, idx], y['train'],
                                     nonzeros=nonzeros, epochs=args.epochs, rounds=6)
        used = int((np.abs(w).sum(0) > 0).sum())
        record('sparse-subset', used, f'top{k}',
               sparse_logreg_params(int((w != 0).sum())),
               accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val']),
               accuracy((x['test'][:, idx] @ w.T + b).argmax(1), y['test']))

    for rank in SPARSE_RANKS:
        nonzeros = BUDGET - 44 * (rank + 1)
        a, c, d = fit_sparse_lowrank_gpu(x['train'], y['train'], rank=rank,
                                         nonzeros=nonzeros, epochs=args.epochs, rounds=6)
        used = int((np.abs(a).sum(1) > 0).sum())
        record('sparse-lowrank', used, f'r{rank}',
               sparse_lowrank_params(nonzeros, rank),
               accuracy(((x['val'] @ a) @ c + d).argmax(1), y['val']),
               accuracy(((x['test'] @ a) @ c + d).argmax(1), y['test']))

    nonzeros = BUDGET - 44
    w, b = fit_sparse_logreg_gpu(x['train'], y['train'], nonzeros=nonzeros,
                                 epochs=args.epochs, rounds=6)
    used = int((np.abs(w).sum(0) > 0).sum())
    record('sparse-pool', used, 'full-rank',
           sparse_logreg_params(int((w != 0).sum())),
           accuracy((x['val'] @ w.T + b).argmax(1), y['val']),
           accuracy((x['test'] @ w.T + b).argmax(1), y['test']))
    np.save(os.path.join(EXPERIMENTS_DIR, 'resisc45_sparse_pool_columns.npy'),
            np.flatnonzero(np.abs(w).sum(0) > 0))

    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    budgeted = [r for r in rows if r['parameters'] <= BUDGET]
    best = max(budgeted, key=lambda r: r['val_accuracy'])
    print(f'best <= {BUDGET} parameters by validation: {best}')


if __name__ == '__main__':
    main()
