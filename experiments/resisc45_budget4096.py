#!/usr/bin/env python
"""Where the RESISC45 frontier stands at 2,048 and 4,096 deployed values.

Every earlier RESISC45 section stops at 1,024 stored values because that was
the target.  The new target is **80% test under 4,096 values**, and the pool's
own linear ceiling is 79.4% test, so two questions decide what to build next:

* how close to the ceiling the existing heads already get at 2,048 and 4,096
  values -- if the gap is small, the pool is the constraint and a new head is
  pointless;
* whether the pool carries information a *linear* head cannot read -- a wide
  one-hidden-layer MLP on the whole pool is a cheap upper bound on that, and if
  it does not beat the linear ceiling either, the information is simply not in
  the pool and a new *pool* is the only lever.

Arms, all fitted on train with ``C`` and the head structure chosen on
validation and test read once per cell:

* ``dense``      -- the unconstrained linear ceiling on the merged pool;
* ``mlp``        -- one-hidden-layer ReLU heads of several widths, no budget;
* ``elementwise``-- prune-and-regrow reference-class head over the whole pool;
* ``cdict``      -- the class-dictionary head (16,384 Gaussian atoms, identity
  column atoms) over the 512-column candidate list and over the whole pool.
  Identity column atoms absorb the standardiser, so its deployed cost is exactly
  ``nnz + 44`` (see ``sep_dict_deployed_values``).

The pair-dictionary arm is run by ``resisc45_standardiser.py --budgets`` at the
same budgets and merged into the summary by ``--summarise``.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import (
    NUM_CLASSES,
    accuracy,
    fit_logreg_gpu,
    fit_mlp_gpu,
    fit_rigl_ref_logreg_gpu,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    mlp_params,
    predict_mlp,
    refit_masked_ref_logreg_gpu,
    refit_masked_sep_dict_ref_logreg_gpu,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_budget4096_result.csv')
BUDGETS = (1024, 2048, 4096)
ROWS = NUM_CLASSES - 1
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DENSE_C = (0.01, 0.03, 0.1, 0.3)
MLP_WIDTHS = (256, 1024)
MLP_DECAY = (1e-4, 1e-3)
RIGL_SETTING = (4000, 100, 0.5)
CANDIDATE_LIST = (512, 128)
SEED = 0


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def evaluate(x, y, w, b) -> dict:
    val = accuracy((x['val'] @ w.T + b).argmax(1), y['val'])
    test = accuracy((x['test'] @ w.T + b).argmax(1), y['test'])
    return {'val_accuracy': val, 'test_accuracy': test,
            'features': int((np.abs(w).sum(0) > 0).sum())}


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def row(arm, budget, deployed, structure, best, seconds) -> dict:
    return {'arm': arm, 'budget': budget, 'deployed_values': deployed,
            'structure': structure, 'features': best['features'],
            'val_accuracy': round(best['val_accuracy'], 4),
            'test_accuracy': round(best['test_accuracy'], 4),
            'seconds': round(seconds, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', type=int, nargs='*', default=list(BUDGETS))
    parser.add_argument('--arms', nargs='*', default=None)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    arms = set(args.arms or ('dense', 'mlp', 'elementwise', 'cdict'))

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    x = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    k = x['train'].shape[1]
    print(f'merged pool {k} columns', flush=True)
    rows: list[dict] = []

    if 'dense' in arms:
        best = None
        t0 = time.time()
        for C in DENSE_C:
            w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
            got = evaluate(x, y, w, b)
            print(f'dense C={C}: val={got["val_accuracy"]:.4f} test={got["test_accuracy"]:.4f}',
                  flush=True)
            if best is None or got['val_accuracy'] > best['val_accuracy']:
                best = dict(got, C=C)
        rows.append(row('dense', 0, ROWS * (k + 1), f'C={best["C"]}', best, time.time() - t0))
        write_rows(rows, args.out)

    if 'mlp' in arms:
        for hidden in MLP_WIDTHS:
            best = None
            t0 = time.time()
            for decay in MLP_DECAY:
                w1, b1, w2, b2 = fit_mlp_gpu(x['train'], y['train'], hidden,
                                             weight_decay=decay)
                got = {
                    'val_accuracy': accuracy(predict_mlp(x['val'], w1, b1, w2, b2), y['val']),
                    'test_accuracy': accuracy(predict_mlp(x['test'], w1, b1, w2, b2), y['test']),
                    'features': k,
                }
                print(f'mlp h={hidden} wd={decay}: val={got["val_accuracy"]:.4f} '
                      f'test={got["test_accuracy"]:.4f}', flush=True)
                if best is None or got['val_accuracy'] > best['val_accuracy']:
                    best = dict(got, decay=decay)
            rows.append(row('mlp', 0, mlp_params(k, hidden), f'h={hidden}/wd={best["decay"]}',
                            best, time.time() - t0))
            write_rows(rows, args.out)

    epochs, updates, drop = RIGL_SETTING
    if 'elementwise' in arms:
        for budget in args.budgets:
            t0 = time.time()
            nnz = budget - ROWS
            w, _ = fit_rigl_ref_logreg_gpu(x['train'], y['train'], nonzeros=nnz,
                                           epochs=epochs, updates=updates,
                                           drop_fraction=drop)
            mask = (w[1:] != 0).astype(np.float32)
            best = None
            for C in C_GRID:
                w, b = refit_masked_ref_logreg_gpu(x['train'], y['train'], mask, C=C,
                                                   steps=args.steps)
                got = evaluate(x, y, w, b)
                if best is None or got['val_accuracy'] > best['val_accuracy']:
                    best = dict(got, C=C)
            rows.append(row('elementwise/pool', budget, nnz + ROWS, f'C={best["C"]}', best,
                            time.time() - t0))
            print(rows[-1], flush=True)
            write_rows(rows, args.out)

    if 'cdict' in arms:
        base_width = base['train'].shape[1]
        size, quota = CANDIDATE_LIST
        base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
        layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
        idx512 = np.concatenate((base_order[:size - quota],
                                 layout_order[:quota] + base_width)).astype(np.int64)
        cdict = gaussian_atoms(16384)
        for list_name, idx in (('list512', idx512), ('pool', np.arange(k))):
            xs = {s: x[s][:, idx] for s in x}
            fdict = np.eye(len(idx))
            for budget in args.budgets:
                t0 = time.time()
                nnz = budget - ROWS
                _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                    xs['train'], y['train'], cdict, fdict, nonzeros=nnz, epochs=epochs,
                    updates=updates, drop_fraction=drop, bias='free')
                best = None
                for C in C_GRID:
                    w, b = refit_masked_sep_dict_ref_logreg_gpu(
                        xs['train'], y['train'], cdict, fdict, support, C=C,
                        steps=args.steps, bias='free')
                    got = evaluate(xs, y, w, b)
                    if best is None or got['val_accuracy'] > best['val_accuracy']:
                        best = dict(got, C=C)
                rows.append(row(f'cdict/{list_name}', budget, nnz + ROWS, f'C={best["C"]}',
                                best, time.time() - t0))
                print(rows[-1], flush=True)
                write_rows(rows, args.out)


if __name__ == '__main__':
    main()
