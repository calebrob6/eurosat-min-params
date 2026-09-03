#!/usr/bin/env python
"""What kind of nonlinearity the RESISC45 pool is hiding from a linear head.

``resisc45_budget4096.py`` shows a one-hidden-layer MLP on the merged pool
beating the pool's linear ceiling by more than two test points.  A budgeted
head cannot afford a dense hidden layer, so the question is which *zero-
parameter* expansion of the pool captures that gain, because an expansion is a
new set of columns and the existing sparse machinery can then buy the ones it
wants.  Arms, each a dense linear head fitted on train with ``C`` chosen on
validation and test read once:

* ``linear``    -- the pool as it is (control);
* ``hinge``     -- each standardised column plus ``relu(x - t)`` at fixed knots,
  an additive (per-column) nonlinearity and nothing else;
* ``pairrelu``  -- ``relu(+-x_i +- x_j)`` over random signed pairs of the
  candidate-list columns, the smallest *interaction* nonlinearity and the ReLU
  analogue of the width-2 column atoms that paid on the linear head;
* ``mlp``       -- one-hidden-layer ReLU heads across widths, to show how narrow
  a hidden layer still reads the nonlinear information (a sparse head has to
  pay for every hidden unit).

The 512-column candidate list is the same one the dictionary heads use, so an
arm restricted to it says whether the information the heads can already reach
is nonlinear, or whether it lives in the 1,572 columns they never read.
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
    accuracy,
    fit_logreg_gpu,
    fit_mlp_gpu,
    group_lasso_rank,
    predict_mlp,
    standardise,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_nonlinear_probe_result.csv')
DENSE_C = (0.003, 0.01, 0.03, 0.1, 0.3)
KNOTS = (-1.0, 0.0, 1.0)
MLP_WIDTHS = (16, 32, 64, 128, 512)
MLP_DECAY = (1e-4, 1e-3)
PAIR_COUNTS = (4096, 16384)
CANDIDATE_LIST = (512, 128)
SEED = 0


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dense_arm(x, y, arm, structure, rows, path, extra=None) -> dict:
    """Dense linear head; ``C`` on validation, test once."""
    best = None
    t0 = time.time()
    for C in DENSE_C:
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        val = accuracy((x['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'val_accuracy': val,
                    'test_accuracy': accuracy((x['test'] @ w.T + b).argmax(1), y['test'])}
    rows.append({'arm': arm, 'structure': f'{structure}/C={best["C"]}',
                 'columns': x['train'].shape[1],
                 'val_accuracy': round(best['val_accuracy'], 4),
                 'test_accuracy': round(best['test_accuracy'], 4),
                 'seconds': round(time.time() - t0, 1), **(extra or {})})
    print(rows[-1], flush=True)
    write_rows(rows, path)
    return best


def hinge(x: dict, knots) -> dict:
    """Standardised columns plus ``relu(x - t)`` at each knot."""
    mu, sigma = standardise(x['train'])
    out = {}
    for split, block in x.items():
        z = ((block - mu) / sigma).astype(np.float32)
        out[split] = np.concatenate([z] + [np.maximum(z - t, 0.0) for t in knots], axis=1)
    return out


def pair_relu(x: dict, count: int, seed: int = SEED) -> dict:
    """``relu(s_i x_i + s_j x_j)`` over ``count`` random signed column pairs."""
    mu, sigma = standardise(x['train'])
    k = x['train'].shape[1]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, k, size=count)
    right = (left + rng.integers(1, k, size=count)) % k
    sign_l = rng.choice((-1.0, 1.0), size=count).astype(np.float32)
    sign_r = rng.choice((-1.0, 1.0), size=count).astype(np.float32)
    out = {}
    for split, block in x.items():
        z = ((block - mu) / sigma).astype(np.float32)
        feats = np.maximum(z[:, left] * sign_l + z[:, right] * sign_r, 0.0)
        out[split] = np.concatenate([z, feats], axis=1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='*', default=None)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    arms = set(args.arms or ('linear', 'hinge', 'pairrelu', 'mlp'))

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    pool = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    size, quota = CANDIDATE_LIST
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    idx = np.concatenate((base_order[:size - quota],
                          layout_order[:quota] + base_width)).astype(np.int64)
    lists = {'pool': pool, 'list512': {s: pool[s][:, idx] for s in pool}}
    rows: list[dict] = []

    for list_name, x in lists.items():
        if 'linear' in arms:
            dense_arm(x, y, f'linear/{list_name}', 'dense', rows, args.out)
        if 'hinge' in arms:
            dense_arm(hinge(x, KNOTS), y, f'hinge/{list_name}',
                      'knots=' + ','.join(f'{t:g}' for t in KNOTS), rows, args.out)
            dense_arm(hinge(x, (0.0,)), y, f'hinge0/{list_name}', 'knots=0', rows, args.out)
        if 'pairrelu' in arms:
            for count in PAIR_COUNTS:
                dense_arm(pair_relu(x, count), y, f'pairrelu{count}/{list_name}',
                          f'pairs={count}', rows, args.out)
        if 'mlp' in arms:
            for hidden in MLP_WIDTHS:
                best = None
                t0 = time.time()
                for decay in MLP_DECAY:
                    w1, b1, w2, b2 = fit_mlp_gpu(x['train'], y['train'], hidden,
                                                 weight_decay=decay)
                    val = accuracy(predict_mlp(x['val'], w1, b1, w2, b2), y['val'])
                    if best is None or val > best['val_accuracy']:
                        best = {'decay': decay, 'val_accuracy': val,
                                'test_accuracy': accuracy(
                                    predict_mlp(x['test'], w1, b1, w2, b2), y['test'])}
                rows.append({'arm': f'mlp{hidden}/{list_name}',
                             'structure': f'h={hidden}/wd={best["decay"]}',
                             'columns': x['train'].shape[1],
                             'val_accuracy': round(best['val_accuracy'], 4),
                             'test_accuracy': round(best['test_accuracy'], 4),
                             'seconds': round(time.time() - t0, 1)})
                print(rows[-1], flush=True)
                write_rows(rows, args.out)


if __name__ == '__main__':
    main()
