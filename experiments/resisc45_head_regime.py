#!/usr/bin/env python
"""Why the RESISC45 dictionary head stops improving above ~2,000 values.

On the 640-column quota list (linear ceiling 81.0% test, 28,204 dense values)
the separable-dictionary head reads 77.9% at 2,326 deployed values and no
better at 3,866, so somewhere between a quarter and an eighth of the dense
head's values the sparse code stops converting budget into accuracy.  This
isolates the cause by holding the list fixed and varying only the head:

* ``elementwise`` -- prune-and-regrow over the list alone (44 x 640 entries),
  the smallest search space;
* ``cdict{atoms}`` -- the class-dictionary head at several atom counts, so the
  search space grows from 44 x 640 to 16,384 x 640 with the same list;
* ``wd`` -- the 16,384-atom head with a ten-fold stronger weight decay during
  the search.

Every row reports train accuracy next to validation and test: a head whose
train accuracy climbs while test stalls is overfitting its *support* to the
training split, which a convex refit at a validation-chosen ``C`` cannot undo
because the support is already chosen.
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

from resisc45_expanded_head import QUOTA, gaussian_atoms
from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import (
    NUM_CLASSES,
    accuracy,
    fit_logreg_gpu,
    fit_rigl_ref_logreg_gpu,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    refit_masked_ref_logreg_gpu,
    refit_masked_sep_dict_ref_logreg_gpu,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_head_regime_result.csv')
ROWS = NUM_CLASSES - 1
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RIGL_SETTING = (4000, 100, 0.5)
BUDGETS = (1024, 2048, 3584)
ATOM_COUNTS = (256, 1024, 4096, 16384)


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(x, y, w, b) -> dict:
    return {f'{s}_accuracy': round(accuracy((x[s] @ w.T + b).argmax(1), y[s]), 4)
            for s in ('train', 'val', 'test')}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', type=int, nargs='*', default=list(BUDGETS))
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    new, _, _ = load_pools(('gpu4_pool',))
    blocks = (base, layout, new)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in blocks]
    x = {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, orders, QUOTA)], axis=1)
         for s in base}
    k = x['train'].shape[1]
    print(f'{k} candidate columns', flush=True)
    epochs, updates, drop = RIGL_SETTING
    rows: list[dict] = []

    best = None
    for C in (0.01, 0.03, 0.1):
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        got = score(x, y, w, b)
        if best is None or got['val_accuracy'] > best['val_accuracy']:
            best = dict(got, C=C)
    rows.append({'arm': 'dense', 'budget': 0, 'deployed_values': ROWS * (k + 1),
                 'structure': f'C={best["C"]}', **{key: best[key] for key in
                                                    ('train_accuracy', 'val_accuracy',
                                                     'test_accuracy')}, 'seconds': 0})
    print(rows[-1], flush=True)

    for budget in args.budgets:
        t0 = time.time()
        w, _ = fit_rigl_ref_logreg_gpu(x['train'], y['train'], nonzeros=budget, epochs=epochs,
                                       updates=updates, drop_fraction=drop)
        mask = (w[1:] != 0).astype(np.float32)
        best = None
        for C in C_GRID:
            w, b = refit_masked_ref_logreg_gpu(x['train'], y['train'], mask, C=C,
                                               steps=args.steps)
            got = score(x, y, w, b)
            if best is None or got['val_accuracy'] > best['val_accuracy']:
                best = dict(got, C=C)
        rows.append({'arm': 'elementwise/list', 'budget': budget,
                     'deployed_values': budget + ROWS, 'structure': f'C={best["C"]}',
                     **{key: best[key] for key in ('train_accuracy', 'val_accuracy',
                                                   'test_accuracy')},
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)

        fdict = np.eye(k)
        for atoms, decay in [(a, 1e-4) for a in ATOM_COUNTS] + [(16384, 1e-3)]:
            t0 = time.time()
            cdict = gaussian_atoms(atoms)
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                x['train'], y['train'], cdict, fdict, nonzeros=budget, epochs=epochs,
                updates=updates, drop_fraction=drop, bias='free', weight_decay=decay)
            best = None
            for C in C_GRID:
                w, b = refit_masked_sep_dict_ref_logreg_gpu(
                    x['train'], y['train'], cdict, fdict, support, C=C, steps=args.steps,
                    bias='free')
                got = score(x, y, w, b)
                if best is None or got['val_accuracy'] > best['val_accuracy']:
                    best = dict(got, C=C)
            rows.append({'arm': f'cdict{atoms}' + ('/wd1e-3' if decay > 1e-4 else ''),
                         'budget': budget, 'deployed_values': budget + ROWS,
                         'structure': f'C={best["C"]}',
                         **{key: best[key] for key in ('train_accuracy', 'val_accuracy',
                                                       'test_accuracy')},
                         'seconds': round(time.time() - t0, 1)})
            print(rows[-1], flush=True)
            write_rows(rows, args.out)


if __name__ == '__main__':
    main()
