#!/usr/bin/env python
"""Are degree-2 products of pool columns worth their stored values?

Every RESISC45 head in this file is *linear* in the pool, so one stored value
buys one column.  A product of two pool columns is still deterministic
arithmetic on a single image -- no learned constants, the same status as every
other pool column -- and standardising it folds into the head exactly like any
other column, so a weight on ``a*b`` costs one stored value like any other
weight.  Two questions follow, and they have different answers.

1. **Does a degree-2 expansion move the parameter frontier?**  Add the products
   of the top 128 ranked columns to the whole 2,084-column pool and re-run the
   prune-and-regrow frontier.

2. **Does it move the *extraction* frontier?**  `resisc45_rigl.py` closed with
   the one cost prune-and-regrow does not pay in stored values: the selected
   head reads 383 of 2,084 pool columns, and the whole-pool arm reads 621, so a
   deployment has to compute most of the pool.  A degree-2 polynomial in ``K``
   base columns needs only those ``K`` columns extracted, however many products
   the head then reads.  Comparing ``linear/topK`` against ``poly2/topK`` at
   equal ``K`` holds extraction fixed and varies only the degree.

Products are formed from the top ``K`` columns of the same train-only L2,1
group-lasso ranking the candidate lists use, giving ``K*(K+1)/2`` extra columns
(the diagonal is the squares).  Every arm searches its support with the same
prune-and-regrow fitter over its whole column set -- no candidate list, no quota
-- refits the support convexly, and picks search settings and ``C`` on
validation.  Test is read once per reported cell.

Parameter accounting is unchanged at ``nonzeros + 44``.  Two costs are reported
separately: ``base_columns``, the pool columns a deployment must extract, and
``index_ids``, the conservative index-pattern reading in which a product weight
spends two base-column ids instead of one.
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
    accuracy,
    fit_logreg_gpu,
    fit_rigl_ref_logreg_gpu,
    group_lasso_rank,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_quadratic_result.csv')
BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
POLY_K = (32, 64, 128, 256)
# The expansion whose products are offered alongside the *whole* linear pool,
# which is the arm that asks question 1.
FRONTIER_K = 128
# (epochs, mask updates, peak fraction of the support swapped per update); the
# two settings `resisc45_rigl.py` selected on validation at most budgets.
RIGL_SETTINGS = ((4000, 100, 0.5), (4000, 200, 0.3))
TARGETS = (0.65, 0.70)


def poly2_columns(x: dict[str, np.ndarray], cols: np.ndarray):
    """Degree-2 polynomial in ``cols``: the columns themselves then all products.

    Returns ``(features, origin)`` where ``origin[j]`` is the pair of *base*
    pool ids column ``j`` is built from (a linear column repeats its own id), so
    the extraction breadth of any support is the number of distinct ids it uses.
    """
    left, right = np.triu_indices(len(cols))
    out = {}
    for split, mat in x.items():
        sub = mat[:, cols]
        out[split] = np.concatenate([sub, (sub[:, left] * sub[:, right])],
                                    axis=1).astype(np.float32)
    origin = np.concatenate([np.stack([cols, cols], axis=1),
                             np.stack([cols[left], cols[right]], axis=1)])
    return out, origin


def summarise(rows: list[dict]) -> None:
    """Print the per-arm frontier and where each target is first cleared."""
    rows = [r for r in rows if r['budget'] > 0]
    arms = sorted({r['arm'] for r in rows})
    print(f'\n{"budget":>7s}  ' + '  '.join(f'{a:>24s}' for a in arms))
    for budget in sorted({r['budget'] for r in rows}):
        cells = []
        for arm in arms:
            hits = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            best = max(hits, key=lambda r: r['val_accuracy']) if hits else None
            cells.append('-' if best is None else
                         f'val {best["val_accuracy"]:.4f} test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>24s}' for c in cells))

    print('\nvalidation-selected arm per budget:')
    selected = []
    for budget in sorted({r['budget'] for r in rows}):
        best = max((r for r in rows if r['budget'] == budget), key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {best["parameters"]:>5d} values  {best["arm"]:<16s} {best["structure"]:<22s} '
              f'val={best["val_accuracy"]:.4f} test={best["test_accuracy"]:.4f} '
              f'({best["features"]} columns from {best["base_columns"]} base columns, '
              f'{best["product_weights"]} product weights, {best["index_ids"]} ids)')
    for target in TARGETS:
        hit = next((r for r in selected if r['val_accuracy'] >= target), None)
        miss = next((r for r in selected if r['test_accuracy'] >= target), None)
        print(f'{target:.0%}: first met on validation at '
              f'{hit["parameters"] if hit else "no"} parameters, on test at '
              f'{miss["parameters"] if miss else "no"} parameters')

    print('\nextraction breadth: base pool columns a deployment must compute')
    print(f'{"budget":>7s}  ' + '  '.join(f'{a:>24s}' for a in arms))
    for budget in sorted({r['budget'] for r in rows}):
        cells = []
        for arm in arms:
            hits = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            best = max(hits, key=lambda r: r['val_accuracy']) if hits else None
            cells.append('-' if best is None else
                         f'{best["base_columns"]:>4d} cols test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>24s}' for c in cells))

    print('\nextraction Pareto front per budget (fewest base columns at each test level):')
    for budget in sorted({r['budget'] for r in rows}):
        best = {}
        for row in (r for r in rows if r['budget'] == budget):
            key = row['arm']
            if key not in best or row['val_accuracy'] > best[key]['val_accuracy']:
                best[key] = row
        front = []
        for row in sorted(best.values(), key=lambda r: r['base_columns']):
            if not front or row['test_accuracy'] > front[-1]['test_accuracy']:
                front.append(row)
        print(f'  {budget:>5d} values: ' + '  ->  '.join(
            f'{r["arm"]} {r["base_columns"]}c/{r["test_accuracy"]:.4f}' for r in front))


def load_rows() -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(RESULT_PATH, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'parameters', 'features', 'base_columns', 'product_columns',
                    'product_weights', 'index_pattern', 'index_ids', 'width'):
            row[key] = int(row[key])
        for key in ('val_accuracy', 'test_accuracy'):
            row[key] = float(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    if args.summarise:
        summarise(load_rows())
        return

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    linear = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    linear_width = linear['train'].shape[1]
    order = group_lasso_rank(linear['train'], y['train'], lam=LAM, epochs=1500)[0]
    all_columns = np.arange(linear_width, dtype=np.int64)

    def linear_origin(cols: np.ndarray) -> np.ndarray:
        return np.stack([cols, cols], axis=1)

    # `arms[name] = (features, origin, n_linear)`; columns from `n_linear` on are
    # products and cost two base-column ids to address.
    arms: dict[str, tuple[dict[str, np.ndarray], np.ndarray, int]] = {
        'linear/whole': (linear, linear_origin(all_columns), linear_width),
    }
    for k in POLY_K:
        cols = order[:k]
        arms[f'linear/top{k}'] = ({s: linear[s][:, cols] for s in linear},
                                  linear_origin(cols), k)
        poly, origin = poly2_columns(linear, cols)
        arms[f'poly2/top{k}'] = (poly, origin, k)
    cols = order[:FRONTIER_K]
    poly, origin = poly2_columns(linear, cols)
    arms[f'poly2/whole+top{FRONTIER_K}'] = (
        {s: np.concatenate([linear[s], poly[s][:, FRONTIER_K:]], axis=1) for s in linear},
        np.concatenate([linear_origin(all_columns), origin[FRONTIER_K:]]),
        linear_width)

    rows: list[dict[str, object]] = []
    for name, (x, origin, n_linear) in arms.items():
        width = x['train'].shape[1]
        fit = max(
            (fit_logreg_gpu(x['train'], y['train'], C=C, steps=400) + (C,) for C in (0.03, 0.1)),
            key=lambda f: accuracy((x['val'] @ f[0].T + f[1]).argmax(1), y['val']),
        )
        rows.append({
            'budget': 0, 'arm': name, 'width': width,
            'parameters': (NUM_CLASSES - 1) * (width + 1), 'features': width,
            'base_columns': len(np.unique(origin)),
            'product_columns': width - n_linear,
            'product_weights': (NUM_CLASSES - 1) * (width - n_linear),
            'index_pattern': (NUM_CLASSES - 1) * width,
            'index_ids': (NUM_CLASSES - 1) * (width + width - n_linear),
            'structure': f'dense/C={fit[2]}',
            'val_accuracy': round(accuracy((x['val'] @ fit[0].T + fit[1]).argmax(1), y['val']), 4),
            'test_accuracy': round(accuracy((x['test'] @ fit[0].T + fit[1]).argmax(1),
                                            y['test']), 4),
        })
        print(f'--- {name}: {width} columns from {rows[-1]["base_columns"]} base columns, '
              f'unconstrained ceiling val={rows[-1]["val_accuracy"]:.4f} '
              f'test={rows[-1]["test_accuracy"]:.4f} at {rows[-1]["parameters"]} values',
              flush=True)

    for budget in BUDGETS:
        nonzeros = budget - (NUM_CLASSES - 1)
        for name, (x, origin, n_linear) in arms.items():
            width = x['train'].shape[1]
            if (NUM_CLASSES - 1) * width <= nonzeros:
                continue
            idx = np.arange(width, dtype=np.int64)
            for epochs, updates, drop in RIGL_SETTINGS:
                w, _ = fit_rigl_ref_logreg_gpu(x['train'], y['train'], nonzeros=nonzeros,
                                               epochs=epochs, updates=updates,
                                               drop_fraction=drop)
                support = w[1:] != 0
                best = sweep_c(x['train'], y['train'], x, y, idx,
                               support.astype(np.float32), args.steps)
                product_weights = int(support[:, n_linear:].sum())
                rows.append({
                    'budget': budget, 'arm': name, 'width': width,
                    'parameters': sparse_logreg_params(best['nonzeros']),
                    'features': best['features'],
                    'base_columns': len(np.unique(origin[best['columns']])),
                    'product_columns': int((best['columns'] >= n_linear).sum()),
                    'product_weights': product_weights,
                    'index_pattern': best['nonzeros'],
                    'index_ids': best['nonzeros'] + product_weights,
                    'structure': f'e{epochs}/u{updates}/d{drop}/C={best["C"]}',
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
