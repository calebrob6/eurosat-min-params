#!/usr/bin/env python
"""What are the 44 intercepts costing, and can the code pay for them instead?

Every RESISC45 head in this run has been budgeted as ``nnz(P) + 44``: the
sparse code plus one free intercept per non-reference class.  That accounting
was harmless while the budget was 1,024 stored values, but iteration 7 pushed
the 65% target down to 208 values, where **21% of the budget is intercept**
and only 164 values are left to spell the weights.

The intercept is a weight on a constant column, so nothing forces it to be
free.  Appending a constant column to the standardised features and a matching
identity atom to ``Df`` makes the head a bias-free ``Dc @ P @ Df.T`` over
``k + 1`` columns, and the prune-and-regrow search then *decides* how many
stored values an intercept is worth -- against every other entry it could grow
instead.  Spending 44 of them reproduces the free intercept exactly whenever
``Dc`` carries the class singletons, so the parameterisation is nested and can
only lose by search.

Three intercept rules are compared at equal total stored values:

* ``free``   -- the incumbent, ``nnz + 44`` values;
* ``coded``  -- the intercept drawn from the same sparse code, ``nnz`` values;
* ``none``   -- no intercept at all, ``nnz`` values.  This is the control that
  separates "a *cheap* intercept is worth something" from "the 44 values were
  simply wasted"; without it, ``coded`` beating ``free`` says nothing about
  where the gain came from.

Two column dictionaries (the ``pairs181``/``pairs256`` enumerations iteration 7
selected) and a second class dictionary carrying the 44 class singletons -- the
one that makes the nesting exact -- are crossed in at the anchor budgets.  ``C``
is chosen on validation, test is read once per row, and every row carries a
paired bootstrap of its test difference against the free-intercept incumbent at
the same budget, because a 6,300-image split only resolves about +-0.6 points.

The accounting convention is unchanged: a deployed head is ``w_eff`` and
``b_eff`` acting on raw features, reconstructed from the stored code, the two
fixed dictionaries and the training standardiser.  A coded intercept is
reconstructed the same way the weights already are, so it is counted the same
way -- and it needs strictly fewer stored values than a free one, never more.
"""
from __future__ import annotations

import argparse
import csv
import math
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
)
from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    accuracy,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    pair_atoms,
    refit_masked_sep_dict_ref_logreg_gpu,
    sep_dict_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_coded_bias_result.csv')
BUDGETS = (96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320,
           384, 448, 512, 640, 768, 1024)
# Budgets that carry the whole cross; the rest carry only the frontier arms.
ANCHORS = (160, 256, 512)
CANDIDATE_LIST = (512, 128)  # (columns, slots reserved for the object-layout pool)
RIGL_SETTING = (4000, 100, 0.5)  # held fixed at iterations 4/6's validated setting
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ROWS = NUM_CLASSES - 1
SEED = 0
BOOTSTRAP = 2000
TARGETS = (0.65, 0.70)
# (intercept rule, class dictionary, column dictionary, every budget or anchors only)
ARMS = (
    ('free', 'gauss16384', 'pairs256', True),
    ('free', 'gauss16384', 'pairs181', True),
    ('coded', 'gauss16384', 'pairs256', True),
    ('coded', 'gauss16384', 'pairs181', True),
    ('none', 'gauss16384', 'pairs256', True),
    ('none', 'gauss16384', 'pairs181', False),
    ('coded', 'gauss16384+I', 'pairs256', False),
    ('free', 'gauss16384+I', 'pairs256', False),
    ('coded', 'gauss16384', 'identity', False),
    ('free', 'gauss16384', 'identity', False),
)
INCUMBENT = 'free/gauss16384/pairs256'


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    """``count`` unit-norm Gaussian directions from a fixed seed."""
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def build_class_dicts() -> dict[str, np.ndarray]:
    """The class dictionaries the arms refer to, keyed by name.

    ``gauss16384+I`` prepends the 44 class singletons, which is what makes a
    coded intercept able to reproduce a free one exactly at 44 stored values.
    """
    gauss = gaussian_atoms(16384)
    return {'gauss16384': gauss,
            'gauss16384+I': np.concatenate((np.eye(ROWS), gauss), axis=1)}


def build_feature_dicts(k: int, order: np.ndarray) -> dict[str, np.ndarray]:
    """The column dictionaries the arms refer to, keyed by name."""
    return {'identity': np.eye(k),
            'pairs181': pair_atoms(k, 181, order),
            'pairs256': pair_atoms(k, 256, order)}


def sweep_c(xtr, ytr, x, y, idx, class_dict, feat_dict, support, steps, bias):
    """Refit a fixed code support at several ``C``; keep the best validation fit."""
    best = None
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(xtr, ytr, class_dict, feat_dict,
                                                    support, C=C, steps=steps, bias=bias)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            pred = (x['test'][:, idx] @ w.T + b).argmax(1)
            best = {
                'C': C, 'val_accuracy': val,
                'test_accuracy': accuracy(pred, y['test']),
                'pred_test': pred,
                'features': int((np.abs(w).sum(0) > 0).sum()),
                'dense_nonzeros': int((np.abs(w) > 1e-12).sum()),
                'intercept_norm': float(np.linalg.norm(b)),
            }
    return best


def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Percentile CI for ``acc(a) - acc(b)`` resampling test images in pairs."""
    correct = (a == y).astype(np.float64) - (b == y).astype(np.float64)
    rng = np.random.default_rng(seed)
    draws = correct[rng.integers(0, len(correct), size=(BOOTSTRAP, len(correct)))].mean(1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def write_rows(rows: list[dict], path: str) -> None:
    """Rewrite the result CSV, so a long run leaves usable partial results."""
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict]) -> None:
    """Print the intercept comparison and where each target is first cleared."""
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    budgets = sorted({r['budget'] for r in rows})
    print(f'\n{"budget":>7s}  ' + '  '.join(f'{a:>32s}' for a in arms))
    for budget in budgets:
        cells = []
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit:
                cells.append('-')
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            cells.append(f'val {best["val_accuracy"]:.4f} test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>32s}' for c in cells))

    print(f'\nvs {INCUMBENT} at the same stored values (paired bootstrap of the test '
          'difference):')
    for budget in budgets:
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit or arm == INCUMBENT:
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            print(f'  {budget:>5d}  {arm:<32s} {best["test_delta"]:+.4f} '
                  f'[{best["delta_lo"]:+.4f}, {best["delta_hi"]:+.4f}]')

    print('\nstored values the coded arms spend on the intercept:')
    for budget in budgets:
        for r in rows:
            if r['budget'] == budget and r['bias'] == 'coded':
                print(f'  {budget:>5d}  {r["arm"]:<32s} {r["bias_values"]:>3d} of '
                      f'{r["index_pattern"]:>4d} entries on the constant column '
                      f'({r["bias_values"] / r["index_pattern"]:.1%}), '
                      f'||b||={r["intercept_norm"]:.2f}')

    print('\nvalidation-selected arm per budget:')
    selected = []
    for budget in budgets:
        best = max((r for r in rows if r['budget'] == budget),
                   key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {best["parameters"]:>5d} values  {best["arm"]:<32s} '
              f'val={best["val_accuracy"]:.4f} test={best["test_accuracy"]:.4f}  '
              f'{best["features"]} columns, {best["dense_nonzeros"]} deployed weights, '
              f'{best["index_bits"]} index bits')
    for target in TARGETS:
        hit = next((r for r in selected if r['val_accuracy'] >= target), None)
        miss = next((r for r in selected if r['test_accuracy'] >= target), None)
        print(f'{target:.0%}: first met on validation at '
              f'{hit["parameters"] if hit else "no"} parameters, on test at '
              f'{miss["parameters"] if miss else "no"} parameters')


def load_rows(path: str) -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'parameters', 'features', 'index_pattern', 'class_atoms',
                    'feature_atoms', 'dense_nonzeros', 'index_bits', 'bias_values'):
            row[key] = int(row[key])
        for key in ('val_accuracy', 'test_accuracy', 'test_delta', 'delta_lo',
                    'delta_hi', 'intercept_norm'):
            row[key] = float(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--budgets', type=int, nargs='*', default=None)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    if args.summarise:
        summarise(load_rows(args.out))
        return

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    size, quota = CANDIDATE_LIST
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    idx = np.concatenate((base_order[:size - quota],
                          layout_order[:quota] + base_width)).astype(np.int64)
    xtr = merged['train'][:, idx]
    inner_order = group_lasso_rank(xtr, y['train'], lam=LAM, epochs=1500)[0]

    class_dicts = build_class_dicts()
    feature_dicts = build_feature_dicts(len(idx), inner_order)
    epochs, updates, drop = RIGL_SETTING
    rows: list[dict[str, object]] = []
    for budget in (args.budgets or BUDGETS):
        preds: dict[str, np.ndarray] = {}
        for bias, class_name, feat_name, every_budget in ARMS:
            if not every_budget and budget not in ANCHORS:
                continue
            nonzeros = budget - (ROWS if bias == 'free' else 0)
            cdict, fdict = class_dicts[class_name], feature_dicts[feat_name]
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                xtr, y['train'], cdict, fdict, nonzeros=nonzeros,
                epochs=epochs, updates=updates, drop_fraction=drop, bias=bias)
            best = sweep_c(xtr, y['train'], merged, y, idx, cdict, fdict, support,
                           args.steps, bias)
            arm = f'{bias}/{class_name}/{feat_name}'
            preds[arm] = best['pred_test']
            reference = preds.get(INCUMBENT)
            if reference is not None and arm != INCUMBENT:
                lo, hi = paired_bootstrap(best['pred_test'], reference, y['test'])
                delta = best['test_accuracy'] - accuracy(reference, y['test'])
            else:
                lo = hi = delta = 0.0
            # The constant column owns the last atom of the augmented dictionary.
            fatoms = fdict.shape[1] + (1 if bias == 'coded' else 0)
            bias_values = (int((support % fatoms == fatoms - 1).sum())
                           if bias == 'coded' else (ROWS if bias == 'free' else 0))
            rows.append({
                'budget': budget, 'arm': arm, 'bias': bias, 'class_dict': class_name,
                'features_dict': feat_name,
                'parameters': sep_dict_logreg_params(len(support), bias),
                'features': best['features'],
                'index_pattern': len(support),
                'bias_values': bias_values,
                'intercept_norm': round(best['intercept_norm'], 4),
                'class_atoms': cdict.shape[1],
                'feature_atoms': fatoms,
                'dense_nonzeros': best['dense_nonzeros'],
                'index_bits': int(round(len(support) * (math.log2(cdict.shape[1])
                                                        + math.log2(fatoms)))),
                'structure': f'e{epochs}/u{updates}/d{drop}/C={best["C"]}',
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
                'test_delta': round(delta, 4),
                'delta_lo': round(lo, 4), 'delta_hi': round(hi, 4),
            })
            print(' '.join(f'{k}={v}' for k, v in rows[-1].items()), flush=True)
        write_rows(rows, args.out)
    summarise(rows)


if __name__ == '__main__':
    main()
