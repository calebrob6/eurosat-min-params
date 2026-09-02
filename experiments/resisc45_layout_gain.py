#!/usr/bin/env python
"""Does an object-layout pool move the RESISC45 parameter frontier?

`resisc45_failure_analysis.py` concluded that the residual RESISC45 errors are
object-layout distinctions and that the pool contained no descriptor which
counts or measures discrete elongated objects.  `resisc45_gpu_features3.py` adds
505 such columns.  This experiment re-runs the sparse reference-class frontier
on the 1,579-column pool and on the 2,084-column extended pool under identical
settings, so the two curves differ only by the new family.

For every budget the sparsity pattern comes from iterative magnitude pruning on
train, the support is then refit convexly, and the candidate-list size and ``C``
are chosen on validation.  Test is read once per reported row.  The per-class
table repeats the failure-analysis comparison for the classes the budget was
previously losing.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import Counter

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import class_names  # noqa: E402
from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    SPLITS,
    accuracy,
    fit_logreg_gpu,
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    load_pool,
    refit_masked_ref_logreg_gpu,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_layout_gain_result.csv')
CLASS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_layout_gain_classes.csv')
BASE_POOLS = ('gpu_pool', 'gpu2_pool', 'rgb_pool')
LAYOUT_POOL = 'gpu3_pool'
C_GRID = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
BUDGETS = (512, 640, 768, 896, 1024)
SUBSET_SIZES = (96, 128, 192, 256)
LAM = 3.0


def load_pools(names: tuple[str, ...]):
    """Concatenate the named cached pools into one feature matrix."""
    loaded = [load_pool(name) for name in names]
    labels = loaded[0][1]
    features = {
        split: np.concatenate([pool[0][split] for pool in loaded], axis=1).astype(np.float32)
        for split in SPLITS
    }
    columns = [name for pool in loaded for name in pool[2]]
    if features['train'].shape[1] != len(columns):
        raise ValueError('pool width does not match the concatenated names')
    return features, labels, columns


def family_of(name: str, layout: bool) -> str:
    """Family of a pool column, used to report which pool the budget spends on.

    Membership of the new pool is decided by column index rather than by name,
    because the original 147-column pool already has ``blob*`` columns.
    """
    stem = re.sub(r'[0-9].*$', '', name.split('_')[0]) or 'other'
    return f'layout:{stem}' if layout else stem


def sweep_c(xtr, ytr, x, y, idx, mask, steps):
    """Refit a fixed support at several ``C`` and keep the best validation fit."""
    best = None
    for C in C_GRID:
        w, b = refit_masked_ref_logreg_gpu(xtr, ytr, mask, C=C, steps=steps)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {
                'C': C, 'val_accuracy': val,
                'test_accuracy': accuracy((x['test'][:, idx] @ w.T + b).argmax(1), y['test']),
                'nonzeros': int((w != 0).sum()),
                'features': int((np.abs(w).sum(0) > 0).sum()),
                'columns': idx[np.abs(w).sum(0) > 0],
                'pred_test': (x['test'][:, idx] @ w.T + b).argmax(1),
            }
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()

    variants = {
        'base': BASE_POOLS,
        'layout': BASE_POOLS + (LAYOUT_POOL,),
    }
    rows: list[dict[str, object]] = []
    best_at_budget: dict[str, dict] = {}
    labels = None

    base_width = sum(len(load_pool(name)[2]) for name in BASE_POOLS)
    for tag, pools in variants.items():
        x, y, columns = load_pools(pools)
        labels = y
        width = x['train'].shape[1]
        print(f'--- {tag}: {width} columns', flush=True)

        ceiling = max(
            (fit_logreg_gpu(x['train'], y['train'], C=C, steps=400) + (C,) for C in (0.03, 0.1)),
            key=lambda fit: accuracy((x['val'] @ fit[0].T + fit[1]).argmax(1), y['val']),
        )
        rows.append({
            'pool': tag, 'head': 'dense-full', 'parameters': (NUM_CLASSES - 1) * (width + 1),
            'features': width, 'structure': f'C={ceiling[2]}',
            'val_accuracy': round(accuracy((x['val'] @ ceiling[0].T + ceiling[1]).argmax(1),
                                           y['val']), 4),
            'test_accuracy': round(accuracy((x['test'] @ ceiling[0].T + ceiling[1]).argmax(1),
                                            y['test']), 4),
            'layout_columns': 0,
        })
        print(rows[-1], flush=True)

        order = group_lasso_rank(x['train'], y['train'], lam=LAM, epochs=1500)[0]
        print(f'{tag}: layout columns in the top 256 = '
              f'{int((order[:256] >= base_width).sum())}', flush=True)

        for budget in BUDGETS:
            nonzeros = budget - (NUM_CLASSES - 1)
            best = None
            for k in SUBSET_SIZES:
                if (NUM_CLASSES - 1) * k < nonzeros:
                    continue
                idx = order[:k]
                w, _ = fit_sparse_logreg_gpu(x['train'][:, idx], y['train'], nonzeros=nonzeros,
                                             epochs=args.epochs, rounds=6)
                mask = (w[1:] != 0).astype(np.float32)
                cand = sweep_c(x['train'][:, idx], y['train'], x, y, idx, mask, args.steps)
                cand['structure'] = f'top{k}/C={cand["C"]}'
                if best is None or cand['val_accuracy'] > best['val_accuracy']:
                    best = cand
            layout_used = int((best['columns'] >= base_width).sum())
            rows.append({
                'pool': tag, 'head': 'sparse',
                'parameters': sparse_logreg_params(best['nonzeros']),
                'features': best['features'], 'structure': best['structure'],
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
                'layout_columns': layout_used,
            })
            print(rows[-1], flush=True)
            if budget == max(BUDGETS):
                best['used'] = [(columns[i], i >= base_width) for i in best['columns']]
                best_at_budget[tag] = best

    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    names = class_names()
    truth = labels['test']
    class_rows = []
    for c in range(NUM_CLASSES):
        sel = truth == c
        entry = {'class': names[c]}
        for tag in variants:
            entry[f'{tag}_accuracy'] = round(float((best_at_budget[tag]['pred_test'][sel]
                                                    == c).mean()), 4)
        entry['delta'] = round(entry['layout_accuracy'] - entry['base_accuracy'], 4)
        class_rows.append(entry)
    class_rows.sort(key=lambda r: -r['delta'])
    with open(CLASS_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]))
        writer.writeheader()
        writer.writerows(class_rows)

    print('\nlargest per-class gains at the top budget:')
    for entry in class_rows[:8]:
        print(f'  {entry["class"]:<22s} {entry["base_accuracy"]:.4f} -> '
              f'{entry["layout_accuracy"]:.4f}  ({entry["delta"]:+.4f})')
    print('largest per-class losses:')
    for entry in class_rows[-5:]:
        print(f'  {entry["class"]:<22s} {entry["base_accuracy"]:.4f} -> '
              f'{entry["layout_accuracy"]:.4f}  ({entry["delta"]:+.4f})')
    print('\nselected-column families at the top budget:')
    for tag in variants:
        counts = Counter(family_of(n, layout) for n, layout in best_at_budget[tag]['used'])
        print(f'  {tag}: ' + ', '.join(f'{k}={v}' for k, v in counts.most_common(10)))


if __name__ == '__main__':
    main()
