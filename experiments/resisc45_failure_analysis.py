#!/usr/bin/env python
"""Where the RESISC45 budget is lost: per-class errors and pool-family usage.

Two heads are compared on the same 1,579-column zero-parameter pool: the
unconstrained 69,520-value head that measures what the pool can express, and the
1,024-value sparse reference-class head that is the actual budgeted model.  The
per-class gap between them separates *pool* failures (classes the features
cannot express at any budget) from *budget* failures (classes the pool handles
but the small head cannot afford).  The selected columns are also grouped by
pool family so the families that survive the budget are visible.

Train fits the heads, validation picks ``C``, and test is read once.
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
    accuracy,
    fit_logreg_gpu,
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    refit_masked_ref_logreg_gpu,
)
from resisc45_min_params import load_all  # noqa: E402

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_failure_analysis_result.csv')
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
BUDGET = 1024
SUBSET = 192


def family_of(name: str) -> str:
    """Collapse a pool column name onto its feature family.

    Column names are ``<family><scale>_<channel>``; dropping the digits and the
    channel suffix leaves the extractor that produced the column.
    """
    return re.sub(r'[\d.]+', '', name.split('_')[0])


def per_class(pred: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    return np.array([float((pred[y == c] == c).mean()) for c in range(n)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1200)
    args = parser.parse_args()

    x, y, names = load_all()
    classes = class_names()
    n_classes = len(classes)

    best = None
    for C in (0.03, 0.1, 0.3):
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        val = accuracy((x['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best[0]:
            best = (val, (x['test'] @ w.T + b).argmax(1), C)
    ceiling_pred = best[1]
    print(f'pool ceiling: C={best[2]} val={best[0]:.4f} test={accuracy(ceiling_pred, y["test"]):.4f}',
          flush=True)

    order = group_lasso_rank(x['train'], y['train'], lam=3.0, epochs=1500)[0]
    idx = order[:SUBSET]
    w0, _ = fit_sparse_logreg_gpu(x['train'][:, idx], y['train'], nonzeros=BUDGET - 44,
                                  epochs=args.epochs, rounds=6)
    mask = (w0[1:] != 0).astype(np.float32)
    budget_best = None
    for C in C_GRID:
        w, b = refit_masked_ref_logreg_gpu(x['train'][:, idx], y['train'], mask, C=C, steps=300)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if budget_best is None or val > budget_best[0]:
            budget_best = (val, (x['test'][:, idx] @ w.T + b).argmax(1), C, w)
    budget_pred = budget_best[1]
    print(f'budget head: C={budget_best[2]} val={budget_best[0]:.4f} '
          f'test={accuracy(budget_pred, y["test"]):.4f}', flush=True)

    ceiling_acc = per_class(ceiling_pred, y['test'], n_classes)
    budget_acc = per_class(budget_pred, y['test'], n_classes)
    rows = []
    for c in range(n_classes):
        wrong = budget_pred[(y['test'] == c) & (budget_pred != c)]
        top = Counter(wrong.tolist()).most_common(1)
        rows.append({
            'class': classes[c],
            'ceiling_accuracy': round(float(ceiling_acc[c]), 4),
            'budget_accuracy': round(float(budget_acc[c]), 4),
            'budget_gap': round(float(ceiling_acc[c] - budget_acc[c]), 4),
            'top_confusion': classes[top[0][0]] if top else '',
            'top_confusion_count': top[0][1] if top else 0,
        })
    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('\nweakest classes for the pool itself (ceiling accuracy):')
    for row in sorted(rows, key=lambda r: r['ceiling_accuracy'])[:10]:
        print(f'  {row["class"]:<22s} ceiling={row["ceiling_accuracy"]:.3f} '
              f'budget={row["budget_accuracy"]:.3f} -> {row["top_confusion"]}')
    print('\nlargest losses caused by the 1,024-value budget:')
    for row in sorted(rows, key=lambda r: -r['budget_gap'])[:10]:
        print(f'  {row["class"]:<22s} ceiling={row["ceiling_accuracy"]:.3f} '
              f'budget={row["budget_accuracy"]:.3f} gap={row["budget_gap"]:.3f} '
              f'-> {row["top_confusion"]}')

    used = idx[np.abs(budget_best[3]).sum(0) > 0]
    counts = Counter(family_of(names[i]) for i in used)
    print(f'\n{len(used)} pool columns used, by family:')
    for family, count in counts.most_common(20):
        print(f'  {family:<40s} {count}')


if __name__ == '__main__':
    main()
