#!/usr/bin/env python
"""Does a prune-and-regrow support search beat iterative magnitude pruning?

Every RESISC45 sparse head so far picks its support by iterative magnitude
pruning (`fit_sparse_logreg_gpu`), which can only ever *remove* weights.  The
support therefore has to start inside a candidate list chosen up front by an
L2,1 group-lasso ranking, and `resisc45_layout_gain.py` showed the weakness of
that ranking: it scores each column against the label and never against the
columns already selected, so a redundant family floods the list and costs
accuracy under a budget.  The quota of `resisc45_layout_frontier.py` patches
this by hand.

`fit_rigl_ref_logreg_gpu` removes the need for the patch.  It holds the
active-weight count at the budget throughout and periodically drops the smallest
active weights and regrows the same number of inactive entries with the largest
*dense* loss gradient, which is evaluated at the current fit -- so a column is
grown only if it explains error the active columns leave behind.  Redundancy is
scored where it matters, and the candidate list can be the entire pool.

Five arms are compared at each budget under one protocol: the two search methods
crossed with a 256- and a 512-column quota list (the 256-column magnitude-pruned
arm is the previous best), plus prune-and-regrow on the whole 2,084-column
merged pool with no candidate list at all.  The crossed design separates the
search method from the candidate width, because a search that can regrow should
be able to exploit a wider list that magnitude pruning cannot.  Every arm refits
its support convexly afterwards.  Search settings and ``C`` are chosen on
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
    fit_rigl_ref_logreg_gpu,
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_rigl_result.csv')
BUDGETS = (256, 384, 512, 640, 768, 896, 1024)
# (candidate columns, slots reserved for the object-layout pool); the quota is a
# quarter of the list in both cases, the fraction `resisc45_layout_frontier.py`
# selected on validation.
CANDIDATE_LISTS = ((256, 64), (512, 128))
# (epochs, mask updates, peak fraction of the support swapped per update)
RIGL_SETTINGS = ((2000, 100, 0.5), (4000, 100, 0.5), (4000, 200, 0.3))
TARGETS = (0.65, 0.70)


def summarise(rows: list[dict]) -> None:
    """Print the per-arm frontier and where each target is first cleared."""
    arms = sorted({r['arm'] for r in rows})
    print(f'\n{"budget":>7s}  ' + '  '.join(f'{a:>22s}' for a in arms))
    for budget in sorted({r['budget'] for r in rows}):
        cells = []
        for arm in arms:
            best = max((r for r in rows if r['budget'] == budget and r['arm'] == arm),
                       key=lambda r: r['val_accuracy'])
            cells.append(f'val {best["val_accuracy"]:.4f} test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>22s}' for c in cells))

    print('\nvalidation-selected arm per budget:')
    selected = []
    for budget in sorted({r['budget'] for r in rows}):
        best = max((r for r in rows if r['budget'] == budget), key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {best["parameters"]:>5d} values  {best["arm"]:<22s} {best["structure"]:<28s} '
              f'val={best["val_accuracy"]:.4f} test={best["test_accuracy"]:.4f} '
              f'({best["features"]} columns, {best["layout_features"]} of them layout)')
    for target in TARGETS:
        hit = next((r for r in selected if r['val_accuracy'] >= target), None)
        miss = next((r for r in selected if r['test_accuracy'] >= target), None)
        print(f'{target:.0%}: first met on validation at '
              f'{hit["parameters"] if hit else "no"} parameters, on test at '
              f'{miss["parameters"] if miss else "no"} parameters')


def load_rows() -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(RESULT_PATH, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'parameters', 'features', 'layout_features', 'index_pattern'):
            row[key] = int(row[key])
        for key in ('val_accuracy', 'test_accuracy'):
            row[key] = float(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--imp-epochs', type=int, default=1200)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()
    if args.summarise:
        summarise(load_rows())
        return

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    width = merged['train'].shape[1]
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    candidates = {
        f'top{size}': np.concatenate((base_order[:size - quota],
                                      layout_order[:quota] + base_width)).astype(np.int64)
        for size, quota in CANDIDATE_LISTS
    }
    candidates['whole-pool'] = np.arange(width, dtype=np.int64)

    rows: list[dict[str, object]] = []
    for budget in BUDGETS:
        nonzeros = budget - (NUM_CLASSES - 1)
        runs: list[tuple[str, np.ndarray, str, np.ndarray]] = []

        for size, _ in CANDIDATE_LISTS:
            idx = candidates[f'top{size}']
            if (NUM_CLASSES - 1) * len(idx) < nonzeros:
                continue
            w, _ = fit_sparse_logreg_gpu(merged['train'][:, idx], y['train'],
                                         nonzeros=nonzeros, epochs=args.imp_epochs, rounds=6)
            runs.append((f'magnitude-prune/top{size}', idx,
                         f'top{size}/imp{args.imp_epochs}', (w[1:] != 0)))

        for name, idx in candidates.items():
            for epochs, updates, drop in RIGL_SETTINGS:
                w, _ = fit_rigl_ref_logreg_gpu(merged['train'][:, idx], y['train'],
                                               nonzeros=nonzeros, epochs=epochs,
                                               updates=updates, drop_fraction=drop)
                runs.append((f'rigl/{name}', idx, f'{name}/e{epochs}/u{updates}/d{drop}',
                             (w[1:] != 0)))

        for arm, idx, tag, support in runs:
            best = sweep_c(merged['train'][:, idx], y['train'], merged, y, idx,
                           support.astype(np.float32), args.steps)
            rows.append({
                'budget': budget, 'arm': arm,
                'parameters': sparse_logreg_params(best['nonzeros']),
                'features': best['features'],
                'index_pattern': best['nonzeros'],
                'layout_features': int((best['columns'] >= base_width).sum()),
                'structure': f'{tag}/C={best["C"]}',
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
