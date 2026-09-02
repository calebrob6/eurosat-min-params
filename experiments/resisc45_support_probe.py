#!/usr/bin/env python
"""Two ways of spending more compute on the RESISC45 support search, both failing.

`resisc45_rigl.py` established that *how* the sparsity pattern is found is worth
more than what is in the pool, and `resisc45_quadratic.py` then found that a
wider pool only raises validation fit.  That combination suggests the support
search is now the place to spend, so this script spends in the two obvious
directions and measures what comes back.

**Bagging.**  If a wider candidate set overfits the support search, stability
selection is the textbook answer: fit the search on several subsamples of train,
keep the entries that recur most often, and refit that support convexly on the
full split.  The votes are counts, broken by the standardised weight magnitude.

**Search length.**  `resisc45_rigl.py` found 4,000 epochs with 100 mask updates
beat 2,000 with 100, so the number of regrow opportunities is a real
hyperparameter.  This sweeps it four-fold further, on all three candidate lists.

Both arms use the same convex refit and the same validation-selected ``C``; test
is read once per row.  The summary reports what a validation-gated protocol
would have picked, which is the number that matters: a setting only helps if
choosing it on validation improves test.  For the search-length probe the
reference is the *validation pick within the existing grid* -- the best of the
three candidate lists at 4,000 epochs and 100 updates, which reproduces the arm
and the test accuracy `resisc45_rigl.py` selected at every budget from 512
upwards -- so the comparison is protocol against protocol rather than cell
against cell.
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
    group_lasso_rank,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_support_probe_result.csv')
# (candidate columns, slots reserved for the object-layout pool), as in
# `resisc45_rigl.py`; 'whole-pool' is the unranked 2,084-column set.
CANDIDATE_LISTS = ((256, 64), (512, 128))
LENGTH_BUDGETS = (384, 512, 640, 768, 896, 1024)
# (epochs, mask updates, peak fraction of the support swapped per update); the
# first is the setting `resisc45_rigl.py` selected at most budgets.
LENGTH_SETTINGS = ((4000, 100, 0.5), (8000, 200, 0.5), (16000, 200, 0.5), (16000, 400, 0.5))
BAG_BUDGETS = (512, 1024)
# (number of subsample fits, fraction of train in each)
BAG_SETTINGS = ((8, 0.8), (16, 0.8), (8, 0.6))


def bagged_support(x_train, y_train, nonzeros: int, bags: int, fraction: float) -> np.ndarray:
    """Support of the ``nonzeros`` entries chosen most often across subsample fits.

    Ties on the count are broken by the summed standardised weight magnitude,
    which is what makes the ranking total; the tie-break is scaled small enough
    that it can never outrank a whole extra vote.
    """
    n = len(x_train)
    votes = np.zeros((NUM_CLASSES - 1, x_train.shape[1]))
    for bag in range(bags):
        rng = np.random.default_rng(1000 + bag)
        sub = rng.choice(n, int(fraction * n), replace=False)
        w, _ = fit_rigl_ref_logreg_gpu(x_train[sub], y_train[sub], nonzeros=nonzeros,
                                       epochs=4000, updates=100, drop_fraction=0.5)
        magnitude = np.abs(w[1:]) * x_train[sub].std(0)[None, :]
        votes += (w[1:] != 0) + 1e-6 * magnitude / (magnitude.max() + 1e-12)
    flat = votes.reshape(-1)
    mask = np.zeros(flat.shape, dtype=np.float32)
    mask[np.argpartition(-flat, nonzeros)[:nonzeros]] = 1.0
    return mask.reshape(votes.shape)


def summarise(rows: list[dict]) -> None:
    """Print each probe against the setting a validation gate would have kept."""
    for probe in ('bagging', 'search-length'):
        subset = [r for r in rows if r['probe'] == probe]
        if not subset:
            continue
        print(f'\n--- {probe}')
        for budget in sorted({r['budget'] for r in subset}):
            cells = [r for r in subset if r['budget'] == budget]
            # The reference is what the pre-existing protocol would have picked,
            # not a single fixed cell.
            reference = max((r for r in cells if r['baseline'] == 'yes'),
                            key=lambda r: r['val_accuracy'])
            chosen = max(cells, key=lambda r: r['val_accuracy'])
            for row in cells:
                flag = '  reference-grid' if row['baseline'] == 'yes' else ''
                flag += '  <- validation pick' if row is chosen else ''
                print(f'  {budget:>5d} values  {row["arm"]:<28s} '
                      f'val={row["val_accuracy"]:.4f} test={row["test_accuracy"]:.4f}{flag}')
            print(f'  {budget:>5d} values  reference test={reference["test_accuracy"]:.4f}, '
                  f'validation gate moves it by '
                  f'{chosen["test_accuracy"] - reference["test_accuracy"]:+.4f}')


def load_rows() -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(RESULT_PATH, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'parameters', 'features'):
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
    x = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    lists = {
        f'top{size}': np.concatenate((base_order[:size - quota],
                                      layout_order[:quota] + base_width)).astype(np.int64)
        for size, quota in CANDIDATE_LISTS
    }
    lists['whole-pool'] = np.arange(x['train'].shape[1], dtype=np.int64)

    rows: list[dict[str, object]] = []

    def record(probe, budget, arm, idx, mask, baseline):
        best = sweep_c(x['train'][:, idx], y['train'], x, y, idx, mask, args.steps)
        rows.append({
            'probe': probe, 'budget': budget, 'arm': arm, 'baseline': baseline,
            'parameters': sparse_logreg_params(best['nonzeros']),
            'features': best['features'], 'structure': f'C={best["C"]}',
            'val_accuracy': round(best['val_accuracy'], 4),
            'test_accuracy': round(best['test_accuracy'], 4),
        })
        print(' '.join(f'{k}={v}' for k, v in rows[-1].items()), flush=True)

    idx = lists['whole-pool']
    for budget in BAG_BUDGETS:
        nonzeros = budget - (NUM_CLASSES - 1)
        w, _ = fit_rigl_ref_logreg_gpu(x['train'], y['train'], nonzeros=nonzeros,
                                       epochs=4000, updates=100, drop_fraction=0.5)
        record('bagging', budget, 'single fit', idx, (w[1:] != 0).astype(np.float32), 'yes')
        for bags, fraction in BAG_SETTINGS:
            mask = bagged_support(x['train'], y['train'], nonzeros, bags, fraction)
            record('bagging', budget, f'bagged {bags} fits at {fraction:.0%}', idx, mask, 'no')

    for budget in LENGTH_BUDGETS:
        nonzeros = budget - (NUM_CLASSES - 1)
        for name, idx in lists.items():
            for epochs, updates, drop in LENGTH_SETTINGS:
                w, _ = fit_rigl_ref_logreg_gpu(x['train'][:, idx], y['train'],
                                               nonzeros=nonzeros, epochs=epochs,
                                               updates=updates, drop_fraction=drop)
                baseline = 'yes' if (epochs, updates, drop) == LENGTH_SETTINGS[0] else 'no'
                record('search-length', budget, f'{name}/e{epochs}/u{updates}/d{drop}',
                       idx, (w[1:] != 0).astype(np.float32), baseline)

    with open(RESULT_PATH, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarise(rows)


if __name__ == '__main__':
    main()
