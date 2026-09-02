#!/usr/bin/env python
"""Why the object-layout pool does not move the RESISC45 frontier.

`resisc45_layout_gain.py` found that adding the 505 object-layout columns of
`resisc45_gpu_features3.py` makes the sparse frontier *worse* at every budget,
even though 113 of the merged pool's top 256 group-lasso columns come from the
new family.  Two explanations are possible:

* **dilution** -- the new columns are informative but crowd the shared candidate
  list, so the head spends weights on them at the expense of better columns;
* **redundancy** -- the new columns carry information the 1,579-column pool
  already has, so no selection strategy can turn them into accuracy.

Three measurements separate the two.  The layout pool's own unconstrained
ceiling says how much the family knows on its own; the merged ceiling says how
much of that is *new*; and a forced-quota candidate list, which reserves a fixed
number of slots for layout columns instead of letting them compete, says whether
a better selector would have found a gain.

Fits and rankings use train only, ``C`` and the quota are chosen on validation,
and test is read once per reported row.
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
    fit_sparse_logreg_gpu,
    group_lasso_rank,
    sparse_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_layout_diagnose_result.csv')
BUDGET = 1024
CANDIDATES = 256
QUOTAS = (0, 16, 32, 64, 128)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs', type=int, default=1200)
    parser.add_argument('--steps', type=int, default=300)
    args = parser.parse_args()

    base, y, base_names = load_pools(BASE_POOLS)
    layout, _, layout_names = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    rows: list[dict[str, object]] = []

    def record(**row) -> None:
        rows.append(row)
        print(' '.join(f'{k}={v}' for k, v in row.items()), flush=True)

    for tag, x in (('base', base), ('layout-only', layout), ('merged', merged)):
        width = x['train'].shape[1]
        best = max(
            (fit_logreg_gpu(x['train'], y['train'], C=C, steps=400) + (C,) for C in (0.03, 0.1)),
            key=lambda fit: accuracy((x['val'] @ fit[0].T + fit[1]).argmax(1), y['val']),
        )
        record(experiment='ceiling', pool=tag, quota='-', width=width,
               parameters=(NUM_CLASSES - 1) * (width + 1), structure=f'C={best[2]}',
               val_accuracy=round(accuracy((x['val'] @ best[0].T + best[1]).argmax(1),
                                           y['val']), 4),
               test_accuracy=round(accuracy((x['test'] @ best[0].T + best[1]).argmax(1),
                                            y['test']), 4))

    # A forced quota reserves candidate slots for the new family instead of
    # letting it compete for them, which separates dilution from redundancy.
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    nonzeros = BUDGET - (NUM_CLASSES - 1)
    for quota in QUOTAS:
        idx = np.concatenate((base_order[:CANDIDATES - quota],
                              layout_order[:quota] + base_width)).astype(np.int64)
        w, _ = fit_sparse_logreg_gpu(merged['train'][:, idx], y['train'], nonzeros=nonzeros,
                                     epochs=args.epochs, rounds=6)
        mask = (w[1:] != 0).astype(np.float32)
        best = sweep_c(merged['train'][:, idx], y['train'], merged, y, idx, mask, args.steps)
        record(experiment='forced-quota', pool='merged', quota=quota, width=len(idx),
               parameters=sparse_logreg_params(best['nonzeros']),
               structure=f'top{CANDIDATES}/C={best["C"]}',
               val_accuracy=round(best['val_accuracy'], 4),
               test_accuracy=round(best['test_accuracy'], 4),
               layout_columns=int((best['columns'] >= base_width).sum()))

    with open(RESULT_PATH, 'w', newline='') as handle:
        fields = sorted({k for row in rows for k in row}, key=lambda k: (k != 'experiment', k))
        writer = csv.DictWriter(handle, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
