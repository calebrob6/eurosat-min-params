#!/usr/bin/env python
"""How much the fourth pool and the hinge expansion move the RESISC45 ceiling.

Before any budgeted head is fitted, this measures what the new columns are
*worth* at the top: the unconstrained linear ceiling of the merged 2,084-column
pool with each new family appended, with the whole fourth pool appended, and
of the fourth pool alone; then the same with the hinge expansion that
``resisc45_nonlinear_probe.py`` showed reads two points of nonlinear headroom;
and a wide ReLU head on the union as the information ceiling.  Every arm is a
head fitted on train with ``C`` (or weight decay) chosen on validation and test
read once.  A family that does not move the ceiling cannot help a budgeted
head, whatever its per-column strength.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import BASE_POOLS, LAYOUT_POOL, load_pools
from resisc45_lib import (
    accuracy,
    fit_logreg_gpu,
    fit_mlp_gpu,
    predict_mlp,
    standardise,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_pool4_ceiling_result.csv')
NEW_POOL = 'gpu4_pool'
DENSE_C = (0.003, 0.01, 0.03, 0.1)
KNOTS = (-1.0, 0.0, 1.0)
MLP_DECAY = (1e-4, 1e-3)
FAMILY = re.compile(r'^(rconv|rtex|ctex|cell)')


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def hinge(x: dict, knots) -> dict:
    mu, sigma = standardise(x['train'])
    out = {}
    for split, block in x.items():
        z = ((block - mu) / sigma).astype(np.float32)
        out[split] = np.concatenate([z] + [np.maximum(z - t, 0.0) for t in knots], axis=1)
    return out


def dense_arm(x, y, arm, rows, path) -> None:
    best = None
    t0 = time.time()
    for C in DENSE_C:
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        val = accuracy((x['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'val_accuracy': val,
                    'test_accuracy': accuracy((x['test'] @ w.T + b).argmax(1), y['test'])}
    rows.append({'arm': arm, 'head': f'dense/C={best["C"]}', 'columns': x['train'].shape[1],
                 'val_accuracy': round(best['val_accuracy'], 4),
                 'test_accuracy': round(best['test_accuracy'], 4),
                 'seconds': round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
    write_rows(rows, path)


def mlp_arm(x, y, arm, hidden, rows, path) -> None:
    best = None
    t0 = time.time()
    for decay in MLP_DECAY:
        w1, b1, w2, b2 = fit_mlp_gpu(x['train'], y['train'], hidden, weight_decay=decay)
        val = accuracy(predict_mlp(x['val'], w1, b1, w2, b2), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'decay': decay, 'val_accuracy': val,
                    'test_accuracy': accuracy(predict_mlp(x['test'], w1, b1, w2, b2), y['test'])}
    rows.append({'arm': arm, 'head': f'mlp/h={hidden}/wd={best["decay"]}',
                 'columns': x['train'].shape[1],
                 'val_accuracy': round(best['val_accuracy'], 4),
                 'test_accuracy': round(best['test_accuracy'], 4),
                 'seconds': round(time.time() - t0, 1)})
    print(rows[-1], flush=True)
    write_rows(rows, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='*', default=None)
    parser.add_argument('--new-pool', default=NEW_POOL)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    arms = set(args.arms or ('families', 'union', 'hinge', 'mlp'))

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    new, _, new_names = load_pools((args.new_pool,))
    families = sorted({FAMILY.match(n).group(1) for n in new_names if FAMILY.match(n)})
    print(f'merged {merged["train"].shape[1]} + new {new["train"].shape[1]} columns; '
          f'families {families}', flush=True)
    rows: list[dict] = []

    def with_columns(cols: np.ndarray) -> dict:
        return {s: np.concatenate([merged[s], new[s][:, cols]], axis=1) for s in merged}

    union = with_columns(np.arange(new['train'].shape[1]))
    if 'families' in arms:
        dense_arm(merged, y, 'merged', rows, args.out)
        for fam in families:
            cols = np.array([i for i, n in enumerate(new_names) if n.startswith(fam)])
            dense_arm(with_columns(cols), y, f'merged+{fam}', rows, args.out)
            dense_arm({s: new[s][:, cols] for s in new}, y, f'{fam} alone', rows, args.out)
    if 'union' in arms:
        dense_arm(union, y, 'merged+new', rows, args.out)
        dense_arm(new, y, 'new alone', rows, args.out)
    if 'hinge' in arms:
        dense_arm(hinge(union, KNOTS), y, 'hinge(merged+new)', rows, args.out)
    if 'mlp' in arms:
        for hidden in (256, 1024):
            mlp_arm(union, y, 'mlp(merged+new)', hidden, rows, args.out)


if __name__ == '__main__':
    main()
