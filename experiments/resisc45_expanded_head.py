#!/usr/bin/env python
"""A budgeted RESISC45 head over the nonlinearly expanded pool at 4,096 values.

``resisc45_pool4_ceiling.py`` and ``resisc45_nonlinear_probe.py`` establish
three zero-parameter ways to move the pool's ceiling: the fourth pool
(+1.3 linear test points), a hinge at each column's mean (+1.7) and ReLUs of
random signed column pairs (+3.3).  This asks the only question that matters
for the target -- how much of that a head can *buy* under 4,096 deployed
values -- by building the expanded candidate pool and fitting the incumbent
separable-dictionary head (16,384 Gaussian class atoms, identity plus
exhaustive signed column pairs) at code budgets that land near the target once
everything a deployment stores is charged.

Candidate lists.  A single group-lasso ranking over the widened pool floods
with the individually strong, collectively redundant families (163 of 512
slots go to the random-texton histograms, 163 to the object-layout pool) and
its linear ceiling drops to 76.8% test; the same ranking over the expanded pool
fills 480 of 504 used columns with pair ReLUs and reads 73%.  ``--ranking
plain`` reproduces that.  The default ``quota`` list takes 384 base, 128
layout and 128 fourth-pool columns from three separate rankings (linear ceiling
81.0% test with 640 columns) and builds the expansions over those.  Expanded
columns are then admitted either all at once (raw + hinge, 1,280 columns) or
through the support of an element-wise prune-and-regrow head over the whole
expanded pool, whose regrow criterion scores a column against the columns
already active rather than against the label alone.

Deployment accounting.  ``sep_dict_deployed_values`` prices the code, the
intercept and the pair-atom scaling ratios.  An expanded column adds constants
of its own that no fold can absorb: a hinge column ``relu(x_j - knot_j)`` costs
its knot (one value: the scale ``1/sigma_j`` is absorbable), and a pair-ReLU
column ``relu(a x_i + b x_j + c)`` costs two (the ratio ``b/a`` and the offset
``c/a``; the common scale is absorbable).  Raw columns cost nothing beyond
what the base accounting already charges.  ``deployed_values`` is the sum.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import (
    NUM_CLASSES,
    accuracy,
    fit_rigl_ref_logreg_gpu,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    pair_atoms,
    refit_masked_ref_logreg_gpu,
    refit_masked_sep_dict_ref_logreg_gpu,
    sep_dict_deployed_values,
    standardise,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_expanded_head_result.csv')
NEW_POOL = 'gpu4_pool'
ROWS = NUM_CLASSES - 1
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
RIGL_SETTING = (4000, 100, 0.5)
QUOTA = (384, 128, 128)     # base, layout, fourth-pool columns
PLAIN_LIST = 512
PAIR_COLUMNS = 8192         # random signed pair-ReLU columns over the raw list
SUPPORT_NNZ = (2048, 4096)  # element-wise searches whose support defines a list
PAIR_TOP = 256
BUDGETS = (1024, 2048, 3072, 3584)
SEED = 0
CLASS_ATOMS = 16384


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def expand(raw: dict, seed: int = SEED) -> tuple[dict, np.ndarray]:
    """Raw list plus hinge-at-mean and random signed pair-ReLU columns.

    Returns the expanded matrices and the per-column constant count a
    deployment must store (0 raw, 1 hinge, 2 pair).
    """
    mu, sigma = standardise(raw['train'])
    k = raw['train'].shape[1]
    rng = np.random.default_rng(seed)
    left = rng.integers(0, k, size=PAIR_COLUMNS)
    right = (left + rng.integers(1, k, size=PAIR_COLUMNS)) % k
    sign_l = rng.choice((-1.0, 1.0), size=PAIR_COLUMNS).astype(np.float32)
    sign_r = rng.choice((-1.0, 1.0), size=PAIR_COLUMNS).astype(np.float32)
    out = {}
    for split, block in raw.items():
        z = ((block - mu) / sigma).astype(np.float32)
        out[split] = np.concatenate([
            block.astype(np.float32),
            np.maximum(z, 0.0),
            np.maximum(z[:, left] * sign_l + z[:, right] * sign_r, 0.0),
        ], axis=1)
    constants = np.concatenate([np.zeros(k), np.ones(k), 2 * np.ones(PAIR_COLUMNS)]).astype(int)
    return out, constants


def column_constants(support: np.ndarray, feat_dict: np.ndarray, fatoms: int,
                     constants: np.ndarray) -> int:
    """Constants owed by every expanded column any live atom touches."""
    atoms = np.unique(np.asarray(support) % fatoms)
    touched = np.flatnonzero((np.asarray(feat_dict)[:, atoms] != 0.0).any(1))
    return int(constants[touched].sum())


def kind_counts(used: np.ndarray, consts: np.ndarray) -> dict:
    return {'raw_columns': int((consts[used] == 0).sum()),
            'hinge_columns': int((consts[used] == 1).sum()),
            'pair_columns': int((consts[used] == 2).sum())}


def sweep_sep(xs, y, cdict, fdict, support, steps):
    best = None
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(
            xs['train'], y['train'], cdict, fdict, support, C=C, steps=steps, bias='free')
        val = accuracy((xs['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'val_accuracy': val, 'w': w,
                    'test_accuracy': accuracy((xs['test'] @ w.T + b).argmax(1), y['test'])}
    return best


def elementwise(xs, y, nnz, steps):
    """Prune-and-regrow element-wise head over ``xs`` plus its convex refit."""
    epochs, updates, drop = RIGL_SETTING
    w, _ = fit_rigl_ref_logreg_gpu(xs['train'], y['train'], nonzeros=nnz, epochs=epochs,
                                   updates=updates, drop_fraction=drop)
    mask = (w[1:] != 0).astype(np.float32)
    best = None
    for C in C_GRID:
        w, b = refit_masked_ref_logreg_gpu(xs['train'], y['train'], mask, C=C, steps=steps)
        val = accuracy((xs['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'val_accuracy': val, 'w': w,
                    'test_accuracy': accuracy((xs['test'] @ w.T + b).argmax(1), y['test'])}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--budgets', type=int, nargs='*', default=list(BUDGETS))
    parser.add_argument('--ranking', choices=('quota', 'plain'), default='quota')
    parser.add_argument('--arms', nargs='*', default=None)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    arms = set(args.arms or ('raw', 'hinge', 'support', 'elementwise'))

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    new, _, _ = load_pools((NEW_POOL,))
    pool = {s: np.concatenate([base[s], layout[s], new[s]], axis=1) for s in base}
    widths = (base['train'].shape[1], layout['train'].shape[1])
    print(f'raw pool {pool["train"].shape[1]} columns', flush=True)
    t0 = time.time()
    if args.ranking == 'plain':
        raw_idx = group_lasso_rank(pool['train'], y['train'], lam=LAM, epochs=1500)[0][:PLAIN_LIST]
    else:
        orders = [group_lasso_rank(block['train'], y['train'], lam=LAM, epochs=1500)[0]
                  for block in (base, layout, new)]
        offsets = (0, widths[0], widths[0] + widths[1])
        raw_idx = np.concatenate([o[:q] + off for o, q, off in zip(orders, QUOTA, offsets)])
    raw_idx = raw_idx.astype(np.int64)
    raw = {s: pool[s][:, raw_idx] for s in pool}
    expanded, constants = expand(raw)
    k_raw = len(raw_idx)
    print(f'{args.ranking} list {k_raw} -> expanded {expanded["train"].shape[1]} columns '
          f'in {time.time() - t0:.0f}s', flush=True)
    cdict = gaussian_atoms(CLASS_ATOMS)
    epochs, updates, drop = RIGL_SETTING
    rows: list[dict] = []

    def sep_arm(name, cols, budgets):
        xs = {s: expanded[s][:, cols] for s in expanded}
        consts = constants[cols]
        inner = group_lasso_rank(xs['train'], y['train'], lam=LAM, epochs=1500)[0]
        fdict = pair_atoms(len(cols), PAIR_TOP, inner)
        fatoms = fdict.shape[1]
        for budget in budgets:
            t1 = time.time()
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                xs['train'], y['train'], cdict, fdict, nonzeros=budget, epochs=epochs,
                updates=updates, drop_fraction=drop, bias='free')
            best = sweep_sep(xs, y, cdict, fdict, support, args.steps)
            cost = sep_dict_deployed_values(support, fdict, fatoms, 'free', True, None)
            extra = column_constants(support, fdict, fatoms, consts)
            used = np.flatnonzero(np.abs(best['w']).sum(0) > 0)
            rows.append({
                'arm': name, 'ranking': args.ranking, 'candidates': len(cols),
                'budget': budget, 'code': cost['code'],
                'intercept_values': cost['intercept'], 'scaling_values': cost['scaling'],
                'column_constants': extra, 'deployed_values': cost['total'] + extra,
                'deployed_bits': round(32 * (cost['total'] + extra)
                                       + len(support) * (math.log2(CLASS_ATOMS)
                                                         + math.log2(fatoms))),
                'features': len(used), **kind_counts(used, consts),
                'structure': f'C={best["C"]}',
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
                'seconds': round(time.time() - t1, 1),
            })
            print(rows[-1], flush=True)
            write_rows(rows, args.out)

    if 'raw' in arms:
        sep_arm('sepdict/raw', np.arange(k_raw), args.budgets)
    if 'hinge' in arms:
        sep_arm('sepdict/raw+hinge', np.arange(2 * k_raw), args.budgets)
    if 'support' in arms:
        for nnz in SUPPORT_NNZ:
            t1 = time.time()
            best = elementwise(expanded, y, nnz, args.steps)
            cols = np.flatnonzero(np.abs(best['w']).sum(0) > 0)
            print(f'support list from nnz={nnz}: {len(cols)} columns '
                  f'{kind_counts(cols, constants)} val={best["val_accuracy"]:.4f} '
                  f'test={best["test_accuracy"]:.4f} in {time.time() - t1:.0f}s', flush=True)
            sep_arm(f'sepdict/support{nnz}', cols, args.budgets)
    if 'elementwise' in arms:
        for budget in args.budgets:
            t1 = time.time()
            best = elementwise(expanded, y, budget, args.steps)
            used = np.flatnonzero(np.abs(best['w']).sum(0) > 0)
            extra = int(constants[used].sum())
            rows.append({
                'arm': 'elementwise/expanded', 'ranking': args.ranking,
                'candidates': expanded['train'].shape[1], 'budget': budget, 'code': budget,
                'intercept_values': ROWS, 'scaling_values': 0, 'column_constants': extra,
                'deployed_values': budget + ROWS + extra,
                'deployed_bits': round(32 * (budget + ROWS + extra)
                                       + budget * math.log2(ROWS * expanded['train'].shape[1])),
                'features': len(used), **kind_counts(used, constants),
                'structure': f'C={best["C"]}',
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
                'seconds': round(time.time() - t1, 1),
            })
            print(rows[-1], flush=True)
            write_rows(rows, args.out)


if __name__ == '__main__':
    main()
