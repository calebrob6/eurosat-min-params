#!/usr/bin/env python
"""Can a fixed class dictionary buy a RESISC45 class more weights than it stores?

Every RESISC45 head so far spends one stored value on one ``(class, column)``
weight.  `resisc45_quadratic.py` showed why that now binds: above about 512
stored values the budgeted head already reads 348 distinct pool columns, so it
is *weight-limited, not column-limited*, and neither a wider pool nor a better
support search relieves it.  The remaining lever is the head's
parameterisation.

Here the ``44 x k`` reference-class weight matrix is generated as ``D @ P``:
``D [44, atoms]`` is a **fixed, zero-parameter** dictionary of unit-norm class
directions and ``P [atoms, k]`` is a sparse code searched by the same
prune-and-regrow procedure and refit convexly afterwards.  One stored value now
buys a whole *pattern* across classes instead of a single class's weight, so a
column that eight classes want can cost one value instead of eight.  Stored
values are ``nnz(P) + 44`` biases, exactly as before.

Six dictionaries are compared:

* ``singleton``   -- the 44 identity atoms, i.e. the incumbent element-wise
  sparse head run through the same code path (it reproduces it exactly);
* ``mean-tree``   -- singletons plus the 43 internal nodes of a Ward tree over
  the class means of the standardised pool;
* ``weight-tree`` -- singletons plus the 43 internal nodes of a Ward tree over
  the rows of the unconstrained dense head, the structure the code is actually
  asked to approximate;
* ``random87``    -- singletons plus 43 random class subsets whose sizes match
  the ``mean-tree`` nodes: the control that separates *which* groups from
  merely having more atoms to search over;
* ``graded2048``  -- singletons plus 2,048 random subsets of graded size, a
  data-independent over-complete indicator dictionary;
* ``gaussN``      -- ``N`` unit-norm Gaussian directions drawn from a fixed
  seed, containing no identity atom at all.

The last three depend on nothing but a seed and a generation rule, so they are
as free as the feature extractors themselves.  ``C`` is chosen on validation and
test is read once per row.  Because iteration 5 established that a 6,300-image
split resolves only about +-0.6 points, every row also carries a paired
bootstrap of its test-accuracy difference against the ``singleton`` arm on the
same candidate list at the same budget, and the index pattern is reported in
bits as well as in entries -- a wider dictionary needs more bits per stored
value, which is the one cost this parameterisation really does pay.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
from scipy.cluster.hierarchy import linkage

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
    dict_logreg_params,
    fit_logreg_gpu,
    fit_rigl_dict_ref_logreg_gpu,
    group_lasso_rank,
    refit_masked_dict_ref_logreg_gpu,
    standardise,
    unit_atoms,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_class_dict_result.csv')
BUDGETS = (128, 192, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
# The three budgets that carry the full dictionary comparison; the rest carry
# only the arms that trace the frontier.
ANCHORS = (256, 512, 1024)
CANDIDATE_LIST = (512, 128)  # (columns, slots reserved for the object-layout pool)
# The prune-and-regrow setting iteration 4 selected on validation at most
# budgets, held fixed so arms differ only by the dictionary.
RIGL_SETTING = (4000, 100, 0.5)
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
# (dictionary, candidate list, every budget or anchors only)
ARMS = (
    ('singleton', 'top512', True),
    ('singleton', 'whole-pool', True),
    ('gauss4096', 'top512', True),
    ('gauss4096', 'whole-pool', True),
    ('gauss16384', 'top512', True),
    ('mean-tree', 'top512', False),
    ('weight-tree', 'top512', False),
    ('random87', 'top512', False),
    ('gauss1024', 'top512', False),
    ('gauss8192', 'top512', False),
    ('graded2048', 'top512', False),
    ('graded2048', 'whole-pool', False),
    ('gauss32768', 'top512', False),
    ('pca44', 'top512', False),
    ('dct44', 'top512', False),
    ('gauss16384-s1', 'top512', False),
    ('gauss16384-s2', 'top512', False),
)
ROWS = NUM_CLASSES - 1
GRADED_SIZES = (2, 3, 4, 6, 8, 12, 16, 22)
SEED = 0
BOOTSTRAP = 2000
# Replications of the winning dictionary under other seeds: reported, but kept
# out of the validation gate, because picking a seed on validation is exactly
# the arm-shopping iteration 5 showed spends the split's resolution on nothing.
REPLICATIONS = ('gauss16384-s1', 'gauss16384-s2')
TARGETS = (0.65, 0.70)


def ward_nodes(rep: np.ndarray) -> list[np.ndarray]:
    """Member sets of every internal node of a Ward tree over ``rep``'s rows.

    Columns are z-scored across the classes first so no single coordinate
    dominates the merge order.  A tree over ``rows`` leaves has ``rows - 1``
    internal nodes, every one of which is a candidate class group.
    """
    z = rep - rep.mean(0, keepdims=True)
    scale = z.std(0, keepdims=True)
    z = z / np.where(scale < 1e-12, 1.0, scale)
    merges = linkage(z, method='ward')
    members: dict[int, np.ndarray] = {i: np.array([i]) for i in range(len(rep))}
    nodes = []
    for step, (a, b, _, _) in enumerate(merges):
        group = np.concatenate((members[int(a)], members[int(b)]))
        members[len(rep) + step] = group
        nodes.append(np.sort(group))
    return nodes


def random_nodes(sizes: list[int], seed: int = SEED) -> list[np.ndarray]:
    """Random class subsets with a prescribed size distribution."""
    rng = np.random.default_rng(seed)
    return [np.sort(rng.choice(ROWS, size=size, replace=False)) for size in sizes]


def gaussian_atoms(count: int, seed: int = SEED) -> np.ndarray:
    """``count`` unit-norm Gaussian class directions from a fixed seed."""
    atoms = np.random.default_rng(seed).standard_normal((ROWS, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def dct_atoms(size: int = ROWS) -> np.ndarray:
    """An orthonormal DCT-II basis over the classes: a fixed change of basis."""
    j = np.arange(size)
    atoms = np.cos(np.pi * (j[:, None] + 0.5) * j[None, :] / size)
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def build_dictionaries(x, y) -> dict[str, np.ndarray]:
    """Every dictionary the arms refer to, keyed by name."""
    xtr = x['train']
    mu, sigma = standardise(xtr)
    xs = (xtr - mu) / sigma
    means = np.stack([xs[y['train'] == c].mean(0) for c in range(1, NUM_CLASSES)])

    w_eff, _ = fit_logreg_gpu(xtr, y['train'], C=0.1, steps=400)
    w_std = w_eff * sigma[None, :]
    w_rows = w_std[1:] - w_std[0][None, :]  # reference-class form, class 0 at zero

    singles = [np.array([i]) for i in range(ROWS)]
    mean_tree = ward_nodes(means)
    rng = np.random.default_rng(SEED)
    graded = [np.sort(rng.choice(ROWS, size=GRADED_SIZES[j % len(GRADED_SIZES)], replace=False))
              for j in range(2048)]
    return {
        'singleton': unit_atoms(singles, ROWS),
        'mean-tree': unit_atoms(singles + mean_tree, ROWS),
        'weight-tree': unit_atoms(singles + ward_nodes(w_rows), ROWS),
        'random87': unit_atoms(singles + random_nodes([len(g) for g in mean_tree]), ROWS),
        'graded2048': unit_atoms(singles + graded, ROWS),
        'gauss1024': gaussian_atoms(1024),
        'gauss4096': gaussian_atoms(4096),
        'gauss8192': gaussian_atoms(8192),
        'gauss16384': gaussian_atoms(16384),
        'gauss32768': gaussian_atoms(32768),
        # Replications of the winning dictionary from two other seeds: the
        # dictionary is random, so the frontier must not depend on which draw.
        'gauss16384-s1': gaussian_atoms(16384, SEED + 1),
        'gauss16384-s2': gaussian_atoms(16384, SEED + 2),
        # Complete 44-atom bases: the model class is *identical* to the
        # element-wise head, only the basis the code is sparse in changes.
        'pca44': np.linalg.svd(w_rows, full_matrices=False)[0],
        'dct44': dct_atoms(),
    }


def sweep_c(xtr, ytr, x, y, idx, dictionary, mask, steps):
    """Refit a fixed code support at several ``C``; keep the best validation fit."""
    best = None
    for C in C_GRID:
        w, b = refit_masked_dict_ref_logreg_gpu(xtr, ytr, dictionary, mask, C=C, steps=steps)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            pred = (x['test'][:, idx] @ w.T + b).argmax(1)
            best = {
                'C': C, 'val_accuracy': val,
                'test_accuracy': accuracy(pred, y['test']),
                'pred_test': pred,
                'features': int((np.abs(w).sum(0) > 0).sum()),
                'dense_nonzeros': int((np.abs(w) > 1e-12).sum()),
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


def total_bits(row: dict) -> int:
    """float32 stored values plus the index pattern, in bits.

    A wider dictionary needs more bits per stored value -- ``log2(atoms)``
    instead of ``log2(44)`` for the atom id -- which is the one price this
    parameterisation genuinely pays, so the frontier is also read at equal bits.
    """
    return 32 * row['parameters'] + row['index_bits']


def bits_matched(rows: list[dict]) -> None:
    """Each budget's best arm against the element-wise head at equal total bits."""
    single = []
    for budget in sorted({r['budget'] for r in rows}):
        same = [r for r in rows if r['budget'] == budget and r['dictionary'] == 'singleton']
        if same:
            pick = max(same, key=lambda r: r['val_accuracy'])
            single.append((total_bits(pick), pick['test_accuracy']))
    xs = np.array([b for b, _ in single], dtype=float)
    ys = np.array([a for _, a in single], dtype=float)
    print('\nequal-bits comparison (against the validation-selected element-wise arm):')
    for budget in sorted({r['budget'] for r in rows}):
        best = max((r for r in rows if r['budget'] == budget
                    and r['dictionary'] not in REPLICATIONS), key=lambda r: r['val_accuracy'])
        if best['dictionary'] == 'singleton':
            continue
        bits = total_bits(best)
        matched = float(np.interp(bits, xs, ys))
        note = '  (beyond the element-wise curve)' if bits > xs.max() else ''
        print(f'  {budget:>5d} values  {best["arm"]:<28s} {bits/8/1024:6.2f} KiB  '
              f'test={best["test_accuracy"]:.4f}  element-wise at equal bits={matched:.4f}  '
              f'{best["test_accuracy"] - matched:+.4f}{note}')


def summarise(rows: list[dict]) -> None:
    """Print the dictionary comparison and where each target is first cleared."""
    arms = [a for a in dict.fromkeys(r['arm'] for r in rows)]
    budgets = sorted({r['budget'] for r in rows})
    print(f'\n{"budget":>7s}  ' + '  '.join(f'{a:>28s}' for a in arms))
    for budget in budgets:
        cells = []
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit:
                cells.append('-')
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            cells.append(f'val {best["val_accuracy"]:.4f} test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>28s}' for c in cells))

    print('\nvs the element-wise head on the same list (paired bootstrap of the test difference):')
    for budget in budgets:
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit or hit[0]['dictionary'] == 'singleton':
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            print(f'  {budget:>5d}  {arm:<28s} {best["test_delta"]:+.4f} '
                  f'[{best["delta_lo"]:+.4f}, {best["delta_hi"]:+.4f}]')

    print('\nvalidation-selected arm per budget (replication arms excluded):')
    selected = []
    for budget in budgets:
        best = max((r for r in rows if r['budget'] == budget
                    and r['dictionary'] not in REPLICATIONS), key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {best["parameters"]:>5d} values  {best["arm"]:<28s} val={best["val_accuracy"]:.4f} '
              f'test={best["test_accuracy"]:.4f}  {best["features"]} columns, '
              f'{best["atoms_used"]} atoms, {best["dense_nonzeros"]} deployed weights, '
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
        for key in ('budget', 'parameters', 'features', 'index_pattern', 'atoms', 'atoms_used',
                    'dense_nonzeros', 'index_bits'):
            row[key] = int(row[key])
        for key in ('val_accuracy', 'test_accuracy', 'test_delta', 'delta_lo', 'delta_hi'):
            row[key] = float(row[key])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--budgets', type=int, nargs='*', default=None)
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--only-dicts', nargs='*', default=None,
                        help='run only these dictionaries (added on the top512 list if '
                             'they are not already an arm), for extending a finished CSV')
    args = parser.parse_args()
    if args.summarise:
        rows = load_rows(args.out)
        summarise(rows)
        bits_matched(rows)
        return

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    width = merged['train'].shape[1]
    size, quota = CANDIDATE_LIST
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    lists = {
        f'top{size}': np.concatenate((base_order[:size - quota],
                                      layout_order[:quota] + base_width)).astype(np.int64),
        'whole-pool': np.arange(width, dtype=np.int64),
    }
    dictionaries = build_dictionaries(merged, y)
    for name, d in dictionaries.items():
        print(f'{name}: {d.shape[1]} atoms', flush=True)

    arms = ARMS
    if args.only_dicts:
        known = {name for name, _, _ in ARMS}
        arms = tuple(a for a in ARMS if a[0] in args.only_dicts)
        arms += tuple((name, f'top{size}', True) for name in args.only_dicts
                      if name not in known)
    epochs, updates, drop = RIGL_SETTING
    rows: list[dict[str, object]] = []
    for budget in (args.budgets or BUDGETS):
        nonzeros = budget - ROWS
        preds: dict[str, np.ndarray] = {}
        for dict_name, list_name, every_budget in arms:
            if not every_budget and budget not in ANCHORS:
                continue
            idx = lists[list_name]
            dictionary = dictionaries[dict_name]
            xtr = merged['train'][:, idx]
            w, _, mask = fit_rigl_dict_ref_logreg_gpu(
                xtr, y['train'], dictionary, nonzeros=nonzeros,
                epochs=epochs, updates=updates, drop_fraction=drop)
            best = sweep_c(xtr, y['train'], merged, y, idx, dictionary, mask, args.steps)
            arm = f'{dict_name}/{list_name}'
            preds[arm] = best['pred_test']
            reference = preds.get(f'singleton/{list_name}')
            if reference is not None and dict_name != 'singleton':
                lo, hi = paired_bootstrap(best['pred_test'], reference, y['test'])
                delta = best['test_accuracy'] - accuracy(reference, y['test'])
            else:
                lo = hi = delta = 0.0
            stored = int(mask.sum())
            rows.append({
                'budget': budget, 'arm': arm, 'dictionary': dict_name, 'candidates': list_name,
                'parameters': dict_logreg_params(stored),
                'features': best['features'],
                'index_pattern': stored,
                'atoms': dictionary.shape[1],
                'atoms_used': int((mask.sum(1) > 0).sum()),
                'dense_nonzeros': best['dense_nonzeros'],
                'index_bits': int(round(stored * (math.log2(dictionary.shape[1])
                                                  + math.log2(len(idx))))),
                'structure': f'{list_name}/e{epochs}/u{updates}/d{drop}/C={best["C"]}',
                'val_accuracy': round(best['val_accuracy'], 4),
                'test_accuracy': round(best['test_accuracy'], 4),
                'test_delta': round(delta, 4),
                'delta_lo': round(lo, 4), 'delta_hi': round(hi, 4),
            })
            print(' '.join(f'{k}={v}' for k, v in rows[-1].items()), flush=True)
        write_rows(rows, args.out)
    summarise(rows)
    bits_matched(rows)


if __name__ == '__main__':
    main()
