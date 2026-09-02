#!/usr/bin/env python
"""Can a stored value buy a *column* pattern as well as a class pattern?

`resisc45_class_dict.py` generated the ``44 x k`` head as ``Dc @ P`` with ``Dc``
a fixed over-complete dictionary of class directions, and that halved the
budget at both RESISC45 targets.  Its lesson was mechanical rather than
semantic: what paid was over-completeness -- letting the support search *choose*
a class pattern per stored value instead of spelling one out class by class.
The obvious question is whether the same trick works on the other axis, where
each stored value still buys exactly one pool column.

Here the head is ``Dc @ P @ Df.T``: ``Dc [44, catoms]`` of unit-norm class
directions, ``Df [k, fatoms]`` of unit-norm *column* directions, and ``P``
sparse.  One stored value buys the rank-1 outer product
``Dc[:, a] Df[:, f].T`` -- a class pattern times a column pattern.  With ``Df``
the identity this is exactly the class-dictionary head, and with both identity
it is the element-wise head, so both controls are nested and can only lose by
search, never by expressiveness.

The column dictionaries differ in how *wide* their atoms are, which is the
control that separates "more atoms to search over" from "the right kind of
atom":

* ``identity``    -- the incumbent, one column per atom;
* ``pairsT``      -- identity plus **every** signed pair ``(x_i +- x_j)/sqrt2``
  over the ``T`` highest-ranked columns of the candidate list, for
  ``T`` in 128, 181, 256 and 320;
* ``rand2xN``     -- ``N`` *sampled* signed pairs over all ``k`` columns, the
  control for exhaustive enumeration against sampling;
* ``rand4x``/``rand8x``/``rand32x`` -- sampled atoms of width 4, 8 and 32;
* ``gaussF``      -- dense random directions over all ``k`` columns, the width
  limit;
* ``graded``      -- pairs plus wider sampled atoms, letting the search choose.

Stored values are ``nnz(P) + 44`` biases as in every earlier head, and both
dictionaries are generated from a fixed seed by a stated rule, so they are as
free as the feature extractors.  ``C`` is chosen on validation, test is read
once per row, and each row carries a paired bootstrap of its test difference
against the class-dictionary incumbent at the same budget, because a
6,300-image split only resolves about +-0.6 points.
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
    dict_logreg_params,
    feature_atoms,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    pair_atoms,
    refit_masked_sep_dict_ref_logreg_gpu,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_feature_dict_result.csv')
BUDGETS = (96, 128, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 352, 384,
           448, 512, 640, 768, 896, 1024)
# Budgets that carry the full column-dictionary comparison; the rest carry only
# the arms that trace the frontier.
ANCHORS = (192, 256, 512, 1024)
CANDIDATE_LIST = (512, 128)  # (columns, slots reserved for the object-layout pool)
# The prune-and-regrow setting iterations 4 and 6 selected on validation, held
# fixed so arms differ only by the dictionaries.
RIGL_SETTING = (4000, 100, 0.5)
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
ROWS = NUM_CLASSES - 1
SEED = 0
BOOTSTRAP = 2000
TARGETS = (0.65, 0.70)
# (class dictionary, column dictionary, every budget or anchors only)
ARMS = (
    ('gauss16384', 'identity', True),
    ('singleton', 'identity', True),
    ('singleton', 'pairs181', True),
    ('gauss16384', 'pairs181', True),
    ('gauss16384', 'pairs256', True),
    ('gauss16384', 'rand2x16384', True),
    ('gauss16384', 'pairs128', False),
    ('gauss16384', 'pairs320', False),
    ('gauss16384', 'rand2x65536', False),
    ('gauss16384', 'rand4x16384', False),
    ('gauss16384', 'rand8x16384', False),
    ('gauss16384', 'rand32x16384', False),
    ('gauss16384', 'gaussF16384', False),
    ('gauss16384', 'graded', False),
    ('gauss4096', 'pairs181', False),
    ('gauss32768', 'pairs181', False),
    ('gauss16384', 'rand2x16384-s1', False),
    ('gauss16384', 'rand2x16384-s2', False),
)
# Replications of a random column dictionary under other seeds: reported, but
# kept out of the validation gate, because picking a draw on validation spends
# the split's resolution on nothing.
REPLICATIONS = ('rand2x16384-s1', 'rand2x16384-s2')
INCUMBENT = 'gauss16384/identity'


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    """``count`` unit-norm Gaussian directions from a fixed seed."""
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def build_class_dicts() -> dict[str, np.ndarray]:
    """The class dictionaries the arms refer to, keyed by name."""
    return {
        'singleton': np.eye(ROWS),
        'gauss4096': gaussian_atoms(4096),
        'gauss16384': gaussian_atoms(16384),
        'gauss32768': gaussian_atoms(32768),
    }


def build_feature_dicts(k: int, order: np.ndarray) -> dict[str, np.ndarray]:
    """The column dictionaries the arms refer to, keyed by name.

    ``order`` ranks the candidate list's own columns, so ``pairsT`` enumerates
    the pairs the ranking says are most likely to matter rather than an
    arbitrary prefix of the list.
    """
    dicts = {
        'identity': np.eye(k),
        'pairs128': pair_atoms(k, 128, order),
        'pairs181': pair_atoms(k, 181, order),
        'pairs256': pair_atoms(k, 256, order),
        'pairs320': pair_atoms(k, 320, order),
        'rand2x16384': feature_atoms(k, 16384, 2, SEED),
        'rand2x65536': feature_atoms(k, 65536, 2, SEED),
        'rand4x16384': feature_atoms(k, 16384, 4, SEED),
        'rand8x16384': feature_atoms(k, 16384, 8, SEED),
        'rand32x16384': feature_atoms(k, 16384, 32, SEED),
        'gaussF16384': feature_atoms(k, 16384, k, SEED),
        'rand2x16384-s1': feature_atoms(k, 16384, 2, SEED + 1),
        'rand2x16384-s2': feature_atoms(k, 16384, 2, SEED + 2),
    }
    # A graded dictionary lets the search pick the width instead of being given
    # one, the column-side analogue of the class-side ``graded2048`` arm.
    dicts['graded'] = np.concatenate((dicts['pairs181'],
                                      feature_atoms(k, 8192, 4, SEED, identity=False),
                                      feature_atoms(k, 4096, 8, SEED, identity=False)), axis=1)
    return dicts


def sweep_c(xtr, ytr, x, y, idx, class_dict, feat_dict, support, steps):
    """Refit a fixed code support at several ``C``; keep the best validation fit."""
    best = None
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(xtr, ytr, class_dict, feat_dict,
                                                    support, C=C, steps=steps)
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

    A stored value now needs ``log2(catoms) + log2(fatoms)`` bits of index
    instead of ``log2(catoms) + log2(k)``, which is the price this
    parameterisation pays, so the frontier is also read at equal bits.
    """
    return 32 * row['parameters'] + row['index_bits']


def bits_matched(rows: list[dict]) -> None:
    """Each budget's best arm against the incumbent head at equal total bits."""
    curve = []
    for budget in sorted({r['budget'] for r in rows}):
        same = [r for r in rows if r['budget'] == budget and r['arm'] == INCUMBENT]
        if same:
            pick = max(same, key=lambda r: r['val_accuracy'])
            curve.append((total_bits(pick), pick['test_accuracy']))
    xs = np.array([b for b, _ in curve], dtype=float)
    ys = np.array([a for _, a in curve], dtype=float)
    print(f'\nequal-bits comparison (against {INCUMBENT}):')
    for budget in sorted({r['budget'] for r in rows}):
        best = max((r for r in rows if r['budget'] == budget
                    and r['features_dict'] not in REPLICATIONS), key=lambda r: r['val_accuracy'])
        if best['arm'] == INCUMBENT:
            continue
        bits = total_bits(best)
        matched = float(np.interp(bits, xs, ys))
        note = '  (beyond the incumbent curve)' if bits > xs.max() else ''
        print(f'  {budget:>5d} values  {best["arm"]:<34s} {bits/8/1024:6.2f} KiB  '
              f'test={best["test_accuracy"]:.4f}  incumbent at equal bits={matched:.4f}  '
              f'{best["test_accuracy"] - matched:+.4f}{note}')


def summarise(rows: list[dict]) -> None:
    """Print the dictionary comparison and where each target is first cleared."""
    arms = list(dict.fromkeys(r['arm'] for r in rows))
    budgets = sorted({r['budget'] for r in rows})
    print(f'\n{"budget":>7s}  ' + '  '.join(f'{a:>34s}' for a in arms))
    for budget in budgets:
        cells = []
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit:
                cells.append('-')
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            cells.append(f'val {best["val_accuracy"]:.4f} test {best["test_accuracy"]:.4f}')
        print(f'{budget:>7d}  ' + '  '.join(f'{c:>34s}' for c in cells))

    print(f'\nvs {INCUMBENT} at the same budget (paired bootstrap of the test difference):')
    for budget in budgets:
        for arm in arms:
            hit = [r for r in rows if r['budget'] == budget and r['arm'] == arm]
            if not hit or arm == INCUMBENT:
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            print(f'  {budget:>5d}  {arm:<34s} {best["test_delta"]:+.4f} '
                  f'[{best["delta_lo"]:+.4f}, {best["delta_hi"]:+.4f}]')

    print('\nvalidation-selected arm per budget (replication arms excluded):')
    selected = []
    for budget in budgets:
        best = max((r for r in rows if r['budget'] == budget
                    and r['features_dict'] not in REPLICATIONS), key=lambda r: r['val_accuracy'])
        selected.append(best)
        print(f'  {best["parameters"]:>5d} values  {best["arm"]:<34s} '
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
                    'feature_atoms', 'dense_nonzeros', 'index_bits'):
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
    size, quota = CANDIDATE_LIST
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    idx = np.concatenate((base_order[:size - quota],
                          layout_order[:quota] + base_width)).astype(np.int64)
    xtr = merged['train'][:, idx]
    # Rank the candidate list's own columns so ``pairsT`` pairs the ones the
    # ranking prefers; the quota list is a concatenation, not a sorted order.
    inner_order = group_lasso_rank(xtr, y['train'], lam=LAM, epochs=1500)[0]

    class_dicts = build_class_dicts()
    feature_dicts = build_feature_dicts(len(idx), inner_order)
    for name, d in feature_dicts.items():
        print(f'{name}: {d.shape[1]} column atoms', flush=True)

    epochs, updates, drop = RIGL_SETTING
    rows: list[dict[str, object]] = []
    for budget in (args.budgets or BUDGETS):
        nonzeros = budget - ROWS
        preds: dict[str, np.ndarray] = {}
        for class_name, feat_name, every_budget in ARMS:
            if not every_budget and budget not in ANCHORS:
                continue
            cdict = class_dicts[class_name]
            fdict = feature_dicts[feat_name]
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                xtr, y['train'], cdict, fdict, nonzeros=nonzeros,
                epochs=epochs, updates=updates, drop_fraction=drop)
            best = sweep_c(xtr, y['train'], merged, y, idx, cdict, fdict, support, args.steps)
            arm = f'{class_name}/{feat_name}'
            preds[arm] = best['pred_test']
            reference = preds.get(INCUMBENT)
            if reference is not None and arm != INCUMBENT:
                lo, hi = paired_bootstrap(best['pred_test'], reference, y['test'])
                delta = best['test_accuracy'] - accuracy(reference, y['test'])
            else:
                lo = hi = delta = 0.0
            stored = len(support)
            rows.append({
                'budget': budget, 'arm': arm, 'class_dict': class_name,
                'features_dict': feat_name,
                'parameters': dict_logreg_params(stored),
                'features': best['features'],
                'index_pattern': stored,
                'class_atoms': cdict.shape[1],
                'feature_atoms': fdict.shape[1],
                'dense_nonzeros': best['dense_nonzeros'],
                'index_bits': int(round(stored * (math.log2(cdict.shape[1])
                                                  + math.log2(fdict.shape[1])))),
                'structure': f'e{epochs}/u{updates}/d{drop}/C={best["C"]}',
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
