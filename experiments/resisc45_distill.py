#!/usr/bin/env python
"""Can a teacher buy RESISC45 accuracy the budget does not have to pay for?

Everything that moved the RESISC45 parameter frontier in this run changed the
*parameterisation* of the head.  Distillation is the obvious lever that changes
neither the head nor the pool: the teacher is a training-time object, so a
student that learns more from soft targets than from hard labels is accuracy
for zero stored values.  The budgeted head is starved -- 65% test is cleared at
208 values against a 79.4% pool ceiling -- which is exactly the regime where a
teacher is supposed to help most, and with 45 classes the soft targets carry a
lot of confusion structure the one-hot labels do not.

The student is iteration 7's separable-dictionary head, fitted with

    (1 - alpha) * CE(hard) + alpha * T^2 * KL(teacher_T || student_T)

in *both* the prune-and-regrow search (the loss it descends and the regrow
criterion, which is the same gradient) and the convex refit, which stays convex
because a cross-entropy against a fixed target distribution is.

The arms separate the three things that could be doing the work:

* ``hard``        -- alpha = 0, the incumbent;
* ``insample``    -- the full-pool logistic teacher's own train logits.  It is
  95.0% accurate on train, so its targets are confident;
* ``xfit``        -- the same teacher 5-fold cross-fitted, so its train targets
  are 79.7% accurate and carry honest uncertainty;
* ``candidate``   -- a teacher fitted on the student's own 512 candidate
  columns, the control for "the teacher knows columns the student cannot see";
* ``smooth``      -- a *uniform* teacher at T = 1, which is exactly label
  smoothing at eps = alpha, the control for "any softening regularises".

``C`` is chosen on validation, test is read once per row, and every row carries
a paired bootstrap of its test difference against the hard-label arm at the
same budget.
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
)
from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    accuracy,
    fit_logreg_gpu,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    pair_atoms,
    refit_masked_sep_dict_ref_logreg_gpu,
    sep_dict_logreg_params,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_distill_result.csv')
BUDGETS = (208, 256)
CANDIDATE_LIST = (512, 128)
RIGL_SETTING = (4000, 100, 0.5)
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
TEACHER_C_GRID = (0.01, 0.03, 0.1, 0.3)
FOLDS = 5
ROWS = NUM_CLASSES - 1
SEED = 0
BOOTSTRAP = 2000
# (teacher, temperature, alpha)
ARMS = (
    ('hard', 1.0, 0.0),
    ('insample', 1.0, 0.5), ('insample', 1.0, 0.9),
    ('insample', 2.0, 0.5), ('insample', 2.0, 0.9),
    ('insample', 4.0, 0.5), ('insample', 4.0, 0.9),
    ('xfit', 1.0, 0.5), ('xfit', 1.0, 0.9),
    ('xfit', 2.0, 0.5), ('xfit', 2.0, 0.9),
    ('xfit', 4.0, 0.5), ('xfit', 4.0, 0.9),
    ('candidate', 2.0, 0.5), ('candidate', 2.0, 0.9),
    ('smooth', 1.0, 0.1), ('smooth', 1.0, 0.3),
)
INCUMBENT = 'hard'


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    """``count`` unit-norm Gaussian directions from a fixed seed."""
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def fit_teacher(x, y, name: str):
    """Full-batch logistic teacher with ``C`` chosen on validation."""
    best = None
    for C in TEACHER_C_GRID:
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        val = accuracy((x['val'] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best[0]:
            best = (val, C, w, b)
    val, C, w, b = best
    test = accuracy((x['test'] @ w.T + b).argmax(1), y['test'])
    train = accuracy((x['train'] @ w.T + b).argmax(1), y['train'])
    print(f'teacher {name}: C={C} train={train:.4f} val={val:.4f} test={test:.4f}',
          flush=True)
    return C, (x['train'] @ w.T + b).astype(np.float32)


def cross_fit(x, y, C: float, name: str) -> np.ndarray:
    """Out-of-fold train logits, so the soft targets are honestly uncertain."""
    logits = np.zeros((len(y['train']), NUM_CLASSES), dtype=np.float32)
    folds = np.random.default_rng(SEED).permutation(np.arange(len(y['train'])) % FOLDS)
    for fold in range(FOLDS):
        keep = folds != fold
        w, b = fit_logreg_gpu(x['train'][keep], y['train'][keep], C=C, steps=400)
        logits[~keep] = (x['train'][~keep] @ w.T + b).astype(np.float32)
    print(f'teacher {name}: cross-fitted train={accuracy(logits.argmax(1), y["train"]):.4f}',
          flush=True)
    return logits


def sweep_c(xtr, ytr, x, y, idx, cdict, fdict, support, steps, teacher, alpha, temp):
    """Refit a fixed code support at several ``C``; keep the best validation fit."""
    best = None
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(
            xtr, ytr, cdict, fdict, support, C=C, steps=steps,
            teacher_logits=teacher, alpha=alpha, temperature=temp)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            pred = (x['test'][:, idx] @ w.T + b).argmax(1)
            best = {'C': C, 'val_accuracy': val, 'pred_test': pred,
                    'test_accuracy': accuracy(pred, y['test']),
                    'features': int((np.abs(w).sum(0) > 0).sum())}
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
    """Print every arm against the hard-label head at the same budget."""
    for budget in sorted({r['budget'] for r in rows}):
        same = [r for r in rows if r['budget'] == budget]
        control = next(r for r in same if r['arm'] == INCUMBENT)
        print(f'\n{budget} stored values -- hard labels: val {control["val_accuracy"]:.4f} '
              f'test {control["test_accuracy"]:.4f}')
        for row in same:
            if row['arm'] == INCUMBENT:
                continue
            print(f'  {row["arm"]:<22s} val {row["val_accuracy"]:.4f} '
                  f'test {row["test_accuracy"]:.4f}  {row["test_delta"]:+.4f} '
                  f'[{row["delta_lo"]:+.4f}, {row["delta_hi"]:+.4f}]')
        best = max(same, key=lambda r: r['val_accuracy'])
        print(f'  validation picks {best["arm"]} '
              f'(test {best["test_accuracy"]:.4f}, '
              f'{best["test_accuracy"] - control["test_accuracy"]:+.4f} against hard labels)')


def load_rows(path: str) -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'parameters', 'features'):
            row[key] = int(row[key])
        for key in ('val_accuracy', 'test_accuracy', 'test_delta', 'delta_lo',
                    'delta_hi', 'alpha', 'temperature'):
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
    narrow = {split: merged[split][:, idx] for split in merged}

    C_full, insample = fit_teacher(merged, y, 'full pool')
    _, candidate = fit_teacher(narrow, y, 'candidate list')
    teachers = {
        'hard': None,
        'insample': insample,
        'xfit': cross_fit(merged, y, C_full, 'full pool'),
        'candidate': candidate,
        # A uniform teacher at T = 1 is exactly label smoothing at eps = alpha.
        'smooth': np.zeros_like(insample),
    }

    cdict = gaussian_atoms(16384)
    fdict = pair_atoms(len(idx), 256, inner_order)
    epochs, updates, drop = RIGL_SETTING
    rows: list[dict[str, object]] = []
    for budget in (args.budgets or BUDGETS):
        reference = None
        for teacher_name, temp, alpha in ARMS:
            teacher = teachers[teacher_name]
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                xtr, y['train'], cdict, fdict, nonzeros=budget - ROWS,
                epochs=epochs, updates=updates, drop_fraction=drop,
                teacher_logits=teacher, alpha=alpha, temperature=temp)
            best = sweep_c(xtr, y['train'], merged, y, idx, cdict, fdict, support,
                           args.steps, teacher, alpha, temp)
            arm = (INCUMBENT if teacher_name == 'hard'
                   else f'{teacher_name} T={temp:g} a={alpha:g}')
            if reference is None:
                reference = best['pred_test']
                lo = hi = delta = 0.0
            else:
                lo, hi = paired_bootstrap(best['pred_test'], reference, y['test'])
                delta = best['test_accuracy'] - accuracy(reference, y['test'])
            rows.append({
                'budget': budget, 'arm': arm, 'teacher': teacher_name,
                'temperature': temp, 'alpha': alpha,
                'parameters': sep_dict_logreg_params(len(support)),
                'features': best['features'],
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
