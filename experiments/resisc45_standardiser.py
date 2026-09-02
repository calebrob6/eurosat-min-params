#!/usr/bin/env python
"""What does the standardiser cost, now that the head is *reconstructed*?

Every section of this file so far has used one accounting convention: "the
feature standardiser is folded into the weights, so it adds no deployed
values".  That is exactly true of a head whose weights are **stored** -- folding
``mu``/``sigma`` into a stored matrix changes the numbers, not how many there
are.  It is *not* true of a head whose weights are **reconstructed** from fixed
dictionaries, which is what every RESISC45 head from the class-dictionary
section onwards has been: a deployment holds a sparse code and rebuilds
``w_eff = (Dc @ P @ Df.T) / sigma`` and ``b_eff = b - sum_j W_j mu_j / sigma_j``,
and those two divisions need numbers that are not in the code.

This experiment prices them exactly and then tries to stop paying:

* **The scaling.** A width-1 column atom needs nothing: ``w_eff[:, j]`` is the
  code entries of column ``j`` divided by ``sigma_j``, so ``sigma`` is
  *absorbable* into values the head already stores.  A width-2 atom ties two
  columns to one value, so its deployed direction ``(s_i/sigma_i, s_j/sigma_j)``
  needs their **ratio**, and ratios compose: the bill is one value per column a
  pair atom touches, minus one per connected component.  The pair dictionary
  that moved the frontier two sections ago therefore carries a charge nobody has
  counted.  Two ways out are tested: identity atoms only (zero scaling cost, and
  a weaker head), or rounding ``sigma`` to a power of two so that columns
  sharing an octave share a ``sigma`` exactly -- their pairs then have ratio 1,
  the common factor is absorbable like a width-1 atom's, and pairs *within an
  octave* cost nothing at all while pairs across octaves cost one value per
  octave rather than one per atom.
* **The centring.** ``b_eff`` is 44 numbers whether or not the head stores an
  intercept, because centring at the training mean *is* an intercept.  The
  intercept section's ``coded``/``none`` heads therefore did not save 44 values
  -- they saved the *second* copy of them.  To actually stop paying, the
  deployed head has to act on uncentred features, which is tested two ways:
  searching the support on uncentred features (a control), and searching on
  centred features and then re-coding the deployed intercept as ``q`` more
  entries of the same sparse code, chosen by matching pursuit over the class
  dictionary and refitted convexly.

Arms are compared at equal *deployed* values -- the code plus everything the
reconstruction needs -- rather than at equal code size, and the frontier is
re-derived under that accounting.  ``C`` is chosen on validation, test is read
once per row, and each row carries a paired bootstrap against the incumbent at
the same code size.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import (  # noqa: E402
    BASE_POOLS,
    LAM,
    LAYOUT_POOL,
    family_of,
    load_pools,
)
from resisc45_lib import (  # noqa: E402
    NUM_CLASSES,
    accuracy,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    pair_atoms,
    refit_masked_sep_dict_ref_logreg_gpu,
    sep_dict_deployed_values,
    standardise,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_standardiser_result.csv')
BUDGETS = (64, 96, 128, 160, 192, 224, 256, 288, 320, 384, 448, 512, 640)
ANCHORS = (160, 256, 384, 512)
CANDIDATE_LIST = (512, 128)  # (columns, slots reserved for the object-layout pool)
RIGL_SETTING = (4000, 100, 0.5)  # held fixed at iterations 4/6's validated setting
C_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
DEFAULT_STEPS = 300  # LBFGS steps in the convex refit; --steps probes convergence
ROWS = NUM_CLASSES - 1
SEED = 0
BOOTSTRAP = 2000
TARGETS = (0.65, 0.70)
PAIR_TOP = 256
# Values of ``q`` for the matching-pursuit re-coding of the deployed intercept.
MP_GRID = (4, 8, 16, 32, 44)
# (arm name, scaling, column dictionary, intercept rule, every budget or anchors only)
ARMS = (
    ('zscore/pairs256/free', 'zscore', 'pairs256', 'free', True),
    ('zscore/pairs256/none', 'zscore', 'pairs256', 'none', True),
    ('zscore/identity/none', 'zscore', 'identity', 'none', True),
    ('zbuck/pairsbuck/none', 'zbuck', 'pairsbuck', 'none', True),
    ('zbuck/pairs256/none', 'zbuck', 'pairs256', 'none', True),
    ('zbuck/pairs256/free', 'zbuck', 'pairs256', 'free', True),
    ('zbuck/pairsbuck/free', 'zbuck', 'pairsbuck', 'free', True),
    ('zscore/identity/free', 'zscore', 'identity', 'free', True),
    ('zfam/pairsfam/none', 'zfam', 'pairsfam', 'none', False),
    ('scale/identity/coded', 'scale', 'identity', 'coded', False),
    ('raw/identity/coded', 'raw', 'identity', 'coded', False),
)
# Arms whose support is re-used for the uncentred intercept re-coding.
MP_ARMS = ('zscore/identity/none', 'zbuck/pairsbuck/none')
INCUMBENT = 'zscore/pairs256/free'
MP_ARM = re.compile(r'(?P<stem>.*)\+(?P<tag>mpu?)(?P<q>\d+)$')


def gaussian_atoms(count: int, rows: int = ROWS, seed: int = SEED) -> np.ndarray:
    """``count`` unit-norm Gaussian class directions from a fixed seed."""
    atoms = np.random.default_rng(seed).standard_normal((rows, count))
    return atoms / np.linalg.norm(atoms, axis=0, keepdims=True)


def group_pair_atoms(columns: int, top: int, order: np.ndarray,
                     groups: np.ndarray) -> np.ndarray:
    """Identity atoms plus every signed pair whose two columns share a group.

    ``pair_atoms`` enumerates all signed pairs over the top-ranked columns, and
    a deployment pays a stored ``sigma_i / sigma_j`` for the columns they touch.
    Restricting the enumeration to pairs whose two columns are scaled by the
    *same* ``sigma`` -- because they fall in the same power-of-two octave, or the
    same feature family -- makes every ratio exactly 1, so the shared factor is
    absorbable into the code value like a width-1 atom's and the scaling bill
    disappears entirely.
    """
    chosen = np.asarray(order)[:top]
    left, right = np.triu_indices(top, k=1)
    same = groups[chosen[left]] == groups[chosen[right]]
    left, right = left[same], right[same]
    count = len(left)
    atoms = np.zeros((columns, 2 * count))
    scale = 1.0 / np.sqrt(2.0)
    span = np.arange(count)
    atoms[chosen[left], span] = scale
    atoms[chosen[right], span] = scale
    atoms[chosen[left], count + span] = scale
    atoms[chosen[right], count + span] = -scale
    return np.concatenate((np.eye(columns), atoms), axis=1)


def build_moments(xtr: np.ndarray, groups: np.ndarray) -> dict[str, tuple]:
    """``(mu, sigma)`` per scaling rule, plus what each one costs to deploy.

    The pair is what the head is fitted and folded with; ``centred`` and
    ``scale_groups`` are what ``sep_dict_deployed_values`` needs to price it.
    """
    mu, sigma = standardise(xtr)
    zero, one = np.zeros_like(mu), np.ones_like(sigma)
    family = np.empty_like(sigma)
    for group in np.unique(groups):
        member = groups == group
        family[member] = np.exp(np.log(sigma[member]).mean())
    octave = np.round(np.log2(sigma)).astype(np.int64)
    single = np.zeros(len(mu), dtype=np.int64)
    return {
        'zscore': (mu, sigma, True, None),
        'zbuck': (mu, np.exp2(octave.astype(np.float64)), True, octave),
        'zfam': (mu, family, True, groups),
        'scale': (zero, sigma, False, None),
        'raw': (zero, one, False, single),
    }


def matching_pursuit(target: np.ndarray, atoms: np.ndarray, count: int) -> np.ndarray:
    """Greedy unit-norm atom indices whose span best explains ``target``."""
    residual = np.array(target, dtype=np.float64)
    chosen: list[int] = []
    for _ in range(count):
        proj = atoms.T @ residual
        proj[chosen] = 0.0
        best = int(np.argmax(np.abs(proj)))
        residual = residual - proj[best] * atoms[:, best]
        chosen.append(best)
    return np.array(chosen, dtype=np.int64)


def coded_support(support: np.ndarray, fatoms: int, extra: np.ndarray) -> np.ndarray:
    """Re-index a bias-free support for the augmented dictionary and add bias entries."""
    rows, cols = support // fatoms, support % fatoms
    return np.concatenate((rows * (fatoms + 1) + cols,
                           extra * (fatoms + 1) + fatoms)).astype(np.int64)


def sweep_c(x, y, idx, class_dict, feat_dict, support, steps, bias, moments,
            penalise_bias=True):
    """Refit a fixed code support at several ``C``; keep the best validation fit."""
    best = None
    xtr = x['train'][:, idx]
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(
            xtr, y['train'], class_dict, feat_dict, support, C=C, steps=steps,
            bias=bias, moments=moments, penalise_bias=penalise_bias)
        val = accuracy((x['val'][:, idx] @ w.T + b).argmax(1), y['val'])
        if best is None or val > best['val_accuracy']:
            pred = (x['test'][:, idx] @ w.T + b).argmax(1)
            best = {
                'C': C, 'val_accuracy': val, 'w': w, 'b': b,
                'test_accuracy': accuracy(pred, y['test']),
                'pred_test': pred,
                'features': int((np.abs(w).sum(0) > 0).sum()),
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


def summarise(rows: list[dict]) -> None:
    """Print the repricing, the intercept re-coding curve and the honest frontier."""
    # The convex refit is a fitting choice, not a stored value, so the frontier
    # may use every row; the equal-code tables stay at one step count so their
    # cells are comparable.
    steps = sorted({r['steps'] for r in rows})
    default = [r for r in rows if r['steps'] == DEFAULT_STEPS]
    arms = list(dict.fromkeys(r['arm'] for r in default))
    budgets = sorted({r['budget'] for r in default})
    base_arms = [a for a in arms if not MP_ARM.match(a)]

    print('\nWhat a deployment stores, by arm, at 256 code values:')
    print(f'{"arm":<24s} {"code":>5s} {"+intercept":>11s} {"+scaling":>9s} '
          f'{"= deployed":>11s} {"bits":>8s}  {"test":>7s}')
    for arm in arms:
        hit = [r for r in default if r['arm'] == arm and r['budget'] == 256]
        if not hit:
            continue
        r = max(hit, key=lambda r: r['val_accuracy'])
        print(f'{arm:<24s} {r["code"]:>5d} {r["intercept_values"]:>11d} '
              f'{r["scaling_values"]:>9d} {r["deployed_values"]:>11d} '
              f'{r["deployed_bits"]:>8d}  {r["test_accuracy"]:>7.4f}')

    print('\nTest accuracy at equal *code* values (the old convention):')
    print(f'{"code":>6s}  ' + '  '.join(f'{a:>24s}' for a in base_arms))
    for budget in budgets:
        cells = []
        for arm in base_arms:
            hit = [r for r in default if r['budget'] == budget and r['arm'] == arm]
            best = max(hit, key=lambda r: r['val_accuracy']) if hit else None
            cells.append(f'{best["test_accuracy"]:.4f}' if best else '-')
        print(f'{budget:>6d}  ' + '  '.join(f'{c:>24s}' for c in cells))

    print('\nDeployed values at the same code sizes:')
    print(f'{"code":>6s}  ' + '  '.join(f'{a:>24s}' for a in base_arms))
    for budget in budgets:
        cells = []
        for arm in base_arms:
            hit = [r for r in default if r['budget'] == budget and r['arm'] == arm]
            best = max(hit, key=lambda r: r['val_accuracy']) if hit else None
            cells.append(f'{best["deployed_values"]:d}' if best else '-')
        print(f'{budget:>6d}  ' + '  '.join(f'{c:>24s}' for c in cells))

    mp_rows = [(m, r) for m, r in ((MP_ARM.match(r['arm']), r) for r in default) if m]
    if mp_rows:
        print('\nRe-coding the deployed intercept as q more code entries '
              '(uncentred features, same support):')
        for stem, tag in dict.fromkeys((m['stem'], m['tag']) for m, _ in mp_rows):
            family = [r for m, r in mp_rows
                      if (m['stem'], m['tag']) == (stem, tag)]
            if not family:
                continue
            qs = sorted({int(MP_ARM.match(r['arm'])['q']) for r in family})
            print(f'  {stem} (+{tag})')
            print(f'  {"code":>6s}  {"centred":>8s}  ' +
                  '  '.join(f'{"q=" + str(q):>13s}' for q in qs))
            for budget in sorted({r['budget'] for r in family}):
                base = [r for r in default
                        if r['arm'] == stem and r['budget'] == budget]
                cells = []
                for q in qs:
                    hit = [r for r in family
                           if r['budget'] == budget
                           and r['arm'].endswith(f'+{tag}{q}')]
                    cells.append(f'{hit[0]["test_accuracy"]:.4f}'
                                 f'/{hit[0]["deployed_values"]:d}' if hit else '-')
                head = (f'{base[0]["test_accuracy"]:.4f}/{base[0]["deployed_values"]:d}'
                        if base else '-')
                print(f'  {budget:>6d}  {head:>8s}  ' +
                      '  '.join(f'{c:>13s}' for c in cells))

    if len(steps) > 1:
        print('\nIs the uncentred refit converged?  Test accuracy by LBFGS steps:')
        pairs = sorted({(r['budget'], r['arm']) for r in rows
                        if MP_ARM.match(r['arm']) and r['steps'] != DEFAULT_STEPS})
        print(f'  {"code":>6s}  {"arm":<30s}  ' + '  '.join(f'{s:>7d}' for s in steps))
        for budget, arm in pairs:
            cells = []
            for step in steps:
                hit = [r for r in rows if r['budget'] == budget
                       and r['arm'] == arm and r['steps'] == step]
                cells.append(f'{hit[0]["test_accuracy"]:.4f}' if hit else '-')
            print(f'  {budget:>6d}  {arm:<30s}  ' + '  '.join(f'{c:>7s}' for c in cells))

    print('\nHonest frontier: best validation accuracy per deployed-value ceiling.')
    ceilings = (128, 160, 192, 224, 256, 272, 288, 304, 320, 352, 384, 416, 448,
                480, 512, 640, 768, 1024)
    selected = []
    for ceiling in ceilings:
        under = [r for r in rows if r['deployed_values'] <= ceiling]
        if not under:
            continue
        best = max(under, key=lambda r: r['val_accuracy'])
        selected.append((ceiling, best))
        print(f'  <= {ceiling:>5d}  {best["arm"]:<28s} '
              f'{best["deployed_values"]:>5d} values ({best["deployed_bits"]:>6d} bits, '
              f'{best["features"]:>3d} columns, {best["steps"]:>4d} steps)  '
              f'val={best["val_accuracy"]:.4f} test={best["test_accuracy"]:.4f}')
    for target in TARGETS:
        hit = next((c for c, r in selected if r['val_accuracy'] >= target), None)
        miss = next((c for c, r in selected if r['test_accuracy'] >= target), None)
        print(f'{target:.0%}: first met on validation at <= {hit} deployed values, '
              f'on test at <= {miss}')

    print(f'\nvs {INCUMBENT} at the same code size (paired bootstrap of the test '
          'difference):')
    for budget in budgets:
        for arm in arms:
            hit = [r for r in default if r['budget'] == budget and r['arm'] == arm]
            if not hit or arm == INCUMBENT:
                continue
            best = max(hit, key=lambda r: r['val_accuracy'])
            print(f'  {budget:>5d}  {arm:<28s} {best["test_delta"]:+.4f} '
                  f'[{best["delta_lo"]:+.4f}, {best["delta_hi"]:+.4f}]')


def load_rows(path: str) -> list[dict]:
    """Re-read a finished result CSV so the summary can be recomputed cheaply."""
    with open(path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ('budget', 'code', 'intercept_values', 'scaling_values',
                    'deployed_values', 'deployed_bits', 'stored_values', 'features',
                    'class_atoms', 'feature_atoms', 'mp_atoms'):
            row[key] = int(row[key])
        row['steps'] = int(row.get('steps') or DEFAULT_STEPS)
        for key in ('val_accuracy', 'test_accuracy', 'test_delta', 'delta_lo',
                    'delta_hi'):
            row[key] = float(row[key])
    return rows


def make_row(arm, budget, scaling, feat_name, bias, support, fdict, fatoms, cdict,
             best, centred, groups, mp, steps=DEFAULT_STEPS) -> dict:
    """One result row, with the deployed-value accounting spelled out."""
    cost = sep_dict_deployed_values(support, fdict, fatoms, bias, centred, groups)
    index_bits = len(support) * (math.log2(cdict.shape[1]) + math.log2(fatoms))
    index_bits += cost['scale_ids'] * math.log2(max(1, cost['scale_groups']))
    return {
        'budget': budget, 'arm': arm, 'scaling': scaling, 'features_dict': feat_name,
        'bias': bias, 'mp_atoms': mp, 'steps': steps,
        'code': cost['code'], 'intercept_values': cost['intercept'],
        'scaling_values': cost['scaling'], 'deployed_values': cost['total'],
        'deployed_bits': int(round(32 * cost['total'] + index_bits)),
        'stored_values': cost['code'] + (ROWS if bias == 'free' else 0),
        'features': best['features'],
        'class_atoms': cdict.shape[1], 'feature_atoms': fatoms,
        'structure': f'C={best["C"]}',
        'val_accuracy': round(best['val_accuracy'], 4),
        'test_accuracy': round(best['test_accuracy'], 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summarise', action='store_true',
                        help='re-print the summary from the existing result CSV')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--budgets', type=int, nargs='*', default=None)
    parser.add_argument('--mp-arms', nargs='*', default=list(MP_ARMS),
                        help='arms whose support is re-used for the intercept '
                             're-coding (the uncentred refit needs many more '
                             'LBFGS steps than the centred one, so --steps 4000 '
                             'is what makes this comparison fair)')
    parser.add_argument('--mp-grid', type=int, nargs='*', default=list(MP_GRID),
                        help='values of q for the intercept re-coding')
    parser.add_argument('--unpenalised-bias', action='store_true',
                        help='also emit +mpu rows: the same intercept re-coding '
                             'with the constant column left out of the L2 penalty')
    parser.add_argument('--arms', nargs='*', default=None,
                        help='restrict to these arm names (the rest of the cross '
                             'is unchanged, so two passes can be merged)')
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--merge', nargs='*', default=None,
                        help='merge these result CSVs into --out and summarise; '
                             'rows are keyed by (budget, arm) and the fitters are '
                             'deterministic, so duplicates must agree')
    args = parser.parse_args()
    if args.merge:
        merged: dict[tuple[int, str], dict] = {}
        for path in args.merge:
            for row in load_rows(path):
                key = (row['budget'], row['arm'], row['steps'])
                if key in merged and merged[key]['test_accuracy'] != row['test_accuracy']:
                    raise ValueError(f'{key} disagrees between result files')
                merged.setdefault(key, row)
        rows = [merged[key] for key in sorted(merged)]
        write_rows(rows, args.out)
        print(f'{len(rows)} rows merged into {args.out}')
        summarise(rows)
        return
    if args.summarise:
        summarise(load_rows(args.out))
        return

    base, y, base_names = load_pools(BASE_POOLS)
    layout, _, layout_names = load_pools((LAYOUT_POOL,))
    merged = {s: np.concatenate([base[s], layout[s]], axis=1) for s in base}
    base_width = base['train'].shape[1]
    names = base_names + layout_names
    size, quota = CANDIDATE_LIST
    base_order = group_lasso_rank(base['train'], y['train'], lam=LAM, epochs=1500)[0]
    layout_order = group_lasso_rank(layout['train'], y['train'], lam=LAM, epochs=1500)[0]
    idx = np.concatenate((base_order[:size - quota],
                          layout_order[:quota] + base_width)).astype(np.int64)
    xtr = merged['train'][:, idx]
    inner_order = group_lasso_rank(xtr, y['train'], lam=LAM, epochs=1500)[0]
    families = np.array([family_of(names[j], j >= base_width) for j in idx])
    groups = np.unique(families, return_inverse=True)[1]
    print(f'{len(idx)} candidate columns over {groups.max() + 1} families', flush=True)

    moments = build_moments(xtr, groups)
    cdict = gaussian_atoms(16384)
    feature_dicts = {
        'identity': np.eye(len(idx)),
        'pairs256': pair_atoms(len(idx), PAIR_TOP, inner_order),
        'pairsbuck': group_pair_atoms(len(idx), PAIR_TOP, inner_order,
                                      moments['zbuck'][3]),
        'pairsfam': group_pair_atoms(len(idx), PAIR_TOP, inner_order, groups),
    }
    for name, dictionary in feature_dicts.items():
        print(f'{name}: {dictionary.shape[1]} atoms', flush=True)

    epochs, updates, drop = RIGL_SETTING
    rows: list[dict[str, object]] = []
    for budget in (args.budgets or BUDGETS):
        preds: dict[str, np.ndarray] = {}
        for arm, scaling, feat_name, bias, every_budget in ARMS:
            if not every_budget and budget not in ANCHORS:
                continue
            if args.arms is not None and arm not in args.arms:
                continue
            mu, sigma, centred, scale_groups = moments[scaling]
            fdict = feature_dicts[feat_name]
            _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
                xtr, y['train'], cdict, fdict, nonzeros=budget, epochs=epochs,
                updates=updates, drop_fraction=drop, bias=bias, moments=(mu, sigma))
            fatoms = fdict.shape[1] + (1 if bias == 'coded' else 0)
            best = sweep_c(merged, y, idx, cdict, fdict, support, args.steps, bias,
                           (mu, sigma))
            rows.append(make_row(arm, budget, scaling, feat_name, bias, support, fdict,
                                 fatoms, cdict, best, centred, scale_groups, 0,
                                 args.steps))
            preds[arm] = best['pred_test']

            if arm in args.mp_arms:
                # The deployed intercept b_eff is 44 numbers the code does not
                # carry.  Re-code it as q more entries on a constant column and
                # refit convexly on *uncentred* features, where b_eff is the only
                # intercept there is.
                # q = 44 spans the whole intercept space, so it separates "the
                # 44 numbers cannot be compressed" from "the uncentred refit is
                # what costs the accuracy".
                atoms = matching_pursuit(best['b'][1:], cdict, max(args.mp_grid))
                zero = np.zeros_like(mu)
                variants = ((True, 'mp'), (False, 'mpu')) if args.unpenalised_bias \
                    else ((True, 'mp'),)
                for q in args.mp_grid:
                    grown = coded_support(support, fdict.shape[1], atoms[:q])
                    for penalise, tag in variants:
                        mp_best = sweep_c(merged, y, idx, cdict, fdict, grown,
                                          args.steps, 'coded', (zero, sigma),
                                          penalise_bias=penalise)
                        mp_arm = f'{arm}+{tag}{q}'
                        rows.append(make_row(mp_arm, budget, scaling, feat_name,
                                             'coded', grown, fdict,
                                             fdict.shape[1] + 1, cdict, mp_best,
                                             False, scale_groups, q, args.steps))
                        preds[mp_arm] = mp_best['pred_test']

        reference = preds.get(INCUMBENT)
        for row in rows:
            if row['budget'] != budget or 'test_delta' in row:
                continue
            if reference is None or row['arm'] == INCUMBENT:
                row.update(test_delta=0.0, delta_lo=0.0, delta_hi=0.0)
                continue
            lo, hi = paired_bootstrap(preds[row['arm']], reference, y['test'])
            row.update(test_delta=round(row['test_accuracy']
                                        - accuracy(reference, y['test']), 4),
                       delta_lo=round(lo, 4), delta_hi=round(hi, 4))
        for row in rows:
            if row['budget'] == budget:
                print(' '.join(f'{k}={v}' for k, v in row.items()), flush=True)
        write_rows(rows, args.out)
    summarise(rows)


if __name__ == '__main__':
    main()
