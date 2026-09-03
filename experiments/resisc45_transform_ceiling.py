#!/usr/bin/env python
"""What a fixed per-column monotone transform is worth at the top.

The dictionary head acts on standardised columns, and the standardiser folds
into the sparse code at no deployed cost (``W_eff = Dc P / sigma``,
``b_eff = b - W_eff mu``).  A fixed *scale-free* function of the raw column
applied before standardisation -- ``sign(x) |x|^p`` -- costs nothing either:
no per-column constant sits inside the nonlinearity.  The pool columns are
skewed (median absolute skewness about 1 on the averaged pools, a fifth above
2), and iteration 1 found that a hinge at each column's mean recovers half of
the MLP headroom, so a concave bend per column may buy part of that headroom
for free where hinge columns could not be bought under the budget.

Arms, all dense multinomial heads on the dihedral-averaged 384/128/128 + 128
list with ``C`` on validation:

* ``raw`` -- the frontier list as it is;
* ``sqrt``, ``cbrt``, ``quart`` -- ``sign(x) |x|^p`` for p = 1/2, 1/3, 1/4 on
  every column (zero deployed cost, same columns);
* ``pick-skew`` -- per column the power in {1, 1/2, 1/3, 1/4} with the
  smallest absolute training skewness (a design choice per column, no
  deployed value);
* ``log1p-median`` -- ``sign(x) log1p(|x| / median|x|)``, one constant per
  column (a diagnostic, priced at one value per column);
* ``rankgauss`` -- the train quantile map to a normal, the upper bound on any
  monotone per-column transform (a diagnostic, not deployable);
* ``raw+sqrt`` -- both copies, to read the collective two-piece value;
* ``sqrt-reranked`` -- the list re-ranked on the transformed pools.

The 30 most confused test pairs are written for every arm.  ``--probe``
re-reads iteration 1's nonlinear headroom (hinge at the mean, three knots,
4,096 pair-ReLUs, MLPs of width 64-1024) on the same averaged list, so the
size of the MLP gap after dihedral averaging is on record.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy.stats import norm, skew

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import class_names
from resisc45_layout_gain import LAM, load_pools
from resisc45_lib import accuracy, fit_logreg_gpu, group_lasso_rank
from resisc45_nonlinear_probe import dense_arm, hinge, pair_relu
from resisc45_pool5_ceiling import pair_counts, write_rows
from resisc45_lib import fit_mlp_gpu, predict_mlp

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_transform_ceiling_result.csv')
PROBE_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_transform_ceiling_probe.csv')
MLP_WIDTHS = (64, 128, 512, 1024)
MLP_DECAY = (1e-4, 1e-3)
PAIRS_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_transform_ceiling_pairs.csv')
BLOCKS = (('gpud8_pool', 'gpu2d8_pool', 'rgb_pool'), ('gpu3d8_pool',), ('gpu4d8_pool',),
          ('gpu5d8_pool',))
QUOTA = (384, 128, 128, 128)
DENSE_C = (0.003, 0.01, 0.03, 0.1, 0.3)
POWERS = {'sqrt': 0.5, 'cbrt': 1.0 / 3.0, 'quart': 0.25}


def power(x: np.ndarray, p: float) -> np.ndarray:
    return (np.sign(x) * np.abs(x) ** p).astype(np.float32)


def apply_power(blocks: dict, p: float) -> dict:
    return {s: power(v, p) for s, v in blocks.items()}


def pick_skew(blocks: dict, powers=(1.0, 0.5, 1.0 / 3.0, 0.25)) -> tuple[dict, np.ndarray]:
    """Per column the power with the smallest absolute training skewness."""
    train = blocks['train']
    table = np.stack([np.abs(skew(power(train, p), axis=0)) for p in powers])
    choice = np.array(powers)[np.nan_to_num(table, nan=np.inf).argmin(0)]
    out = {}
    for s, v in blocks.items():
        out[s] = np.stack([power(v[:, j], choice[j]) for j in range(v.shape[1])], axis=1)
    return out, choice


def log1p_median(blocks: dict) -> dict:
    scale = np.median(np.abs(blocks['train']), axis=0)
    scale = np.where(scale > 0, scale, np.abs(blocks['train']).mean(0) + 1e-6)
    return {s: (np.sign(v) * np.log1p(np.abs(v) / scale)).astype(np.float32)
            for s, v in blocks.items()}


def rankgauss(blocks: dict) -> dict:
    """Train quantile map to a standard normal, applied by interpolation."""
    train = blocks['train']
    n, k = train.shape
    out = {s: np.empty_like(v) for s, v in blocks.items()}
    for j in range(k):
        col = train[:, j]
        u, inv = np.unique(col, return_inverse=True)
        ranks = np.empty(n)
        order = np.argsort(col, kind='stable')
        ranks[order] = np.arange(n)
        target = np.zeros(len(u))
        np.add.at(target, inv, ranks)
        target /= np.bincount(inv, minlength=len(u))
        target = norm.ppf((target + 0.5) / n)
        for s, v in blocks.items():
            out[s][:, j] = np.interp(v[:, j], u, target)
    return {s: v.astype(np.float32) for s, v in out.items()}


def dense(x, y):
    best = None
    for C in DENSE_C:
        w, b = fit_logreg_gpu(x['train'], y['train'], C=C, steps=400)
        pred = {s: (x[s] @ w.T + b).argmax(1) for s in x}
        val = accuracy(pred['val'], y['val'])
        if best is None or val > best['val_accuracy']:
            best = {'C': C, 'train_accuracy': accuracy(pred['train'], y['train']),
                    'val_accuracy': val, 'test_accuracy': accuracy(pred['test'], y['test']),
                    'pred': pred['test']}
    return best


def probe(x, y, path) -> None:
    rows: list[dict] = []
    dense_arm(x, y, 'linear/list768-d8', 'dense', rows, path)
    dense_arm(hinge(x, (0.0,)), y, 'hinge0/list768-d8', 'knots=0', rows, path)
    dense_arm(hinge(x, (-1.0, 0.0, 1.0)), y, 'hinge/list768-d8', 'knots=-1,0,1', rows, path)
    dense_arm(pair_relu(x, 4096), y, 'pairrelu4096/list768-d8', 'pairs=4096', rows, path)
    for hidden in MLP_WIDTHS:
        best, t0 = None, time.time()
        for decay in MLP_DECAY:
            w1, b1, w2, b2 = fit_mlp_gpu(x['train'], y['train'], hidden, weight_decay=decay)
            val = accuracy(predict_mlp(x['val'], w1, b1, w2, b2), y['val'])
            if best is None or val > best['val_accuracy']:
                best = {'decay': decay, 'val_accuracy': val,
                        'test_accuracy': accuracy(predict_mlp(x['test'], w1, b1, w2, b2),
                                                  y['test'])}
        rows.append({'arm': f'mlp{hidden}/list768-d8', 'structure': f'h={hidden}/wd={best["decay"]}',
                     'columns': x['train'].shape[1],
                     'val_accuracy': round(best['val_accuracy'], 4),
                     'test_accuracy': round(best['test_accuracy'], 4),
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default=RESULT_PATH)
    parser.add_argument('--probe', action='store_true',
                        help='read the nonlinear headroom on the averaged list instead')
    parser.add_argument('--probe-out', default=PROBE_PATH)
    parser.add_argument('--pairs', default=PAIRS_PATH)
    parser.add_argument('--quota', type=int, nargs=4, default=list(QUOTA))
    parser.add_argument('--skip-rerank', action='store_true')
    args = parser.parse_args()
    quota = tuple(args.quota)

    pools, y = [], None
    for names in BLOCKS:
        b, y, _ = load_pools(names)
        pools.append(b)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in pools]

    def build(blocks, ords=orders):
        return {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, ords, quota)],
                                  axis=1) for s in y}

    def cat(a, b):
        return {s: np.concatenate([a[s], b[s]], axis=1) for s in a}

    raw = build(pools)
    if args.probe:
        probe(raw, y, args.probe_out)
        return
    arms = [('raw', raw)]
    for name, p in POWERS.items():
        arms.append((name, apply_power(raw, p)))
    picked, choice = pick_skew(raw)
    print('pick-skew powers:', {p: int((choice == p).sum()) for p in np.unique(choice)},
          flush=True)
    arms += [('pick-skew', picked),
             ('log1p-median', log1p_median(raw)),
             ('rankgauss', rankgauss(raw)),
             ('raw+sqrt', cat(raw, apply_power(raw, 0.5)))]
    if not args.skip_rerank:
        sq = [apply_power(b, 0.5) for b in pools]
        orders_sq = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in sq]
        for o, osq, q in zip(orders, orders_sq, quota):
            print(f'{len(set(o[:q]) & set(osq[:q]))}/{q} top columns shared', flush=True)
        arms.append(('sqrt-reranked', build(sq, orders_sq)))

    rows, preds = [], {}
    for arm, x in arms:
        t0 = time.time()
        best = dense(x, y)
        preds[arm] = best.pop('pred')
        rows.append({'arm': arm, 'columns': x['train'].shape[1],
                     **{k: round(v, 4) if isinstance(v, float) else v for k, v in best.items()},
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)

    names = class_names()
    counts = {arm: pair_counts(preds[arm], y['test']) for arm in preds}
    top_pairs = [p for p, _ in counts['raw'].most_common(30)]
    pair_rows = [{'pair': f'{names[p[0]]} / {names[p[1]]}',
                  **{arm: counts[arm][p] for arm in preds}} for p in top_pairs]
    write_rows(pair_rows, args.pairs)
    for row in pair_rows[:12]:
        print(row, flush=True)


if __name__ == '__main__':
    main()
