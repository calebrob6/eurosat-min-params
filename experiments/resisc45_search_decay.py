#!/usr/bin/env python
"""Tune the prune-and-regrow search's own regulariser at 2,000-3,600 values.

``resisc45_head_regime.py`` found that the 16,384-atom class-dictionary head on
the 640-column quota list stops improving above ~2,300 deployed values because
the *search* overfits its support, and that raising the search weight decay
from 1e-4 to 1e-3 is worth +1.1 to +1.4 test points.  Only that one value was
tried.  This sweeps the decay over a decade and a half, repeats settings under
"seeds" that perturb the decay by 0.2% each (the search itself is
deterministic, so this measures its chaos rather than seed noise), asks in the
``quota`` arm whether a wider candidate list pays once the search is
regularised, and in the ``union`` arm averages over the chaos instead of
selecting over it.

Arms:

* ``decay`` -- weight decay in ``--decays`` x seeds x budgets on the fixed
  384/128/128 quota list;
* ``quota`` -- alternative quota lists at the decays in ``--decays``;
* ``union`` -- the union of the supports found by ``--union-seeds`` searches,
  refitted convexly and magnitude-pruned back to the budget over
  ``--prune-rounds`` geometric steps (each step a fresh convex refit), so the
  search's chaos is averaged over rather than selected over.

Every row reports train accuracy next to validation and test, the convex
refit selects ``C`` on validation, and deployed values are the code plus the
44 intercepts (identity column atoms absorb the standardiser).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_expanded_head import QUOTA, gaussian_atoms
from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import (
    NUM_CLASSES,
    _to_device,
    accuracy,
    fit_rigl_sep_dict_ref_logreg_gpu,
    group_lasso_rank,
    refit_masked_sep_dict_ref_logreg_gpu,
    standardise,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_search_decay_result.csv')
ROWS = NUM_CLASSES - 1
C_GRID = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RIGL_SETTING = (4000, 100, 0.5)
BUDGETS = (2048, 3072, 3584)
DECAYS = (1e-3, 3e-3, 1e-2, 3e-2)
SEEDS = (0, 1, 2)
ATOMS = 16384
QUOTAS = {
    'q384/128/128': (384, 128, 128),
    'q512/128/128': (512, 128, 128),
    'q384/128/256': (384, 128, 256),
    'q384/256/128': (384, 256, 128),
    'q512/192/192': (512, 192, 192),
    'q256/128/128': (256, 128, 128),
}


def write_rows(rows: list[dict], path: str) -> None:
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(x, y, w, b) -> dict:
    return {f'{s}_accuracy': round(accuracy((x[s] @ w.T + b).argmax(1), y[s]), 4)
            for s in ('train', 'val', 'test')}


def search(x, y, cdict, fdict, budget, decay, seed, drop, epochs=None, updates=None):
    e, u, _ = RIGL_SETTING
    epochs = epochs or e
    updates = updates or u
    # The search is deterministic (top-k start, no sampling), so a seed only
    # changes the support through a tiny perturbation of the decay: 0.2% per
    # seed, far below the decade the sweep spans, which makes repeated seeds a
    # measurement of the search's own chaos rather than of the decay.
    _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
        x['train'], y['train'], cdict, fdict, nonzeros=budget, epochs=epochs,
        updates=updates, drop_fraction=drop, bias='free',
        weight_decay=decay * (1.0 + 0.002 * seed), seed=seed)
    return support


def select_c(x, y, cdict, fdict, support, steps):
    best = None
    for C in C_GRID:
        w, b = refit_masked_sep_dict_ref_logreg_gpu(
            x['train'], y['train'], cdict, fdict, support, C=C, steps=steps, bias='free')
        got = score(x, y, w, b)
        if best is None or got['val_accuracy'] > best['val_accuracy']:
            best = dict(got, C=C)
    return best


def run_head(x, y, cdict, budget, decay, seed, steps, drop=0.5):
    fdict = np.eye(x['train'].shape[1])
    support = search(x, y, cdict, fdict, budget, decay, seed, drop)
    return select_c(x, y, cdict, fdict, support, steps)


def refit_values(x, y, cdict, fdict, support, C, steps, device='cuda'):
    """Convex refit of a separable-code support that returns the code values.

    Same objective as ``refit_masked_sep_dict_ref_logreg_gpu`` with a free
    intercept; the values are what magnitude pruning needs.
    """
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    dc = _to_device(cdict, device)
    df = _to_device(fdict, device)
    n = xs.shape[0]
    fatoms = df.shape[1]
    index = torch.as_tensor(support, device=device, dtype=torch.long)
    dc_sel, df_sel = dc[:, index // fatoms], df[:, index % fatoms]
    value = torch.zeros(len(index), device=device, requires_grad=True)
    b = torch.zeros(ROWS, device=device, requires_grad=True)
    l2 = 1.0 / (2.0 * C * n)
    opt = torch.optim.LBFGS([value, b], max_iter=steps, history_size=20, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        w = (dc_sel * value) @ df_sel.T
        logits = torch.cat((torch.zeros(n, 1, device=device), xs @ w.T + b), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * (value * value).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return value.detach().abs().cpu().numpy()


def run_union(x, y, cdict, budget, decay, seeds, steps, drop, rounds, prune_c, scale=1.0):
    fdict = np.eye(x['train'].shape[1])
    supports = [search(x, y, cdict, fdict, int(budget * scale), decay, s, drop) for s in seeds]
    support = np.unique(np.concatenate(supports))
    sizes = [int(round(budget * (len(support) / budget) ** (1 - r / rounds)))
             for r in range(1, rounds + 1)]
    print(f'union of {len(seeds)} supports: {len(support)} entries -> {sizes}', flush=True)
    for size in sizes:
        magnitude = refit_values(x['train'], y['train'], cdict, fdict, support, prune_c, steps)
        support = support[np.argsort(-magnitude)[:size]]
    return dict(select_c(x, y, cdict, fdict, support, steps), union=len(np.unique(
        np.concatenate(supports))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('decay', 'quota', 'union'), default='decay')
    parser.add_argument('--union-seeds', type=int, nargs='*', default=[2, 4, 8])
    parser.add_argument('--prune-rounds', type=int, default=3)
    parser.add_argument('--prune-c', type=float, default=0.1)
    parser.add_argument('--search-scale', type=float, default=1.0,
                        help='search at this multiple of the budget before pruning; with one '
                             'seed this is the control that prunes a single over-provisioned '
                             'search instead of a union')
    parser.add_argument('--seed-offset', type=int, default=0,
                        help='first search seed of a union, to draw disjoint search sets')
    parser.add_argument('--budgets', type=int, nargs='*', default=list(BUDGETS))
    parser.add_argument('--decays', type=float, nargs='*', default=list(DECAYS))
    parser.add_argument('--seeds', type=int, nargs='*', default=list(SEEDS))
    parser.add_argument('--quotas', nargs='*', default=list(QUOTAS))
    parser.add_argument('--drops', type=float, nargs='*', default=[0.5])
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()

    base, y, _ = load_pools(BASE_POOLS)
    layout, _, _ = load_pools((LAYOUT_POOL,))
    new, _, _ = load_pools(('gpu4_pool',))
    blocks = (base, layout, new)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in blocks]

    def build(quota):
        return {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, orders, quota)],
                                  axis=1) for s in base}

    cdict = gaussian_atoms(ATOMS)
    rows: list[dict] = []
    if args.arm == 'decay':
        settings = [(QUOTA, 'q384/128/128', d, drop, s, b)
                    for b in args.budgets for d in args.decays for drop in args.drops
                    for s in args.seeds]
    elif args.arm == 'union':
        settings = [(QUOTA, 'q384/128/128', d, drop, s, b)
                    for b in args.budgets for d in args.decays for drop in args.drops
                    for s in args.union_seeds]
    else:
        settings = [(QUOTAS[q], q, d, drop, s, b)
                    for b in args.budgets for q in args.quotas for d in args.decays
                    for drop in args.drops for s in args.seeds]
    cache = {}
    for quota, qname, decay, drop, seed, budget in settings:
        if qname not in cache:
            cache[qname] = build(quota)
        x = cache[qname]
        t0 = time.time()
        if args.arm == 'union':
            seeds = list(range(args.seed_offset, args.seed_offset + seed))
            best = run_union(x, y, cdict, budget, decay, seeds, args.steps, drop,
                             args.prune_rounds, args.prune_c, args.search_scale)
        else:
            best = run_head(x, y, cdict, budget, decay, seed, args.steps, drop=drop)
        rows.append({'arm': args.arm, 'list': qname, 'columns': x['train'].shape[1],
                     'budget': budget, 'deployed_values': budget + ROWS, 'decay': decay,
                     'drop': drop, 'seed': seed, 'seed_offset': args.seed_offset,
                     'rounds': args.prune_rounds, 'prune_c': args.prune_c,
                     'search_scale': args.search_scale,
                     'union': best.get('union', 0), 'C': best['C'],
                     **{key: best[key] for key in ('train_accuracy', 'val_accuracy',
                                                   'test_accuracy')},
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)


if __name__ == '__main__':
    main()
