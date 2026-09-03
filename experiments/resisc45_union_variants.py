#!/usr/bin/env python
"""Variants of the union-of-supports head, read with repeats.

``resisc45_search_decay.py --arm union`` cleared 80% at 3,628 deployed values
by taking the union of eight decay-perturbed prune-and-regrow supports and
magnitude-pruning it back with convex refits.  Its remaining knobs were all
read against single searches, which that script showed are chaotic (0.64
points of test spread), so they are re-read here under the union itself, with
two disjoint search sets per setting.

Arms:

* ``quota`` -- the six quota candidate lists under the union head (the
  single-search comparison in ``--arm quota`` was inside the noise);
* ``subsample`` -- diversity from the data instead of the decay: each search
  sees a different random ``--fractions`` subsample of the training set, the
  union is refitted and pruned on the full set;
* ``mixed`` -- diversity from the search settings: the eight searches cycle
  over drop fractions and decays instead of a 0.2% decay perturbation;
* ``union2`` -- a union of ``--groups`` union heads (each already pruned to the
  budget), pruned back to the budget once more.

Every row reports train accuracy next to validation and test, the convex
refit selects ``C`` on validation, and deployed values are the code plus the
44 intercepts.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_expanded_head import gaussian_atoms
from resisc45_layout_gain import BASE_POOLS, LAM, LAYOUT_POOL, load_pools
from resisc45_lib import fit_rigl_sep_dict_ref_logreg_gpu, group_lasso_rank
from resisc45_search_decay import (
    ATOMS,
    QUOTAS,
    RIGL_SETTING,
    ROWS,
    refit_values,
    select_c,
    write_rows,
)

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_union_variants_result.csv')
DECAY = 1.5e-3
DROP = 0.3
SEARCHES = 8
OFFSETS = (0, 16)
MIXED_DROPS = (0.2, 0.3, 0.4, 0.5)
MIXED_DECAYS = (1e-3, 1.5e-3)


def one_search(x_train, y_train, cdict, fdict, budget, decay, drop, seed):
    e, u, _ = RIGL_SETTING
    _, _, support = fit_rigl_sep_dict_ref_logreg_gpu(
        x_train, y_train, cdict, fdict, nonzeros=budget, epochs=e, updates=u,
        drop_fraction=drop, bias='free', weight_decay=decay * (1.0 + 0.002 * seed), seed=seed)
    return support


def prune(x, y, cdict, fdict, support, budget, rounds, prune_c, steps):
    sizes = [int(round(budget * (len(support) / budget) ** (1 - r / rounds)))
             for r in range(1, rounds + 1)]
    print(f'union: {len(support)} entries -> {sizes}', flush=True)
    for size in sizes:
        magnitude = refit_values(x['train'], y['train'], cdict, fdict, support, prune_c, steps)
        support = support[np.argsort(-magnitude)[:size]]
    return support


def union_support(x, y, cdict, fdict, budget, seeds, arm, fraction=1.0):
    supports = []
    for i, seed in enumerate(seeds):
        xt, yt = x['train'], y['train']
        decay, drop = DECAY, DROP
        if arm == 'subsample':
            rng = np.random.default_rng(1000 + seed)
            keep = rng.random(len(yt)) < fraction
            xt, yt = xt[keep], yt[keep]
            seed = 0  # the data, not the decay, supplies the diversity
        elif arm == 'mixed':
            drop = MIXED_DROPS[i % len(MIXED_DROPS)]
            decay = MIXED_DECAYS[(i // len(MIXED_DROPS)) % len(MIXED_DECAYS)]
            seed = seeds[0]  # the offset perturbs the decay so repeats differ
        supports.append(one_search(xt, yt, cdict, fdict, budget, decay, drop, seed))
    return np.unique(np.concatenate(supports))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=('quota', 'subsample', 'mixed', 'union2'),
                        default='quota')
    parser.add_argument('--budgets', type=int, nargs='*', default=[3584])
    parser.add_argument('--quotas', nargs='*', default=list(QUOTAS))
    parser.add_argument('--fractions', type=float, nargs='*', default=[0.5, 0.7, 0.85])
    parser.add_argument('--searches', type=int, default=SEARCHES)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--offsets', type=int, nargs='*', default=list(OFFSETS))
    parser.add_argument('--prune-rounds', type=int, default=3)
    parser.add_argument('--prune-c', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--c-grid', type=float, nargs='*', default=None,
                        help='refit C grid for the final validation pick (the union rows all '
                             'chose 0.03, the edge of the default grid)')
    parser.add_argument('--extra-pool', default='gpu5_pool',
                        help='fourth quota block; a quota name with four parts '
                             '(q384/128/128/64) admits its top columns')
    parser.add_argument('--extra-pool2', default='gpu6_pool',
                        help='fifth quota block; a quota name with five parts '
                             '(q384/128/128/96/64) admits its top columns')
    parser.add_argument('--d8', action='store_true',
                        help='read the dihedral-averaged pools (resisc45_dihedral_pools.py) '
                             'in place of every GPU pool')
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()
    if args.c_grid:
        import resisc45_search_decay
        resisc45_search_decay.C_GRID = tuple(args.c_grid)

    def pool_name(name):
        return name.replace('_pool', 'd8_pool') if args.d8 and name != 'rgb_pool' else name

    base, y, _ = load_pools(tuple(pool_name(p) for p in BASE_POOLS))
    layout, _, _ = load_pools((pool_name(LAYOUT_POOL),))
    new, _, _ = load_pools((pool_name('gpu4_pool'),))
    extra, _, _ = load_pools((pool_name(args.extra_pool),))
    extra2, _, _ = load_pools((args.extra_pool2,))
    blocks = (base, layout, new, extra, extra2)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in blocks]

    def build(quota):
        return {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(blocks, orders, quota)],
                                  axis=1) for s in base}

    cdict = gaussian_atoms(ATOMS)
    if args.arm == 'quota':
        settings = [(q, 1.0, b, o) for b in args.budgets for q in args.quotas
                    for o in args.offsets]
    elif args.arm == 'subsample':
        settings = [('q384/128/128', f, b, o) for b in args.budgets for f in args.fractions
                    for o in args.offsets]
    else:
        settings = [('q384/128/128', 1.0, b, o) for b in args.budgets for o in args.offsets]
    cache, rows = {}, []
    for qname, fraction, budget, offset in settings:
        if qname not in cache:
            quota = QUOTAS.get(qname) or tuple(int(q) for q in qname.lstrip('q').split('/'))
            cache[qname] = build(quota)
        x = cache[qname]
        fdict = np.eye(x['train'].shape[1])
        t0 = time.time()
        if args.arm == 'union2':
            pruned = []
            for g in range(args.groups):
                seeds = list(range(offset + g * args.searches, offset + (g + 1) * args.searches))
                sup = union_support(x, y, cdict, fdict, budget, seeds, 'quota')
                pruned.append(prune(x, y, cdict, fdict, sup, budget, args.prune_rounds,
                                    args.prune_c, args.steps))
            support = np.unique(np.concatenate(pruned))
        else:
            seeds = list(range(offset, offset + args.searches))
            support = union_support(x, y, cdict, fdict, budget, seeds, args.arm, fraction)
        union = len(support)
        support = prune(x, y, cdict, fdict, support, budget, args.prune_rounds, args.prune_c,
                        args.steps)
        best = select_c(x, y, cdict, fdict, support, args.steps)
        rows.append({'arm': args.arm, 'list': qname, 'prune_c': args.prune_c, 'columns': x['train'].shape[1],
                     'budget': budget, 'deployed_values': budget + ROWS, 'fraction': fraction,
                     'searches': args.searches, 'groups': args.groups if args.arm == 'union2' else 1,
                     'seed_offset': offset, 'union': union, 'C': best['C'],
                     **{key: best[key] for key in ('train_accuracy', 'val_accuracy',
                                                   'test_accuracy')},
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)


if __name__ == '__main__':
    main()
