#!/usr/bin/env python
"""Honest multi-seed train-CV of the masked-multinomial IMP pipeline.

Preliminary single-split runs (experiments/masked_imp.py) suggest element-wise
IMP at low weight-decay reaches val/test >= 0.940 down to ~200-210 params, below
the 260-param dense-backward frontier.  This measures the honest floor: the
WHOLE IMP prune-retrain pipeline is run inside every CV fold (standardise on
fold-train, prune by magnitude, retrain survivors, score the held-out fold at
each sparsity target).  The floor is the smallest target whose mean multi-seed
held-out CV clears 0.940 -- an estimate IMP never optimised (it prunes by
magnitude, not by held-out accuracy).

A full-train IMP additionally reports the true held-out val and test curves.

Env: WD (weight decay), SEEDS (comma list, default 0,1,2,3,4), DEVICE index,
MODE=cv|full (default both).
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

BASE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')

TARGETS = [300, 280, 260, 250, 240, 230, 220, 210, 200, 195, 190, 185, 180,
           175, 170, 160, 150]


def load(split):
    return (np.load(os.path.join(BASE, f'{split}_feat_o6.npy')),
            np.load(os.path.join(BASE, f'{split}_y.npy')))


def imp_curve(Xtr, ytr, Xev, yev, wd, dev, max_iter=200):
    """Fit full multinomial on standardised Xtr, then IMP prune+retrain.

    Returns dict target -> held-out accuracy on (Xev, yev) at that nnz target.
    """
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=dev)
    Zev = torch.tensor((Xev - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr, dtype=torch.long, device=dev)
    K, F = int(ytr.max() + 1), Ztr.shape[1]
    W = torch.zeros(K, F, device=dev, requires_grad=True)
    b = torch.zeros(K, device=dev, requires_grad=True)
    mask = torch.ones(K, F, device=dev)

    def train():
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter,
                                tolerance_grad=1e-7, tolerance_change=1e-9,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            logits = Ztr @ (W * mask).t() + b
            loss = torch.nn.functional.cross_entropy(logits, yt) \
                + wd * ((W * mask) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            W.mul_(mask)

    @torch.no_grad()
    def acc():
        return float(((Zev @ (W * mask).t() + b).argmax(1).cpu().numpy() == yev).mean())

    train()
    out = {}
    for tgt in TARGETS:
        with torch.no_grad():
            flat = (W.abs() * mask).flatten()
            nz = int((flat > 0).sum())
            if tgt < nz:
                thresh = torch.topk(flat, tgt, largest=True).values.min()
                mask.copy_((W.abs() >= thresh).float() * mask)
                W.mul_(mask)
        train()
        out[tgt] = (acc(), int((W != 0).sum()))
    return out


def main():
    wd = float(os.environ.get('WD', '5e-4'))
    seeds = [int(s) for s in os.environ.get('SEEDS', '0,1,2,3,4').split(',')]
    dev = f"cuda:{os.environ.get('DEVICE', '0')}" if torch.cuda.is_available() else 'cpu'
    mode = os.environ.get('MODE', 'both')

    Xtr, ytr = load('train')
    print(f'WD={wd} seeds={seeds} dev={dev}', flush=True)

    if mode in ('both', 'cv'):
        acc_by_tgt = {t: [] for t in TARGETS}
        for s in seeds:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=s)
            for tri, vai in skf.split(Xtr, ytr):
                res = imp_curve(Xtr[tri], ytr[tri], Xtr[vai], ytr[vai], wd, dev)
                for t in TARGETS:
                    acc_by_tgt[t].append(res[t][0])
            print(f'  seed {s} done', flush=True)
        print(f'\n== CV (WD={wd}, {len(seeds)} seeds x5) ==')
        print(f'{"params":>6} {"cvAcc":>7} {"se":>6} {"nfold":>5}')
        for t in TARGETS:
            a = np.array(acc_by_tgt[t])
            print(f'{t + 10:>6} {a.mean():>7.4f} {a.std() / np.sqrt(len(a)):>6.4f} '
                  f'{len(a):>5}', flush=True)

    if mode in ('both', 'full'):
        Xva, yva = load('val')
        Xte, yte = load('test')
        rv = imp_curve(Xtr, ytr, Xva, yva, wd, dev)
        rt = imp_curve(Xtr, ytr, Xte, yte, wd, dev)
        print(f'\n== full-train (WD={wd}) ==')
        print(f'{"params":>6} {"val":>7} {"test":>7} {"nnz":>5}')
        for t in TARGETS:
            print(f'{t + 10:>6} {rv[t][0]:>7.4f} {rt[t][0]:>7.4f} {rt[t][1]:>5}',
                  flush=True)
    print('CVIMP_DONE', flush=True)


if __name__ == '__main__':
    main()
