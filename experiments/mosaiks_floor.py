#!/usr/bin/env python
"""Parameter floor of RANDOM MOSAIKS + linear head: how few features clear 94%?

Random MOSAIKS features are seed-free (~0 extractor params), so the deployed
parameter count is just the linear head: 10*(k+1) for k KEPT features. The ceiling
sweep showed 4096 features -> test 0.958 but 40970 head params. Here we L1-rank the
2K features and read the accuracy-vs-k curve, then a backward-greedy tighten near
the frontier's k=18 (190 params), to locate where MOSAIKS crosses 94% vs the
hand-crafted linear frontier (k=18, 190 params, test 0.9433).
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import load_cached  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=1024)
    ap.add_argument('--patch', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import torch.nn.functional as Fn
    from sklearn.metrics import accuracy_score

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    g = torch.Generator().manual_seed(args.seed)

    def load(s):
        x, y = load_cached(s); return x.astype(np.float32), y
    xtr, ytr = load('train'); xva, yva = load('val'); xte, yte = load('test')
    C = xtr.shape[1]; p = args.patch
    mu = xtr.mean((0, 2, 3), keepdims=True); sd = xtr.std((0, 2, 3), keepdims=True) + 1e-6
    filt = torch.randn(args.K, C, p, p, generator=g)
    filt = filt - filt.mean((1, 2, 3), keepdim=True)
    filt = (filt / (filt.flatten(1).norm(dim=1)[:, None, None, None] + 1e-6)).to(dev)

    def feat(x):
        X = torch.tensor((x - mu) / sd); outs = []
        for i in range(0, X.shape[0], 128):
            c = Fn.conv2d(X[i:i+128].to(dev), filt)
            outs.append(torch.cat([Fn.relu(c).mean((2, 3)),
                                   Fn.relu(-c).mean((2, 3))], 1).cpu())
        return torch.cat(outs).numpy()
    Ftr, Fva, Fte = feat(xtr), feat(xva), feat(xte)
    print(f'random MOSAIKS K={args.K}: {Ftr.shape[1]} features (seed-free extractor)')

    order = l1_rank(Ftr, ytr)
    print(f'\n{"k":>4} {"params":>6} {"verCV":>7} {"val":>7} {"test":>7}')
    for k in [12, 18, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, Ftr.shape[1]]:
        if k > Ftr.shape[1]:
            continue
        idx = order[:k]
        w, b, fi = fit_folded_logreg(Ftr, ytr, idx, C=1.0)
        ver = mean_cv(Ftr, ytr, idx, range(10, 20), C=1.0)
        va = accuracy_score(yva, predict(Fva, w, b, fi))
        te = accuracy_score(yte, predict(Fte, w, b, fi))
        flag = '  >=95' if te >= .95 else ('  >=94' if te >= .94 else '')
        print(f'{k:>4} {10*(k+1):>6} {ver:>7.4f} {va:>7.4f} {te:>7.4f}{flag}', flush=True)

    # backward-greedy tighten from top-64 -> 14 (does selection beat L1 top-k here?)
    _, _, subs = backward_eliminate(Ftr, ytr, order[:64], 14, select_seeds=range(6),
                                    C=1.0, workers=8, record=True)
    print(f'\nbackward-greedy from top-64:')
    for k in sorted(subs, reverse=True):
        if k > 40:
            continue
        s = subs[k]
        w, b, fi = fit_folded_logreg(Ftr, ytr, s, C=1.0)
        ver = mean_cv(Ftr, ytr, s, range(10, 20), C=1.0)
        va = accuracy_score(yva, predict(Fva, w, b, fi))
        te = accuracy_score(yte, predict(Fte, w, b, fi))
        flag = '  >=94' if te >= .94 and va >= .94 and ver >= .94 else ''
        print(f'{k:>4} {10*(k+1):>6} {ver:>7.4f} {va:>7.4f} {te:>7.4f}{flag}', flush=True)


if __name__ == '__main__':
    main()
