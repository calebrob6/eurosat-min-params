#!/usr/bin/env python
"""MOSAIKS (random convolutional features) on EuroSAT, empirical & random filters.

Rolf et al. 2021: convolve the image with K random p x p x C filters, apply ReLU
(here both +/- branches -> 2K features), global-average-pool -> a K-of-2K feature
vector, then a simple head. The filters are NOT learned.

  * mode=random    : filters ~ N(0,1), SEEDED -> regenerable from one int, so the
                     feature extractor stores ~0 parameters (a seed-hack the brief
                     explicitly allows), like our other parameter-free features.
  * mode=empirical : filters ARE real p x p patches sampled from TRAIN images
                     (canonical MOSAIKS). These cannot be regenerated from a seed,
                     so an honest deploy must STORE K*C*p*p filter values ->
                     that many parameters. Reported, but rarely frontier-viable.

Heads: linear (sklearn logreg, params 10*(F+1)) and a tiny MLP (F->H->10, params
F*H+H + H*10+10). Input band standardisation folds into the (fixed) conv, 0 params.
Selection stays honest: choose by val, report test once. One config per GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import load_cached  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['random', 'empirical'], default='random')
    ap.add_argument('--K', type=int, default=512, help='number of filters (features=2K)')
    ap.add_argument('--patch', type=int, default=3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--img-bs', type=int, default=128)
    ap.add_argument('--hidden', type=int, nargs='+', default=[16, 32])
    ap.add_argument('--mlp-epochs', type=int, default=150)
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(args.seed)

    def load(split):
        x, y = load_cached(split)
        return x.astype(np.float32), y
    xtr, ytr = load('train'); xva, yva = load('val'); xte, yte = load('test')
    C = xtr.shape[1]
    mu = xtr.mean((0, 2, 3), keepdims=True); sd = xtr.std((0, 2, 3), keepdims=True) + 1e-6
    Xtr = torch.tensor((xtr - mu) / sd); Xva = torch.tensor((xva - mu) / sd)
    Xte = torch.tensor((xte - mu) / sd)

    # ---- build the (fixed) filter bank -------------------------------------
    p = args.patch
    if args.mode == 'random':
        filt = torch.randn(args.K, C, p, p, generator=g)
    else:
        ii = torch.randint(0, Xtr.shape[0], (args.K,), generator=g)
        yy = torch.randint(0, 64 - p + 1, (args.K,), generator=g)
        xx = torch.randint(0, 64 - p + 1, (args.K,), generator=g)
        filt = torch.stack([Xtr[ii[k], :, yy[k]:yy[k]+p, xx[k]:xx[k]+p]
                            for k in range(args.K)])
    filt = filt - filt.mean((1, 2, 3), keepdim=True)
    filt = filt / (filt.flatten(1).norm(dim=1)[:, None, None, None] + 1e-6)
    filt = filt.to(dev)

    def featurize(X):
        outs = []
        for i in range(0, X.shape[0], args.img_bs):
            xb = X[i:i + args.img_bs].to(dev)
            c = F.conv2d(xb, filt)                       # (b, K, h, w)
            fpos = F.relu(c).mean((2, 3))
            fneg = F.relu(-c).mean((2, 3))
            outs.append(torch.cat([fpos, fneg], 1).cpu())
            del c
        return torch.cat(outs).numpy()
    Ftr, Fva, Fte = featurize(Xtr), featurize(Xva), featurize(Xte)
    Fdim = Ftr.shape[1]
    ext = 'seed-only (~0 params)' if args.mode == 'random' else f'{args.K*C*p*p} stored filter params'
    print(f'[{args.mode} K={args.K} p={p}] features={Fdim}  extractor={ext}', flush=True)

    sc = StandardScaler().fit(Ftr)
    Ztr, Zva, Zte = sc.transform(Ftr), sc.transform(Fva), sc.transform(Fte)

    # ---- linear head (ceiling with ALL features) ---------------------------
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(Ztr, ytr)
    lva = accuracy_score(yva, clf.predict(Zva)); lte = accuracy_score(yte, clf.predict(Zte))
    print(f'  linear head: head_params={10*(Fdim+1)} val={lva:.4f} test={lte:.4f}'
          f'{"  >=94" if lte >= .94 else ""}{"  >=95" if lte >= .95 else ""}', flush=True)

    # ---- tiny MLP head -----------------------------------------------------
    Ztr_t = torch.tensor(Ztr, device=dev); Zva_t = torch.tensor(Zva, device=dev)
    Zte_t = torch.tensor(Zte, device=dev); ytr_t = torch.tensor(ytr, device=dev)
    yva_t = torch.tensor(yva, device=dev); yte_t = torch.tensor(yte, device=dev)
    for H in args.hidden:
        torch.manual_seed(args.seed)
        mlp = nn.Sequential(nn.Linear(Fdim, H), nn.ReLU(), nn.Linear(H, 10)).to(dev)
        hp = Fdim * H + H + H * 10 + 10
        opt = torch.optim.AdamW(mlp.parameters(), lr=3e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.mlp_epochs)
        bva, bte, bep = 0., 0., -1
        n = Ztr_t.shape[0]
        for ep in range(args.mlp_epochs):
            mlp.train(); perm = torch.randperm(n, device=dev)
            for i in range(0, n, 256):
                idx = perm[i:i+256]
                loss = F.cross_entropy(mlp(Ztr_t[idx]), ytr_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            sch.step()
            mlp.eval()
            with torch.no_grad():
                va = (mlp(Zva_t).argmax(1) == yva_t).float().mean().item()
                if va > bva:
                    bva = va; bte = (mlp(Zte_t).argmax(1) == yte_t).float().mean().item(); bep = ep
        print(f'  MLP H={H}: head_params={hp} val={bva:.4f}@ep{bep} test={bte:.4f}'
              f'{"  >=94" if bte >= .94 else ""}{"  >=95" if bte >= .95 else ""}', flush=True)


if __name__ == '__main__':
    main()
