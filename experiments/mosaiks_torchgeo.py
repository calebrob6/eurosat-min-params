#!/usr/bin/env python
"""MOSAIKS via torchgeo's CANONICAL RCF (with ZCA-whitened empirical filters).

Corrects experiments/mosaiks.py, which hand-rolled the featurizer and (a) skipped
ZCA whitening on empirical patches and (b) unit-normalised the gaussian filters
with bias 0. torchgeo.models.RCF is the reference MOSAIKS implementation:
  * gaussian: raw N(0,1) filters + bias=-1.0 ReLU threshold, +/- ReLU, GAP.
  * empirical: patches sampled from the dataset then ZCA-whitened.
Feeds per-band standardised 13-band inputs; linear head; test reported once.
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
    ap.add_argument('--mode', choices=['gaussian', 'empirical'], default='gaussian')
    ap.add_argument('--K', type=int, default=512, help='num filters (features=2K)')
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--bias', type=float, default=-1.0)
    ap.add_argument('--seed', type=int, default=1)   # torchgeo ignores seed=0
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    from torchgeo.models import RCF

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    def load(s):
        x, y = load_cached(s); return x.astype(np.float32), y
    xtr, ytr = load('train'); xva, yva = load('val'); xte, yte = load('test')
    C = xtr.shape[1]
    mu = xtr.mean((0, 2, 3), keepdims=True); sd = xtr.std((0, 2, 3), keepdims=True) + 1e-6
    Xtr = ((xtr - mu) / sd).astype(np.float32)
    Xva = ((xva - mu) / sd).astype(np.float32)
    Xte = ((xte - mu) / sd).astype(np.float32)

    class ArrDS:  # minimal NonGeoDataset-like: dataset[i]['image'], len(dataset)
        def __init__(self, x): self.x = x
        def __len__(self): return len(self.x)
        def __getitem__(self, i): return {'image': torch.tensor(self.x[i])}

    kwargs = dict(in_channels=C, features=2 * args.K, kernel_size=args.kernel,
                  bias=args.bias, seed=args.seed, mode=args.mode)
    if args.mode == 'empirical':
        kwargs['dataset'] = ArrDS(Xtr)
    rcf = RCF(**kwargs).to(dev).eval()

    @torch.inference_mode()
    def feat(x):
        outs = []
        xt = torch.tensor(x)
        for i in range(0, len(xt), 128):
            outs.append(rcf(xt[i:i + 128].to(dev)).cpu().numpy())
        return np.concatenate(outs)
    Ftr, Fva, Fte = feat(Xtr), feat(Xva), feat(Xte)
    print(f'[torchgeo RCF {args.mode} K={args.K} k={args.kernel} bias={args.bias}] '
          f'features={Ftr.shape[1]}', flush=True)

    sc = StandardScaler().fit(Ftr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Ftr), ytr)
    va = accuracy_score(yva, clf.predict(sc.transform(Fva)))
    te = accuracy_score(yte, clf.predict(sc.transform(Fte)))
    print(f'  linear head params={10*(Ftr.shape[1]+1)} val={va:.4f} test={te:.4f}'
          f'{"  >=94" if te >= .94 else ""}{"  >=95" if te >= .95 else ""}', flush=True)


if __name__ == '__main__':
    main()
