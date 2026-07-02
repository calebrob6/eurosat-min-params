#!/usr/bin/env python
"""MOSAIKS (torchgeo gaussian RCF, seed-free) parameter floor at the 95% gate.

Gaussian RCF filters are seed-free (~0 extractor params). Featurize with a large K,
L1-rank the 2K features, and read the accuracy-vs-k curve to find the fewest
features whose reference-class head (9*(k+1) params) clears verify-CV/val/test 0.95.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault('OMP_NUM_THREADS', '8')

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import load_cached  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--K', type=int, default=2048)
    ap.add_argument('--kernel', type=int, default=3)
    ap.add_argument('--gpu', type=int, default=0)
    args = ap.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    from sklearn.metrics import accuracy_score
    from torchgeo.models import RCF

    def load(s):
        x, y = load_cached(s); return x.astype(np.float32), y
    xtr, ytr = load('train'); xva, yva = load('val'); xte, yte = load('test')
    C = xtr.shape[1]
    mu = xtr.mean((0, 2, 3), keepdims=True); sd = xtr.std((0, 2, 3), keepdims=True) + 1e-6
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    rcf = RCF(in_channels=C, features=2 * args.K, kernel_size=args.kernel,
              bias=-1.0, seed=1, mode='gaussian').to(dev).eval()

    @torch.inference_mode()
    def feat(x):
        X = torch.tensor(((x - mu) / sd).astype(np.float32)); o = []
        for i in range(0, len(X), 128):
            o.append(rcf(X[i:i+128].to(dev)).cpu().numpy())
        return np.concatenate(o)
    Ftr, Fva, Fte = feat(xtr), feat(xva), feat(xte)
    print(f'gaussian RCF K={args.K}: {Ftr.shape[1]} seed-free features')

    order = l1_rank(Ftr, ytr)
    print(f'\n{"k":>4} {"p(9x)":>6} {"verCV3":>7} {"val":>7} {"test":>7}  gate?')
    for k in [64, 128, 256, 384, 512, 768, 1024, 1536, Ftr.shape[1]]:
        if k > Ftr.shape[1]:
            continue
        idx = order[:k]
        w, b, fi = fit_folded_logreg(Ftr, ytr, idx, C=1.0)
        ver = mean_cv(Ftr, ytr, idx, range(10, 13), C=1.0)
        va = accuracy_score(yva, predict(Fva, w, b, fi))
        te = accuracy_score(yte, predict(Fte, w, b, fi))
        g = '  >=95!' if (ver >= .95 and va >= .95 and te >= .95) else ('  test>=95' if te >= .95 else '')
        print(f'{k:>4} {9*(k+1):>6} {ver:>7.4f} {va:>7.4f} {te:>7.4f}{g}', flush=True)


if __name__ == '__main__':
    main()
