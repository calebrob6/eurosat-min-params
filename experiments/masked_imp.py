#!/usr/bin/env python
"""Masked-multinomial iterative magnitude pruning (IMP) on the o6 pool.

The fair test of *element-wise* weight sparsity as a parameter-reduction lever.
OvR-L1 (experiments/sparse_l1_sweep-style probe) needs ~563 params for val>=0.94
-- but OvR-L1 carries two handicaps vs the dense backward-greedy k=25 (260-param)
frontier: it fits each class independently (not joint multinomial) and it pays
the L1 shrinkage bias.  This removes both: a JOINT multinomial head trained with
a fixed 0/1 mask on the weight matrix, pruned by global magnitude and RETRAINED
(no L1 bias) at each sparsity level.

Params counted honestly as nnz(W) + nnz(b) = nnz(W) + 10, the same convention
that lets prior submissions count 10*(k+1) (the sparsity *pattern* is structural,
only the non-zero *values* are parameters).  Features are standardised (unit
variance) so |W| is comparable across columns for global magnitude pruning; the
scaler folds into the deployed weights, preserving zeros.

Reference frontier: sub09 dense k=25 = 260 params, val 0.9404, test 0.9443.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

BASE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def load(split):
    return (np.load(os.path.join(BASE, f'{split}_feat_o6.npy')),
            np.load(os.path.join(BASE, f'{split}_y.npy')))


def main():
    wd = float(os.environ.get('WD', '2e-3'))
    Xtr, ytr = load('train')
    Xva, yva = load('val')
    Xte, yte = load('test')
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Ztr = torch.tensor((Xtr - mu) / sd, dtype=torch.float32, device=DEV)
    Zva = torch.tensor((Xva - mu) / sd, dtype=torch.float32, device=DEV)
    Zte = torch.tensor((Xte - mu) / sd, dtype=torch.float32, device=DEV)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEV)
    F = Ztr.shape[1]
    K = int(ytr.max() + 1)

    W = torch.zeros(K, F, device=DEV, requires_grad=True)
    b = torch.zeros(K, device=DEV, requires_grad=True)
    mask = torch.ones(K, F, device=DEV)

    def train():
        opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=300,
                                tolerance_grad=1e-7, tolerance_change=1e-9,
                                line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            logits = Ztr @ (W * mask).t() + b
            loss = torch.nn.functional.cross_entropy(logits, ytr_t)
            loss = loss + wd * ((W * mask) ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            W.mul_(mask)

    @torch.no_grad()
    def acc(Z, y):
        pred = (Z @ (W * mask).t() + b).argmax(1).cpu().numpy()
        return float((pred == y).mean())

    # full-support fit (calibration point)
    train()
    nnz = int((mask.bool() & (W != 0)).sum())
    print(f'WD={wd}  full-support: nnz={nnz} params={nnz + 10} '
          f'val={acc(Zva, yva):.4f} test={acc(Zte, yte):.4f}')
    print(f'\n{"targetNNZ":>9} {"params":>6} {"val":>7} {"test":>7}')

    targets = [700, 600, 500, 450, 400, 350, 300, 280, 260, 250,
               240, 230, 220, 210, 200, 190, 180, 160, 140]
    for tgt in targets:
        with torch.no_grad():
            flat = (W.abs() * mask).flatten()
            nz = int((flat > 0).sum())
            if tgt >= nz:
                continue
            thresh = torch.topk(flat, tgt, largest=True).values.min()
            mask.copy_((W.abs() >= thresh).float() * mask)
            W.mul_(mask)
        train()  # retrain survivors jointly
        nnz = int((W != 0).sum())
        print(f'{tgt:>9} {nnz + 10:>6} {acc(Zva, yva):>7.4f} {acc(Zte, yte):>7.4f}',
              flush=True)
    print('IMP_DONE', flush=True)


if __name__ == '__main__':
    main()
