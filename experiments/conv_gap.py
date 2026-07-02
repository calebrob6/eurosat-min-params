#!/usr/bin/env python
"""Tiny conv -> GAP -> logreg on EuroSAT (last night's idea C), RGB or all bands.

The deployed model is Conv2d(C_in->F, 3x3, pad=1) -> BN -> ReLU -> GlobalAvgPool
-> Linear(F, 10). BN and the fixed per-band input standardisation are affine, so
they FOLD into the conv at deploy time (exactly like the StandardScaler folding in
the linear submissions) and cost no deployed parameters. Deployed params are
therefore:

    F*C_in*9 + F   (conv)  +  10*F + 10   (head)  =  F*(9*C_in + 11) + 10

Optional knowledge distillation: pass --teacher-logits <npz with train/val/test
logits> to add a temperature-scaled KL term (soft targets from a strong teacher
whose params do NOT count -- only the student ships).

Selection stays honest: choose the epoch by VAL, report TEST once. Run one config
per GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import load_cached  # noqa: E402
from src.data import B_BLUE, B_GREEN, B_RED  # noqa: E402

RGB = [B_RED, B_GREEN, B_BLUE]   # [3,2,1]


def deploy_params(F: int, c_in: int) -> int:
    return F * (9 * c_in + 11) + 10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--bands', choices=['rgb', 'all'], default='rgb')
    ap.add_argument('--filters', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=5e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--teacher-logits', type=str, default=None)
    ap.add_argument('--kd-T', type=float, default=4.0)
    ap.add_argument('--kd-alpha', type=float, default=0.7)
    ap.add_argument('--save-logits', type=str, default=None,
                    help='if set, also train as a TEACHER and dump its logits here')
    ap.add_argument('--teacher', action='store_true',
                    help='use the larger teacher CNN instead of the tiny student')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    import torch
    import torch.nn as nn
    import torch.nn.functional as F_

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bands = RGB if args.bands == 'rgb' else list(range(13))
    c_in = len(bands)

    def load(split):
        x, y = load_cached(split)
        return x[:, bands].astype(np.float32), y
    xtr, ytr = load('train')
    xva, yva = load('val')
    xte, yte = load('test')
    # fixed per-band standardisation (folds into conv1 at deploy -> 0 params)
    mu = xtr.mean((0, 2, 3), keepdims=True)
    sd = xtr.std((0, 2, 3), keepdims=True) + 1e-6
    def norm(x):
        return torch.tensor((x - mu) / sd, device=dev)
    Xtr, Xva, Xte = norm(xtr), norm(xva), norm(xte)
    Ytr = torch.tensor(ytr, device=dev)
    yva_t = torch.tensor(yva, device=dev)
    yte_t = torch.tensor(yte, device=dev)

    teach = None
    if args.teacher_logits:
        tl = np.load(args.teacher_logits)
        teach = torch.tensor(tl['train'], device=dev)

    class Student(nn.Module):
        def __init__(self, F):
            super().__init__()
            self.c = nn.Conv2d(c_in, F, 3, padding=1)
            self.bn = nn.BatchNorm2d(F)
            self.fc = nn.Linear(F, 10)

        def forward(self, x):
            h = F_.relu(self.bn(self.c(x)))
            h = h.mean((2, 3))              # global average pool
            return self.fc(h)

    class Teacher(nn.Module):
        """Larger net (params don't ship); provides soft targets for distillation."""
        def __init__(self):
            super().__init__()
            def blk(a, b):
                return nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b),
                                     nn.ReLU(), nn.Conv2d(b, b, 3, padding=1),
                                     nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
            self.f = nn.Sequential(blk(c_in, 32), blk(32, 64), blk(64, 128))
            self.head = nn.Linear(128, 10)

        def forward(self, x):
            h = self.f(x).mean((2, 3))
            return self.head(h)

    net = (Teacher() if args.teacher else Student(args.filters)).to(dev)
    n_train_params = sum(p.numel() for p in net.parameters())
    dp = deploy_params(args.filters, c_in) if not args.teacher else n_train_params
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ntr = len(Ytr)

    def evaluate(X, y):
        net.eval()
        with torch.no_grad():
            return (net(X).argmax(1) == y).float().mean().item()

    best_va, best_te, best_ep = 0.0, 0.0, -1
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, args.bs):
            idx = perm[i:i + args.bs]
            xb, yb = Xtr[idx], Ytr[idx]
            # cheap augmentation: random flips + 90-deg rotations (EuroSAT is ~iso)
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, [3])
            if torch.rand(1).item() < 0.5:
                xb = torch.flip(xb, [2])
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                xb = torch.rot90(xb, k, [2, 3])
            logits = net(xb)
            loss = F_.cross_entropy(logits, yb)
            if teach is not None:
                T = args.kd_T
                kd = F_.kl_div(F_.log_softmax(logits / T, 1),
                               F_.softmax(teach[idx] / T, 1),
                               reduction='batchmean') * (T * T)
                loss = (1 - args.kd_alpha) * loss + args.kd_alpha * kd
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        va = evaluate(Xva, yva_t)
        if va > best_va:
            best_va, best_te, best_ep = va, evaluate(Xte, yte_t), ep

    tag = 'TEACHER' if args.teacher else f'student F={args.filters}'
    kd = f' KD(T={args.kd_T},a={args.kd_alpha})' if teach is not None else ''
    print(f'[{args.bands} {tag}{kd}] train_params={n_train_params} deploy_params={dp} '
          f'| best_val={best_va:.4f}@ep{best_ep} test@best_val={best_te:.4f} '
          f'{">=94!" if best_te >= 0.94 else ""}', flush=True)

    if args.save_logits:
        net.eval()
        with torch.no_grad():
            out = {s: net(X).cpu().numpy() for s, X in
                   (('train', Xtr), ('val', Xva), ('test', Xte))}
        np.savez(args.save_logits, **out)
        print(f'saved teacher logits -> {args.save_logits}', flush=True)


if __name__ == '__main__':
    main()
