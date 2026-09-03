#!/usr/bin/env python
"""What a few learned hidden units are worth next to a dense linear head.

Iterations 4 and 5 named 'a head with a few learned hidden units' as one of
two remaining levers, and ``resisc45_transform_ceiling.py --probe`` shows the
nonlinear headroom on the dihedral-averaged 768-column list is intact (MLP
width 1,024 reads 86.3% against 83.3% linear).  Before building a budgeted
version, this reads the lever at the ceiling with the accounting in view.

Model: ``logits = z W + b + relu(z V + c) U`` on standardised columns ``z``,
a dense linear head plus ``H`` ReLU units, trained jointly (Adam, cosine
schedule, weight decay on validation).  Then each unit's input vector ``V[:,
h]`` is magnitude-pruned to its top ``s`` columns and the model retrained
under the mask from the dense solution, which is what a deployment would
store: ``s + 1`` values per unit for its inputs and bias plus 44 output
weights, next to the linear head.

Rows: ``H`` in ``--hidden`` with dense inputs, then ``H x s`` for every
``--sparsity``, all with train, validation and test accuracy.  The dense
linear head (``H = 0``) is the control.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_layout_gain import LAM, load_pools
from resisc45_lib import accuracy, group_lasso_rank, standardise
from resisc45_pool5_ceiling import write_rows
from resisc45_transform_ceiling import BLOCKS, QUOTA

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_hidden_units_result.csv')
DECAYS = (1e-4, 3e-4, 1e-3, 3e-3)
ROWS = 44


def fit_skip_mlp(z, y, hidden, decay, epochs, lr=0.03, mask=None, init=None, seed=0,
                 device='cuda'):
    torch.manual_seed(seed)
    xs = torch.as_tensor(z['train'], device=device)
    yt = torch.as_tensor(y['train'], device=device, dtype=torch.long)
    k = xs.shape[1]
    n_classes = int(yt.max()) + 1
    if init is None:
        w = (torch.randn(k, n_classes, device=device) * 0.01).requires_grad_(True)
        b = torch.zeros(n_classes, device=device, requires_grad=True)
        v = (torch.randn(k, hidden, device=device) / np.sqrt(k)).requires_grad_(True)
        c = torch.zeros(hidden, device=device, requires_grad=True)
        u = (torch.randn(hidden, n_classes, device=device) / np.sqrt(max(hidden, 1))
             ).requires_grad_(True)
    else:
        w, b, v, c, u = [torch.as_tensor(p, device=device).clone().requires_grad_(True)
                         for p in init]
    m = None if mask is None else torch.as_tensor(mask, device=device, dtype=torch.float32)
    params = [w, b] + ([v, c, u] if hidden > 0 else [])
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = xs @ w + b
        if hidden > 0:
            vv = v if m is None else v * m
            logits = logits + torch.relu(xs @ vv + c) @ u
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss = loss + decay * ((w * w).sum() + ((vv * vv).sum() + (u * u).sum()
                                                if hidden > 0 else 0.0))
        loss.backward()
        opt.step()
        sched.step()
    out = [p.detach().cpu().numpy() for p in (w, b, v, c, u)]
    if m is not None:
        out[2] = out[2] * mask
    return out


def predict(z, params, hidden):
    w, b, v, c, u = params
    logits = z @ w + b
    if hidden > 0:
        logits = logits + np.maximum(z @ v + c, 0.0) @ u
    return logits.argmax(1)


def evaluate(z, y, params, hidden):
    return {s: accuracy(predict(z[s], params, hidden), y[s]) for s in z}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, nargs='*', default=[0, 4, 8, 16, 32, 64])
    parser.add_argument('--sparsity', type=int, nargs='*', default=[8, 16, 32, 64])
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--retrain-epochs', type=int, default=1500)
    parser.add_argument('--out', default=RESULT_PATH)
    args = parser.parse_args()

    pools, y = [], None
    for names in BLOCKS:
        b, y, _ = load_pools(names)
        pools.append(b)
    orders = [group_lasso_rank(b['train'], y['train'], lam=LAM, epochs=1500)[0] for b in pools]
    raw = {s: np.concatenate([b[s][:, o[:q]] for b, o, q in zip(pools, orders, QUOTA)], axis=1)
           for s in y}
    mu, sigma = standardise(raw['train'])
    z = {s: ((v - mu) / sigma).astype(np.float32) for s, v in raw.items()}
    k = z['train'].shape[1]

    rows = []

    def record(arm, hidden, sparsity, decay, acc, t0):
        hidden_values = hidden * (sparsity + 1 + ROWS) if hidden else 0
        rows.append({'arm': arm, 'hidden': hidden, 'inputs_per_unit': sparsity,
                     'hidden_values': hidden_values, 'decay': decay,
                     'train_accuracy': round(acc['train'], 4),
                     'val_accuracy': round(acc['val'], 4),
                     'test_accuracy': round(acc['test'], 4),
                     'seconds': round(time.time() - t0, 1)})
        print(rows[-1], flush=True)
        write_rows(rows, args.out)

    for hidden in args.hidden:
        best, t0 = None, time.time()
        for decay in DECAYS:
            params = fit_skip_mlp(z, y, hidden, decay, args.epochs)
            acc = evaluate(z, y, params, hidden)
            if best is None or acc['val'] > best[1]['val']:
                best = (params, acc, decay)
        record('dense-inputs', hidden, k, best[2], best[1], t0)
        if hidden == 0:
            continue
        w, b, v, c, u = best[0]
        for sparsity in args.sparsity:
            if sparsity >= k:
                continue
            t0 = time.time()
            mask = np.zeros_like(v)
            top = np.argsort(-np.abs(v), axis=0)[:sparsity]
            np.put_along_axis(mask, top, 1.0, axis=0)
            best_s = None
            for decay in DECAYS:
                params = fit_skip_mlp(z, y, hidden, decay, args.retrain_epochs, mask=mask,
                                      init=(w, b, v * mask, c, u))
                acc = evaluate(z, y, params, hidden)
                if best_s is None or acc['val'] > best_s[1]['val']:
                    best_s = (params, acc, decay)
            record('sparse-inputs', hidden, sparsity, best_s[2], best_s[1], t0)


if __name__ == '__main__':
    main()
