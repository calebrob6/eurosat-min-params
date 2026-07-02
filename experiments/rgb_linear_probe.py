#!/usr/bin/env python
"""How costly is RGB-ONLY for the linear-on-features model? (params = 10*(k+1))

Builds a parameter-free feature set from ONLY the RGB bands (mean/std/percentiles
+ multi-scale gradient-magnitude stats) and finds the backward-greedy honest floor
(smallest k with verify-CV >= 0.940 on disjoint seeds AND val >= 0.940). Compares
to the all-13-band frontier (18 feats @ test 0.9433). Point: dropping NIR/SWIR does
NOT lower params (params depend on k, not band count) and typically NEEDS MORE
features, so RGB-only is a worse param trade for the LINEAR model -- it only saves
params in the conv model, where input channels multiply the conv weights.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import load_cached  # noqa: E402
from src.data import B_BLUE, B_GREEN, B_RED  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402

RGB = [B_RED, B_GREEN, B_BLUE]
PCT = [10, 25, 50, 75, 90]


def _gradmag(imgs):  # (N,C,H,W) -> (N,C,H,W) magnitude
    gx = np.zeros_like(imgs); gy = np.zeros_like(imgs)
    gx[..., :, 1:] = np.diff(imgs, axis=3)
    gy[..., 1:, :] = np.diff(imgs, axis=2)
    return np.sqrt(gx * gx + gy * gy)


def _pool(imgs, s):
    if s == 1:
        return imgs
    n, c, h, w = imgs.shape
    return imgs.reshape(n, c, h // s, s, w // s, s).mean((3, 5))


def feats(split):
    x, y = load_cached(split)
    x = x[:, RGB].astype(np.float32)          # (N,3,64,64)
    n = x.shape[0]
    flat = x.reshape(n, 3, -1)
    parts = [flat.mean(2), flat.std(2)]
    names = [f'{b}_mean' for b in 'RGB'] + [f'{b}_std' for b in 'RGB']
    pcts = np.percentile(flat, PCT, axis=2)   # (5,N,3)
    for i, p in enumerate(PCT):
        parts.append(pcts[i]); names += [f'{b}_p{p}' for b in 'RGB']
    for s in (1, 2, 4):                        # multi-scale gradient-magnitude stats
        gm = _gradmag(_pool(x, s)).reshape(n, 3, -1)
        parts += [gm.mean(2), gm.std(2)]
        names += [f'{b}_gmean{s}' for b in 'RGB'] + [f'{b}_gstd{s}' for b in 'RGB']
    return np.concatenate(parts, 1).astype(np.float32), y, names


def main():
    ftr, ytr, names = feats('train')
    fva, yva, _ = feats('val')
    fte, yte, _ = feats('test')
    print(f'RGB-only feature pool: {ftr.shape[1]} features (from 3 bands)')

    # full-pool accuracy, then honest backward-greedy floor
    allidx = np.arange(ftr.shape[1])
    w, b, fi = fit_folded_logreg(ftr, ytr, allidx, C=10.0)
    print(f'full pool k={ftr.shape[1]} params={10*(ftr.shape[1]+1)} '
          f'val={accuracy_score(yva, predict(fva,w,b,fi)):.4f} '
          f'test={accuracy_score(yte, predict(fte,w,b,fi)):.4f}')

    _, _, subs = backward_eliminate(ftr, ytr, allidx, 8, select_seeds=range(10),
                                    C=10.0, workers=12, record=True)
    print(f'\n{"k":>3} {"params":>6} {"verCV":>7} {"val":>7} {"test":>7}  >=.94?')
    floor = None
    for k in sorted(subs, reverse=True):
        s = subs[k]
        w, b, fi = fit_folded_logreg(ftr, ytr, s, C=10.0)
        ver = mean_cv(ftr, ytr, s, range(10, 20), C=10.0)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        ok = ver >= 0.94 and va >= 0.94
        if ok:
            floor = k
        print(f'{k:>3} {10*(k+1):>6} {ver:>7.4f} {va:>7.4f} {te:>7.4f}  {"PASS" if ok else "fail"}')
    print(f'\nRGB-only honest floor: k={floor} -> params={10*(floor+1) if floor else None} '
          f'(all-13-band frontier is k=18, 190 params, test 0.9433)')


if __name__ == '__main__':
    main()
