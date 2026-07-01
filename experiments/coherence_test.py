#!/usr/bin/env python
"""Does adding structure-tensor *coherence* features lower the honest CV floor?

Coherence measures how *directional* the local gradient field is (high for
linear structures like roads/Highway, low for isotropic fields).  It is
parameter-free (fixed arithmetic on the bands) and captures signal absent from
mean/std/percentile/gradient-magnitude.  We concatenate coherence (2 scales x
13 bands = 26 features) to the cached 169, L1-rank the augmented pool, and run
the same de-noised multi-seed 5-fold CV sweep used to establish the 660-param
floor.  If CV>=0.940 now triggers below k=65, the features help.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.data import CLASSES  # noqa: E402
from src.linmodel import fit_folded_logreg, l1_rank, predict  # noqa: E402


def _pool2(imgs):
    n, c, h, w = imgs.shape
    return imgs.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def coherence_features(imgs, scales=2, eps=1e-6):
    """Structure-tensor coherence per band per scale -> (N, 13*scales)."""
    parts, names = [], []
    cur = imgs
    for s in range(scales):
        gx = np.diff(cur, axis=3)[:, :, :-1, :]
        gy = np.diff(cur, axis=2)[:, :, :, :-1]
        sxx = (gx * gx).mean((2, 3))
        syy = (gy * gy).mean((2, 3))
        sxy = (gx * gy).mean((2, 3))
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / (sxx + syy + eps)
        parts.append(coh.astype(np.float32))
        names += [f'coh{s}_b{i}' for i in range(imgs.shape[1])]
        cur = _pool2(cur)
    return np.concatenate(parts, 1), names


def load_aug(split):
    base = np.load(os.path.join(CACHE_DIR, f'{split}_feat.npy'))
    y = np.load(os.path.join(CACHE_DIR, f'{split}_y.npy'))
    x, _ = load_cached(split)
    coh, _ = coherence_features(x)
    return np.concatenate([base, coh], 1), y


def main():
    ftr, ytr = load_aug('train')
    fva, yva = load_aug('val')
    fte, yte = load_aug('test')
    print(f'augmented feature dim: {ftr.shape[1]} (169 base + 26 coherence)')
    order = l1_rank(ftr, ytr)
    # how many coherence features land in the top-65?
    n_coh_top = int((order[:65] >= 169).sum())
    print(f'coherence features in top-65 by L1 rank: {n_coh_top}')

    seeds = [0, 1, 2, 3, 4]
    print(f'{"k":>4} {"params":>7} {"cv_mean":>8} {"val":>7} {"test":>7}')
    for k in [45, 50, 55, 60, 65]:
        idx = order[:k]
        vals = []
        for sd in seeds:
            skf = StratifiedKFold(5, shuffle=True, random_state=sd)
            accs = [accuracy_score(ytr[te], predict(ftr[te],
                    *fit_folded_logreg(ftr[tr], ytr[tr], feature_idx=idx, C=10.0)))
                    for tr, te in skf.split(ftr, ytr)]
            vals.append(np.mean(accs))
        cv = float(np.mean(vals))
        w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=10.0)
        va = accuracy_score(yva, predict(fva, w, b, fi))
        te = accuracy_score(yte, predict(fte, w, b, fi))
        print(f'{k:>4} {10*(k+1):>7} {cv:>8.4f} {va:>7.4f} {te:>7.4f}')

    # per-class test acc at k=55 to see Highway effect
    idx = order[:55]
    w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=10.0)
    cm = confusion_matrix(yte, predict(fte, w, b, fi))
    per = cm.diagonal() / cm.sum(1)
    print('\nper-class test acc @ k=55 (augmented):')
    for c, a in sorted(zip(CLASSES, per), key=lambda t: t[1]):
        print(f'  {c:22s} {a:.3f}')


if __name__ == '__main__':
    main()
