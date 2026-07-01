#!/usr/bin/env python
"""Independently evaluate the saved submission-02 model.

Recomputes features directly from the raw patches (bypassing the feature cache)
so this is a genuine end-to-end check of extraction + folded linear model.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.cache import load_cached  # noqa: E402
from src.data import CLASSES  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import predict  # noqa: E402

HERE = os.path.dirname(__file__)


def main() -> None:
    m = np.load(os.path.join(HERE, 'model.npz'))
    w, b, fi = m['W'], m['b'], m['feature_idx']
    print(f'model: k={int(m["k"])} C={float(m["C"]):.0f} params={int(m["params"])}')

    for split in ('val', 'test'):
        x, y = load_cached(split)
        f, _ = patch_features(x)
        pred = predict(f, w, b, fi)
        acc = accuracy_score(y, pred)
        print(f'{split}: acc={acc:.4f}  (n={len(y)})')
        if split == 'test':
            cm = confusion_matrix(y, pred)
            per_cls = cm.diagonal() / cm.sum(1)
            for c, a in sorted(zip(CLASSES, per_cls), key=lambda t: t[1]):
                print(f'  {c:22s} {a:.3f}')


if __name__ == '__main__':
    main()
