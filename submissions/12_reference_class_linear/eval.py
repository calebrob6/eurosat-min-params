#!/usr/bin/env python
"""Independently evaluate the saved submission-12 model.

Recomputes ALL features directly from the raw patches (bypassing every cache),
using exactly the feature configuration stored in the checkpoint, then applies
the 9-row reference-class linear head (class ``ref_class`` has an implicit
constant 0 logit).  A genuine end-to-end check of extraction + the deployed
affine map, and of the stored parameter count.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.data import CLASSES, iter_images  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import predict_reference_class  # noqa: E402

HERE = os.path.dirname(__file__)


def main() -> None:
    m = np.load(os.path.join(HERE, 'model.npz'))
    w, b, fi, ref = m['W'], m['b'], m['feature_idx'], int(m['ref_class'])
    cfg = dict(
        coherence_scales=int(m['coherence_scales']),
        orient_entropy_bins=int(m['orient_entropy_bins']),
        orient_hist_bins=int(m['orient_hist_bins']),
        spectral_peak=bool(m['spectral_peak']),
        xband=bool(m['xband']),
        index_texture=bool(m['index_texture']),
        hough_lines=bool(m['hough_lines']),
        harris_corners=bool(m['harris_corners']),
    )
    params = int(w.size + b.size)
    assert params == int(m['params']) == (len(CLASSES) - 1) * (len(fi) + 1)
    print(f'model: k={int(m["k"])} C={float(m["C"]):.0f} head=(K-1)={w.shape[0]} rows '
          f'ref_class={ref} params={params} cfg={cfg}')

    for split in ('val', 'test'):
        predictions, labels = [], []
        for x, y in iter_images(split):
            f, _ = patch_features(x, **cfg)
            predictions.append(predict_reference_class(f, w, b, fi, ref=ref))
            labels.append(y)
        pred, y = np.concatenate(predictions), np.concatenate(labels)
        acc = accuracy_score(y, pred)
        print(f'{split}: acc={acc:.4f}  (n={len(y)})')
        if split == 'test':
            cm = confusion_matrix(y, pred)
            per_cls = cm.diagonal() / cm.sum(1)
            for c, a in sorted(zip(CLASSES, per_cls), key=lambda t: t[1]):
                print(f'  {c:22s} {a:.3f}')


if __name__ == '__main__':
    main()
