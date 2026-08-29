#!/usr/bin/env python
"""Per-band image-statistics baseline with validation-tuned logistic regression.

Each 13-band patch becomes 52 fixed features: mean, standard deviation, minimum,
and maximum for each band. The feature standardizer is folded into the logistic
regression weights, and the final head is converted to reference-class form so
its deployed parameter count is ``(10 - 1) * (52 + 1) = 477``.

Only validation accuracy is used to choose C. Test accuracy is computed once for
the selected C.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR, build_cache  # noqa: E402
from src.data import BAND_NAMES, CLASSES  # noqa: E402
from src.linmodel import (fit_folded_logreg, predict,  # noqa: E402
                          predict_reference_class, to_reference_class)

HERE = os.path.dirname(__file__)
RESULT_PATH = os.path.join(HERE, 'image_statistics_baseline_result.txt')


def image_statistics(split: str, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load or compute per-band mean, std, min, and max for one split."""
    feature_path = os.path.join(CACHE_DIR, f'{split}_feat_image_statistics.npy')
    label_path = os.path.join(CACHE_DIR, f'{split}_y.npy')
    image_path = os.path.join(CACHE_DIR, f'{split}_x_uint16.npy')

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        build_cache(split)
    if os.path.exists(feature_path):
        return np.load(feature_path), np.load(label_path)

    images = np.load(image_path, mmap_mode='r')
    features = np.empty((len(images), len(BAND_NAMES) * 4), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        stop = min(start + batch_size, len(images))
        flat = images[start:stop].astype(np.float32).reshape(
            stop - start, len(BAND_NAMES), -1,
        )
        batch_features = np.stack(
            (flat.mean(2), flat.std(2), flat.min(2), flat.max(2)),
            axis=2,
        )
        features[start:stop] = batch_features.reshape(stop - start, -1)

    np.save(feature_path, features)
    return features, np.load(label_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--Cs',
        type=float,
        nargs='+',
        default=[
            0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
            1.0, 3.0, 10.0, 30.0, 100.0, 150.0, 200.0, 250.0,
            300.0, 400.0, 500.0, 700.0, 1000.0, 2000.0,
        ],
    )
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()

    ftr, ytr = image_statistics('train', args.batch_size)
    fva, yva = image_statistics('val', args.batch_size)
    fte, yte = image_statistics('test', args.batch_size)

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    emit('PER-BAND IMAGE-STATISTICS + LOGISTIC-REGRESSION BASELINE')
    emit(f'features: {ftr.shape[1]} (13 bands x mean,std,min,max)')
    emit(f'splits: train={len(ytr)} val={len(yva)} test={len(yte)}')
    emit(f'{"C":>9} {"val":>7}')

    best = None
    for C in args.Cs:
        w, b, fi = fit_folded_logreg(ftr, ytr, C=C)
        val_acc = accuracy_score(yva, predict(fva, w, b, fi))
        emit(f'{C:>9g} {val_acc:>7.4f}')
        if best is None or val_acc > best[0]:
            best = (val_acc, C, w, b, fi)

    val_acc, C, w, b, fi = best
    full_val_pred = predict(fva, w, b, fi)
    full_test_pred = predict(fte, w, b, fi)
    w_ref, b_ref = to_reference_class(w, b, ref=0)
    val_pred = predict_reference_class(fva, w_ref, b_ref, fi, ref=0)
    test_pred = predict_reference_class(fte, w_ref, b_ref, fi, ref=0)
    assert np.array_equal(val_pred, full_val_pred)
    assert np.array_equal(test_pred, full_test_pred)

    test_acc = accuracy_score(yte, test_pred)
    full_params = int(w.size + b.size)
    reference_params = int(w_ref.size + b_ref.size)
    emit()
    emit(f'selected C={C:g} on validation')
    emit(f'val={val_acc:.4f}')
    emit(f'test={test_acc:.4f}')
    emit(f'parameters: full-head={full_params} reference-class={reference_params}')
    emit('per-class test accuracy:')
    cm = confusion_matrix(yte, test_pred)
    per_class = cm.diagonal() / cm.sum(1)
    for name, acc in sorted(zip(CLASSES, per_class), key=lambda item: item[1]):
        emit(f'  {name:22s} {acc:.4f}')

    with open(RESULT_PATH, 'w') as result_file:
        result_file.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
