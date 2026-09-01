#!/usr/bin/env python
"""Remove one feature from the verified 34-feature >96% candidate.

All 34 leave-one-out candidates are ranked using train-only CV seeds 0..2 at
the fixed C=3. The winner is checked on disjoint CV seeds 10..19 and 30..39
and on validation. Test is loaded and evaluated only if all three independent
checks remain at or above 96%.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from region_shape_ceiling96 import load_pool, load_region_shape  # noqa: E402
from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402

POOL_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 206, 195, 298, 75, 37, 317, 287,
    86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 300, 314, 289, 65, 32,
    365, 2,
])
FINAL_POOL_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 206, 195, 298, 75, 37, 317, 287,
    86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 314, 289, 65, 32, 365,
    2,
])
REGION_SHAPE_NAME = 'tail_aniso_low_ndvi'
C = 3.0
GATE = 0.960


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=34)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__),
            'region_shape_prune_ceiling96_33_result.txt',
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, names = load_region_shape('train')
    rva, val_names = load_region_shape('val')
    if names != val_names or rtr.shape[1] != len(names):
        raise ValueError('train/validation region-shape schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + names.index(REGION_SHAPE_NAME)
    init_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    emit('One-step backward prune of verified 34-feature >96% candidate')
    emit('select seeds=0..2; verify A=10..19; verify B=30..39; C=3')
    emit('test policy: evaluate only after verify A, verify B, and val >= 0.960')

    idx, trace = backward_eliminate(
        ftr,
        ytr,
        init_idx,
        33,
        select_seeds=range(3),
        C=C,
        workers=args.workers,
    )
    dropped = int(next(iter(set(init_idx) - set(idx))))
    expected_idx = np.append(FINAL_POOL_IDX, region_idx)
    if not np.array_equal(idx, expected_idx):
        raise ValueError('backward elimination did not reproduce the final subset')
    select_cv = trace[-1][1]
    verify_a = mean_cv(ftr, ytr, idx, range(10, 20), C=C)
    verify_b = mean_cv(ftr, ytr, idx, range(30, 40), C=C)
    w, b, fi = fit_folded_logreg(ftr, ytr, idx, C=C)
    val = accuracy_score(yva, predict(fva, w, b, fi))

    emit()
    if dropped < pool_dim:
        emit(f'dropped_feature=pool:{dropped}')
    else:
        emit(f'dropped_feature=region_shape:{names[dropped-pool_dim]}')
    emit(f'k={len(idx)}')
    emit(f'reference-class parameters={9 * (len(idx) + 1)}')
    emit(f'select CV seeds 0..2={select_cv:.4f}')
    emit(f'verify CV seeds 10..19={verify_a:.4f}')
    emit(f'verify CV seeds 30..39={verify_b:.4f}')
    emit(f'validation={val:.4f}')

    passed = verify_a >= GATE and verify_b >= GATE and val >= GATE
    emit(f'independent_gate={"PASS" if passed else "FAIL"}')
    if passed:
        fte = load_pool('test')
        rte, test_names = load_region_shape('test')
        if names != test_names:
            raise ValueError('test region-shape schema differs')
        fte = np.concatenate((fte, rte), axis=1).astype(np.float32)
        yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
        test = accuracy_score(yte, predict(fte, w, b, fi))
        emit(f'test={test:.4f}')
        emit(f'objective_gate={">96 PASS" if test > GATE else "FAIL"}')
    else:
        emit('test=not evaluated')

    emit()
    emit('pool_feature_idx')
    emit(','.join(map(str, idx[idx < pool_dim])))
    emit('region_shape_features')
    emit(','.join(names[value - pool_dim] for value in idx if value >= pool_dim))

    with open(args.output, 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
