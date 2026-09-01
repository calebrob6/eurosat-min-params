#!/usr/bin/env python
"""Remove one feature from the verified 52-feature >96% candidate.

All 52 leave-one-out candidates are ranked using train-only CV seeds 0..2 at
the fixed C=3.  The winner is checked on disjoint CV seeds 10..19 and 30..39
and on validation.  Test is loaded and evaluated only if all three independent
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

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import backward_eliminate, mean_cv  # noqa: E402

FAMILIES = [
    'feat_o6',
    'linefam_line',
    'gs2fam_corn2',
    'gs2fam_lbp2',
    'gs2fam_blob2',
    'gs2fam_sslope2',
    'difam_ixcoh',
    'ixfam_ixtex2',
    'ofam_xcorr',
    'ofam_oent2',
]
INIT_IDX = np.array([
    351, 92, 325, 297, 316, 294, 144, 94, 145, 348, 292, 111, 206, 195, 298,
    75, 37, 317, 287, 86, 291, 26, 321, 205, 323, 95, 36, 320, 17, 310, 91,
    29, 10, 93, 63, 197, 68, 33, 329, 300, 319, 60, 38, 358, 45, 314, 289,
    65, 32, 361, 273, 365,
])
C = 3.0
GATE = 0.960


def load_pool(split: str) -> np.ndarray:
    parts = [
        np.load(os.path.join(CACHE_DIR, f'{split}_{family}.npy'))
        for family in FAMILIES
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'prune_ceiling96_thirteen_result.txt'
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    emit('Thirteenth one-step backward prune of verified 52-feature >96% candidate')
    emit('select seeds=0..2; verify A=10..19; verify B=30..39; C=3')
    emit('test policy: evaluate only after verify A, verify B, and val >= 0.960')

    idx, trace = backward_eliminate(
        ftr,
        ytr,
        INIT_IDX,
        51,
        select_seeds=range(3),
        C=C,
        workers=args.workers,
    )
    dropped = int(next(iter(set(INIT_IDX) - set(idx))))
    select_cv = trace[-1][1]
    verify_a = mean_cv(ftr, ytr, idx, range(10, 20), C=C)
    verify_b = mean_cv(ftr, ytr, idx, range(30, 40), C=C)
    w, b, fi = fit_folded_logreg(ftr, ytr, idx, C=C)
    val = accuracy_score(yva, predict(fva, w, b, fi))

    emit()
    emit(f'dropped_feature={dropped}')
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
        yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
        test = accuracy_score(yte, predict(fte, w, b, fi))
        emit(f'test={test:.4f}')
        emit(f'objective_gate={">96 PASS" if test > GATE else "FAIL"}')
    else:
        emit('test=not evaluated')

    emit()
    emit('feature_idx')
    emit(','.join(map(str, idx)))

    with open(args.output, 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
