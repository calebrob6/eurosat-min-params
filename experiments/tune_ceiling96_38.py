#!/usr/bin/env python
"""Retune regularization for the best backward-pruned 38-feature candidate.

The candidate was selected at C=3 but narrowly missed the independent
validation gate. Select C using train-only CV seeds 0..2, then check the winner
on disjoint CV seeds 10..19 and 30..39 plus validation. Test remains gated
behind all three independent checks.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402

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
FEATURE_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 292, 111, 206, 195, 298, 75, 37,
    317, 287, 86, 291, 205, 323, 95, 17, 310, 29, 63, 197, 68, 329, 300,
    319, 358, 314, 289, 65, 32, 361, 365,
])
C_GRID = (0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0)
GATE = 0.960

_X: np.ndarray | None = None
_Y: np.ndarray | None = None


def load_pool(split: str) -> np.ndarray:
    parts = [
        np.load(os.path.join(CACHE_DIR, f'{split}_{family}.npy'))
        for family in FAMILIES
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def init_worker(x: np.ndarray, y: np.ndarray) -> None:
    global _X, _Y
    _X, _Y = x, y


def score_c(c: float) -> tuple[float, float]:
    return c, mean_cv(_X, _Y, FEATURE_IDX, range(3), C=c)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'tune_ceiling96_38_result.txt'
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    with ProcessPoolExecutor(
        max_workers=min(args.workers, len(C_GRID)),
        initializer=init_worker,
        initargs=(ftr, ytr),
    ) as executor:
        scores = list(executor.map(score_c, C_GRID))

    best_c, select_cv = max(scores, key=lambda item: item[1])
    verify_a = mean_cv(ftr, ytr, FEATURE_IDX, range(10, 20), C=best_c)
    verify_b = mean_cv(ftr, ytr, FEATURE_IDX, range(30, 40), C=best_c)
    w, b, fi = fit_folded_logreg(ftr, ytr, FEATURE_IDX, C=best_c)
    val = accuracy_score(yva, predict(fva, w, b, fi))

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    emit('Regularization retune of best backward-pruned 38-feature candidate')
    emit('C selected with train CV seeds=0..2; verify A=10..19; verify B=30..39')
    emit('test policy: evaluate only after verify A, verify B, and val >= 0.960')
    emit()
    emit('C grid selection CV')
    for c, score in scores:
        emit(f'C={c:g} select_cv={score:.4f}')

    emit()
    emit(f'selected_C={best_c:g}')
    emit(f'k={len(FEATURE_IDX)}')
    emit(f'reference-class parameters={9 * (len(FEATURE_IDX) + 1)}')
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
    emit(','.join(map(str, FEATURE_IDX)))

    with open(args.output, 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
