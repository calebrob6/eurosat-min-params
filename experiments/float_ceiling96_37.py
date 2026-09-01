#!/usr/bin/env python
"""Try one floating exchange from the failed nested 37-feature candidate.

The verified 38-feature frontier cannot be pruned directly: dropping feature
319 gives the strongest nested 37-feature subset but misses the validation
gate.  This experiment searches for a non-nested child in two train-only steps:

1. Add every feature outside the 38-feature parent to the failed child and keep
   the best additions by seed-0 CV.
2. For each beam addition, remove each feature from the failed child and rank
   the resulting 37-feature exchanges by CV seeds 0..2.

The winning exchange is checked on disjoint CV seeds 10..19 and 30..39 plus
validation.  Test remains inaccessible unless all three checks reach 96%.
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
PARENT_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 292, 111, 206, 195, 298, 75, 37,
    317, 287, 86, 291, 205, 323, 95, 17, 310, 29, 63, 197, 68, 329, 300,
    319, 358, 314, 289, 65, 32, 361, 365,
])
FAILED_DROP = 319
C = 3.0
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


def score_subset(
    candidate: tuple[tuple[int, int], np.ndarray, tuple[int, ...]],
) -> tuple[tuple[int, int], float]:
    tag, idx, seeds = candidate
    return tag, mean_cv(_X, _Y, idx, seeds, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', type=int, default=12)
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'float_ceiling96_37_result.txt'
        ),
    )
    args = parser.parse_args()

    if args.beam < 1:
        parser.error('--beam must be positive')

    ftr = load_pool('train')
    fva = load_pool('val')
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    child = PARENT_IDX[PARENT_IDX != FAILED_DROP]
    parent_set = set(int(feature) for feature in PARENT_IDX)
    additions = [
        feature for feature in range(ftr.shape[1]) if feature not in parent_set
    ]

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    emit('One floating exchange from failed nested 37-feature candidate')
    emit('screen additions with seed=0; rank exchanges with seeds=0..2; C=3')
    emit('verify A=10..19; verify B=30..39; test gated by both plus validation')
    emit(f'pool_features={ftr.shape[1]} non_parent_additions={len(additions)}')
    emit(f'beam={min(args.beam, len(additions))}')

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(ftr, ytr),
    ) as executor:
        screen_jobs = [
            ((feature, -1), np.append(child, feature), (0,))
            for feature in additions
        ]
        screen_scores = list(executor.map(score_subset, screen_jobs))
        screen_scores.sort(key=lambda item: (-item[1], item[0][0]))
        beam = screen_scores[:args.beam]

        emit()
        emit('addition screen (top beam)')
        for (added, _), score in beam:
            emit(f'add={added} seed0_cv={score:.4f}')

        exchange_jobs = []
        for (added, _), _score in beam:
            for dropped in child:
                idx = np.append(child[child != dropped], added)
                exchange_jobs.append(
                    ((int(added), int(dropped)), idx, (0, 1, 2))
                )
        exchange_scores = list(executor.map(score_subset, exchange_jobs))

    exchange_scores.sort(
        key=lambda item: (-item[1], item[0][0], item[0][1])
    )
    (added, dropped), select_cv = exchange_scores[0]
    winner = np.append(child[child != dropped], added)

    child_cv = mean_cv(ftr, ytr, child, range(3), C=C)
    parent_cv = mean_cv(ftr, ytr, PARENT_IDX, range(3), C=C)
    verify_a = mean_cv(ftr, ytr, winner, range(10, 20), C=C)
    verify_b = mean_cv(ftr, ytr, winner, range(30, 40), C=C)
    w, b, fi = fit_folded_logreg(ftr, ytr, winner, C=C)
    val = accuracy_score(yva, predict(fva, w, b, fi))

    emit()
    emit(f'baseline_parent_select_cv={parent_cv:.4f}')
    emit(f'baseline_nested_child_select_cv={child_cv:.4f}')
    emit(f'added_feature={added}')
    emit(f'dropped_feature={dropped}')
    emit(f'k={len(winner)}')
    emit(f'reference-class parameters={9 * (len(winner) + 1)}')
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
    emit(','.join(map(str, winner)))

    with open(args.output, 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
