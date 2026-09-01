#!/usr/bin/env python
"""Try region-shape one-for-two exchanges below the 36-feature frontier.

The region-shape prescreen identified low-pan and low-NDVI tail anisotropy as
the two candidates with the clearest train-CV/validation evidence.  For each
candidate, this experiment removes every pair of frontier features, screens all
resulting 35-feature subsets with seed-0 CV, and reranks a small beam with seeds
0..2.  The winner is checked on disjoint CV seeds 10..19 and 30..39 plus
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
from itertools import combinations

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from region_shape_ceiling96 import load_pool, load_region_shape  # noqa: E402
from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402

FRONTIER_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 292, 111, 206, 195, 298, 75, 37,
    317, 287, 86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 300, 358,
    314, 289, 65, 32, 365, 2,
])
ADDITION_NAMES = ('tail_aniso_low_pan', 'tail_aniso_low_ndvi')
C = 3.0
GATE = 0.960

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def init_worker(ftr: np.ndarray, ytr: np.ndarray) -> None:
    global _FTR, _YTR
    _FTR, _YTR = ftr, ytr


def score_subset(
    candidate: tuple[tuple[int, int, int], np.ndarray, tuple[int, ...]],
) -> tuple[tuple[int, int, int], float]:
    tag, idx, seeds = candidate
    return tag, mean_cv(_FTR, _YTR, idx, seeds, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--beam', type=int, default=24)
    parser.add_argument('--workers', type=int, default=40)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__),
            'region_shape_prune_ceiling96_35_result.txt',
        ),
    )
    args = parser.parse_args()
    if args.beam < 1:
        parser.error('--beam must be positive')

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, names = load_region_shape('train')
    rva, val_names = load_region_shape('val')
    if names != val_names or rtr.shape[1] != len(names):
        raise ValueError('train/validation region-shape schemas differ')

    pool_dim = ftr.shape[1]
    ftr = np.concatenate((ftr, rtr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    additions = [
        pool_dim + names.index(candidate_name)
        for candidate_name in ADDITION_NAMES
    ]

    lines: list[str] = []

    def emit(message: str = '') -> None:
        print(message, flush=True)
        lines.append(message)

    pair_count = len(FRONTIER_IDX) * (len(FRONTIER_IDX) - 1) // 2
    emit('Region-shape one-for-two exchange below 36-feature frontier')
    emit('screen all drop pairs with seed=0; rerank beam with seeds=0..2; C=3')
    emit('verify A=10..19; verify B=30..39; test gated by both plus validation')
    emit(f'additions={",".join(ADDITION_NAMES)}')
    emit(f'drop_pairs_per_addition={pair_count} beam={args.beam}')

    screen_jobs = []
    for added in additions:
        for dropped_a, dropped_b in combinations(FRONTIER_IDX, 2):
            kept = FRONTIER_IDX[
                (FRONTIER_IDX != dropped_a) & (FRONTIER_IDX != dropped_b)
            ]
            idx = np.append(kept, added)
            screen_jobs.append(
                (
                    (int(added), int(dropped_a), int(dropped_b)),
                    idx,
                    (0,),
                )
            )

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(ftr, ytr),
    ) as executor:
        screen_scores = list(executor.map(score_subset, screen_jobs))
        screen_scores.sort(key=lambda item: (-item[1], item[0]))
        beam = screen_scores[:min(args.beam, len(screen_scores))]

        emit()
        emit('seed-0 screen (top beam)')
        for (added, dropped_a, dropped_b), score in beam:
            emit(
                f'add={names[added-pool_dim]} '
                f'drop={dropped_a},{dropped_b} seed0_cv={score:.4f}'
            )

        rerank_jobs = []
        for (added, dropped_a, dropped_b), _score in beam:
            kept = FRONTIER_IDX[
                (FRONTIER_IDX != dropped_a) & (FRONTIER_IDX != dropped_b)
            ]
            rerank_jobs.append(
                (
                    (added, dropped_a, dropped_b),
                    np.append(kept, added),
                    (0, 1, 2),
                )
            )
        rerank_scores = list(executor.map(score_subset, rerank_jobs))

    rerank_scores.sort(key=lambda item: (-item[1], item[0]))
    (added, dropped_a, dropped_b), select_cv = rerank_scores[0]
    winner = np.append(
        FRONTIER_IDX[
            (FRONTIER_IDX != dropped_a) & (FRONTIER_IDX != dropped_b)
        ],
        added,
    )

    parent_cv = mean_cv(ftr, ytr, FRONTIER_IDX, range(3), C=C)
    verify_a = mean_cv(ftr, ytr, winner, range(10, 20), C=C)
    verify_b = mean_cv(ftr, ytr, winner, range(30, 40), C=C)
    w, b, fi = fit_folded_logreg(ftr, ytr, winner, C=C)
    val = accuracy_score(yva, predict(fva, w, b, fi))

    emit()
    emit(f'baseline_parent_select_cv={parent_cv:.4f}')
    emit(f'added_feature={names[added-pool_dim]}')
    emit(f'dropped_features={dropped_a},{dropped_b}')
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
    emit(','.join(map(str, winner[winner < pool_dim])))
    emit(f'region_shape_feature={names[added-pool_dim]}')

    with open(args.output, 'w') as out:
        out.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
