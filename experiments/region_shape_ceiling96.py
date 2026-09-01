#!/usr/bin/env python
"""Prescreen spectral-tail region shape against the 36-feature >96% frontier.

The current pool measures region size, local edge direction, and global straight
lines, but not the shape of spectrally distinct regions.  A Highway's low-NDVI
pixels should form an elongated strip, unlike the more compact low-NDVI regions
in fields or industrial scenes.  This experiment summarizes the spatial second
moments of the low and high quartile tails of pan, NDVI, and NDBI.

Each candidate is appended individually to the frozen 36-feature frontier and
scored with train-only CV seeds 0..2 plus held-out validation.  Test is never
loaded: this is only a bounded family prescreen for a later pruning search.
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

from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.data import B_NIR, B_RED, B_SWIR1  # noqa: E402
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
FRONTIER_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 292, 111, 206, 195, 298, 75, 37,
    317, 287, 86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 300, 358,
    314, 289, 65, 32, 365, 2,
])
C = 3.0
SELECT_SEEDS = (0, 1, 2)
_EPS = 1e-6

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def load_pool(split: str) -> np.ndarray:
    parts = [
        np.load(os.path.join(CACHE_DIR, f'{split}_{family}.npy'))
        for family in FAMILIES
    ]
    return np.concatenate(parts, axis=1).astype(np.float32)


def tail_region_shape(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Spatial anisotropy and spread of low/high channel-value tails."""
    _, _, height, width = imgs.shape
    coords_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    coords_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(coords_y, coords_x, indexing='ij')
    xx = xx.reshape(1, -1)
    yy = yy.reshape(1, -1)
    parts: list[np.ndarray] = []
    names: list[str] = []

    for channel_name in ('pan', 'ndvi', 'ndbi'):
        if channel_name == 'pan':
            channel = imgs.mean(axis=1)
        elif channel_name == 'ndvi':
            channel = (
                (imgs[:, B_NIR] - imgs[:, B_RED])
                / (imgs[:, B_NIR] + imgs[:, B_RED] + _EPS)
            )
        else:
            channel = (
                (imgs[:, B_SWIR1] - imgs[:, B_NIR])
                / (imgs[:, B_SWIR1] + imgs[:, B_NIR] + _EPS)
            )

        flat = channel.reshape(channel.shape[0], -1)
        q25, q75 = np.percentile(flat, (25, 75), axis=1).astype(np.float32)
        for tail, weights in (
            ('low', np.maximum(q25[:, None] - flat, 0.0)),
            ('high', np.maximum(flat - q75[:, None], 0.0)),
        ):
            mass = weights.sum(axis=1) + _EPS
            mean_x = (weights * xx).sum(axis=1) / mass
            mean_y = (weights * yy).sum(axis=1) / mass
            dx = xx - mean_x[:, None]
            dy = yy - mean_y[:, None]
            sxx = (weights * dx * dx).sum(axis=1) / mass
            syy = (weights * dy * dy).sum(axis=1) / mass
            sxy = (weights * dx * dy).sum(axis=1) / mass
            trace = sxx + syy
            anisotropy = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (
                trace + _EPS
            )
            spread = np.sqrt(trace)
            parts.extend((anisotropy[:, None], spread[:, None]))
            names.extend((
                f'tail_aniso_{tail}_{channel_name}',
                f'tail_spread_{tail}_{channel_name}',
            ))

        del channel, flat

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('tail-region features contain non-finite values')
    return features, names


def load_region_shape(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_region_shape.npy')
    names = [
        f'tail_{stat}_{tail}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for tail in ('low', 'high')
        for stat in ('aniso', 'spread')
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = tail_region_shape(imgs)
    if computed_names != names:
        raise ValueError('region-shape feature order does not match cache schema')
    np.save(path, features)
    return features, names


def init_worker(ftr: np.ndarray, ytr: np.ndarray) -> None:
    global _FTR, _YTR
    _FTR, _YTR = ftr, ytr


def score_candidate(idx: np.ndarray) -> float:
    return mean_cv(_FTR, _YTR, idx, SELECT_SEEDS, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=13)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'region_shape_ceiling96_result.txt'
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
    ftr = np.concatenate((ftr, rtr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [FRONTIER_IDX]
    candidates.extend(
        np.append(FRONTIER_IDX, pool_dim + offset)
        for offset in range(len(names))
    )
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=init_worker,
        initargs=(ftr, ytr),
    ) as executor:
        cv_scores = list(executor.map(score_candidate, candidates))

    val_scores = []
    for idx in candidates:
        w, b, fi = fit_folded_logreg(ftr, ytr, idx, C=C)
        val_scores.append(accuracy_score(yva, predict(fva, w, b, fi)))

    baseline_cv = cv_scores[0]
    baseline_val = val_scores[0]
    rows = sorted(
        zip(names, cv_scores[1:], val_scores[1:]),
        key=lambda row: (-row[1], -row[2], row[0]),
    )

    lines = [
        'Spectral-tail region-shape prescreen on frozen 36-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>31} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>31} {cv_score:>7.4f} {cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
