#!/usr/bin/env python
"""Prescreen spectral-tail morphology on the 33-feature >96% frontier.

The existing frontier and candidate families describe tail-region anisotropy,
connectivity, boundary density, and straight lines, but not region thickness.
A narrow Highway strip and a broad crop region can have similar elongation and
boundary statistics while responding very differently to morphological
opening.

For low and high quartile masks of pan, NDVI, and NDBI, this experiment measures
the fraction of tail pixels retained after opening with disks of radius 1, 2,
and 4. Each candidate is appended individually to the frozen 33-feature
frontier and scored with train-only CV seeds 0..2 plus held-out validation.
Test is never loaded.
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
from scipy.ndimage import binary_opening
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from region_shape_ceiling96 import load_pool, load_region_shape  # noqa: E402
from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.data import B_NIR, B_RED, B_SWIR1  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import mean_cv  # noqa: E402

POOL_IDX = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 206, 195, 298, 75, 37, 317, 287,
    86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 314, 289, 65, 32, 365,
    2,
])
REGION_SHAPE_NAME = 'tail_aniso_low_ndvi'
C = 3.0
SELECT_SEEDS = (0, 1, 2)
RADII = (1, 2, 4)
_EPS = 1e-6

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _disk(radius: int) -> np.ndarray:
    coords = np.arange(-radius, radius + 1)
    yy, xx = np.meshgrid(coords, coords, indexing='ij')
    return (xx * xx + yy * yy <= radius * radius)[None, :, :]


def _opening_survival(masks: np.ndarray) -> list[np.ndarray]:
    """Return the foreground fraction retained at each opening radius."""
    mass = masks.sum(axis=(1, 2)) + _EPS
    return [
        (
            binary_opening(masks, structure=_disk(radius)).sum(axis=(1, 2))
            / mass
        ).astype(np.float32)
        for radius in RADII
    ]


def morphological_profile(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Opening granulometry of low/high pan, NDVI, and NDBI quartile masks."""
    nir, red = imgs[:, B_NIR], imgs[:, B_RED]
    swir1 = imgs[:, B_SWIR1]
    channels = (
        ('pan', imgs.mean(axis=1)),
        ('ndvi', (nir - red) / (nir + red + _EPS)),
        ('ndbi', (swir1 - nir) / (swir1 + nir + _EPS)),
    )
    parts: list[np.ndarray] = []
    names: list[str] = []

    for channel_name, channel in channels:
        flat = channel.reshape(channel.shape[0], -1)
        q25, q75 = np.percentile(flat, (25, 75), axis=1)
        for tail, masks in (
            ('low', channel < q25[:, None, None]),
            ('high', channel > q75[:, None, None]),
        ):
            survival = _opening_survival(masks)
            parts.extend(value[:, None] for value in survival)
            names.extend(
                f'tail_open_r{radius}_{tail}_{channel_name}'
                for radius in RADII
            )

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('morphological-profile features contain non-finite values')
    return features, names


def load_morphological_profile(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_morphological_profile.npy')
    names = [
        f'tail_open_r{radius}_{tail}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for tail in ('low', 'high')
        for radius in RADII
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = morphological_profile(imgs)
    if computed_names != names:
        raise ValueError('morphological-profile feature order does not match schema')
    np.save(path, features)
    return features, names


def init_worker(ftr: np.ndarray, ytr: np.ndarray) -> None:
    global _FTR, _YTR
    _FTR, _YTR = ftr, ytr


def score_candidate(idx: np.ndarray) -> float:
    return mean_cv(_FTR, _YTR, idx, SELECT_SEEDS, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=19)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__),
            'morphological_profile_ceiling96_result.txt',
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    mtr, names = load_morphological_profile('train')
    mva, val_names = load_morphological_profile('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or mtr.shape[1] != len(names):
        raise ValueError('train/validation morphological-profile schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    morphology_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, mtr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, mva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, morphology_start + offset)
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
        'Morphological-profile prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>34} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>34} {cv_score:>7.4f} '
            f'{cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
