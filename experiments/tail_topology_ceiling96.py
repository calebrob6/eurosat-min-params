#!/usr/bin/env python
"""Prescreen spectral-tail topology against the 33-feature >96% frontier.

The frontier's low-NDVI tail anisotropy measures whether spectrally distinct
pixels form an elongated distribution, but not whether that distribution is one
continuous region or many disconnected fragments. A Highway should often have
one connected low-NDVI strip, while crop and herbaceous patches can have several
separate low-NDVI regions.

For the low and high quartile masks of pan, NDVI, and NDBI, this experiment
measures largest-component fraction, component count, and boundary density.
Each candidate is appended individually to the frozen 33-feature frontier and
scored with train-only CV seeds 0..2 plus held-out validation. Test is never
loaded.
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
from scipy.ndimage import label
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
_EPS = 1e-6

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _mask_topology(masks: np.ndarray) -> tuple[np.ndarray, ...]:
    """Summarize connectivity and fragmentation of binary masks."""
    count = masks.shape[0]
    largest = np.empty(count, np.float32)
    components = np.empty(count, np.float32)
    boundary = np.empty(count, np.float32)
    eight_connected = np.ones((3, 3), np.int8)

    for i, mask in enumerate(masks):
        regions, num_regions = label(mask, structure=eight_connected)
        mass = float(mask.sum()) + _EPS
        sizes = np.bincount(regions.ravel())[1:]
        largest[i] = (sizes.max() / mass) if num_regions else 0.0
        components[i] = np.log1p(num_regions)
        transitions = (
            np.count_nonzero(mask[:, 1:] != mask[:, :-1])
            + np.count_nonzero(mask[1:, :] != mask[:-1, :])
            + np.count_nonzero(mask[0])
            + np.count_nonzero(mask[-1])
            + np.count_nonzero(mask[:, 0])
            + np.count_nonzero(mask[:, -1])
        )
        boundary[i] = transitions / mass

    return largest, components, boundary


def tail_topology(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Topology of low/high pan, NDVI, and NDBI quartile masks."""
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
            largest, components, boundary = _mask_topology(masks)
            parts.extend((
                largest[:, None],
                components[:, None],
                boundary[:, None],
            ))
            names.extend((
                f'tail_largest_{tail}_{channel_name}',
                f'tail_components_{tail}_{channel_name}',
                f'tail_boundary_{tail}_{channel_name}',
            ))

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('tail-topology features contain non-finite values')
    return features, names


def load_tail_topology(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_tail_topology.npy')
    names = [
        f'tail_{stat}_{tail}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for tail in ('low', 'high')
        for stat in ('largest', 'components', 'boundary')
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = tail_topology(imgs)
    if computed_names != names:
        raise ValueError('tail-topology feature order does not match cache schema')
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
            os.path.dirname(__file__), 'tail_topology_ceiling96_result.txt'
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    ttr, names = load_tail_topology('train')
    tva, val_names = load_tail_topology('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or ttr.shape[1] != len(names):
        raise ValueError('train/validation tail-topology schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    topology_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, ttr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, tva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, topology_start + offset)
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
        'Spectral-tail topology prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>34} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>34} {cv_score:>7.4f} {cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
