#!/usr/bin/env python
"""Prescreen spectral-tail component shape on the 33-feature >96% frontier.

The frontier's low-NDVI anisotropy measures the spatial second moment of an
entire spectral tail. It cannot distinguish one elongated region from several
separated compact regions with the same global distribution. Existing topology
features measure component counts and sizes but discard each component's shape.

For low and high quartile masks of pan, NDVI, and NDBI, this experiment measures
the largest component's anisotropy, the area-weighted mean component anisotropy,
and the largest component's bounding-box fill. Each candidate is appended
individually to the frozen 33-feature frontier and scored with train-only CV
seeds 0..2 plus held-out validation. Test is never loaded.
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


def _component_shape(masks: np.ndarray) -> tuple[np.ndarray, ...]:
    """Summarize the geometry of connected foreground components."""
    count, height, width = masks.shape
    largest_aniso = np.empty(count, np.float32)
    weighted_aniso = np.empty(count, np.float32)
    largest_fill = np.empty(count, np.float32)
    yy, xx = np.indices((height, width), dtype=np.float32)
    xx = xx.ravel()
    yy = yy.ravel()
    eight_connected = np.ones((3, 3), np.int8)

    for i, mask in enumerate(masks):
        regions, num_regions = label(mask, structure=eight_connected)
        if num_regions == 0:
            largest_aniso[i] = 0.0
            weighted_aniso[i] = 0.0
            largest_fill[i] = 0.0
            continue

        region_ids = regions.ravel()
        sizes = np.bincount(region_ids, minlength=num_regions + 1)[1:].astype(
            np.float32
        )
        sum_x = np.bincount(
            region_ids, weights=xx, minlength=num_regions + 1
        )[1:]
        sum_y = np.bincount(
            region_ids, weights=yy, minlength=num_regions + 1
        )[1:]
        sum_xx = np.bincount(
            region_ids, weights=xx * xx, minlength=num_regions + 1
        )[1:]
        sum_yy = np.bincount(
            region_ids, weights=yy * yy, minlength=num_regions + 1
        )[1:]
        sum_xy = np.bincount(
            region_ids, weights=xx * yy, minlength=num_regions + 1
        )[1:]

        mean_x = sum_x / sizes
        mean_y = sum_y / sizes
        sxx = np.maximum(sum_xx / sizes - mean_x * mean_x, 0.0)
        syy = np.maximum(sum_yy / sizes - mean_y * mean_y, 0.0)
        sxy = sum_xy / sizes - mean_x * mean_y
        trace = sxx + syy
        anisotropy = np.sqrt((sxx - syy) ** 2 + 4.0 * sxy * sxy) / (
            trace + _EPS
        )

        largest = int(np.argmax(sizes))
        largest_label = largest + 1
        largest_aniso[i] = anisotropy[largest]
        weighted_aniso[i] = np.sum(anisotropy * sizes) / np.sum(sizes)
        ly, lx = np.nonzero(regions == largest_label)
        bbox_area = (ly.max() - ly.min() + 1) * (lx.max() - lx.min() + 1)
        largest_fill[i] = sizes[largest] / bbox_area

    return largest_aniso, weighted_aniso, largest_fill


def tail_component_shape(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Component geometry of low/high pan, NDVI, and NDBI quartile masks."""
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
            largest_aniso, weighted_aniso, largest_fill = _component_shape(masks)
            parts.extend((
                largest_aniso[:, None],
                weighted_aniso[:, None],
                largest_fill[:, None],
            ))
            names.extend((
                f'tail_component_largest_aniso_{tail}_{channel_name}',
                f'tail_component_weighted_aniso_{tail}_{channel_name}',
                f'tail_component_largest_fill_{tail}_{channel_name}',
            ))

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('tail-component-shape features contain non-finite values')
    return features, names


def load_component_shape(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_component_shape.npy')
    names = [
        f'tail_component_{stat}_{tail}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for tail in ('low', 'high')
        for stat in ('largest_aniso', 'weighted_aniso', 'largest_fill')
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = tail_component_shape(imgs)
    if computed_names != names:
        raise ValueError('component-shape feature order does not match cache schema')
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
            os.path.dirname(__file__), 'component_shape_ceiling96_result.txt'
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    ctr, names = load_component_shape('train')
    cva, val_names = load_component_shape('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or ctr.shape[1] != len(names):
        raise ValueError('train/validation component-shape schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    component_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, ctr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, cva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, component_start + offset)
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
        'Spectral-tail component-shape prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>49} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>49} {cv_score:>7.4f} '
            f'{cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
