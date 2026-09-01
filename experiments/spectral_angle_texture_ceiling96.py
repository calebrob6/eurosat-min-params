#!/usr/bin/env python
"""Prescreen spectral-angle edge texture on the 33-feature >96% frontier.

The existing pool measures spatial gradients one band or spectral index at a
time, while the spectral-mixture experiment discards where different material
spectra occur. Neither captures brightness-invariant transitions between the
full spectra of neighboring pixels. Those transitions should distinguish a
road crossing vegetation from brightness variation within one material, and
sharp crop boundaries from diffuse herbaceous texture.

For all 13 bands and the six principal optical bands, this experiment measures
the mean, standard deviation, 90th percentile, and tail-to-mean ratio of
neighboring-pixel spectral angles at 1x, 2x, and 4x spatial scales. Each
candidate is appended individually to the frozen 33-feature frontier and
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
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from region_shape_ceiling96 import load_pool, load_region_shape  # noqa: E402
from src.cache import CACHE_DIR, load_cached  # noqa: E402
from src.data import B_BLUE, B_GREEN, B_NIR, B_RED, B_SWIR1, B_SWIR2  # noqa: E402
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
SPECTRAL_SETS = (
    ('all', tuple(range(13))),
    ('key6', (B_BLUE, B_GREEN, B_RED, B_NIR, B_SWIR1, B_SWIR2)),
)
SCALES = (1, 2, 4)
STAT_NAMES = ('mean', 'std', 'p90', 'tail_ratio')
_EPS = 1e-6

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _pool2(imgs: np.ndarray) -> np.ndarray:
    count, channels, height, width = imgs.shape
    return imgs.reshape(
        count, channels, height // 2, 2, width // 2, 2
    ).mean(axis=(3, 5))


def _spectral_angle_stats(
    imgs: np.ndarray,
    bands: tuple[int, ...],
    chunk: int = 256,
) -> np.ndarray:
    """Summarize neighboring-pixel spectral angles at three spatial scales."""
    count = imgs.shape[0]
    features = np.empty(
        (count, len(SCALES) * len(STAT_NAMES)),
        dtype=np.float32,
    )

    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        current = imgs[start:stop, bands].astype(np.float32)
        for scale_offset, _ in enumerate(SCALES):
            spectra = current.transpose(0, 2, 3, 1)
            norm = np.sqrt(np.sum(spectra * spectra, axis=3, keepdims=True))
            unit = spectra / (norm + _EPS)
            horizontal = np.arccos(np.clip(
                np.sum(unit[:, :, 1:] * unit[:, :, :-1], axis=3),
                -1.0,
                1.0,
            ))
            vertical = np.arccos(np.clip(
                np.sum(unit[:, 1:] * unit[:, :-1], axis=3),
                -1.0,
                1.0,
            ))
            angles = np.concatenate((
                horizontal.reshape(stop - start, -1),
                vertical.reshape(stop - start, -1),
            ), axis=1)
            mean = angles.mean(axis=1)
            p90 = np.percentile(angles, 90, axis=1)
            offset = scale_offset * len(STAT_NAMES)
            features[start:stop, offset] = mean
            features[start:stop, offset + 1] = angles.std(axis=1)
            features[start:stop, offset + 2] = p90
            features[start:stop, offset + 3] = p90 / (mean + _EPS)
            if scale_offset + 1 < len(SCALES):
                current = _pool2(current)

    return features


def spectral_angle_texture(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Multiscale brightness-invariant spectral edge summaries."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    for set_name, bands in SPECTRAL_SETS:
        parts.append(_spectral_angle_stats(imgs, bands))
        names.extend(
            f'sam_edge_{stat}_s{scale}_{set_name}'
            for scale in SCALES
            for stat in STAT_NAMES
        )

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('spectral-angle texture contains non-finite values')
    return features, names


def load_spectral_angle_texture(
    split: str,
) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_spectral_angle_texture.npy')
    names = [
        f'sam_edge_{stat}_s{scale}_{set_name}'
        for set_name, _ in SPECTRAL_SETS
        for scale in SCALES
        for stat in STAT_NAMES
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = spectral_angle_texture(imgs)
    if computed_names != names:
        raise ValueError('spectral-angle feature order does not match cache schema')
    np.save(path, features)
    return features, names


def init_worker(ftr: np.ndarray, ytr: np.ndarray) -> None:
    global _FTR, _YTR
    _FTR, _YTR = ftr, ytr


def score_candidate(idx: np.ndarray) -> float:
    return mean_cv(_FTR, _YTR, idx, SELECT_SEEDS, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=25)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__),
            'spectral_angle_texture_ceiling96_result.txt',
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    atr, names = load_spectral_angle_texture('train')
    ava, val_names = load_spectral_angle_texture('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or atr.shape[1] != len(names):
        raise ValueError('train/validation spectral-angle schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    angle_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, atr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, ava), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, angle_start + offset)
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
        'Spectral-angle texture prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>36} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>36} {cv_score:>7.4f} '
            f'{cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
