#!/usr/bin/env python
"""Prescreen spectral periodicity on the 33-feature >96% frontier.

PermanentCrop remains one of the frontier's weakest classes and is most often
confused with HerbaceousVegetation and AnnualCrop. The established pool has raw
band FFT peakiness, but it mixes frequency scales and none of those features
survived into the frontier. This experiment instead normalizes power within
each radial frequency ring, then summarizes periodicity on pan, NDVI, and NDBI.

For each channel it measures the strongest ring-normalized peak, its frequency,
the entropy of the radially whitened spectrum, and rotation-invariant angular
concentration. Each candidate is appended individually to the frozen
33-feature frontier and scored with train-only CV seeds 0..2 plus held-out
validation. Test is never loaded.
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
_EPS = 1e-12
_RMIN = 3
_RMAX = 20

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _periodicity_stats(channel: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Return scale-separated periodicity summaries for one image channel."""
    count, height, width = channel.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height) * height)
    fx = np.fft.fftshift(np.fft.fftfreq(width) * width)
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    radius = np.rint(np.sqrt(xx * xx + yy * yy)).astype(np.int16)
    angle_phase = np.exp(2j * np.arctan2(yy, xx))
    ring_masks = [radius == r for r in range(_RMIN, _RMAX + 1)]
    annulus = (radius >= _RMIN) & (radius <= _RMAX)
    annulus_radii = radius[annulus]
    annulus_phase = angle_phase[annulus]
    annulus_size = int(annulus.sum())
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]

    features = np.empty((count, 4), np.float32)
    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        values = channel[start:stop].astype(np.float64)
        values -= values.mean(axis=(1, 2), keepdims=True)
        power = np.abs(
            np.fft.fftshift(
                np.fft.fft2(values * window[None, :, :]),
                axes=(1, 2),
            )
        ) ** 2

        whitened = np.empty((stop - start, annulus_size), np.float64)
        contrasts = np.empty((stop - start, len(ring_masks)), np.float64)
        for ring_offset, mask in enumerate(ring_masks):
            ring_power = power[:, mask]
            ring_mean = ring_power.mean(axis=1, keepdims=True) + _EPS
            normalized = ring_power / ring_mean
            contrasts[:, ring_offset] = normalized.max(axis=1)
            whitened[:, annulus_radii == ring_offset + _RMIN] = normalized

        strongest = contrasts.argmax(axis=1)
        mass = whitened.sum(axis=1) + _EPS
        probability = whitened / mass[:, None]
        entropy = -(
            probability * np.log(probability + _EPS)
        ).sum(axis=1) / np.log(annulus_size)
        angular = np.abs(whitened @ annulus_phase) / mass

        features[start:stop, 0] = np.log1p(
            contrasts[np.arange(stop - start), strongest]
        )
        features[start:stop, 1] = (
            strongest + _RMIN
        ) / _RMAX
        features[start:stop, 2] = entropy
        features[start:stop, 3] = angular

    return features


def spectral_periodicity(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Periodic texture summaries on pan, NDVI, and NDBI channels."""
    nir, red = imgs[:, B_NIR], imgs[:, B_RED]
    swir1 = imgs[:, B_SWIR1]
    channels = (
        ('pan', imgs.mean(axis=1)),
        ('ndvi', (nir - red) / (nir + red + 1e-6)),
        ('ndbi', (swir1 - nir) / (swir1 + nir + 1e-6)),
    )
    parts: list[np.ndarray] = []
    names: list[str] = []
    for channel_name, channel in channels:
        parts.append(_periodicity_stats(channel))
        names.extend((
            f'fft_ring_peak_{channel_name}',
            f'fft_ring_scale_{channel_name}',
            f'fft_white_entropy_{channel_name}',
            f'fft_angular_aniso_{channel_name}',
        ))

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('spectral-periodicity features contain non-finite values')
    return features, names


def load_spectral_periodicity(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_spectral_periodicity.npy')
    names = [
        f'fft_{stat}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for stat in ('ring_peak', 'ring_scale', 'white_entropy', 'angular_aniso')
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = spectral_periodicity(imgs)
    if computed_names != names:
        raise ValueError('spectral-periodicity feature order does not match cache schema')
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
            os.path.dirname(__file__),
            'spectral_periodicity_ceiling96_result.txt',
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    ptr, names = load_spectral_periodicity('train')
    pva, val_names = load_spectral_periodicity('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or ptr.shape[1] != len(names):
        raise ValueError('train/validation spectral-periodicity schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    periodicity_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, ptr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, pva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, periodicity_start + offset)
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
        'Spectral-periodicity prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>31} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>31} {cv_score:>7.4f} '
            f'{cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
