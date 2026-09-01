#!/usr/bin/env python
"""Prescreen scale-separated spectral anisotropy on the 33-feature frontier.

The all-scale NDBI angular FFT anisotropy improved both train-only CV and
validation, but may average together unrelated coarse boundaries and fine crop
rows. This experiment measures rotation-invariant second- and fourth-order
angular concentration in three radial frequency bands on pan, NDVI, and NDBI.
Ring whitening prevents a channel's radial power profile from dominating the
directional summaries.

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
RADIAL_BANDS = (
    ('coarse', 3, 7),
    ('mid', 8, 13),
    ('fine', 14, 20),
)
HARMONICS = (2, 4)
_EPS = 1e-12

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _band_anisotropy(channel: np.ndarray, chunk: int = 512) -> np.ndarray:
    """Return angular power concentration by radial band and harmonic."""
    count, height, width = channel.shape
    fy = np.fft.fftshift(np.fft.fftfreq(height) * height)
    fx = np.fft.fftshift(np.fft.fftfreq(width) * width)
    yy, xx = np.meshgrid(fy, fx, indexing='ij')
    radius = np.rint(np.sqrt(xx * xx + yy * yy)).astype(np.int16)
    angle = np.arctan2(yy, xx)
    ring_masks = {
        r: radius == r
        for r in range(RADIAL_BANDS[0][1], RADIAL_BANDS[-1][2] + 1)
    }
    band_masks = [
        (radius >= low) & (radius <= high)
        for _, low, high in RADIAL_BANDS
    ]
    phases = {
        harmonic: np.exp(1j * harmonic * angle)
        for harmonic in HARMONICS
    }
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]

    features = np.empty(
        (count, len(RADIAL_BANDS) * len(HARMONICS)),
        np.float32,
    )
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

        whitened = np.empty_like(power)
        whitened.fill(0.0)
        for mask in ring_masks.values():
            ring_power = power[:, mask]
            whitened[:, mask] = ring_power / (
                ring_power.mean(axis=1, keepdims=True) + _EPS
            )

        offset = 0
        for mask in band_masks:
            band_power = whitened[:, mask]
            mass = band_power.sum(axis=1) + _EPS
            for harmonic in HARMONICS:
                features[start:stop, offset] = (
                    np.abs(band_power @ phases[harmonic][mask]) / mass
                )
                offset += 1

    return features


def spectral_band_anisotropy(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Scale-separated angular FFT concentration on pan, NDVI, and NDBI."""
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
        parts.append(_band_anisotropy(channel))
        names.extend(
            f'fft_band_h{harmonic}_{band_name}_{channel_name}'
            for band_name, _, _ in RADIAL_BANDS
            for harmonic in HARMONICS
        )

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('spectral-band-anisotropy features contain non-finite values')
    return features, names


def load_spectral_band_anisotropy(
    split: str,
) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_spectral_band_anisotropy.npy')
    names = [
        f'fft_band_h{harmonic}_{band_name}_{channel}'
        for channel in ('pan', 'ndvi', 'ndbi')
        for band_name, _, _ in RADIAL_BANDS
        for harmonic in HARMONICS
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = spectral_band_anisotropy(imgs)
    if computed_names != names:
        raise ValueError(
            'spectral-band-anisotropy feature order does not match cache schema'
        )
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
            'spectral_band_anisotropy_ceiling96_result.txt',
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    atr, names = load_spectral_band_anisotropy('train')
    ava, val_names = load_spectral_band_anisotropy('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or atr.shape[1] != len(names):
        raise ValueError('train/validation spectral-band schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    anisotropy_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, atr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, ava), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, anisotropy_start + offset)
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
        'Spectral-band anisotropy prescreen on frozen 33-feature frontier',
        'train-only select CV seeds=0..2; C=3; test not loaded',
        f'baseline select_cv={baseline_cv:.4f} validation={baseline_val:.4f}',
        '',
        f'{"candidate":>35} {"selCV":>7} {"dCV":>8} '
        f'{"val":>7} {"dval":>8}',
    ]
    for name, cv_score, val_score in rows:
        lines.append(
            f'{name:>35} {cv_score:>7.4f} '
            f'{cv_score-baseline_cv:>+8.4f} '
            f'{val_score:>7.4f} {val_score-baseline_val:>+8.4f}'
        )

    output = '\n'.join(lines) + '\n'
    print(output, end='', flush=True)
    with open(args.output, 'w') as out:
        out.write(output)


if __name__ == '__main__':
    main()
