#!/usr/bin/env python
"""Prescreen spectral-mixture geometry on the 33-feature >96% frontier.

The frontier describes each band and spectral-index map separately, but does
not measure how many distinct spectral materials coexist within a patch. This
is relevant to Highway and PermanentCrop, the two weakest validation classes:
a road crossing vegetation or a mixed crop patch can have similar marginal
statistics but different per-pixel spectral shapes.

This experiment L2-normalizes each pixel spectrum to remove brightness, then
summarizes its angular dispersion around the patch mean and the eigenspectrum
of the within-patch spectral-shape covariance. The statistics are computed for
all 13 bands and for the six principal optical bands.

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
STAT_NAMES = (
    'angle_mean',
    'angle_std',
    'angle_p90',
    'shape_var',
    'eig1_frac',
    'eig2_frac',
    'eig_entropy',
)
_EPS = 1e-12

_FTR: np.ndarray | None = None
_YTR: np.ndarray | None = None


def _spectral_mixture_stats(
    imgs: np.ndarray,
    bands: tuple[int, ...],
    chunk: int = 256,
) -> np.ndarray:
    """Return angular dispersion and covariance-spectrum shape statistics."""
    count = imgs.shape[0]
    features = np.empty((count, len(STAT_NAMES)), np.float32)
    entropy_scale = np.log(len(bands))

    for start in range(0, count, chunk):
        stop = min(start + chunk, count)
        spectra = imgs[start:stop, bands].transpose(0, 2, 3, 1)
        spectra = spectra.reshape(stop - start, -1, len(bands)).astype(np.float64)
        unit = spectra / (
            np.linalg.norm(spectra, axis=2, keepdims=True) + _EPS
        )
        mean_shape = unit.mean(axis=1)
        mean_direction = mean_shape / (
            np.linalg.norm(mean_shape, axis=1, keepdims=True) + _EPS
        )
        cosine = np.einsum('npc,nc->np', unit, mean_direction)
        angles = np.arccos(np.clip(cosine, -1.0, 1.0))

        centered = unit - mean_shape[:, None, :]
        covariance = np.einsum(
            'npc,npd->ncd', centered, centered, optimize=True
        ) / unit.shape[1]
        eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
        shape_var = eigenvalues.sum(axis=1)
        probability = eigenvalues / (shape_var[:, None] + _EPS)
        eig_entropy = -(
            probability * np.log(probability + _EPS)
        ).sum(axis=1) / entropy_scale

        features[start:stop, 0] = angles.mean(axis=1)
        features[start:stop, 1] = angles.std(axis=1)
        features[start:stop, 2] = np.percentile(angles, 90, axis=1)
        features[start:stop, 3] = shape_var
        features[start:stop, 4] = eigenvalues[:, -1] / (shape_var + _EPS)
        features[start:stop, 5] = eigenvalues[:, -2] / (shape_var + _EPS)
        features[start:stop, 6] = eig_entropy

    return features


def spectral_mixture_features(
    imgs: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """Brightness-invariant spectral-mixture summaries for two band sets."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    for set_name, bands in SPECTRAL_SETS:
        parts.append(_spectral_mixture_stats(imgs, bands))
        names.extend(f'spectral_{stat}_{set_name}' for stat in STAT_NAMES)

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError('spectral-mixture features contain non-finite values')
    return features, names


def load_spectral_mixture(split: str) -> tuple[np.ndarray, list[str]]:
    path = os.path.join(CACHE_DIR, f'{split}_spectral_mixture.npy')
    names = [
        f'spectral_{stat}_{set_name}'
        for set_name, _ in SPECTRAL_SETS
        for stat in STAT_NAMES
    ]
    if os.path.exists(path):
        return np.load(path), names
    imgs, _ = load_cached(split)
    features, computed_names = spectral_mixture_features(imgs)
    if computed_names != names:
        raise ValueError('spectral-mixture feature order does not match cache schema')
    np.save(path, features)
    return features, names


def init_worker(ftr: np.ndarray, ytr: np.ndarray) -> None:
    global _FTR, _YTR
    _FTR, _YTR = ftr, ytr


def score_candidate(idx: np.ndarray) -> float:
    return mean_cv(_FTR, _YTR, idx, SELECT_SEEDS, C=C)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=15)
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'spectral_mixture_ceiling96_result.txt'
        ),
    )
    args = parser.parse_args()

    ftr = load_pool('train')
    fva = load_pool('val')
    rtr, region_names = load_region_shape('train')
    rva, val_region_names = load_region_shape('val')
    mtr, names = load_spectral_mixture('train')
    mva, val_names = load_spectral_mixture('val')
    if region_names != val_region_names:
        raise ValueError('train/validation region-shape schemas differ')
    if names != val_names or mtr.shape[1] != len(names):
        raise ValueError('train/validation spectral-mixture schemas differ')

    pool_dim = ftr.shape[1]
    region_idx = pool_dim + region_names.index(REGION_SHAPE_NAME)
    mixture_start = pool_dim + rtr.shape[1]
    frontier_idx = np.append(POOL_IDX, region_idx)
    ftr = np.concatenate((ftr, rtr, mtr), axis=1).astype(np.float32)
    fva = np.concatenate((fva, rva, mva), axis=1).astype(np.float32)
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))

    candidates = [frontier_idx]
    candidates.extend(
        np.append(frontier_idx, mixture_start + offset)
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
        'Spectral-mixture prescreen on frozen 33-feature frontier',
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
