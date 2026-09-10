"""RESISC45 RGB pool from commit 68480cb7aca0e73289da2ac36ec95faeab020b2d.

The arithmetic and column order match experiments/resisc45_33_feature.py.
Only its feature functions are retained; no research caches or downloads.
"""
from __future__ import annotations

import numpy as np

from experiments.gstruct2_features_lib import (
    _blob_stats, _corner_stats, _lbp_stats, _radial_slope,
)
from experiments.line_features_lib import _hough_line_stats
from src.features import patch_features

_EPS = 1e-6
ARCHIVE_COMMIT = '68480cb7aca0e73289da2ac36ec95faeab020b2d'
SELECTED_INDICES = (
    26, 74, 20, 81, 83, 13, 8, 21, 25, 38, 0, 35, 2, 15, 112, 23, 47,
    85, 33, 128, 45, 27, 98, 7, 14, 105, 107, 34, 76, 32, 129, 24, 31,
)
SELECTED_NAMES = (
    'g0std_b2', 'rgbgm_green_blue', 'p90_b2', 'rgbgs_red_blue',
    'rgbgs_lightness', 'p50_b1', 'p10_b2', 'g0mean_b0', 'g0std_b1',
    'g2std_b2', 'mean_b0', 'g2mean_b2', 'mean_b2', 'p75_b0', 'sspectral_pan',
    'g0mean_b2', 'oent_b2', 'rgbspr_green_red', 'g2mean_b0',
    'cornermag_saturation', 'oent_b0', 'g1mean_b0', 'rgb2gm_green_blue',
    'p10_b1', 'p50_b2', 'cornerfrac_pan', 'lbpent_pan', 'g2mean_b1',
    'rgbgm_saturation', 'g1std_b2', 'lbpent_saturation', 'g0std_b0', 'g1std_b1',
)


def pool2(images: np.ndarray) -> np.ndarray:
    n, c, h, w = images.shape
    return images.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def gradient_magnitude(images: np.ndarray) -> np.ndarray:
    gx = np.diff(images, axis=3)[:, :, :-1, :]
    gy = np.diff(images, axis=2)[:, :, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def rgb_maps(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    red, green, blue = images[:, 0], images[:, 1], images[:, 2]

    def ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left - right) / (left + right + _EPS)

    total = red + green + blue + _EPS
    maximum = images.max(axis=1)
    minimum = images.min(axis=1)
    maps = np.stack((
        (2 * green - red - blue) / total,
        ratio(green, red), ratio(green, blue), ratio(red, blue),
        (maximum - minimum) / (maximum + _EPS),
        (maximum + minimum) / 510.0,
    ), axis=1)
    names = ['exg', 'green_red', 'green_blue', 'red_blue', 'saturation', 'lightness']
    return maps.astype(np.float32), names


def rgb_index_texture(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    maps, map_names = rgb_maps(images)
    n, channels = maps.shape[:2]
    flat = maps.reshape(n, channels, -1)
    gradient = gradient_magnitude(maps).reshape(n, channels, -1)
    p10, p90 = np.percentile(flat, (10, 90), axis=2)
    fine_parts = (
        flat.std(axis=2), gradient.mean(axis=2), gradient.std(axis=2), p90 - p10,
    )
    fine_names = [
        f'{prefix}_{name}'
        for prefix in ('rgbstd', 'rgbgm', 'rgbgs', 'rgbspr')
        for name in map_names
    ]
    coarse_maps, _ = rgb_maps(pool2(images))
    coarse_flat = coarse_maps.reshape(n, channels, -1)
    coarse_gradient = gradient_magnitude(coarse_maps).reshape(n, channels, -1)
    coarse_parts = (coarse_flat.std(axis=2), coarse_gradient.mean(axis=2))
    coarse_names = [
        f'{prefix}_{name}'
        for prefix in ('rgb2std', 'rgb2gm')
        for name in map_names
    ]
    features = np.concatenate((*fine_parts, *coarse_parts), axis=1)
    return features.astype(np.float32), fine_names + coarse_names


def rgb_cross_correlation(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    flat = images.reshape(len(images), 3, -1)
    centered = flat - flat.mean(axis=2, keepdims=True)
    standard_deviation = np.sqrt((centered * centered).mean(axis=2)) + _EPS
    pairs = ((0, 1, 'red_green'), (0, 2, 'red_blue'), (1, 2, 'green_blue'))
    parts, names = [], []
    for left, right, name in pairs:
        covariance = (centered[:, left] * centered[:, right]).mean(axis=1)
        parts.append(covariance / (
            standard_deviation[:, left] * standard_deviation[:, right]
        ))
        names.append(f'rgbcorr_{name}')
    return np.column_stack(parts).astype(np.float32), names


def structural_channels(images: np.ndarray) -> dict[str, np.ndarray]:
    maps, names = rgb_maps(images)
    lookup = {name: maps[:, index] for index, name in enumerate(names)}
    return {
        'pan': images.mean(axis=1),
        'exg': lookup['exg'],
        'saturation': lookup['saturation'],
    }


def rgb_global_structure(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    parts, names = [], []
    for channel_name, channel in structural_channels(images).items():
        values = (
            *_hough_line_stats(channel), *_corner_stats(channel),
            *_lbp_stats(channel), *_blob_stats(channel), _radial_slope(channel),
        )
        prefixes = (
            'linepf', 'linepl', 'linet3', 'cornerfrac', 'cornermag',
            'lbpent', 'lbpuni', 'bloblargest', 'blobcount', 'blobmean', 'sspectral',
        )
        parts.extend(value[:, None] for value in values)
        names.extend(f'{prefix}_{channel_name}' for prefix in prefixes)
    return np.concatenate(parts, axis=1).astype(np.float32), names


def rgb_tail_shape(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    _, _, height, width = images.shape
    coordinate_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    coordinate_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(coordinate_y, coordinate_x, indexing='ij')
    xx, yy = xx.reshape(1, -1), yy.reshape(1, -1)
    parts, names = [], []
    for channel_name, channel in structural_channels(images).items():
        flat = channel.reshape(len(channel), -1)
        q25, q75 = np.percentile(flat, (25, 75), axis=1).astype(np.float32)
        for tail, weights in (
            ('low', np.maximum(q25[:, None] - flat, 0.0)),
            ('high', np.maximum(flat - q75[:, None], 0.0)),
        ):
            mass = weights.sum(axis=1) + _EPS
            mean_x = (weights * xx).sum(axis=1) / mass
            mean_y = (weights * yy).sum(axis=1) / mass
            delta_x, delta_y = xx - mean_x[:, None], yy - mean_y[:, None]
            sxx = (weights * delta_x * delta_x).sum(axis=1) / mass
            syy = (weights * delta_y * delta_y).sum(axis=1) / mass
            sxy = (weights * delta_x * delta_y).sum(axis=1) / mass
            trace = sxx + syy
            anisotropy = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (trace + _EPS)
            parts.extend((anisotropy[:, None], np.sqrt(trace)[:, None]))
            names.extend((
                f'tail_aniso_{tail}_{channel_name}',
                f'tail_spread_{tail}_{channel_name}',
            ))
    return np.concatenate(parts, axis=1).astype(np.float32), names


def rgb_feature_pool(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return the archived 147 columns, including its historical b0/b1/b2 names."""
    images = np.asarray(images, dtype=np.float32)
    if images.ndim != 4 or images.shape[1:] != (3, 64, 64):
        raise ValueError('expected RGB images shaped (N, 3, 64, 64)')
    base = patch_features(
        images, pcts=(10, 25, 50, 75, 90), grad_scales=3, coherence_scales=2,
        orient_entropy_bins=8, orient_hist_bins=4, spectral_peak=True,
    )
    families = (
        base, rgb_cross_correlation(images), rgb_index_texture(images),
        rgb_global_structure(images), rgb_tail_shape(images),
    )
    features = np.concatenate([values for values, _ in families], axis=1).astype(np.float32)
    names = [name for _, family_names in families for name in family_names]
    if features.shape[1] != 147 or len(names) != len(set(names)):
        raise ValueError('RGB feature pool schema changed')
    if tuple(names[i] for i in SELECTED_INDICES) != SELECTED_NAMES:
        raise ValueError('archived RESISC45 feature order changed')
    if not np.isfinite(features).all():
        raise ValueError('RGB feature pool contains non-finite values')
    return features, names
