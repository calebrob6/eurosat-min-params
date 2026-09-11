"""Named NumPy feature pools used by EuroSAT experiments."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .extra_features import (
    blob_features,
    index_coherence,
    index_texture_scale2,
    lbp_features,
    orient_entropy_scale2,
    sslope_features,
    tail_region_shape,
    xband_corr,
)
from .features import patch_features
from .frontier import CORE_CONFIG

FeatureFunction = Callable[[np.ndarray], tuple[np.ndarray, list[str]]]

CORE_GROUPS = (
    ('spectral_statistics', 91),
    ('multiscale_gradients', 78),
    ('coherence', 26),
    ('orientation_entropy', 13),
    ('orientation_histogram', 52),
    ('spectral_peaks', 13),
    ('cross_band', 8),
    ('index_texture', 24),
    ('hough_lines', 9),
    ('harris_corners', 6),
)
EXTRA_FAMILIES: tuple[tuple[str, FeatureFunction, int], ...] = (
    ('lbp', lbp_features, 6),
    ('blobs', blob_features, 9),
    ('spectral_slope', sslope_features, 3),
    ('index_coherence', index_coherence, 6),
    ('index_texture_scale2', index_texture_scale2, 12),
    ('cross_band_correlation', xband_corr, 8),
    ('orientation_entropy_scale2', orient_entropy_scale2, 13),
)
HISTORICAL_POOL_SIZE = 377
REGION_SHAPE_SIZE = 12
FULL_POOL_SIZE = HISTORICAL_POOL_SIZE + REGION_SHAPE_SIZE


def historical_pool_features(
    images: np.ndarray, core: tuple[np.ndarray, list[str]] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute the historical 377-feature pool and its column schema."""
    if (
        images.dtype != np.float32
        or images.ndim != 4
        or images.shape[1:] != (13, 64, 64)
    ):
        raise ValueError('expected float32 TIFF-order images of shape (N,13,64,64)')
    core_values, core_names = core or patch_features(images, **CORE_CONFIG)
    if core_values.shape != (len(images), 320) or len(core_names) != 320:
        raise ValueError('core schema is no longer 320 columns')
    parts = [core_values]
    names = list(core_names)
    families = [group for group, width in CORE_GROUPS for _ in range(width)]
    for group, function, width in EXTRA_FAMILIES:
        values, family_names = function(images)
        if values.shape != (len(images), width) or len(family_names) != width:
            raise ValueError(f'{group}: unexpected feature width')
        parts.append(values)
        names.extend(family_names)
        families.extend([group] * width)
    features = np.concatenate(parts, axis=1).astype(np.float32)
    if (
        features.shape != (len(images), HISTORICAL_POOL_SIZE)
        or len(names) != HISTORICAL_POOL_SIZE
        or len(families) != HISTORICAL_POOL_SIZE
        or not np.isfinite(features).all()
    ):
        raise ValueError('invalid historical feature pool')
    return features, names, families


def full_pool_features(
    images: np.ndarray, core: tuple[np.ndarray, list[str]] | None = None
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute all 389 historical and region-shape candidate features."""
    historical, names, families = historical_pool_features(images, core)
    region, region_names = tail_region_shape(images)
    if (
        region.shape != (len(images), REGION_SHAPE_SIZE)
        or len(region_names) != REGION_SHAPE_SIZE
    ):
        raise ValueError('region_shape: unexpected feature width')
    features = np.concatenate((historical, region), axis=1).astype(np.float32)
    names.extend(region_names)
    families.extend(['region_shape'] * REGION_SHAPE_SIZE)
    if (
        features.shape != (len(images), FULL_POOL_SIZE)
        or len(names) != FULL_POOL_SIZE
        or not np.isfinite(features).all()
    ):
        raise ValueError('invalid full feature pool')
    return features, names, families
