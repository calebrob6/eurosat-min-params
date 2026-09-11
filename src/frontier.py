"""Frozen EuroSAT feature recipe for the published 306-value head.

The historical numeric channel choices are intentional: some spectral-index
aliases do not match physical bands, and changing them invalidates the
checked-in classifier.
"""
from __future__ import annotations

import numpy as np

from .extra_features import (
    blob_features, index_texture_scale2, lbp_features,
    orient_entropy_scale2, tail_region_shape,
)
from .features import patch_features

RECIPE = 'eurosat306-v1-historical-channels'
CORE_CONFIG = dict(
    pcts=(10, 25, 50, 75, 90), grad_scales=3, coherence_scales=2,
    orient_entropy_bins=8, orient_hist_bins=4, spectral_peak=True,
    xband=True, index_texture=True, hough_lines=True, harris_corners=True,
)
# Indices into the historical 377-feature mega-pool, in trained weight order.
POOL_INDICES = np.array([
    325, 297, 316, 294, 144, 94, 145, 348, 206, 195, 298, 75, 37, 317, 287,
    86, 291, 323, 95, 17, 310, 29, 63, 197, 68, 329, 314, 289, 65, 32, 365, 2,
])


def frontier_features(
    images: np.ndarray, core: tuple[np.ndarray, list[str]] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the 33 deployed features from raw TIFF-order float32 pixels."""
    if images.dtype != np.float32 or images.shape[1:] != (13, 64, 64):
        raise ValueError('expected float32 images with shape (N, 13, 64, 64)')
    core_values, core_names = core or patch_features(images, **CORE_CONFIG)
    # Compute only families used by the final head, preserving old pool offsets.
    families = [
        (0, core_values, core_names),
        (320, *lbp_features(images)),
        (326, *blob_features(images)),
        (344, *index_texture_scale2(images)),
        (364, *orient_entropy_scale2(images)),
    ]
    lookup = {
        start + i: (values[:, i], names[i])
        for start, values, names in families for i in range(values.shape[1])
    }
    selected = [lookup[int(index)] for index in POOL_INDICES]
    region, region_names = tail_region_shape(images)
    tail_index = region_names.index('tail_aniso_low_ndvi')
    selected.append((region[:, tail_index], region_names[tail_index]))
    features = np.column_stack([values for values, _ in selected]).astype(np.float32)
    if features.shape[1] != 33 or not np.isfinite(features).all():
        raise ValueError('invalid frontier feature matrix')
    return features, [name for _, name in selected]
