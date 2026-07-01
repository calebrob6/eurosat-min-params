"""Spatial-texture-of-spectral-INDEX feature families (iteration 6 candidate).

Every index feature in the deployed pool is a per-band *mean* ratio -- one scalar
NDVI/NDWI/... per patch -- so all within-patch spatial structure of the indices is
discarded.  This module computes the per-pixel index *map* (e.g. NDVI(x,y)) and
summarises its spatial heterogeneity: how uniform vs mixed the vegetation/water/
built-up signal is across the 64x64 patch.  That is a fresh axis orthogonal to
every per-band intensity/gradient/coherence/orientation and to the cross-band
correlation family (which measures band co-variation, not index heterogeneity).

Physical intuition: a PermanentCrop patch is spatially near-uniform in NDVI (one
managed crop) whereas HerbaceousVegetation / a mixed field varies pixel-to-pixel;
Highway has a sharp low-NDVI paved streak crossing a vegetated background, giving
a high NDVI *gradient* but modest std.  None of that is visible to a mean index.

All functions take raw patches ``imgs`` (N, 13, 64, 64) float32 and return
``(feats [N, F], names)``.  Fixed arithmetic, zero learned parameters.
"""
from __future__ import annotations

import numpy as np

from src.data import B_BLUE, B_GREEN, B_NIR, B_RED, B_SWIR1, B_SWIR2

_EPS = 1e-6


def _pool2(imgs: np.ndarray) -> np.ndarray:
    n, c, h, w = imgs.shape
    return imgs.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def _index_maps(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per-pixel normalised-difference index maps, shape (N, K, H, W)."""
    b = imgs
    nir, red, grn, blu = b[:, B_NIR], b[:, B_RED], b[:, B_GREEN], b[:, B_BLUE]
    sw1, sw2 = b[:, B_SWIR1], b[:, B_SWIR2]

    def nd(a, c):
        return (a - c) / (a + c + _EPS)

    ndvi = nd(nir, red)
    ndwi = nd(grn, nir)
    ndbi = nd(sw1, nir)
    ndmi = nd(nir, sw1)
    nbr = nd(nir, sw2)
    bsi = ((sw1 + red) - (nir + blu)) / ((sw1 + red) + (nir + blu) + _EPS)
    maps = np.stack([ndvi, ndwi, ndbi, ndmi, nbr, bsi], axis=1)  # (N, 6, H, W)
    names = ['ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi']
    return maps.astype(np.float32), names


def _grad_mag(m: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of maps (N, K, H, W) -> (N, K, H-1, W-1)."""
    gx = np.diff(m, axis=3)[:, :, :-1, :]
    gy = np.diff(m, axis=2)[:, :, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def index_texture(imgs: np.ndarray):
    """Spatial-heterogeneity summary of each index map.

    Per index: spatial std, gradient-magnitude mean, gradient-magnitude std, and
    robust spread p90-p10.  ``6 indices * 4 = 24`` features.
    """
    maps, inames = _index_maps(imgs)
    n, k = maps.shape[:2]
    flat = maps.reshape(n, k, -1)
    std = flat.std(2)
    g = _grad_mag(maps).reshape(n, k, -1)
    gmean = g.mean(2)
    gstd = g.std(2)
    p10, p90 = np.percentile(flat, [10, 90], axis=2)
    spread = p90 - p10
    feats = np.concatenate([std, gmean, gstd, spread], axis=1)
    names = ([f'ixstd_{s}' for s in inames]
             + [f'ixgm_{s}' for s in inames]
             + [f'ixgs_{s}' for s in inames]
             + [f'ixspr_{s}' for s in inames])
    return feats.astype(np.float32), names


def index_entropy(imgs: np.ndarray, nbins: int = 16):
    """Shannon entropy of each index map's value histogram (uniform vs mixed).

    The index is min-max normalised per patch to [0,1] then binned into ``nbins``;
    entropy is high for spatially varied indices (mixed vegetation), low for
    uniform ones (single managed crop / open water).  ``6`` features.
    """
    maps, inames = _index_maps(imgs)
    n, k = maps.shape[:2]
    flat = maps.reshape(n, k, -1)
    lo = flat.min(2, keepdims=True)
    hi = flat.max(2, keepdims=True)
    norm = (flat - lo) / (hi - lo + _EPS)
    idx = np.minimum((norm * nbins).astype(np.int64), nbins - 1)
    npix = flat.shape[2]
    ent = np.empty((n, k), np.float64)
    # histogram per (patch, index) via bincount over the flattened bins
    for j in range(k):
        for i in range(n):
            counts = np.bincount(idx[i, j], minlength=nbins).astype(np.float64)
            p = counts / npix
            nz = p > 0
            ent[i, j] = -(p[nz] * np.log(p[nz])).sum()
    names = [f'ixent_{s}' for s in inames]
    return ent.astype(np.float32), names


def index_texture_scale2(imgs: np.ndarray):
    """Same as :func:`index_texture` std+grad on a 2x2-pooled (coarser) patch.

    Tests whether index heterogeneity at a coarser scale adds signal.  ``6*2=12``
    features (coarse std, coarse gradient mean).
    """
    coarse = _pool2(imgs)
    maps, inames = _index_maps(coarse)
    n, k = maps.shape[:2]
    flat = maps.reshape(n, k, -1)
    std = flat.std(2)
    g = _grad_mag(maps).reshape(n, k, -1)
    gmean = g.mean(2)
    feats = np.concatenate([std, gmean], axis=1)
    names = ([f'ix2std_{s}' for s in inames] + [f'ix2gm_{s}' for s in inames])
    return feats.astype(np.float32), names


FAMILIES = {
    'ixtex': index_texture,          # 24 feats: std, grad-mean, grad-std, spread
    'ixent': index_entropy,          # 6 feats: value-histogram entropy
    'ixtex2': index_texture_scale2,  # 12 feats: coarse std + grad-mean
}
