"""Directional-index-texture feature families (iteration 7 candidates).

Submission 06's ``index_texture`` family summarises each per-pixel spectral-index
map (NDVI/NDWI/NDBI/NDMI/NBR/BSI) by its *isotropic* spatial heterogeneity --
std, gradient-magnitude mean/std, robust spread.  It sees *how much* an index
varies across the patch but not *in which direction* the index edges run.

The raw-band directional families that earned earlier param cuts --
structure-tensor coherence (sub 03) and gradient-orientation entropy (sub 04) --
were never applied to the index maps.  A Highway is a single low-NDVI paved
streak crossing a vegetated background: its index gradients share one dominant
direction, whereas a mixed field's index boundaries are isotropic.  Directional
statistics of the index maps should therefore separate Highway (the persistent
weakest class) from field/vegetation classes on an axis orthogonal to the
isotropic index-magnitude texture already in the pool.

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
    """Per-pixel normalised-difference index maps, shape (N, 6, H, W)."""
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
    maps = np.stack([ndvi, ndwi, ndbi, ndmi, nbr, bsi], axis=1)
    names = ['ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi']
    return maps.astype(np.float32), names


def index_coherence(imgs: np.ndarray, eps: float = _EPS):
    """Structure-tensor coherence of each index map (isotropy of index edges).

    ``coh = sqrt((Sxx-Syy)^2 + 4 Sxy^2)/(Sxx+Syy)`` in [0,1] on the per-pixel
    index gradients: high when the index edges are directional (a Highway's
    single-direction low-NDVI streak, crop rows), low when isotropic (mixed
    fields, forest).  ``6`` features.
    """
    maps, inames = _index_maps(imgs)
    gx = np.diff(maps, axis=3)[:, :, :-1, :]
    gy = np.diff(maps, axis=2)[:, :, :, :-1]
    sxx = (gx * gx).mean((2, 3))
    syy = (gy * gy).mean((2, 3))
    sxy = (gx * gy).mean((2, 3))
    coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (sxx + syy + eps)
    names = [f'ixcoh_{s}' for s in inames]
    return coh.astype(np.float32), names


def index_coherence2(imgs: np.ndarray, eps: float = _EPS):
    """Index-map coherence at 2 scales (fine + 2x2-pooled).  ``12`` features."""
    parts, names = [], []
    cur = imgs
    for s in range(2):
        maps, inames = _index_maps(cur)
        gx = np.diff(maps, axis=3)[:, :, :-1, :]
        gy = np.diff(maps, axis=2)[:, :, :, :-1]
        sxx = (gx * gx).mean((2, 3))
        syy = (gy * gy).mean((2, 3))
        sxy = (gx * gy).mean((2, 3))
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (sxx + syy + eps)
        parts.append(coh.astype(np.float32))
        names += [f'ixcoh{s}_{n}' for n in inames]
        cur = _pool2(cur)
    return np.concatenate(parts, 1).astype(np.float32), names


def index_orient_entropy(imgs: np.ndarray, nbins: int = 8, eps: float = _EPS):
    """Shannon entropy of each index map's magnitude-weighted orientation hist.

    Low entropy => index edges share one dominant direction (Highway streak, crop
    rows); high entropy => orientation is spread out (isotropic vegetation /
    water).  A different statistic from coherence: entropy sees the whole
    orientation *distribution* (one vs two vs uniform), coherence is a
    single-edge-dominated second moment.  ``6`` features.
    """
    maps, inames = _index_maps(imgs)
    gx = np.diff(maps, axis=3)[:, :, :-1, :]
    gy = np.diff(maps, axis=2)[:, :, :, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    n, k = mag.shape[:2]
    bin_idx = np.minimum((ang / (np.pi / nbins)).astype(np.int64), nbins - 1)
    mag_f = mag.reshape(n, k, -1)
    bin_f = bin_idx.reshape(n, k, -1)
    hist = np.empty((n, k, nbins), np.float64)
    for b in range(nbins):
        hist[:, :, b] = np.where(bin_f == b, mag_f, 0.0).sum(2)
    hist /= hist.sum(2, keepdims=True) + eps
    ent = -(hist * np.log(hist + eps)).sum(2)
    names = [f'ixoent_{s}' for s in inames]
    return ent.astype(np.float32), names


def index_dir(imgs: np.ndarray):
    """Coherence + orientation entropy of index maps combined.  ``12`` features."""
    coh, cn = index_coherence(imgs)
    oent, on = index_orient_entropy(imgs)
    return np.concatenate([coh, oent], 1).astype(np.float32), cn + on


FAMILIES = {
    'ixcoh': index_coherence,          # 6 feats: index-map structure-tensor coherence
    'ixcoh2': index_coherence2,        # 12 feats: 2-scale index coherence
    'ixoent': index_orient_entropy,    # 6 feats: index-map orientation entropy
    'ixdir': index_dir,                # 12 feats: coherence + orientation entropy
}
