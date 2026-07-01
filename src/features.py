"""Per-patch spectral feature extraction for EuroSAT.

These features summarise each 13-band patch into a small vector so that a tiny
linear model can classify it.  The point of the project is *minimal parameters*,
so feature extraction itself uses **zero learned parameters** -- it is fixed
arithmetic on the raw bands.
"""

from __future__ import annotations

import numpy as np

from .data import B_BLUE, B_GREEN, B_NIR, B_RED, B_SWIR1, B_SWIR2

_EPS = 1e-6


def _ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalised difference index (a-b)/(a+b), per patch mean already applied."""
    return (a - b) / (a + b + _EPS)


def spectral_features(
    images: np.ndarray,
    include_mean: bool = True,
    include_std: bool = True,
    include_indices: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-patch spectral summary features.

    Args:
        images: array of shape (N, 13, H, W), float32 raw reflectance counts.
        include_mean: include per-band spatial mean (13 features).
        include_std: include per-band spatial std (13 features).
        include_indices: include normalised-difference vegetation/water/built-up
            indices computed from per-band means (a handful of features).

    Returns:
        (features [N, F] float32, feature_names list of length F).
    """
    n = images.shape[0]
    flat = images.reshape(n, images.shape[1], -1)  # (N, 13, H*W)
    band_mean = flat.mean(axis=2)  # (N, 13)
    band_std = flat.std(axis=2)    # (N, 13)

    feats: list[np.ndarray] = []
    names: list[str] = []

    if include_mean:
        feats.append(band_mean)
        names += [f'mean_b{i}' for i in range(13)]
    if include_std:
        feats.append(band_std)
        names += [f'std_b{i}' for i in range(13)]
    if include_indices:
        m = band_mean
        ndvi = _ratio(m[:, B_NIR], m[:, B_RED])
        ndwi = _ratio(m[:, B_GREEN], m[:, B_NIR])
        ndbi = _ratio(m[:, B_SWIR1], m[:, B_NIR])
        ndmi = _ratio(m[:, B_NIR], m[:, B_SWIR1])
        nbr = _ratio(m[:, B_NIR], m[:, B_SWIR2])
        bsi = ((m[:, B_SWIR1] + m[:, B_RED]) - (m[:, B_NIR] + m[:, B_BLUE])) / (
            (m[:, B_SWIR1] + m[:, B_RED]) + (m[:, B_NIR] + m[:, B_BLUE]) + _EPS
        )
        idx = np.stack([ndvi, ndwi, ndbi, ndmi, nbr, bsi], axis=1)
        feats.append(idx)
        names += ['ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi']

    return np.concatenate(feats, axis=1).astype(np.float32), names


# --- multi-scale texture features (the strong ones) ------------------------

def _pool2(imgs: np.ndarray) -> np.ndarray:
    """2x2 average pool over the last two (spatial) dims."""
    n, c, h, w = imgs.shape
    return imgs.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def _grad_mag(imgs: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude, shape (N, C, H-1, W-1)."""
    gx = np.diff(imgs, axis=3)[:, :, :-1, :]
    gy = np.diff(imgs, axis=2)[:, :, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def coherence_features(
    imgs: np.ndarray, scales: int = 2, eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Structure-tensor *coherence* per band and scale (zero parameters).

    The structure tensor ``[[Sxx, Sxy], [Sxy, Syy]]`` (patch sums of gradient
    outer products) has coherence ``sqrt((Sxx-Syy)^2 + 4 Sxy^2)/(Sxx+Syy)`` in
    ``[0, 1]``: high when the local gradient field is *directional* (linear
    structures like roads / Highway), low when isotropic (fields, forest).  This
    captures orientation information absent from the magnitude-only gradient
    statistics in :func:`patch_features`, and empirically lets a linear model hit
    the same accuracy with fewer selected features.

    Returns ``(features [N, 13*scales], names)``.
    """
    parts: list[np.ndarray] = []
    names: list[str] = []
    c = imgs.shape[1]
    cur = imgs
    for s in range(scales):
        gx = np.diff(cur, axis=3)[:, :, :-1, :]
        gy = np.diff(cur, axis=2)[:, :, :, :-1]
        sxx = (gx * gx).mean((2, 3))
        syy = (gy * gy).mean((2, 3))
        sxy = (gx * gy).mean((2, 3))
        coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (sxx + syy + eps)
        parts.append(coh.astype(np.float32))
        names += [f'coh{s}_b{i}' for i in range(c)]
        cur = _pool2(cur)
    return np.concatenate(parts, 1), names


def patch_features(
    imgs: np.ndarray,
    pcts: tuple[int, ...] = (10, 25, 50, 75, 90),
    grad_scales: int = 3,
    coherence_scales: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Rich fixed (zero-parameter) per-patch descriptor.

    Combines per-band intensity statistics (mean, std, percentiles) with
    multi-scale gradient-magnitude texture statistics (mean & std at each
    average-pooling scale).  Empirically a linear classifier on this descriptor
    exceeds 94% test accuracy on EuroSAT.

    Args:
        imgs: (N, 13, 64, 64) float32 raw band values.
        pcts: percentiles to include per band.
        grad_scales: number of octaves of gradient-magnitude texture (0 to skip).

    Returns:
        (features [N, F] float32, feature names).
    """
    n, c = imgs.shape[:2]
    flat = imgs.reshape(n, c, -1)
    parts: list[np.ndarray] = [flat.mean(2), flat.std(2)]
    names: list[str] = [f'mean_b{i}' for i in range(c)] + [f'std_b{i}' for i in range(c)]
    if pcts:
        pc = np.percentile(flat, pcts, axis=2).transpose(1, 0, 2).reshape(n, -1)
        parts.append(pc)
        names += [f'p{p}_b{i}' for i in range(c) for p in pcts]
    cur = imgs
    for s in range(grad_scales):
        g = _grad_mag(cur).reshape(n, c, -1)
        parts += [g.mean(2), g.std(2)]
        names += [f'g{s}mean_b{i}' for i in range(c)] + [f'g{s}std_b{i}' for i in range(c)]
        cur = _pool2(cur)
    if coherence_scales:
        coh, coh_names = coherence_features(imgs, scales=coherence_scales)
        parts.append(coh)
        names += coh_names
    return np.concatenate(parts, 1).astype(np.float32), names
