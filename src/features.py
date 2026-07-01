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


def orientation_entropy_features(
    imgs: np.ndarray, nbins: int = 8, eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Shannon entropy of the magnitude-weighted gradient-orientation histogram.

    For each band we build a ``nbins``-bin *unsigned* (mod-pi) orientation
    histogram of the per-pixel gradients, weighted by gradient magnitude and
    normalised to sum 1, then take its Shannon entropy.  Low entropy means the
    local edges share one dominant direction (roads, crop rows); high entropy
    means orientation is spread out (isotropic forest / water).

    This is a *different* statistic from structure-tensor coherence: coherence is
    a second-moment scalar that is easily dominated by a single strong edge,
    whereas the entropy sees the whole orientation *distribution* and so
    separates "one direction" from "two directions" from "uniform".  Empirically
    it is the single most useful extra parameter-free feature family found: it
    lets a linear model reach the 0.940 honest bar with 13 fewer selected
    features than coherence alone.  One feature per band, zero learned
    parameters.

    Returns ``(features [N, 13], names)``.
    """
    gx = np.diff(imgs, axis=3)[:, :, :-1, :]
    gy = np.diff(imgs, axis=2)[:, :, :, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)                 # 0..pi (unsigned)
    n, c = mag.shape[:2]
    bin_idx = np.minimum((ang / (np.pi / nbins)).astype(np.int64), nbins - 1)
    mag_f = mag.reshape(n, c, -1)
    bin_f = bin_idx.reshape(n, c, -1)
    hist = np.empty((n, c, nbins), np.float64)
    for b in range(nbins):
        hist[:, :, b] = np.where(bin_f == b, mag_f, 0.0).sum(2)
    hist /= hist.sum(2, keepdims=True) + eps
    ent = -(hist * np.log(hist + eps)).sum(2)
    names = [f'oent_b{i}' for i in range(c)]
    return ent.astype(np.float32), names


def orientation_histogram_features(
    imgs: np.ndarray, nbins: int = 4, eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Magnitude-weighted, L1-normalised unsigned orientation histogram per band.

    Exposes the full *shape* of the per-band gradient-orientation distribution
    (``nbins`` bins), of which :func:`orientation_entropy_features` is a scalar
    summary.  Complements the entropy: the histogram distinguishes *which*
    direction dominates, not just how concentrated it is.  ``13*nbins`` features,
    zero learned parameters.
    """
    gx = np.diff(imgs, axis=3)[:, :, :-1, :]
    gy = np.diff(imgs, axis=2)[:, :, :, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    n, c = mag.shape[:2]
    bin_idx = np.minimum((ang / (np.pi / nbins)).astype(np.int64), nbins - 1)
    mag_f = mag.reshape(n, c, -1)
    bin_f = bin_idx.reshape(n, c, -1)
    hist = np.empty((n, c, nbins), np.float64)
    for b in range(nbins):
        hist[:, :, b] = np.where(bin_f == b, mag_f, 0.0).sum(2)
    hist /= hist.sum(2, keepdims=True) + eps
    names = [f'hog0_b{j}_o{b}' for j in range(c) for b in range(nbins)]
    return hist.reshape(n, c * nbins).astype(np.float32), names


def spectral_peak_features(
    imgs: np.ndarray, rmin: int = 4, rmax: int = 24, eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Per-band spectral peakiness: max/mean power in a mid-frequency annulus.

    Periodic textures (crop rows in PermanentCrop / AnnualCrop) place a sharp
    peak in the mid-frequency ring of the 2D power spectrum; random textures do
    not.  Feature = log1p(max power / mean power) over the annulus, per band.
    ``13`` features, zero learned parameters.
    """
    n, c, h, w = imgs.shape
    x = imgs - imgs.mean((2, 3), keepdims=True)
    power = np.fft.fftshift(np.abs(np.fft.fft2(x)) ** 2, axes=(2, 3))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ann = (r >= rmin) & (r <= rmax)
    P = power[:, :, ann]
    feat = P.max(2) / (P.mean(2) + eps)
    names = [f'fftpk_b{i}' for i in range(c)]
    return np.log1p(feat).astype(np.float32), names


def patch_features(
    imgs: np.ndarray,
    pcts: tuple[int, ...] = (10, 25, 50, 75, 90),
    grad_scales: int = 3,
    coherence_scales: int = 0,
    orient_entropy_bins: int = 0,
    orient_hist_bins: int = 0,
    spectral_peak: bool = False,
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
    if orient_entropy_bins:
        oent, oent_names = orientation_entropy_features(imgs, nbins=orient_entropy_bins)
        parts.append(oent)
        names += oent_names
    if orient_hist_bins:
        oh, oh_names = orientation_histogram_features(imgs, nbins=orient_hist_bins)
        parts.append(oh)
        names += oh_names
    if spectral_peak:
        sp, sp_names = spectral_peak_features(imgs)
        parts.append(sp)
        names += sp_names
    return np.concatenate(parts, 1).astype(np.float32), names
