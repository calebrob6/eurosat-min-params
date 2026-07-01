"""Candidate zero-parameter feature families to push the CV>=0.940 floor lower.

All functions take raw patches ``imgs`` of shape (N, 13, 64, 64) float32 and
return ``(feats [N, F], names)``.  They use only fixed arithmetic -- no learned
parameters -- so they can be appended to the linear model's feature pool without
changing the honest parameter count (which is 10*(k+1) on the *selected* k).
"""
from __future__ import annotations

import numpy as np

from src.data import B_BLUE, B_GREEN, B_NIR, B_RED

_EPS = 1e-6


def _grads(cur: np.ndarray):
    """Aligned per-pixel gradients gx, gy (both shape N,C,H-1,W-1)."""
    gx = np.diff(cur, axis=3)[:, :, :-1, :]
    gy = np.diff(cur, axis=2)[:, :, :, :-1]
    return gx, gy


def _pool2(imgs: np.ndarray) -> np.ndarray:
    n, c, h, w = imgs.shape
    return imgs.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def hog_features(imgs: np.ndarray, nbins: int = 6, bands=None, scales: int = 1):
    """Magnitude-weighted, L1-normalised unsigned orientation histogram per band.

    For each band the local gradient orientation (mod pi, so unsigned) is binned
    into ``nbins`` bins weighted by gradient magnitude, then normalised to sum 1
    over bins.  Directional textures (roads, crop rows) concentrate energy in few
    bins; isotropic textures (forest, water) spread it evenly.  This is richer
    than the single coherence scalar because it exposes the *shape* of the
    orientation distribution.
    """
    if bands is None:
        bands = list(range(imgs.shape[1]))
    parts, names = [], []
    cur = imgs[:, bands]
    nb = len(bands)
    for s in range(scales):
        gx, gy = _grads(cur)
        mag = np.sqrt(gx * gx + gy * gy)                 # (N, nb, h, w)
        ang = np.mod(np.arctan2(gy, gx), np.pi)          # 0..pi
        bin_idx = np.minimum((ang / (np.pi / nbins)).astype(np.int64), nbins - 1)
        n, c = mag.shape[:2]
        mag_f = mag.reshape(n, c, -1)
        bin_f = bin_idx.reshape(n, c, -1)
        hist = np.zeros((n, c, nbins), np.float64)
        # accumulate magnitude into orientation bins per (sample, band)
        for b in range(nbins):
            hist[:, :, b] = np.where(bin_f == b, mag_f, 0.0).sum(2)
        hist /= hist.sum(2, keepdims=True) + _EPS
        parts.append(hist.reshape(n, c * nbins).astype(np.float32))
        names += [f'hog{s}_b{bands[j]}_o{b}' for j in range(c) for b in range(nbins)]
        cur = _pool2(cur)
    return np.concatenate(parts, 1), names


def hog_pan(imgs: np.ndarray, nbins: int = 8, scales: int = 2):
    """HOG on a single panchromatic composite (mean of B/G/R/NIR)."""
    pan = imgs[:, [B_BLUE, B_GREEN, B_RED, B_NIR]].mean(1, keepdims=True)
    return hog_features(pan, nbins=nbins, bands=[0], scales=scales)


def coherence_scale3(imgs: np.ndarray):
    """3rd octave (scale index 2) of structure-tensor coherence, 13 features."""
    cur = _pool2(_pool2(imgs))
    gx, gy = _grads(cur)
    sxx = (gx * gx).mean((2, 3))
    syy = (gy * gy).mean((2, 3))
    sxy = (gx * gy).mean((2, 3))
    coh = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (sxx + syy + _EPS)
    names = [f'coh2_b{i}' for i in range(imgs.shape[1])]
    return coh.astype(np.float32), names


def orient_entropy(imgs: np.ndarray, nbins: int = 8, bands=None):
    """Shannon entropy of the magnitude-weighted orientation histogram per band.

    Complements coherence: high entropy = isotropic, low = one dominant
    direction.  One feature per band.
    """
    hist, _ = hog_features(imgs, nbins=nbins, bands=bands, scales=1)
    if bands is None:
        bands = list(range(imgs.shape[1]))
    nb = len(bands)
    hist = hist.reshape(hist.shape[0], nb, nbins).astype(np.float64)
    ent = -(hist * np.log(hist + _EPS)).sum(2)
    names = [f'oent_b{bands[j]}' for j in range(nb)]
    return ent.astype(np.float32), names


def fft_periodicity(imgs: np.ndarray, bands=None):
    """Spectral peakiness per band: max mid-frequency power / mean power.

    Periodic textures (crop rows in PermanentCrop/AnnualCrop) put a sharp peak in
    the mid-frequency annulus of the 2D power spectrum; random textures do not.
    Feature = (max power in mid annulus) / (mean power in annulus), per band.
    """
    if bands is None:
        bands = list(range(imgs.shape[1]))
    x = imgs[:, bands]
    n, c, h, w = x.shape
    x = x - x.mean((2, 3), keepdims=True)
    F = np.fft.fftshift(np.abs(np.fft.fft2(x)) ** 2, axes=(2, 3))
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    ann = (r >= 4) & (r <= 24)                 # mid-frequency ring
    P = F[:, :, ann]                            # (N, c, npix)
    feat = P.max(2) / (P.mean(2) + _EPS)
    names = [f'fftpk_b{bands[j]}' for j in range(c)]
    return np.log1p(feat).astype(np.float32), names
