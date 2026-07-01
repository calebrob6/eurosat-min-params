"""Orthogonal zero-parameter feature families to push the CV>=0.940 floor below 410.

Iteration-4 notes concluded the next lever toward k=35 (360 params) must be an
*orthogonal* family, not more orientation.  The existing 273-feature pool captures
per-band intensity statistics, gradient-magnitude texture, structure-tensor
coherence, gradient-orientation entropy/histogram, and FFT peakiness -- but every
one of those is either a per-pixel gradient statistic or a per-band-independent
spectral statistic.  Two axes are untouched:

  * **texture COARSENESS** -- how fast the texture decorrelates with distance.
    A normalised directional variogram gamma(d) = 0.5*E[(I(x)-I(x+d))^2]/var(I)
    (= 1 - spatial autocorrelation at lag d) grows from 0 (smooth) to 1
    (decorrelated) as lag d crosses the texture correlation length.  The *shape*
    of that curve over several lags is a coarseness signature orthogonal to the
    offset-1 gradient magnitude the pool already uses.

  * **cross-band JOINT structure** -- every existing feature is computed one band
    at a time.  The Pearson spatial correlation of two bands over the patch
    pixels captures whether their spatial patterns co-vary (vegetation couples
    RED/NIR very differently from built-up), which no per-band feature can see.

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


def variogram(imgs: np.ndarray, lags=(2, 4, 8, 16)):
    """Normalised isotropic variogram per band and lag (texture coarseness).

    gamma(d) = 0.5 * mean_dir mean_pix (I(x) - I(x+d))^2 / var(I), averaged over
    the horizontal and vertical directions.  Equals 1 - autocorrelation at lag d,
    so it rises from ~0 for smooth patches to ~1 once the lag exceeds the texture
    correlation length.  The set over several lags encodes *how coarse* the
    texture is, an axis the offset-1 gradient statistics cannot resolve.
    ``13 * len(lags)`` features.
    """
    n, c, h, w = imgs.shape
    var = imgs.var((2, 3)) + _EPS                       # (N, C)
    parts, names = [], []
    for d in lags:
        dh = imgs[:, :, :, d:] - imgs[:, :, :, :-d]
        dv = imgs[:, :, d:, :] - imgs[:, :, :-d, :]
        gh = (dh * dh).mean((2, 3))
        gv = (dv * dv).mean((2, 3))
        g = 0.5 * 0.5 * (gh + gv) / var                 # (N, C)
        parts.append(g.astype(np.float32))
        names += [f'vgram_d{d}_b{i}' for i in range(c)]
    return np.concatenate(parts, 1), names


def glcm_homogeneity(imgs: np.ndarray, lags=(1, 2, 4)):
    """GLCM-style homogeneity per band and lag: mean 1/(1 + (dI/scale)^2).

    Homogeneity weights small-difference pixel pairs heavily, so it is high for
    smooth/uniform texture and low for busy texture -- a bounded, saturation-free
    complement to the (unbounded, coarseness-only) variogram.  Differences are
    scaled by each band's std so the measure is intensity-invariant.
    ``13 * len(lags)`` features.
    """
    n, c, h, w = imgs.shape
    sd = imgs.std((2, 3), keepdims=True) + _EPS
    z = imgs / sd
    parts, names = [], []
    for d in lags:
        dh = z[:, :, :, d:] - z[:, :, :, :-d]
        dv = z[:, :, d:, :] - z[:, :, :-d, :]
        hh = (1.0 / (1.0 + dh * dh)).mean((2, 3))
        hv = (1.0 / (1.0 + dv * dv)).mean((2, 3))
        homog = 0.5 * (hh + hv)
        parts.append(homog.astype(np.float32))
        names += [f'homog_d{d}_b{i}' for i in range(c)]
    return np.concatenate(parts, 1), names


# informative Sentinel-2 band pairs for cross-band joint structure
_XPAIRS = [
    (B_RED, B_NIR), (B_GREEN, B_NIR), (B_BLUE, B_NIR),
    (B_SWIR1, B_NIR), (B_RED, B_SWIR1), (B_GREEN, B_RED),
    (B_NIR, B_SWIR2), (B_SWIR1, B_SWIR2),
]


def xband_corr(imgs: np.ndarray, pairs=_XPAIRS):
    """Pearson spatial correlation of band pairs over the patch pixels.

    corr = mean((a-abar)(b-bbar)) / (std_a std_b) over the 64x64 pixels, one
    feature per band pair.  Captures whether two bands' spatial patterns co-vary
    -- a joint-structure axis no per-band feature exposes.  ``len(pairs)`` feats.
    """
    n, c, h, w = imgs.shape
    flat = imgs.reshape(n, c, -1)
    mu = flat.mean(2, keepdims=True)
    z = flat - mu
    sd = np.sqrt((z * z).mean(2)) + _EPS                # (N, C)
    parts, names = [], []
    for a, b in pairs:
        cov = (z[:, a] * z[:, b]).mean(1)
        corr = cov / (sd[:, a] * sd[:, b])
        parts.append(corr.astype(np.float32)[:, None])
        names.append(f'xcorr_b{a}_b{b}')
    return np.concatenate(parts, 1), names


def orient_entropy_scale2(imgs: np.ndarray, nbins: int = 8):
    """2nd-octave gradient-orientation entropy (control: 'more orientation').

    Same statistic as the deployed oent family but computed on a 2x2-pooled
    (coarser) patch, to test whether a second orientation SCALE adds anything the
    full-res orientation features do not.  13 features.
    """
    cur = _pool2(imgs)
    gx = np.diff(cur, axis=3)[:, :, :-1, :]
    gy = np.diff(cur, axis=2)[:, :, :, :-1]
    mag = np.sqrt(gx * gx + gy * gy)
    ang = np.mod(np.arctan2(gy, gx), np.pi)
    n, c = mag.shape[:2]
    bin_idx = np.minimum((ang / (np.pi / nbins)).astype(np.int64), nbins - 1)
    mag_f = mag.reshape(n, c, -1)
    bin_f = bin_idx.reshape(n, c, -1)
    hist = np.empty((n, c, nbins), np.float64)
    for bb in range(nbins):
        hist[:, :, bb] = np.where(bin_f == bb, mag_f, 0.0).sum(2)
    hist /= hist.sum(2, keepdims=True) + _EPS
    ent = -(hist * np.log(hist + _EPS)).sum(2)
    names = [f'oent2_b{i}' for i in range(c)]
    return ent.astype(np.float32), names


FAMILIES = {
    'vgram': lambda x: variogram(x),
    'homog': lambda x: glcm_homogeneity(x),
    'xcorr': lambda x: xband_corr(x),
    'oent2': lambda x: orient_entropy_scale2(x),
}
