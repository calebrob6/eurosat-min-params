"""Global straight-line (Hough/Radon) feature families -- iteration 13 candidate.

Every directional feature already in the ``o6`` pool is *local*: structure-tensor
coherence, gradient-orientation entropy/histogram and their index-map variants all
aggregate per-pixel gradient directions.  A patch of many short parallel edges
(crop rows) and a patch with one long straight streak (a Highway) can look
identical to those statistics -- both have low orientation entropy / high
coherence.  What separates them is *global collinearity*: a Highway is a single
long line spanning the patch, whereas crop rows are many short parallel lines and
forest/field texture has no line at all.

This family measures that with a Hough (== Radon of a binary edge image)
transform: a straight run of ``L`` collinear edge pixels produces an accumulator
peak of value ``L``, so the peak (relative to the edge budget) is the length of
the single longest straight line in the patch -- an axis no local statistic sees.

All functions take raw patches ``imgs`` (N, 13, 64, 64) float32 and return
``(feats [N, F], names)``.  Fixed arithmetic, zero learned parameters.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from src.data import B_BLUE, B_GREEN, B_NIR, B_RED, B_SWIR1

_EPS = 1e-6

# Hough discretisation (fixed constants, no learned parameters).
_NTHETA = 60                       # orientation resolution over [0, pi)
_RHO_BIN = 2.0                     # rho quantisation, pixels
_EDGE_PCTL = 85.0                  # per-patch gradient percentile => edge pixels


def _grad_mag(chan: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of a single-channel image stack (N,H,W)."""
    gx = np.diff(chan, axis=2)[:, :-1, :]
    gy = np.diff(chan, axis=1)[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def _hough_matrix(h: int, w: int, ntheta: int, rho_bin: float):
    """Projection matrix ``M`` (ntheta*nrho, H*W) as a sparse CSR: the Hough
    accumulator of an edge image ``e`` (flattened) is ``M @ e``.  Each pixel votes
    for exactly one (theta, rho) bin per angle, so ``M`` has ntheta*H*W nonzeros.
    Built once and reused for all patches (the pixel grid is fixed)."""
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.ravel().astype(np.float64)
    ys = ys.ravel().astype(np.float64)
    hw = h * w
    thetas = np.linspace(0.0, np.pi, ntheta, endpoint=False)
    diag = np.hypot(h, w)
    nrho = int(np.ceil(2 * diag / rho_bin)) + 1
    rows = np.empty((ntheta, hw), np.int64)
    for t, th in enumerate(thetas):
        rho = xs * np.cos(th) + ys * np.sin(th)          # (H*W,)
        b = np.floor((rho + diag) / rho_bin).astype(np.int64)
        np.clip(b, 0, nrho - 1, out=b)
        rows[t] = t * nrho + b
    row = rows.reshape(-1)
    col = np.tile(np.arange(hw), ntheta)
    data = np.ones(row.shape[0], np.float32)
    M = csr_matrix((data, (row, col)), shape=(ntheta * nrho, hw))
    return M, nrho, diag


def _hough_line_stats(chan: np.ndarray, chunk: int = 4000):
    """For a single-channel stack (N,H,W) return (peakfrac, peaklen, top3frac).

    * peakfrac  = (longest straight line length) / (edge-pixel count)  in (0,1]
    * peaklen   = (longest straight line length) / patch diagonal      -- absolute
    * top3frac  = (sum of best line at each of the 3 strongest angles) / edges
    """
    n = chan.shape[0]
    mag = _grad_mag(chan)                                  # (N, H-1, W-1)
    he, we = mag.shape[1], mag.shape[2]
    thr = np.percentile(mag.reshape(n, -1), _EDGE_PCTL, axis=1)  # (N,)
    edge = (mag > thr[:, None, None]).astype(np.float32)
    edge_flat = edge.reshape(n, -1)                        # (N, He*We)
    ecount = edge_flat.sum(1) + _EPS                       # (N,)

    M, nrho, diag = _hough_matrix(he, we, _NTHETA, _RHO_BIN)
    ntheta = _NTHETA

    peak = np.empty(n, np.float32)
    top3 = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        e = edge_flat[s:s + chunk]                         # (nb, HW)
        acc = (M @ e.T)                                    # (ntheta*nrho, nb)
        acc = acc.reshape(ntheta, nrho, acc.shape[1])
        per_angle = acc.max(1)                             # (ntheta, nb) best line/angle
        peak[s:s + chunk] = per_angle.max(0)
        top3[s:s + chunk] = np.sort(per_angle, 0)[-3:].sum(0)
    peakfrac = (peak / ecount).astype(np.float32)
    peaklen = (peak / diag).astype(np.float32)
    top3frac = (top3 / ecount).astype(np.float32)
    return peakfrac, peaklen, top3frac


def _channels(imgs: np.ndarray):
    """Highway-relevant single channels: panchromatic + NDVI + NDBI index maps."""
    b = imgs
    pan = b.mean(1)                                        # (N,H,W) overall brightness
    nir, red, grn, blu = b[:, B_NIR], b[:, B_RED], b[:, B_GREEN], b[:, B_BLUE]
    sw1 = b[:, B_SWIR1]
    ndvi = (nir - red) / (nir + red + _EPS)
    ndbi = (sw1 - nir) / (sw1 + nir + _EPS)
    return {'pan': pan, 'ndvi': ndvi, 'ndbi': ndbi}


def hough_lines(imgs: np.ndarray):
    """9-feature global-line family: (peakfrac, peaklen, top3frac) x 3 channels."""
    chans = _channels(imgs.astype(np.float32))
    parts, names = [], []
    for cname, chan in chans.items():
        pf, pl, t3 = _hough_line_stats(chan)
        parts += [pf[:, None], pl[:, None], t3[:, None]]
        names += [f'linepf_{cname}', f'linepl_{cname}', f'linet3_{cname}']
    return np.concatenate(parts, 1).astype(np.float32), names


def hough_pan(imgs: np.ndarray):
    """3-feature panchromatic-only variant (peakfrac, peaklen, top3frac)."""
    pan = imgs.astype(np.float32).mean(1)
    pf, pl, t3 = _hough_line_stats(pan)
    feats = np.stack([pf, pl, t3], 1).astype(np.float32)
    return feats, ['linepf_pan', 'linepl_pan', 'linet3_pan']


FAMILIES = {
    'line': hough_lines,      # 9 feats: global-line stats on pan + NDVI + NDBI
    'linepan': hough_pan,     # 3 feats: panchromatic-only global-line stats
}
