"""Global-structure candidate families, reimplemented -- iteration 15.

Iteration 14 cached four candidate parameter-free families (blob / spectral
slope / corner / LBP on the pan, NDVI and NDBI channels) but its generating
source was lost when the iteration errored.  These are clean reimplementations
of the same concepts (NOT bit-identical to the lost cache -- cached under the
distinct ``gs2fam`` prefix), so any submission built on them is reproducible
from code in the repo.

Why these axes are candidates: everything in the ``o6+line`` pool is either a
local gradient statistic, a global-line (Hough) statistic, or a pixel-value
moment.  None of them see *region granularity* (how the patch tiles into
contiguous blobs -- Residential's many small blocks vs AnnualCrop's few large
fields), *junction density* (corners occur where edges MEET -- grid layouts vs
parallel rows), *micro-texture pattern statistics* (LBP: the distribution of
local binary patterns), or the *scale composition* of the spectrum (radial FFT
power slope: fine-grained vs smooth scenes).

All functions take raw patches ``imgs`` (N, 13, 64, 64) float32 and return
``(feats [N, F], names)``.  Fixed arithmetic, zero learned parameters.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.ndimage import label, uniform_filter

from src.data import B_BLUE, B_GREEN, B_NIR, B_RED, B_SWIR1

_EPS = 1e-6


def _channels(imgs: np.ndarray):
    """Same channel trio the line family uses: panchromatic + NDVI + NDBI."""
    b = imgs
    pan = b.mean(1)
    nir, red = b[:, B_NIR], b[:, B_RED]
    sw1 = b[:, B_SWIR1]
    ndvi = (nir - red) / (nir + red + _EPS)
    ndbi = (sw1 - nir) / (sw1 + nir + _EPS)
    return {'pan': pan, 'ndvi': ndvi, 'ndbi': ndbi}


# ---------------------------------------------------------------- blob (9)
def _blob_stats(chan: np.ndarray):
    """Connected components of the above-median mask: region granularity."""
    n, h, w = chan.shape
    npix = float(h * w)
    med = np.median(chan.reshape(n, -1), axis=1)
    masks = chan > med[:, None, None]
    eight = np.ones((3, 3), np.int32)
    lrg = np.empty(n, np.float32)
    ncc = np.empty(n, np.float32)
    msz = np.empty(n, np.float32)
    for i in range(n):
        lab, k = label(masks[i], structure=eight)
        if k == 0:
            lrg[i] = ncc[i] = msz[i] = 0.0
            continue
        sizes = np.bincount(lab.ravel())[1:]
        lrg[i] = sizes.max() / npix
        ncc[i] = np.log1p(k)
        msz[i] = sizes.mean() / npix
    return lrg, ncc, msz


def blob_features(imgs: np.ndarray):
    chans = _channels(imgs.astype(np.float32))
    parts, names = [], []
    for cname, chan in chans.items():
        lrg, ncc, msz = _blob_stats(chan)
        parts += [lrg[:, None], ncc[:, None], msz[:, None]]
        names += [f'blob2lrg_{cname}', f'blob2nc_{cname}', f'blob2msz_{cname}']
    return np.concatenate(parts, 1).astype(np.float32), names


# -------------------------------------------------------------- corner (6)
def _corner_stats(chan: np.ndarray):
    """Harris corner response: junction density among active-gradient pixels."""
    gx = np.gradient(chan, axis=2)
    gy = np.gradient(chan, axis=1)
    jxx = uniform_filter(gx * gx, size=(1, 3, 3))
    jyy = uniform_filter(gy * gy, size=(1, 3, 3))
    jxy = uniform_filter(gx * gy, size=(1, 3, 3))
    tr = jxx + jyy
    resp = (jxx * jyy - jxy * jxy) - 0.05 * tr * tr
    n = chan.shape[0]
    tr_f = tr.reshape(n, -1)
    resp_f = resp.reshape(n, -1)
    act = tr_f > np.median(tr_f, axis=1)[:, None]          # strong-gradient pixels
    nact = act.sum(1) + _EPS
    frac = ((resp_f > 0) & act).sum(1) / nact              # corner fraction
    # scale-invariant corner strength: sqrt(R+) has units of tr
    mag = np.sqrt(np.clip(resp_f, 0, None)).sum(1) / (tr_f.sum(1) + _EPS)
    return frac.astype(np.float32), mag.astype(np.float32)


def corner_features(imgs: np.ndarray):
    chans = _channels(imgs.astype(np.float32))
    parts, names = [], []
    for cname, chan in chans.items():
        fr, mg = _corner_stats(chan)
        parts += [fr[:, None], mg[:, None]]
        names += [f'corn2frac_{cname}', f'corn2mag_{cname}']
    return np.concatenate(parts, 1).astype(np.float32), names


# ----------------------------------------------------------------- lbp (6)
_OFFS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def _lbp_stats(chan: np.ndarray, chunk: int = 2000):
    """Rotation-invariant uniform LBP (8 neighbours -> 10 bins): pattern stats."""
    n = chan.shape[0]
    ent = np.empty(n, np.float32)
    uni = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        c = chan[s:s + chunk]
        ctr = c[:, 1:-1, 1:-1]
        bits = np.stack([(c[:, 1 + dy:c.shape[1] - 1 + dy,
                            1 + dx:c.shape[2] - 1 + dx] > ctr)
                         for dy, dx in _OFFS], 0).astype(np.int8)  # (8,nb,H-2,W-2)
        trans = np.abs(bits - np.roll(bits, 1, axis=0)).sum(0)     # circular transitions
        ones = bits.sum(0)
        riu = np.where(trans <= 2, ones, 9)                        # bins 0..9
        nb, npix = riu.shape[0], riu.shape[1] * riu.shape[2]
        flat = riu.reshape(nb, -1)
        hist = np.stack([(flat == v).sum(1) for v in range(10)], 1) / npix
        p = np.clip(hist, 1e-12, None)
        ent[s:s + chunk] = -(p * np.log(p)).sum(1)
        uni[s:s + chunk] = hist[:, :9].sum(1)
    return ent, uni


def lbp_features(imgs: np.ndarray):
    chans = _channels(imgs.astype(np.float32))
    parts, names = [], []
    for cname, chan in chans.items():
        ent, uni = _lbp_stats(chan)
        parts += [ent[:, None], uni[:, None]]
        names += [f'lbp2ent_{cname}', f'lbp2uni_{cname}']
    return np.concatenate(parts, 1).astype(np.float32), names


# -------------------------------------------------------------- sslope (3)
def _radial_slope(chan: np.ndarray, chunk: int = 2000):
    """Slope of log radial-mean FFT power vs log radius: scale composition."""
    n, h, w = chan.shape
    fy = np.fft.fftfreq(h) * h
    fx = np.fft.fftfreq(w) * w
    r = np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    rbin = np.round(r).astype(np.int64).ravel()
    rmax = min(h, w) // 2
    keep = (rbin >= 1) & (rbin <= rmax)
    onehot = np.zeros((rmax, h * w), np.float32)
    onehot[rbin[keep] - 1, np.nonzero(keep)[0]] = 1.0
    onehot /= onehot.sum(1, keepdims=True)
    logr = np.log(np.arange(1, rmax + 1).astype(np.float64))
    lc = logr - logr.mean()
    slope = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        c = chan[s:s + chunk]
        c = c - c.mean(axis=(1, 2), keepdims=True)
        pw = np.abs(np.fft.fft2(c)) ** 2
        rad = pw.reshape(pw.shape[0], -1) @ onehot.T           # (nb, rmax)
        lp = np.log(rad + _EPS)
        slope[s:s + chunk] = (lp @ lc) / (lc @ lc)
    return slope


def sslope_features(imgs: np.ndarray):
    chans = _channels(imgs.astype(np.float32))
    parts, names = [], []
    for cname, chan in chans.items():
        parts.append(_radial_slope(chan)[:, None])
        names.append(f'sslope2_{cname}')
    return np.concatenate(parts, 1).astype(np.float32), names


FAMILIES = {
    'blob2': blob_features,      # 9: connected-component region granularity
    'corn2': corner_features,    # 6: Harris junction density/strength
    'lbp2': lbp_features,        # 6: rotation-invariant-uniform LBP stats
    'sslope2': sslope_features,  # 3: radial FFT power slope
}

_CH = ['pan', 'ndvi', 'ndbi']
FAMILY_NAMES = {
    'blob2': [f'blob2{s}_{c}' for c in _CH for s in ['lrg', 'nc', 'msz']],
    'corn2': [f'corn2{s}_{c}' for c in _CH for s in ['frac', 'mag']],
    'lbp2': [f'lbp2{s}_{c}' for c in _CH for s in ['ent', 'uni']],
    'sslope2': [f'sslope2_{c}' for c in _CH],
}


if __name__ == '__main__':
    import os
    import sys
    import time

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.cache import CACHE_DIR, load_cached

    for split in ['train', 'val', 'test']:
        x, _ = load_cached(split)
        for fam, fn in FAMILIES.items():
            fp = os.path.join(CACHE_DIR, f'{split}_gs2fam_{fam}.npy')
            if os.path.exists(fp):
                continue
            t0 = time.time()
            feats, names = fn(x)
            np.save(fp, feats)
            print(f'{split} {fam}: {feats.shape} in {time.time()-t0:.1f}s '
                  f'({", ".join(names)})', flush=True)
