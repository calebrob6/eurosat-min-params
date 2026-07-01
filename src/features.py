"""Per-patch spectral feature extraction for EuroSAT.

These features summarise each 13-band patch into a small vector so that a tiny
linear model can classify it.  The point of the project is *minimal parameters*,
so feature extraction itself uses **zero learned parameters** -- it is fixed
arithmetic on the raw bands.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

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


# informative Sentinel-2 band pairs for cross-band joint spatial structure
_XBAND_PAIRS = [
    (B_RED, B_NIR), (B_GREEN, B_NIR), (B_BLUE, B_NIR),
    (B_SWIR1, B_NIR), (B_RED, B_SWIR1), (B_GREEN, B_RED),
    (B_NIR, B_SWIR2), (B_SWIR1, B_SWIR2),
]


def xband_corr_features(
    imgs: np.ndarray, pairs=_XBAND_PAIRS, eps: float = 1e-6,
) -> tuple[np.ndarray, list[str]]:
    """Pearson spatial correlation of band pairs over the patch pixels.

    ``corr = mean((a-abar)(b-bbar)) / (std_a std_b)`` over the 64x64 pixels, one
    feature per band pair.  Every other feature in this module is computed one
    band at a time; this is the only family that looks at the *joint* spatial
    structure of two bands, i.e. whether their patterns co-vary.  Vegetation
    couples RED/NIR very differently from built-up or water, so a handful of these
    correlations is a discriminative axis orthogonal to all the per-band
    intensity, gradient, coherence and orientation statistics.  ``len(pairs)``
    features, zero learned parameters.

    Returns ``(features [N, len(pairs)], names)``.
    """
    n, c = imgs.shape[:2]
    flat = imgs.reshape(n, c, -1)
    z = flat - flat.mean(2, keepdims=True)
    sd = np.sqrt((z * z).mean(2)) + eps                     # (N, C)
    parts, names = [], []
    for a, b in pairs:
        cov = (z[:, a] * z[:, b]).mean(1)
        parts.append((cov / (sd[:, a] * sd[:, b])).astype(np.float32)[:, None])
        names.append(f'xcorr_b{a}_b{b}')
    return np.concatenate(parts, 1), names


def _index_maps(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per-pixel normalised-difference index maps, shape (N, 6, H, W)."""
    b = imgs
    nir, red, grn, blu = b[:, B_NIR], b[:, B_RED], b[:, B_GREEN], b[:, B_BLUE]
    sw1, sw2 = b[:, B_SWIR1], b[:, B_SWIR2]
    ndvi = _ratio(nir, red)
    ndwi = _ratio(grn, nir)
    ndbi = _ratio(sw1, nir)
    ndmi = _ratio(nir, sw1)
    nbr = _ratio(nir, sw2)
    bsi = ((sw1 + red) - (nir + blu)) / ((sw1 + red) + (nir + blu) + _EPS)
    maps = np.stack([ndvi, ndwi, ndbi, ndmi, nbr, bsi], axis=1)  # (N, 6, H, W)
    return maps.astype(np.float32), ['ndvi', 'ndwi', 'ndbi', 'ndmi', 'nbr', 'bsi']


def index_texture_features(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Spatial-heterogeneity summary of each per-pixel spectral-index map.

    Every *other* index feature (in :func:`spectral_features`) is a per-band
    *mean* ratio -- one scalar NDVI/NDWI/... per patch -- discarding all within-
    patch spatial structure.  This family instead computes the per-pixel index
    map (e.g. ``NDVI(x, y)``) and summarises how uniform vs mixed it is across the
    64x64 patch: for each of 6 indices the spatial ``std``, gradient-magnitude
    ``mean`` and ``std``, and robust spread ``p90 - p10``.  ``6 * 4 = 24``
    features, zero learned parameters.

    Physical intuition: a managed PermanentCrop patch is near-uniform in NDVI
    while HerbaceousVegetation varies pixel-to-pixel; a Highway is a sharp
    low-NDVI streak (high NDVI *gradient*) across a vegetated background.  None of
    that is visible to a mean index, so this axis is orthogonal to every per-band
    intensity/gradient/coherence/orientation statistic and to the cross-band
    correlation family -- and it is the single strongest parameter-free family
    found, dropping the honest CV>=0.940 floor from k=38 to k=34.

    Returns ``(features [N, 24], names)``.
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


# --- global straight-line (Hough) features ---------------------------------

_HOUGH_NTHETA = 60      # orientation resolution over [0, pi)
_HOUGH_RHO_BIN = 2.0    # rho quantisation, pixels
_HOUGH_EDGE_PCTL = 85.0  # per-patch gradient percentile => edge pixels


def _chan_grad_mag(chan: np.ndarray) -> np.ndarray:
    """Per-pixel gradient magnitude of a single-channel stack (N, H, W)."""
    gx = np.diff(chan, axis=2)[:, :-1, :]
    gy = np.diff(chan, axis=1)[:, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def _hough_matrix(h: int, w: int, ntheta: int, rho_bin: float):
    """CSR projection matrix ``M`` s.t. the Hough accumulator of a flattened edge
    image ``e`` is ``M @ e``.  Each pixel votes for one (theta, rho) bin per
    angle; built once and reused for every patch (the pixel grid is fixed)."""
    ys, xs = np.mgrid[0:h, 0:w]
    xs = xs.ravel().astype(np.float64)
    ys = ys.ravel().astype(np.float64)
    hw = h * w
    thetas = np.linspace(0.0, np.pi, ntheta, endpoint=False)
    diag = np.hypot(h, w)
    nrho = int(np.ceil(2 * diag / rho_bin)) + 1
    rows = np.empty((ntheta, hw), np.int64)
    for t, th in enumerate(thetas):
        rho = xs * np.cos(th) + ys * np.sin(th)
        b = np.floor((rho + diag) / rho_bin).astype(np.int64)
        np.clip(b, 0, nrho - 1, out=b)
        rows[t] = t * nrho + b
    row = rows.reshape(-1)
    col = np.tile(np.arange(hw), ntheta)
    data = np.ones(row.shape[0], np.float32)
    return csr_matrix((data, (row, col)), shape=(ntheta * nrho, hw)), nrho, diag


def _hough_line_stats(chan: np.ndarray, chunk: int = 4000):
    """(peakfrac, peaklen, top3frac) global-line stats for a stack (N, H, W).

    ``peakfrac`` = longest straight line length / edge-pixel count; ``peaklen`` =
    longest line / patch diagonal; ``top3frac`` = sum of the best line at the 3
    strongest orientations / edge count.  Zero learned parameters.
    """
    n = chan.shape[0]
    mag = _chan_grad_mag(chan)
    he, we = mag.shape[1], mag.shape[2]
    thr = np.percentile(mag.reshape(n, -1), _HOUGH_EDGE_PCTL, axis=1)
    edge = (mag > thr[:, None, None]).astype(np.float32)
    edge_flat = edge.reshape(n, -1)
    ecount = edge_flat.sum(1) + _EPS
    M, nrho, diag = _hough_matrix(he, we, _HOUGH_NTHETA, _HOUGH_RHO_BIN)
    ntheta = _HOUGH_NTHETA
    peak = np.empty(n, np.float32)
    top3 = np.empty(n, np.float32)
    for s in range(0, n, chunk):
        acc = (M @ edge_flat[s:s + chunk].T).reshape(ntheta, nrho, -1)
        per_angle = acc.max(1)
        peak[s:s + chunk] = per_angle.max(0)
        top3[s:s + chunk] = np.sort(per_angle, 0)[-3:].sum(0)
    return ((peak / ecount).astype(np.float32),
            (peak / diag).astype(np.float32),
            (top3 / ecount).astype(np.float32))


def hough_line_features(imgs: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Global straight-line (Hough) statistics on pan + NDVI + NDBI channels.

    Every directional feature already in the pool (coherence, orientation
    entropy/histogram, their index-map variants) is *local* -- an aggregate of
    per-pixel gradient directions.  Many short parallel edges (crop rows) and one
    long streak (a Highway, a River) can look identical to those.  This family
    measures *global collinearity* via a Hough transform (== Radon of the binary
    edge image): a run of ``L`` collinear edge pixels makes an accumulator peak of
    value ``L``, so the peak is the length of the single longest straight line in
    the patch -- an axis no local statistic sees.  ``3 channels * 3 stats = 9``
    features, zero learned parameters.
    """
    b = imgs.astype(np.float32)
    pan = b.mean(1)
    nir, red = b[:, B_NIR], b[:, B_RED]
    sw1 = b[:, B_SWIR1]
    ndvi = (nir - red) / (nir + red + _EPS)
    ndbi = (sw1 - nir) / (sw1 + nir + _EPS)
    parts, names = [], []
    for cname, chan in (('pan', pan), ('ndvi', ndvi), ('ndbi', ndbi)):
        pf, pl, t3 = _hough_line_stats(chan)
        parts += [pf[:, None], pl[:, None], t3[:, None]]
        names += [f'linepf_{cname}', f'linepl_{cname}', f'linet3_{cname}']
    return np.concatenate(parts, 1).astype(np.float32), names


def patch_features(
    imgs: np.ndarray,
    pcts: tuple[int, ...] = (10, 25, 50, 75, 90),
    grad_scales: int = 3,
    coherence_scales: int = 0,
    orient_entropy_bins: int = 0,
    orient_hist_bins: int = 0,
    spectral_peak: bool = False,
    xband: bool = False,
    index_texture: bool = False,
    hough_lines: bool = False,
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
    if xband:
        xb, xb_names = xband_corr_features(imgs)
        parts.append(xb)
        names += xb_names
    if index_texture:
        ix, ix_names = index_texture_features(imgs)
        parts.append(ix)
        names += ix_names
    if hough_lines:
        hl, hl_names = hough_line_features(imgs)
        parts.append(hl)
        names += hl_names
    return np.concatenate(parts, 1).astype(np.float32), names
