"""NumPy reference measurements used by the full pool and RGB baseline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label

from .features import (
    _grad_mag,
    _index_maps,
    _pool2,
    coherence_features,
    orientation_entropy_features,
    xband_corr_features,
)

EPS = 1e-6
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1))


def channels(images: np.ndarray) -> dict[str, np.ndarray]:
    """Return panchromatic, NDVI, and historical NDBI channel stacks."""
    maps, _ = _index_maps(images)
    return {'pan': images.mean(1), 'ndvi': maps[:, 0], 'ndbi': maps[:, 2]}


def blob_stats(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure sizes and counts of eight-connected above-median regions."""
    count, height, width = channel.shape
    masks = channel > np.median(channel.reshape(count, -1), axis=1)[:, None, None]
    largest = np.zeros(count, np.float32)
    regions = np.zeros(count, np.float32)
    average = np.zeros(count, np.float32)
    for index, mask in enumerate(masks):
        labels, n = label(mask, structure=np.ones((3, 3), np.int32))
        if n:
            sizes = np.bincount(labels.ravel())[1:]
            largest[index] = sizes.max() / (height * width)
            regions[index] = np.log1p(n)
            average[index] = sizes.mean() / (height * width)
    return largest, regions, average


def blob_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return nine connected-component summaries."""
    parts, names = [], []
    for name, channel in channels(images).items():
        parts.extend(value[:, None] for value in blob_stats(channel))
        names.extend(f'blob2{stat}_{name}' for stat in ('lrg', 'nc', 'msz'))
    return np.concatenate(parts, axis=1).astype(np.float32), names


def lbp_stats(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return uniform local-binary-pattern entropy and uniform-pattern share."""
    center = channel[:, 1:-1, 1:-1]
    bits = np.stack([
        channel[:, 1 + dy:channel.shape[1] - 1 + dy, 1 + dx:channel.shape[2] - 1 + dx] > center
        for dy, dx in OFFSETS
    ]).astype(np.int8)
    transitions = np.abs(bits - np.roll(bits, 1, axis=0)).sum(0)
    patterns = np.where(transitions <= 2, bits.sum(0), 9).reshape(len(channel), -1)
    histogram = np.stack([(patterns == i).sum(1) for i in range(10)], axis=1) / patterns.shape[1]
    probability = np.clip(histogram, 1e-12, None)
    entropy = -(probability * np.log(probability)).sum(1)
    return entropy.astype(np.float32), histogram[:, :9].sum(1).astype(np.float32)


def lbp_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return six local-binary-pattern summaries."""
    parts, names = [], []
    for name, channel in channels(images).items():
        parts.extend(value[:, None] for value in lbp_stats(channel))
        names.extend((f'lbp2ent_{name}', f'lbp2uni_{name}'))
    return np.concatenate(parts, axis=1).astype(np.float32), names


def radial_slope(channel: np.ndarray) -> np.ndarray:
    """Fit log radial FFT power against log spatial frequency."""
    count, height, width = channel.shape
    fy, fx = np.fft.fftfreq(height) * height, np.fft.fftfreq(width) * width
    bins = np.round(np.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)).astype(np.int64).ravel()
    radius = min(height, width) // 2
    keep = (bins >= 1) & (bins <= radius)
    projection = np.zeros((radius, height * width), np.float32)
    projection[bins[keep] - 1, np.flatnonzero(keep)] = 1
    projection /= projection.sum(1, keepdims=True)
    frequency = np.log(np.arange(1, radius + 1).astype(np.float64))
    frequency -= frequency.mean()
    centered = channel - channel.mean((1, 2), keepdims=True)
    power = np.abs(np.fft.fft2(centered)) ** 2
    radial = power.reshape(count, -1) @ projection.T
    return ((np.log(radial + EPS) @ frequency) / (frequency @ frequency)).astype(np.float32)


def sslope_features(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return a radial FFT power slope for each structural channel."""
    values = channels(images)
    return np.column_stack([radial_slope(value) for value in values.values()]), [
        f'sslope2_{name}' for name in values
    ]


def index_coherence(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return coherence for each spectral-index map."""
    maps, names = _index_maps(images)
    values, _ = coherence_features(maps, scales=1)
    return values, [f'ixcoh_{name}' for name in names]


def index_texture_scale2(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return index standard deviation and gradient mean after one pooling step."""
    maps, names = _index_maps(_pool2(images))
    flat = maps.reshape(len(images), len(names), -1)
    gradient = _grad_mag(maps).reshape(len(images), len(names), -1)
    values = np.concatenate((flat.std(2), gradient.mean(2)), axis=1).astype(np.float32)
    return values, [f'ix2std_{name}' for name in names] + [f'ix2gm_{name}' for name in names]


def orient_entropy_scale2(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return gradient-direction entropy after one pooling step."""
    values, names = orientation_entropy_features(_pool2(images), nbins=8)
    return values, [name.replace('oent_', 'oent2_') for name in names]


def xband_corr(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return the second copy of cross-band correlations in the saved pool order."""
    return xband_corr_features(images)


def tail_region_shape(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return weighted low/high-tail spatial anisotropy and spread."""
    _, _, height, width = images.shape
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, height, dtype=np.float32),
        np.linspace(-1, 1, width, dtype=np.float32), indexing='ij',
    )
    xx, yy = xx.reshape(1, -1), yy.reshape(1, -1)
    parts, names = [], []
    for name, channel in channels(images).items():
        flat = channel.reshape(len(images), -1)
        q25, q75 = np.percentile(flat, (25, 75), axis=1).astype(np.float32)
        for tail, weights in (
            ('low', np.maximum(q25[:, None] - flat, 0.0)),
            ('high', np.maximum(flat - q75[:, None], 0.0)),
        ):
            mass = weights.sum(1) + EPS
            mean_x, mean_y = (weights * xx).sum(1) / mass, (weights * yy).sum(1) / mass
            dx, dy = xx - mean_x[:, None], yy - mean_y[:, None]
            sxx = (weights * dx * dx).sum(1) / mass
            syy = (weights * dy * dy).sum(1) / mass
            sxy = (weights * dx * dy).sum(1) / mass
            trace = sxx + syy
            anisotropy = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (trace + EPS)
            parts.extend((anisotropy[:, None], np.sqrt(trace)[:, None]))
            names.extend((f'tail_aniso_{tail}_{name}', f'tail_spread_{tail}_{name}'))
    return np.concatenate(parts, axis=1).astype(np.float32), names
