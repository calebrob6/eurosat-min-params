#!/usr/bin/env python
"""Second zero-parameter RESISC45 feature pool: oriented, periodic, morphological.

The first GPU pool summarises intensity distributions and isotropic texture.
RESISC45's man-made classes are instead separated by *oriented* and *periodic*
structure -- runways, freeways, railways, bridges, and regular field or building
lattices.  This module adds a fixed Gabor bank, uniform local binary patterns,
normalised autocorrelation periodicity, directional projection-profile
regularity, and percentile-thresholded morphology.  All constants are fixed a
priori or derived per image, so nothing here is learned.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from src.cache import CACHE_DIR  # noqa: E402

EPS = 1e-6
MAP_NAMES = ('pan', 'exg', 'sat', 'gr')
GABOR_SCALES = (4.0, 8.0, 16.0, 32.0)
GABOR_ORIENTATIONS = 6


def _maps(x: torch.Tensor) -> torch.Tensor:
    """Four structural maps: panchromatic, excess green, saturation, green-red."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    total = r + g + b + EPS
    mx = x.max(1).values
    mn = x.min(1).values
    return torch.stack((
        (r + g + b) / 3.0,
        (2 * g - r - b) / total,
        (mx - mn) / (mx + EPS),
        (g - r) / (g + r + EPS),
    ), dim=1)


def _log_gabor_bank(size: int, device: str) -> torch.Tensor:
    """Fixed one-sided log-Gabor transfer functions over scales and angles.

    Building the bank in the frequency domain makes the filtering a single
    elementwise product, avoiding the huge spatial kernels that a 16-pixel
    wavelength would otherwise need.
    """
    fy = torch.fft.fftfreq(size, device=device)[:, None]
    fx = torch.fft.fftfreq(size, device=device)[None, :]
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    radius[0, 0] = 1.0
    angle = torch.atan2(fy.expand_as(radius), fx.expand_as(radius))
    filters = []
    for wavelength in GABOR_SCALES:
        f0 = 1.0 / wavelength
        radial = torch.exp(-(torch.log(radius / f0) ** 2) / (2 * np.log(0.65) ** 2))
        radial[0, 0] = 0.0
        for o in range(GABOR_ORIENTATIONS):
            theta = np.pi * o / GABOR_ORIENTATIONS
            delta = torch.atan2(torch.sin(angle - theta), torch.cos(angle - theta))
            spread = torch.exp(-(delta ** 2) / (2 * (np.pi / GABOR_ORIENTATIONS) ** 2))
            filters.append(radial * spread)
    return torch.stack(filters)


def _gabor(maps: torch.Tensor, bank: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Log-Gabor amplitude statistics per wavelength and orientation."""
    n, c, h, w = maps.shape
    field = torch.nn.functional.avg_pool2d(maps, 2)
    size = field.shape[-1]
    flat = field.reshape(n * c, size, size)
    spectrum = torch.fft.fft2(flat)[:, None]
    amplitude = torch.fft.ifft2(spectrum * bank[None]).abs()
    amplitude = amplitude.reshape(n, c, len(GABOR_SCALES), GABOR_ORIENTATIONS, -1)
    mean = amplitude.mean(-1)
    std = amplitude.std(-1)
    out: list[torch.Tensor] = []
    for s, wavelength in enumerate(GABOR_SCALES):
        band = mean[:, :, s]
        total = band.sum(-1, keepdim=True) + EPS
        out.append(torch.log(band.mean(-1) + 1e-8))
        names.extend(f'gab{wavelength:g}mean_{m}' for m in MAP_NAMES)
        share = band / total
        srt = torch.sort(share, dim=-1, descending=True).values
        for i in (0, 1, GABOR_ORIENTATIONS - 1):
            out.append(srt[:, :, i])
            names.extend(f'gab{wavelength:g}s{i}_{m}' for m in MAP_NAMES)
        out.append(-(share * torch.log(share + EPS)).sum(-1))
        names.extend(f'gab{wavelength:g}ent_{m}' for m in MAP_NAMES)
        cv = (std[:, :, s] / (mean[:, :, s] + EPS))
        out.append(cv.mean(-1))
        names.extend(f'gab{wavelength:g}cv_{m}' for m in MAP_NAMES)
        out.append(cv.max(-1).values - cv.min(-1).values)
        names.extend(f'gab{wavelength:g}cvrange_{m}' for m in MAP_NAMES)
    return out


_LBP_OFFSETS = tuple(
    (int(round(-np.sin(2 * np.pi * i / 8))), int(round(np.cos(2 * np.pi * i / 8))))
    for i in range(8)
)


def _lbp(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Rotation-invariant uniform LBP histograms at three radii."""
    out: list[torch.Tensor] = []
    n, c = maps.shape[0], maps.shape[1]
    for radius in (1, 2, 4):
        centre = maps[:, :, 4 * radius:-4 * radius or None, 4 * radius:-4 * radius or None]
        bits = []
        for dy, dx in _LBP_OFFSETS:
            oy, ox = dy * radius, dx * radius
            y0 = 4 * radius + oy
            x0 = 4 * radius + ox
            shifted = maps[:, :, y0:y0 + centre.shape[2], x0:x0 + centre.shape[3]]
            bits.append((shifted >= centre).float())
        stack = torch.stack(bits, dim=-1)
        transitions = (stack - stack.roll(1, dims=-1)).abs().sum(-1)
        count = stack.sum(-1)
        uniform = transitions <= 2
        code = torch.where(uniform, count, torch.full_like(count, 9.0)).long()
        flat = code.reshape(n, c, -1)
        hist = torch.zeros(n, c, 10, device=maps.device)
        hist.scatter_add_(2, flat, torch.ones_like(flat, dtype=torch.float32))
        hist = hist / (hist.sum(-1, keepdim=True) + EPS)
        for i in range(10):
            out.append(hist[:, :, i])
            names.extend(f'lbp{radius}_{i}_{m}' for m in MAP_NAMES)
    return out


def _autocorrelation(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Periodicity strength and scale from the normalised autocorrelation."""
    n, c, h, w = maps.shape
    centred = maps - maps.reshape(n, c, -1).mean(-1)[:, :, None, None]
    spectrum = torch.fft.rfft2(centred)
    corr = torch.fft.irfft2(spectrum.abs() ** 2, s=(h, w))
    corr = corr / (corr[:, :, :1, :1] + EPS)
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    yy, xx = torch.meshgrid(
        torch.arange(h, device=maps.device) - h // 2,
        torch.arange(w, device=maps.device) - w // 2,
        indexing='ij',
    )
    radius = torch.sqrt((yy.float() ** 2 + xx.float() ** 2))
    out: list[torch.Tensor] = []
    for lo, hi in ((3, 6), (6, 12), (12, 24), (24, 48), (48, 96)):
        mask = (radius >= lo) & (radius < hi)
        values = corr[:, :, mask]
        out.append(values.max(-1).values)
        names.extend(f'acmax{lo}_{m}' for m in MAP_NAMES)
        out.append(values.mean(-1))
        names.extend(f'acmean{lo}_{m}' for m in MAP_NAMES)
        out.append(values.max(-1).values - values.mean(-1))
        names.extend(f'acpeak{lo}_{m}' for m in MAP_NAMES)
    return out


def _projection(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Row/column projection-profile regularity on the axis-aligned directions."""
    out: list[torch.Tensor] = []
    n, c = maps.shape[0], maps.shape[1]
    for tag, profile in (('row', maps.mean(-1)), ('col', maps.mean(-2))):
        centred = profile - profile.mean(-1, keepdim=True)
        out.append(centred.std(-1))
        names.extend(f'prof{tag}std_{m}' for m in MAP_NAMES)
        spectrum = torch.fft.rfft(centred, dim=-1).abs() ** 2
        share = spectrum[:, :, 1:] / (spectrum[:, :, 1:].sum(-1, keepdim=True) + EPS)
        out.append(share.max(-1).values)
        names.extend(f'prof{tag}pk_{m}' for m in MAP_NAMES)
        out.append(share.argmax(-1).float())
        names.extend(f'prof{tag}fr_{m}' for m in MAP_NAMES)
        out.append(-(share * torch.log(share + EPS)).sum(-1))
        names.extend(f'prof{tag}ent_{m}' for m in MAP_NAMES)
    return out


def _morphology(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Shape of per-image percentile-thresholded regions at two scales."""
    out: list[torch.Tensor] = []
    n, c, h, w = maps.shape
    flat = maps.reshape(n, c, -1)
    for q in (0.1, 0.5, 0.9):
        thresh = torch.quantile(flat.float(), q, dim=-1)[:, :, None, None]
        binary = (maps >= thresh).float()
        opened = -F.max_pool2d(-binary, 3, stride=1, padding=1)
        opened = F.max_pool2d(opened, 3, stride=1, padding=1)
        closed = F.max_pool2d(binary, 3, stride=1, padding=1)
        closed = -F.max_pool2d(-closed, 3, stride=1, padding=1)
        area = binary.reshape(n, c, -1).mean(-1) + EPS
        out.append(opened.reshape(n, c, -1).mean(-1) / area)
        names.extend(f'morphopen{q}_{m}' for m in MAP_NAMES)
        out.append(closed.reshape(n, c, -1).mean(-1) / area)
        names.extend(f'morphclose{q}_{m}' for m in MAP_NAMES)
        perimeter = (binary[:, :, :, 1:] != binary[:, :, :, :-1]).float().mean((2, 3))
        perimeter = perimeter + (binary[:, :, 1:] != binary[:, :, :-1]).float().mean((2, 3))
        out.append(perimeter / area)
        names.extend(f'morphperim{q}_{m}' for m in MAP_NAMES)
        blocks = F.avg_pool2d(binary, 16).reshape(n, c, -1)
        out.append(blocks.std(-1) / (blocks.mean(-1) + EPS))
        names.extend(f'morphclump{q}_{m}' for m in MAP_NAMES)
    return out


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 128) -> tuple[np.ndarray, list[str]]:
    """Extract the oriented/periodic pool for uint8 ``(N, 3, H, W)`` images."""
    bank = _log_gabor_bank(images.shape[-1] // 2, device)
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                            device=device).float() / 255.0
        maps = _maps(x)
        local: list[str] = []
        parts = (
            _gabor(maps, bank, local)
            + _lbp(maps, local)
            + _autocorrelation(maps, local)
            + _projection(maps, local)
            + _morphology(maps, local)
        )
        block = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1)
        chunks.append(block.float().cpu().numpy())
        if not names:
            names = local
    out = np.concatenate(chunks, axis=0)
    if out.shape[1] != len(names):
        raise ValueError(f'{out.shape[1]} columns but {len(names)} names')
    return out, names


if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
        images = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_256.npy'), mmap_mode='r')
        features, names = extract(images)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu2_pool.npy'), features)
        print(split, features.shape, flush=True)
    with open(os.path.join(CACHE_DIR, 'resisc45_gpu2_pool_names.txt'), 'w') as handle:
        handle.write('\n'.join(names) + '\n')
    print('names', len(names))
