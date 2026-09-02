#!/usr/bin/env python
"""Build a large zero-parameter RGB feature pool for RESISC45 on the GPU.

Every column is a deterministic arithmetic summary of a single image, so the
extractor contributes no learned parameters.  Families are computed from the
native 256x256 cache and cover colour distributions, fixed colour-bin
occupancies, multiscale gradient and structure-tensor texture, Haar subband
energies, Fourier ring/wedge power, grey-level co-occurrence, box-counting
lacunarity, and coarse spatial layout.  Rotation-sensitive histograms are stored
both raw and sorted so the selector can pick whichever form generalises.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from src.cache import CACHE_DIR  # noqa: E402

EPS = 1e-6
MAP_NAMES = ('r', 'g', 'b', 'pan', 'exg', 'sat', 'light', 'gr', 'gb', 'rb')


def _maps(x: torch.Tensor) -> torch.Tensor:
    """Ten fixed colour maps from an RGB batch scaled to ``[0, 1]``."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    total = r + g + b + EPS
    mx = x.max(1).values
    mn = x.min(1).values
    return torch.stack((
        r, g, b,
        (r + g + b) / 3.0,
        (2 * g - r - b) / total,
        (mx - mn) / (mx + EPS),
        (mx + mn) / 2.0,
        (g - r) / (g + r + EPS),
        (g - b) / (g + b + EPS),
        (r - b) / (r + b + EPS),
    ), dim=1)


def _pool(m: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return m
    return torch.nn.functional.avg_pool2d(m, factor)


def _grad(m: torch.Tensor) -> torch.Tensor:
    gx = m[:, :, :, 1:] - m[:, :, :, :-1]
    gy = m[:, :, 1:, :] - m[:, :, :-1, :]
    return torch.sqrt(gx[:, :, :-1, :] ** 2 + gy[:, :, :, :-1] ** 2 + EPS)


def _moments(flat: torch.Tensor, prefix: str, names: list[str]) -> list[torch.Tensor]:
    """Mean, standard deviation, skew, kurtosis and five percentiles."""
    mean = flat.mean(-1)
    centred = flat - mean[..., None]
    var = (centred ** 2).mean(-1)
    std = torch.sqrt(var + EPS)
    skew = (centred ** 3).mean(-1) / (std ** 3 + EPS)
    kurt = (centred ** 4).mean(-1) / (var ** 2 + EPS)
    q = torch.quantile(
        flat.float(), torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95], device=flat.device), dim=-1
    )
    out = [mean, std, skew, kurt] + [q[i] for i in range(5)]
    for tag in ('mean', 'std', 'skew', 'kurt', 'p05', 'p25', 'p50', 'p75', 'p95'):
        names.extend(f'{prefix}{tag}_{m}' for m in MAP_NAMES)
    return out


def _colour_bins(x: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Occupancy of fixed HSV-style and vegetation/greyness bins."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    mx = x.max(1).values
    mn = x.min(1).values
    chroma = mx - mn
    hue = torch.zeros_like(mx)
    is_r = (mx == r) & (chroma > EPS)
    is_g = (mx == g) & (chroma > EPS)
    is_b = (mx == b) & (chroma > EPS)
    hue = torch.where(is_r, ((g - b) / (chroma + EPS)) % 6.0, hue)
    hue = torch.where(is_g, (b - r) / (chroma + EPS) + 2.0, hue)
    hue = torch.where(is_b, (r - g) / (chroma + EPS) + 4.0, hue)
    hue = hue / 6.0
    sat = chroma / (mx + EPS)
    n = x.shape[0]
    out: list[torch.Tensor] = []
    hue_flat = hue.reshape(n, -1)
    sat_flat = sat.reshape(n, -1)
    val_flat = mx.reshape(n, -1)
    for i in range(8):
        mask = (hue_flat >= i / 8.0) & (hue_flat < (i + 1) / 8.0) & (sat_flat > 0.15)
        out.append(mask.float().mean(-1))
        names.append(f'huebin{i}')
    for i, (lo, hi) in enumerate(((0.0, 0.1), (0.1, 0.2), (0.2, 0.35), (0.35, 1.01))):
        out.append(((sat_flat >= lo) & (sat_flat < hi)).float().mean(-1))
        names.append(f'satbin{i}')
    for i, (lo, hi) in enumerate(((0.0, 0.25), (0.25, 0.45), (0.45, 0.65), (0.65, 0.85), (0.85, 1.01))):
        out.append(((val_flat >= lo) & (val_flat < hi)).float().mean(-1))
        names.append(f'valbin{i}')
    exg = ((2 * g - r - b) / (r + g + b + EPS)).reshape(n, -1)
    for thr in (0.02, 0.06, 0.12):
        out.append((exg > thr).float().mean(-1))
        names.append(f'exgfrac{thr}')
    return out


def _grad_texture(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Gradient-magnitude statistics at four averaging scales."""
    out: list[torch.Tensor] = []
    n = maps.shape[0]
    for factor in (1, 2, 4, 8):
        gm = _grad(_pool(maps, factor)).reshape(n, maps.shape[1], -1)
        mean = gm.mean(-1)
        out.extend((mean, gm.std(-1), torch.quantile(gm.float(), 0.9, dim=-1)))
        names.extend(f'g{factor}{tag}_{m}' for tag in ('mean', 'std', 'p90') for m in MAP_NAMES)
        out.append(gm.std(-1) / (mean + EPS))
        names.extend(f'g{factor}cv_{m}' for m in MAP_NAMES)
    return out


def _structure(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Structure-tensor coherence plus raw and sorted orientation histograms."""
    out: list[torch.Tensor] = []
    n = maps.shape[0]
    bins = 8
    for factor in (1, 2, 4):
        m = _pool(maps, factor)
        gx = (m[:, :, :, 1:] - m[:, :, :, :-1])[:, :, :-1, :]
        gy = (m[:, :, 1:, :] - m[:, :, :-1, :])[:, :, :, :-1]
        jxx = (gx * gx).reshape(n, m.shape[1], -1).mean(-1)
        jyy = (gy * gy).reshape(n, m.shape[1], -1).mean(-1)
        jxy = (gx * gy).reshape(n, m.shape[1], -1).mean(-1)
        trace = jxx + jyy
        disc = torch.sqrt((jxx - jyy) ** 2 + 4 * jxy ** 2 + EPS)
        out.append(disc / (trace + EPS))
        names.extend(f'coh{factor}_{mm}' for mm in MAP_NAMES)
        mag = torch.sqrt(gx * gx + gy * gy + EPS)
        ang = torch.atan2(gy, gx) % np.pi
        index = torch.clamp((ang / np.pi * bins).long(), 0, bins - 1)
        flat_mag = mag.reshape(n, m.shape[1], -1)
        hist = torch.zeros(n, m.shape[1], bins, device=maps.device)
        hist.scatter_add_(2, index.reshape(n, m.shape[1], -1), flat_mag)
        hist = hist / (hist.sum(-1, keepdim=True) + EPS)
        entropy = -(hist * torch.log(hist + EPS)).sum(-1)
        out.append(entropy)
        names.extend(f'oent{factor}_{mm}' for mm in MAP_NAMES)
        srt = torch.sort(hist, dim=-1, descending=True).values
        for i in range(bins):
            out.append(srt[:, :, i])
            names.extend(f'ohs{factor}_{i}_{mm}' for mm in MAP_NAMES)
    return out


def _haar(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Log energies of five Haar wavelet levels in three orientations."""
    out: list[torch.Tensor] = []
    n = maps.shape[0]
    cur = maps
    for level in range(1, 6):
        a = cur[:, :, 0::2, 0::2]
        b = cur[:, :, 0::2, 1::2]
        c = cur[:, :, 1::2, 0::2]
        d = cur[:, :, 1::2, 1::2]
        ll = (a + b + c + d) / 4.0
        for tag, sub in (('lh', (a + b - c - d) / 4.0),
                         ('hl', (a - b + c - d) / 4.0),
                         ('hh', (a - b - c + d) / 4.0)):
            energy = (sub ** 2).reshape(n, maps.shape[1], -1).mean(-1)
            out.append(torch.log(energy + 1e-8))
            names.extend(f'haar{level}{tag}_{m}' for m in MAP_NAMES)
        cur = ll
    return out


def _fourier(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Log power in eight radial rings and six sorted angular wedges."""
    n, c, h, w = maps.shape
    m = maps - maps.reshape(n, c, -1).mean(-1)[:, :, None, None]
    power = torch.fft.rfft2(m).abs() ** 2
    fy = torch.fft.fftfreq(h, device=maps.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=maps.device)[None, :]
    radius = torch.sqrt(fy ** 2 + fx ** 2)
    angle = torch.atan2(fy.expand_as(radius), fx.expand_as(radius)) % np.pi
    total = power.reshape(n, c, -1).sum(-1) + EPS
    out: list[torch.Tensor] = []
    edges = torch.tensor([0.0, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.33, 1.0], device=maps.device)
    for i in range(8):
        mask = ((radius >= edges[i]) & (radius < edges[i + 1])).float()
        share = (power * mask).reshape(n, c, -1).sum(-1) / total
        out.append(torch.log(share + 1e-8))
        names.extend(f'fftring{i}_{mm}' for mm in MAP_NAMES)
    wedges = []
    for i in range(6):
        mask = ((angle >= i * np.pi / 6) & (angle < (i + 1) * np.pi / 6) & (radius > 0.01)).float()
        wedges.append((power * mask).reshape(n, c, -1).sum(-1) / total)
    stacked = torch.stack(wedges, -1)
    stacked = stacked / (stacked.sum(-1, keepdim=True) + EPS)
    srt = torch.sort(stacked, dim=-1, descending=True).values
    for i in range(6):
        out.append(srt[:, :, i])
        names.extend(f'fftwedge{i}_{mm}' for mm in MAP_NAMES)
    return out


def _glcm(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Direction-averaged co-occurrence contrast and correlation per offset."""
    out: list[torch.Tensor] = []
    n, c = maps.shape[0], maps.shape[1]
    flat = maps.reshape(n, c, -1)
    mean = flat.mean(-1)[:, :, None, None]
    std = flat.std(-1)[:, :, None, None] + EPS
    z = (maps - mean) / std
    for d in (1, 2, 4, 8, 16):
        pairs = ((z[:, :, :, d:], z[:, :, :, :-d]), (z[:, :, d:, :], z[:, :, :-d, :]))
        contrast = sum(((p - q) ** 2).reshape(n, c, -1).mean(-1) for p, q in pairs) / 2
        corr = sum((p * q).reshape(n, c, -1).mean(-1) for p, q in pairs) / 2
        out.extend((torch.log(contrast + EPS), corr))
        names.extend(f'glcm{d}con_{m}' for m in MAP_NAMES)
        names.extend(f'glcm{d}cor_{m}' for m in MAP_NAMES)
    return out


def _lacunarity(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Box-mean dispersion across five box sizes."""
    out: list[torch.Tensor] = []
    n, c = maps.shape[0], maps.shape[1]
    for box in (2, 4, 8, 16, 32):
        blocks = _pool(maps, box).reshape(n, c, -1)
        mean = blocks.mean(-1)
        out.append(blocks.std(-1) / (mean.abs() + EPS))
        names.extend(f'lac{box}_{m}' for m in MAP_NAMES)
    return out


def _layout(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Coarse spatial-layout contrasts that survive image flips."""
    out: list[torch.Tensor] = []
    n, c = maps.shape[0], maps.shape[1]
    grid = _pool(maps, maps.shape[-1] // 4)
    blocks = grid.reshape(n, c, -1)
    out.append(blocks.std(-1))
    names.extend(f'layoutstd_{m}' for m in MAP_NAMES)
    rows = grid.mean(-1)
    cols = grid.mean(-2)
    out.append((rows[:, :, :2].mean(-1) - rows[:, :, 2:].mean(-1)).abs())
    names.extend(f'layoutvert_{m}' for m in MAP_NAMES)
    out.append((cols[:, :, :2].mean(-1) - cols[:, :, 2:].mean(-1)).abs())
    names.extend(f'layouthoriz_{m}' for m in MAP_NAMES)
    centre = grid[:, :, 1:3, 1:3].reshape(n, c, -1).mean(-1)
    out.append(centre - blocks.mean(-1))
    names.extend(f'layoutcentre_{m}' for m in MAP_NAMES)
    srt = torch.sort(blocks, dim=-1).values
    out.append(srt[:, :, -1] - srt[:, :, 0])
    names.extend(f'layoutrange_{m}' for m in MAP_NAMES)
    return out


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 256) -> tuple[np.ndarray, list[str]]:
    """Extract the full GPU pool for a stack of uint8 ``(N, 3, H, W)`` images."""
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                            device=device).float() / 255.0
        maps = _maps(x)
        local: list[str] = []
        parts = (
            _moments(maps.reshape(maps.shape[0], maps.shape[1], -1), '', local)
            + _grad_texture(maps, local)
            + _structure(maps, local)
            + _haar(maps, local)
            + _fourier(maps, local)
            + _glcm(maps, local)
            + _lacunarity(maps, local)
            + _layout(maps, local)
        )
        block = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1)
        extra = _colour_bins(x, local)
        block = torch.cat([block, torch.stack(extra, dim=1)], dim=1)
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
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu_pool.npy'), features)
        print(split, features.shape, flush=True)
    with open(os.path.join(CACHE_DIR, 'resisc45_gpu_pool_names.txt'), 'w') as handle:
        handle.write('\n'.join(names) + '\n')
    print('names', len(names))
