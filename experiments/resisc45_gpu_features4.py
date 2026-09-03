#!/usr/bin/env python
"""Fourth zero-parameter RESISC45 pool: random local features and colour-conditioned texture.

The merged 2,084-column pool has a linear ceiling of 79.4% test, and a wide
ReLU head on the same columns reads 82.6%, so the pool is short of both
*information* and *nonlinearity*.  Every column so far is a global statistic of
a hand-designed map; the classes that remain confused (palace/church,
basketball/tennis court, roundabout/intersection, railway/railway_station) are
told apart by *which local patterns occur*, not by how much of each global
quantity there is.  This module adds four families that summarise local
pattern occurrence directly, all deterministic given a fixed seed so the
extractor still contributes no learned parameters:

* ``rconv`` -- MOSAIKS-style random convolutional features: seeded Gaussian
  ``5 x 5 x 3`` filters at three image scales, ReLU on both signs, global mean
  and max pooling.  Each filter is a random local pattern and the pooled
  response is how strongly and how often it occurs;
* ``rtex``  -- random texton histograms: the sign pattern of six seeded random
  ``3 x 3`` filters hashes every pixel into one of 64 codes, whose histogram is
  a rotation-*variant* cousin of the LBP family that already dominates the
  selected columns, computed on two maps at three scales;
* ``ctex``  -- colour-conditioned texture: gradient energy, blob compactness and
  brightness *inside* each of eleven fixed hue/grey masks, the joint statistic
  that a product of a colour fraction and a global texture value cannot give
  (a tennis court is a flat green or blue rectangle with white lines; a
  basketball court is a flat red or grey one);
* ``cell``  -- sorted 4 x 4 grid-cell statistics of four maps and their
  gradient energy: the order statistics of the cells are rotation- and
  translation-insensitive summaries of how *unevenly* a property is laid out.
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

from src.cache import CACHE_DIR

EPS = 1e-6
SEED = 0
RCONV_FILTERS = 128
RCONV_PATCH = 5
RCONV_SCALES = (2, 4, 8)  # average-pool factors from 256 x 256
RTEX_BITS = 6
RTEX_SCALES = (2, 4, 8)
RTEX_MAPS = ('pan', 'exg')
CELL_GRID = 4
CELL_MAPS = ('pan', 'exg', 'sat', 'gr')


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


def _grad_magnitude(m: torch.Tensor) -> torch.Tensor:
    gx = F.pad(m[:, :, :, 1:] - m[:, :, :, :-1], (0, 1))
    gy = F.pad(m[:, :, 1:, :] - m[:, :, :-1, :], (0, 0, 0, 1))
    return torch.sqrt(gx ** 2 + gy ** 2 + EPS)


def _rconv_filters(device: str) -> torch.Tensor:
    """Seeded zero-mean unit-norm Gaussian ``5 x 5 x 3`` filters."""
    rng = np.random.default_rng(SEED)
    w = rng.standard_normal((RCONV_FILTERS, 3, RCONV_PATCH, RCONV_PATCH))
    w -= w.mean(axis=(1, 2, 3), keepdims=True)
    w /= np.linalg.norm(w.reshape(RCONV_FILTERS, -1), axis=1)[:, None, None, None]
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def _rtex_filters(device: str) -> torch.Tensor:
    """Seeded zero-mean random ``3 x 3`` filters, one per hash bit."""
    rng = np.random.default_rng(SEED + 1)
    w = rng.standard_normal((RTEX_BITS, 1, 3, 3))
    w -= w.mean(axis=(1, 2, 3), keepdims=True)
    return torch.as_tensor(w, dtype=torch.float32, device=device)


def _rconv(x: torch.Tensor, filters: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Random convolutional features: +/- ReLU, mean and max pooled, per scale."""
    out: list[torch.Tensor] = []
    # Per-channel centring by a fixed constant keeps the response independent of
    # overall brightness without any data-derived statistic.
    xc = x - 0.5
    for factor in RCONV_SCALES:
        xs = F.avg_pool2d(xc, factor)
        resp = F.conv2d(xs, filters)
        n = resp.shape[0]
        for sign, tag in ((1.0, 'p'), (-1.0, 'n')):
            act = torch.relu(sign * resp).reshape(n, RCONV_FILTERS, -1)
            out.append(act.mean(-1))
            names.extend(f'rconv{factor}{tag}mean_{i}' for i in range(RCONV_FILTERS))
            out.append(act.amax(-1))
            names.extend(f'rconv{factor}{tag}max_{i}' for i in range(RCONV_FILTERS))
    return out


def _rtex(maps: torch.Tensor, filters: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Histogram of the sign code of random 3 x 3 filters, per map and scale."""
    out: list[torch.Tensor] = []
    weights = torch.as_tensor([2 ** b for b in range(RTEX_BITS)], dtype=torch.float32,
                              device=maps.device)
    for mi, mname in enumerate(RTEX_MAPS):
        m = maps[:, mi:mi + 1]
        for factor in RTEX_SCALES:
            ms = F.avg_pool2d(m, factor)
            resp = F.conv2d(ms, filters)
            code = ((resp > 0).float() * weights[None, :, None, None]).sum(1)
            n = code.shape[0]
            hist = torch.zeros(n, 2 ** RTEX_BITS, device=maps.device)
            hist.scatter_add_(1, code.reshape(n, -1).long(),
                              torch.ones(n, code[0].numel(), device=maps.device))
            hist = hist / code[0].numel()
            out.append(hist)
            names.extend(f'rtex{factor}_{mname}_{c}' for c in range(2 ** RTEX_BITS))
    return out


def _colour_masks(x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Eleven fixed masks: eight hue bins for saturated pixels, three grey levels."""
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
    masks, tags = [], []
    for i in range(8):
        masks.append((hue >= i / 8.0) & (hue < (i + 1) / 8.0) & (sat > 0.15))
        tags.append(f'hue{i}')
    grey = sat <= 0.15
    for i, (lo, hi) in enumerate(((0.0, 0.35), (0.35, 0.65), (0.65, 1.01))):
        masks.append(grey & (mx >= lo) & (mx < hi))
        tags.append(f'grey{i}')
    return torch.stack(masks, dim=1).float(), tags


def _ctex(x: torch.Tensor, maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Texture and brightness inside each fixed colour mask."""
    masks, tags = _colour_masks(x)
    n = x.shape[0]
    pan = maps[:, :1]
    grad = _grad_magnitude(pan)
    grad4 = _grad_magnitude(F.avg_pool2d(pan, 4))
    smooth = F.avg_pool2d(masks, 9, stride=1, padding=4, count_include_pad=False)
    masks4 = F.avg_pool2d(masks, 4)
    flat = masks.reshape(n, masks.shape[1], -1)
    frac = flat.mean(-1)
    out: list[torch.Tensor] = []
    out.append(frac)
    names.extend(f'ctexfrac_{t}' for t in tags)
    cover = frac + EPS
    out.append((flat * grad.reshape(n, 1, -1)).mean(-1) / cover)
    names.extend(f'ctexgrad_{t}' for t in tags)
    out.append((masks4.reshape(n, masks.shape[1], -1) * grad4.reshape(n, 1, -1)).mean(-1)
               / (masks4.reshape(n, masks.shape[1], -1).mean(-1) + EPS))
    names.extend(f'ctexgrad4_{t}' for t in tags)
    # Mean smoothed occupancy inside the mask: 1 for solid blobs, small for
    # scattered pixels -- a compactness of the colour region.
    out.append((flat * smooth.reshape(n, masks.shape[1], -1)).mean(-1) / cover)
    names.extend(f'ctexblob_{t}' for t in tags)
    out.append((flat * pan.reshape(n, 1, -1)).mean(-1) / cover)
    names.extend(f'ctexpan_{t}' for t in tags)
    # Coarse-grid spread: how many of the 16 cells the colour occupies.
    cells = F.adaptive_avg_pool2d(masks, CELL_GRID).reshape(n, masks.shape[1], -1)
    out.append((cells > 0.05).float().mean(-1))
    names.extend(f'ctexspread_{t}' for t in tags)
    return out


def _cell(maps: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Order statistics of 4 x 4 grid-cell means of each map and its gradient energy."""
    out: list[torch.Tensor] = []
    n = maps.shape[0]
    fields = torch.cat((maps, _grad_magnitude(maps)), dim=1)
    tags = list(CELL_MAPS) + [f'{m}grad' for m in CELL_MAPS]
    cells = F.adaptive_avg_pool2d(fields, CELL_GRID).reshape(n, fields.shape[1], -1)
    sorted_cells = cells.sort(-1).values
    mean = cells.mean(-1, keepdim=True)
    std = cells.std(-1) + EPS
    for j, tag in enumerate(tags):
        s = sorted_cells[:, j]
        out.append(s[:, 0])
        names.append(f'cellmin_{tag}')
        out.append(s[:, -1])
        names.append(f'cellmax_{tag}')
        out.append(s[:, -1] - s[:, 0])
        names.append(f'cellrange_{tag}')
        out.append(std[:, j])
        names.append(f'cellstd_{tag}')
        out.append((s[:, -1] - s[:, -2]) / (std[:, j]))
        names.append(f'celltopgap_{tag}')
        out.append((s[:, 1] - s[:, 0]) / (std[:, j]))
        names.append(f'cellbotgap_{tag}')
        out.append(((cells[:, j] - mean[:, j]) ** 3).mean(-1) / std[:, j] ** 3)
        names.append(f'cellskew_{tag}')
    return out


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 128) -> tuple[np.ndarray, list[str]]:
    """Extract the fourth pool for uint8 ``(N, 3, H, W)`` images."""
    rconv_filters = _rconv_filters(device)
    rtex_filters = _rtex_filters(device)
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                            device=device).float() / 255.0
        local: list[str] = []
        maps = _maps(x)
        parts = (
            _rconv(x, rconv_filters, local)
            + _rtex(maps, rtex_filters, local)
            + _ctex(x, maps, local)
            + _cell(maps, local)
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
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu4_pool.npy'), features)
        print(split, features.shape, flush=True)
    with open(os.path.join(CACHE_DIR, 'resisc45_gpu4_pool_names.txt'), 'w') as handle:
        handle.write('\n'.join(names) + '\n')
    print('names', len(names))
