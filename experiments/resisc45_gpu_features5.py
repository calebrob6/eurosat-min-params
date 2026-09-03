#!/usr/bin/env python
"""Fifth zero-parameter RESISC45 pool: line geometry and region shape.

The union-of-supports head sits at 80.0% test against its list's 81.0% linear
ceiling, and the residual confusions (``resisc45_union_failures_pairs.csv``)
are the same pairs that the dense 4,126-column head and a wide MLP on the same
columns also confuse: church/palace, basketball/tennis court, lake/river,
rectangular farmland/terrace, railway/railway station, baseball
diamond/ground-track field.  Those are *geometry* distinctions that none of the
four pools measures: a basketball court has curved white markings and a tennis
court only straight ones, a river is an elongated water region touching two
borders and a lake a compact one, a terrace has curved contour edges and a
field straight ones.  This module adds three families, all fixed arithmetic on
one image so the extractor contributes no learned parameters:

* ``line`` -- thin bright (white top-hat) and thin dark (black top-hat) line
  pixels: their fraction, a rotation-normalised orientation histogram, its
  entropy and rectilinearity (mass in the dominant and orthogonal bins),
  *curvature* (orientation change a few pixels along the tangent, which is
  zero on straight and crossing lines and positive on arcs), and for bright
  lines the colour composition of a ring around them (the court surface next
  to its markings);
* ``curv`` -- the same curvature and straightness statistics for ordinary
  edges of the panchromatic and excess-green maps at three scales;
* ``cc``   -- connected components of seven fixed colour/brightness masks at
  128 x 128: component count, largest-component area, elongation, box fill,
  border sides touched, boundary-to-area ratio, second-largest area and the
  entropy of the component area distribution.
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_gpu_features4 import _colour_masks, _maps

from src.cache import CACHE_DIR

EPS = 1e-6
ORIENT_BINS = 8
CURV_OFFSETS = (4, 8)
CURV_SCALES = (1, 2, 4)
CC_RES = 128
CC_MIN_AREA = 4
CC_MASKS = ('water', 'veg', 'tan', 'bright', 'dark', 'red', 'mark')


def _sobel(m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                      device=m.device).view(1, 1, 3, 3) / 8.0
    gx = F.conv2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(m, (1, 1, 1, 1), mode='replicate'), kx.transpose(2, 3))
    return gx, gy


def _box(m: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(m, k, stride=1, padding=k // 2, count_include_pad=False)


def _orientation_field(m: torch.Tensor, k: int = 5):
    """Smoothed structure tensor: doubled-angle unit vector, coherence, energy."""
    gx, gy = _sobel(m)
    c = _box(gx * gx - gy * gy, k)
    s = _box(2 * gx * gy, k)
    e = _box(gx * gx + gy * gy, k)
    norm = torch.sqrt(c * c + s * s + EPS)
    return c / norm, s / norm, norm / (e + EPS), e, gx, gy


def _tangent_shift(nc: torch.Tensor, ns: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the doubled-angle field ``d`` pixels along the local line direction.

    The doubled angle ``2t = atan2(ns, nc)`` is the *gradient* orientation;
    the line runs perpendicular to it, so the tangent is ``(-sin t, cos t)``.
    """
    _, _, h, w = nc.shape
    theta = 0.5 * torch.atan2(ns, nc)
    tx = -torch.sin(theta)
    ty = torch.cos(theta)
    ys, xs = torch.meshgrid(torch.arange(h, device=nc.device, dtype=nc.dtype),
                            torch.arange(w, device=nc.device, dtype=nc.dtype), indexing='ij')
    gx = (xs[None, None] + d * tx) / (w - 1) * 2 - 1
    gy = (ys[None, None] + d * ty) / (h - 1) * 2 - 1
    grid = torch.stack((gx[:, 0], gy[:, 0]), dim=-1)
    field = torch.cat((nc, ns), dim=1)
    shifted = F.grid_sample(field, grid, mode='bilinear', padding_mode='border',
                            align_corners=True)
    return shifted[:, :1], shifted[:, 1:]


def _masked_mean(v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    n = v.shape[0]
    flat_v = v.reshape(n, -1)
    flat_m = mask.reshape(n, -1)
    return (flat_v * flat_m).sum(1) / (flat_m.sum(1) + EPS)


def _orientation_stats(nc, ns, coh, mask, tag: str, names: list[str], out: list[torch.Tensor],
                       offsets=CURV_OFFSETS, histogram: bool = True, frac: bool = True) -> None:
    """Orientation histogram, entropy, rectilinearity, coherence and curvature on a mask."""
    n = nc.shape[0]
    if frac:
        out.append(mask.reshape(n, -1).mean(1))
        names.append(f'{tag}frac')
    theta = (0.5 * torch.atan2(ns, nc)) % np.pi  # [0, pi)
    bins = (theta / np.pi * ORIENT_BINS).long().clamp(0, ORIENT_BINS - 1)
    hist = torch.zeros(n, ORIENT_BINS, device=nc.device)
    hist.scatter_add_(1, bins.reshape(n, -1), mask.reshape(n, -1))
    hist = hist / (hist.sum(1, keepdim=True) + EPS)
    dom = hist.argmax(1)
    idx = (torch.arange(ORIENT_BINS, device=nc.device)[None] + dom[:, None]) % ORIENT_BINS
    rolled = hist.gather(1, idx)
    if histogram:
        for i in range(ORIENT_BINS):
            out.append(rolled[:, i])
            names.append(f'{tag}ohist{i}')
    out.append(-(hist * torch.log(hist + EPS)).sum(1))
    names.append(f'{tag}oent')
    out.append(rolled[:, 0] + rolled[:, ORIENT_BINS // 2])
    names.append(f'{tag}rect')
    out.append(_masked_mean(coh, mask))
    names.append(f'{tag}coh')
    for d in offsets:
        sc, ss = _tangent_shift(nc, ns, d)
        agreement = (nc * sc + ns * ss).clamp(-1, 1)
        curvature = 1.0 - agreement
        out.append(_masked_mean(curvature, mask))
        names.append(f'{tag}curv{d}')
        out.append(_masked_mean((curvature > 0.3).float(), mask))
        names.append(f'{tag}bent{d}')


def _tophat(pan: torch.Tensor, k: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """White top-hat (pan - opening) and black top-hat (closing - pan)."""
    pad = k // 2
    erode = -F.max_pool2d(-pan, k, stride=1, padding=pad)
    opening = F.max_pool2d(erode, k, stride=1, padding=pad)
    dilate = F.max_pool2d(pan, k, stride=1, padding=pad)
    closing = -F.max_pool2d(-dilate, k, stride=1, padding=pad)
    return pan - opening, closing - pan


def _line(x: torch.Tensor, maps: torch.Tensor, names: list[str], out: list[torch.Tensor]) -> torch.Tensor:
    pan = maps[:, :1]
    white, black = _tophat(pan)
    bright = ((white > 0.10) & (pan > 0.45)).float()
    dark = ((black > 0.10) & (pan < 0.55)).float()
    nc, ns, coh, _, _, _ = _orientation_field(pan, 5)
    _orientation_stats(nc, ns, coh, bright, 'lineb_', names, out)
    _orientation_stats(nc, ns, coh, dark, 'lined_', names, out, histogram=False)
    # Colour of the ring around bright lines (the surface the markings are on).
    ring = (F.max_pool2d(bright, 9, stride=1, padding=4) - bright).clamp(0, 1)
    colour, tags = _colour_masks(x)
    n = x.shape[0]
    ring_flat = ring.reshape(n, 1, -1)
    cover = ring_flat.sum(-1) + EPS
    frac = (colour.reshape(n, colour.shape[1], -1) * ring_flat).sum(-1) / cover
    for i, t in enumerate(tags):
        out.append(frac[:, i])
        names.append(f'linering_{t}')
    for j, t in enumerate(('pan', 'exg', 'sat', 'gr')):
        out.append(_masked_mean(maps[:, j:j + 1], ring))
        names.append(f'linering_{t}')
    out.append(_masked_mean(pan, bright))
    names.append('lineb_pan')
    out.append(_masked_mean(white, bright))
    names.append('lineb_contrast')
    out.append(_masked_mean(black, dark))
    names.append('lined_contrast')
    return bright


def _curv(maps: torch.Tensor, names: list[str], out: list[torch.Tensor]) -> None:
    n = maps.shape[0]
    for j, tag in ((0, 'pan'), (1, 'exg')):
        m = maps[:, j:j + 1]
        for s in CURV_SCALES:
            ms = F.avg_pool2d(m, s) if s > 1 else m
            nc, ns, coh, e, _, _ = _orientation_field(ms, 5)
            flat = e.reshape(n, -1)
            thresh = flat.quantile(0.8, dim=1).view(n, 1, 1, 1)
            edge = (e > thresh).float()
            _orientation_stats(nc, ns, coh, edge, f'curv_{tag}{s}_', names, out,
                               offsets=(4,), histogram=False, frac=False)


def _cc_masks(x: torch.Tensor, maps: torch.Tensor, bright: torch.Tensor) -> torch.Tensor:
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    pan, exg, sat = maps[:, 0], maps[:, 1], maps[:, 2]
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
    water = ((b >= g) & (b > r) & (pan < 0.65)) | ((sat < 0.12) & (pan < 0.3))
    veg = exg > 0.08
    tan = (hue < 0.14) & (sat > 0.12) & (sat < 0.55) & (pan > 0.35)
    brightm = pan > 0.7
    darkm = pan < 0.25
    red = ((hue < 0.08) | (hue > 0.92)) & (sat > 0.35) & (pan > 0.2)
    masks = torch.stack((water, veg, tan, brightm, darkm, red), dim=1).float()
    masks = (F.adaptive_avg_pool2d(masks, CC_RES) >= 0.5)
    mark = F.adaptive_max_pool2d(bright, CC_RES) > 0.5
    return torch.cat((masks, mark), dim=1)


CC_STATS = ('frac', 'count', 'largest', 'elong', 'fill', 'border', 'boundary', 'second',
            'areaent', 'largest_dist')


def _cc_one(item: np.ndarray) -> np.ndarray:
    """Component statistics for one image's ``(M, H, W)`` bool masks."""
    result = np.zeros((item.shape[0], len(CC_STATS)), dtype=np.float32)
    h, w = item.shape[1:]
    total = float(h * w)
    for j in range(item.shape[0]):
        mask = item[j]
        frac = mask.mean()
        result[j, 0] = frac
        if frac == 0.0:
            continue
        lab, num = ndimage.label(mask)
        if num == 0:
            continue
        sizes = ndimage.sum(mask, lab, np.arange(1, num + 1))
        keep = sizes >= CC_MIN_AREA
        sizes_kept = sizes[keep]
        result[j, 1] = keep.sum()
        if sizes_kept.size == 0:
            continue
        order = np.argsort(-sizes)
        big = order[0] + 1
        result[j, 2] = sizes[order[0]] / total
        if order.size > 1:
            result[j, 7] = sizes[order[1]] / total
        p = sizes_kept / sizes_kept.sum()
        result[j, 8] = -(p * np.log(p + 1e-9)).sum()
        sl = ndimage.find_objects(lab == big, max_label=1)[0]
        comp = lab[sl] == big
        ys, xs = np.nonzero(comp)
        area = float(ys.size)
        ys = ys.astype(np.float64)
        xs = xs.astype(np.float64)
        cyy = ((ys - ys.mean()) ** 2).mean()
        cxx = ((xs - xs.mean()) ** 2).mean()
        cxy = ((ys - ys.mean()) * (xs - xs.mean())).mean()
        tr = cyy + cxx
        det = cyy * cxx - cxy * cxy
        disc = max(tr * tr / 4 - det, 0.0) ** 0.5
        l1 = tr / 2 + disc
        l2 = tr / 2 - disc
        result[j, 3] = 1.0 - l2 / (l1 + 1e-9)
        bh = sl[0].stop - sl[0].start
        bw = sl[1].stop - sl[1].start
        result[j, 4] = area / float(bh * bw)
        result[j, 5] = float(sl[0].start == 0) + float(sl[0].stop == h) + \
            float(sl[1].start == 0) + float(sl[1].stop == w)
        interior = ndimage.binary_erosion(comp)
        result[j, 6] = (area - interior.sum()) / (area ** 0.5)
        # Distance of the largest component's centroid from the image centre.
        cy = ys.mean() + sl[0].start
        cx = xs.mean() + sl[1].start
        result[j, 9] = (((cy / h - 0.5) ** 2 + (cx / w - 0.5) ** 2) ** 0.5)
    return result


def _cc(pool: Pool, masks: torch.Tensor, names: list[str]) -> np.ndarray:
    items = list(masks.cpu().numpy().astype(bool))
    stats = np.stack(pool.map(_cc_one, items, chunksize=8), axis=0)  # (N, M, S)
    if not names:
        for m in CC_MASKS:
            for s in CC_STATS:
                names.append(f'cc_{m}_{s}')
    return stats.reshape(stats.shape[0], -1)


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 256,
            workers: int = 32) -> tuple[np.ndarray, list[str]]:
    """Extract the fifth pool for uint8 ``(N, 3, H, W)`` images."""
    chunks: list[np.ndarray] = []
    names: list[str] = []
    cc_names: list[str] = []
    with Pool(workers) as pool:
        for start in range(0, len(images), batch):
            x = torch.as_tensor(np.array(images[start:start + batch]),
                                device=device).float() / 255.0
            local: list[str] = []
            out: list[torch.Tensor] = []
            with torch.no_grad():
                maps = _maps(x)
                bright = _line(x, maps, local, out)
                _curv(maps, local, out)
                cc_masks = _cc_masks(x, maps, bright)
            block = torch.stack([p.reshape(p.shape[0]) for p in out], dim=1).float().cpu().numpy()
            cc_block = _cc(pool, cc_masks, cc_names)
            chunks.append(np.concatenate((block, cc_block), axis=1))
            if not names:
                names = local + cc_names
            if (start // batch) % 10 == 0:
                print(f'  {start + len(x)}/{len(images)}', flush=True)
    result = np.concatenate(chunks, axis=0)
    if result.shape[1] != len(names):
        raise ValueError(f'{result.shape[1]} columns but {len(names)} names')
    return result, names


if __name__ == '__main__':
    import time
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    for split in ('train', 'val', 'test'):
        images = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_256.npy'), mmap_mode='r')
        if limit:
            images = images[:limit]
        t0 = time.time()
        features, names = extract(images)
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        print(split, features.shape, f'{time.time() - t0:.0f}s', flush=True)
        if limit:
            const = [names[i] for i in range(features.shape[1]) if features[:, i].std() < 1e-8]
            print('constant columns:', const)
            print('abs max:', float(np.abs(features).max()))
            break
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu5_pool.npy'), features)
    if not limit:
        with open(os.path.join(CACHE_DIR, 'resisc45_gpu5_pool_names.txt'), 'w') as handle:
            handle.write('\n'.join(names) + '\n')
        print('names', len(names))
