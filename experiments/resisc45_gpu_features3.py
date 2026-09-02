#!/usr/bin/env python
"""Third zero-parameter RESISC45 pool: elongated objects and scene layout.

The failure analysis in `resisc45_failure_analysis.py` showed that the residual
RESISC45 errors are *object-layout* distinctions rather than texture ones: a
bridge is a river plus one elongated crossing structure, a harbour is water plus
repeated docked hulls, a roundabout is an intersection whose arms meet on a
ring.  The first two pools summarise intensity distributions, isotropic texture,
and oriented *energy*, but nothing in them counts or measures discrete elongated
objects.

This module adds five such families, all deterministic arithmetic on a single
image so the extractor still contributes no learned parameters:

* ``radon``  -- projection profiles at twelve fixed angles, which turn a long
  straight structure into a single profile peak and a repeated parallel set into
  a periodic profile;
* ``ridge``  -- Hessian ridge (vesselness) strength, orientation coherence, and
  linearity at three fixed scales;
* ``run``    -- directional run lengths of thresholded ridge, edge, and dark
  masks, which measure how *far* a thin structure continues;
* ``blob``   -- difference-of-Gaussian local-maximum counts and the second
  moments of the maximum cloud, which count repeated discrete objects and
  measure whether they lie along a line;
* ``polar``  -- rotational/mirror self-similarity and log-polar radial and
  angular profiles, which separate ring layouts from radial-arm layouts;
* ``mask``   -- second moments, border spanning, and profile breaks of
  percentile-thresholded regions.
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
FIELD_NAMES = MAP_NAMES + ('edge',)
RES = 128
N_ANGLES = 12
RIDGE_SCALES = (2.0, 4.0, 8.0)
BLOB_SCALES = (2.0, 4.0, 8.0)
RUN_DIRS = ((0, 1), (1, 0), (1, 1), (1, -1))
RUN_MAX = 48
POLAR_RADII = 32
POLAR_ANGLES = 64


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


def _gauss_kernel(sigma: float, device: str) -> torch.Tensor:
    radius = int(np.ceil(3.0 * sigma))
    t = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    k = torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


def _blur(m: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Separable Gaussian blur with replicate padding, applied per channel."""
    c = m.shape[1]
    radius = (kernel.numel() - 1) // 2
    kx = kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = kernel.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    m = F.conv2d(F.pad(m, (radius, radius, 0, 0), mode='replicate'), kx, groups=c)
    return F.conv2d(F.pad(m, (0, 0, radius, radius), mode='replicate'), ky, groups=c)


def _shift(t: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Translate by ``(dy, dx)`` with zero fill (no wraparound)."""
    h, w = t.shape[-2], t.shape[-1]
    out = torch.zeros_like(t)
    out[..., max(dy, 0):h - max(-dy, 0), max(dx, 0):w - max(-dx, 0)] = \
        t[..., max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


def _quantile(t: torch.Tensor, q: float) -> torch.Tensor:
    """Per-(image, channel) quantile of a ``(N, C, H, W)`` tensor."""
    n, c = t.shape[0], t.shape[1]
    return torch.quantile(t.reshape(n, c, -1).float(), q, dim=-1)[:, :, None, None]


def _moments2(mass: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor):
    """Centroid offset and sorted principal spreads of a nonnegative mass map."""
    n, c = mass.shape[0], mass.shape[1]
    flat = mass.reshape(n, c, -1)
    total = flat.sum(-1) + EPS
    my = (flat * yy.reshape(1, 1, -1)).sum(-1) / total
    mx = (flat * xx.reshape(1, 1, -1)).sum(-1) / total
    cyy = (flat * yy.reshape(1, 1, -1) ** 2).sum(-1) / total - my ** 2
    cxx = (flat * xx.reshape(1, 1, -1) ** 2).sum(-1) / total - mx ** 2
    cxy = (flat * (yy * xx).reshape(1, 1, -1)).sum(-1) / total - my * mx
    half = (cyy + cxx) / 2.0
    root = torch.sqrt(torch.clamp(((cyy - cxx) / 2.0) ** 2 + cxy ** 2, min=0.0))
    big = torch.sqrt(torch.clamp(half + root, min=0.0))
    small = torch.sqrt(torch.clamp(half - root, min=0.0))
    offset = torch.sqrt(my ** 2 + mx ** 2)
    return offset, big, small, small / (big + EPS)


def _rotation_grids(size: int, device: str) -> torch.Tensor:
    """One sampling grid per fixed projection angle, shape ``(A, H, W, 2)``."""
    lin = torch.linspace(-1.0, 1.0, size, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    grids = []
    for a in range(N_ANGLES):
        theta = np.pi * a / N_ANGLES
        cos, sin = float(np.cos(theta)), float(np.sin(theta))
        grids.append(torch.stack((cos * xx - sin * yy, sin * xx + cos * yy), dim=-1))
    return torch.stack(grids)


def _polar_grid(device: str) -> torch.Tensor:
    """Log-spaced radial by uniform angular sampling grid, shape ``(1, R, A, 2)``."""
    radii = torch.logspace(np.log10(0.05), 0.0, POLAR_RADII, device=device)
    angles = torch.linspace(0.0, 2.0 * np.pi, POLAR_ANGLES + 1, device=device)[:-1]
    rr = radii[:, None]
    aa = angles[None, :]
    return torch.stack((rr * torch.cos(aa), rr * torch.sin(aa)), dim=-1)[None]


def _sorted_share(values: torch.Tensor, count: int) -> torch.Tensor:
    share = values / (values.sum(-1, keepdim=True) + EPS)
    return torch.sort(share, dim=-1, descending=True).values[..., :count]


def _radon(fields: torch.Tensor, grids: torch.Tensor, disk: torch.Tensor,
           names: list[str]) -> list[torch.Tensor]:
    """Projection-profile statistics at twelve fixed angles.

    A long straight structure projects to one sharp profile peak at the angle
    parallel to it, and a set of parallel structures projects to a periodic
    profile, so peak height, peak count, and profile periodicity measure exactly
    the discrete elongated objects the texture pools cannot see.
    """
    n, c, h, w = fields.shape
    columns = disk.sum(0) + EPS
    centred = fields - (fields * disk).sum((-2, -1), keepdim=True) / (disk.sum() + EPS)
    masked = centred * disk
    profiles = []
    for a in range(N_ANGLES):
        grid = grids[a][None].expand(n, -1, -1, -1)
        rotated = F.grid_sample(masked, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=False)
        profiles.append(rotated.sum(-2) / columns[None, None])
    profile = torch.stack(profiles, dim=2)                       # (N, C, A, W)
    energy = profile.var(-1)                                     # (N, C, A)
    out: list[torch.Tensor] = []
    out.append(torch.log(energy.mean(-1) + 1e-8))
    names.extend(f'radonE_{f}' for f in FIELD_NAMES)
    share = _sorted_share(energy, 3)
    for i in range(3):
        out.append(share[..., i])
        names.extend(f'radonS{i}_{f}' for f in FIELD_NAMES)
    norm = energy / (energy.sum(-1, keepdim=True) + EPS)
    out.append(-(norm * torch.log(norm + EPS)).sum(-1))
    names.extend(f'radonEnt_{f}' for f in FIELD_NAMES)
    out.append(energy.max(-1).values / (energy.mean(-1) + EPS))
    names.extend(f'radonAniso_{f}' for f in FIELD_NAMES)
    theta = torch.tensor([np.pi * a / N_ANGLES for a in range(N_ANGLES)],
                         device=fields.device, dtype=torch.float32)
    for harmonic in (2, 4):  # 2 = one dominant direction, 4 = orthogonal grid
        cos = (norm * torch.cos(harmonic * theta)).sum(-1)
        sin = (norm * torch.sin(harmonic * theta)).sum(-1)
        out.append(torch.sqrt(cos ** 2 + sin ** 2))
        names.extend(f'radonH{harmonic}_{f}' for f in FIELD_NAMES)
    best = energy.argmax(-1)[:, :, None, None].expand(-1, -1, -1, w)
    peak = torch.gather(profile, 2, best)[:, :, 0]                # (N, C, W)
    z = (peak - peak.mean(-1, keepdim=True)) / (peak.std(-1, keepdim=True) + EPS)
    out.append(z.max(-1).values)
    names.extend(f'radonPk_{f}' for f in FIELD_NAMES)
    out.append(-z.min(-1).values)
    names.extend(f'radonTr_{f}' for f in FIELD_NAMES)
    out.append((z.abs() > 2.0).float().mean(-1))
    names.extend(f'radonNpk_{f}' for f in FIELD_NAMES)
    out.append((z ** 4).mean(-1))
    names.extend(f'radonKur_{f}' for f in FIELD_NAMES)
    spectrum = torch.fft.rfft(z, dim=-1).abs() ** 2
    spec = spectrum[..., 1:] / (spectrum[..., 1:].sum(-1, keepdim=True) + EPS)
    out.append(spec.max(-1).values)
    names.extend(f'radonFpk_{f}' for f in FIELD_NAMES)
    out.append(spec.argmax(-1).float())
    names.extend(f'radonFfr_{f}' for f in FIELD_NAMES)
    out.append(-(spec * torch.log(spec + EPS)).sum(-1))
    names.extend(f'radonFent_{f}' for f in FIELD_NAMES)
    strength = profile.abs().max(-1).values                       # (N, C, A)
    top2 = torch.sort(strength, dim=-1, descending=True).values[..., :2]
    out.append(top2[..., 0])
    names.extend(f'radonL0_{f}' for f in FIELD_NAMES)
    out.append(top2[..., 1] / (top2[..., 0] + EPS))
    names.extend(f'radonL1r_{f}' for f in FIELD_NAMES)
    return out


def _ridge(maps: torch.Tensor, kernels: dict[float, torch.Tensor],
           names: list[str]) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Hessian ridge strength, coherence, and linearity at three fixed scales."""
    out: list[torch.Tensor] = []
    keep: list[torch.Tensor] = []
    for sigma in RIDGE_SCALES:
        smooth = F.pad(_blur(maps, kernels[sigma]), (1, 1, 1, 1), mode='replicate')
        centre = smooth[:, :, 1:-1, 1:-1]
        ixx = smooth[:, :, 1:-1, 2:] + smooth[:, :, 1:-1, :-2] - 2 * centre
        iyy = smooth[:, :, 2:, 1:-1] + smooth[:, :, :-2, 1:-1] - 2 * centre
        ixy = 0.25 * (smooth[:, :, 2:, 2:] + smooth[:, :, :-2, :-2]
                      - smooth[:, :, 2:, :-2] - smooth[:, :, :-2, 2:])
        half = (ixx + iyy) / 2.0
        root = torch.sqrt(torch.clamp(((ixx - iyy) / 2.0) ** 2 + ixy ** 2, min=0.0))
        lo, hi = half - root, half + root
        big = torch.where(lo.abs() >= hi.abs(), lo, hi)
        small = torch.where(lo.abs() >= hi.abs(), hi, lo)
        linearity = 1.0 - small.abs() / (big.abs() + EPS)
        strength = big.abs() * linearity * (sigma ** 2)
        keep.append(strength[:, :1])
        tag = f's{sigma:g}'
        out.append(torch.log(strength.mean((-2, -1)) + 1e-8))
        names.extend(f'ridge{tag}mean_{m}' for m in MAP_NAMES)
        out.append(strength.std((-2, -1)) / (strength.mean((-2, -1)) + EPS))
        names.extend(f'ridge{tag}cv_{m}' for m in MAP_NAMES)
        q90 = _quantile(strength, 0.9)
        out.append(strength.amax((-2, -1)) / (q90[:, :, 0, 0] + EPS))
        names.extend(f'ridge{tag}pk_{m}' for m in MAP_NAMES)
        top = (strength >= q90).float()
        out.append((linearity * top).sum((-2, -1)) / (top.sum((-2, -1)) + EPS))
        names.extend(f'ridge{tag}lin_{m}' for m in MAP_NAMES)
        out.append((big * top).sum((-2, -1)) / ((big.abs() * top).sum((-2, -1)) + EPS))
        names.extend(f'ridge{tag}sgn_{m}' for m in MAP_NAMES)
        angle = 0.5 * torch.atan2(2 * ixy, ixx - iyy)
        weight = strength * top
        total = weight.sum((-2, -1)) + EPS
        cos = (weight * torch.cos(2 * angle)).sum((-2, -1)) / total
        sin = (weight * torch.sin(2 * angle)).sum((-2, -1)) / total
        out.append(torch.sqrt(cos ** 2 + sin ** 2))
        names.extend(f'ridge{tag}coh_{m}' for m in MAP_NAMES)
        out.append((strength > strength.mean((-2, -1), keepdim=True)
                    + 2 * strength.std((-2, -1), keepdim=True)).float().mean((-2, -1)))
        names.extend(f'ridge{tag}frac_{m}' for m in MAP_NAMES)
    return out, torch.cat(keep, dim=1)


def _runlength(masks: torch.Tensor, tags: tuple[str, ...],
               names: list[str]) -> list[torch.Tensor]:
    """Directional run lengths of binary masks: how far a thin structure runs."""
    out: list[torch.Tensor] = []
    longest, emphasis = [], []
    for dy, dx in RUN_DIRS:
        run = masks.clone()
        for _ in range(RUN_MAX - 1):
            run = masks * (1.0 + _shift(run, dy, dx))
        longest.append(run.amax((-2, -1)))
        emphasis.append((run ** 2).sum((-2, -1)) / (run.sum((-2, -1)) + EPS))
        out.append(run.sum((-2, -1)) / (masks.sum((-2, -1)) + EPS))
        names.extend(f'runmean{dy}{dx}_{t}' for t in tags)
    stack_long = torch.stack(longest, dim=-1)
    stack_emph = torch.stack(emphasis, dim=-1)
    srt_long = torch.sort(stack_long, dim=-1, descending=True).values
    srt_emph = torch.sort(stack_emph, dim=-1, descending=True).values
    for i in range(len(RUN_DIRS)):
        out.append(srt_long[..., i])
        names.extend(f'runlong{i}_{t}' for t in tags)
        out.append(srt_emph[..., i])
        names.extend(f'runemph{i}_{t}' for t in tags)
    out.append(srt_long[..., 0] / (stack_long.mean(-1) + EPS))
    names.extend(f'runaniso_{t}' for t in tags)
    out.append(srt_long[..., 1] / (srt_long[..., 0] + EPS))
    names.extend(f'runsecond_{t}' for t in tags)
    return out


def _blobs(maps: torch.Tensor, kernels: dict[float, torch.Tensor],
           yy: torch.Tensor, xx: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Difference-of-Gaussian local-maximum counts and maximum-cloud moments.

    Repeated discrete objects -- docked hulls, storage tanks, parked aircraft --
    show up as a count of local maxima at a fixed scale, and their arrangement
    (a line of ships along a quay against a scattered field) shows up in the
    second moments of the maximum locations.
    """
    out: list[torch.Tensor] = []
    n, c, h, w = maps.shape
    for sigma in BLOB_SCALES:
        dog = _blur(maps, kernels[sigma]) - _blur(maps, kernels[sigma * 1.6])
        response = dog.abs() * (sigma ** 2)
        peak = F.max_pool2d(response, 5, stride=1, padding=2)
        threshold = response.mean((-2, -1), keepdim=True) + 2 * response.std((-2, -1), keepdim=True)
        maxima = ((response >= peak) & (response > threshold)).float()
        tag = f'b{sigma:g}'
        count = maxima.sum((-2, -1))
        out.append(torch.log1p(count))
        names.extend(f'blob{tag}n_{m}' for m in MAP_NAMES)
        out.append((response * maxima).sum((-2, -1)) / (count + EPS))
        names.extend(f'blob{tag}resp_{m}' for m in MAP_NAMES)
        offset, big, small, ratio = _moments2(maxima, yy, xx)
        out.append(big)
        names.extend(f'blob{tag}spread_{m}' for m in MAP_NAMES)
        out.append(ratio)
        names.extend(f'blob{tag}elong_{m}' for m in MAP_NAMES)
        out.append(offset)
        names.extend(f'blob{tag}off_{m}' for m in MAP_NAMES)
        cells = F.avg_pool2d(maxima, h // 4).reshape(n, c, -1)
        out.append(cells.std(-1) / (cells.mean(-1) + EPS))
        names.extend(f'blob{tag}disp_{m}' for m in MAP_NAMES)
        out.append(cells.max(-1).values / (cells.sum(-1) + EPS))
        names.extend(f'blob{tag}top_{m}' for m in MAP_NAMES)
    return out


def _polar(fields: torch.Tensor, grid: torch.Tensor, names: list[str]) -> list[torch.Tensor]:
    """Rotational/mirror self-similarity and log-polar radial and angular profiles."""
    n, c = fields.shape[0], fields.shape[1]
    centred = fields - fields.mean((-2, -1), keepdim=True)
    denom = (centred ** 2).sum((-2, -1)) + EPS
    out: list[torch.Tensor] = []
    for tag, other in (('r90', torch.rot90(centred, 1, (-2, -1))),
                       ('r180', torch.rot90(centred, 2, (-2, -1))),
                       ('fx', torch.flip(centred, (-1,))),
                       ('fy', torch.flip(centred, (-2,)))):
        out.append((centred * other).sum((-2, -1)) / denom)
        names.extend(f'sym{tag}_{f}' for f in FIELD_NAMES)
    sampled = F.grid_sample(fields, grid.expand(n, -1, -1, -1), mode='bilinear',
                            padding_mode='border', align_corners=False)
    radial = sampled.mean(-1)                                    # (N, C, R)
    angular = sampled.mean(-2)                                   # (N, C, A)
    scale = radial.mean(-1, keepdim=True).abs() + EPS
    for i in range(8):
        band = radial[..., i * (POLAR_RADII // 8):(i + 1) * (POLAR_RADII // 8)].mean(-1)
        out.append(band / scale[..., 0])
        names.extend(f'polR{i}_{f}' for f in FIELD_NAMES)
    spectrum = torch.fft.rfft(angular - angular.mean(-1, keepdim=True), dim=-1).abs() ** 2
    share = spectrum[..., 1:] / (spectrum[..., 1:].sum(-1, keepdim=True) + EPS)
    for k in range(6):  # k-fold angular symmetry: 2 = strip, 4 = crossroads
        out.append(share[..., k])
        names.extend(f'polA{k + 1}_{f}' for f in FIELD_NAMES)
    out.append(-(share * torch.log(share + EPS)).sum(-1))
    names.extend(f'polAent_{f}' for f in FIELD_NAMES)
    out.append(angular.std(-1) / (radial.std(-1) + EPS))
    names.extend(f'polAR_{f}' for f in FIELD_NAMES)
    ring = sampled.std(-1)                                       # angular variance per radius
    out.append(ring.max(-1).values / (ring.mean(-1) + EPS))
    names.extend(f'polRing_{f}' for f in FIELD_NAMES)
    out.append(ring.argmax(-1).float() / POLAR_RADII)
    names.extend(f'polRingR_{f}' for f in FIELD_NAMES)
    return out


def _mask_layout(maps: torch.Tensor, yy: torch.Tensor, xx: torch.Tensor,
                 names: list[str]) -> list[torch.Tensor]:
    """Second moments, border spanning, and profile breaks of thresholded regions."""
    out: list[torch.Tensor] = []
    n, c, h, w = maps.shape
    for q in (0.1, 0.9):
        binary = (maps >= _quantile(maps, q)).float() if q > 0.5 else \
            (maps <= _quantile(maps, q)).float()
        tag = f'q{q:g}'
        offset, big, small, ratio = _moments2(binary, yy, xx)
        out.append(big)
        names.extend(f'mask{tag}big_{m}' for m in MAP_NAMES)
        out.append(ratio)
        names.extend(f'mask{tag}elong_{m}' for m in MAP_NAMES)
        out.append(offset)
        names.extend(f'mask{tag}off_{m}' for m in MAP_NAMES)
        sides = torch.stack((binary[:, :, 0].mean(-1), binary[:, :, -1].mean(-1),
                             binary[:, :, :, 0].mean(-1), binary[:, :, :, -1].mean(-1)), dim=-1)
        out.append(sides.mean(-1))
        names.extend(f'mask{tag}border_{m}' for m in MAP_NAMES)
        out.append((sides > 0.05).float().sum(-1))
        names.extend(f'mask{tag}sides_{m}' for m in MAP_NAMES)
        for axis, atag in ((-1, 'row'), (-2, 'col')):
            profile = binary.mean(axis)
            level = profile.mean(-1, keepdim=True)
            crossings = ((profile[..., 1:] >= level) != (profile[..., :-1] >= level)).float()
            out.append(crossings.sum(-1))
            names.extend(f'mask{tag}{atag}br_{m}' for m in MAP_NAMES)
            out.append(profile.max(-1).values - profile.min(-1).values)
            names.extend(f'mask{tag}{atag}rng_{m}' for m in MAP_NAMES)
    return out


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 64) -> tuple[np.ndarray, list[str]]:
    """Extract the object-layout pool for uint8 ``(N, 3, H, W)`` images."""
    grids = _rotation_grids(RES, device)
    polar = _polar_grid(device)
    sigmas = sorted(set(RIDGE_SCALES) | set(BLOB_SCALES) | {s * 1.6 for s in BLOB_SCALES})
    kernels = {s: _gauss_kernel(s, device) for s in sigmas}
    lin = torch.linspace(-1.0, 1.0, RES, device=device)
    yy, xx = torch.meshgrid(lin, lin, indexing='ij')
    disk = ((yy ** 2 + xx ** 2) <= 1.0).float()
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                            device=device).float() / 255.0
        maps = F.avg_pool2d(_maps(x), 2)
        edge = _grad_magnitude(maps[:, :1])
        fields = torch.cat((maps, edge), dim=1)
        local: list[str] = []
        ridge_parts, ridge_maps = _ridge(maps, kernels, local)
        run_masks = torch.cat((
            (ridge_maps >= _quantile(ridge_maps, 0.95)).float(),
            (edge >= _quantile(edge, 0.9)).float(),
            (maps[:, :1] <= _quantile(maps[:, :1], 0.1)).float(),
        ), dim=1)
        run_tags = tuple(f'ridge{s:g}' for s in RIDGE_SCALES) + ('edge', 'dark')
        parts = (
            _radon(fields, grids, disk, local)
            + ridge_parts
            + _runlength(run_masks, run_tags, local)
            + _blobs(maps, kernels, yy, xx, local)
            + _polar(fields, polar, local)
            + _mask_layout(maps, yy, xx, local)
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
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu3_pool.npy'), features)
        print(split, features.shape, flush=True)
    with open(os.path.join(CACHE_DIR, 'resisc45_gpu3_pool_names.txt'), 'w') as handle:
        handle.write('\n'.join(names) + '\n')
    print('names', len(names))
