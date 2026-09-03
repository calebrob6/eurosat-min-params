#!/usr/bin/env python
"""Seventh zero-parameter RESISC45 pool: cross-part configuration.

Earlier pools describe each colour region, edge field, or connected component
independently. Persistent church/palace and basketball/tennis errors instead
depend on how parts are arranged. This pool measures pairwise containment,
adjacency, centroid spacing, radial order, principal-axis agreement, and coarse
layout correlation among eleven fixed colour masks plus edge, bright-marking,
and dark-line masks.
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

from resisc45_gpu_features4 import (  # noqa: E402
    _colour_masks,
    _grad_magnitude,
    _maps,
)
from src.cache import CACHE_DIR  # noqa: E402

EPS = 1e-6
MASK_RESOLUTION = 32
CELL_GRID = 8
ADJACENCY_KERNELS = (3, 7, 15)


def _part_masks(x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
    """Return soft fixed masks at a common 32x32 working resolution."""
    colour, names = _colour_masks(x)
    maps = _maps(x)
    pan = maps[:, :1]
    saturation = maps[:, 2:3]
    gradient = _grad_magnitude(pan)
    threshold = torch.quantile(
        gradient.flatten(2).float(), 0.82, dim=2
    )[:, :, None, None]
    edge = (gradient > threshold).float()

    eroded = -F.max_pool2d(-pan, 5, stride=1, padding=2)
    opened = F.max_pool2d(eroded, 5, stride=1, padding=2)
    white_tophat = pan - opened
    bright_mark = (
        (white_tophat > 0.08) & (pan > 0.5) & (saturation < 0.35)
    ).float()

    dilated = F.max_pool2d(pan, 5, stride=1, padding=2)
    closed = -F.max_pool2d(-dilated, 5, stride=1, padding=2)
    black_tophat = closed - pan
    dark_line = ((black_tophat > 0.08) & (pan < 0.55)).float()

    masks = torch.cat((colour, edge, bright_mark, dark_line), dim=1)
    masks = F.adaptive_avg_pool2d(masks, MASK_RESOLUTION)
    return masks, names + ['edge', 'bright_mark', 'dark_line']


def _configuration_features(
    masks: torch.Tensor, mask_names: list[str]
) -> tuple[torch.Tensor, list[str]]:
    """Summarize individual masks and every unordered mask pair."""
    batch, count, height, width = masks.shape
    flat = masks.flatten(2)
    mass = flat.sum(2) + EPS
    fraction = mass / (height * width)

    coordinate_y = torch.linspace(-1, 1, height, device=masks.device)
    coordinate_x = torch.linspace(-1, 1, width, device=masks.device)
    yy, xx = torch.meshgrid(coordinate_y, coordinate_x, indexing='ij')
    x_flat = xx.flatten()[None, None]
    y_flat = yy.flatten()[None, None]
    mean_x = (flat * x_flat).sum(2) / mass
    mean_y = (flat * y_flat).sum(2) / mass
    delta_x = x_flat - mean_x[:, :, None]
    delta_y = y_flat - mean_y[:, :, None]
    covariance_xx = (flat * delta_x.square()).sum(2) / mass
    covariance_yy = (flat * delta_y.square()).sum(2) / mass
    covariance_xy = (flat * delta_x * delta_y).sum(2) / mass
    trace = covariance_xx + covariance_yy
    anisotropy_numerator = torch.sqrt(
        (covariance_xx - covariance_yy).square()
        + 4 * covariance_xy.square()
    ).clamp_min(0)
    anisotropy = anisotropy_numerator / (trace + EPS)
    axis_cos = (covariance_xx - covariance_yy) / (
        anisotropy_numerator + EPS
    )
    axis_sin = 2 * covariance_xy / (anisotropy_numerator + EPS)
    radius = torch.sqrt(mean_x.square() + mean_y.square())
    spread = torch.sqrt(trace.clamp_min(0))

    outputs: list[torch.Tensor] = [fraction, radius, spread, anisotropy]
    names = [
        f'part_{stat}_{name}'
        for stat in ('fraction', 'radius', 'spread', 'anisotropy')
        for name in mask_names
    ]

    left, right = torch.triu_indices(count, count, offset=1, device=masks.device)
    overlap = torch.bmm(flat, flat.transpose(1, 2))
    overlap_pair = overlap[:, left, right]
    left_mass = mass[:, left]
    right_mass = mass[:, right]
    union = left_mass + right_mass - overlap_pair + EPS
    containment_left = overlap_pair / left_mass
    containment_right = overlap_pair / right_mass
    iou = overlap_pair / union

    centroid_distance = torch.sqrt(
        (mean_x[:, left] - mean_x[:, right]).square()
        + (mean_y[:, left] - mean_y[:, right]).square()
    )
    radial_gap = (radius[:, left] - radius[:, right]).abs()
    axis_agreement = (
        axis_cos[:, left] * axis_cos[:, right]
        + axis_sin[:, left] * axis_sin[:, right]
    )

    cells = F.adaptive_avg_pool2d(masks, CELL_GRID).flatten(2)
    cells = cells - cells.mean(2, keepdim=True)
    cell_norm = torch.sqrt(cells.square().sum(2) + EPS)
    cell_correlation = torch.bmm(cells, cells.transpose(1, 2))
    cell_correlation = (
        cell_correlation
        / (cell_norm[:, :, None] * cell_norm[:, None, :] + EPS)
    )[:, left, right]

    pair_outputs = [
        iou,
        containment_left,
        containment_right,
        centroid_distance,
        radial_gap,
        axis_agreement,
        cell_correlation,
    ]
    pair_stats = [
        'iou',
        'contain_left',
        'contain_right',
        'centroid_distance',
        'radial_gap',
        'axis_agreement',
        'cell_correlation',
    ]

    for kernel in ADJACENCY_KERNELS:
        expanded = F.max_pool2d(
            masks, kernel, stride=1, padding=kernel // 2
        ).flatten(2)
        contact = torch.bmm(flat, expanded.transpose(1, 2))
        contact = contact + contact.transpose(1, 2)
        contact_pair = contact[:, left, right] / (
            left_mass + right_mass + EPS
        )
        pair_outputs.append(contact_pair)
        pair_stats.append(f'adjacency{kernel}')

    outputs.extend(pair_outputs)
    pair_names = [
        (mask_names[int(i)], mask_names[int(j)])
        for i, j in zip(left.cpu(), right.cpu(), strict=True)
    ]
    names.extend(
        f'partpair_{stat}_{left_name}__{right_name}'
        for stat in pair_stats
        for left_name, right_name in pair_names
    )
    features = torch.cat(outputs, dim=1).reshape(batch, -1)
    return features, names


def extract(
    images: np.ndarray, device: str = 'cuda', batch: int = 256
) -> tuple[np.ndarray, list[str]]:
    """Extract cross-part configuration features from uint8 RGB images."""
    chunks: list[np.ndarray] = []
    names: list[str] = []
    for start in range(0, len(images), batch):
        x = torch.as_tensor(
            np.ascontiguousarray(images[start:start + batch]),
            device=device,
            dtype=torch.float32,
        ) / 255.0
        masks, mask_names = _part_masks(x)
        block, local_names = _configuration_features(masks, mask_names)
        chunks.append(block.float().cpu().numpy())
        if not names:
            names = local_names
        elif names != local_names:
            raise ValueError('feature schema changed between batches')
    features = np.concatenate(chunks, axis=0)
    if features.shape[1] != len(names):
        raise ValueError('feature names do not match extracted columns')
    return features, names


def main() -> None:
    for split in ('train', 'val', 'test'):
        images = np.load(
            os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_256.npy'),
            mmap_mode='r',
        )
        with torch.no_grad():
            features, names = extract(images)
        features = np.nan_to_num(
            features, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        np.save(
            os.path.join(CACHE_DIR, f'resisc45_{split}_gpu7_pool.npy'),
            features,
        )
        print(split, features.shape, flush=True)
    with open(
        os.path.join(CACHE_DIR, 'resisc45_gpu7_pool_names.txt'), 'w'
    ) as destination:
        destination.write('\n'.join(names) + '\n')


if __name__ == '__main__':
    main()
