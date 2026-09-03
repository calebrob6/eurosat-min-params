#!/usr/bin/env python
"""Sixth zero-parameter RESISC45 pool: multi-layer random convolutional features.

Five pools of global statistics leave the same confusions standing (church /
palace, basketball / tennis court, railway / railway station), and iteration 4
concluded that what those pairs need is *part configuration*: not which local
patterns occur but how they are arranged relative to one another.  The fourth
pool's single-layer random convolutions (MOSAIKS) summarise local pattern
occurrence; stacking two more seeded random layers on top of them, with ReLU
and max pooling between, gives responses to *compositions* of local patterns
over a receptive field of about a fifth of the image.  Random deep features are
known to carry much of the value of trained ones (Saxe et al. 2011) and cost
no learned parameters: every filter is drawn from a fixed-seed generator and
scaled by its fan-in, so the extractor is as deterministic as any hand-designed
map in the earlier pools.

* ``rdeep`` -- for each of two input scales (average-pooled by 2 and by 4 from
  the 256 x 256 image), a three-layer random ReLU network (5 x 5 x 3 -> 32,
  3 x 3 -> 64, 3 x 3 -> 128 filters, max pooling by 2 after the first two
  layers); the pool records the global mean and max of every channel of the
  second and third layers.  Each scale uses its own seed so the two networks
  are different random draws rather than the same one at two resolutions.

The output is 2 scales x (64 + 128) channels x (mean, max) = 768 columns.
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

SEED = 600
SCALES = (2, 4)
LAYERS = ((3, 32, 5), (32, 64, 3), (64, 128, 3))  # (in, out, kernel)
POOL_AFTER = (True, True, False)
RECORD = (False, True, True)


def _network(seed: int, device: str) -> list[torch.Tensor]:
    """Seeded He-scaled Gaussian filters for the three layers."""
    rng = np.random.default_rng(seed)
    filters = []
    for cin, cout, k in LAYERS:
        w = rng.standard_normal((cout, cin, k, k)) * np.sqrt(2.0 / (cin * k * k))
        filters.append(torch.as_tensor(w, dtype=torch.float32, device=device))
    return filters


def _rdeep(x: torch.Tensor, nets: dict[int, list[torch.Tensor]], names: list[str]) -> list[torch.Tensor]:
    out: list[torch.Tensor] = []
    xc = x - 0.5
    for factor in SCALES:
        h = F.avg_pool2d(xc, factor)
        for depth, (w, pool, record) in enumerate(zip(nets[factor], POOL_AFTER, RECORD)):
            h = torch.relu(F.conv2d(h, w))
            if record:
                flat = h.reshape(h.shape[0], h.shape[1], -1)
                out.append(flat.mean(-1))
                names.extend(f'rdeep{factor}l{depth}mean_{i}' for i in range(h.shape[1]))
                out.append(flat.amax(-1))
                names.extend(f'rdeep{factor}l{depth}max_{i}' for i in range(h.shape[1]))
            if pool:
                h = F.max_pool2d(h, 2)
    return out


def extract(images: np.ndarray, device: str = 'cuda', batch: int = 128) -> tuple[np.ndarray, list[str]]:
    """Extract the sixth pool for uint8 ``(N, 3, H, W)`` images."""
    nets = {factor: _network(SEED + factor, device) for factor in SCALES}
    chunks: list[np.ndarray] = []
    names: list[str] = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                                device=device).float() / 255.0
            local: list[str] = []
            parts = _rdeep(x, nets, local)
            block = torch.cat([p.reshape(p.shape[0], -1) for p in parts], dim=1)
            chunks.append(block.float().cpu().numpy())
            if not names:
                names = local
    out = np.concatenate(chunks, axis=0)
    if out.shape[1] != len(names):
        raise ValueError(f'{out.shape[1]} columns but {len(names)} names')
    return out, names


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
            print('constant columns:', len(const), const[:10])
            print('abs max:', float(np.abs(features).max()), 'mean abs:', float(np.abs(features).mean()))
            break
        np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_gpu6_pool.npy'), features)
    if not limit:
        with open(os.path.join(CACHE_DIR, 'resisc45_gpu6_pool_names.txt'), 'w') as handle:
            handle.write('\n'.join(names) + '\n')
        print('names', len(names))
