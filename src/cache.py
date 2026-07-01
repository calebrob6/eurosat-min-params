"""Cache raw EuroSAT patches to .npy for fast repeated experiments."""

from __future__ import annotations

import os

import numpy as np

from .data import list_split, read_tif

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')
CACHE_DIR = os.path.abspath(CACHE_DIR)


def _paths(split: str) -> tuple[str, str]:
    return (
        os.path.join(CACHE_DIR, f'{split}_x_uint16.npy'),
        os.path.join(CACHE_DIR, f'{split}_y.npy'),
    )


def build_cache(split: str, max_workers: int = 16) -> None:
    """Read every patch in a split and store as a single uint16 npy array."""
    from concurrent.futures import ProcessPoolExecutor

    os.makedirs(CACHE_DIR, exist_ok=True)
    paths, labels = list_split(split)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        imgs = list(ex.map(read_tif, paths, chunksize=32))
    x = np.stack(imgs).astype(np.uint16)
    xp, yp = _paths(split)
    np.save(xp, x)
    np.save(yp, labels)
    print(f'cached {split}: x={x.shape} {x.dtype} -> {xp}')


def load_cached(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load cached patches as (N,13,64,64) float32 and integer labels."""
    xp, yp = _paths(split)
    if not os.path.exists(xp):
        build_cache(split)
    x = np.load(xp).astype(np.float32)
    y = np.load(yp)
    return x, y


if __name__ == '__main__':
    for s in ('train', 'val', 'test'):
        if not os.path.exists(_paths(s)[0]):
            build_cache(s)
        else:
            print(f'{s} already cached')
