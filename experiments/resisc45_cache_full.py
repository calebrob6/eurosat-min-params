#!/usr/bin/env python
"""Cache the RESISC45 splits at native 256x256 resolution.

The existing pool was extracted from a bilinear 64x64 resize, which discards the
fine structure that separates RESISC45's man-made classes.  This builds the
full-resolution uint8 cache so later feature families can choose their own
scale.  Split membership and labels are validated against the official TorchGeo
listings by reusing ``resisc45_33_feature.list_split``.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_33_feature import list_split  # noqa: E402
from src.cache import CACHE_DIR  # noqa: E402

SIZE = 256


def read_rgb(path: str) -> np.ndarray:
    """Read one RGB JPEG at native 256x256 in channel-first order."""
    with Image.open(path) as handle:
        image = handle.convert('RGB')
        if image.size != (SIZE, SIZE):
            image = image.resize((SIZE, SIZE), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)


def build(split: str, workers: int = 32) -> None:
    paths, labels = list_split(split)
    out = os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_{SIZE}.npy')
    if os.path.exists(out):
        print(f'{out} exists')
        return
    images = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.uint8, shape=(len(paths), 3, SIZE, SIZE)
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for index, image in enumerate(executor.map(read_rgb, paths, chunksize=64)):
            images[index] = image
    images.flush()
    del images
    stored = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_y.npy'))
    if not np.array_equal(stored, labels):
        raise ValueError(f'{split} labels disagree with the cached split order')
    print(f'wrote {out} ({len(paths)} images)')


if __name__ == '__main__':
    for split in ('train', 'val', 'test'):
        build(split)
