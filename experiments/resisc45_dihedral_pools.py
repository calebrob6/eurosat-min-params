#!/usr/bin/env python
"""Dihedral-averaged RESISC45 pools: every GPU pool re-extracted under the eight
rotations and flips of the image and averaged.

Aerial scenes have no canonical orientation, but many pool columns are not
rotation-invariant (gradient orientation histograms, line orientation, sorted
grid cells, layout, random convolutions).  Averaging a column over the eight
dihedral transforms of the image makes it invariant without any learned
parameter and removes the orientation-dependent part of its variance, the same
test-time-augmentation idea that lifts trained networks.  The averaged pool is
cached under the same names with a ``d8`` suffix on the pool stem
(``gpu_pool`` -> ``gpud8_pool``) so ``load_pools`` reads it like any other.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from src.cache import CACHE_DIR

MODULES = {
    'gpu_pool': 'resisc45_gpu_features',
    'gpu2_pool': 'resisc45_gpu_features2',
    'gpu3_pool': 'resisc45_gpu_features3',
    'gpu4_pool': 'resisc45_gpu_features4',
    'gpu5_pool': 'resisc45_gpu_features5',
    'gpu6_pool': 'resisc45_gpu_features6',
    'gpu7_pool': 'resisc45_gpu_features7',
}


def transforms(images: np.ndarray):
    """The eight dihedral views of an ``(N, 3, H, W)`` array, as views."""
    for flip in (False, True):
        base = images[:, :, :, ::-1] if flip else images
        for k in range(4):
            yield f'rot{k}{"f" if flip else ""}', np.rot90(base, k, axes=(2, 3))


def averaged_name(pool: str) -> str:
    stem, _, suffix = pool.partition('_')
    return f'{stem}d8_{suffix}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pools', nargs='*', default=list(MODULES))
    parser.add_argument('--splits', nargs='*', default=['train', 'val', 'test'])
    args = parser.parse_args()
    for pool in args.pools:
        module = importlib.import_module(MODULES[pool])
        out_name = averaged_name(pool)
        names = None
        for split in args.splits:
            images = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_256.npy'),
                             mmap_mode='r')
            total = None
            t0 = time.time()
            for tag, view in transforms(images):
                with torch.no_grad():
                    features, names = module.extract(view)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                if tag == 'rot0':
                    cached = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_{pool}.npy'))
                    diff = float(np.abs(cached - features).max())
                    print(f'{pool} {split} identity check: max |diff| = {diff:.2e}', flush=True)
                total = features.astype(np.float64) if total is None else total + features
                print(f'  {pool} {split} {tag} {time.time() - t0:.0f}s', flush=True)
            mean = (total / 8).astype(np.float32)
            np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_{out_name}.npy'), mean)
            print(pool, split, mean.shape, f'{time.time() - t0:.0f}s', flush=True)
        with open(os.path.join(CACHE_DIR, f'resisc45_{out_name}_names.txt'), 'w') as handle:
            handle.write('\n'.join(names) + '\n')


if __name__ == '__main__':
    main()
