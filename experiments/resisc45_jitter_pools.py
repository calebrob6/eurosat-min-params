#!/usr/bin/env python
"""Translation- and scale-averaged RESISC45 pools, at the dihedral pools' cost.

Iteration 5 showed that averaging every GPU pool over the eight dihedral views
of the image (``resisc45_dihedral_pools.py``) removes an orientation nuisance
worth +0.8 test points at the dense ceiling and +0.5 under the union head at
no deployed cost.  Orientation is not the only nuisance an aerial scene
carries: the field of view is arbitrary (the scene continues beyond the
frame) and the ground sample distance varies by a factor of about a hundred
across the dataset.  This script averages each pool over eight views that
combine the eight dihedral elements with a *second* nuisance, so every family
costs the same eight extractions as the dihedral pools:

* ``c224`` -- each dihedral view is paired with a distinct 224 x 224 crop of
  the 256 x 256 image (four corners and four mid-edges), so the average is over
  eight (crop, rotation) pairs: translation and field-of-view jitter;
* ``s192`` / ``s320`` -- the eight dihedral views of the whole image resized to
  192 x 192 (area) or 320 x 320 (bicubic): the same pool read at 0.75x / 1.25x
  the pixel scale, so their average with the ``d8`` pool is a scale average.

Every column keeps its definition (the extractors take any ``(N, 3, H, W)``
input; pooling factors and cell grids are relative to the image, so a column
at 192 or 320 is the same statistic at a different pixel scale).  The layout
pool works at a fixed 128-pixel resolution, so its views are resized back to
256 x 256 first (crop-and-zoom, or a blurred / sharpened image).  The pools
are cached as ``resisc45_{split}_gpu{,2,3,4,5}{family}_pool.npy`` with the
original names, so ``load_pools`` reads them like any other, and
``resisc45_jitter_ceiling.py`` reads what each family and their averages are
worth on the frontier list.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from src.cache import CACHE_DIR
from resisc45_dihedral_pools import MODULES, transforms

CROP = 224
CROP_OFFSETS = ((0, 0), (0, 32), (32, 0), (32, 32), (0, 16), (16, 0), (32, 16), (16, 32))
SCALES = {'s192': 192, 's320': 320}


def family_name(pool: str, family: str) -> str:
    stem, _, suffix = pool.partition('_')
    return f'{stem}{family}_{suffix}'


def resized(images: np.ndarray, size: int, device: str = 'cuda', batch: int = 512) -> np.ndarray:
    """The uint8 stack resized to ``size`` (area down, bicubic up), materialised."""
    out = np.empty((images.shape[0], 3, size, size), dtype=np.uint8)
    mode = 'area' if size < images.shape[-1] else 'bicubic'
    for start in range(0, len(images), batch):
        x = torch.as_tensor(np.ascontiguousarray(images[start:start + batch]),
                            device=device).float()
        kwargs = {} if mode == 'area' else {'align_corners': False}
        y = F.interpolate(x, size=(size, size), mode=mode, **kwargs)
        out[start:start + batch] = y.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    return out


def views(images: np.ndarray, family: str, native: bool = False,
          crop_shift: int = 0):
    """Eight ``(tag, view)`` pairs for the family, dihedral views over crops or scales.

    With ``native`` every view is resized back to the cached 256 x 256 before it
    is yielded (the layout pool works at a fixed 128-pixel resolution): a crop
    becomes a crop-and-zoom, a rescaled image a blurred or sharpened one.
    """
    size = images.shape[-1]
    if family == 'c224':
        shift = crop_shift % len(CROP_OFFSETS)
        offsets = CROP_OFFSETS[shift:] + CROP_OFFSETS[:shift]
        for (tag, view), (dy, dx) in zip(transforms(images), offsets):
            crop = view[:, :, dy:dy + CROP, dx:dx + CROP]
            yield f'{tag}-y{dy}x{dx}', resized(crop, size) if native else crop
    elif family in SCALES:
        scaled = resized(images, SCALES[family])
        yield from transforms(resized(scaled, size) if native else scaled)
    else:
        raise ValueError(family)


def average_families(pools, families, name, splits) -> None:
    """Cache the mean of already-extracted families as pool suffix ``name``."""
    for pool in pools:
        for split in splits:
            parts = [np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_{family_name(pool, f)}.npy'))
                     for f in families]
            mean = np.mean(parts, axis=0).astype(np.float32)
            np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_{family_name(pool, name)}.npy'), mean)
            print(pool, split, name, mean.shape, flush=True)
        names_path = os.path.join(CACHE_DIR, f'resisc45_{family_name(pool, families[0])}_names.txt')
        with open(names_path) as src, \
                open(os.path.join(CACHE_DIR, f'resisc45_{family_name(pool, name)}_names.txt'), 'w') as dst:
            dst.write(src.read())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', choices=('c224',) + tuple(SCALES))
    parser.add_argument('--crop-shift', type=int, default=0,
                        help='cyclic shift of crop offsets relative to dihedral views; '
                             'use with --name to cache a complementary c224 pairing')
    parser.add_argument('--average', nargs='*', default=None,
                        help='cache the mean of these cached families (e.g. d8 c224 s192 s320) '
                             'under the suffix given by --name instead of extracting')
    parser.add_argument('--name', default=None,
                        help='output pool suffix for extraction or --average (e.g. c224s1, j24)')
    parser.add_argument(
        '--pools',
        nargs='*',
        default=[
            pool
            for pool in MODULES
            if pool not in ('gpu6_pool', 'gpu7_pool')
        ],
    )
    parser.add_argument('--splits', nargs='*', default=['train', 'val', 'test'])
    parser.add_argument('--smoke', type=int, default=0,
                        help='extract only this many images per split and do not cache')
    args = parser.parse_args()
    if args.average:
        if not args.name:
            parser.error('--name is required with --average')
        average_families(args.pools, args.average, args.name, args.splits)
        return
    if args.crop_shift and args.family != 'c224':
        parser.error('--crop-shift is only valid with --family c224')
    if args.crop_shift and not args.name:
        parser.error('--name is required when --crop-shift is nonzero')
    for pool in args.pools:
        module = importlib.import_module(MODULES[pool])
        out_name = family_name(pool, args.name or args.family)
        names = None
        for split in args.splits:
            images = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_x_uint8_256.npy'),
                             mmap_mode='r')
            if args.smoke:
                images = np.array(images[:args.smoke])
            total = None
            t0 = time.time()
            for tag, view in views(
                images,
                args.family,
                native=pool == 'gpu3_pool',
                crop_shift=args.crop_shift,
            ):
                with torch.no_grad():
                    features, names = module.extract(view)
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                total = features.astype(np.float64) if total is None else total + features
                print(f'  {pool} {split} {tag} {tuple(view.shape[-2:])} {time.time() - t0:.0f}s',
                      flush=True)
            mean = (total / 8).astype(np.float32)
            cached = np.load(os.path.join(CACHE_DIR, f'resisc45_{split}_{pool}.npy'),
                             mmap_mode='r')
            if mean.shape[1] != cached.shape[1]:
                raise ValueError(f'{pool} {split}: {mean.shape[1]} columns, cached {cached.shape[1]}')
            corr = np.corrcoef(np.nan_to_num(np.array(cached[:len(mean)]), nan=0.0).ravel(),
                               mean.ravel())[0, 1]
            print(pool, split, mean.shape, f'corr with cached {corr:.3f}', f'{time.time() - t0:.0f}s',
                  flush=True)
            if args.smoke:
                continue
            np.save(os.path.join(CACHE_DIR, f'resisc45_{split}_{out_name}.npy'), mean)
        if not args.smoke:
            with open(os.path.join(CACHE_DIR, f'resisc45_{out_name}_names.txt'), 'w') as handle:
                handle.write('\n'.join(names) + '\n')


if __name__ == '__main__':
    main()
