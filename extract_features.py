#!/usr/bin/env python
"""Extract PyTorch features from 13-band, 64x64 TIFF patches."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import torch

from eurosat_features import TIFF_BAND_NAMES, EuroSATFeatures


def input_paths(inputs: list[Path]) -> list[Path]:
    """Collect TIFF files from individual paths and directories.

    Args:
        inputs: TIFF paths or directories to search recursively.

    Returns:
        File paths in input order, with each directory's files sorted.

    Raises:
        ValueError: If a path is invalid, a directory has no TIFFs, or files repeat.
    """
    paths = []
    for source in inputs:
        if source.is_dir():
            candidates = sorted(
                path
                for path in source.rglob('*')
                if path.is_file() and path.suffix.lower() in ('.tif', '.tiff')
            )
            if not candidates:
                raise ValueError(f'no TIFF patches found in {source}')
            paths.extend(candidates)
        elif source.is_file() and source.suffix.lower() in ('.tif', '.tiff'):
            paths.append(source)
        else:
            raise ValueError(f'expected a TIFF file or directory: {source}')
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError('the inputs contain duplicate files')
    return paths


def main() -> None:
    """Extract the requested features and save them with filenames and column names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'inputs',
        type=Path,
        nargs='+',
        help='TIFF files or directories searched recursively',
    )
    parser.add_argument('--features', choices=('33', '377', '52'), default='33')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.output.suffix != '.npz':
        parser.error('--output must end with .npz')
    if args.output.exists():
        parser.error(f'output already exists: {args.output}')
    paths = input_paths(args.inputs)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    extractor = EuroSATFeatures(feature_set=args.features).to(device)
    parts = []
    for start in range(0, len(paths), args.batch_size):
        images = []
        for path in paths[start : start + args.batch_size]:
            with rasterio.open(path) as handle:
                image = handle.read(out_dtype='float32')
            if image.shape != (13, 64, 64):
                raise ValueError(f'{path}: expected (13, 64, 64), got {image.shape}')
            images.append(image)
        batch = torch.from_numpy(np.stack(images)).to(device)
        parts.append(extractor(batch).cpu().numpy())
        print(f'{start + len(images)}/{len(paths)} patches', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=args.output.parent, suffix='.npz', delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with temporary.open('wb') as handle:
            np.savez_compressed(
                handle,
                features=np.concatenate(parts),
                filenames=np.asarray([str(path) for path in paths]),
                feature_names=np.asarray(extractor.feature_names),
                feature_set=args.features,
                band_names=np.asarray(TIFF_BAND_NAMES),
                input_scale='original TIFF pixel values',
            )
        # Publish a complete file without replacing an existing destination.
        os.link(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f'Saved {len(paths)} x {extractor.num_features} features to {args.output}')


if __name__ == '__main__':
    main()
