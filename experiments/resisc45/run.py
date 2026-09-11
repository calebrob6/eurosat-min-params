"""Fit ImageStats and the fixed RGB 33-feature model on RESISC45."""

from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
from pathlib import Path
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

from src.linmodel import fit_folded_logreg, predict, predict_reference_class, to_reference_class
from .download import prepare_dataset
from .rgb import SELECTED_INDICES, rgb_feature_pool

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')
SIZES = (18900, 6300, 6300)
SPLIT_SHA256 = (
    'ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b',
    '08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342',
    'e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805',
)
CS = (.01, .03, .1, .3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000)


def split_records(root: Path) -> tuple[dict, tuple[str, ...]]:
    """Read and validate the official split files and their image paths."""
    image_root = root / 'NWPU-RESISC45'
    classes = tuple(sorted(path.name for path in image_root.iterdir() if path.is_dir()))
    if len(classes) != 45:
        raise ValueError('expected 45 RESISC45 class directories')
    records, seen = {}, set()
    for split, size, checksum in zip(SPLITS, SIZES, SPLIT_SHA256, strict=True):
        file = root / f'resisc45-{split}.txt'
        if hashlib.sha256(file.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'wrong official split checksum: {file}')
        names = [line.strip() for line in file.read_text().splitlines() if line.strip()]
        if len(names) != size or len(set(names)) != size or seen.intersection(names):
            raise ValueError(f'{split}: wrong split size or overlapping images')
        seen.update(names)
        paths, labels = [], []
        for name in names:
            if Path(name).name != name or Path(name).suffix != '.jpg':
                raise ValueError(f'invalid image filename: {name}')
            label = Path(name).stem.rsplit('_', 1)[0]
            path = image_root / label / name
            if label not in classes or not path.is_file():
                raise ValueError(f'missing image or invalid class: {path}')
            paths.append(path)
            labels.append(classes.index(label))
        records[split] = {'paths': paths, 'labels': np.asarray(labels), 'filenames': np.asarray(names)}
    return records, classes


def read_image(path: Path) -> np.ndarray:
    """Read an original JPEG using the saved bilinear resize convention."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if (source.count, source.height, source.width) != (3, 256, 256):
                raise ValueError(f'{path}: expected an original 3-band 256x256 image')
            image = source.read(out_shape=(3, 64, 64), resampling=Resampling.bilinear)
    if image.dtype != np.uint8:
        raise ValueError(f'{path}: expected uint8 JPEG values')
    return image


def image_statistics(images: np.ndarray) -> np.ndarray:
    """Return per-band mean, population standard deviation, minimum, and maximum."""
    flat = images.astype(np.float32).reshape(len(images), 3, -1)
    return np.stack((flat.mean(2), flat.std(2), flat.min(2), flat.max(2)), axis=2).reshape(len(images), 12)


def extract(record: dict, batch_size: int) -> dict:
    """Compute both feature sets in bounded batches from original images."""
    statistics, pool = [], []
    names = None
    for start in range(0, len(record['paths']), batch_size):
        images = np.stack([read_image(path) for path in record['paths'][start:start + batch_size]])
        statistics.append(image_statistics(images))
        values, names = rgb_feature_pool(images)
        pool.append(values)
    return {**record, 'imagestats': np.concatenate(statistics),
            'selected33': np.concatenate(pool), 'pool_names': names}


def main() -> None:
    """Run both baselines using train-only fitting and validation selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/RESISC45')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/resisc45')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--batch-size', type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    output = args.output.resolve()
    if output == ROOT / 'output' or not output.is_relative_to(ROOT / 'output'):
        parser.error('--output must be a subdirectory of output/')
    if output.exists() and any(output.iterdir()):
        parser.error('output directory is not empty; choose a new --output')
    if args.download:
        prepare_dataset(args.root, SPLIT_SHA256)
    records, classes = split_records(args.root)
    train, val = (extract(records[split], args.batch_size) for split in ('train', 'val'))
    fitted = {}
    for key, indices, grid in (
        ('imagestats', np.arange(12), CS),
        ('selected33', np.asarray(SELECTED_INDICES), (30.0,)),
    ):
        best = None
        for c in grid:
            model = fit_folded_logreg(train[key], train['labels'], indices, C=c)
            score = float((predict(val[key], *model) == val['labels']).mean())
            if best is None or score > best[0]:
                best = (score, c, model)
        fitted[key] = best
    test = extract(records['test'], args.batch_size)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, (_, c, (w, b, indices)) in fitted.items():
        wr, br = to_reference_class(w, b)
        feature_names = (
            [train['pool_names'][i] for i in indices] if key == 'selected33'
            else [f'{stat}_{band}' for band in ('red', 'green', 'blue') for stat in ('mean', 'std', 'min', 'max')]
        )
        np.savez_compressed(output / f'{key}.npz', W=wr, b=br, feature_idx=indices,
                            feature_names=np.asarray(feature_names), C=c, ref_class=0,
                            classes=np.asarray(classes), preprocessing='rasterio bilinear RGB uint8 64x64')
        for split, data in (('val', val), ('test', test)):
            predictions = predict_reference_class(data[key], wr, br, indices)
            correct = int((predictions == data['labels']).sum())
            rows.append(dict(model=key, split=split, features=len(indices),
                             learned_parameters=wr.size + br.size, C=c, n_images=len(predictions),
                             correct=correct, accuracy=correct / len(predictions)))
            print(f'{key} {split}: {100 * correct / len(predictions):.2f}% ({wr.size + br.size} values)', flush=True)
    with (output / 'results.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
