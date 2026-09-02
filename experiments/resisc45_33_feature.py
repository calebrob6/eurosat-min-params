#!/usr/bin/env python
"""Adapt the 33-feature handcrafted linear model to RGB RESISC45.

RESISC45 has RGB imagery rather than EuroSAT's 13 Sentinel-2 bands. This trial
builds an analogous zero-parameter RGB pool from color distributions,
multiscale gradients, coherence/orientation, RGB index-map texture, and global
line/corner/LBP/blob/region-shape summaries. Train-only L1 ranking selects
exactly 33 features, validation selects C, and test is evaluated only after
those choices are fixed.

For 45 classes, a 33-feature reference-class affine head stores exactly
``(45 - 1) * (33 + 1) = 1,496`` learned values. The learning curve refits that
fixed feature subset and C on one stratified subsample (seed 0) per fraction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import sys
import tempfile
import warnings
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from gstruct2_features_lib import (  # noqa: E402
    _blob_stats,
    _corner_stats,
    _lbp_stats,
    _radial_slope,
)
from line_features_lib import _hough_line_stats  # noqa: E402
from src.cache import CACHE_DIR  # noqa: E402
from src.features import patch_features  # noqa: E402
from src.linmodel import (  # noqa: E402
    fit_folded_logreg,
    l1_rank,
    predict,
    predict_reference_class,
    to_reference_class,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPOSITORY_ROOT / 'data' / 'RESISC45'
IMAGE_ROOT = DATA_ROOT / 'NWPU-RESISC45'
ARCHIVE_PATH = DATA_ROOT / 'NWPU-RESISC45.zip'
DATA_REVISION = '883edc0eee77b2c84225472f10f126e3ed83fa6e'
BASE_URL = f'https://hf.co/datasets/isaaccorley/resisc45/resolve/{DATA_REVISION}'
ARCHIVE_SHA256 = 'beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53'
SPLIT_SHA256 = {
    'train': 'ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b',
    'val': '08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342',
    'test': 'e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805',
}
EXPECTED_SPLIT_SIZES = {'train': 18900, 'val': 6300, 'test': 6300}
FRACTIONS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
C_GRID = (
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
    300.0,
    1000.0,
    3000.0,
    10000.0,
)
SELECTED_FEATURES = 33
NUM_CLASSES = 45
REFERENCE_CLASS_PARAMETERS = (NUM_CLASSES - 1) * (SELECTED_FEATURES + 1)
SEED = 0
_EPS = 1e-6


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, path: Path, expected_sha256: str) -> None:
    """Download one checksum-pinned file using an atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urlopen(url) as response:
            with temporary.open('wb') as dst:
                shutil.copyfileobj(response, dst)
        actual_sha256 = sha256(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f'{path.name} SHA-256 is {actual_sha256}, '
                f'expected {expected_sha256}'
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_dataset(download: bool) -> None:
    """Validate, and optionally download, the TorchGeo RESISC45 release."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for split, expected_sha256 in SPLIT_SHA256.items():
        path = DATA_ROOT / f'resisc45-{split}.txt'
        if not path.exists():
            if not download:
                raise FileNotFoundError(f'{path} is missing; rerun with --download')
            download_file(
                f'{BASE_URL}/resisc45-{split}.txt', path, expected_sha256
            )
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f'{path} SHA-256 is {actual_sha256}, expected {expected_sha256}'
            )

    if not IMAGE_ROOT.exists():
        if not ARCHIVE_PATH.exists():
            if not download:
                raise FileNotFoundError(
                    f'{IMAGE_ROOT} is missing; rerun with --download'
                )
            download_file(
                f'{BASE_URL}/NWPU-RESISC45.zip',
                ARCHIVE_PATH,
                ARCHIVE_SHA256,
            )
        actual_sha256 = sha256(ARCHIVE_PATH)
        if actual_sha256 != ARCHIVE_SHA256:
            raise ValueError(
                f'{ARCHIVE_PATH} SHA-256 is {actual_sha256}, '
                f'expected {ARCHIVE_SHA256}'
            )
        with zipfile.ZipFile(ARCHIVE_PATH) as archive:
            archive.extractall(DATA_ROOT)


def class_names() -> tuple[str, ...]:
    """Return the 45 alphabetically ordered ImageFolder class names."""
    names = tuple(sorted(path.name for path in IMAGE_ROOT.iterdir() if path.is_dir()))
    if len(names) != NUM_CLASSES:
        raise ValueError(f'expected {NUM_CLASSES} classes, found {len(names)}')
    return names


def list_split(split: str) -> tuple[list[str], np.ndarray]:
    """Return image paths and alphabetical class labels for one split."""
    names = [
        line.strip()
        for line in (DATA_ROOT / f'resisc45-{split}.txt').read_text().splitlines()
        if line.strip()
    ]
    expected_size = EXPECTED_SPLIT_SIZES[split]
    if len(names) != expected_size or len(names) != len(set(names)):
        raise ValueError(
            f'{split} split has {len(names)} rows and {len(set(names))} unique '
            f'names; expected {expected_size}'
        )
    classes = class_names()
    class_to_index = {name: index for index, name in enumerate(classes)}
    paths: list[str] = []
    labels: list[int] = []
    for name in names:
        class_name = Path(name).stem.rsplit('_', 1)[0]
        if class_name not in class_to_index:
            raise ValueError(f'unknown RESISC45 class in {name!r}')
        path = IMAGE_ROOT / class_name / name
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(str(path))
        labels.append(class_to_index[class_name])
    return paths, np.asarray(labels, dtype=np.int64)


def read_rgb64(path: str) -> np.ndarray:
    """Read one RGB JPEG and resize it bilinearly to 64x64."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(path) as src:
            if src.count != 3:
                raise ValueError(f'{path} has {src.count} channels, expected 3')
            image = src.read(
                out_shape=(3, 64, 64), resampling=Resampling.bilinear
            )
    return image.astype(np.uint8)


def load_rgb64(
    split: str, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load or build the uint8 64x64 RGB cache for one split."""
    image_path = Path(CACHE_DIR) / f'resisc45_{split}_x_uint8_64.npy'
    label_path = Path(CACHE_DIR) / f'resisc45_{split}_y.npy'
    paths, expected_labels = list_split(split)
    if not image_path.exists() or not label_path.exists():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        images = np.lib.format.open_memmap(
            image_path,
            mode='w+',
            dtype=np.uint8,
            shape=(len(paths), 3, 64, 64),
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = executor.map(read_rgb64, paths, chunksize=32)
            for index, image in enumerate(rows):
                images[index] = image
        images.flush()
        del images
        np.save(label_path, expected_labels)

    images = np.load(image_path, mmap_mode='r')
    labels = np.load(label_path)
    if images.shape != (len(paths), 3, 64, 64):
        raise ValueError(f'{image_path} has unexpected shape {images.shape}')
    if not np.array_equal(labels, expected_labels):
        raise ValueError(f'{label_path} does not match the official split')
    return images, labels


def pool2(images: np.ndarray) -> np.ndarray:
    """Fixed 2x2 average pooling."""
    n, c, h, w = images.shape
    return images.reshape(n, c, h // 2, 2, w // 2, 2).mean((3, 5))


def gradient_magnitude(images: np.ndarray) -> np.ndarray:
    """Gradient magnitude for a stack of maps."""
    gx = np.diff(images, axis=3)[:, :, :-1, :]
    gy = np.diff(images, axis=2)[:, :, :, :-1]
    return np.sqrt(gx * gx + gy * gy)


def rgb_maps(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Six fixed RGB color maps analogous to the EuroSAT index maps."""
    red, green, blue = images[:, 0], images[:, 1], images[:, 2]

    def ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return (left - right) / (left + right + _EPS)

    total = red + green + blue + _EPS
    maximum = images.max(axis=1)
    minimum = images.min(axis=1)
    maps = np.stack((
        (2 * green - red - blue) / total,
        ratio(green, red),
        ratio(green, blue),
        ratio(red, blue),
        (maximum - minimum) / (maximum + _EPS),
        (maximum + minimum) / 510.0,
    ), axis=1)
    names = ['exg', 'green_red', 'green_blue', 'red_blue', 'saturation', 'lightness']
    return maps.astype(np.float32), names


def rgb_index_texture(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Fine and coarse heterogeneity summaries of the six RGB maps."""
    maps, map_names = rgb_maps(images)
    n, channels = maps.shape[:2]
    flat = maps.reshape(n, channels, -1)
    gradient = gradient_magnitude(maps).reshape(n, channels, -1)
    p10, p90 = np.percentile(flat, (10, 90), axis=2)
    fine_parts = (
        flat.std(axis=2),
        gradient.mean(axis=2),
        gradient.std(axis=2),
        p90 - p10,
    )
    fine_names = [
        f'{prefix}_{name}'
        for prefix in ('rgbstd', 'rgbgm', 'rgbgs', 'rgbspr')
        for name in map_names
    ]

    coarse_maps, _ = rgb_maps(pool2(images))
    coarse_flat = coarse_maps.reshape(n, channels, -1)
    coarse_gradient = gradient_magnitude(coarse_maps).reshape(n, channels, -1)
    coarse_parts = (coarse_flat.std(axis=2), coarse_gradient.mean(axis=2))
    coarse_names = [
        f'{prefix}_{name}'
        for prefix in ('rgb2std', 'rgb2gm')
        for name in map_names
    ]
    features = np.concatenate((*fine_parts, *coarse_parts), axis=1)
    return features.astype(np.float32), fine_names + coarse_names


def rgb_cross_correlation(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Spatial Pearson correlation for the three RGB channel pairs."""
    n = len(images)
    flat = images.reshape(n, 3, -1)
    centered = flat - flat.mean(axis=2, keepdims=True)
    standard_deviation = np.sqrt((centered * centered).mean(axis=2)) + _EPS
    pairs = ((0, 1, 'red_green'), (0, 2, 'red_blue'), (1, 2, 'green_blue'))
    parts: list[np.ndarray] = []
    names: list[str] = []
    for left, right, name in pairs:
        covariance = (centered[:, left] * centered[:, right]).mean(axis=1)
        parts.append(
            covariance / (
                standard_deviation[:, left] * standard_deviation[:, right]
            )
        )
        names.append(f'rgbcorr_{name}')
    return np.column_stack(parts).astype(np.float32), names


def structural_channels(images: np.ndarray) -> dict[str, np.ndarray]:
    """Panchromatic, excess-green, and saturation maps."""
    maps, names = rgb_maps(images)
    map_lookup = {name: maps[:, index] for index, name in enumerate(names)}
    return {
        'pan': images.mean(axis=1),
        'exg': map_lookup['exg'],
        'saturation': map_lookup['saturation'],
    }


def rgb_global_structure(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Line, corner, LBP, blob, and radial-slope features on RGB maps."""
    parts: list[np.ndarray] = []
    names: list[str] = []
    for channel_name, channel in structural_channels(images).items():
        line_peak, line_length, line_top3 = _hough_line_stats(channel)
        corner_fraction, corner_magnitude = _corner_stats(channel)
        lbp_entropy, lbp_uniform = _lbp_stats(channel)
        blob_largest, blob_count, blob_mean = _blob_stats(channel)
        spectral_slope = _radial_slope(channel)
        parts.extend((
            line_peak[:, None],
            line_length[:, None],
            line_top3[:, None],
            corner_fraction[:, None],
            corner_magnitude[:, None],
            lbp_entropy[:, None],
            lbp_uniform[:, None],
            blob_largest[:, None],
            blob_count[:, None],
            blob_mean[:, None],
            spectral_slope[:, None],
        ))
        names.extend((
            f'linepf_{channel_name}',
            f'linepl_{channel_name}',
            f'linet3_{channel_name}',
            f'cornerfrac_{channel_name}',
            f'cornermag_{channel_name}',
            f'lbpent_{channel_name}',
            f'lbpuni_{channel_name}',
            f'bloblargest_{channel_name}',
            f'blobcount_{channel_name}',
            f'blobmean_{channel_name}',
            f'sspectral_{channel_name}',
        ))
    return np.concatenate(parts, axis=1).astype(np.float32), names


def rgb_tail_shape(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Spatial anisotropy and spread of low/high RGB-map quartile tails."""
    _, _, height, width = images.shape
    coordinate_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    coordinate_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    yy, xx = np.meshgrid(coordinate_y, coordinate_x, indexing='ij')
    xx = xx.reshape(1, -1)
    yy = yy.reshape(1, -1)
    parts: list[np.ndarray] = []
    names: list[str] = []
    for channel_name, channel in structural_channels(images).items():
        flat = channel.reshape(len(channel), -1)
        q25, q75 = np.percentile(flat, (25, 75), axis=1).astype(np.float32)
        for tail, weights in (
            ('low', np.maximum(q25[:, None] - flat, 0.0)),
            ('high', np.maximum(flat - q75[:, None], 0.0)),
        ):
            mass = weights.sum(axis=1) + _EPS
            mean_x = (weights * xx).sum(axis=1) / mass
            mean_y = (weights * yy).sum(axis=1) / mass
            delta_x = xx - mean_x[:, None]
            delta_y = yy - mean_y[:, None]
            sxx = (weights * delta_x * delta_x).sum(axis=1) / mass
            syy = (weights * delta_y * delta_y).sum(axis=1) / mass
            sxy = (weights * delta_x * delta_y).sum(axis=1) / mass
            trace = sxx + syy
            anisotropy = np.sqrt((sxx - syy) ** 2 + 4 * sxy * sxy) / (
                trace + _EPS
            )
            spread = np.sqrt(trace)
            parts.extend((anisotropy[:, None], spread[:, None]))
            names.extend((
                f'tail_aniso_{tail}_{channel_name}',
                f'tail_spread_{tail}_{channel_name}',
            ))
    return np.concatenate(parts, axis=1).astype(np.float32), names


def rgb_feature_pool(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build the complete RGB-adapted zero-parameter candidate pool."""
    base, base_names = patch_features(
        images,
        pcts=(10, 25, 50, 75, 90),
        grad_scales=3,
        coherence_scales=2,
        orient_entropy_bins=8,
        orient_hist_bins=4,
        spectral_peak=True,
    )
    correlation, correlation_names = rgb_cross_correlation(images)
    texture, texture_names = rgb_index_texture(images)
    structure, structure_names = rgb_global_structure(images)
    tail, tail_names = rgb_tail_shape(images)
    features = np.concatenate(
        (base, correlation, texture, structure, tail), axis=1
    ).astype(np.float32)
    names = (
        base_names
        + correlation_names
        + texture_names
        + structure_names
        + tail_names
    )
    if features.shape[1] != len(names) or len(names) != len(set(names)):
        raise ValueError('RGB feature pool names are inconsistent')
    if not np.isfinite(features).all():
        raise ValueError('RGB feature pool contains non-finite values')
    return features, names


def load_feature_pool(
    split: str, batch_size: int, workers: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load or build the RGB-adapted feature cache for one split."""
    feature_path = Path(CACHE_DIR) / f'resisc45_{split}_rgb_pool.npy'
    names_path = Path(CACHE_DIR) / 'resisc45_rgb_pool_names.txt'
    images, labels = load_rgb64(split, workers)
    _, expected_names = rgb_feature_pool(images[:1].astype(np.float32))
    if feature_path.exists():
        features = np.load(feature_path)
        names = expected_names
        if (
            not names_path.exists()
            or names_path.read_text().splitlines() != expected_names
        ):
            names_path.write_text('\n'.join(expected_names) + '\n')
    else:
        feature_memmap: np.memmap | None = None
        names: list[str] | None = None
        for start in range(0, len(images), batch_size):
            stop = min(start + batch_size, len(images))
            batch_features, batch_names = rgb_feature_pool(
                images[start:stop].astype(np.float32)
            )
            if feature_memmap is None:
                names = batch_names
                feature_memmap = np.lib.format.open_memmap(
                    feature_path,
                    mode='w+',
                    dtype=np.float32,
                    shape=(len(images), batch_features.shape[1]),
                )
            elif names != batch_names:
                raise ValueError('RGB feature order changed between batches')
            feature_memmap[start:stop] = batch_features
            print(
                f'{split} features {stop}/{len(images)}',
                flush=True,
            )
        if feature_memmap is None or names is None:
            raise ValueError(f'{split} split is empty')
        feature_memmap.flush()
        del feature_memmap
        names_path.write_text('\n'.join(names) + '\n')
        features = np.load(feature_path)

    if features.shape[0] != len(labels) or features.shape[1] != len(names):
        raise ValueError(f'{split} feature cache shape is inconsistent')
    return features, labels, names


def subset_indices(labels: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Return one deterministic stratified subset."""
    if fraction == 100:
        return np.arange(len(labels))
    train_size = round(len(labels) * fraction / 100)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, random_state=seed
    )
    indices, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--Cs', type=float, nargs='+', default=list(C_GRID))
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'resisc45_33_feature_fractions.csv'
        ),
    )
    args = parser.parse_args()

    ensure_dataset(args.download)
    train_features, train_labels, names = load_feature_pool(
        'train', args.batch_size, args.workers
    )
    val_features, val_labels, val_names = load_feature_pool(
        'val', args.batch_size, args.workers
    )
    if names != val_names:
        raise ValueError('train and validation feature schemas differ')

    ranking = l1_rank(train_features, train_labels)
    selected = ranking[:SELECTED_FEATURES]
    selected_names = [names[index] for index in selected]
    validation_results: list[tuple[float, float]] = []
    for regularization in args.Cs:
        w, b, feature_idx = fit_folded_logreg(
            train_features,
            train_labels,
            feature_idx=selected,
            C=regularization,
        )
        val_accuracy = accuracy_score(
            val_labels, predict(val_features, w, b, feature_idx)
        )
        print(f'C={regularization:g} validation={val_accuracy:.4f}', flush=True)
        validation_results.append((regularization, val_accuracy))
    if not validation_results:
        raise ValueError('C grid is empty')
    max_val_accuracy = max(accuracy for _, accuracy in validation_results)
    standard_error = np.sqrt(
        max_val_accuracy * (1 - max_val_accuracy) / len(val_labels)
    )
    one_se_threshold = max_val_accuracy - standard_error
    selected_c, selection_val_accuracy = min(
        (regularization, accuracy)
        for regularization, accuracy in validation_results
        if accuracy >= one_se_threshold
    )
    print(
        f'one-SE threshold={one_se_threshold:.4f}; selected C={selected_c:g} '
        f'validation={selection_val_accuracy:.4f}',
        flush=True,
    )

    test_features, test_labels, test_names = load_feature_pool(
        'test', args.batch_size, args.workers
    )
    if names != test_names:
        raise ValueError('train and test feature schemas differ')

    rows: list[dict[str, str | int | float]] = []
    for fraction in FRACTIONS:
        indices = subset_indices(train_labels, fraction, SEED)
        w, b, feature_idx = fit_folded_logreg(
            train_features[indices],
            train_labels[indices],
            feature_idx=selected,
            C=selected_c,
        )
        predictions = predict(test_features, w, b, feature_idx)
        w_reference, b_reference = to_reference_class(w, b)
        reference_predictions = predict_reference_class(
            test_features,
            w_reference,
            b_reference,
            feature_idx,
        )
        if not np.array_equal(predictions, reference_predictions):
            raise ValueError('reference-class conversion changed predictions')
        if w_reference.size + b_reference.size != REFERENCE_CLASS_PARAMETERS:
            raise ValueError('reference-class parameter count is inconsistent')
        test_accuracy = accuracy_score(test_labels, predictions)
        rows.append({
            'dataset': 'RESISC45',
            'split_protocol': 'TorchGeo fixed random train/val/test',
            'model': 'folded reference-class multinomial logistic regression',
            'feature_pool': 'RGB-adapted handcrafted zero-parameter pool',
            'pool_features': train_features.shape[1],
            'selection': 'train-only L1 coefficient ranking, fixed top 33',
            'selected_features': SELECTED_FEATURES,
            'selected_feature_indices': ';'.join(map(str, selected)),
            'selected_feature_names': ';'.join(selected_names),
            'classes': NUM_CLASSES,
            'learned_parameters': REFERENCE_CLASS_PARAMETERS,
            'regularization_C': selected_c,
            'C_selection': 'smallest C within one standard error of best validation',
            'C_selection_validation_accuracy': f'{selection_val_accuracy:.6f}',
            'max_C_sweep_validation_accuracy': f'{max_val_accuracy:.6f}',
            'one_standard_error_threshold': f'{one_se_threshold:.6f}',
            'resize': 'bilinear 256x256 to 64x64',
            'train_fraction_percent': fraction,
            'n_train': len(indices),
            'seed': SEED,
            'n_test': len(test_labels),
            'test_accuracy': f'{test_accuracy:.6f}',
        })
        print(
            f'fraction={fraction:g}% n={len(indices)} '
            f'test={test_accuracy:.4f}',
            flush=True,
        )

    with Path(args.output).open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
