#!/usr/bin/env python
"""Evaluate the fixed 306-parameter model on TorchGeo's spatial splits.

The selected feature set and C=3 regularization are held fixed. For each
training-data fraction, only the affine head and its folded standardizer are
refit. Smaller fractions use deterministic stratified subsamples of the spatial
training split; the spatial validation and test splits remain unchanged.

The existing random-split feature caches cover all 27,000 EuroSAT samples.
This script remaps those rows by filename instead of recomputing the same fixed
features, and verifies that the random and spatial splits contain the same
sample universe before fitting anything.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from region_shape_ceiling96 import load_pool, load_region_shape  # noqa: E402
from region_shape_prune_ceiling96_33 import (  # noqa: E402
    C,
    FINAL_POOL_IDX,
    REGION_SHAPE_NAME,
)
from src.cache import CACHE_DIR  # noqa: E402
from src.data import CLASSES, CLASS_TO_IDX, DATA_ROOT  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402

SOURCE_SPLITS = ('train', 'val', 'test')
DEFAULT_FRACTIONS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)
DEFAULT_SEEDS = tuple(range(10))
TORCHGEO_DATA_REVISION = '1ce6f1bfb56db63fd91b6ecc466ea67f2509774c'
TORCHGEO_DATA_URL = (
    f'https://hf.co/datasets/torchgeo/eurosat/resolve/{TORCHGEO_DATA_REVISION}'
)
SPATIAL_SPLIT_SHA256 = {
    'train': '2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f',
    'val': '6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b',
    'test': 'de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452',
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_spatial_splits(download: bool) -> dict[str, Path]:
    """Validate, and optionally download, TorchGeo's spatial split files."""
    paths: dict[str, Path] = {}
    root = Path(DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for split, expected_sha256 in SPATIAL_SPLIT_SHA256.items():
        filename = f'eurosat-spatial-{split}.txt'
        path = root / filename
        if not path.exists():
            if not download:
                raise FileNotFoundError(
                    f'{path} is missing; rerun with --download-splits'
                )
            fd, temporary_name = tempfile.mkstemp(
                prefix=f'.{filename}.', dir=root
            )
            os.close(fd)
            temporary = Path(temporary_name)
            try:
                with urlopen(f'{TORCHGEO_DATA_URL}/{filename}') as response:
                    with temporary.open('wb') as dst:
                        shutil.copyfileobj(response, dst)
                actual_sha256 = sha256(temporary)
                if actual_sha256 != expected_sha256:
                    raise ValueError(
                        f'{filename} SHA-256 is {actual_sha256}, '
                        f'expected {expected_sha256}'
                    )
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        actual_sha256 = sha256(path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f'{path} SHA-256 is {actual_sha256}, expected {expected_sha256}'
            )
        paths[split] = path
    return paths


def read_split(path: Path) -> list[str]:
    """Read non-empty sample names from a TorchGeo split file."""
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f'{path} contains duplicate sample names')
    return names


def labels_from_names(names: list[str]) -> np.ndarray:
    """Derive project-order class labels from EuroSAT filenames."""
    labels: list[int] = []
    for name in names:
        class_name = Path(name).stem.rsplit('_', 1)[0]
        if class_name not in CLASS_TO_IDX:
            raise ValueError(f'unknown EuroSAT class in split entry {name!r}')
        labels.append(CLASS_TO_IDX[class_name])
    return np.asarray(labels, dtype=np.int64)


def load_source_features() -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load the final 33 features and their filenames across random splits."""
    all_names: list[str] = []
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    region_column: int | None = None

    for split in SOURCE_SPLITS:
        names = read_split(Path(DATA_ROOT) / f'eurosat-{split}.txt')
        pool = load_pool(split)
        region, region_names = load_region_shape(split)
        if region_column is None:
            region_column = region_names.index(REGION_SHAPE_NAME)
        elif region_names[region_column] != REGION_SHAPE_NAME:
            raise ValueError('region-shape feature order differs between splits')
        labels = np.load(os.path.join(CACHE_DIR, f'{split}_y.npy'))
        if not (len(names) == pool.shape[0] == region.shape[0] == len(labels)):
            raise ValueError(f'{split} cache rows do not match its split file')
        expected_labels = labels_from_names(names)
        if not np.array_equal(labels, expected_labels):
            raise ValueError(f'{split} cached labels do not match its split file')

        selected = np.column_stack(
            (pool[:, FINAL_POOL_IDX], region[:, region_column])
        ).astype(np.float32)
        all_names.extend(names)
        feature_parts.append(selected)
        label_parts.append(labels)

    if len(all_names) != len(set(all_names)):
        raise ValueError('random train/validation/test splits overlap')
    features = np.concatenate(feature_parts)
    labels = np.concatenate(label_parts)
    if features.shape != (27000, 33):
        raise ValueError(f'expected a (27000, 33) feature matrix, got {features.shape}')
    return all_names, features, labels


def remap_spatial_splits(
    split_paths: dict[str, Path],
    source_names: list[str],
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Remap cached feature rows into TorchGeo's spatial split order."""
    spatial_names = {split: read_split(path) for split, path in split_paths.items()}
    source_set = set(source_names)
    spatial_set = set().union(*(set(names) for names in spatial_names.values()))
    if source_set != spatial_set:
        missing = len(source_set - spatial_set)
        extra = len(spatial_set - source_set)
        raise ValueError(
            f'random and spatial splits differ: {missing} missing, {extra} extra'
        )
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = set(spatial_names[left]) & set(spatial_names[right])
        if overlap:
            raise ValueError(
                f'spatial {left}/{right} splits overlap by {len(overlap)} samples'
            )

    source_index = {name: index for index, name in enumerate(source_names)}
    remapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, names in spatial_names.items():
        indices = np.fromiter(
            (source_index[name] for name in names), dtype=np.int64, count=len(names)
        )
        split_labels = labels[indices]
        expected_labels = labels_from_names(names)
        if not np.array_equal(split_labels, expected_labels):
            raise ValueError(f'spatial {split} labels do not match filenames')
        remapped[split] = (features[indices], split_labels)
    return remapped


def spatial_test_selection_overlap(split_paths: dict[str, Path]) -> tuple[int, int]:
    """Count spatial-test samples seen during random-split model selection."""
    random_selection_names = set(
        read_split(Path(DATA_ROOT) / 'eurosat-train.txt')
    )
    random_selection_names.update(
        read_split(Path(DATA_ROOT) / 'eurosat-val.txt')
    )
    spatial_test_names = read_split(split_paths['test'])
    overlap = len(random_selection_names & set(spatial_test_names))
    return overlap, len(spatial_test_names)


def subset_indices(
    labels: np.ndarray, fraction_percent: float, seed: int
) -> np.ndarray:
    """Return one deterministic stratified training subset."""
    if fraction_percent == 100:
        return np.arange(len(labels))
    n_train = round(len(labels) * fraction_percent / 100)
    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=n_train, random_state=seed
    )
    indices, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return indices


def metric_pair(labels: np.ndarray, predictions: np.ndarray) -> tuple[float, float]:
    """Return ordinary and macro per-class accuracy."""
    return (
        accuracy_score(labels, predictions),
        balanced_accuracy_score(labels, predictions),
    )


def format_mean_std(values: list[float]) -> str:
    """Format a mean and between-subsample standard deviation."""
    array = np.asarray(values)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return f'{array.mean():.4f} +/- {std:.4f}'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=list(DEFAULT_FRACTIONS),
        help='percentages of the 16,200-image spatial training split',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(DEFAULT_SEEDS),
        help='stratified-subsample seeds (100%% is fit once)',
    )
    parser.add_argument(
        '--download-splits',
        action='store_true',
        help='download missing official TorchGeo spatial split files',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'eval_spatial_fractions_result.txt'
        ),
    )
    args = parser.parse_args()

    fractions = sorted(set(args.fractions))
    if not fractions or any(value <= 0 or value > 100 for value in fractions):
        parser.error('--fractions must contain values in (0, 100]')
    if not args.seeds:
        parser.error('--seeds must contain at least one integer')

    split_paths = ensure_spatial_splits(args.download_splits)
    selection_overlap, spatial_test_size = spatial_test_selection_overlap(
        split_paths
    )
    source_names, features, labels = load_source_features()
    spatial = remap_spatial_splits(split_paths, source_names, features, labels)
    x_train, y_train = spatial['train']
    x_val, y_val = spatial['val']
    x_test, y_test = spatial['test']

    runs: list[dict[str, float | int]] = []
    full_test_predictions: np.ndarray | None = None
    for fraction in fractions:
        seeds = (args.seeds[0],) if fraction == 100 else args.seeds
        for seed in seeds:
            indices = subset_indices(y_train, fraction, seed)
            w, b, feature_idx = fit_folded_logreg(
                x_train[indices], y_train[indices], C=C
            )
            val_predictions = predict(x_val, w, b, feature_idx)
            test_predictions = predict(x_test, w, b, feature_idx)
            val_accuracy, val_balanced = metric_pair(y_val, val_predictions)
            test_accuracy, test_balanced = metric_pair(y_test, test_predictions)
            runs.append({
                'fraction': fraction,
                'seed': seed,
                'n_train': len(indices),
                'val_accuracy': val_accuracy,
                'val_balanced': val_balanced,
                'test_accuracy': test_accuracy,
                'test_balanced': test_balanced,
            })
            if fraction == 100:
                full_test_predictions = test_predictions

    lines = [
        'EuroSAT 306-parameter model on TorchGeo longitude-based spatial splits',
        '',
        'Protocol',
        '--------',
        'fixed feature subset=33 features from region_shape_prune_ceiling96_33',
        f'fixed C={C:g}',
        'head=9-row reference-class logistic regression, 9*(33+1)=306 parameters',
        'fractions are stratified subsets of the 16,200-image spatial train split',
        f'subsample seeds={",".join(map(str, args.seeds))}',
        '100% is deterministic and fit once',
        'spatial validation/test remain fixed at 5,400 images each',
        'accuracy=sample-weighted; balanced=mean per-class recall',
        f'TorchGeo split revision={TORCHGEO_DATA_REVISION}',
        'model-selection caveat=post-hoc repartition of the same 27,000 samples',
        (
            'spatial-test samples previously in random train/validation='
            f'{selection_overlap}/{spatial_test_size} '
            f'({selection_overlap / spatial_test_size:.1%})'
        ),
        '',
        'Summary (mean +/- standard deviation across subsample seeds)',
        '------------------------------------------------------------',
        'fraction n_train repeats val_accuracy val_balanced test_accuracy test_balanced',
    ]
    for fraction in fractions:
        selected_runs = [run for run in runs if run['fraction'] == fraction]
        lines.append(
            f'{fraction:7g}% '
            f'{int(selected_runs[0]["n_train"]):7d} '
            f'{len(selected_runs):7d} '
            f'{format_mean_std([float(run["val_accuracy"]) for run in selected_runs])} '
            f'{format_mean_std([float(run["val_balanced"]) for run in selected_runs])} '
            f'{format_mean_std([float(run["test_accuracy"]) for run in selected_runs])} '
            f'{format_mean_std([float(run["test_balanced"]) for run in selected_runs])}'
        )

    lines.extend([
        '',
        'Individual runs',
        '---------------',
        'fraction seed n_train val_accuracy val_balanced test_accuracy test_balanced',
    ])
    for run in runs:
        lines.append(
            f'{float(run["fraction"]):7g}% '
            f'{int(run["seed"]):4d} '
            f'{int(run["n_train"]):7d} '
            f'{float(run["val_accuracy"]):.4f} '
            f'{float(run["val_balanced"]):.4f} '
            f'{float(run["test_accuracy"]):.4f} '
            f'{float(run["test_balanced"]):.4f}'
        )

    if full_test_predictions is not None:
        lines.extend(['', '100% training fraction: test accuracy by class', '------------------------------------------------'])
        for class_index, class_name in enumerate(CLASSES):
            mask = y_test == class_index
            class_accuracy = accuracy_score(
                y_test[mask], full_test_predictions[mask]
            )
            lines.append(
                f'{class_name:22s} n={mask.sum():4d} accuracy={class_accuracy:.4f}'
            )

    output = '\n'.join(lines) + '\n'
    print(output, end='')
    Path(args.output).write_text(output)


if __name__ == '__main__':
    main()
