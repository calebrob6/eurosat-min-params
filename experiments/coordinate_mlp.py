#!/usr/bin/env python
"""Centroid-only MLP parameter/accuracy sweep on the random EuroSAT split.

Each GeoTIFF centroid is transformed to WGS84 latitude/longitude. Deterministic
raw, spherical, polynomial, and Fourier encodings are compared across compact
one- and two-hidden-layer ReLU MLPs. Candidate selection uses train and
validation only; test is evaluated only for the final multi-seed validation
Pareto frontier.

The input standardizer can be folded into the first affine layer. The 10-class
output can likewise be stored with one implicit reference class, so parameter
counts include every hidden weight and bias plus only 9 output rows.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
import rasterio
from rasterio.warp import transform
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.data import list_split  # noqa: E402

SCREEN_SEED = 0
ROBUST_SEEDS = (0, 1, 2, 3, 4)
BASE_ENCODING_NAMES = (
    'raw',
    'sphere',
    'poly2',
    'poly3',
    'poly4',
    'fourier_axes3',
    'fourier_axes5',
    'fourier_axes8',
    'fourier_axes11',
    'fourier_diag4',
    'fourier_diag7',
)
BASE_ARCHITECTURES = (
    (2,),
    (4,),
    (8,),
    (16,),
    (32,),
    (4, 2),
    (4, 4),
    (8, 4),
    (8, 8),
    (16, 4),
    (16, 8),
    (16, 16),
    (32, 8),
    (32, 16),
)
FOURIER_WAVELENGTHS = {
    'fourier_axes3': (0.02, 0.2, 2.0),
    'fourier_axes5': (0.01, 0.03, 0.1, 0.3, 1.0),
    'fourier_axes8': (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0),
    'fourier_axes11': (
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ),
    'fourier_axes16': (
        0.005,
        0.0075,
        0.01,
        0.015,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.3,
        0.5,
        1.0,
        2.0,
        5.0,
    ),
    'fourier_diag4': (0.03, 0.1, 0.3, 1.0),
    'fourier_diag7': (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
    'fourier_dir4': (
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ),
    'fourier_dir8': (
        0.01,
        0.02,
        0.05,
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
    ),
}
ADVANCED_ARCHITECTURES = {
    'raw': ((64,), (128,), (64, 32), (64, 64), (128, 32), (128, 64)),
    'poly3': ((64,), (128,), (64, 32), (64, 64)),
    'fourier_axes11': (
        (64,),
        (128,),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
    ),
    'fourier_axes16': (
        (8,),
        (16,),
        (32,),
        (64,),
        (128,),
        (16, 8),
        (32, 16),
        (64, 32),
        (64, 64),
        (128, 64),
    ),
    'fourier_dir4': (
        (8,),
        (16,),
        (32,),
        (64,),
        (16, 8),
        (32, 16),
        (64, 32),
    ),
    'fourier_dir8': (
        (8,),
        (16,),
        (32,),
        (64,),
        (16, 8),
        (32, 16),
        (64, 32),
    ),
    'rbf_grid8': ((2,), (4,), (8,), (16,), (32,), (8, 4), (16, 8)),
    'rbf_grid4': ((2,), (4,), (8,), (16,), (32,), (8, 4), (16, 8)),
    'rbf_grid2': ((2,), (4,), (8,), (16,), (8, 4), (16, 8)),
    'rbf_grid2_narrow': ((2,), (4,), (8,), (16,), (8, 4), (16, 8)),
}


@dataclass(frozen=True)
class Candidate:
    """One coordinate encoding and MLP architecture."""

    encoding: str
    architecture: tuple[int, ...]
    input_dim: int
    parameters: int

    @property
    def key(self) -> str:
        architecture = 'x'.join(map(str, self.architecture))
        return f'{self.encoding}__{architecture}'


@dataclass
class TrainedMLP:
    """A fitted standardizer and MLP."""

    scaler: StandardScaler
    classifier: MLPClassifier
    train_accuracy: float
    val_accuracy: float
    iterations: int
    converged: bool


def tif_centroid_latlon(path: str) -> tuple[float, float]:
    """Read one raster footprint centroid as WGS84 latitude/longitude."""
    with rasterio.open(path) as src:
        if src.crs is None:
            raise ValueError(f'{path} has no CRS')
        x = (src.bounds.left + src.bounds.right) / 2
        y = (src.bounds.bottom + src.bounds.top) / 2
        longitude, latitude = transform(src.crs, 'EPSG:4326', [x], [y])
    return float(latitude[0]), float(longitude[0])


def load_centroids(split: str, workers: int) -> tuple[np.ndarray, np.ndarray]:
    """Load or build the centroid cache for one split."""
    cache_path = Path(CACHE_DIR) / f'{split}_centroid_latlon.npy'
    paths, labels = list_split(split)
    if cache_path.exists():
        coordinates = np.load(cache_path)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = executor.map(tif_centroid_latlon, paths, chunksize=64)
            coordinates = np.asarray(list(rows), dtype=np.float64)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, coordinates)

    if coordinates.shape != (len(paths), 2):
        raise ValueError(
            f'{cache_path} has shape {coordinates.shape}, expected {(len(paths), 2)}'
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f'{cache_path} contains non-finite coordinates')
    if not (
        np.all((-90 <= coordinates[:, 0]) & (coordinates[:, 0] <= 90))
        and np.all((-180 <= coordinates[:, 1]) & (coordinates[:, 1] <= 180))
    ):
        raise ValueError(f'{cache_path} contains invalid latitude/longitude')
    return coordinates, labels


def polynomial_features(coordinates: np.ndarray, degree: int) -> np.ndarray:
    """Fixed two-dimensional polynomial basis through a total degree."""
    latitude = (coordinates[:, 0] - 50.0) / 20.0
    longitude = (coordinates[:, 1] - 10.0) / 40.0
    parts = [
        (latitude ** lat_power) * (longitude ** lon_power)
        for total_degree in range(1, degree + 1)
        for lat_power in range(total_degree, -1, -1)
        for lon_power in (total_degree - lat_power,)
    ]
    return np.column_stack(parts).astype(np.float32)


def axis_fourier_features(
    coordinates: np.ndarray, wavelengths: tuple[float, ...]
) -> np.ndarray:
    """Raw coordinates plus separable latitude/longitude Fourier features."""
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    parts: list[np.ndarray] = [latitude, longitude]
    for wavelength in wavelengths:
        for values in (latitude, longitude):
            phase = 2 * np.pi * values / wavelength
            parts.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(parts).astype(np.float32)


def diagonal_fourier_features(
    coordinates: np.ndarray, wavelengths: tuple[float, ...]
) -> np.ndarray:
    """Raw coordinates plus Fourier features along axes and diagonals."""
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    projections = (
        latitude,
        longitude,
        (latitude + longitude) / math.sqrt(2),
        (latitude - longitude) / math.sqrt(2),
    )
    parts: list[np.ndarray] = [latitude, longitude]
    for wavelength in wavelengths:
        for values in projections:
            phase = 2 * np.pi * values / wavelength
            parts.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(parts).astype(np.float32)


def directional_fourier_features(
    coordinates: np.ndarray,
    wavelengths: tuple[float, ...],
    directions: int,
) -> np.ndarray:
    """Fourier features projected along evenly spaced directions."""
    latitude = coordinates[:, 0]
    longitude = coordinates[:, 1]
    angles = np.linspace(0, np.pi, directions, endpoint=False)
    projections = [
        latitude * np.cos(angle) + longitude * np.sin(angle)
        for angle in angles
    ]
    parts: list[np.ndarray] = [latitude, longitude]
    for wavelength in wavelengths:
        for values in projections:
            phase = 2 * np.pi * values / wavelength
            parts.extend((np.sin(phase), np.cos(phase)))
    return np.column_stack(parts).astype(np.float32)


def rbf_grid_features(
    coordinates: np.ndarray, spacing: float, sigma: float
) -> np.ndarray:
    """Gaussian features on a fixed Europe-wide latitude/longitude grid."""
    latitude_centers = np.arange(28.0, 66.0 + spacing / 2, spacing)
    longitude_centers = np.arange(-22.0, 34.0 + spacing / 2, spacing)
    center_latitude, center_longitude = np.meshgrid(
        latitude_centers, longitude_centers, indexing='ij'
    )
    centers = np.column_stack((
        center_latitude.ravel(),
        center_longitude.ravel(),
    )).astype(np.float32)

    features = np.empty((len(coordinates), len(centers)), dtype=np.float32)
    longitude_scale = np.float32(np.cos(np.deg2rad(50.0)))
    denominator = np.float32(2 * sigma * sigma)
    for start in range(0, len(coordinates), 512):
        stop = min(start + 512, len(coordinates))
        rows = coordinates[start:stop].astype(np.float32)
        latitude_delta = rows[:, None, 0] - centers[None, :, 0]
        longitude_delta = (
            rows[:, None, 1] - centers[None, :, 1]
        ) * longitude_scale
        distance_squared = latitude_delta**2 + longitude_delta**2
        features[start:stop] = np.exp(-distance_squared / denominator)
    return features


def encode(coordinates: np.ndarray, name: str) -> np.ndarray:
    """Apply one deterministic coordinate encoding."""
    latitude = np.deg2rad(coordinates[:, 0])
    longitude = np.deg2rad(coordinates[:, 1])
    if name == 'raw':
        features = coordinates
    elif name == 'sphere':
        features = np.column_stack((
            np.cos(latitude) * np.cos(longitude),
            np.cos(latitude) * np.sin(longitude),
            np.sin(latitude),
        ))
    elif name.startswith('poly'):
        features = polynomial_features(coordinates, int(name.removeprefix('poly')))
    elif name.startswith('fourier_axes'):
        features = axis_fourier_features(coordinates, FOURIER_WAVELENGTHS[name])
    elif name.startswith('fourier_diag'):
        features = diagonal_fourier_features(coordinates, FOURIER_WAVELENGTHS[name])
    elif name.startswith('fourier_dir'):
        directions = int(name.removeprefix('fourier_dir'))
        features = directional_fourier_features(
            coordinates, FOURIER_WAVELENGTHS[name], directions
        )
    elif name.startswith('rbf_grid'):
        narrow = name.endswith('_narrow')
        spacing_text = name.removeprefix('rbf_grid').removesuffix('_narrow')
        spacing = float(spacing_text)
        sigma = spacing / 2 if narrow else spacing
        features = rbf_grid_features(coordinates, spacing, sigma)
    else:
        raise ValueError(f'unknown coordinate encoding {name!r}')

    features = np.asarray(features, dtype=np.float32)
    if not np.isfinite(features).all():
        raise ValueError(f'{name} encoding contains non-finite values')
    return features


def candidate_specs() -> list[tuple[str, tuple[int, ...]]]:
    """Return the broad base grid plus targeted higher-capacity candidates."""
    specs = [
        (encoding, architecture)
        for encoding in BASE_ENCODING_NAMES
        for architecture in BASE_ARCHITECTURES
    ]
    specs.extend(
        (encoding, architecture)
        for encoding, architectures in ADVANCED_ARCHITECTURES.items()
        for architecture in architectures
    )
    return list(dict.fromkeys(specs))


def reference_class_parameter_count(
    input_dim: int, architecture: tuple[int, ...]
) -> int:
    """Count all hidden parameters and nine explicit output rows."""
    parameters = 0
    previous = input_dim
    for width in architecture:
        parameters += previous * width + width
        previous = width
    return parameters + 9 * (previous + 1)


def train_candidate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    architecture: tuple[int, ...],
    seed: int,
    max_iter: int,
) -> TrainedMLP:
    """Train one standardized MLP without consulting the external test set."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    val_scaled = scaler.transform(x_val).astype(np.float32)
    classifier = MLPClassifier(
        hidden_layer_sizes=architecture,
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=3e-3,
        max_iter=max_iter,
        shuffle=True,
        random_state=seed,
        tol=1e-5,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        classifier.fit(train_scaled, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(train_scaled))
    val_accuracy = accuracy_score(y_val, classifier.predict(val_scaled))
    return TrainedMLP(
        scaler=scaler,
        classifier=classifier,
        train_accuracy=train_accuracy,
        val_accuracy=val_accuracy,
        iterations=classifier.n_iter_,
        converged=classifier.n_iter_ < max_iter,
    )


def pareto_frontier(
    records: list[dict[str, object]], accuracy_key: str
) -> list[dict[str, object]]:
    """Return candidates that improve accuracy as parameters increase."""
    ordered = sorted(
        records,
        key=lambda record: (
            int(record['parameters']),
            -float(record[accuracy_key]),
        ),
    )
    frontier: list[dict[str, object]] = []
    best_accuracy = -1.0
    seen_parameters: set[int] = set()
    for record in ordered:
        parameters = int(record['parameters'])
        if parameters in seen_parameters:
            continue
        seen_parameters.add(parameters)
        accuracy = float(record[accuracy_key])
        if accuracy > best_accuracy:
            frontier.append(record)
            best_accuracy = accuracy
    return frontier


def select_robust_candidates(
    screen_records: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Keep the screen frontier plus strong candidates across parameter scales."""
    selected = {
        str(record['candidate']): record
        for record in pareto_frontier(screen_records, 'val_accuracy')
    }
    for record in sorted(
        screen_records, key=lambda item: float(item['val_accuracy']), reverse=True
    )[:10]:
        selected[str(record['candidate'])] = record

    bounds = (50, 100, 200, 400, 800, 1600, 3200, math.inf)
    lower = 0
    for upper in bounds:
        in_bin = [
            record
            for record in screen_records
            if lower <= int(record['parameters']) < upper
        ]
        for record in sorted(
            in_bin, key=lambda item: float(item['val_accuracy']), reverse=True
        )[:2]:
            selected[str(record['candidate'])] = record
        lower = upper
    return sorted(selected.values(), key=lambda record: int(record['parameters']))


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    """Write records using their common insertion-ordered schema."""
    with Path(path).open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values: list[float]) -> tuple[float, float]:
    """Return mean and sample standard deviation."""
    array = np.asarray(values)
    return float(array.mean()), float(array.std(ddof=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--max-iter', type=int, default=400)
    parser.add_argument(
        '--screen-output',
        default=os.path.join(
            os.path.dirname(__file__), 'coordinate_mlp_screen.csv'
        ),
    )
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'coordinate_mlp_result.csv'
        ),
    )
    args = parser.parse_args()

    coordinates: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split in ('train', 'val', 'test'):
        coordinates[split], labels[split] = load_centroids(split, args.workers)

    specs = candidate_specs()
    encoding_names = list(dict.fromkeys(encoding for encoding, _ in specs))
    encoded = {
        name: {
            split: encode(coordinates[split], name)
            for split in ('train', 'val', 'test')
        }
        for name in encoding_names
    }
    candidates = [
        Candidate(
            encoding=name,
            architecture=architecture,
            input_dim=encoded[name]['train'].shape[1],
            parameters=reference_class_parameter_count(
                encoded[name]['train'].shape[1], architecture
            ),
        )
        for name, architecture in specs
    ]

    screen_records: list[dict[str, object]] = []
    for number, candidate in enumerate(candidates, start=1):
        model = train_candidate(
            encoded[candidate.encoding]['train'],
            labels['train'],
            encoded[candidate.encoding]['val'],
            labels['val'],
            candidate.architecture,
            SCREEN_SEED,
            args.max_iter,
        )
        record: dict[str, object] = {
            'candidate': candidate.key,
            'encoding': candidate.encoding,
            'input_dim': candidate.input_dim,
            'architecture': 'x'.join(map(str, candidate.architecture)),
            'parameters': candidate.parameters,
            'seed': SCREEN_SEED,
            'train_accuracy': f'{model.train_accuracy:.6f}',
            'val_accuracy': f'{model.val_accuracy:.6f}',
            'iterations': model.iterations,
            'converged': model.converged,
        }
        screen_records.append(record)
        print(
            f'screen {number:3d}/{len(candidates)} {candidate.key:25s} '
            f'p={candidate.parameters:4d} val={model.val_accuracy:.4f}',
            flush=True,
        )
    write_csv(args.screen_output, screen_records)

    candidate_lookup = {candidate.key: candidate for candidate in candidates}
    robust_candidates = select_robust_candidates(screen_records)
    robust_records: list[dict[str, object]] = []
    robust_models: dict[tuple[str, int], TrainedMLP] = {}
    for number, screen_record in enumerate(robust_candidates, start=1):
        candidate = candidate_lookup[str(screen_record['candidate'])]
        val_accuracies: list[float] = []
        train_accuracies: list[float] = []
        iterations: list[int] = []
        for seed in ROBUST_SEEDS:
            model = train_candidate(
                encoded[candidate.encoding]['train'],
                labels['train'],
                encoded[candidate.encoding]['val'],
                labels['val'],
                candidate.architecture,
                seed,
                args.max_iter,
            )
            robust_models[(candidate.key, seed)] = model
            val_accuracies.append(model.val_accuracy)
            train_accuracies.append(model.train_accuracy)
            iterations.append(model.iterations)
        val_mean, val_std = mean_std(val_accuracies)
        train_mean, train_std = mean_std(train_accuracies)
        robust_records.append({
            'candidate': candidate.key,
            'encoding': candidate.encoding,
            'input_dim': candidate.input_dim,
            'architecture': 'x'.join(map(str, candidate.architecture)),
            'parameters': candidate.parameters,
            'seeds': ';'.join(map(str, ROBUST_SEEDS)),
            'train_accuracy_mean': train_mean,
            'train_accuracy_stdev': train_std,
            'val_accuracy_mean': val_mean,
            'val_accuracy_stdev': val_std,
            'iterations_mean': float(np.mean(iterations)),
        })
        print(
            f'robust {number:2d}/{len(robust_candidates)} {candidate.key:25s} '
            f'p={candidate.parameters:4d} val={val_mean:.4f}+/-{val_std:.4f}',
            flush=True,
        )

    selected_records = pareto_frontier(robust_records, 'val_accuracy_mean')
    selected_keys = {str(record['candidate']) for record in selected_records}
    final_records: list[dict[str, object]] = []
    for record in robust_records:
        candidate_key = str(record['candidate'])
        selected = candidate_key in selected_keys
        test_accuracies: list[float] = []
        if selected:
            encoding = str(record['encoding'])
            test_features = encoded[encoding]['test']
            for seed in ROBUST_SEEDS:
                model = robust_models[(candidate_key, seed)]
                predictions = model.classifier.predict(
                    model.scaler.transform(test_features).astype(np.float32)
                )
                test_accuracies.append(
                    accuracy_score(labels['test'], predictions)
                )
            test_mean, test_std = mean_std(test_accuracies)
            print(
                f'test {candidate_key:25s} p={int(record["parameters"]):4d} '
                f'acc={test_mean:.4f}+/-{test_std:.4f}',
                flush=True,
            )
        else:
            test_mean = math.nan
            test_std = math.nan
        final_records.append({
            **record,
            'validation_pareto': selected,
            'test_accuracies': ';'.join(
                f'{accuracy:.6f}' for accuracy in test_accuracies
            ),
            'test_accuracy_mean': test_mean,
            'test_accuracy_stdev': test_std,
        })
    write_csv(args.output, final_records)


if __name__ == '__main__':
    main()
