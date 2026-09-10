#!/usr/bin/env python
"""Rank the full EuroSAT feature pool and trace recursive elimination."""

from __future__ import annotations

import os

# Keep coefficient paths and solver stopping deterministic across runs.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import json
import platform
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from reproduce import CHECKSUMS, prepare_data
from src.data import CLASSES, iter_images, list_split
from src.feature_pool import FULL_POOL_SIZE, full_pool_features
from src.frontier import POOL_INDICES

ROOT = Path(__file__).resolve().parents[2]
SPLITS = ('train', 'val', 'test')
SIZES = {'train': 16200, 'val': 5400, 'test': 5400}
DEFAULT_CS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0)
SOURCE_FILES = (
    'src/data.py',
    'src/features.py',
    'src/extra_features.py',
    'src/frontier.py',
    'src/feature_pool.py',
)


@dataclass
class FittedModel:
    """A classifier and its train-fitted feature standardizer."""

    scaler: StandardScaler
    estimator: LogisticRegression
    feature_indices: np.ndarray
    importances: np.ndarray


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write deterministic JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a nonempty sequence of dictionaries as CSV."""
    if not rows:
        raise ValueError(f'cannot write empty table: {path}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def cache_identity() -> dict:
    """Describe every source and split input that defines cached features."""
    return {
        'schema': 'eurosat-full-pool-389-v2',
        'source_sha256': {name: sha256(ROOT / name) for name in SOURCE_FILES},
        'split_sha256': {name: CHECKSUMS[f'eurosat-{name}.txt'] for name in SPLITS},
    }


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Atomically save a compressed NumPy archive."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix='.npz', delete=False
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(handle, **arrays)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(cache: Path) -> dict:
    """Validate and return an existing feature-cache manifest."""
    path = cache / 'manifest.json'
    if not path.is_file():
        raise FileNotFoundError(f'missing feature cache manifest: {path}')
    manifest = json.loads(path.read_text())
    if manifest.get('identity') != cache_identity():
        raise ValueError(f'{cache}: feature-generating inputs changed; use a new cache')
    expected_files = {f'{split}.npz' for split in SPLITS}
    if set(manifest.get('files', {})) != expected_files:
        raise ValueError(f'{cache}: cache manifest has incomplete split coverage')
    for name, expected in manifest['files'].items():
        file = cache / name
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f'{cache}: checksum mismatch for {name}')
    schema = manifest.get('schema', {})
    if (
        len(schema.get('names', [])) != FULL_POOL_SIZE
        or len(schema.get('families', [])) != FULL_POOL_SIZE
    ):
        raise ValueError(f'{cache}: invalid full-pool schema')
    return manifest


def prepare_cache(cache: Path, batch_size: int, download: bool) -> dict:
    """Extract all 389 features from original TIFFs into an authenticated cache."""
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    prepare_data(download, False)
    manifest_path = cache / 'manifest.json'
    if manifest_path.exists():
        manifest = load_manifest(cache)
        print(f'reusing feature cache: {cache}', flush=True)
        return manifest
    cache.mkdir(parents=True, exist_ok=True)
    if any(cache.iterdir()):
        raise ValueError(
            f'{cache}: incomplete cache is not overwritten; choose a new cache'
        )

    schema = None
    for split in SPLITS:
        paths, labels = list_split(split)
        if (
            len(paths) != SIZES[split]
            or len(labels) != SIZES[split]
            or len(set(paths)) != SIZES[split]
        ):
            raise ValueError(f'{split}: invalid split membership')
        parts = []
        seen = 0
        for images, batch_labels in iter_images(split, batch_size):
            if not np.array_equal(batch_labels, labels[seen : seen + len(images)]):
                raise ValueError(f'{split}: TIFF iterator labels changed')
            values, names, families = full_pool_features(images)
            current_schema = {'names': names, 'families': families}
            if schema is not None and current_schema != schema:
                raise ValueError(
                    'feature names or family order changed between batches'
                )
            schema = current_schema
            parts.append(values)
            seen += len(images)
            if seen % (batch_size * 10) == 0:
                print(f'{split}: extracted {seen}/{len(paths)}', flush=True)
        features = np.concatenate(parts)
        if (
            features.shape != (SIZES[split], FULL_POOL_SIZE)
            or not np.isfinite(features).all()
        ):
            raise ValueError(f'{split}: invalid extracted feature matrix')
        save_npz(
            cache / f'{split}.npz',
            features=features,
            labels=labels,
            filenames=np.asarray([Path(path).name for path in paths]),
        )
        print(f'{split}: extracted {len(paths)} TIFFs', flush=True)

    files = {f'{split}.npz': sha256(cache / f'{split}.npz') for split in SPLITS}
    manifest = {'identity': cache_identity(), 'schema': schema, 'files': files}
    write_json(manifest_path, manifest)
    return manifest


def load_splits(cache: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    """Load all authenticated feature matrices and labels."""
    manifest = load_manifest(cache)
    result = {}
    filenames = []
    for split in SPLITS:
        with np.load(cache / f'{split}.npz', allow_pickle=False) as source:
            result[split] = {
                key: source[key] for key in ('features', 'labels', 'filenames')
            }
        if (
            result[split]['features'].shape != (SIZES[split], FULL_POOL_SIZE)
            or result[split]['labels'].shape != (SIZES[split],)
            or result[split]['filenames'].shape != (SIZES[split],)
            or not np.isfinite(result[split]['features']).all()
        ):
            raise ValueError(f'{split}: invalid cached arrays')
        filenames.extend(result[split]['filenames'].tolist())
    if len(set(filenames)) != sum(SIZES.values()):
        raise ValueError('cached splits overlap')
    return result, manifest


def fit_classifier(
    features: np.ndarray,
    labels: np.ndarray,
    feature_indices: np.ndarray,
    c: float,
    max_iter: int,
) -> FittedModel:
    """Fit standardized multinomial logistic regression and measure coefficients."""
    feature_indices = np.asarray(feature_indices, dtype=np.int64)
    if (
        features.ndim != 2
        or labels.shape != (len(features),)
        or feature_indices.ndim != 1
        or len(feature_indices) < 1
        or len(set(feature_indices.tolist())) != len(feature_indices)
        or feature_indices.min() < 0
        or feature_indices.max() >= features.shape[1]
    ):
        raise ValueError('invalid classifier arrays or feature indices')
    if not np.isfinite(c) or c <= 0 or max_iter < 1:
        raise ValueError('C and max_iter must be positive')
    scaler = StandardScaler()
    standardized = scaler.fit_transform(features[:, feature_indices])
    estimator = LogisticRegression(
        C=c, max_iter=max_iter, random_state=0, solver='lbfgs'
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always', ConvergenceWarning)
        estimator.fit(standardized, labels)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError(
            f'logistic regression did not converge within {max_iter} iterations '
            f'with {len(feature_indices)} features and C={c:g}'
        )
    importances = np.linalg.norm(estimator.coef_, axis=0)
    if (
        importances.shape != (len(feature_indices),)
        or not np.isfinite(importances).all()
    ):
        raise ValueError('invalid standardized coefficient importances')
    return FittedModel(scaler, estimator, feature_indices, importances)


def score(model: FittedModel, features: np.ndarray, labels: np.ndarray) -> float:
    """Return classification accuracy for a fitted model."""
    standardized = model.scaler.transform(features[:, model.feature_indices])
    return float(model.estimator.score(standardized, labels))


def select_regularization(
    train: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    feature_indices: np.ndarray,
    c_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[FittedModel, list[dict]]:
    """Select C for one feature set, preferring stronger regularization on ties."""
    if not c_grid or any(not np.isfinite(c) or c <= 0 for c in c_grid):
        raise ValueError('C grid must contain positive finite values')
    if tuple(sorted(set(c_grid))) != c_grid:
        raise ValueError(
            'C grid must be unique and sorted from strongest regularization'
        )
    rows = []
    best_row = None
    best_model = None
    for c in c_grid:
        model = fit_classifier(
            train['features'], train['labels'], feature_indices, c, max_iter
        )
        row = {
            'C': c,
            'features_kept': len(feature_indices),
            'validation_accuracy': score(
                model, validation['features'], validation['labels']
            ),
            'max_iterations_used': int(model.estimator.n_iter_.max()),
        }
        rows.append(row)
        if (
            best_row is None
            or row['validation_accuracy'] > best_row['validation_accuracy']
        ):
            best_row = row
            best_model = model
    for row in rows:
        row['selected'] = row is best_row
    return best_model, rows


def importance_rows(
    model: FittedModel, names: list[str], families: list[str], class_names: list[str]
) -> list[dict]:
    """Describe every full-model standardized coefficient and aggregate importance."""
    coefficients = model.estimator.coef_
    order = np.lexsort((model.feature_indices, -model.importances))
    rows = []
    for rank, position in enumerate(order, 1):
        feature_index = int(model.feature_indices[position])
        values = coefficients[:, position]
        row = {
            'importance_rank': rank,
            'pool_index': feature_index,
            'feature_name': names[feature_index],
            'family': families[feature_index],
            'l2_importance': float(model.importances[position]),
            'mean_absolute_coefficient': float(np.abs(values).mean()),
            'maximum_absolute_coefficient': float(np.abs(values).max()),
        }
        for label, value in zip(model.estimator.classes_, values, strict=True):
            row[f'coefficient_{class_names[int(label)]}'] = float(value)
        rows.append(row)
    return rows


def recursive_elimination(
    datasets: dict[str, dict[str, np.ndarray]],
    names: list[str],
    families: list[str],
    class_names: list[str],
    c_grid: tuple[float, ...],
    step: int,
    minimum_features: int,
    max_iter: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    """Refit and remove the five least-important standardized coefficients per step."""
    feature_count = datasets['train']['features'].shape[1]
    if (
        step < 1
        or minimum_features < 1
        or minimum_features > feature_count
        or (feature_count - minimum_features) % step
    ):
        raise ValueError(
            'step must reach minimum_features using exact equal-sized removals'
        )
    active = np.arange(feature_count)
    scores = []
    c_sweeps = []
    removals = []
    full_importances = None
    initial_rank = {}
    initial_importance = {}
    round_number = 0
    while True:
        model, sweep = select_regularization(
            datasets['train'], datasets['val'], active, c_grid, max_iter
        )
        for candidate in sweep:
            c_sweeps.append(
                {
                    'elimination_round': round_number,
                    'features_kept': len(active),
                    **candidate,
                }
            )
        selected_c = float(model.estimator.C)
        row = {
            'elimination_round': round_number,
            'features_kept': len(active),
            'reference_class_parameters': (len(model.estimator.classes_) - 1)
            * (len(active) + 1),
            'C': selected_c,
            'train_accuracy': score(
                model, datasets['train']['features'], datasets['train']['labels']
            ),
            'validation_accuracy': score(
                model, datasets['val']['features'], datasets['val']['labels']
            ),
            'test_accuracy': score(
                model, datasets['test']['features'], datasets['test']['labels']
            ),
            'max_iterations_used': int(model.estimator.n_iter_.max()),
        }
        scores.append(row)
        print(
            f'k={len(active):3d} C={selected_c:g}: '
            f'train={row["train_accuracy"]:.6f} '
            f'validation={row["validation_accuracy"]:.6f} '
            f'test={row["test_accuracy"]:.6f}',
            flush=True,
        )
        if full_importances is None:
            full_importances = importance_rows(model, names, families, class_names)
            initial_rank = {
                int(item['pool_index']): int(item['importance_rank'])
                for item in full_importances
            }
            initial_importance = {
                int(item['pool_index']): float(item['l2_importance'])
                for item in full_importances
            }
        if len(active) == minimum_features:
            for position, feature_index in enumerate(active):
                removals.append(
                    {
                        'removal_round': '',
                        'features_before_removal': len(active),
                        'features_after_removal': len(active),
                        'within_round_least_importance_rank': '',
                        'pool_index': int(feature_index),
                        'feature_name': names[feature_index],
                        'family': families[feature_index],
                        'importance_at_removal': '',
                        'initial_importance_rank': initial_rank[int(feature_index)],
                        'initial_l2_importance': initial_importance[int(feature_index)],
                        'final_kept': True,
                        'final_l2_importance': float(model.importances[position]),
                    }
                )
            break
        drop_positions = np.lexsort((active, model.importances))[:step]
        for rank, position in enumerate(drop_positions, 1):
            feature_index = int(active[position])
            removals.append(
                {
                    'removal_round': round_number + 1,
                    'features_before_removal': len(active),
                    'features_after_removal': len(active) - step,
                    'within_round_least_importance_rank': rank,
                    'pool_index': feature_index,
                    'feature_name': names[feature_index],
                    'family': families[feature_index],
                    'importance_at_removal': float(model.importances[position]),
                    'initial_importance_rank': initial_rank[feature_index],
                    'initial_l2_importance': initial_importance[feature_index],
                    'final_kept': False,
                    'final_l2_importance': '',
                }
            )
        active = np.delete(active, drop_positions)
        round_number += 1
    return scores, full_importances, removals, c_sweeps


def frontier33_indices(names: list[str]) -> np.ndarray:
    """Map the exact published 33-feature set into the 389-column pool."""
    matches = [
        index for index, name in enumerate(names) if name == 'tail_aniso_low_ndvi'
    ]
    if len(matches) != 1 or matches[0] < 377:
        raise ValueError(
            'full pool does not contain one region-shape tail_aniso_low_ndvi'
        )
    indices = np.append(POOL_INDICES, matches[0]).astype(np.int64)
    if len(indices) != 33 or len(set(indices.tolist())) != 33:
        raise ValueError('invalid published 33-feature mapping')
    return indices


def evaluate_feature_set(
    datasets: dict[str, dict[str, np.ndarray]],
    names: list[str],
    families: list[str],
    feature_indices: np.ndarray,
    c_grid: tuple[float, ...],
    max_iter: int,
) -> tuple[dict, list[dict], list[dict]]:
    """Tune and evaluate one fixed feature set under the curve's protocol."""
    model, sweep = select_regularization(
        datasets['train'], datasets['val'], feature_indices, c_grid, max_iter
    )
    result = {
        'features_kept': len(feature_indices),
        'reference_class_parameters': (len(model.estimator.classes_) - 1)
        * (len(feature_indices) + 1),
        'C': float(model.estimator.C),
        'train_accuracy': score(
            model, datasets['train']['features'], datasets['train']['labels']
        ),
        'validation_accuracy': score(
            model, datasets['val']['features'], datasets['val']['labels']
        ),
        'test_accuracy': score(
            model, datasets['test']['features'], datasets['test']['labels']
        ),
        'max_iterations_used': int(model.estimator.n_iter_.max()),
    }
    feature_rows = [
        {
            'position': position,
            'full_pool_index': int(index),
            'feature_name': names[index],
            'family': families[index],
        }
        for position, index in enumerate(feature_indices)
    ]
    return result, sweep, feature_rows


def plot_scores(path: Path, rows: list[dict], frontier33: dict | None = None) -> None:
    """Plot held-out accuracy against the number of retained features."""
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    ordered = sorted(rows, key=lambda row: row['features_kept'])
    counts = [row['features_kept'] for row in ordered]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        counts,
        [100 * row['validation_accuracy'] for row in ordered],
        marker='o',
        markersize=2.5,
        linewidth=1.4,
        label='validation',
    )
    axis.plot(
        counts,
        [100 * row['test_accuracy'] for row in ordered],
        marker='o',
        markersize=2.5,
        linewidth=1.4,
        label='test',
    )
    if frontier33 is not None:
        axis.plot(
            frontier33['features_kept'],
            100 * frontier33['validation_accuracy'],
            marker='*',
            markeredgecolor='black',
            markersize=12,
            linestyle='none',
            color='tab:blue',
            label='published 33-feature set (validation)',
        )
        axis.plot(
            frontier33['features_kept'],
            100 * frontier33['test_accuracy'],
            marker='*',
            markeredgecolor='black',
            markersize=12,
            linestyle='none',
            color='tab:orange',
            label='published 33-feature set (test)',
        )
    axis.set(
        xlabel='Features kept',
        ylabel='Accuracy (%)',
        title='Recursive feature elimination (C tuned at each step)',
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_importances(path: Path, rows: list[dict], limit: int = 30) -> None:
    """Plot the strongest full-model standardized coefficient norms."""
    import matplotlib

    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    selected = list(reversed(rows[:limit]))
    labels = [
        f'{row["pool_index"]}: {row["feature_name"]} [{row["family"]}]'
        for row in selected
    ]
    figure, axis = plt.subplots(figsize=(9, 8))
    axis.barh(labels, [row['l2_importance'] for row in selected])
    axis.set(
        xlabel='L2 norm across standardized class coefficients',
        title=f'Top {len(selected)} full-pool feature importances',
    )
    axis.grid(axis='x', alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_experiment(
    datasets: dict[str, dict[str, np.ndarray]],
    schema: dict,
    output: Path,
    c_grid: tuple[float, ...],
    step: int,
    minimum_features: int,
    max_iter: int,
    reference_indices: np.ndarray,
) -> None:
    """Select regularization, run elimination, and write tables and plots."""
    scores, importances, removals, sweeps = recursive_elimination(
        datasets,
        schema['names'],
        schema['families'],
        CLASSES,
        c_grid,
        step,
        minimum_features,
        max_iter,
    )
    frontier33, frontier_sweep, frontier_features = evaluate_feature_set(
        datasets,
        schema['names'],
        schema['families'],
        reference_indices,
        c_grid,
        max_iter,
    )
    write_csv(output / 'c_sweep.csv', sweeps)
    write_csv(output / 'feature_importances.csv', importances)
    write_csv(output / 'elimination_order.csv', removals)
    write_csv(output / 'scores.csv', scores)
    write_csv(output / 'frontier33.csv', [frontier33])
    write_csv(output / 'frontier33_c_sweep.csv', frontier_sweep)
    write_csv(output / 'frontier33_features.csv', frontier_features)
    plot_scores(output / 'score_by_features.png', scores, frontier33)
    plot_importances(output / 'top_feature_importances.png', importances)
    best = max(
        scores, key=lambda row: (row['validation_accuracy'], -row['features_kept'])
    )
    summary = {
        'regularization_selection': (
            'validation-tuned independently at every retained-feature count'
        ),
        'C_grid': c_grid,
        'importance_definition': (
            'L2 norm across class coefficients after train-fitted per-feature standardization'
        ),
        'removal_rule': f'refit and remove the {step} least-important active features',
        'frontier33_refit': frontier33,
        'full_pool': scores[0],
        'best_validation': best,
        'final': scores[-1],
    }
    write_json(output / 'summary.json', summary)


def main() -> None:
    """Run the command-line experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument(
        '--cache', type=Path, default=ROOT / 'output/feature-importance-cache-v2'
    )
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'output/feature-importance'
    )
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--step', type=int, default=5)
    parser.add_argument('--minimum-features', type=int, default=4)
    parser.add_argument('--max-iter', type=int, default=4000)
    parser.add_argument('--c-grid', type=float, nargs='+', default=DEFAULT_CS)
    args = parser.parse_args()

    cache = args.cache.resolve()
    output = args.output.resolve()
    output_root = (ROOT / 'output').resolve()
    for argument, path in (('--cache', cache), ('--output', output)):
        if path == output_root or not path.is_relative_to(output_root):
            parser.error(f'{argument} must be a subdirectory of output/')
    if output.exists() and any(output.iterdir()):
        parser.error('--output must be empty; choose a new results directory')
    if (
        args.step < 1
        or args.minimum_features < 1
        or (FULL_POOL_SIZE - args.minimum_features) % args.step
    ):
        parser.error(
            '--step must reach --minimum-features using exact equal-sized removals'
        )
    if args.batch_size < 1 or args.max_iter < 1:
        parser.error('--batch-size and --max-iter must be positive')
    c_grid = tuple(args.c_grid)
    if tuple(sorted(set(c_grid))) != c_grid or any(c <= 0 for c in c_grid):
        parser.error('--c-grid must be positive, unique, and sorted ascending')

    manifest = prepare_cache(cache, args.batch_size, args.download)
    datasets, manifest = load_splits(cache)
    reference_indices = frontier33_indices(manifest['schema']['names'])
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / 'provenance.json',
        {
            'python': platform.python_version(),
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'scikit_learn': sklearn.__version__,
            'blas_threads': 1,
            'feature_cache': str(cache.relative_to(ROOT)),
            'cache_manifest_sha256': sha256(cache / 'manifest.json'),
            'cache_identity': manifest['identity'],
            'feature_schema': manifest['schema'],
            'C_grid': c_grid,
            'regularization_selection': (
                'validation-tuned independently at every retained-feature count'
            ),
            'elimination_step': args.step,
            'minimum_features': args.minimum_features,
            'max_iter': args.max_iter,
            'frontier33_full_pool_indices': reference_indices.tolist(),
            'test_use': 'reported at every fixed elimination step; never used for selection',
        },
    )
    run_experiment(
        datasets,
        manifest['schema'],
        output,
        c_grid,
        args.step,
        args.minimum_features,
        args.max_iter,
        reference_indices,
    )


if __name__ == '__main__':
    main()
