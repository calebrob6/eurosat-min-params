#!/usr/bin/env python
"""Evaluate the fixed 306-parameter model across data fractions and splits.

Runs five deterministic stratified subsamples for each requested fraction on
both the default random EuroSAT split and TorchGeo's longitude-based spatial
split. The feature subset, C, validation split, and test split are fixed; only
the folded affine head is refit. The output CSV stores every seed accuracy plus
their mean and sample standard deviation.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np
from sklearn.metrics import accuracy_score

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from eval_spatial_fractions import (  # noqa: E402
    DEFAULT_FRACTIONS,
    ensure_spatial_splits,
    load_source_features,
    read_split,
    subset_indices,
)
from region_shape_prune_ceiling96_33 import (  # noqa: E402
    C,
    FINAL_POOL_IDX,
    REGION_SHAPE_NAME,
)
from src.data import DATA_ROOT  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402

DEFAULT_SEEDS = tuple(range(5))
EXPECTED_SPLIT_SIZES = {'train': 16200, 'val': 5400, 'test': 5400}
MODEL_ID = 'reference_class_logreg_c3_306'
MODEL_TYPE = 'folded reference-class multinomial logistic regression'
FEATURE_SET_ID = 'region_shape_prune_ceiling96_33'
FEATURE_COMBINATION = (
    '32 selected features from the 377-feature mega-pool '
    f'plus {REGION_SHAPE_NAME}'
)
FEATURE_INDICES = ';'.join(
    [*(f'pool:{index}' for index in FINAL_POOL_IDX), f'region_shape:{REGION_SHAPE_NAME}']
)


def remap_splits(
    split_paths: dict[str, Path],
    source_names: list[str],
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Map full-dataset cached features into a named split protocol."""
    names_by_split = {
        split: read_split(path) for split, path in split_paths.items()
    }
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        actual_size = len(names_by_split[split])
        if actual_size != expected_size:
            raise ValueError(
                f'{split_paths[split]} contains {actual_size} samples, '
                f'expected {expected_size}'
            )

    name_sets = {split: set(names) for split, names in names_by_split.items()}
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        overlap = name_sets[left] & name_sets[right]
        if overlap:
            raise ValueError(
                f'{left}/{right} splits overlap by {len(overlap)} samples'
            )
    protocol_names = set().union(*name_sets.values())
    if protocol_names != set(source_names):
        raise ValueError('split protocol does not cover the source sample universe')

    source_index = {name: index for index, name in enumerate(source_names)}
    remapped: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, names in names_by_split.items():
        indices = np.fromiter(
            (source_index[name] for name in names), dtype=np.int64, count=len(names)
        )
        remapped[split] = (features[indices], labels[indices])
    return remapped


def evaluate_protocol(
    protocol: str,
    split_description: str,
    split_data: dict[str, tuple[np.ndarray, np.ndarray]],
    fractions: list[float],
    seeds: list[int],
) -> list[dict[str, str | int | float]]:
    """Evaluate one split protocol and return one CSV row per fraction."""
    x_train, y_train = split_data['train']
    x_test, y_test = split_data['test']
    rows: list[dict[str, str | int | float]] = []

    for fraction in fractions:
        accuracies: list[float] = []
        n_train: int | None = None
        for seed in seeds:
            indices = subset_indices(y_train, fraction, seed)
            n_train = len(indices)
            w, b, feature_idx = fit_folded_logreg(
                x_train[indices], y_train[indices], C=C
            )
            predictions = predict(x_test, w, b, feature_idx)
            accuracies.append(accuracy_score(y_test, predictions))

        accuracy_array = np.asarray(accuracies)
        rows.append({
            'model_id': MODEL_ID,
            'model_type': MODEL_TYPE,
            'feature_set_id': FEATURE_SET_ID,
            'feature_combination': FEATURE_COMBINATION,
            'feature_indices': FEATURE_INDICES,
            'num_features': features_per_model(),
            'learned_parameters': 9 * (features_per_model() + 1),
            'regularization_C': C,
            'split_protocol': protocol,
            'split_description': split_description,
            'train_fraction_percent': fraction,
            'n_train': int(n_train),
            'n_test': len(y_test),
            'seeds': ';'.join(map(str, seeds)),
            **{
                f'seed_{seed}_test_accuracy': f'{accuracy:.6f}'
                for seed, accuracy in zip(seeds, accuracies, strict=True)
            },
            'test_accuracy_mean': f'{accuracy_array.mean():.6f}',
            'test_accuracy_stdev': f'{accuracy_array.std(ddof=1):.6f}',
        })
    return rows


def features_per_model() -> int:
    """Return the exact number of fixed features consumed by the head."""
    return len(FINAL_POOL_IDX) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fractions',
        type=float,
        nargs='+',
        default=list(DEFAULT_FRACTIONS),
        help='percentages of each 16,200-image training split',
    )
    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=list(DEFAULT_SEEDS),
        help='stratified-subsample seeds',
    )
    parser.add_argument(
        '--download-spatial-splits',
        action='store_true',
        help='download missing official TorchGeo spatial split files',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'eval_training_fractions_result.csv'
        ),
    )
    args = parser.parse_args()

    fractions = sorted(set(args.fractions))
    if not fractions or any(value <= 0 or value > 100 for value in fractions):
        parser.error('--fractions must contain values in (0, 100]')
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must contain at least two unique integers')

    source_names, features, labels = load_source_features()
    if features.shape[1] != features_per_model():
        raise ValueError(
            f'loaded {features.shape[1]} features, expected {features_per_model()}'
        )

    random_paths = {
        split: Path(DATA_ROOT) / f'eurosat-{split}.txt'
        for split in EXPECTED_SPLIT_SIZES
    }
    spatial_paths = ensure_spatial_splits(args.download_spatial_splits)
    protocols = [
        (
            'random',
            'default EuroSAT random 60/20/20 split',
            remap_splits(random_paths, source_names, features, labels),
        ),
        (
            'spatial',
            'TorchGeo longitude-based EuroSATSpatial 60/20/20 split',
            remap_splits(spatial_paths, source_names, features, labels),
        ),
    ]

    rows: list[dict[str, str | int | float]] = []
    for protocol, description, split_data in protocols:
        rows.extend(
            evaluate_protocol(
                protocol,
                description,
                split_data,
                fractions,
                args.seeds,
            )
        )

    fieldnames = list(rows[0])
    with Path(args.output).open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        'split fraction n_train test_accuracy_mean test_accuracy_stdev',
        flush=True,
    )
    for row in rows:
        print(
            f'{row["split_protocol"]:7s} '
            f'{float(row["train_fraction_percent"]):7g}% '
            f'{int(row["n_train"]):7d} '
            f'{row["test_accuracy_mean"]} '
            f'{row["test_accuracy_stdev"]}',
            flush=True,
        )


if __name__ == '__main__':
    main()
