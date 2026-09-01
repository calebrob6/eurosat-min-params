#!/usr/bin/env python
"""Evaluate the 52-feature ImageStats baseline across data fractions.

Uses the baseline's validation-selected C=300 and refits the folded logistic
head on stratified 1%, 2%, 5%, 10%, 20%, 50%, and 100% training subsets of
both the default random split and TorchGeo's longitude-based spatial split.
The CSV stores five seed accuracies plus their mean and sample standard
deviation for each fraction and split protocol.
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
    ensure_spatial_splits,
    labels_from_names,
    read_split,
    subset_indices,
)
from eval_training_fractions import (  # noqa: E402
    DEFAULT_FRACTIONS,
    DEFAULT_SEEDS,
    EXPECTED_SPLIT_SIZES,
    remap_splits,
)
from image_statistics_baseline import image_statistics  # noqa: E402
from src.data import DATA_ROOT  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402

C = 300.0
NUM_FEATURES = 52
LEARNED_PARAMETERS = 9 * (NUM_FEATURES + 1)
MODEL_ID = 'imagestats_reference_class_logreg_c300'
MODEL_TYPE = 'folded reference-class multinomial logistic regression'
FEATURE_SET_ID = 'per_band_mean_std_min_max'
FEATURE_COMBINATION = '13 bands x mean, standard deviation, minimum, maximum'


def load_source_features(
    batch_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Load ImageStats features in the default split-file order."""
    names: list[str] = []
    feature_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for split, expected_size in EXPECTED_SPLIT_SIZES.items():
        split_names = read_split(Path(DATA_ROOT) / f'eurosat-{split}.txt')
        features, labels = image_statistics(split, batch_size)
        if not (len(split_names) == len(features) == len(labels) == expected_size):
            raise ValueError(f'{split} ImageStats rows do not match split file')
        if not np.array_equal(labels, labels_from_names(split_names)):
            raise ValueError(f'{split} labels do not match split filenames')
        names.extend(split_names)
        feature_parts.append(features)
        label_parts.append(labels)
    if len(names) != len(set(names)):
        raise ValueError('default train/validation/test split files overlap')
    all_features = np.concatenate(feature_parts).astype(np.float32)
    if all_features.shape != (27000, NUM_FEATURES):
        raise ValueError(
            f'expected a (27000, {NUM_FEATURES}) feature matrix, '
            f'got {all_features.shape}'
        )
    return names, all_features, np.concatenate(label_parts)


def evaluate_protocol(
    protocol: str,
    split_description: str,
    split_data: dict[str, tuple[np.ndarray, np.ndarray]],
    fractions: list[float],
    seeds: list[int],
) -> list[dict[str, str | int | float]]:
    """Evaluate one split protocol and return one row per fraction."""
    x_train, y_train = split_data['train']
    x_test, y_test = split_data['test']
    rows: list[dict[str, str | int | float]] = []
    for fraction in fractions:
        accuracies: list[float] = []
        n_train = 0
        for seed in seeds:
            indices = subset_indices(y_train, fraction, seed)
            n_train = len(indices)
            w, b, feature_idx = fit_folded_logreg(
                x_train[indices], y_train[indices], C=C
            )
            accuracies.append(
                accuracy_score(y_test, predict(x_test, w, b, feature_idx))
            )
        accuracy_array = np.asarray(accuracies)
        rows.append({
            'model_id': MODEL_ID,
            'model_type': MODEL_TYPE,
            'feature_set_id': FEATURE_SET_ID,
            'feature_combination': FEATURE_COMBINATION,
            'num_features': NUM_FEATURES,
            'learned_parameters': LEARNED_PARAMETERS,
            'regularization_C': C,
            'split_protocol': protocol,
            'split_description': split_description,
            'train_fraction_percent': fraction,
            'n_train': n_train,
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
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument(
        '--download-spatial-splits',
        action='store_true',
        help='download missing official TorchGeo spatial split files',
    )
    parser.add_argument(
        '--output',
        default=os.path.join(
            os.path.dirname(__file__), 'eval_imagestats_fractions_result.csv'
        ),
    )
    args = parser.parse_args()

    fractions = sorted(set(args.fractions))
    if not fractions or any(value <= 0 or value > 100 for value in fractions):
        parser.error('--fractions must contain values in (0, 100]')
    if len(args.seeds) < 2 or len(args.seeds) != len(set(args.seeds)):
        parser.error('--seeds must contain at least two unique integers')

    source_names, features, labels = load_source_features(args.batch_size)
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
                protocol, description, split_data, fractions, args.seeds
            )
        )

    with Path(args.output).open('w', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print('split fraction n_train test_accuracy_mean test_accuracy_stdev')
    for row in rows:
        print(
            f'{row["split_protocol"]:7s} '
            f'{float(row["train_fraction_percent"]):7g}% '
            f'{int(row["n_train"]):7d} '
            f'{row["test_accuracy_mean"]} '
            f'{row["test_accuracy_stdev"]}'
        )


if __name__ == '__main__':
    main()
