#!/usr/bin/env python
"""Evaluate persisted RESISC45 shared-value codebook models."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
from pathlib import Path

import numpy as np

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_codebook import (  # noqa: E402
    BLOCK_POOLS,
    CANDIDATE_MAPPING_BITS,
    CLASS_ATOMS,
    SCALE_EXPONENT_ORIGIN_BITS,
    accuracy,
    gaussian_atoms,
    standard_weights,
    suffixed,
)
from resisc45_layout_gain import load_pools  # noqa: E402

DEFAULT_DIRECTORY = os.path.join(
    EXPERIMENTS_DIR, 'resisc45_codebook_models'
)
DEFAULT_OUTPUT = os.path.join(
    EXPERIMENTS_DIR, 'resisc45_codebook_models_result.csv'
)
REPOSITORY_ROOT = Path(EXPERIMENTS_DIR).parent


def file_hash(path: Path) -> str:
    """Return an artifact's SHA-256."""
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_features(
    artifact: np.lib.npyio.NpzFile,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Reconstruct the learned 768-column candidate-list mapping."""
    suffix = str(artifact['pool_suffix'])
    mapping = artifact['candidate_mapping']
    offsets = artifact['candidate_mapping_offsets']
    quota = artifact['quota']
    if len(BLOCK_POOLS) != len(quota) or len(offsets) != len(quota) + 1:
        raise ValueError('candidate mapping metadata is inconsistent')

    parts: dict[str, list[np.ndarray]] = {
        split: [] for split in ('train', 'val', 'test')
    }
    labels = None
    for block_index, pools in enumerate(BLOCK_POOLS):
        block, labels, _ = load_pools(
            tuple(suffixed(pool, suffix) for pool in pools)
        )
        selected = mapping[offsets[block_index]:offsets[block_index + 1]]
        if len(selected) != int(quota[block_index]):
            raise ValueError('candidate mapping does not match its quota')
        for split in parts:
            parts[split].append(block[split][:, selected])
    features = {
        split: np.concatenate(blocks, axis=1).astype(np.float32)
        for split, blocks in parts.items()
    }
    return features, labels


def evaluate(path: Path) -> dict[str, object]:
    """Reconstruct one model and evaluate validation/test accuracy."""
    with np.load(path) as artifact:
        features, labels = load_candidate_features(artifact)
        if str(artifact['class_dictionary']) != 'gaussian':
            raise ValueError('unsupported class dictionary')
        if int(artifact['reference_class']) != 0:
            raise ValueError('only reference class 0 is supported')
        class_dictionary = gaussian_atoms(
            int(artifact['class_dictionary_atoms']),
            seed=int(artifact['class_dictionary_seed']),
        )
        support = artifact['support'].astype(np.int64)
        assignment = artifact['level_assignment'].astype(np.int64)
        levels = artifact['feature_levels'].astype(np.float64)
        sigma_exponent = artifact['sigma_exponent'].astype(np.int64)
        sigma = np.exp2(sigma_exponent.astype(np.float64))
        columns = features['train'].shape[1]
        weights = standard_weights(
            class_dictionary,
            support,
            levels[assignment],
            columns,
        ) / sigma[None, :]
        bias_atoms = artifact['bias_atoms'].astype(np.int64)
        bias_values = artifact['raw_bias_values'].astype(np.float64)
        if len(bias_atoms):
            bias = class_dictionary[:, bias_atoms] @ bias_values
            bias_structure_bits = len(bias_atoms) * math.ceil(
                math.log2(CLASS_ATOMS)
            )
            bias_kind = 'class_dictionary'
        else:
            bias = bias_values
            bias_structure_bits = 0
            bias_kind = 'full'

        learned_values = len(levels) + len(bias_values)
        level_bits = math.ceil(math.log2(len(levels)))
        atom_bits = math.ceil(math.log2(CLASS_ATOMS))
        column_bits = math.ceil(math.log2(columns))
        touched = np.unique(support % columns)
        exponent_range = int(
            sigma_exponent[touched].max()
            - sigma_exponent[touched].min()
            + 1
        )
        exponent_bits = max(1, math.ceil(math.log2(exponent_range)))
        structure_bits = len(support) * (
            atom_bits + column_bits + level_bits
        )
        structure_bits += len(touched) * exponent_bits
        structure_bits += SCALE_EXPONENT_ORIGIN_BITS
        structure_bits += CANDIDATE_MAPPING_BITS + bias_structure_bits
        learned_bits = learned_values * 32

        return {
            'artifact': str(path.relative_to(REPOSITORY_ROOT)),
            'artifact_sha256': file_hash(path),
            'support': len(support),
            'feature_levels': len(levels),
            'bias_kind': bias_kind,
            'bias_values': len(bias_values),
            'learned_values': learned_values,
            'structure_bits': structure_bits,
            'learned_value_bits': learned_bits,
            'total_model_bits': structure_bits + learned_bits,
            'physical_file_bytes': path.stat().st_size,
            'C': f'{float(artifact["C"]):g}',
            'validation_accuracy': f'{accuracy(features["val"], labels["val"], weights, bias):.6f}',
            'test_accuracy': f'{accuracy(features["test"], labels["test"], weights, bias):.6f}',
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default=DEFAULT_DIRECTORY)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = sorted(Path(args.directory).glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'no model artifacts in {args.directory}')
    rows = [evaluate(path) for path in paths]
    with open(args.output, 'w', newline='') as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{Path(str(row['artifact'])).name}: "
            f"{row['learned_values']} values, "
            f"val={row['validation_accuracy']} test={row['test_accuracy']}"
        )


if __name__ == '__main__':
    main()
