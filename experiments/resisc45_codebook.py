#!/usr/bin/env python
"""Share sparse RESISC45 code values through a learned scalar codebook.

The existing separable-dictionary head stores one floating-point value per
active class-atom/feature entry. Here a much larger fixed support reuses Q
learned scalar levels. Support locations and level assignments are discrete
learned structure and are reported in bits, while the learned-value count is
Q levels plus 44 folded reference-class intercepts.

Feature standard deviations are rounded to powers of two. Their exponents are
stored as discrete ids, allowing raw-weight reconstruction without introducing
one learned floating-point scale per feature.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import os
import sys
import time

import numpy as np
import torch

EXPERIMENTS_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, EXPERIMENTS_DIR)
sys.path.insert(0, os.path.abspath(os.path.join(EXPERIMENTS_DIR, '..')))

from resisc45_expanded_head import gaussian_atoms  # noqa: E402
from resisc45_layout_gain import (  # noqa: E402
    BASE_POOLS,
    LAM,
    LAYOUT_POOL,
    load_pools,
)
from resisc45_lib import (  # noqa: E402
    _to_device,
    group_lasso_rank,
    standardise,
)
from resisc45_union_variants import prune, union_support  # noqa: E402

RESULT_PATH = os.path.join(EXPERIMENTS_DIR, 'resisc45_codebook_result.csv')
POOL_SUFFIX = 'j32crop'
RANK_SUFFIX = 'd8'
BLOCK_POOLS = (
    ('gpu_pool', 'gpu2_pool', 'rgb_pool'),
    ('gpu3_pool',),
    ('gpu4_pool',),
    ('gpu5_pool',),
)
QUOTA = (384, 128, 128, 128)
CLASS_ATOMS = 16384
ROWS = 44
C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1)
LEVELS = (4, 8, 16, 32, 64, 128, 256)
SEARCHES = 8
DECAY = 1.5e-3
DROP = 0.3
# Four learned quota mappings: 384/1,579, 128/505, 128/2,042, 128/142.
CANDIDATE_MAPPING_BITS = 384 * 11 + 128 * 9 + 128 * 11 + 128 * 8
SCALE_EXPONENT_ORIGIN_BITS = 16


def suffixed(pool: str, suffix: str) -> str:
    """Insert a cached view suffix into a pool stem."""
    if pool == 'rgb_pool':
        return pool
    return pool.replace('_pool', f'{suffix}_pool')


def load_list() -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    tuple[np.ndarray, ...],
]:
    """Build the j32crop values on the validation-winning d8 column list."""
    value_blocks = []
    rank_blocks = []
    labels = None
    for pools in BLOCK_POOLS:
        values, labels, _ = load_pools(
            tuple(suffixed(pool, POOL_SUFFIX) for pool in pools)
        )
        ranking, _, _ = load_pools(
            tuple(suffixed(pool, RANK_SUFFIX) for pool in pools)
        )
        value_blocks.append(values)
        rank_blocks.append(ranking)
    orders = [
        group_lasso_rank(
            block['train'], labels['train'], lam=LAM, epochs=1500
        )[0]
        for block in rank_blocks
    ]
    mappings = tuple(
        order[:quota].astype(np.int32)
        for order, quota in zip(orders, QUOTA, strict=True)
    )
    features = {
        split: np.concatenate(
            [
                block[split][:, order[:quota]]
                for block, order, quota in zip(
                    value_blocks, orders, QUOTA, strict=True
                )
            ],
            axis=1,
        ).astype(np.float32)
        for split in labels
    }
    return features, labels, mappings


def rounded_moments(
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, power-of-two sigma, and integer sigma exponents."""
    mean, sigma = standardise(train)
    exponent = np.round(np.log2(sigma)).astype(np.int16)
    rounded_sigma = np.exp2(exponent.astype(np.float64))
    return mean, rounded_sigma, exponent


def array_hash(array: np.ndarray) -> str:
    """Stable SHA-256 for a discrete model-structure array."""
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def save_model_artifact(
    directory: str,
    tag: str,
    feature_levels: np.ndarray,
    raw_bias_values: np.ndarray,
    support: np.ndarray,
    level_assignment: np.ndarray,
    bias_atoms: np.ndarray,
    sigma_exponent: np.ndarray,
    mappings: tuple[np.ndarray, ...],
    C: float,
) -> tuple[str, str]:
    """Persist every value and discrete array needed to reconstruct a head."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f'{tag}.npz')
    mapping_offsets = np.cumsum((0, *(len(mapping) for mapping in mappings)))
    np.savez_compressed(
        path,
        feature_levels=np.asarray(feature_levels, dtype=np.float32),
        raw_bias_values=np.asarray(raw_bias_values, dtype=np.float32),
        support=np.asarray(support, dtype=np.int32),
        level_assignment=np.asarray(level_assignment, dtype=np.uint16),
        bias_atoms=np.asarray(bias_atoms, dtype=np.uint16),
        sigma_exponent=np.asarray(sigma_exponent, dtype=np.int16),
        candidate_mapping=np.concatenate(mappings).astype(np.int32),
        candidate_mapping_offsets=mapping_offsets.astype(np.int16),
        quota=np.asarray(QUOTA, dtype=np.int16),
        source_blocks=np.asarray(
            ('base', 'layout', 'pool4', 'pool5'), dtype='U8'
        ),
        pool_suffix=np.asarray(POOL_SUFFIX),
        ranking_suffix=np.asarray(RANK_SUFFIX),
        class_dictionary=np.asarray('gaussian'),
        class_dictionary_seed=np.int16(0),
        class_dictionary_atoms=np.int32(CLASS_ATOMS),
        reference_class=np.int16(0),
        C=np.float32(C),
    )
    with open(path, 'rb') as source:
        digest = hashlib.sha256(source.read()).hexdigest()
    return path, digest


def fit_continuous(
    train: np.ndarray,
    labels: np.ndarray,
    class_dictionary: np.ndarray,
    support: np.ndarray,
    C: float,
    mean: np.ndarray,
    sigma: np.ndarray,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convexly refit signed support values and an explicit intercept."""
    standardized = _to_device((train - mean) / sigma, device)
    targets = _to_device(labels, device, torch.long)
    dictionary = _to_device(class_dictionary, device)
    columns = train.shape[1]
    index = torch.as_tensor(support, device=device, dtype=torch.long)
    class_vectors = dictionary[:, index // columns]
    feature_index = index % columns
    feature_vectors = torch.nn.functional.one_hot(
        feature_index, num_classes=columns
    ).to(standardized.dtype)
    values = torch.zeros(len(index), device=device, requires_grad=True)
    intercept = torch.zeros(ROWS, device=device, requires_grad=True)
    regularization = 1.0 / (2.0 * C * len(train))
    optimizer = torch.optim.LBFGS(
        [values, intercept],
        max_iter=steps,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        weights = (class_vectors * values[None, :]) @ feature_vectors
        logits = torch.cat(
            (
                torch.zeros(len(train), 1, device=device),
                standardized @ weights.T + intercept,
            ),
            dim=1,
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = loss + regularization * values.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return values.detach().cpu().numpy(), intercept.detach().cpu().numpy()


def scalar_codebook(values: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic one-dimensional Lloyd quantization."""
    if levels >= len(values):
        centers = np.zeros(levels, dtype=np.float64)
        centers[:len(values)] = values
        return centers, np.arange(len(values), dtype=np.int64)
    probabilities = (np.arange(levels) + 0.5) / levels
    centers = np.quantile(values, probabilities).astype(np.float64)
    assignment = np.zeros(len(values), dtype=np.int64)
    for _ in range(100):
        updated = np.abs(values[:, None] - centers[None, :]).argmin(1)
        if np.array_equal(updated, assignment):
            break
        assignment = updated
        for level in range(levels):
            members = values[assignment == level]
            if len(members):
                centers[level] = members.mean()
    return centers, assignment


def fit_codebook(
    train: np.ndarray,
    labels: np.ndarray,
    class_dictionary: np.ndarray,
    support: np.ndarray,
    assignment: np.ndarray,
    initial_levels: np.ndarray,
    C: float,
    mean: np.ndarray,
    sigma: np.ndarray,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convexly refit shared levels and the folded intercept."""
    standardized = _to_device((train - mean) / sigma, device)
    targets = _to_device(labels, device, torch.long)
    dictionary = _to_device(class_dictionary, device)
    columns = train.shape[1]
    index = torch.as_tensor(support, device=device, dtype=torch.long)
    class_vectors = dictionary[:, index // columns]
    feature_index = index % columns
    feature_vectors = torch.nn.functional.one_hot(
        feature_index, num_classes=columns
    ).to(standardized.dtype)
    code_index = torch.as_tensor(
        assignment, device=device, dtype=torch.long
    )
    levels = torch.as_tensor(
        initial_levels, device=device, dtype=torch.float32
    ).requires_grad_(True)
    intercept = torch.zeros(ROWS, device=device, requires_grad=True)
    regularization = 1.0 / (2.0 * C * len(train))
    optimizer = torch.optim.LBFGS(
        [levels, intercept],
        max_iter=steps,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        expanded = levels[code_index]
        weights = (class_vectors * expanded[None, :]) @ feature_vectors
        logits = torch.cat(
            (
                torch.zeros(len(train), 1, device=device),
                standardized @ weights.T + intercept,
            ),
            dim=1,
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = loss + regularization * expanded.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return levels.detach().cpu().numpy(), intercept.detach().cpu().numpy()


def standard_weights(
    class_dictionary: np.ndarray,
    support: np.ndarray,
    values: np.ndarray,
    columns: int,
) -> np.ndarray:
    """Reconstruct reference-class weights in standardized feature space."""
    class_vectors = class_dictionary[:, support // columns]
    feature_index = support % columns
    weights = np.zeros((ROWS, columns), dtype=np.float64)
    np.add.at(
        weights,
        (slice(None), feature_index),
        class_vectors * values[None, :],
    )
    return weights


def fit_bias_codebook(
    train: np.ndarray,
    labels: np.ndarray,
    class_dictionary: np.ndarray,
    support: np.ndarray,
    assignment: np.ndarray,
    initial_levels: np.ndarray,
    standardized_intercept: np.ndarray,
    bias_level_count: int,
    C: float,
    mean: np.ndarray,
    sigma: np.ndarray,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly refit feature levels and a codebook for the raw intercept."""
    columns = train.shape[1]
    initial_weights = standard_weights(
        class_dictionary,
        support,
        initial_levels[assignment],
        columns,
    )
    raw_intercept = standardized_intercept - (
        initial_weights * (mean / sigma)[None, :]
    ).sum(1)
    initial_bias_levels, bias_assignment = scalar_codebook(
        raw_intercept, bias_level_count
    )

    standardized = _to_device((train - mean) / sigma, device)
    targets = _to_device(labels, device, torch.long)
    dictionary = _to_device(class_dictionary, device)
    index = torch.as_tensor(support, device=device, dtype=torch.long)
    class_vectors = dictionary[:, index // columns]
    feature_index = index % columns
    feature_vectors = torch.nn.functional.one_hot(
        feature_index, num_classes=columns
    ).to(standardized.dtype)
    code_index = torch.as_tensor(
        assignment, device=device, dtype=torch.long
    )
    bias_index = torch.as_tensor(
        bias_assignment, device=device, dtype=torch.long
    )
    feature_levels = torch.as_tensor(
        initial_levels, device=device, dtype=torch.float32
    ).requires_grad_(True)
    bias_levels = torch.as_tensor(
        initial_bias_levels, device=device, dtype=torch.float32
    ).requires_grad_(True)
    mean_over_sigma = _to_device(mean / sigma, device)
    regularization = 1.0 / (2.0 * C * len(train))
    optimizer = torch.optim.LBFGS(
        [feature_levels, bias_levels],
        max_iter=steps,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        expanded = feature_levels[code_index]
        weights = (class_vectors * expanded[None, :]) @ feature_vectors
        raw_bias = bias_levels[bias_index]
        intercept = raw_bias + (weights * mean_over_sigma[None, :]).sum(1)
        logits = torch.cat(
            (
                torch.zeros(len(train), 1, device=device),
                standardized @ weights.T + intercept,
            ),
            dim=1,
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = loss + regularization * expanded.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        feature_levels.detach().cpu().numpy(),
        bias_levels.detach().cpu().numpy(),
        bias_assignment,
    )


def bias_dictionary_initialization(
    target: np.ndarray,
    class_dictionary: np.ndarray,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedily choose class atoms and least-squares fit their coefficients."""
    residual = np.asarray(target, dtype=np.float64)
    selected: list[int] = []
    coefficients = np.empty(0)
    for _ in range(count):
        projection = class_dictionary.T @ residual
        projection[selected] = 0.0
        selected.append(int(np.argmax(np.abs(projection))))
        atoms = class_dictionary[:, selected]
        coefficients = np.linalg.lstsq(atoms, target, rcond=None)[0]
        residual = target - atoms @ coefficients
    return np.asarray(selected, dtype=np.int64), coefficients


def fit_bias_dictionary(
    train: np.ndarray,
    labels: np.ndarray,
    class_dictionary: np.ndarray,
    support: np.ndarray,
    assignment: np.ndarray,
    initial_levels: np.ndarray,
    standardized_intercept: np.ndarray,
    bias_atom_count: int,
    C: float,
    mean: np.ndarray,
    sigma: np.ndarray,
    steps: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Jointly refit feature levels and sparse class-dictionary bias values."""
    columns = train.shape[1]
    initial_weights = standard_weights(
        class_dictionary,
        support,
        initial_levels[assignment],
        columns,
    )
    raw_intercept = standardized_intercept - (
        initial_weights * (mean / sigma)[None, :]
    ).sum(1)
    bias_atoms, initial_bias_values = bias_dictionary_initialization(
        raw_intercept, class_dictionary, bias_atom_count
    )

    standardized = _to_device((train - mean) / sigma, device)
    targets = _to_device(labels, device, torch.long)
    dictionary = _to_device(class_dictionary, device)
    index = torch.as_tensor(support, device=device, dtype=torch.long)
    class_vectors = dictionary[:, index // columns]
    feature_index = index % columns
    feature_vectors = torch.nn.functional.one_hot(
        feature_index, num_classes=columns
    ).to(standardized.dtype)
    code_index = torch.as_tensor(
        assignment, device=device, dtype=torch.long
    )
    bias_vectors = dictionary[:, torch.as_tensor(
        bias_atoms, device=device, dtype=torch.long
    )]
    feature_levels = torch.as_tensor(
        initial_levels, device=device, dtype=torch.float32
    ).requires_grad_(True)
    bias_values = torch.as_tensor(
        initial_bias_values, device=device, dtype=torch.float32
    ).requires_grad_(True)
    mean_over_sigma = _to_device(mean / sigma, device)
    regularization = 1.0 / (2.0 * C * len(train))
    optimizer = torch.optim.LBFGS(
        [feature_levels, bias_values],
        max_iter=steps,
        history_size=20,
        tolerance_grad=1e-9,
        tolerance_change=1e-12,
        line_search_fn='strong_wolfe',
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        expanded = feature_levels[code_index]
        weights = (class_vectors * expanded[None, :]) @ feature_vectors
        raw_bias = bias_vectors @ bias_values
        intercept = raw_bias + (weights * mean_over_sigma[None, :]).sum(1)
        logits = torch.cat(
            (
                torch.zeros(len(train), 1, device=device),
                standardized @ weights.T + intercept,
            ),
            dim=1,
        )
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss = loss + regularization * expanded.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        feature_levels.detach().cpu().numpy(),
        bias_values.detach().cpu().numpy(),
        bias_atoms,
    )


def folded_head(
    class_dictionary: np.ndarray,
    support: np.ndarray,
    values: np.ndarray,
    mean: np.ndarray,
    sigma: np.ndarray,
    intercept: np.ndarray,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct and fold a reference-class head into raw feature space."""
    class_vectors = class_dictionary[:, support // columns]
    feature_index = support % columns
    standardized_weights = np.zeros((ROWS, columns), dtype=np.float64)
    np.add.at(
        standardized_weights,
        (slice(None), feature_index),
        class_vectors * values[None, :],
    )
    raw_weights = standardized_weights / sigma[None, :]
    raw_intercept = intercept - (
        standardized_weights * (mean / sigma)[None, :]
    ).sum(1)
    return raw_weights, raw_intercept


def codebook_raw_head(
    class_dictionary: np.ndarray,
    support: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    bias_levels: np.ndarray,
    bias_assignment: np.ndarray,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct raw weights and a codebook-quantized raw intercept."""
    weights = standard_weights(
        class_dictionary, support, values, columns
    )
    return weights / sigma[None, :], bias_levels[bias_assignment]


def dictionary_bias_raw_head(
    class_dictionary: np.ndarray,
    support: np.ndarray,
    values: np.ndarray,
    sigma: np.ndarray,
    bias_values: np.ndarray,
    bias_atoms: np.ndarray,
    columns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct raw weights and a class-dictionary-coded raw intercept."""
    weights = standard_weights(
        class_dictionary, support, values, columns
    )
    return (
        weights / sigma[None, :],
        class_dictionary[:, bias_atoms] @ bias_values,
    )


def accuracy(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    intercept: np.ndarray,
) -> float:
    """Reference-class classification accuracy."""
    rest = features @ weights.T + intercept
    logits = np.concatenate(
        (np.zeros((len(features), 1)), rest), axis=1
    )
    return float((logits.argmax(1) == labels).mean())


def write_rows(rows: list[dict[str, object]], path: str) -> None:
    """Rewrite partial results after each completed row."""
    with open(path, 'w', newline='') as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(rows)


def offset_paths(support: int, prefix: str | None = None) -> tuple[str, ...]:
    """Return the four raw repeat paths for a support size."""
    prefix = prefix or (
        'resisc45_codebook'
        if support == 3840
        else f'resisc45_codebook_s{support}'
    )
    return tuple(
        os.path.join(EXPERIMENTS_DIR, f'{prefix}_offset{offset}.csv')
        for offset in (0, 16, 32, 48)
    )


def summarize_offsets(
    path: str, support: int, prefix: str | None = None
) -> None:
    """Aggregate four support searches and select C by mean validation."""
    raw: list[dict[str, str]] = []
    for source_path in offset_paths(support, prefix):
        with open(source_path, newline='') as source:
            raw.extend(csv.DictReader(source))
    groups: dict[tuple[int, float], list[dict[str, str]]] = {}
    for row in raw:
        key = (int(row['levels']), float(row['C']))
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (level_count, C), rows in sorted(groups.items()):
        offsets = sorted(int(row['search_offset']) for row in rows)
        if offsets != [0, 16, 32, 48]:
            raise ValueError(
                f'levels={level_count} C={C} has offsets {offsets}'
            )
        validation = np.array([float(row['val_accuracy']) for row in rows])
        test = np.array([float(row['test_accuracy']) for row in rows])
        continuous_validation = np.array(
            [float(row['continuous_val']) for row in rows]
        )
        continuous_test = np.array(
            [float(row['continuous_test']) for row in rows]
        )
        assignment_bits = np.array(
            [int(row['assignment_bits']) for row in rows]
        )
        total_bits = np.array(
            [int(row['total_model_bits']) for row in rows]
        )
        touched = np.array([int(row['touched_columns']) for row in rows])
        summaries.append({
            'levels': level_count,
            'learned_values': level_count + ROWS,
            'support': int(rows[0]['support']),
            'repeats': len(rows),
            'search_offsets': ';'.join(map(str, offsets)),
            'C': C,
            'validation_mean': f'{validation.mean():.6f}',
            'validation_stdev': f'{validation.std(ddof=1):.6f}',
            'test_mean': f'{test.mean():.6f}',
            'test_stdev': f'{test.std(ddof=1):.6f}',
            'test_min': f'{test.min():.6f}',
            'test_max': f'{test.max():.6f}',
            'continuous_validation_mean': (
                f'{continuous_validation.mean():.6f}'
            ),
            'continuous_test_mean': f'{continuous_test.mean():.6f}',
            'touched_columns_mean': f'{touched.mean():.2f}',
            'assignment_bits_mean': f'{assignment_bits.mean():.2f}',
            'total_model_bits_mean': f'{total_bits.mean():.2f}',
            'validation_selected_for_levels': False,
        })

    for level_count in sorted({int(row['levels']) for row in summaries}):
        candidates = [
            row for row in summaries if int(row['levels']) == level_count
        ]
        selected = max(
            candidates, key=lambda row: float(row['validation_mean'])
        )
        selected['validation_selected_for_levels'] = True
    write_rows(summaries, path)
    for row in summaries:
        if row['validation_selected_for_levels']:
            print(
                f"{row['learned_values']} learned values: "
                f"C={row['C']} validation={row['validation_mean']} "
                f"test={row['test_mean']} +/- {row['test_stdev']}"
            )


def summarize_bias_offsets(path: str, prefix: str) -> None:
    """Aggregate repeated feature- and intercept-codebook rows."""
    raw: list[dict[str, str]] = []
    for source_path in offset_paths(8192, prefix):
        with open(source_path, newline='') as source:
            raw.extend(csv.DictReader(source))
    groups: dict[tuple[int, int, float], list[dict[str, str]]] = {}
    for row in raw:
        key = (
            int(row['feature_levels']),
            int(row['bias_levels']),
            float(row['C']),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (feature_levels, bias_levels, C), rows in sorted(groups.items()):
        offsets = sorted(int(row['search_offset']) for row in rows)
        if offsets != [0, 16, 32, 48]:
            raise ValueError(
                f'Q={feature_levels} B={bias_levels} C={C} '
                f'has offsets {offsets}'
            )
        validation = np.array([float(row['val_accuracy']) for row in rows])
        test = np.array([float(row['test_accuracy']) for row in rows])
        assignment_bits = np.array(
            [int(row['assignment_bits']) for row in rows]
        )
        total_bits = np.array(
            [int(row['total_model_bits']) for row in rows]
        )
        summaries.append({
            'feature_levels': feature_levels,
            'bias_levels': bias_levels,
            'learned_values': feature_levels + bias_levels,
            'support': int(rows[0]['support']),
            'repeats': len(rows),
            'search_offsets': ';'.join(map(str, offsets)),
            'C': C,
            'validation_mean': f'{validation.mean():.6f}',
            'validation_stdev': f'{validation.std(ddof=1):.6f}',
            'test_mean': f'{test.mean():.6f}',
            'test_stdev': f'{test.std(ddof=1):.6f}',
            'test_min': f'{test.min():.6f}',
            'test_max': f'{test.max():.6f}',
            'assignment_bits_mean': f'{assignment_bits.mean():.2f}',
            'total_model_bits_mean': f'{total_bits.mean():.2f}',
            'validation_selected_for_values': False,
        })
    for learned_values in sorted(
        {int(row['learned_values']) for row in summaries}
    ):
        candidates = [
            row
            for row in summaries
            if int(row['learned_values']) == learned_values
        ]
        selected = max(
            candidates, key=lambda row: float(row['validation_mean'])
        )
        selected['validation_selected_for_values'] = True
    write_rows(summaries, path)
    for row in summaries:
        if row['validation_selected_for_values']:
            print(
                f"{row['learned_values']} values: "
                f"Q={row['feature_levels']} B={row['bias_levels']} "
                f"C={row['C']} validation={row['validation_mean']} "
                f"test={row['test_mean']} +/- {row['test_stdev']}"
            )


def summarize_bias_dictionary_offsets(path: str, prefix: str) -> None:
    """Aggregate repeated class-dictionary-coded intercept rows."""
    raw: list[dict[str, str]] = []
    for source_path in offset_paths(8192, prefix):
        with open(source_path, newline='') as source:
            raw.extend(csv.DictReader(source))
    groups: dict[tuple[int, int, float], list[dict[str, str]]] = {}
    for row in raw:
        key = (
            int(row['feature_levels']),
            int(row['bias_atoms']),
            float(row['C']),
        )
        groups.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    for (feature_levels, bias_atoms, C), rows in sorted(groups.items()):
        offsets = sorted(int(row['search_offset']) for row in rows)
        if offsets != [0, 16, 32, 48]:
            raise ValueError(
                f'Q={feature_levels} bias_atoms={bias_atoms} C={C} '
                f'has offsets {offsets}'
            )
        validation = np.array([float(row['val_accuracy']) for row in rows])
        test = np.array([float(row['test_accuracy']) for row in rows])
        assignment_bits = np.array(
            [int(row['assignment_bits']) for row in rows]
        )
        total_bits = np.array(
            [int(row['total_model_bits']) for row in rows]
        )
        summaries.append({
            'feature_levels': feature_levels,
            'bias_atoms': bias_atoms,
            'learned_values': feature_levels + bias_atoms,
            'support': int(rows[0]['support']),
            'repeats': len(rows),
            'search_offsets': ';'.join(map(str, offsets)),
            'C': C,
            'validation_mean': f'{validation.mean():.6f}',
            'validation_stdev': f'{validation.std(ddof=1):.6f}',
            'test_mean': f'{test.mean():.6f}',
            'test_stdev': f'{test.std(ddof=1):.6f}',
            'test_min': f'{test.min():.6f}',
            'test_max': f'{test.max():.6f}',
            'assignment_bits_mean': f'{assignment_bits.mean():.2f}',
            'total_model_bits_mean': f'{total_bits.mean():.2f}',
            'validation_selected_for_values': False,
        })
    for learned_values in sorted(
        {int(row['learned_values']) for row in summaries}
    ):
        candidates = [
            row
            for row in summaries
            if int(row['learned_values']) == learned_values
        ]
        selected = max(
            candidates, key=lambda row: float(row['validation_mean'])
        )
        selected['validation_selected_for_values'] = True
    write_rows(summaries, path)
    for row in summaries:
        if row['validation_selected_for_values']:
            print(
                f"{row['learned_values']} values: "
                f"Q={row['feature_levels']} bias_atoms={row['bias_atoms']} "
                f"C={row['C']} validation={row['validation_mean']} "
                f"test={row['test_mean']} +/- {row['test_stdev']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--support', type=int, default=3840)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--levels', type=int, nargs='*', default=list(LEVELS))
    parser.add_argument('--bias-levels', type=int, nargs='*', default=[])
    parser.add_argument('--bias-atoms', type=int, nargs='*', default=[])
    parser.add_argument(
        '--bias-configs',
        nargs='*',
        default=[],
        help='optional feature-level/bias-atom pairs, e.g. 3/17 6/16',
    )
    parser.add_argument('--Cs', type=float, nargs='*', default=list(C_GRID))
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=RESULT_PATH)
    parser.add_argument('--save-model-dir', default=None)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--summarize-support', type=int, default=3840)
    parser.add_argument('--summarize-prefix', default=None)
    parser.add_argument('--summarize-bias', action='store_true')
    parser.add_argument('--summarize-bias-atoms', action='store_true')
    args = parser.parse_args()
    if args.summarize_bias_atoms:
        if not args.summarize_prefix:
            parser.error(
                '--summarize-prefix is required with --summarize-bias-atoms'
            )
        summarize_bias_dictionary_offsets(
            args.output, args.summarize_prefix
        )
        return
    if args.bias_levels and args.bias_atoms:
        parser.error('--bias-levels and --bias-atoms are mutually exclusive')
    bias_configs = {
        tuple(int(value) for value in config.split('/'))
        for config in args.bias_configs
    }
    if any(len(config) != 2 for config in bias_configs):
        parser.error('--bias-configs entries must have feature/bias form')
    if args.summarize_bias:
        if not args.summarize_prefix:
            parser.error('--summarize-prefix is required with --summarize-bias')
        summarize_bias_offsets(args.output, args.summarize_prefix)
        return
    if args.summarize:
        summarize_offsets(
            args.output, args.summarize_support, args.summarize_prefix
        )
        return

    features, labels, mappings = load_list()
    class_dictionary = gaussian_atoms(CLASS_ATOMS)
    feature_dictionary = np.eye(features['train'].shape[1])
    seeds = range(args.offset, args.offset + SEARCHES)
    start = time.time()
    combined = union_support(
        features,
        labels,
        class_dictionary,
        feature_dictionary,
        args.support,
        seeds,
        'quota',
    )
    support = prune(
        features,
        labels,
        class_dictionary,
        feature_dictionary,
        combined,
        args.support,
        3,
        0.1,
        300,
    )
    mean, sigma, exponent = rounded_moments(features['train'])
    touched = np.unique(support % features['train'].shape[1])
    exponent_range = int(exponent[touched].max() - exponent[touched].min() + 1)
    scale_bits = max(1, math.ceil(math.log2(exponent_range)))
    atom_bits = math.ceil(math.log2(CLASS_ATOMS))
    column_bits = math.ceil(math.log2(features['train'].shape[1]))

    rows: list[dict[str, object]] = []
    for C in args.Cs:
        continuous, continuous_intercept = fit_continuous(
            features['train'],
            labels['train'],
            class_dictionary,
            support,
            C,
            mean,
            sigma,
            args.steps,
            args.device,
        )
        continuous_weights, continuous_bias = folded_head(
            class_dictionary,
            support,
            continuous,
            mean,
            sigma,
            continuous_intercept,
            features['train'].shape[1],
        )
        continuous_val = accuracy(
            features['val'],
            labels['val'],
            continuous_weights,
            continuous_bias,
        )
        continuous_test = accuracy(
            features['test'],
            labels['test'],
            continuous_weights,
            continuous_bias,
        )
        for level_count in args.levels:
            initial, assignment = scalar_codebook(
                continuous, level_count
            )
            levels, intercept = fit_codebook(
                features['train'],
                labels['train'],
                class_dictionary,
                support,
                assignment,
                initial,
                C,
                mean,
                sigma,
                args.steps,
                args.device,
            )
            expanded = levels[assignment]
            level_bits = math.ceil(math.log2(level_count))
            assignment_bits = len(support) * (
                atom_bits + column_bits + level_bits
            )
            assignment_bits += len(touched) * scale_bits
            assignment_bits += (
                CANDIDATE_MAPPING_BITS + SCALE_EXPONENT_ORIGIN_BITS
            )
            if args.bias_levels:
                for bias_level_count in args.bias_levels:
                    feature_levels, bias_levels, bias_assignment = (
                        fit_bias_codebook(
                            features['train'],
                            labels['train'],
                            class_dictionary,
                            support,
                            assignment,
                            levels,
                            intercept,
                            bias_level_count,
                            C,
                            mean,
                            sigma,
                            args.steps,
                            args.device,
                        )
                    )
                    weights, bias = codebook_raw_head(
                        class_dictionary,
                        support,
                        feature_levels[assignment],
                        sigma,
                        bias_levels,
                        bias_assignment,
                        features['train'].shape[1],
                    )
                    bias_bits = ROWS * math.ceil(
                        math.log2(bias_level_count)
                    )
                    row = {
                        'support': len(support),
                        'search_offset': args.offset,
                        'feature_levels': level_count,
                        'bias_levels': bias_level_count,
                        'learned_values': level_count + bias_level_count,
                        'touched_columns': len(touched),
                        'assignment_bits': assignment_bits + bias_bits,
                        'learned_value_bits': (
                            level_count + bias_level_count
                        ) * 32,
                        'total_model_bits': (
                            assignment_bits
                            + bias_bits
                            + (level_count + bias_level_count) * 32
                        ),
                        'candidate_mapping_bits': CANDIDATE_MAPPING_BITS,
                        'scale_exponent_bits': (
                            len(touched) * scale_bits
                            + SCALE_EXPONENT_ORIGIN_BITS
                        ),
                        'support_sha256': array_hash(support),
                        'level_assignment_sha256': array_hash(assignment),
                        'bias_assignment_sha256': array_hash(bias_assignment),
                        'C': C,
                        'continuous_val': round(continuous_val, 6),
                        'continuous_test': round(continuous_test, 6),
                        'val_accuracy': round(
                            accuracy(
                                features['val'],
                                labels['val'],
                                weights,
                                bias,
                            ),
                            6,
                        ),
                        'test_accuracy': round(
                            accuracy(
                                features['test'],
                                labels['test'],
                                weights,
                                bias,
                            ),
                            6,
                        ),
                        'seconds': round(time.time() - start, 1),
                    }
                    rows.append(row)
                    print(row, flush=True)
                    write_rows(rows, args.output)
            elif args.bias_atoms:
                for bias_atom_count in args.bias_atoms:
                    if bias_configs and (
                        level_count,
                        bias_atom_count,
                    ) not in bias_configs:
                        continue
                    feature_levels, bias_values, bias_atoms = (
                        fit_bias_dictionary(
                            features['train'],
                            labels['train'],
                            class_dictionary,
                            support,
                            assignment,
                            levels,
                            intercept,
                            bias_atom_count,
                            C,
                            mean,
                            sigma,
                            args.steps,
                            args.device,
                        )
                    )
                    weights, bias = dictionary_bias_raw_head(
                        class_dictionary,
                        support,
                        feature_levels[assignment],
                        sigma,
                        bias_values,
                        bias_atoms,
                        features['train'].shape[1],
                    )
                    artifact = ''
                    artifact_sha256 = ''
                    if args.save_model_dir:
                        tag = (
                            f's{len(support)}_o{args.offset}_'
                            f'q{level_count}_b{bias_atom_count}_'
                            f'c{C:g}'
                        )
                        artifact, artifact_sha256 = save_model_artifact(
                            args.save_model_dir,
                            tag,
                            feature_levels,
                            bias_values,
                            support,
                            assignment,
                            bias_atoms,
                            exponent,
                            mappings,
                            C,
                        )
                    row = {
                        'support': len(support),
                        'search_offset': args.offset,
                        'feature_levels': level_count,
                        'bias_atoms': bias_atom_count,
                        'learned_values': level_count + bias_atom_count,
                        'touched_columns': len(touched),
                        'assignment_bits': (
                            assignment_bits + bias_atom_count * atom_bits
                        ),
                        'learned_value_bits': (
                            level_count + bias_atom_count
                        ) * 32,
                        'total_model_bits': (
                            assignment_bits
                            + bias_atom_count * atom_bits
                            + (level_count + bias_atom_count) * 32
                        ),
                        'candidate_mapping_bits': CANDIDATE_MAPPING_BITS,
                        'scale_exponent_bits': (
                            len(touched) * scale_bits
                            + SCALE_EXPONENT_ORIGIN_BITS
                        ),
                        'support_sha256': array_hash(support),
                        'level_assignment_sha256': array_hash(assignment),
                        'bias_atoms_sha256': array_hash(bias_atoms),
                        'artifact': artifact,
                        'artifact_sha256': artifact_sha256,
                        'C': C,
                        'continuous_val': round(continuous_val, 6),
                        'continuous_test': round(continuous_test, 6),
                        'val_accuracy': round(
                            accuracy(
                                features['val'],
                                labels['val'],
                                weights,
                                bias,
                            ),
                            6,
                        ),
                        'test_accuracy': round(
                            accuracy(
                                features['test'],
                                labels['test'],
                                weights,
                                bias,
                            ),
                            6,
                        ),
                        'seconds': round(time.time() - start, 1),
                    }
                    rows.append(row)
                    print(row, flush=True)
                    write_rows(rows, args.output)
            else:
                weights, bias = folded_head(
                    class_dictionary,
                    support,
                    expanded,
                    mean,
                    sigma,
                    intercept,
                    features['train'].shape[1],
                )
                artifact = ''
                artifact_sha256 = ''
                if args.save_model_dir:
                    tag = (
                        f's{len(support)}_o{args.offset}_'
                        f'q{level_count}_fullbias_c{C:g}'
                    )
                    artifact, artifact_sha256 = save_model_artifact(
                        args.save_model_dir,
                        tag,
                        levels,
                        bias,
                        support,
                        assignment,
                        np.empty(0, dtype=np.int64),
                        exponent,
                        mappings,
                        C,
                    )
                row = {
                    'support': len(support),
                    'search_offset': args.offset,
                    'levels': level_count,
                    'learned_values': level_count + ROWS,
                    'touched_columns': len(touched),
                    'assignment_bits': assignment_bits,
                    'learned_value_bits': (level_count + ROWS) * 32,
                    'total_model_bits': (
                        assignment_bits + (level_count + ROWS) * 32
                    ),
                    'candidate_mapping_bits': CANDIDATE_MAPPING_BITS,
                    'scale_exponent_bits': (
                        len(touched) * scale_bits
                        + SCALE_EXPONENT_ORIGIN_BITS
                    ),
                    'support_sha256': array_hash(support),
                    'level_assignment_sha256': array_hash(assignment),
                    'artifact': artifact,
                    'artifact_sha256': artifact_sha256,
                    'C': C,
                    'continuous_val': round(continuous_val, 6),
                    'continuous_test': round(continuous_test, 6),
                    'val_accuracy': round(
                        accuracy(
                            features['val'], labels['val'], weights, bias
                        ),
                        6,
                    ),
                    'test_accuracy': round(
                        accuracy(
                            features['test'], labels['test'], weights, bias
                        ),
                        6,
                    ),
                    'seconds': round(time.time() - start, 1),
                }
                rows.append(row)
                print(row, flush=True)
                write_rows(rows, args.output)


if __name__ == '__main__':
    main()
