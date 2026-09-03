#!/usr/bin/env python
"""Summarize the repeated j16 RESISC45 heads near the 4,096-value cap."""
from __future__ import annotations

import csv
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(__file__)
SOURCES = (
    ('resisc45_union_jitter_j16.csv', 'j16', 'own'),
    ('resisc45_union_j16_cap_q96.csv', 'j16', 'own'),
    ('resisc45_union_j16_cap_q128.csv', 'j16', 'own'),
    ('resisc45_union_j32crop_own.csv', 'j32crop', 'own'),
    ('resisc45_union_j32crop_d8rank.csv', 'j32crop', 'd8'),
    ('resisc45_union_j32crop_4096.csv', 'j32crop', 'd8'),
)
OUTPUT = os.path.join(HERE, 'resisc45_exact_cap_result.csv')


def load_rows() -> list[dict[str, str]]:
    """Load and validate all raw repeated-head rows."""
    rows: list[dict[str, str]] = []
    for filename, pool, ranking in SOURCES:
        path = os.path.join(HERE, filename)
        with open(path, newline='') as source:
            for row in csv.DictReader(source):
                rows.append({**row, '_pool': pool, '_ranking': ranking})
    expected_offsets = {0, 16, 32, 48}
    grouped: dict[tuple[int, str, str, str], set[int]] = defaultdict(set)
    for row in rows:
        grouped[(
            int(row['deployed_values']),
            row['list'],
            row['_pool'],
            row['_ranking'],
        )].add(
            int(row['seed_offset'])
        )
    for key, offsets in grouped.items():
        if offsets != expected_offsets:
            raise ValueError(f'{key} has offsets {sorted(offsets)}')
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Aggregate four offsets and mark the mean-validation pick per budget."""
    groups: dict[
        tuple[int, str, str, str], list[dict[str, str]]
    ] = defaultdict(list)
    for row in rows:
        groups[(
            int(row['deployed_values']),
            row['list'],
            row['_pool'],
            row['_ranking'],
        )].append(row)

    summaries: list[dict[str, object]] = []
    for (
        deployed,
        candidate_list,
        pool,
        ranking,
    ), group in sorted(groups.items()):
        validation = np.array([float(row['val_accuracy']) for row in group])
        test = np.array([float(row['test_accuracy']) for row in group])
        summaries.append({
            'pool': pool,
            'ranking_pool': ranking,
            'candidate_list': candidate_list,
            'code_values': int(group[0]['budget']),
            'deployed_values': deployed,
            'repeats': len(group),
            'seed_offsets': ';'.join(row['seed_offset'] for row in group),
            'validation_mean': f'{validation.mean():.6f}',
            'validation_stdev': f'{validation.std(ddof=1):.6f}',
            'test_mean': f'{test.mean():.6f}',
            'test_stdev': f'{test.std(ddof=1):.6f}',
            'test_min': f'{test.min():.6f}',
            'test_max': f'{test.max():.6f}',
            'validation_selected_at_budget': False,
            'validation_selected_under_cap': False,
        })

    for deployed in sorted({int(row['deployed_values']) for row in summaries}):
        candidates = [
            row for row in summaries if int(row['deployed_values']) == deployed
        ]
        selected = max(candidates, key=lambda row: float(row['validation_mean']))
        selected['validation_selected_at_budget'] = True
    selected_under_cap = max(
        summaries, key=lambda row: float(row['validation_mean'])
    )
    selected_under_cap['validation_selected_under_cap'] = True
    return summaries


def main() -> None:
    rows = summarize(load_rows())
    with open(OUTPUT, 'w', newline='') as destination:
        writer = csv.DictWriter(
            destination, fieldnames=list(rows[0]), lineterminator='\n'
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['validation_selected_at_budget']:
            print(
                f"{row['deployed_values']} values: {row['pool']} "
                f"{row['ranking_pool']}-rank {row['candidate_list']} "
                f"validation={row['validation_mean']} "
                f"test={row['test_mean']} +/- {row['test_stdev']}"
            )


if __name__ == '__main__':
    main()
