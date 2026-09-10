#!/usr/bin/env python
"""Compare a fresh representation study with the recorded article results."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TABLES = {
    'classification.csv': (
        ('model', 'variant', 'representation', 'context'),
        ('accuracy', 'correct', 'errors', 'log_loss'),
    ),
    'decoding.csv': (
        ('model', 'variant', 'representation', 'context', 'direction'),
        ('native_r2', 'macro_r2', 'native_ci_low', 'native_ci_high', 'undefined_features'),
    ),
    'complementarity.csv': (
        ('model', 'variant', 'representation', 'context'),
        ('accuracy_delta', 'accuracy_ci_low', 'accuracy_ci_high', 'gained_correct', 'lost_correct',
         'log_loss_improvement', 'mcnemar_pvalue', 'mcnemar_holm_pvalue'),
    ),
    'features.csv': (
        ('model', 'variant', 'representation', 'position', 'feature'),
        ('r2',),
    ),
    'geometry.csv': (
        ('model', 'variant', 'representation', 'condition', 'seed'),
        ('cka',),
    ),
    'paired_reconstruction.csv': (
        ('model', 'variant'),
        ('delta_r2', 'delta_ci_low', 'delta_ci_high'),
    ),
}


def indexed(path: Path, keys: tuple[str, ...]) -> dict[tuple, dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    result = {tuple(row[key] for key in keys): row for row in rows}
    if len(result) != len(rows) or not rows:
        raise ValueError(f'{path}: duplicate or missing result rows')
    return result


def compare(reference: Path, fresh: Path, tolerance: float) -> list[dict]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    output = []
    for name, (keys, metrics) in TABLES.items():
        expected, actual = indexed(reference / name, keys), indexed(fresh / name, keys)
        if expected.keys() != actual.keys():
            raise ValueError(f'{name}: fresh and reference rows do not cover the same cases')
        for key in expected:
            for metric in metrics:
                before, after = expected[key][metric], actual[key][metric]
                difference = ''
                if before == '' or after == '':
                    match = before == after
                else:
                    a, b = float(before), float(after)
                    if not math.isfinite(a) or not math.isfinite(b):
                        raise ValueError(f'{name}/{key}/{metric}: non-finite result')
                    difference = b - a
                    match = abs(difference) <= tolerance
                output.append({
                    'table': name, 'case': '/'.join(key), 'metric': metric,
                    'reference': before, 'fresh': after, 'difference': difference, 'match': match,
                })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=ROOT / 'experiments/representation_overlap/results')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/overlap-comparison.csv')
    parser.add_argument('--atol', type=float, default=1e-6)
    parser.add_argument('--check', action='store_true', help='fail after writing the comparison if a value differs')
    args = parser.parse_args()
    rows = compare(args.reference, args.results, args.atol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    different = sum(not row['match'] for row in rows)
    print(f'{different}/{len(rows)} values differ by more than {args.atol:g}; comparison: {args.output}')
    if args.check and different:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
