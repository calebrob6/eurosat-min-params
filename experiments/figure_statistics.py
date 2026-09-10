#!/usr/bin/env python
"""Recompute the training-patch statistics quoted in Figures 3 and 4."""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reproduce import prepare_data
from src.data import CLASSES, list_split, read_tif
from src.features import patch_features


def sample_indices(labels: np.ndarray, per_class: int = 120) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.concatenate([
        rng.choice(np.flatnonzero(labels == index), size=per_class, replace=False)
        for index in range(len(CLASSES))
    ])


def write(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def check_results(output: Path) -> None:
    for name, reference in (('orientation_examples.csv', 'figure_examples.csv'),
                            ('gradient_medians.csv', 'figure_gradient_medians.csv')):
        with (output / name).open() as handle:
            actual = list(csv.DictReader(handle))
        with (ROOT / 'results' / reference).open() as handle:
            expected = list(csv.DictReader(handle))
        if actual != expected:
            raise ValueError(f'figure measurements differ from results/{reference}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/figure-statistics')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    prepare_data(args.download, False)
    paths, labels = list_split('train')
    chosen = sample_indices(labels)
    rows = []
    for start in range(0, len(chosen), 64):
        indices = chosen[start:start + 64]
        images = np.stack([read_tif(paths[index]) for index in indices])
        features, names = patch_features(
            images, pcts=(), grad_scales=3, coherence_scales=1, orient_entropy_bins=8,
        )
        columns = [names.index(name) for name in ('g2mean_b2', 'oent_b7', 'coh0_b7')]
        for index, values in zip(indices, features[:, columns], strict=True):
            rows.append({
                'filename': Path(paths[index]).name, 'class': CLASSES[labels[index]],
                'b03_coarse_gradient_mean': float(values[0]),
                'b08_orientation_entropy': float(values[1]),
                'b08_coherence': float(values[2]),
            })
    demos = []
    for name, quantile in (('AnnualCrop', .05), ('River', .10), ('Residential', .50), ('Forest', .50)):
        candidates = [row for row in rows if row['class'] == name]
        entropies = np.asarray([row['b08_orientation_entropy'] for row in candidates])
        index = int(np.argmin(np.abs(entropies - np.quantile(entropies, quantile))))
        demos.append({'figure': 4, 'selection_quantile': quantile, **candidates[index]})
    distributions = []
    for name in CLASSES:
        selected = [row['b03_coarse_gradient_mean'] for row in rows if row['class'] == name]
        distributions.append({'figure': 3, 'class': name, 'patches': len(selected),
                              'median_b03_coarse_gradient': float(np.median(selected))})
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'training_distributions.csv', rows)
    write(args.output / 'orientation_examples.csv', demos)
    write(args.output / 'gradient_medians.csv', distributions)
    if args.check:
        check_results(args.output)
    print(f'Saved {len(rows)} training-patch measurements to {args.output}')
    for row in demos:
        print(f'{row["class"]}: entropy={row["b08_orientation_entropy"]:.2f}, '
              f'coherence={row["b08_coherence"]:.2f}')


if __name__ == '__main__':
    main()
