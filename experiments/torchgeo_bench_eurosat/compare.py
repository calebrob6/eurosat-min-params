#!/usr/bin/env python
"""Compare fresh upstream measurements with the imported original CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
IMPORTED = ROOT / 'experiments/imported'
FRESH = HERE / 'reproduced'
FULL = 'eurosat-13band-merge-val-false-20260901'
OLMO = 'olmoearth-train-fractions-20260901'
STEMS = {
    'earthloc': 'tgeo_earthloc_s2_resnet50',
    'moco': 'tgeo_resnet50_s2all_moco',
    'resnet18': 'resnet18',
    'resnet50': 'resnet50',
    'vit_base': 'vit_base_patch16_224',
    'convnext_tiny': 'convnext_tiny',
    'dofa_base': 'tgeo_dofa_base',
    'dofa_large': 'tgeo_dofa_large',
    'olmoearth_nano': 'olmoearth_v1_2_nano',
    'olmoearth_small': 'olmoearth_v1_2_small',
    'olmoearth_base': 'olmoearth_v1_2_base',
}
CURVES = ('resnet50', 'convnext_tiny', 'dofa_base', 'dofa_large',
          'olmoearth_nano', 'olmoearth_small', 'olmoearth_base')


def linear_rows(path: Path) -> dict[int, dict]:
    with path.open() as handle:
        rows = [r for r in csv.DictReader(handle) if r['method'] == 'linear']
    by_count = {int(row['n_train']): row for row in rows}
    if len(by_count) != len(rows):
        raise ValueError(f'duplicate linear training counts in {path}')
    return by_count


def original_path(model: str, dataset: str, full: bool = False,
                  root: Path = IMPORTED) -> Path:
    if model.startswith('olmoearth'):
        protocol = 'normal' if dataset == 'eurosat' else 'spatial'
        return root / OLMO / protocol / f'{STEMS[model]}.csv'
    if full:
        return root / FULL / f'{STEMS[model]}.csv'
    folder = ('eurosat-train-fractions-20260901' if dataset == 'eurosat'
              else 'eurosat-spatial-train-fractions-20260901')
    return root / folder / f'{STEMS[model]}.csv'


def display_path(path: Path) -> str:
    resolved = path.resolve()
    return (resolved.relative_to(ROOT).as_posix()
            if resolved.is_relative_to(ROOT) else resolved.as_posix())


def comparison(model: str, dataset: str, count: int, old: dict, new: dict,
               old_path: Path, new_path: Path) -> dict:
    for label, row in [('original', old), ('fresh', new)]:
        if int(row['n_val']) != 5400 or int(row['n_test']) != 5400:
            raise ValueError(f'{model}: {label} split size differs')
        if row['merge_val'].lower() != 'false':
            raise ValueError(f'{model}: {label} merged validation into training')
    previous = float(old['metric_value'])
    current = float(new['metric_value'])
    return dict(
        model=model, dataset=dataset, train_fraction_percent=count * 100 / 16200,
        n_train=count, n_val=5400, n_test=5400,
        original_accuracy=previous, fresh_accuracy=current,
        difference_percentage_points=100 * (current - previous),
        difference_correct_images=round((current - previous) * 5400),
        original_best_c=float(old['best_c']), fresh_best_c=float(new['best_c']),
        original_ci_lower=float(old['ci_lower']), original_ci_upper=float(old['ci_upper']),
        fresh_ci_lower=float(new['ci_lower']), fresh_ci_upper=float(new['ci_upper']),
        matches_article_rounding=f'{100 * previous:.2f}' == f'{100 * current:.2f}',
        original_csv=display_path(old_path),
        fresh_csv=display_path(new_path),
    )


def write(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=FRESH)
    parser.add_argument('--originals-dir', type=Path, default=IMPORTED)
    parser.add_argument('--scoreboard-only', action='store_true')
    args = parser.parse_args()
    fresh, imported = args.results_dir, args.originals_dir
    scoreboard = []
    for model in STEMS:
        old_path = original_path(model, 'eurosat', full=True, root=imported)
        new_path = fresh / 'scoreboard' / f'{model}_eurosat.csv'
        row = comparison(model, 'eurosat', 16200, linear_rows(old_path)[16200],
                         linear_rows(new_path)[16200], old_path, new_path)
        scoreboard.append(row)
        print(f'{model}: {100 * row["original_accuracy"]:.2f}% -> '
              f'{100 * row["fresh_accuracy"]:.2f}% '
              f'({row["difference_correct_images"]:+d} images)')
    write(fresh / 'scoreboard_comparison.csv', scoreboard)
    if args.scoreboard_only:
        return

    curves = []
    for model in CURVES:
        for dataset in ('eurosat', 'eurosat-spatial'):
            old_path = original_path(model, dataset, root=imported)
            new_path = fresh / f'{model}_{dataset}.csv'
            old_rows, new_rows = linear_rows(old_path), linear_rows(new_path)
            if set(new_rows) != {162, 324, 810, 1620, 3240, 8100, 16200}:
                raise ValueError(f'{new_path}: incomplete fresh curve')
            for count, new in sorted(new_rows.items()):
                source = old_path
                if count not in old_rows:
                    if dataset != 'eurosat' or count != 16200:
                        raise ValueError(f'{old_path}: missing original fraction')
                    source = original_path(model, dataset, full=True, root=imported)
                    old = linear_rows(source)[count]
                else:
                    old = old_rows[count]
                curves.append(comparison(model, dataset, count, old, new, source, new_path))
    write(fresh / 'fraction_comparison.csv', curves)
    largest = max(curves, key=lambda r: abs(r['difference_percentage_points']))
    print(f'{len(curves)} curve points; largest absolute difference: '
          f'{largest["model"]} {largest["dataset"]} {largest["train_fraction_percent"]:g}% '
          f'{largest["difference_percentage_points"]:+.4f} pp')


if __name__ == '__main__':
    main()
