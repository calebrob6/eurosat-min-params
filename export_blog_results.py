#!/usr/bin/env python
"""Export current local and representation-comparison CSVs without fitting."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
LOCAL_MODELS = {
    'Ours 33 features': '306',
    'ImageStats': 'imagestats',
}
BACKBONE_SCOREBOARD = ROOT / 'experiments/torchgeo_bench_eurosat/blog_post/scoreboard.csv'
SUPPORTING_RESULTS = ROOT / 'experiments/resisc45/results.csv'
TRAIN_COUNTS = (162, 324, 810, 1620, 3240, 8100, 16200)
OVERLAP = ROOT / 'experiments/representation_overlap/results'
OVERLAP_MODELS = {
    'resnet50': 'ResNet-50',
    'convnext_tiny': 'ConvNeXt-Tiny',
    'dofa_large': 'DOFA Large',
    'olmoearth_nano': 'OlmoEarth v1.2 Nano',
    'olmoearth_base': 'OlmoEarth v1.2 Base',
}


def read_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'cannot export an empty table: {path.name}')
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def one_row(rows: list[dict], **conditions) -> dict:
    found = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(found) != 1:
        raise ValueError(f'expected one result for {conditions}, found {len(found)}')
    return found[0]


def validate_backbone(row: dict) -> None:
    """Check a recorded frozen-backbone result's split and parameter counts."""
    dimensions = int(row['feature_dim'])
    if (row['dataset'] != 'eurosat' or row['merge_val'].lower() != 'false'
            or (int(row['n_train']), int(row['n_val']), int(row['n_test']))
            != (16200, 5400, 5400) or int(row['seed']) != 0
            or dimensions < 1 or int(row['backbone_parameters']) < 1
            or int(row['probe_parameters']) != 10 * (dimensions + 1)):
        raise ValueError('recorded backbone result has an unexpected protocol or parameter count')
    if not 0 <= float(row['test_accuracy_reported']) <= 1:
        raise ValueError('recorded backbone accuracy is invalid')


def scoreboard() -> list[dict]:
    """Combine the two local models with the optional recorded backbones."""
    local_path = ROOT / 'results/eurosat_models.csv'
    local = read_rows(local_path)
    rows = []
    for name, model in LOCAL_MODELS.items():
        original = one_row(local, model=model, split='test')
        dimensions = 33 if model == '306' else 52
        parameters = 9 * (dimensions + 1)
        accuracy = float(original['accuracy'])
        if (int(original['parameters']) != parameters or int(original['images']) != 5400
                or not 0 <= int(original['correct']) <= 5400
                or not math.isclose(accuracy, int(original['correct']) / 5400,
                                    rel_tol=0, abs_tol=1e-12)):
            raise ValueError(f'{name}: inconsistent local result')
        rows.append(dict(
            model=name, feature_dim=dimensions, backbone_parameters=0,
            probe_parameters=parameters, learned_parameters=parameters,
            test_accuracy=original['accuracy'], test_accuracy_percent=f'{100 * accuracy:.2f}',
            n_test=5400, source_kind='checked_in_local_result',
            source_csv=local_path.relative_to(ROOT).as_posix(),
        ))
    if BACKBONE_SCOREBOARD.exists():
        backbones = read_rows(BACKBONE_SCOREBOARD)
        if not backbones or len({row['model_config'] for row in backbones}) != len(backbones):
            raise ValueError('expected distinct recorded backbone configurations')
        for original in backbones:
            validate_backbone(original)
            backbone, probe = int(original['backbone_parameters']), int(original['probe_parameters'])
            rows.append(dict(
                model=original['display_name'], feature_dim=int(original['feature_dim']),
                backbone_parameters=backbone, probe_parameters=probe,
                learned_parameters=backbone + probe,
                test_accuracy=original['test_accuracy_reported'],
                test_accuracy_percent=f'{100 * float(original["test_accuracy_reported"]):.2f}',
                n_test=5400, source_kind='recorded_backbone_scoreboard',
                source_csv=BACKBONE_SCOREBOARD.relative_to(ROOT).as_posix(),
            ))
    if len({row['model'] for row in rows}) != len(rows):
        raise ValueError('duplicate scoreboard model names')
    return sorted(rows, key=lambda row: float(row['test_accuracy']))


def training_fractions() -> list[dict]:
    """Export the two fixed-C, five-seed random and spatial learning curves."""
    rows = []
    for name, filename, parameters, C in [
        ('Ours 33 features', 'frontier_fractions.csv', '306', 3),
        ('ImageStats', 'imagestats_fractions.csv', '477', 300),
    ]:
        source = ROOT / 'results' / filename
        for row in read_rows(source):
            if row['seeds'] != '0;1;2;3;4':
                raise ValueError('local curve must contain five training-subsample seeds')
            if (row['learned_parameters'] != parameters or float(row['regularization_C']) != C
                    or not 0 <= float(row['test_accuracy_mean']) <= 1
                    or not 0 <= float(row['test_accuracy_stdev']) <= 1
                    or any(not 0 <= float(row[f'seed_{seed}_test_accuracy']) <= 1 for seed in range(5))):
                raise ValueError('local curve has unexpected parameters or accuracy statistics')
            rows.append(dict(
                model=name, split_protocol=row['split_protocol'],
                train_fraction_percent=float(row['train_fraction_percent']),
                n_train=int(row['n_train']), n_test=int(row['n_test']),
                test_accuracy=row['test_accuracy_mean'],
                test_accuracy_stdev=row['test_accuracy_stdev'],
                uncertainty='sample_stdev_over_5_training_seeds',
                source_csv=source.relative_to(ROOT).as_posix(),
            ))
    keys = {(r['model'], r['split_protocol'], r['n_train']) for r in rows}
    expected = {
        (name, protocol, count)
        for name in LOCAL_MODELS
        for protocol in ('random', 'spatial') for count in TRAIN_COUNTS
    }
    if keys != expected or len(rows) != len(expected):
        raise ValueError('expected seven fractions for both local models and both protocols')
    if any(row['n_test'] != 5400 or row['train_fraction_percent'] != row['n_train'] * 100 / 16200
           for row in rows):
        raise ValueError('learning-curve split sizes or fractions are inconsistent')
    return rows


def representation_tables() -> dict[str, list[dict]]:
    """Export the three current overlap tables, checking their paired sources."""
    classification = read_rows(OVERLAP / 'classification.csv')
    comparisons = read_rows(OVERLAP / 'complementarity.csv')
    decoding = read_rows(OVERLAP / 'decoding.csv')
    features = read_rows(OVERLAP / 'features.csv')
    combined, explained, reverse = [], [], []
    for model, label in OVERLAP_MODELS.items():
        base = one_row(classification, model=model, context='standalone')
        accuracy = float(base['accuracy'])
        if not math.isclose(accuracy, int(base['correct']) / 5400, abs_tol=1e-12):
            raise ValueError(f'{model}: inconsistent backbone accuracy')
        combined_row = {'model': label, 'embedding_accuracy_percent': f'{100 * accuracy:.2f}'}
        explained_row = {'model': label}
        for rep, short in (('frontier33', '33'), ('pool377', '377')):
            clf = one_row(classification, model=model, representation=rep, context='combined', variant='original')
            delta = one_row(comparisons, model=model, representation=rep, context='combined', variant='original')
            if (delta['primary'] != 'True' or not math.isclose(
                    float(clf['accuracy']) - accuracy, float(delta['accuracy_delta']), abs_tol=1e-12)):
                raise ValueError(f'{model}/{rep}: inconsistent paired comparison')
            if not math.isclose(float(clf['accuracy']), int(clf['correct']) / 5400, abs_tol=1e-12):
                raise ValueError(f'{model}/{rep}: inconsistent combined accuracy')
            combined_row.update({
                f'plus_{short}_accuracy_percent': f'{100 * float(clf["accuracy"]):.2f}',
                f'plus_{short}_gain_points': float(delta['accuracy_delta']) * 100,
                f'plus_{short}_gain_ci_low': float(delta['accuracy_ci_low']) * 100,
                f'plus_{short}_gain_ci_high': float(delta['accuracy_ci_high']) * 100,
                f'plus_{short}_holm_pvalue': delta['mcnemar_holm_pvalue'],
            })
            forward = one_row(decoding, model=model, representation=rep, direction='h_to_z',
                              context='ordinary', variant='original')
            explained_row.update({
                f'features_{short}_r2_percent': f'{100 * float(forward["native_r2"]):.1f}',
                f'features_{short}_ci_low_percent': float(forward['native_ci_low']) * 100,
                f'features_{short}_ci_high_percent': float(forward['native_ci_high']) * 100,
            })
        combined.append(combined_row)
        explained.append(explained_row)
        rows = [row for row in features if row['model'] == model
                and row['representation'] == 'frontier33' and row['variant'] == 'original']
        if len(rows) != 33 or len({row['feature'] for row in rows}) != 33:
            raise ValueError(f'{model}: expected 33 unique decoded features')
        values = {row['feature']: float(row['r2']) for row in rows}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f'{model}: undefined decoded feature')
        mean = statistics.mean(values.values())
        aggregate = one_row(decoding, model=model, representation='frontier33', direction='z_to_h',
                            context='ordinary', variant='original')
        if not math.isclose(mean, float(aggregate['macro_r2']), abs_tol=1e-12):
            raise ValueError(f'{model}: feature scores disagree with their mean')
        reverse.append({
            'model': label, 'mean_r2': f'{mean:.3f}',
            'largest_ndvi_region_r2': f'{values["blob2lrg_ndvi"]:.3f}',
            'low_ndvi_anisotropy_r2': f'{values["tail_aniso_low_ndvi"]:.3f}',
        })
    return {
        'combined_features.csv': combined, 'explained_variance.csv': explained,
        'decoded_features.csv': reverse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/blog-results')
    args = parser.parse_args()
    tables = {
        'scoreboard.csv': scoreboard(), 'training_fractions.csv': training_fractions(),
        **representation_tables(),
    }
    for output_name, source_name in (
        ('models.csv', 'eurosat_models.csv'),
        ('features_33.csv', 'eurosat_306_features.csv'),
        ('class_accuracy.csv', 'eurosat_per_class.csv'),
    ):
        rows = read_rows(ROOT / 'results' / source_name)
        if output_name in ('models.csv', 'class_accuracy.csv'):
            rows = [row for row in rows if row['model'] in LOCAL_MODELS.values()]
        tables[output_name] = rows
    if SUPPORTING_RESULTS.exists():
        tables['supporting_results.csv'] = read_rows(SUPPORTING_RESULTS)
    args.output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_rows(args.output / name, rows)
    print(f'Exported {len(tables)} current result tables to {args.output}')


if __name__ == '__main__':
    main()
