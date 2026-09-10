#!/usr/bin/env python
"""Export the final article's tables from recorded results; no model fitting.

Includes the 14-row scoreboard, 126 learning-curve points, feature list, class
scores, and the three representation-comparison tables. Original scoreboard
measurements stay separate from the later controlled probe fits.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import statistics

from experiments.torchgeo_bench_eurosat.compare import (
    CURVES,
    ROOT,
    linear_rows,
    original_path,
    write,
)

LOCAL_MODELS = {
    'ImageStats': 'imagestats',
    'Ours 18 features': '171',
    'Ours 30 features': '279',
    'Ours 33 features': '306',
}
BACKBONE_MODELS = {
    'torchgeo/earthloc_s2_resnet50': 'earthloc',
    'torchgeo/resnet50_s2_all_moco': 'moco',
    'timm/resnet18': 'resnet18',
    'timm/resnet50': 'resnet50',
    'timm/vit_base_patch16_224': 'vit_base',
    'timm/convnext_tiny': 'convnext_tiny',
    'OlmoEarth v1.2 Nano': 'olmoearth_nano',
    'torchgeo/dofa_base': 'dofa_base',
    'torchgeo/dofa_large': 'dofa_large',
    'OlmoEarth v1.2 Small': 'olmoearth_small',
    'OlmoEarth v1.2 Base': 'olmoearth_base',
}
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


def one_row(rows: list[dict], **conditions) -> dict:
    found = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(found) != 1:
        raise ValueError(f'expected one result for {conditions}, found {len(found)}')
    return found[0]


def validate_backbone(row: dict, dataset: str, count: int) -> None:
    if (row['dataset'] != dataset or row['method'] != 'linear'
            or row['metric_name'] != 'accuracy'
            or row['merge_val'].lower() != 'false'
            or (int(row['n_train']), int(row['n_val']), int(row['n_test']))
            != (count, 5400, 5400)
            or int(row['seed']) != 0 or int(row['bootstrap']) != 200
            or int(row['num_classes']) != 10 or row['bands'] != 'all'):
        raise ValueError('original backbone result has an unexpected protocol')
    if not 0 <= float(row['ci_lower']) <= float(row['metric_value']) <= float(row['ci_upper']) <= 1:
        raise ValueError('original backbone accuracy or confidence interval is invalid')


def scoreboard() -> list[dict]:
    local_path = ROOT / 'results/eurosat_models.csv'
    with local_path.open() as handle:
        local_rows = [r for r in csv.DictReader(handle) if r['split'] == 'test']
    local = {row['model']: row for row in local_rows}
    if len(local_rows) != len(LOCAL_MODELS) or set(local) != set(LOCAL_MODELS.values()):
        raise ValueError('expected one test result for each of the four local models')
    with (ROOT / 'results/article_claims.csv').open() as handle:
        claims = list(csv.DictReader(handle))
    rows = []
    for claim in claims:
        name = claim['claim']
        if claim['status'] == 'historical_note_only':
            continue
        if name in LOCAL_MODELS:
            source = local_path
            original = local[LOCAL_MODELS[name]]
            accuracy = original['accuracy']
            if (original['parameters'] != claim['parameters'] or int(original['images']) != 5400
                    or not 0 <= int(original['correct']) <= 5400
                    or abs(float(accuracy) - int(original['correct']) / 5400) > 1e-12):
                raise ValueError(f'{name}: local parameters, split size, or accuracy differs')
            kind = 'local_model_count_reproduced_from_raw_tiffs'
        elif name in BACKBONE_MODELS:
            source = original_path(BACKBONE_MODELS[name], 'eurosat', full=True)
            original = linear_rows(source)[16200]
            validate_backbone(original, 'eurosat', 16200)
            accuracy = original['metric_value']
            kind = 'original_uploaded_benchmark'
        else:
            raise ValueError(f'no source mapping for article claim {name!r}')
        percent = f'{100 * float(accuracy):.2f}'
        if percent != claim['test_accuracy_percent']:
            raise ValueError(f'{name}: original result {percent}% differs from the article')
        rows.append(dict(
            model=name, parameters_as_reported_in_article=claim['parameters'],
            test_accuracy=accuracy, test_accuracy_percent=percent, n_test=5400,
            source_kind=kind, source_csv=source.relative_to(ROOT).as_posix(),
        ))
    if len(rows) != 15 or len({row['model'] for row in rows}) != 15:
        raise ValueError('expected all 15 distinct article scoreboard models')
    return sorted(rows, key=lambda row: float(row['test_accuracy']))


def training_fractions() -> list[dict]:
    rows = []
    for name, filename, parameters in [
        ('Ours 33 features', 'eval_training_fractions_result.csv', '306'),
        ('ImageStats', 'eval_imagestats_fractions_result.csv', '477'),
    ]:
        source = ROOT / 'experiments' / filename
        with source.open() as handle:
            for row in csv.DictReader(handle):
                if row['seeds'] != '0;1;2;3;4':
                    raise ValueError('local curve must contain five training-subsample seeds')
                if (row['learned_parameters'] != parameters
                        or not 0 <= float(row['test_accuracy_mean']) <= 1
                        or not 0 <= float(row['test_accuracy_stdev']) <= 1):
                    raise ValueError('local curve has unexpected parameters or accuracy statistics')
                rows.append(dict(
                    model=name, split_protocol=row['split_protocol'],
                    train_fraction_percent=float(row['train_fraction_percent']),
                    n_train=int(row['n_train']), n_test=int(row['n_test']),
                    test_accuracy=row['test_accuracy_mean'],
                    test_accuracy_stdev=row['test_accuracy_stdev'],
                    ci_lower='', ci_upper='',
                    uncertainty='sample_stdev_over_5_training_seeds',
                    source_kind='original_local_learning_curve',
                    source_csv=source.relative_to(ROOT).as_posix(),
                ))
    labels = {key: label for label, key in BACKBONE_MODELS.items()}
    for model in CURVES:
        for dataset, protocol in [('eurosat', 'random'), ('eurosat-spatial', 'spatial')]:
            curve_path = original_path(model, dataset)
            curve = linear_rows(curve_path)
            for count in TRAIN_COUNTS:
                source = curve_path
                if count in curve:
                    original = curve[count]
                elif count == 16200 and dataset == 'eurosat':
                    # Four original random curves stop at 50%; use their
                    # full-data scoreboard runs, as the article figures do.
                    source = original_path(model, dataset, full=True)
                    original = linear_rows(source)[count]
                else:
                    raise ValueError(f'{curve_path}: missing training count {count}')
                validate_backbone(original, dataset, count)
                rows.append(dict(
                    model=labels[model], split_protocol=protocol,
                    train_fraction_percent=count * 100 / 16200, n_train=count, n_test=5400,
                    test_accuracy=original['metric_value'], test_accuracy_stdev='',
                    ci_lower=original['ci_lower'], ci_upper=original['ci_upper'],
                    uncertainty='95pct_test_bootstrap_200_resamples_seed0',
                    source_kind='original_uploaded_benchmark',
                    source_csv=source.relative_to(ROOT).as_posix(),
                ))
    keys = {(r['model'], r['split_protocol'], r['n_train']) for r in rows}
    expected = {
        (name, protocol, count)
        for name in ['Ours 33 features', 'ImageStats', *(labels[key] for key in CURVES)]
        for protocol in ('random', 'spatial') for count in TRAIN_COUNTS
    }
    if keys != expected or len(rows) != len(expected):
        raise ValueError('expected seven fractions for all nine models and both protocols')
    if any(row['n_test'] != 5400 or row['train_fraction_percent'] != row['n_train'] * 100 / 16200
           for row in rows):
        raise ValueError('learning-curve split sizes or fractions are inconsistent')
    return rows


def representation_tables() -> dict[str, list[dict]]:
    classification = read_rows(OVERLAP / 'classification.csv')
    comparisons = read_rows(OVERLAP / 'complementarity.csv')
    decoding = read_rows(OVERLAP / 'decoding.csv')
    features = read_rows(OVERLAP / 'features.csv')
    combined, explained, reverse, centered = [], [], [], []
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
            within = one_row(decoding, model=model, representation=rep, direction='h_to_z',
                             context='within_class', variant='original')
            centered.append({'model': label, 'features': short, 'r2_percent': float(within['native_r2']) * 100})
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
        'decoded_features.csv': reverse, 'class_centered_variance.csv': centered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/blog-results')
    parser.add_argument('--include-archive', action='store_true', help='include EarthLoc from the earlier scoreboard')
    args = parser.parse_args()
    table, curves = scoreboard(), training_fractions()
    if not args.include_archive:
        table = [row for row in table if row['model'] != 'torchgeo/earthloc_s2_resnet50']
    extra = representation_tables()
    for output_name, source_name in (
        ('features_33.csv', 'eurosat_306_features.csv'),
        ('class_accuracy.csv', 'eurosat_per_class.csv'),
        ('figure_examples.csv', 'figure_examples.csv'),
        ('figure_gradient_medians.csv', 'figure_gradient_medians.csv'),
    ):
        extra[output_name] = read_rows(ROOT / 'results' / source_name)
    baselines = ROOT / 'experiments/article_baselines'
    extra['supporting_results.csv'] = read_rows(baselines / 'measured/metrics.csv')
    extra['historical_claims.csv'] = read_rows(baselines / 'archive_manifest.csv')
    extra['archived_resisc33.csv'] = read_rows(baselines / 'archived/resisc45_33_full.csv')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output / 'scoreboard.csv', table)
    write(args.output / 'training_fractions.csv', curves)
    for name, rows in extra.items():
        write(args.output / name, rows)
    print(f'Exported {len(table)} scoreboard rows, {len(curves)} curve points, and {len(extra)} additional tables to {args.output}')


if __name__ == '__main__':
    main()
