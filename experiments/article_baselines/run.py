#!/usr/bin/env python
"""Refit article side baselines from original TIFFs/JPEGs, never old caches."""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
import scipy
import sklearn
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.article_baselines.rgb import (  # noqa: E402
    ARCHIVE_COMMIT, SELECTED_INDICES, rgb_feature_pool,
)
from src.data import CLASSES, TIFF_BAND_NAMES  # noqa: E402
from src.linmodel import (  # noqa: E402
    fit_folded_logreg, l1_rank, predict, predict_reference_class, to_reference_class,
)

SPLITS = ('train', 'val', 'test')
SIZES = {'eurosat': (16200, 5400, 5400), 'resisc45': (18900, 6300, 6300)}
SPLIT_SHA256 = {
    'eurosat': (
        '1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695',
        'b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330',
        'cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592',
    ),
    'resisc45': (
        'ecfa963be4d85eac83665f8be8634abcb4fe6f3546472cc0e87999e2cab4449b',
        '08d81f642526bec240589000af7f49a47e8d071a6e7925b0f36246a78ab64342',
        'e0927e80130b47317a2f18520d98382b6fc56f0d3edd3345140f7d02267c3805',
    ),
}
EUROSAT_CS = (
    .0001, .0003, .001, .003, .01, .03, .1, .3, 1, 3, 10, 30,
    100, 150, 200, 250, 300, 400, 500, 700, 1000, 2000,
)
RESISC_CS = (.01, .03, .1, .3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000)
PREPROCESS = {
    'eurosat': 'raw 13-band 64x64 uint16 TIFF; float32 statistics',
    'resisc45': (
        'rasterio/GDAL JPEG read out_shape=(3,64,64), Resampling.bilinear; '
        'uint8 quantization before float32 features; RGB order'
    ),
}
ARTICLE_ACCURACY = {
    ('eurosat', 'means'): 74.8,
    ('eurosat', 'mean_std'): 87.8,
    ('eurosat', 'imagestats'): 90.96,
    ('resisc45', 'imagestats'): 36.63,
    ('resisc45', 'selected33'): 59.06,
}


def file_digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def statistics(images: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Per-band mean/population std/min/max, in band-major order."""
    images = np.asarray(images, dtype=np.float32)
    flat = images.reshape(len(images), images.shape[1], -1)
    values = np.stack(
        (flat.mean(2), flat.std(2), flat.min(2), flat.max(2)), axis=2,
    ).reshape(len(images), -1)
    names = [
        f'{stat}_band{band}' for band in range(images.shape[1])
        for stat in ('mean', 'std', 'min', 'max')
    ]
    return values, names


def statistics_indices(bands: int, kind: str) -> np.ndarray:
    columns = {'means': (0,), 'mean_std': (0, 1), 'imagestats': (0, 1, 2, 3)}[kind]
    return np.asarray([4 * band + column for band in range(bands) for column in columns])


def dataset_lists(dataset: str, root: Path) -> tuple[dict, tuple[str, ...]]:
    """Check pinned split bytes, sizes, disjointness, class order and raw files."""
    if dataset == 'eurosat':
        image_root = root / 'ds/images/remote_sensing/otherDatasets/sentinel_2/tif'
        classes = tuple(CLASSES)
    else:
        image_root = root / 'NWPU-RESISC45'
        if not image_root.is_dir():
            raise FileNotFoundError(f'{image_root}: install the official raw RESISC45 data first')
        classes = tuple(sorted(p.name for p in image_root.iterdir() if p.is_dir()))
        if len(classes) != 45:
            raise ValueError(f'expected 45 RESISC45 class directories, found {len(classes)}')
    class_to_index = {name: i for i, name in enumerate(classes)}
    records, universe = {}, set()
    for split, count, checksum in zip(SPLITS, SIZES[dataset], SPLIT_SHA256[dataset], strict=True):
        path = root / f'{dataset}-{split}.txt'
        if not path.is_file():
            raise FileNotFoundError(f'{path}: see experiments/article_baselines/README.md')
        if file_digest(path) != checksum:
            raise ValueError(f'official split checksum mismatch: {path}')
        names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(names) != count or len(set(names)) != count or universe.intersection(names):
            raise ValueError(f'{split}: incorrect size, duplicate images or overlapping splits')
        universe.update(names)
        paths, labels = [], []
        for name in names:
            if Path(name).name != name or Path(name).suffix != '.jpg':
                raise ValueError(f'unsafe or unexpected image filename: {name!r}')
            class_name = Path(name).stem.rsplit('_', 1)[0]
            if class_name not in class_to_index:
                raise ValueError(f'unknown class: {class_name}')
            image_path = image_root / class_name / (
                Path(name).with_suffix('.tif').name if dataset == 'eurosat' else name
            )
            if not image_path.is_file():
                raise FileNotFoundError(f'raw image missing: {image_path}; caches are not accepted')
            paths.append(str(image_path))
            labels.append(class_to_index[class_name])
        records[split] = dict(
            paths=paths, labels=np.asarray(labels, dtype=np.int64),
            filenames=np.asarray(names), split_sha256=checksum,
        )
    return records, classes


def read_image(task: tuple[str, str]) -> np.ndarray:
    dataset, path = task
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', NotGeoreferencedWarning)
        with rasterio.open(path) as source:
            if dataset == 'resisc45':
                if (source.count, source.height, source.width) != (3, 256, 256):
                    raise ValueError(f'{path}: expected original 3-band 256x256 JPEG')
                image = source.read(out_shape=(3, 64, 64), resampling=Resampling.bilinear)
                if image.dtype != np.uint8:
                    raise ValueError(f'{path}: expected uint8 JPEG pixels')
            else:
                image = source.read()
                if image.shape != (13, 64, 64) or image.dtype != np.uint16:
                    raise ValueError(f'{path}: expected original 13-band 64x64 uint16 TIFF')
    return image


def extract_split(dataset: str, record: dict, batch_size: int, workers: int) -> dict:
    """Hash decoded inputs in split order; read no derived image/feature files."""
    parts, pool_parts, digest = [], [], hashlib.sha256()
    feature_names, pool_names = None, None
    executor = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for start in range(0, len(record['paths']), batch_size):
            tasks = [(dataset, path) for path in record['paths'][start:start + batch_size]]
            images = np.stack(list(
                executor.map(read_image, tasks, chunksize=16)
                if executor else map(read_image, tasks)
            ))
            digest.update(images.tobytes())
            features, feature_names = statistics(images)
            parts.append(features)
            if dataset == 'resisc45':
                pool, pool_names = rgb_feature_pool(images)
                pool_parts.append(pool)
    finally:
        if executor:
            executor.shutdown()
    result = dict(record, statistics=np.concatenate(parts), statistic_names=feature_names,
                  decoded_image_sha256=digest.hexdigest())
    if pool_parts:
        result.update(pool=np.concatenate(pool_parts), pool_names=pool_names)
    for key in ('statistics', 'pool'):
        if key in result and not np.isfinite(result[key]).all():
            raise ValueError(f'{key} contains non-finite values')
    return result


def choose_c(sweep: list[dict], n_val: int, one_se: bool = False) -> float:
    """Validation-only selection; a tie uses first C, or smallest C within 1 SE."""
    if not sweep:
        raise ValueError('C grid is empty')
    best = max(sweep, key=lambda row: row['validation_accuracy'])
    if not one_se:
        return best['C']
    accuracy = best['validation_accuracy']
    threshold = accuracy - np.sqrt(accuracy * (1 - accuracy) / n_val)
    return min(row['C'] for row in sweep if row['validation_accuracy'] >= threshold)


def fit_candidates(
    train: dict, val: dict, kind: str, indices: np.ndarray, grid: tuple,
    one_se: bool = False,
) -> tuple[tuple, list[dict], float]:
    key = 'pool' if kind == 'selected33' else 'statistics'
    models, sweep = {}, []
    for c in grid:
        model = fit_folded_logreg(train[key], train['labels'], indices, C=c)
        accuracy = float((predict(val[key], *model) == val['labels']).mean())
        models[c] = model
        sweep.append(dict(model=kind, C=c, validation_accuracy=accuracy))
        print(f'{kind} C={c:g} validation={accuracy:.6f}', flush=True)
    c = choose_c(sweep, len(val['labels']), one_se)
    return models[c], sweep, c


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def output_directory(dataset: str, requested: Path | None) -> Path:
    parent = (ROOT / 'output').resolve()
    output = (requested or parent / 'article-baselines' / dataset).resolve()
    if output == parent or not output.is_relative_to(parent):
        raise ValueError('outputs must be inside output/; historical artifacts are protected')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; choose a new --output-dir')
    output.mkdir(parents=True, exist_ok=True)
    return output


def check_results(output: Path, dataset: str) -> None:
    """Compare a new run with the frozen measured reference, not missing originals."""
    with (Path(__file__).parent / 'measured/metrics.csv').open(newline='') as handle:
        expected = {
            (row['model'], row['split']): row
            for row in csv.DictReader(handle) if row['dataset'] == dataset
        }
    with (output / 'metrics.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    actual = {(row['model'], row['split']): row for row in rows}
    if (len(actual) != len(rows) or actual.keys() != expected.keys()
            or any(row['dataset'] != dataset for row in rows)):
        raise ValueError('reference result model/split membership differs')
    for key, row in expected.items():
        for field in ('features', 'learned_parameters', 'C', 'n_images', 'correct', 'accuracy'):
            if float(actual[key][field]) != float(row[field]):
                raise ValueError(f'reference results differ: {dataset} {key} {field}')
    print(f'{dataset}: matches frozen measured reference (not missing historical fits)', flush=True)


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    dataset = args.dataset
    root = args.data_root or ROOT / 'data' / ('EuroSAT' if dataset == 'eurosat' else 'RESISC45')
    records, classes = dataset_lists(dataset, root)
    output = output_directory(dataset, args.output_dir)
    data = {}
    for split in ('train', 'val'):
        data[split] = extract_split(dataset, records[split], args.batch_size, args.workers)
        print(f'{dataset} {split}: {len(records[split]["labels"])} raw images', flush=True)
    models, sweeps = {}, []
    kinds = ('means', 'mean_std', 'imagestats') if dataset == 'eurosat' else ('imagestats', 'selected33')
    for kind in kinds:
        if kind == 'selected33':
            indices = (
                l1_rank(data['train']['pool'], data['train']['labels'])[:33]
                if args.reselect_resisc33 else np.asarray(SELECTED_INDICES)
            )
            grid = RESISC_CS if args.reselect_resisc33 else (30.0,)
            status = 'new_selection' if args.reselect_resisc33 else 'historical_recipe_refit'
        else:
            indices = statistics_indices(13 if dataset == 'eurosat' else 3, kind)
            grid = EUROSAT_CS if dataset == 'eurosat' else RESISC_CS
            status = (
                'historical_recipe_refit' if dataset == 'eurosat' and kind == 'imagestats'
                else 'new_reference_missing_historical_fit_metadata'
            )
        model, sweep, c = fit_candidates(
            data['train'], data['val'], kind, indices, grid,
            one_se=kind == 'selected33' and args.reselect_resisc33,
        )
        models[kind] = (model, c, status)
        sweeps.extend(sweep)
    # All fitted models and selection decisions are frozen before test extraction.
    data['test'] = extract_split(dataset, records['test'], args.batch_size, args.workers)
    print(f'{dataset} test: {len(records["test"]["labels"])} raw images', flush=True)
    rows, predictions, schema = [], [], []
    for kind, (model, c, status) in models.items():
        w, b, indices = model
        wr, br = to_reference_class(w, b)
        key = 'pool' if kind == 'selected33' else 'statistics'
        names = data['train']['pool_names' if kind == 'selected33' else 'statistic_names']
        selected_names = [names[i] for i in indices]
        parameters = int(wr.size + br.size)
        if parameters != (len(classes) - 1) * (len(indices) + 1):
            raise ValueError('incorrect reference-class parameter count')
        np.savez_compressed(
            output / f'{kind}.npz', W=wr, b=br, feature_idx=indices,
            feature_names=np.asarray(selected_names), C=c, ref_class=0,
            classes=np.asarray(classes), dataset=dataset, preprocessing=PREPROCESS[dataset],
            feature_recipe='archived_rgb147' if kind == 'selected33' else 'band_mean_std_min_max',
            source_commit=ARCHIVE_COMMIT if dataset == 'resisc45' and kind == 'selected33' else '',
            provenance_status=status, split_sha256=np.asarray(SPLIT_SHA256[dataset]),
        )
        schema.extend(dict(model=kind, position=i, pool_index=int(index), feature_name=names[index])
                      for i, index in enumerate(indices))
        for split in ('val', 'test'):
            predicted = predict_reference_class(data[split][key], wr, br, indices)
            if not np.array_equal(predicted, predict(data[split][key], *model)):
                raise ValueError('float32 reference-class conversion changed predictions')
            labels = data[split]['labels']
            accuracy = float((predicted == labels).mean())
            quoted = ARTICLE_ACCURACY[(dataset, kind)]
            rows.append(dict(
                dataset=dataset, model=kind, split=split, features=len(indices),
                learned_parameters=parameters, C=c, n_images=len(labels),
                correct=int((predicted == labels).sum()), accuracy=accuracy,
                article_test_accuracy_percent=quoted, provenance_status=status,
                difference_from_article_pp=100 * accuracy - quoted if split == 'test' else '',
            ))
            predictions.extend(dict(
                model=kind, split=split, filename=str(name), label=int(label), prediction=int(pred),
            ) for name, label, pred in zip(
                data[split]['filenames'], labels, predicted, strict=True,
            ))
            print(f'{kind} {split}: {100 * accuracy:.4f}% ({parameters} learned values)', flush=True)
    write_csv(output / 'metrics.csv', rows)
    write_csv(output / 'validation_sweep.csv', sweeps)
    write_csv(output / 'predictions.csv', predictions)
    write_csv(output / 'features.csv', schema)
    metadata = dict(
        dataset=dataset, command=sys.argv, data_root=str(root.resolve()), classes=classes,
        preprocessing=PREPROCESS[dataset], raw_inputs_only=True, test_used_for_selection=False,
        resisc33_selection=(
            ('train_L1_then_validation_one_SE' if args.reselect_resisc33 else 'archived_top33_C30')
            if dataset == 'resisc45' else None
        ),
        split_sha256=dict(zip(SPLITS, SPLIT_SHA256[dataset], strict=True)),
        decoded_image_sha256={split: data[split]['decoded_image_sha256'] for split in SPLITS},
        python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        sklearn=sklearn.__version__, rasterio=rasterio.__version__, gdal=rasterio.__gdal_version__,
        physical_band_order=TIFF_BAND_NAMES if dataset == 'eurosat' else ['red', 'green', 'blue'],
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    with (output / 'environment.json').open('x') as handle:
        json.dump(metadata, handle, indent=2)
        handle.write('\n')
    print(f'wrote {output} in {metadata["elapsed_seconds"]:.1f}s', flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all', 'eurosat', 'resisc45'), default='all')
    parser.add_argument('--data-root', type=Path,
                        help='dataset root, or parent of EuroSAT/ and RESISC45/ with --dataset all')
    parser.add_argument('--output', '--output-dir', dest='output_dir', type=Path,
                        help='new directory inside output/; both datasets get separate subdirectories')
    parser.add_argument('--download', action='store_true',
                        help='download missing official data only; verify checksums and preserve existing files')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--reselect-resisc33', action='store_true',
                        help='rerun train-only L1 top33 and the archived validation one-SE C sweep')
    parser.add_argument('--check', action='store_true',
                        help='check against measured/metrics.csv, not the missing historical fits')
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 1:
        parser.error('--batch-size and --workers must be positive')
    if args.reselect_resisc33 and args.dataset == 'eurosat':
        parser.error('--reselect-resisc33 requires --dataset resisc45 or all')
    datasets = ('eurosat', 'resisc45') if args.dataset == 'all' else (args.dataset,)
    base = output_directory('all', args.output_dir) if args.dataset == 'all' else None
    combined = []
    for dataset in datasets:
        current = argparse.Namespace(**vars(args))
        current.dataset = dataset
        current.reselect_resisc33 = args.reselect_resisc33 and dataset == 'resisc45'
        if base is not None:
            current.output_dir = base / dataset
            current.data_root = (
                args.data_root / ('EuroSAT' if dataset == 'eurosat' else 'RESISC45')
                if args.data_root else None
            )
        if args.download:
            from experiments.article_baselines.download import prepare_dataset
            root = current.data_root or ROOT / 'data' / (
                'EuroSAT' if dataset == 'eurosat' else 'RESISC45'
            )
            prepare_dataset(dataset, root, SPLIT_SHA256[dataset])
        output = run(current)
        if args.check:
            check_results(output, dataset)
        with (output / 'metrics.csv').open(newline='') as handle:
            combined.extend(csv.DictReader(handle))
    if base is not None:
        write_csv(base / 'metrics.csv', combined)
        shutil.copyfile(Path(__file__).parent / 'archive_manifest.csv', base / 'archive_manifest.csv')


if __name__ == '__main__':
    main()
