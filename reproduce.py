#!/usr/bin/env python
"""Reproduce the EuroSAT article from TIFFs without historical feature caches."""
from __future__ import annotations

import os

# Set before numerical imports; L-BFGS stopping and the C sweep are thread-sensitive.
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import json
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import rasterio
import scipy
import sklearn
from sklearn.model_selection import StratifiedShuffleSplit

from src.data import CLASSES, DATA_ROOT, TIFF_BAND_NAMES, iter_images, list_split
from src.features import patch_features
from src.frontier import CORE_CONFIG, POOL_INDICES, RECIPE, frontier_features
from src.linmodel import fit_folded_logreg, predict, predict_reference_class, to_reference_class

ROOT = Path(__file__).resolve().parent
SPLITS = ('train', 'val', 'test')
SIZES = (16200, 5400, 5400)
DATA_URL = 'https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c'
CHECKSUMS = {
    'EuroSATallBands.zip': '751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59',
    'eurosat-train.txt': '1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695',
    'eurosat-val.txt': 'b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330',
    'eurosat-test.txt': 'cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592',
    'eurosat-spatial-train.txt': '2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f',
    'eurosat-spatial-val.txt': '6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b',
    'eurosat-spatial-test.txt': 'de22dec83d350cac3b3e4ca8e285cb6733c81ab94bf5bcf9213a567993402452',
}
FRACTIONS = (1, 2, 5, 10, 20, 50, 100)
BASELINE_CS = (
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
    1, 3, 10, 30, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 2000,
)
MODEL306 = ROOT / 'submissions/14_reference_class_96/model.npz'


def digest(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def download(path: Path) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with urlopen(f'{DATA_URL}/{path.name}', timeout=120) as response:
            with temporary.open('wb') as target:
                shutil.copyfileobj(response, target)
        if digest(temporary) != CHECKSUMS[path.name]:
            raise ValueError(f'checksum mismatch downloading {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data(allow_download: bool, spatial: bool) -> None:
    root = Path(DATA_ROOT)
    root.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        if name.endswith('.zip') or ('spatial' in name and not spatial):
            continue
        path = root / name
        if not path.exists() and allow_download:
            download(path)
        if not path.exists():
            raise FileNotFoundError(f'{path}; pass --download to obtain the data')
        if digest(path) != checksum:
            raise ValueError(f'checksum mismatch: {path}')
    paths = [path for split in SPLITS for path in list_split(split)[0]]
    if all(Path(path).is_file() for path in paths):
        return
    archive = root / 'EuroSATallBands.zip'
    if not archive.exists() and allow_download:
        download(archive)
    if not archive.exists():
        raise FileNotFoundError('EuroSAT TIFFs missing; pass --download')
    if digest(archive) != CHECKSUMS[archive.name]:
        raise ValueError('EuroSAT archive checksum mismatch')
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            if not (root / member.filename).resolve().is_relative_to(root.resolve()):
                raise ValueError('archive member escapes data directory')
        handle.extractall(root)
    if not all(Path(path).is_file() for path in paths):
        raise ValueError('archive did not supply every split image')


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def extract_features(batch_size: int, output: Path) -> dict:
    """Always read TIFFs, never data/cache or previously derived features."""
    result = {}
    feature_names = None
    for split, expected_size in zip(SPLITS, SIZES, strict=True):
        paths, labels = list_split(split)
        if len(paths) != expected_size or len(set(paths)) != expected_size:
            raise ValueError(f'{split} split membership is invalid')
        core_parts, frontier_parts, stats_parts = [], [], []
        for images, _ in iter_images(split, batch_size):
            core = patch_features(images, **CORE_CONFIG)
            frontier, names = frontier_features(images, core)
            if feature_names is not None and feature_names != names:
                raise ValueError('feature names changed between batches')
            feature_names = names
            flat = images.reshape(len(images), 13, -1)
            statistics = np.stack(
                (flat.mean(2), flat.std(2), flat.min(2), flat.max(2)), axis=2
            ).reshape(len(images), 52)
            core_parts.append(core[0])
            frontier_parts.append(frontier)
            stats_parts.append(statistics)
        result[split] = {
            'core': np.concatenate(core_parts),
            'frontier': np.concatenate(frontier_parts),
            'imagestats': np.concatenate(stats_parts),
            'labels': labels,
            'filenames': np.asarray([Path(path).name for path in paths]),
        }
        if any(not np.isfinite(result[split][key]).all()
               for key in ('core', 'frontier', 'imagestats')):
            raise ValueError(f'{split} features contain non-finite values')
        print(f'{split}: extracted {expected_size} TIFFs', flush=True)
    universe = np.concatenate([result[split]['filenames'] for split in SPLITS])
    if len(np.unique(universe)) != 27000:
        raise ValueError('random splits overlap')
    result['feature_names'] = feature_names
    write_csv(output / 'features_306.csv', [
        {'position': i, 'historical_pool_index': int(POOL_INDICES[i]) if i < 32 else 381,
         'feature_name': name}
        for i, name in enumerate(feature_names)
    ])
    return result


def fitted_reference(x: np.ndarray, y: np.ndarray, C: float) -> tuple:
    w, b, indices = fit_folded_logreg(x, y, C=C)
    wr, br = to_reference_class(w, b)
    return wr, br, indices


def prediction(features: np.ndarray, model: tuple) -> np.ndarray:
    return predict_reference_class(features, *model)


def evaluate(data: dict, output: Path, refit: bool) -> None:
    rows, per_class = [], []

    def record(name: str, split: str, features: np.ndarray, model: tuple) -> None:
        labels = data[split]['labels']
        predicted = prediction(features, model)
        row = dict(model=name, split=split, parameters=model[0].size + model[1].size,
                   images=len(labels), correct=int((predicted == labels).sum()),
                   accuracy=float((predicted == labels).mean()))
        rows.append(row)
        print(row, flush=True)
        for i, class_name in enumerate(CLASSES):
            mask = labels == i
            per_class.append(dict(model=name, split=split, class_name=class_name,
                                  images=int(mask.sum()),
                                  accuracy=float((predicted[mask] == labels[mask]).mean())))

    for folder, name in [('12_reference_class_linear', '171'), ('13_reference_class_95', '279')]:
        with np.load(ROOT / f'submissions/{folder}/model.npz', allow_pickle=False) as m:
            model = (m['W'], m['b'], m['feature_idx'])
            if int(m['ref_class']) != 0:
                raise ValueError('unexpected reference class')
            for key, value in CORE_CONFIG.items():
                if key in m and not np.array_equal(m[key], value):
                    raise ValueError(f'{folder}: different extraction configuration')
        for split in ('val', 'test'):
            record(name, split, data[split]['core'], model)

    if refit:
        model = fitted_reference(data['train']['frontier'], data['train']['labels'], 3.0)
        cfg = dict(CORE_CONFIG)
        np.savez_compressed(
            output / 'model_306.npz', W=model[0], b=model[1], feature_idx=model[2],
            params=306, k=33, C=3.0, ref_class=0, recipe=RECIPE, **cfg,
            feature_names=np.asarray(data['feature_names']),
            historical_pool_indices=np.append(POOL_INDICES, 381),
            tiff_band_names=np.asarray(TIFF_BAND_NAMES), legacy_swir_indices=[11, 12],
        )
    else:
        with np.load(MODEL306, allow_pickle=False) as m:
            if str(m['recipe']) != RECIPE or m['feature_names'].tolist() != data['feature_names']:
                raise ValueError('306 model recipe does not match feature schema')
            model = (m['W'], m['b'], m['feature_idx'])
    if model[0].size + model[1].size != 306:
        raise ValueError('306-value checkpoint has an incorrect parameter count')
    for split in ('val', 'test'):
        record('306', split, data[split]['frontier'], model)

    sweep = []
    best = None
    for C in BASELINE_CS:
        model = fitted_reference(data['train']['imagestats'], data['train']['labels'], C)
        acc = float((prediction(data['val']['imagestats'], model) == data['val']['labels']).mean())
        sweep.append(dict(C=C, validation_accuracy=acc))
        if best is None or acc > best[0]:
            best = (acc, C, model)
    write_csv(output / 'imagestats_c_sweep.csv', sweep)
    print(f'ImageStats validation-selected C={best[1]}', flush=True)
    for split in ('val', 'test'):
        record('imagestats', split, data[split]['imagestats'], best[2])
    write_csv(output / 'models.csv', rows)
    write_csv(output / 'per_class.csv', per_class)


def learning_curves(data: dict, output: Path) -> None:
    names = np.concatenate([data[s]['filenames'] for s in SPLITS])
    labels = np.concatenate([data[s]['labels'] for s in SPLITS])
    lookup = {name: i for i, name in enumerate(names)}
    spatial = {}
    for split, size in zip(SPLITS, SIZES, strict=True):
        path = Path(DATA_ROOT) / f'eurosat-spatial-{split}.txt'
        spatial[split] = np.asarray([
            lookup[Path(line.strip()).with_suffix('.tif').name]
            for line in path.read_text().splitlines() if line.strip()
        ])
        if len(spatial[split]) != size:
            raise ValueError('wrong spatial split size')
    if len(np.unique(np.concatenate(list(spatial.values())))) != 27000:
        raise ValueError('spatial splits overlap or omit samples')
    random = {'train': np.arange(16200), 'test': np.arange(21600, 27000)}
    for family, C, parameters in [('frontier', 3.0, 306), ('imagestats', 300.0, 477)]:
        features = np.concatenate([data[s][family] for s in SPLITS])
        rows = []
        for protocol, split in [('random', random), ('spatial', spatial)]:
            train, test = split['train'], split['test']
            for fraction in FRACTIONS:
                scores = []
                for seed in range(5):
                    n = round(len(train) * fraction / 100)
                    if fraction == 100:
                        selected = train
                    else:
                        selector = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
                        keep, _ = next(selector.split(np.zeros(len(train)), labels[train]))
                        selected = train[keep]
                    # Match the published curve's float32 full-logit arithmetic.
                    # Reference-class conversion is algebraic; rounding near a
                    # class tie can otherwise move an individual prediction.
                    w, b, indices = fit_folded_logreg(features[selected], labels[selected], C=C)
                    scores.append(float((predict(features[test], w, b, indices) == labels[test]).mean()))
                rows.append(dict(
                    model=family, parameters=parameters, C=C, split_protocol=protocol,
                    train_fraction_percent=fraction, n_train=n, n_test=len(test),
                    **{f'seed_{seed}_test_accuracy': score for seed, score in enumerate(scores)},
                    test_accuracy_mean=float(np.mean(scores)),
                    test_accuracy_stdev=float(np.std(scores, ddof=1)),
                ))
                print(f'{family} {protocol} {fraction}%: {np.mean(scores):.6f}', flush=True)
        write_csv(output / f'{family}_fractions.csv', rows)


def check_results(output: Path, fractions: bool) -> None:
    """Fail rather than silently accepting a different published result."""
    for generated, published in [
        ('models.csv', 'eurosat_models.csv'),
        ('per_class.csv', 'eurosat_per_class.csv'),
        ('features_306.csv', 'eurosat_306_features.csv'),
        ('imagestats_c_sweep.csv', 'imagestats_c_sweep.csv'),
    ]:
        with (ROOT / 'results' / published).open() as handle:
            expected = list(csv.DictReader(handle))
        with (output / generated).open() as handle:
            actual = list(csv.DictReader(handle))
        if actual != expected:
            raise ValueError(f'results differ from results/{published}')
    if fractions:
        for family, source in [('frontier', 'eval_training_fractions_result.csv'),
                               ('imagestats', 'eval_imagestats_fractions_result.csv')]:
            with (ROOT / 'experiments' / source).open() as handle:
                expected_rows = list(csv.DictReader(handle))
            with (output / f'{family}_fractions.csv').open() as handle:
                actual_rows = list(csv.DictReader(handle))
            if len(expected_rows) != len(actual_rows):
                raise ValueError(f'{family}: wrong number of curve rows')
            for wanted, got in zip(expected_rows, actual_rows, strict=True):
                if (wanted['split_protocol'] != got['split_protocol']
                        or float(wanted['train_fraction_percent']) != float(got['train_fraction_percent'])):
                    raise ValueError(f'{family}: mismatched split/fraction')
                metrics = [f'seed_{i}_test_accuracy' for i in range(5)]
                metrics += ['test_accuracy_mean', 'test_accuracy_stdev']
                for metric in metrics:
                    if abs(float(wanted[metric]) - float(got[metric])) > 1e-6:
                        raise ValueError(f'{family} {got["split_protocol"]} '
                                         f'{got["train_fraction_percent"]}%: {metric} differs')
    print('Published local model counts and requested learning curves reproduced.', flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--fractions', action='store_true', help='also refit both five-seed learning curves')
    parser.add_argument('--refit-306', action='store_true', help='write a train-only refit to output/model_306.npz')
    parser.add_argument('--check', action='store_true', help='fail if local results differ from checked-in article measurements')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'reproduce')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_data(args.download, args.fractions)
    metadata = {
        'recipe': RECIPE, 'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__,
        'rasterio': rasterio.__version__, 'blas_threads': 1,
        'data_url': DATA_URL, 'split_sha256': {k: v for k, v in CHECKSUMS.items() if k.endswith('.txt')},
        'source': 'raw TIFFs; no image or feature caches read',
    }
    (args.output / 'environment.json').write_text(json.dumps(metadata, indent=2) + '\n')
    data = extract_features(args.batch_size, args.output)
    evaluate(data, args.output, args.refit_306)
    if args.fractions:
        learning_curves(data, args.output)
    if args.check:
        check_results(args.output, args.fractions)


if __name__ == '__main__':
    main()
