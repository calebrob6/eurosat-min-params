"""Raw handcrafted extraction and filename-aligned, authenticated embeddings."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from reproduce import CHECKSUMS, prepare_data
from src.data import CLASSES, DATA_ROOT, TIFF_BAND_NAMES, iter_images, list_split
from src.feature_pool import historical_pool_features
from src.features import patch_features
from src.frontier import CORE_CONFIG, POOL_INDICES, frontier_features

from .protocol import (
    BENCH_REVISION,
    MODELS,
    REPRESENTATIONS,
    ROOT,
    SIZES,
    json_value,
    save_npz,
    sha256,
    write_json,
)

logger = logging.getLogger(__name__)
SOURCE_FILES = (
    'src/data.py', 'src/features.py', 'src/feature_pool.py', 'src/frontier.py',
    'src/extra_features.py', 'experiments/representation_overlap/data.py',
)


def canonical_names(split: str) -> tuple[np.ndarray, np.ndarray]:
    paths, labels = list_split(split)
    names = np.asarray([Path(path).name for path in paths])
    if len(names) != SIZES[split] or len(set(names)) != len(names):
        raise ValueError(f'{split}: invalid canonical membership')
    return names, labels


def align(
    features: np.ndarray, names: np.ndarray, labels: np.ndarray,
    expected_names: np.ndarray, expected_labels: np.ndarray,
) -> np.ndarray:
    if features.ndim != 2 or names.ndim != 1 or labels.ndim != 1:
        raise ValueError('expected a feature matrix and one-dimensional sample IDs/labels')
    if len(features) != len(names) or len(labels) != len(names):
        raise ValueError('feature, ID, and label lengths differ')
    if len(set(names.tolist())) != len(names):
        raise ValueError('duplicate sample IDs')
    if len(set(expected_names.tolist())) != len(expected_names):
        raise ValueError('duplicate canonical sample IDs')
    if set(names.tolist()) != set(expected_names.tolist()):
        raise ValueError('sample IDs differ from canonical split membership')
    lookup = {name: index for index, name in enumerate(names.tolist())}
    indices = np.asarray([lookup[name] for name in expected_names.tolist()])
    if not np.array_equal(labels[indices], expected_labels):
        raise ValueError('sample labels disagree after filename alignment')
    result = features[indices]
    if not np.isfinite(result).all():
        raise ValueError('non-finite feature matrix')
    return result


def feature_matrices(images: np.ndarray, *, b10_zeroed: bool = False) -> tuple[dict, dict]:
    if images.dtype != np.float32 or images.ndim != 4 or images.shape[1:] != (13, 64, 64):
        raise ValueError('expected float32 TIFF-order images of shape (N,13,64,64)')
    if b10_zeroed:
        images = images.copy()
        images[:, TIFF_BAND_NAMES.index('B10')] = 0
    core = patch_features(images, **CORE_CONFIG)
    if core[0].shape[1] != 320:
        raise ValueError('core schema is no longer 320 columns')
    pool, names, groups = historical_pool_features(images, core)
    frontier, frontier_names = frontier_features(images, core)
    np.testing.assert_array_equal(frontier[:, :32], pool[:, POOL_INDICES])
    flat = images.reshape(len(images), 13, -1)
    stats = np.stack((flat.mean(2), flat.std(2), flat.min(2), flat.max(2)), axis=2).reshape(len(images), 52)
    matrices = {'frontier33': frontier, 'pool377': pool, 'imagestats52': stats}
    schema = {
        'frontier33': {
            'names': frontier_names,
            'families': [groups[index] for index in POOL_INDICES] + ['region_shape'],
            'pool_indices': [int(x) for x in POOL_INDICES] + [None],
        },
        'pool377': {'names': names, 'families': groups, 'pool_indices': list(range(377))},
        'imagestats52': {
            'names': [f'{stat}_{band}' for band in TIFF_BAND_NAMES for stat in ('mean', 'std', 'min', 'max')],
            'families': ['image_statistics'] * 52,
            'pool_indices': [None] * 52,
        },
    }
    for key, width in REPRESENTATIONS.items():
        if matrices[key].shape != (len(images), width) or not np.isfinite(matrices[key]).all():
            raise ValueError(f'{key}: invalid handcrafted features')
    return matrices, schema


def extraction_identity() -> dict:
    return json_value({
        'schema': 'raw-handcrafted-overlap-v1',
        'source_sha256': {path: sha256(ROOT / path) for path in SOURCE_FILES},
        'core_config': CORE_CONFIG,
        'physical_bands': TIFF_BAND_NAMES,
        'legacy_index_bands': {'swir1': 'B12', 'swir2': 'B8A'},
        'split_sha256': {name: digest for name, digest in CHECKSUMS.items()
                         if name.startswith('eurosat-') and not name.startswith('eurosat-spatial-')},
        'variants': ['original', 'b10_zeroed'],
        'numpy': np.__version__,
    })


def prepare(cache: Path, batch_size: int = 128, *, download: bool = False) -> None:
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    prepare_data(download, False)
    identity = extraction_identity()
    manifest_path = cache / 'features.json'
    if manifest_path.exists():
        load_feature_manifest(cache)
        logger.info('reusing authenticated handcrafted cache %s', cache)
        return
    cache.mkdir(parents=True, exist_ok=True)
    existing = list(cache.glob('*.npz'))
    if existing:
        raise ValueError(f'incomplete cache {cache}: choose a new directory; existing files are not overwritten')
    all_names = []
    schema = None
    for split in SIZES:
        names, labels = canonical_names(split)
        all_names.extend(names.tolist())
        values = {variant: {rep: [] for rep in REPRESENTATIONS}
                  for variant in ('original', 'b10_zeroed')}
        seen = 0
        for images, batch_labels in iter_images(split, batch_size):
            if not np.array_equal(batch_labels, labels[seen:seen + len(images)]):
                raise ValueError('TIFF iterator labels changed')
            for variant in values:
                matrices, current = feature_matrices(images, b10_zeroed=variant == 'b10_zeroed')
                if schema is not None and current != schema:
                    raise ValueError('feature names or family order changed between batches')
                schema = current
                for rep in REPRESENTATIONS:
                    values[variant][rep].append(matrices[rep])
            seen += len(images)
            if seen % (batch_size * 10) == 0:
                logger.info('handcrafted %s: %d/%d', split, seen, len(names))
        if seen != len(names):
            raise ValueError(f'{split}: TIFF iterator omitted samples')
        for variant, reps in values.items():
            save_npz(cache / f'{variant}_{split}.npz', filenames=names, labels=labels,
                     **{rep: np.concatenate(parts) for rep, parts in reps.items()})
    if len(set(all_names)) != sum(SIZES.values()):
        raise ValueError('canonical splits overlap')
    with np.load(ROOT / 'models/eurosat_33.npz', allow_pickle=False) as model:
        if schema['frontier33']['names'] != model['feature_names'].tolist():
            raise ValueError('frontier names differ from the published checkpoint')
    write_json(manifest_path, {
        'identity': identity,
        'schema': schema,
        'files': {path.name: sha256(path) for path in sorted(cache.glob('*.npz'))},
        'classes': CLASSES,
        'raw_source': str(Path(DATA_ROOT)),
    })


def load_feature_manifest(cache: Path) -> dict:
    manifest = json.loads((cache / 'features.json').read_text())
    if manifest['identity'] != extraction_identity():
        raise ValueError('handcrafted generating source/configuration changed; build a new cache')
    expected = {f'{variant}_{split}.npz' for variant in ('original', 'b10_zeroed') for split in SIZES}
    if set(manifest['files']) != expected:
        raise ValueError('handcrafted cache manifest has missing or extra splits')
    for name, digest in manifest['files'].items():
        if sha256(cache / name) != digest:
            raise ValueError(f'handcrafted checksum mismatch: {cache / name}')
    return manifest


def load_handcrafted(cache: Path, variant: str, splits: tuple[str, ...]) -> dict:
    if variant not in ('original', 'b10_zeroed') or any(split not in SIZES for split in splits):
        raise ValueError('unknown handcrafted variant or split')
    result = {}
    for split in splits:
        names, labels = canonical_names(split)
        with np.load(cache / f'{variant}_{split}.npz', allow_pickle=False) as handle:
            reps = {
                rep: align(handle[rep], handle['filenames'], handle['labels'], names, labels)
                for rep in REPRESENTATIONS
            }
        for rep, width in REPRESENTATIONS.items():
            if reps[rep].shape[1] != width:
                raise ValueError(f'{variant}/{split}/{rep}: unexpected feature width')
        result[split] = {**reps, 'filenames': names, 'labels': labels}
    return result


def backbone_metadata(model: str, cache: Path, metadata_dir: Path) -> dict:
    if model not in MODELS:
        raise ValueError(f'unknown study backbone: {model}')
    metadata = json.loads((metadata_dir / f'{model}_eurosat.json').read_text())
    if (metadata['model_key'] != model or metadata['dataset'] != 'eurosat'
            or metadata['torchgeo_bench_revision'] != BENCH_REVISION):
        raise ValueError(f'{model}: backbone configuration does not match study')
    for split in SIZES:
        name = f'eurosat-{split}.txt'
        if metadata['dataset_split_sha256'][name] != CHECKSUMS[name]:
            raise ValueError(f'{model}: wrong {split} split checksum')
    expected_bands = [band for band in TIFF_BAND_NAMES if not (model.startswith('olmoearth') and band == 'B10')]
    if metadata['used_bands'] != expected_bands:
        raise ValueError(f'{model}: unexpected physical bands')
    if sha256(cache / f'{model}_eurosat.npz') != metadata['embedding_sha256']:
        raise ValueError(f'{model}: embedding checksum mismatch')
    return metadata


def load_backbone(model: str, cache: Path, splits: tuple[str, ...]) -> dict:
    if model not in MODELS or any(split not in SIZES for split in splits):
        raise ValueError('unknown backbone or split')
    result = {}
    with np.load(cache / f'{model}_eurosat.npz', allow_pickle=False) as handle:
        for split in splits:
            names, labels = canonical_names(split)
            features = align(handle[f'{split}_features'], handle[f'{split}_filenames'],
                             handle[f'{split}_labels'], names, labels)
            if features.shape != (SIZES[split], MODELS[model]):
                raise ValueError(f'{model}/{split}: unexpected embedding dimensions')
            result[split] = {'features': features, 'filenames': names, 'labels': labels}
    return result
