"""Raw handcrafted extraction and filename-aligned, authenticated embeddings."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch

from experiments.data import CHECKSUMS, CLASSES, DATA_ROOT, default_device, iter_images, list_split, prepare_data
from patch_features import EUROSAT_377, EUROSAT_BANDS, EuroSATFeatures

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
SOURCE_FILES = ('patch_features.py', 'experiments/data.py', 'experiments/representation_overlap/data.py')
EXTRACTORS = {'frontier33': '33', 'pool377': '377', 'imagestats52': '52'}


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


def feature_matrices(
    images: np.ndarray, extractors: dict, device: str, *, b10_zeroed: bool = False
) -> tuple[dict, dict]:
    if images.ndim != 4 or images.shape[1:] != (13, 64, 64):
        raise ValueError('expected TIFF-order images of shape (N,13,64,64)')
    x = torch.from_numpy(np.ascontiguousarray(images, dtype=np.float32)).to(device)
    if b10_zeroed:
        x = x.clone()
        x[:, EUROSAT_BANDS.index('B10')] = 0
    matrices = {rep: extractor(x).cpu().numpy() for rep, extractor in extractors.items()}
    schema = {
        rep: {
            'names': list(extractor.feature_names),
            'families': list(extractor.feature_families),
            'pool_indices': [EUROSAT_377.index(n) if n in EUROSAT_377 else None for n in extractor.feature_names]
            if rep != 'imagestats52' else [None] * 52,
        }
        for rep, extractor in extractors.items()
    }
    schema['pool377']['pool_indices'] = list(range(377))
    for key, width in REPRESENTATIONS.items():
        if matrices[key].shape != (len(images), width) or not np.isfinite(matrices[key]).all():
            raise ValueError(f'{key}: invalid handcrafted features')
    return matrices, schema


def extraction_identity() -> dict:
    return json_value({
        'schema': 'raw-handcrafted-overlap-v2',
        'source_sha256': {path: sha256(ROOT / path) for path in SOURCE_FILES},
        'physical_bands': EUROSAT_BANDS,
        'legacy_index_bands': {'swir1': 'B12', 'swir2': 'B8A'},
        'split_sha256': {name: digest for name, digest in CHECKSUMS.items()
                         if name.startswith('eurosat-') and not name.startswith('eurosat-spatial-')},
        'variants': ['original', 'b10_zeroed'],
        'torch': torch.__version__,
    })


def prepare(cache: Path, batch_size: int = 128, *, download: bool = False, device: str | None = None) -> None:
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    prepare_data(download)
    device = device or default_device()
    extractors = {rep: EuroSATFeatures(key).to(device) for rep, key in EXTRACTORS.items()}
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
                matrices, current = feature_matrices(
                    images, extractors, device, b10_zeroed=variant == 'b10_zeroed'
                )
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
    expected_bands = [band for band in EUROSAT_BANDS if not (model.startswith('olmoearth') and band == 'B10')]
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
