#!/usr/bin/env python
"""Fresh EuroSAT backbone measurements using pinned TorchGeo-bench APIs.

Run from the repository root with output/benchmark-env/bin/python. The upstream
checkout is unmodified. We reuse its model configs, dataset resize/band logic,
feature extractor, logistic C sweep, final refit and bootstrap. Fraction subsets
follow the imported original runner: a nested seed-0 permutation of embeddings
from the shuffled training loader. Filename order is recorded, not assumed.
"""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
import torchgeo_bench
from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.main import embed_split, evaluate_logistic, resolve_model_config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.data import CHECKSUMS, DATA_ROOT, prepare_data

REVISION = '9c8e4afab46675d7279c88828dfcbf0ca99b3a07'
SOURCE = ROOT / 'output/torchgeo-bench'
RESULTS = ROOT / 'output/backbones'
MODELS = {
    'resnet50': 'timm/resnet50',
    'convnext_tiny': 'timm/convnext_tiny',
    'dofa_large': 'torchgeo/dofa_large',
    'olmoearth_nano': 'olmoearth_v1_2_nano',
    'olmoearth_base': 'olmoearth_v1_2_base',
    'resnet18': 'timm/resnet18',
    'vit_base': 'timm/vit/vit_base_patch16_224',
    'dofa_base': 'torchgeo/dofa_base',
    'olmoearth_small': 'olmoearth_v1_2_small',
    'earthloc_s2_resnet50': 'torchgeo/earthloc_s2_resnet50',
    'resnet50_s2all_moco': 'torchgeo/resnet50_s2_all_moco',
}
DEFAULT_MODELS = ('resnet50', 'convnext_tiny', 'dofa_large', 'olmoearth_nano', 'olmoearth_base')
FRACTIONS = (1, 2, 5, 10, 20, 50, 100)
SPLITS = ('train', 'val', 'test')
SAMPLING = 'nested_random_permutation_of_shuffled_embeddings_v1'
DEFAULT_BATCH = {
    'resnet50': 64, 'convnext_tiny': 64, 'dofa_large': 16,
    'olmoearth_nano': 32, 'olmoearth_base': 32,
    'resnet18': 64, 'vit_base': 64, 'dofa_base': 16, 'olmoearth_small': 32,
    'earthloc_s2_resnet50': 64, 'resnet50_s2all_moco': 64,
}
logger = logging.getLogger(__name__)


class IndexedDataset(Dataset):
    """Add source indices without changing images, labels, or random state."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], '_source_index': index}


class RecordingLoader:
    """Record the upstream sampler order while using upstream extraction."""

    def __init__(self, loader):
        self.indices = []
        self.dataset = loader.dataset
        self.loader = DataLoader(
            IndexedDataset(loader.dataset), batch_size=loader.batch_size,
            sampler=loader.sampler, num_workers=loader.num_workers,
            pin_memory=loader.pin_memory, drop_last=loader.drop_last,
            persistent_workers=loader.persistent_workers,
            generator=loader.generator, worker_init_fn=loader.worker_init_fn,
        )

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        for batch in self.loader:
            self.indices.extend(batch['_source_index'].tolist())
            yield batch


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def check_source() -> None:
    sha = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if sha != REVISION or not Path(torchgeo_bench.__file__).resolve().is_relative_to(SOURCE):
        raise RuntimeError('install the pinned output/torchgeo-bench checkout in the benchmark environment')
    dirty = subprocess.check_output(['git', '-C', str(SOURCE), 'diff', '--name-only'], text=True)
    if dirty.strip():
        raise RuntimeError('benchmark source must be unmodified')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def environment() -> dict:
    packages = {
        name: importlib.metadata.version(name)
        for name in ('torch', 'torchvision', 'torchgeo', 'timm', 'numpy',
                     'scipy', 'scikit-learn', 'olmoearth-pretrain-minimal',
                     'huggingface-hub', 'torchmetrics', 'omegaconf')
    }
    return {
        'torchgeo_bench_revision': REVISION, 'python': platform.python_version(),
        'packages': packages, 'gpu': torch.cuda.get_device_name(0),
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'dataset_split_sha256': {k: v for k, v in CHECKSUMS.items() if k.endswith('.txt')},
    }


def extract(model_key: str, dataset: str, device: str, workers: int, batch_size: int,
            output: Path, cache: Path, fresh: bool) -> tuple[dict, dict]:
    cache_path = cache / f'{model_key}_{dataset}.npz'
    meta_path = output / f'{model_key}_{dataset}.json'
    cfg = compose_config([
        f'model={MODELS[model_key]}', f'dataset.names=[{dataset}]',
        'dataset.bands=all', 'dataset.image_size=224',
        'dataset.normalization=bandspec_zscore', 'eval.merge_val=false', 'seed=0',
    ])
    model_cfg = resolve_model_config(cfg.model, dataset)
    image_size = model_cfg.get('image_size', cfg.dataset.image_size)
    interpolation = model_cfg.get('interpolation', cfg.dataset.interpolation)
    resolved = OmegaConf.to_container(model_cfg, resolve=True)
    if not fresh and cache_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text())
        if (metadata['torchgeo_bench_revision'] != REVISION or metadata['model_config'] != resolved
                or metadata.get('sampling') != SAMPLING or metadata['batch_size'] != batch_size
                or metadata['packages'] != environment()['packages']):
            raise ValueError('cached embedding configuration differs')
        if sha256(cache_path) != metadata['embedding_sha256']:
            raise ValueError(f'embedding checksum differs: {cache_path}')
        with np.load(cache_path, allow_pickle=False) as handle:
            return {key: handle[key] for key in handle.files}, metadata

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_float32_matmul_precision('highest')
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset, return_val=True, bands='all', partition_name='default',
        image_size=image_size, interpolation=interpolation,
        batch_size=batch_size, num_workers=workers,
    )
    bench = get_bench_dataset_class(dataset)()
    specs = bench.select_band_specs(None)
    model_cfg.pop('interpolation', None)
    model = instantiate(model_cfg, bands=specs, normalization='bandspec_zscore')
    model.to(device).eval()
    if model_key.startswith('olmoearth'):
        used_bands = [b.source_name for b in specs if b.source_name != 'B10']
    else:
        used_bands = [b.source_name for b in specs]
    metadata = {
        **environment(), 'model_key': model_key, 'model_config_name': MODELS[model_key],
        'model_config': resolved, 'dataset': dataset,
        'requested_bands': [b.source_name for b in specs], 'used_bands': used_bands,
        'normalization': 'bandspec_zscore (model-specific normalization overrides apply)',
        'image_size': image_size, 'interpolation': interpolation,
        'backbone_parameters': sum(p.numel() for p in model.parameters()),
        'backbone_state_sha256': state_hash(model),
        'batch_size': batch_size, 'num_workers': workers,
        'sampling': SAMPLING,
        'embedding_order': 'upstream train loader shuffle=True; validation/test shuffle=False; filenames recorded',
        'embedding_matmul_precision': 'highest',
    }
    data = {}
    start = time.monotonic()
    for split, original_loader in zip(SPLITS, [train_loader, val_loader, test_loader], strict=True):
        dataset_object = original_loader.dataset
        loader = RecordingLoader(original_loader)
        embeddings, labels = embed_split(model, loader, torch.device(device), False, split)
        paths = [Path(dataset_object.samples[i][0]).name for i in loader.indices]
        if embeddings.shape[0] != len(paths) or not np.isfinite(embeddings).all():
            raise ValueError(f'{model_key}/{dataset}/{split}: invalid embedding matrix')
        np.testing.assert_array_equal(labels, [dataset_object.samples[i][1] for i in loader.indices])
        if len(set(paths)) != len(dataset_object):
            raise ValueError('sampler duplicated or omitted an image')
        data[f'{split}_features'] = embeddings
        data[f'{split}_labels'] = labels
        data[f'{split}_filenames'] = np.asarray(paths)
        logger.info('%s %s %s: %s', model_key, dataset, split, embeddings.shape)
    metadata['extraction_seconds'] = time.monotonic() - start
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **data)
    temporary.replace(cache_path)
    metadata['embedding_sha256'] = sha256(cache_path)
    output.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + '\n')
    del model
    torch.cuda.empty_cache()
    return data, metadata


def evaluate(model_key: str, dataset: str, data: dict, metadata: dict,
             fractions: tuple[int, ...], device: str, output: Path, *, nested: bool) -> None:
    result_path = output / f'{model_key}_{dataset}.csv'
    n_train = len(data['train_labels'])
    if (n_train, len(data['val_labels']), len(data['test_labels'])) != (16200, 5400, 5400):
        raise ValueError('unexpected EuroSAT split sizes')
    rows = []
    permutation = np.random.default_rng(0).permutation(n_train)
    for fraction in fractions:
        count = round(n_train * fraction / 100)
        # Original non-Olmo scoreboard rows used the loader order directly;
        # fraction sweeps (including Olmo's later 100% rows) permuted it.
        selected = permutation[:count] if nested else np.arange(n_train)
        if not nested and fraction != 100:
            raise ValueError('only a full-data scoreboard can use unpermuted loader order')
        start = time.monotonic()
        metric, lo, hi, best_c, _, _ = evaluate_logistic(
            data['train_features'][selected], data['train_labels'][selected],
            data['val_features'], data['val_labels'],
            data['test_features'], data['test_labels'],
            c_values=10 ** np.linspace(-6, 4, 40), seed=0, n_bootstrap=200,
            merge_val=False, device=device, temp_scale=False,
        )
        selected_names = data['train_filenames'][selected]
        subset_hash = hashlib.sha256('\n'.join(selected_names.tolist()).encode()).hexdigest()
        dim = data['train_features'].shape[1]
        rows.append(dict(
            model=model_key, model_config=MODELS[model_key], dataset=dataset,
            method='linear', metric_name='accuracy', train_fraction_percent=fraction,
            sampling=SAMPLING if nested else 'full_training_loader_order',
            n_train=count, n_val=5400, n_test=5400, seed=0, bootstrap=200,
            merge_val=False, normalization=metadata['normalization'],
            image_size=metadata['image_size'] or 64, bands=';'.join(metadata['used_bands']),
            feature_dim=dim, backbone_parameters=metadata['backbone_parameters'],
            head_parameters=10 * (dim + 1), best_c=best_c,
            metric_value=metric, ci_lower=lo, ci_upper=hi,
            c_range_start=-6, c_range_stop=4, c_range_num=40,
            training_subset_sha256=subset_hash,
            backbone_state_sha256=metadata['backbone_state_sha256'],
            embedding_sha256=metadata['embedding_sha256'],
            torchgeo_bench_revision=REVISION, seconds=round(time.monotonic() - start, 3),
        ))
        write_csv(result_path, rows)
        logger.info('%s %s %s%%: %.4f C=%.4g', model_key, dataset, fraction, metric, best_c)


def evaluation_jobs(model: str, dataset: str, fractions: bool, output: Path) -> list[tuple]:
    """Keep full-data scoreboard files separate from complete curve files."""
    full_nested = model.startswith('olmoearth') or dataset == 'eurosat-spatial'
    if not fractions:
        return [((100,), output / 'scoreboard', full_nested)]
    jobs = [(FRACTIONS, output, True)]
    if dataset == 'eurosat':
        jobs.append(((100,), output / 'scoreboard', full_nested))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument('--models', nargs='+', choices=MODELS)
    model_selection.add_argument('--all-models', action='store_true',
                                 help='run all eleven archived scoreboard backbones instead of the five overlap-study defaults')
    parser.add_argument('--download', action='store_true', help='download missing official EuroSAT data/splits')
    parser.add_argument('--fractions', action='store_true')
    parser.add_argument('--datasets', nargs='+', choices=('eurosat', 'eurosat-spatial'), default=['eurosat'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--cache', type=Path, default=ROOT / 'output/benchmark-embeddings')
    parser.add_argument('--output', type=Path, default=RESULTS)
    parser.add_argument('--fresh', action='store_true', help='recompute embeddings instead of reading verified cache')
    parser.add_argument('--extract-only', action='store_true', help='write embeddings and metadata without fitting or scoring classifiers')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.workers < 0:
        parser.error('--workers cannot be negative')
    if not torch.cuda.is_available():
        raise RuntimeError('this benchmark configuration requires CUDA')
    check_source()
    prepare_data(args.download)
    bench_data = ROOT / 'data/eurosat'
    if not bench_data.exists():
        bench_data.symlink_to('EuroSAT', target_is_directory=True)
    if bench_data.resolve() != Path(DATA_ROOT).resolve():
        raise ValueError('data/eurosat must point to the checked data/EuroSAT download')
    models = list(MODELS) if args.all_models else args.models or DEFAULT_MODELS
    for model_key in models:
        for dataset in args.datasets:
            data, metadata = extract(model_key, dataset, args.device, args.workers,
                                     args.batch_size or DEFAULT_BATCH[model_key],
                                     args.output, args.cache, args.fresh)
            if not args.extract_only:
                for fractions, destination, nested in evaluation_jobs(
                    model_key, dataset, args.fractions, args.output
                ):
                    evaluate(model_key, dataset, data, metadata, fractions,
                             args.device, destination, nested=nested)


if __name__ == '__main__':
    main()
