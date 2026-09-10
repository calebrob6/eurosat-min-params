#!/usr/bin/env python
"""Explicit, opt-in reruns of the historical tiny-CNN and MOSAIKS configurations.

These are new measurements, not recovered checkpoints. Raw TIFFs are staged in
a new output directory, never read from or written to the historical cache.
"""
from __future__ import annotations

import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import argparse
import contextlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.article_baselines.run import (  # noqa: E402
    SPLITS, dataset_lists, file_digest, output_directory, read_image, write_csv,
)
from src.linmodel import (  # noqa: E402
    fit_folded_logreg, l1_rank, predict, predict_reference_class, to_reference_class,
)


def prepare_raw_arrays(records: dict, output: Path) -> dict[str, Path]:
    """Fresh, explicitly named uint16 arrays; existing output directories fail."""
    paths = {}
    for split in SPLITS:
        path = output / f'{split}_raw_uint16.npy'
        paths[split] = path
        images = np.lib.format.open_memmap(
            path, mode='w+', dtype=np.uint16,
            shape=(len(records[split]['paths']), 13, 64, 64),
        )
        for index, source in enumerate(records[split]['paths']):
            images[index] = read_image(('eurosat', source))
        images.flush()
        del images
        print(f'{split}: staged original TIFFs in {path}', flush=True)
    return paths


def tiny_cnn(paths: dict, records: dict, output: Path, epochs: int, gpu: int) -> dict:
    """Use the archived training source, replacing only its cache reader."""
    from experiments import conv_gap

    original_loader, original_argv = conv_gap.load_cached, sys.argv
    conv_gap.load_cached = lambda split: (
        np.load(paths[split], mmap_mode='r'), records[split]['labels'],
    )
    sys.argv = [
        'experiments/conv_gap.py', '--bands', 'all', '--filters', '16',
        '--epochs', str(epochs), '--bs', '256', '--lr', '0.003',
        '--wd', '0.0005', '--seed', '0', '--gpu', str(gpu),
    ]
    try:
        with (output / 'training.log').open('x') as handle:
            with contextlib.redirect_stdout(handle):
                conv_gap.main()
    finally:
        conv_gap.load_cached, sys.argv = original_loader, original_argv
    print((output / 'training.log').read_text(), end='')
    return dict(
        source='experiments/conv_gap.py', learned_parameters=2058, epochs=epochs,
        seed=0, filters=16, bands=13,
        historical_limit='The note records 60–80 epochs, not the exact seed/epoch/checkpoint.',
    )


def mosaiks_parameter_counts(w: np.ndarray, b: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    head = int(w.size + b.size)
    normalization = int(mu.size + sd.size)
    return dict(learned_parameters=head + normalization, head_parameters=head,
                normalization_parameters=normalization)


def mosaiks(paths: dict, records: dict, output: Path, device: str, batch_size: int) -> dict:
    """One fixed 512-of-4096 Gaussian RCF configuration, not a test-gated sweep."""
    import torch
    from torchgeo.models import RCF

    # Original source computes population moments on the whole float32 train
    # tensor. Retain that reduction rather than substituting streaming moments.
    train = np.load(paths['train'], mmap_mode='r').astype(np.float32)
    mu = train.mean((0, 2, 3), keepdims=True)
    sd = train.std((0, 2, 3), keepdims=True) + 1e-6
    del train
    rcf = RCF(
        in_channels=13, features=4096, kernel_size=3, bias=-1.0,
        seed=1, mode='gaussian',
    ).to(device).eval()
    features = {}
    for split in SPLITS:
        raw = np.load(paths[split], mmap_mode='r')
        parts = []
        with torch.inference_mode():
            for start in range(0, len(raw), batch_size):
                x = (raw[start:start + batch_size].astype(np.float32) - mu) / sd
                parts.append(rcf(torch.from_numpy(x).to(device)).cpu().numpy())
        features[split] = np.concatenate(parts)
    indices = l1_rank(features['train'], records['train']['labels'])[:512]
    model = fit_folded_logreg(features['train'], records['train']['labels'], indices, C=1.0)
    wr, br = to_reference_class(model[0], model[1])
    counts = mosaiks_parameter_counts(wr, br, mu, sd)
    rows = []
    for split in ('val', 'test'):
        predicted = predict_reference_class(features[split], wr, br, indices)
        if not np.array_equal(predicted, predict(features[split], *model)):
            raise ValueError('reference-class conversion changed predictions')
        labels = records[split]['labels']
        rows.append(dict(
            model='mosaiks_gaussian_top512', split=split, **counts,
            n_images=len(labels), correct=int((predicted == labels).sum()),
            accuracy=float((predicted == labels).mean()), C=1, seed=1,
            provenance_status='new_reference_historical_rounded_note_only',
        ))
    np.savez_compressed(
        output / 'mosaiks.npz', W=wr, b=br, feature_idx=indices, ref_class=0,
        C=1.0, RCF_features=4096, RCF_kernel=3, RCF_bias=-1.0, RCF_seed=1,
        mode='gaussian', input_mean=mu, input_std=sd,
    )
    write_csv(output / 'metrics.csv', rows)
    for row in rows:
        print(row, flush=True)
    return dict(
        source='experiments/mosaiks_floor95.py', **counts,
        generated_features=4096, selected_features=512, seed=1, C=1,
        historical_limit=(
            'No archived selected-filter list, fitted head, or exact 512-feature '
            'test score. The original floor script used test accuracy in its gate; '
            'this command fixes 512 and C=1 before any test scoring.'
        ),
        torch=torch.__version__,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('tiny-cnn', 'mosaiks'), required=True)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data/EuroSAT')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu', help='MOSAIKS device, e.g. cuda:0')
    parser.add_argument('--gpu', type=int, default=0, help='CNN CUDA device ordinal')
    parser.add_argument('--epochs', type=int, default=60, help='CNN rerun epoch budget; historical exact value missing')
    parser.add_argument('--batch-size', type=int, default=128, help='MOSAIKS extraction batch size')
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1:
        parser.error('--batch-size and --epochs must be positive')
    # Resolve optional dependencies before writing multi-gigabyte raw arrays.
    import torch
    if args.method == 'tiny-cnn':
        if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
            parser.error('the CNN rerun requires an available CUDA GPU; CPU fallback is disabled')
    else:
        import torchgeo.models  # noqa: F401
        device = torch.device(args.device)
        if device.type not in ('cpu', 'cuda'):
            parser.error('MOSAIKS supports CPU or CUDA devices')
        if device.type == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA requested but unavailable')
        if device.type == 'cuda' and device.index is not None and device.index >= torch.cuda.device_count():
            parser.error('requested CUDA device is unavailable')
    started = time.monotonic()
    records, _ = dataset_lists('eurosat', args.data_root)
    output = output_directory(args.method, args.output_dir)
    paths = prepare_raw_arrays(records, output)
    if args.method == 'tiny-cnn':
        metadata = tiny_cnn(paths, records, output, args.epochs, args.gpu)
    else:
        metadata = mosaiks(paths, records, output, args.device, args.batch_size)
    metadata.update(
        command=sys.argv, elapsed_seconds=time.monotonic() - started,
        raw_tiffs_only=True, test_used_for_selection=False, torch=torch.__version__,
        split_sha256={split: records[split]['split_sha256'] for split in SPLITS},
        result_kind='new_rerun_not_historical_checkpoint_evaluation',
        python=platform.python_version(), numpy=np.__version__,
        sklearn=importlib.metadata.version('scikit-learn'),
        rasterio=importlib.metadata.version('rasterio'),
        source_sha256=file_digest(ROOT / metadata['source']),
        adapter_sha256=file_digest(Path(__file__)),
    )
    if args.method == 'mosaiks':
        metadata['torchgeo'] = importlib.metadata.version('torchgeo')
    with (output / 'environment.json').open('x') as handle:
        json.dump(metadata, handle, indent=2)
        handle.write('\n')


if __name__ == '__main__':
    main()
