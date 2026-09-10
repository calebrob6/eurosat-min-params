"""Prespecified settings and artifact handling for the representation study."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parent
MODELS = {
    'resnet50': 2048,
    'convnext_tiny': 768,
    'dofa_large': 1024,
    'olmoearth_nano': 128,
    'olmoearth_base': 768,
}
REPRESENTATIONS = {'frontier33': 33, 'pool377': 377, 'imagestats52': 52}
SIZES = {'train': 16200, 'val': 5400, 'test': 5400}
BENCH_REVISION = '9c8e4afab46675d7279c88828dfcbf0ca99b3a07'
ALPHAS = tuple(float(x) for x in np.logspace(-8, 4, 13))
CS = tuple(float(x) for x in np.logspace(-4, 8, 13))
BETAS = (0.0, 0.25, 1.0, 4.0)
NULL_SEEDS = (0, 1, 2)
RANKS = (1, 2, 4, 8, 16, 33, 64, 128, 256, 377, 512, 1024, 2048)
BOOTSTRAP = 200
PROBE_MAX_ITER = 8000
PROBE_TOL = 1e-6


def sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_value(value):
    """Represent explicitly undefined numerical scores as JSON null."""
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), sort_keys=True, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def runtime() -> dict:
    return {
        'python': platform.python_version(),
        'packages': {
            name: importlib.metadata.version(name)
            for name in ('numpy', 'scipy', 'scikit-learn', 'rasterio', 'torch', 'matplotlib')
        },
    }


def specification(models: list[str], device: str) -> dict:
    import torch
    import torchgeo_bench
    from torchgeo_bench import linear

    if not models or len(set(models)) != len(models) or any(key not in MODELS for key in models):
        raise ValueError('models must be a nonempty, unique subset of the five study backbones')
    source = ROOT / 'output/torchgeo-bench'
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'diff', '--name-only'], text=True).strip()
    if (revision != BENCH_REVISION or dirty
            or not Path(torchgeo_bench.__file__).resolve().is_relative_to(source)):
        raise ValueError('install the unmodified pinned output/torchgeo-bench checkout in the benchmark environment')
    return {
        'schema': 'eurosat-representation-overlap-v2',
        'models': models,
        'representations': REPRESENTATIONS,
        'split_sizes': SIZES,
        'alphas': ALPHAS,
        'ridge_objective': 'mean_squared_error + alpha * squared_weight_norm',
        'c_values': CS,
        'betas': BETAS,
        'primary_metric': 'accuracy',
        'selection_ties': ['validation_log_loss', 'smaller_C', 'beta_order'],
        'bootstrap': BOOTSTRAP,
        'seed': 0,
        'null_seeds': NULL_SEEDS,
        'classification_control_seed': 0,
        'pca_ranks': RANKS,
        'device': device,
        'tf32': False,
        'merge_val': False,
        'probe_max_iter': PROBE_MAX_ITER,
        'probe_retry_max_iter': 2 * PROBE_MAX_ITER,
        'probe_tol': PROBE_TOL,
        'probe_stopping': 'require return before both iteration and function-evaluation budgets; not a gradient-convergence claim',
        'band_control': 'physical B10 replaced by zero before all handcrafted computations',
        'source_sha256': {path.name: sha256(path) for path in sorted(PACKAGE.glob('*.py'))},
        'runtime': runtime(),
        'solver_source_sha256': sha256(Path(linear.__file__)),
        'torchgeo_bench_revision': revision,
        'gpu': torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else None,
        'cuda': torch.version.cuda,
    }


def require_same(path: Path, expected: dict) -> None:
    if json.loads(path.read_text()) != json_value(expected):
        raise ValueError(f'configuration changed: use a new output directory instead of {path.parent}')
