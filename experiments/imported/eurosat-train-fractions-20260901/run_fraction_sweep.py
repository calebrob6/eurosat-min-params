"""Run nested random EuroSAT train-fraction probes from one feature extraction."""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from torchgeo_bench.cli import _setup_logging
from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.knn import resolve_knn_device
from torchgeo_bench.main import (
    embed_split,
    evaluate_knn,
    evaluate_logistic,
    resolve_model_config,
)
from torchgeo_bench.results import append_rows_atomic, metric_row
from torchgeo_bench.resume import _normalize_bands_value, _resume_config_hash
from torchgeo_bench.utils import resolve_device

logger = logging.getLogger(__name__)


def _completed_methods(path: Path) -> dict[str, set[str]]:
    """Return completed methods keyed by partition name."""
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        rows = csv.DictReader(file)
        completed: dict[str, set[str]] = {}
        for row in rows:
            completed.setdefault(row["partition"], set()).add(row["method"])
    return completed


def main() -> None:
    """Extract features once and evaluate nested random train subsets."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    _setup_logging()
    fractions = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    cfg = compose_config(
        [
            f"model={args.model}",
            "dataset.names=[eurosat]",
            "dataset.bands=all",
            "eval.merge_val=false",
            f"dataset.batch_size={args.batch_size}",
            f"device={args.device}",
            f"output={args.output}",
        ]
    )
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    model_eval = cfg.model.get("eval", None) if "eval" in cfg.model else None
    eval_cfg = OmegaConf.merge(cfg.eval, model_eval or {})
    c_start, c_stop, c_num = eval_cfg.c_range
    c_values = [
        float(value)
        for value in np.logspace(float(c_start), float(c_stop), int(c_num)).tolist()
    ]

    model_cfg = resolve_model_config(cfg.model, "eurosat")
    image_size = model_cfg.get("image_size", cfg.dataset.image_size)
    interpolation = model_cfg.get("interpolation", cfg.dataset.interpolation)
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name="eurosat",
        partition_name="default",
        batch_size=cfg.dataset.batch_size,
        num_workers=int(cfg.dataset.num_workers),
        return_val=True,
        image_size=image_size,
        interpolation=interpolation,
        bands="all",
    )

    bench = get_bench_dataset_class("eurosat")()
    bands = bench.select_band_specs(None)
    model_cfg.pop("interpolation", None)
    model = instantiate(
        model_cfg,
        bands=bands,
        normalization=str(cfg.dataset.normalization),
    )
    model.to(device).eval()

    logger.info("[%s] Extracting full EuroSAT embeddings once", model_cfg.name)
    x_train, y_train = embed_split(model, train_loader, device, verbose=True)
    x_val, y_val = embed_split(model, val_loader, device, verbose=True)
    x_test, y_test = embed_split(model, test_loader, device, verbose=True)
    feature_dim = x_train.shape[1]

    permutation = np.random.default_rng(cfg.seed).permutation(len(x_train))
    completed = _completed_methods(args.output)
    knn_k = int(eval_cfg.knn_k)
    knn_device = resolve_knn_device(eval_cfg.get("knn_device"), cfg.device)
    calibration = eval_cfg.calibration
    knn_bins = calibration.get("n_bins_knn", None)
    linear_bins = int(calibration.get("n_bins_linear", 15))
    temp_scale = bool(calibration.get("temp_scale", False))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for fraction in fractions:
        percent = int(round(fraction * 100))
        partition = f"random_{percent:02d}pct_train"
        count = max(1, int(round(len(x_train) * fraction)))
        indices = permutation[:count]
        x_fraction = x_train[indices]
        y_fraction = y_train[indices]
        methods = completed.get(partition, set())

        cfg.dataset.partition = partition
        common_meta = {
            "dataset": "eurosat",
            "seed": cfg.seed,
            "model": model_cfg._target_,
            "name": model_cfg.name,
            "normalization": str(cfg.dataset.normalization),
            "image_size": image_size,
            "interpolation": interpolation,
            "partition": partition,
            "bands": _normalize_bands_value(cfg.dataset.bands),
            "num_classes": bench.num_classes,
            "config_hash": _resume_config_hash(cfg),
            "c_range_start": float(c_start),
            "c_range_stop": float(c_stop),
            "c_range_num": int(c_num),
            "merge_val": False,
            "bootstrap": int(eval_cfg.bootstrap),
            "res": model_cfg.get("res"),
            "pool": model_cfg.get("pool"),
        }
        n_counts = {
            "train": count,
            "val": len(x_val),
            "test": len(x_test),
        }
        rows: list[dict] = []
        logger.info(
            "[%s] Evaluating %s (%d/%d train samples)",
            model_cfg.name,
            partition,
            count,
            len(x_train),
        )

        if f"knn{knn_k}" not in methods:
            score, lower, upper, cal, bins = evaluate_knn(
                x_fraction,
                y_fraction,
                x_test,
                y_test,
                cfg.seed,
                int(eval_cfg.bootstrap),
                verbose=True,
                device=knn_device,
                n_neighbors=knn_k,
                calibration_n_bins=knn_bins,
            )
            rows.append(
                metric_row(
                    common_meta,
                    method=f"knn{knn_k}",
                    metric_name="accuracy",
                    metric_value=score,
                    ci_lower=lower,
                    ci_upper=upper,
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                    ece=cal["ece"],
                    rms_ce=cal["rms_ce"],
                    mce=cal["mce"],
                    calibration_n_bins=bins,
                )
            )

        if "linear" not in methods:
            score, lower, upper, best_c, cal, cal_ts = evaluate_logistic(
                x_fraction,
                y_fraction,
                x_val,
                y_val,
                x_test,
                y_test,
                c_values,
                cfg.seed,
                int(eval_cfg.bootstrap),
                False,
                cfg.device,
                verbose=True,
                calibration_n_bins=linear_bins,
                temp_scale=temp_scale,
            )
            rows.append(
                metric_row(
                    common_meta,
                    method="linear",
                    metric_name="accuracy",
                    metric_value=score,
                    ci_lower=lower,
                    ci_upper=upper,
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                    best_c=best_c,
                    ece=cal["ece"],
                    rms_ce=cal["rms_ce"],
                    mce=cal["mce"],
                    ece_ts=cal_ts["ece_ts"],
                    rms_ce_ts=cal_ts["rms_ce_ts"],
                    mce_ts=cal_ts["mce_ts"],
                    temperature=cal_ts["temperature"],
                    calibration_n_bins=linear_bins,
                )
            )
        append_rows_atomic(str(args.output), rows)


if __name__ == "__main__":
    main()
