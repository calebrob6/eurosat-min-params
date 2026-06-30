#!/usr/bin/env python
"""Train a deep learning model on the EuroSAT dataset.

Uses torchgeo's ``EuroSATDataModule`` for data loading and ``ClassificationTask``
(a PyTorch Lightning module wrapping a timm backbone) for the model, optimizer,
and metrics. Configuration is handled entirely through argparse.

Example:
    python train.py --data-dir data/EuroSAT --model resnet18 --epochs 10
"""

import argparse
import os

import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import CSVLogger

from torchgeo.datamodules import EuroSATDataModule
from torchgeo.datasets import EuroSAT
from torchgeo.trainers import ClassificationTask

# Band selections supported on the command line.
BAND_SETS = {
    'all': EuroSAT.all_band_names,  # all 13 Sentinel-2 bands
    'rgb': EuroSAT.rgb_bands,  # B04, B03, B02
}

NUM_CLASSES = 10


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a classification model on EuroSAT.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data
    parser.add_argument(
        '--data-dir',
        default='data/EuroSAT',
        help='Root directory of the EuroSAT dataset.',
    )
    parser.add_argument(
        '--bands',
        default='all',
        choices=sorted(BAND_SETS.keys()),
        help="Which bands to use ('all' = 13 bands, 'rgb' = 3 bands).",
    )
    parser.add_argument('--batch-size', type=int, default=64, help='Mini-batch size.')
    parser.add_argument(
        '--num-workers', type=int, default=8, help='Data loading workers.'
    )

    # Model
    parser.add_argument(
        '--model', default='resnet18', help='Name of the timm model to train.'
    )
    parser.add_argument(
        '--pretrained',
        action='store_true',
        help='Initialize the backbone with ImageNet weights.',
    )
    parser.add_argument(
        '--freeze-backbone',
        action='store_true',
        help='Freeze the backbone and only train the classifier head.',
    )

    # Optimization
    parser.add_argument('--epochs', type=int, default=10, help='Maximum epochs.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Patience (epochs) for the LR scheduler.',
    )
    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=0,
        help='Stop early after this many epochs without val_loss improvement '
        '(0 disables early stopping).',
    )

    # Hardware / runtime
    parser.add_argument(
        '--accelerator', default='auto', help="Lightning accelerator (e.g. 'gpu')."
    )
    parser.add_argument(
        '--devices', default='auto', help="Devices to use (e.g. '1' or 'auto')."
    )
    parser.add_argument(
        '--precision', default='32-true', help='Lightning precision setting.'
    )
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument(
        '--output-dir', default='output', help='Directory for logs and checkpoints.'
    )
    parser.add_argument(
        '--fast-dev-run',
        action='store_true',
        help='Run a single train/val/test batch to smoke-test the pipeline.',
    )

    return parser.parse_args()


def main() -> None:
    """Train and evaluate a model on EuroSAT."""
    args = parse_args()

    pl.seed_everything(args.seed, workers=True)

    bands = BAND_SETS[args.bands]

    datamodule = EuroSATDataModule(
        root=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bands=bands,
        download=False,
    )

    task = ClassificationTask(
        model=args.model,
        weights=True if args.pretrained else None,
        in_channels=len(bands),
        num_classes=NUM_CLASSES,
        loss='ce',
        lr=args.lr,
        patience=args.patience,
        freeze_backbone=args.freeze_backbone,
    )

    logger = CSVLogger(save_dir=args.output_dir, name='eurosat')

    callbacks: list[pl.Callback] = [
        ModelCheckpoint(monitor='val_loss', mode='min', save_last=True),
        LearningRateMonitor(logging_interval='epoch'),
    ]
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                mode='min',
                patience=args.early_stopping_patience,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=args.output_dir,
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(model=task, datamodule=datamodule)

    ckpt_path = None if args.fast_dev_run else 'best'
    trainer.test(model=task, datamodule=datamodule, ckpt_path=ckpt_path)

    if not args.fast_dev_run:
        best = trainer.checkpoint_callback.best_model_path
        print(f'\nBest checkpoint: {best}')
        print(f'Logs written to: {os.path.join(logger.log_dir)}')


if __name__ == '__main__':
    main()
