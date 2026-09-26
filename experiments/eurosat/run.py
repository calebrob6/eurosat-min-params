"""Evaluate the 306-parameter EuroSAT model and fit the ImageStats baseline.

    python -m experiments.eurosat.run --download [--refit] [--fractions]
"""

from __future__ import annotations

import os

# L-BFGS stopping and the C sweep are thread-sensitive.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from experiments.data import CLASSES, ROOT, SPLITS, default_device, extract, list_split, prepare_data
from experiments.linear import fit_folded_logreg, predict, stored_parameters, to_reference_class
from patch_features import EuroSATFeatures

MODEL_PATH = ROOT / 'models' / 'eurosat306.pt'
FRACTIONS = (1, 2, 5, 10, 20, 50, 100)
IMAGESTATS_CS = (
    0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3,
    1, 3, 10, 30, 100, 150, 200, 250, 300, 400, 500, 700, 1000, 2000,
)  # fmt: skip


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def fit_head(x: np.ndarray, y: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    return to_reference_class(*fit_folded_logreg(x, y, C=C))


def evaluate(data: dict, output: Path, refit: bool) -> None:
    rows, per_class = [], []

    def record(name: str, split: str, x: np.ndarray, head: tuple) -> None:
        labels = data[split]['labels']
        predicted = predict(x, *head)
        rows.append(dict(
            model=name, split=split, parameters=stored_parameters(*head), images=len(labels),
            correct=int((predicted == labels).sum()), accuracy=float((predicted == labels).mean()),
        ))
        print(rows[-1], flush=True)
        for i, class_name in enumerate(CLASSES):
            mask = labels == i
            per_class.append(dict(
                model=name, split=split, class_name=class_name, images=int(mask.sum()),
                accuracy=float((predicted[mask] == i).mean()),
            ))

    if refit:
        head = fit_head(data['train']['33'], data['train']['labels'], C=3.0)
        linear = torch.nn.Linear(33, 10)
        linear.load_state_dict({'weight': torch.from_numpy(head[0]), 'bias': torch.from_numpy(head[1])})
        torch.save(linear.state_dict(), output / 'eurosat306.pt')
    else:
        state = torch.load(MODEL_PATH, weights_only=True)
        head = (state['weight'].double().numpy(), state['bias'].double().numpy())
    for split in ('val', 'test'):
        record('306', split, data[split]['33'], head)

    sweep, best = [], None
    for C in IMAGESTATS_CS:
        head = fit_head(data['train']['52'], data['train']['labels'], C)
        accuracy = float((predict(data['val']['52'], *head) == data['val']['labels']).mean())
        sweep.append(dict(C=C, validation_accuracy=accuracy))
        if best is None or accuracy > best[0]:
            best = (accuracy, C, head)
    write_csv(output / 'imagestats_c_sweep.csv', sweep)
    print(f'ImageStats validation-selected C={best[1]}', flush=True)
    for split in ('val', 'test'):
        record('imagestats', split, data[split]['52'], best[2])
    write_csv(output / 'models.csv', rows)
    write_csv(output / 'per_class.csv', per_class)


def learning_curves(data: dict, output: Path) -> None:
    """Refit both heads on five stratified subsamples per fraction, on both split protocols."""
    names = np.concatenate([data[s]['filenames'] for s in SPLITS])
    labels = np.concatenate([data[s]['labels'] for s in SPLITS])
    lookup = {name: i for i, name in enumerate(names)}
    spatial = {s: np.asarray([lookup[p.name] for p in list_split(f'spatial-{s}')[0]]) for s in SPLITS}
    n_test = len(data['test']['labels'])
    random = {'train': np.arange(len(data['train']['labels'])), 'test': np.arange(len(names) - n_test, len(names))}
    for family, C in (('33', 3.0), ('52', 300.0)):
        x = np.concatenate([data[s][family] for s in SPLITS])
        rows = []
        for protocol, split in (('random', random), ('spatial', spatial)):
            train, test = split['train'], split['test']
            for fraction in FRACTIONS:
                n = round(len(train) * fraction / 100)
                scores = []
                for seed in range(5):
                    selected = train
                    if fraction < 100:
                        splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
                        selected = train[next(splitter.split(np.zeros(len(train)), labels[train]))[0]]
                    w, b = fit_folded_logreg(x[selected], labels[selected], C=C)
                    scores.append(float((predict(x[test], w, b) == labels[test]).mean()))
                rows.append(dict(
                    features=family, C=C, split_protocol=protocol, train_fraction_percent=fraction,
                    n_train=n, n_test=len(test),
                    **{f'seed_{seed}_test_accuracy': score for seed, score in enumerate(scores)},
                    test_accuracy_mean=float(np.mean(scores)),
                    test_accuracy_stdev=float(np.std(scores, ddof=1)),
                ))
                print(f'{family} features, {protocol} {fraction}%: {np.mean(scores):.4f}', flush=True)
        write_csv(output / f'fractions_{family}.csv', rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--download', action='store_true', help='fetch EuroSAT if it is missing')
    parser.add_argument('--refit', action='store_true', help='refit the 33-feature head at C=3 instead of loading it')
    parser.add_argument('--fractions', action='store_true', help='also compute the learning curves')
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--output', type=Path, default=ROOT / 'output' / 'eurosat')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare_data(args.download)
    data = {}
    for split in SPLITS:
        paths, labels = list_split(split)
        data[split] = extract({key: EuroSATFeatures(key) for key in ('33', '52')}, split, args.device)
        data[split] |= {'labels': labels, 'filenames': np.asarray([p.name for p in paths])}
        print(f'{split}: {len(labels)} images', flush=True)
    evaluate(data, args.output, args.refit)
    if args.fractions:
        learning_curves(data, args.output)
    print(f'tables written to {args.output}')


if __name__ == '__main__':
    main()
