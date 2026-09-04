#!/usr/bin/env python
"""Evaluate the frozen 306-value checkpoint directly from EuroSAT TIFFs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from src.data import iter_images
from src.frontier import RECIPE, frontier_features
from src.linmodel import predict_reference_class


def main() -> None:
    with np.load(Path(__file__).with_name('model.npz'), allow_pickle=False) as model:
        w, b, indices = model['W'], model['b'], model['feature_idx']
        names = model['feature_names'].tolist()
        if str(model['recipe']) != RECIPE or w.size + b.size != 306:
            raise ValueError('checkpoint recipe or parameter count differs')
    for split in ('val', 'test'):
        correct = total = 0
        for images, labels in iter_images(split):
            features, actual_names = frontier_features(images)
            if actual_names != names:
                raise ValueError('feature order differs from checkpoint')
            predicted = predict_reference_class(features, w, b, indices)
            correct += int((predicted == labels).sum())
            total += len(labels)
        print(f'{split}: {correct}/{total} = {correct / total:.6f}; parameters=306')


if __name__ == '__main__':
    main()
