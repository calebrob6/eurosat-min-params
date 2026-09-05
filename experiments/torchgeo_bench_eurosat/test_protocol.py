"""CPU-only protocol tests run inside the optional benchmark environment."""
import contextlib
import csv
import io
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

import compare
from run import FRACTIONS, RecordingLoader, evaluation_jobs


class ToyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {'image': torch.tensor([float(index)]), 'label': index}


class ProtocolTests(unittest.TestCase):
    def test_source_recording_preserves_original_shuffle(self):
        torch.manual_seed(0)
        original = DataLoader(ToyDataset(), batch_size=4, shuffle=True)
        expected = torch.cat([batch['label'] for batch in original]).tolist()
        torch.manual_seed(0)
        wrapped = RecordingLoader(DataLoader(ToyDataset(), batch_size=4, shuffle=True))
        actual = torch.cat([batch['label'] for batch in wrapped]).tolist()
        self.assertEqual(expected, actual)
        self.assertEqual(actual, wrapped.indices)

    def test_full_only_never_targets_curve_csv(self):
        output = Path('/tmp/results')
        for model in compare.STEMS:
            for dataset in ('eurosat', 'eurosat-spatial'):
                jobs = evaluation_jobs(model, dataset, False, output)
                self.assertEqual(len(jobs), 1)
                fractions, destination, nested = jobs[0]
                self.assertEqual(fractions, (100,))
                self.assertEqual(destination, output / 'scoreboard')
                self.assertEqual(nested, model.startswith('olmoearth') or dataset == 'eurosat-spatial')

    def test_curve_and_scoreboard_orders_are_explicit(self):
        output = Path('/tmp/results')
        self.assertEqual(evaluation_jobs('resnet50', 'eurosat', True, output), [
            (FRACTIONS, output, True), ((100,), output / 'scoreboard', False),
        ])
        self.assertEqual(evaluation_jobs('olmoearth_nano', 'eurosat', True, output), [
            (FRACTIONS, output, True), ((100,), output / 'scoreboard', True),
        ])

    def test_comparison_uses_requested_result_directory(self):
        original_report = (compare.FRESH / 'scoreboard_comparison.csv').read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            shutil.copytree(compare.FRESH / 'scoreboard', destination / 'scoreboard')
            for model in compare.CURVES:
                for dataset in ('eurosat', 'eurosat-spatial'):
                    name = f'{model}_{dataset}.csv'
                    shutil.copy2(compare.FRESH / name, destination / name)
            with patch.object(sys, 'argv', ['compare.py', '--results-dir', directory]):
                with contextlib.redirect_stdout(io.StringIO()):
                    compare.main()
            with (destination / 'fraction_comparison.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 98)
            self.assertTrue(all(row['fresh_csv'].startswith(directory) for row in rows))
            with (destination / 'scoreboard_comparison.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 11)
        self.assertEqual((compare.FRESH / 'scoreboard_comparison.csv').read_bytes(), original_report)


if __name__ == '__main__':
    unittest.main()
