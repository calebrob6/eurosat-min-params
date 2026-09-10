"""CPU-only protocol tests run inside the optional benchmark environment."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, Dataset

import run as benchmark_run
from run import FRACTIONS, RecordingLoader, evaluation_jobs


class ToyDataset(Dataset):
    def __len__(self):
        return 20

    def __getitem__(self, index):
        return {'image': torch.tensor([float(index)]), 'label': index}


class ProtocolTests(unittest.TestCase):
    def test_every_archived_backbone_has_a_resolvable_config_and_batch_size(self):
        with (benchmark_run.ROOT / 'experiments/torchgeo_bench_eurosat/blog_post/scoreboard.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        configured_names = set()
        for model, config in benchmark_run.MODELS.items():
            cfg = benchmark_run.compose_config([f'model={config}'])
            configured_names.add(cfg.model.name)
            self.assertGreater(benchmark_run.DEFAULT_BATCH[model], 0)
        self.assertEqual(configured_names, {row['model_config'] for row in rows})

    def test_historical_curve_targets_preserve_full_figure_protocol(self):
        curve_models = (
            'resnet50', 'convnext_tiny', 'dofa_base', 'dofa_large',
            'olmoearth_nano', 'olmoearth_small', 'olmoearth_base',
        )
        archive = benchmark_run.ROOT / 'experiments/torchgeo_bench_eurosat'
        linear_rows = 0
        for protocol, dataset in (('random', 'eurosat'), ('spatial', 'eurosat-spatial')):
            for model in curve_models:
                cfg = benchmark_run.compose_config([f'model={benchmark_run.MODELS[model]}'])
                path = archive / 'blog_post/training_fractions' / protocol / f'{cfg.model.name}.csv'
                with path.open() as handle:
                    rows = [row for row in csv.DictReader(handle) if row['method'] == 'linear']
                counts = [162, 324, 810, 1620, 3240, 8100]
                if protocol == 'spatial' or model.startswith('olmoearth'):
                    counts.append(16200)
                with self.subTest(model=model, protocol=protocol):
                    self.assertEqual([int(row['n_train']) for row in rows], counts)
                    for row in rows:
                        self.assertEqual(row['dataset'], dataset)
                        self.assertEqual(row['normalization'], 'bandspec_zscore')
                        self.assertEqual(row['merge_val'], 'False')
                        self.assertEqual(
                            tuple(int(row[key]) for key in ('n_val', 'n_test', 'seed', 'bootstrap')),
                            (5400, 5400, 0, 200),
                        )
                        self.assertEqual(
                            tuple(float(row[key]) for key in ('c_range_start', 'c_range_stop', 'c_range_num')),
                            (-6.0, 4.0, 40.0),
                        )
                linear_rows += len(rows)
        self.assertEqual(linear_rows, 94)
        with (archive / 'reproduced/fraction_comparison.csv').open() as handle:
            previous = list(csv.DictReader(handle))
        self.assertEqual(len(previous), 98)
        self.assertEqual(
            {(row['model'], row['dataset'], int(float(row['train_fraction_percent']))) for row in previous},
            {(model, dataset, fraction) for model in curve_models
             for dataset in ('eurosat', 'eurosat-spatial') for fraction in FRACTIONS},
        )

    def test_default_and_all_model_extraction_coverage(self):
        for flags, expected in (
            ([], ('resnet50', 'convnext_tiny', 'dofa_large', 'olmoearth_nano', 'olmoearth_base')),
            (['--all-models'], tuple(benchmark_run.MODELS)),
        ):
            with self.subTest(flags=flags), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / 'data/EuroSAT'
                data.mkdir(parents=True)
                (root / 'data/eurosat').symlink_to('EuroSAT')
                with (
                    patch.object(benchmark_run, 'ROOT', root),
                    patch.object(benchmark_run, 'DATA_ROOT', data),
                    patch.object(benchmark_run, 'check_source'),
                    patch.object(benchmark_run, 'prepare_data'),
                    patch.object(benchmark_run.torch.cuda, 'is_available', return_value=True),
                    patch.object(benchmark_run, 'extract', return_value=({}, {})) as extract,
                    patch.object(benchmark_run, 'evaluate') as evaluate,
                    patch.object(sys, 'argv', ['run.py', '--extract-only', *flags]),
                ):
                    benchmark_run.main()
                    self.assertEqual([call.args[0] for call in extract.call_args_list], list(expected))
                    evaluate.assert_not_called()

    def test_extract_only_never_fits_or_evaluates_a_probe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data/EuroSAT'
            data.mkdir(parents=True)
            (root / 'data/eurosat').symlink_to('EuroSAT')
            with (
                patch.object(benchmark_run, 'ROOT', root),
                patch.object(benchmark_run, 'DATA_ROOT', data),
                patch.object(benchmark_run, 'check_source'),
                patch.object(benchmark_run, 'prepare_data'),
                patch.object(benchmark_run.torch.cuda, 'is_available', return_value=True),
                patch.object(benchmark_run, 'extract', return_value=({}, {})) as extract,
                patch.object(benchmark_run, 'evaluate') as evaluate,
                patch.object(sys, 'argv', ['run.py', '--models', 'resnet50', '--extract-only']),
            ):
                benchmark_run.main()
                extract.assert_called_once()
                evaluate.assert_not_called()

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
        for model in benchmark_run.MODELS:
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

if __name__ == '__main__':
    unittest.main()
