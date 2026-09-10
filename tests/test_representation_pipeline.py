"""Tiny end-to-end study fixture, including lock and report coverage."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.representation_overlap import data, report, run
from experiments.representation_overlap.protocol import REPRESENTATIONS, write_json


class PipelineTests(unittest.TestCase):
    def test_all_stage_keeps_selection_before_test_evaluation(self):
        calls = Mock()
        with (
            patch.object(run, 'select', calls.select),
            patch.object(run, 'lock_selection', calls.lock),
            patch.object(run, 'evaluate', calls.evaluate),
            patch.object(data, 'prepare', calls.prepare),
            patch.object(report, 'export', calls.export),
            patch.object(run.logging, 'basicConfig'),
            patch('sys.argv', ['run', 'all']),
        ):
            run.main()
        self.assertEqual([entry[0] for entry in calls.mock_calls],
                         ['prepare', 'select', 'lock', 'evaluate', 'export'])

    def test_pilot_cannot_be_locked(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_json(output / 'protocol.json', {'pilot': True})
            with self.assertRaisesRegex(ValueError, 'pilot cannot be locked'):
                run.lock_selection(Namespace(output=output))

    def test_select_lock_evaluate_export_without_early_test_access(self):
        rng = np.random.default_rng(25)
        _, schema = data.feature_matrices(rng.uniform(1, 100, (1, 13, 64, 64)).astype(np.float32))
        source = {}
        for split, size in (('train', 40), ('val', 20), ('test', 20)):
            labels = np.arange(size) % 10
            z = np.eye(10)[labels] + rng.normal(scale=.1, size=(size, 10))
            source[split] = {
                'features': z, 'labels': labels,
                'filenames': np.array([f'{split}{index}.tif' for index in range(size)]),
                **{rep: rng.normal(size=(size, width)) for rep, width in REPRESENTATIONS.items()},
            }
        requested_splits = []

        def handcrafted(cache, variant, splits):
            requested_splits.extend(splits)
            return {split: {key: value.copy() for key, value in source[split].items() if key != 'features'}
                    for split in splits}

        def backbone(model, cache, splits):
            requested_splits.extend(splits)
            return {split: {key: source[split][key] for key in ('features', 'labels', 'filenames')}
                    for split in splits}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / 'cache'
            manifest = {'identity': {}, 'schema': schema, 'files': {}, 'classes': data.CLASSES}
            write_json(cache / 'features.json', manifest)
            args = Namespace(
                cache=cache, embeddings=root, metadata=root, output=root / 'study',
                models=['olmoearth_nano'], device='cpu', pilot=False, component='all',
            )
            with (
                patch.object(data, 'load_feature_manifest', return_value=manifest),
                patch.object(data, 'load_handcrafted', side_effect=handcrafted),
                patch.object(data, 'load_backbone', side_effect=backbone),
                patch.object(data, 'backbone_metadata', return_value={
                    'embedding_sha256': 'embedding-fixture', 'backbone_state_sha256': 'backbone-fixture',
                }),
                patch.object(run, 'ALPHAS', (.01,)),
                patch.object(run, 'CS', (1.,)),
                patch.object(run, 'BETAS', (0., 1.)),
                patch.object(run, 'NULL_SEEDS', (0,)),
                patch.object(run, 'BOOTSTRAP', 5),
            ):
                run.select(args)
                self.assertNotIn('test', requested_splits)
                run.lock_selection(args)
                self.assertNotIn('test', requested_splits)
                with (
                    patch.object(run.decoding, 'select_ridge', side_effect=AssertionError('test refit')),
                    patch.object(run.probes, 'select_probe', side_effect=AssertionError('test refit')),
                ):
                    run.evaluate(args)
                result = json.loads((args.output / 'results.json').read_text())
                self.assertEqual(len(result['evaluations']), len(run.expected_job_ids(args.models)))
                self.assertEqual(sum(row['primary'] for row in result['comparisons']), 2)
                self.assertEqual(len(result['paired_reconstruction']), 2)
                destination = root / 'export'
                report.export(args.output, destination)
                self.assertTrue((destination / 'complementarity.csv').exists())
                self.assertTrue((destination / 'accuracy_complementarity.png').exists())
                self.assertTrue((destination / 'README.md').exists())
                report.export(args.output, destination)


if __name__ == '__main__':
    unittest.main()
