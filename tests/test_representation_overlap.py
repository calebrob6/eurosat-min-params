"""Data-free representation-study schema and alignment checks."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.representation_overlap.data import align, feature_matrices
from experiments.representation_overlap.protocol import json_value, require_same, write_json
from src.data import TIFF_BAND_NAMES
from src.frontier import POOL_INDICES, frontier_features


class RepresentationDataTests(unittest.TestCase):
    def test_alignment_is_by_sample_id(self):
        names = np.array(['a.tif', 'b.tif', 'c.tif'])
        labels = np.array([0, 1, 2])
        permutation = np.array([2, 0, 1])
        features = np.arange(6).reshape(3, 2)
        actual = align(features[permutation], names[permutation], labels[permutation], names, labels)
        np.testing.assert_array_equal(actual, features)

    def test_alignment_rejects_invalid_membership(self):
        names = np.array(['a.tif', 'b.tif'])
        features, labels = np.ones((2, 3)), np.array([0, 1])
        for invalid in (np.array(['a.tif', 'a.tif']), np.array(['a.tif', 'c.tif'])):
            with self.assertRaises(ValueError):
                align(features, invalid, labels, names, labels)
        with self.assertRaisesRegex(ValueError, 'labels'):
            align(features, names, labels[::-1], names, labels)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            align(features * np.nan, names, labels, names, labels)

    def test_raw_recipe_preserves_frontier_and_pool_order(self):
        from experiments.representation_overlap.report import physical_definition

        images = np.random.default_rng(7).uniform(1, 1000, (2, 13, 64, 64)).astype(np.float32)
        matrices, schema = feature_matrices(images)
        expected, names = frontier_features(images)
        np.testing.assert_array_equal(matrices['frontier33'], expected)
        np.testing.assert_array_equal(matrices['frontier33'][:, :32], matrices['pool377'][:, POOL_INDICES])
        self.assertEqual(schema['frontier33']['names'], names)
        self.assertEqual(schema['frontier33']['families'][-1], 'region_shape')
        self.assertEqual(schema['frontier33']['pool_indices'][-1], None)
        self.assertEqual(matrices['pool377'].shape, (2, 377))
        self.assertEqual(len(schema['pool377']['families']), 377)
        self.assertEqual(matrices['imagestats52'].shape, (2, 52))
        for recipe in schema.values():
            for name in recipe['names']:
                self.assertTrue(physical_definition(name))

    def test_b10_ablation_removes_all_b10_dependence_without_mutation(self):
        images = np.random.default_rng(8).uniform(1, 1000, (2, 13, 64, 64)).astype(np.float32)
        original = images.copy()
        changed = images.copy()
        changed[:, TIFF_BAND_NAMES.index('B10')] += 10000
        first, schema = feature_matrices(images, b10_zeroed=True)
        second, other_schema = feature_matrices(changed, b10_zeroed=True)
        self.assertEqual(schema, other_schema)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        np.testing.assert_array_equal(images, original)

    def test_configuration_and_undefined_scores_are_explicit(self):
        self.assertEqual(json_value({'score': float('nan'), 'undefined_features': 1}),
                         {'score': None, 'undefined_features': 1})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            write_json(path, {'seed': 0, 'grid': (1, 2)})
            require_same(path, {'seed': 0, 'grid': [1, 2]})
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                require_same(path, {'seed': 1, 'grid': [1, 2]})

    def test_test_evaluation_requires_an_intact_lock(self):
        from experiments.representation_overlap.run import locked_jobs
        from experiments.representation_overlap.protocol import sha256

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaisesRegex(ValueError, 'requires a locked selection'):
                locked_jobs(output)
            artifact = output / 'selected.json'
            artifact.write_text('original')
            write_json(output / 'locked.json', {
                'files': {'selected.json': sha256(artifact)}, 'jobs': [],
            })
            self.assertEqual(locked_jobs(output), [])
            artifact.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'locked artifact changed'):
                locked_jobs(output)

    def test_prespecified_coverage_includes_band_controls(self):
        from experiments.representation_overlap.protocol import MODELS
        from experiments.representation_overlap.run import expected_job_ids

        identifiers = expected_job_ids(list(MODELS))
        self.assertEqual(len(identifiers), 249)
        for model in ('olmoearth_nano', 'olmoearth_base'):
            for rep in ('frontier33', 'pool377'):
                self.assertIn(f'probe-{model}-b10_zeroed-{rep}-combined', identifiers)
                self.assertIn(f'ridge-{model}-b10_zeroed-{rep}-ordinary-h_to_z', identifiers)
                self.assertIn(f'ridge-{model}-b10_zeroed-{rep}-ordinary-z_to_h', identifiers)
        self.assertEqual(len(expected_job_ids(['olmoearth_nano'], pilot=True)), 9)

    def test_handcrafted_manifest_rejects_corruption_and_source_drift(self):
        from experiments.representation_overlap.data import load_feature_manifest
        from experiments.representation_overlap.protocol import save_npz, sha256

        identity = {'schema': 'fixture'}
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            names = [f'{variant}_{split}.npz' for variant in ('original', 'b10_zeroed')
                     for split in ('train', 'val', 'test')]
            for name in names:
                save_npz(cache / name, fixture=np.arange(3))
            manifest = {'identity': identity, 'files': {name: sha256(cache / name) for name in names}}
            write_json(cache / 'features.json', manifest)
            with patch('experiments.representation_overlap.data.extraction_identity', return_value=identity):
                self.assertEqual(load_feature_manifest(cache), manifest)
                save_npz(cache / names[0], fixture=np.arange(4))
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    load_feature_manifest(cache)
            with patch('experiments.representation_overlap.data.extraction_identity', return_value={'schema': 'new'}):
                with self.assertRaisesRegex(ValueError, 'generating source/configuration changed'):
                    load_feature_manifest(cache)


if __name__ == '__main__':
    unittest.main()
