"""Small, data-free checks for the public EuroSAT reproduction contract."""
import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduce import (
    MODEL_PATH, check_results, evaluate, extract_features, fitted_reference,
    learning_curves, write_csv,
)
from src.data import BAND_NAMES, B_SWIR1, B_SWIR2, TIFF_BAND_NAMES, iter_images
from src.features import patch_features
from src.frontier import CORE_CONFIG, RECIPE, frontier_features
from src.linmodel import predict, predict_reference_class, to_reference_class

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_band_labels_are_physical_without_changing_legacy_indices(self):
        expected = [
            'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
            'B08', 'B09', 'B10', 'B11', 'B12', 'B8A',
        ]
        self.assertEqual(BAND_NAMES, expected)
        self.assertEqual(TIFF_BAND_NAMES, expected)
        self.assertEqual((B_SWIR1, B_SWIR2), (11, 12))
        with np.load(MODEL_PATH, allow_pickle=False) as model:
            self.assertEqual(BAND_NAMES, model['tiff_band_names'].tolist())
            np.testing.assert_array_equal(model['legacy_swir_indices'], [B_SWIR1, B_SWIR2])

    def test_percentile_names_match_values(self):
        images = np.random.default_rng(0).uniform(1, 100, (2, 13, 64, 64)).astype(np.float32)
        features, names = patch_features(images, grad_scales=0)
        for percentile in (10, 25, 50, 75, 90):
            expected = np.percentile(images.reshape(2, 13, -1), percentile, axis=2)
            for band in range(13):
                np.testing.assert_allclose(
                    features[:, names.index(f'p{percentile}_b{band}')],
                    expected[:, band], rtol=1e-6,
                )

    def test_frontier_schema_matches_checkpoint(self):
        images = np.random.default_rng(1).uniform(1, 100, (2, 13, 64, 64)).astype(np.float32)
        features, names = frontier_features(images)
        with np.load(MODEL_PATH, allow_pickle=False) as model:
            self.assertEqual(str(model['recipe']), RECIPE)
            self.assertEqual(names, model['feature_names'].tolist())
            self.assertEqual(model['W'].shape, (9, 33))
            self.assertEqual(model['W'].size + model['b'].size, 306)
            for key, value in CORE_CONFIG.items():
                np.testing.assert_array_equal(model[key], value)
        self.assertEqual(features.shape, (2, 33))
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaises(ValueError):
            frontier_features(images.astype(np.uint16))

    def test_reference_class_prediction(self):
        rng = np.random.default_rng(2)
        features, w, b = rng.normal(size=(40, 33)), rng.normal(size=(10, 33)), rng.normal(size=10)
        wr, br = to_reference_class(w, b)
        np.testing.assert_array_equal(
            predict(features, w, b, np.arange(33)),
            predict_reference_class(features, wr, br, np.arange(33)),
        )
        self.assertEqual(wr.size + br.size, 306)

    def test_raw_iterator_preserves_split_order(self):
        labels = np.array([1, 2, 3])
        with patch('src.data.list_split', return_value=(['one', 'two', 'three'], labels)):
            with patch('src.data.read_tif', side_effect=[
                np.full((13, 64, 64), i, dtype=np.float32) for i in (1, 2, 3)
            ]) as read:
                batches = list(iter_images('val', batch_size=2))
                self.assertEqual([len(x) for x, _ in batches], [2, 1])
                self.assertEqual([call.args[0] for call in read.call_args_list], ['one', 'two', 'three'])
                np.testing.assert_array_equal(np.concatenate([y for _, y in batches]), labels)

    def test_extraction_streams_only_current_feature_sets(self):
        images = np.random.default_rng(3).uniform(1, 100, (2, 13, 64, 64)).astype(np.float32)
        labels = np.array([0, 1])
        with np.load(MODEL_PATH, allow_pickle=False) as model:
            names = model['feature_names'].tolist()

        def split_members(split):
            return [f'{split}_{i}.tif' for i in range(2)], labels

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('reproduce.SIZES', (2, 2, 2)):
                with patch('reproduce.list_split', side_effect=split_members):
                    with patch('reproduce.iter_images', return_value=[(images, labels)]) as stream:
                        with patch('reproduce.frontier_features', return_value=(np.zeros((2, 33)), names)):
                            data = extract_features(2, output)
            self.assertEqual([call.args for call in stream.call_args_list],
                             [('train', 2), ('val', 2), ('test', 2)])
            flat = images.reshape(2, 13, -1)
            expected = np.stack((flat.mean(2), flat.std(2), flat.min(2), flat.max(2)), axis=2)
            for split in ('train', 'val', 'test'):
                self.assertEqual(set(data[split]), {'frontier', 'imagestats', 'labels', 'filenames'})
                np.testing.assert_array_equal(data[split]['imagestats'], expected.reshape(2, 52))
            self.assertEqual((output / 'features_306.csv').read_text(),
                             (ROOT / 'results/eurosat_306_features.csv').read_text())
            with self.assertRaisesRegex(ValueError, 'batch_size must be positive'):
                extract_features(0, output)

    def test_reference_fit_folds_scaler_and_removes_one_class(self):
        features = np.random.default_rng(4).normal(size=(100, 33)).astype(np.float32)
        labels = np.tile(np.arange(10), 10)
        w, b, indices = fitted_reference(features, labels, 3.0)
        self.assertEqual(w.shape, (9, 33))
        self.assertEqual(b.shape, (9,))
        np.testing.assert_array_equal(indices, np.arange(33))
        self.assertTrue(np.isfinite(w).all())

    def test_evaluation_fits_train_and_selects_c_on_validation(self):
        data = self.synthetic_data()

        def fake_fit(features, labels, C):
            self.assertIs(features, data['train']['imagestats'])
            self.assertIs(labels, data['train']['labels'])
            return np.full((9, 52), C), np.zeros(9), np.arange(52)

        def fake_predict(features, model):
            if features is data['test']['imagestats']:
                self.assertEqual(model[0][0, 0], 1)
            correct = features.shape[1] == 33 or model[0][0, 0] == 1
            return np.arange(10) if correct else np.zeros(10, dtype=int)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('reproduce.BASELINE_CS', (1, 3)):
                with patch('reproduce.fitted_reference', side_effect=fake_fit) as fit:
                    with patch('reproduce.prediction', side_effect=fake_predict):
                        evaluate(data, output, refit=False)
            self.assertEqual(fit.call_count, 2)
            with (output / 'models.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row['model'], row['split']) for row in rows],
                             [('306', 'val'), ('306', 'test'), ('imagestats', 'val'), ('imagestats', 'test')])
            self.assertEqual([int(row['parameters']) for row in rows], [306, 306, 477, 477])
            self.assertFalse((output / 'eurosat_33.npz').exists())

    def test_refit_uses_fixed_training_recipe_and_only_output_checkpoint(self):
        data = self.synthetic_data()
        original = MODEL_PATH.read_bytes()

        def fake_fit(features, labels, C):
            self.assertIs(labels, data['train']['labels'])
            self.assertIs(features, data['train']['frontier' if features.shape[1] == 33 else 'imagestats'])
            self.assertEqual(C, 3 if features.shape[1] == 33 else 300)
            return np.zeros((9, features.shape[1]), dtype=np.float32), np.zeros(9), np.arange(features.shape[1])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch('reproduce.BASELINE_CS', (300,)):
                with patch('reproduce.fitted_reference', side_effect=fake_fit):
                    evaluate(data, output, refit=True)
            with np.load(output / 'eurosat_33.npz', allow_pickle=False) as model:
                self.assertEqual(float(model['C']), 3.0)
                self.assertEqual(model['feature_names'].tolist(), data['feature_names'])
                self.assertEqual(str(model['recipe']), RECIPE)
            self.assertEqual(MODEL_PATH.read_bytes(), original)
        with self.assertRaisesRegex(ValueError, 'must not overwrite'):
            evaluate(data, MODEL_PATH.parent, refit=True)

    def test_learning_curves_keep_fixed_c_and_spatial_membership(self):
        data = self.synthetic_data()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for split in ('train', 'val', 'test'):
                (output / f'eurosat-spatial-{split}.txt').write_text(
                    '\n'.join(str(name).replace('.tif', '.jpg') for name in data[split]['filenames']) + '\n')
            with patch('reproduce.DATA_ROOT', directory), patch('reproduce.SIZES', (10, 10, 10)):
                with patch('reproduce.FRACTIONS', (100,)):
                    with patch('reproduce.fit_folded_logreg') as fit:
                        fit.side_effect = lambda x, y, C: (np.zeros((10, x.shape[1])), np.zeros(10), np.arange(x.shape[1]))
                        learning_curves(data, output)
                    self.assertEqual(fit.call_count, 20)
                    for call in fit.call_args_list:
                        features, labels = call.args
                        family = 'frontier' if features.shape[1] == 33 else 'imagestats'
                        np.testing.assert_array_equal(features, data['train'][family])
                        np.testing.assert_array_equal(labels, data['train']['labels'])
                        self.assertEqual(call.kwargs['C'], 3 if family == 'frontier' else 300)
                    path = output / 'eurosat-spatial-train.txt'
                    path.write_text(path.read_text().replace('train_0.jpg', 'unknown.jpg'))
                    with self.assertRaisesRegex(ValueError, 'unknown image'):
                        learning_curves(data, output)

    def synthetic_data(self):
        data = {}
        for i, split in enumerate(('train', 'val', 'test')):
            data[split] = {
                'frontier': np.full((10, 33), i, dtype=np.float32),
                'imagestats': np.full((10, 52), i, dtype=np.float32),
                'labels': np.arange(10),
                'filenames': np.asarray([f'{split}_{j}.tif' for j in range(10)]),
            }
        with np.load(MODEL_PATH, allow_pickle=False) as model:
            data['feature_names'] = model['feature_names'].tolist()
        return data

    def test_published_count_drift_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            expected = (ROOT / 'results/eurosat_models.csv').read_text()
            (path / 'models.csv').write_text(expected)
            for generated, source in [('per_class.csv', 'eurosat_per_class.csv'),
                                      ('features_306.csv', 'eurosat_306_features.csv'),
                                      ('imagestats_c_sweep.csv', 'imagestats_c_sweep.csv')]:
                (path / generated).write_text((ROOT / 'results' / source).read_text())
            check_results(path, False)
            (path / 'models.csv').write_text(expected.replace('5186', '5185'))
            with self.assertRaisesRegex(ValueError, 'results differ'):
                check_results(path, False)

    def test_check_curves_rejects_nonfinite_metrics_and_wrong_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            for generated, source in [('models.csv', 'eurosat_models.csv'),
                                      ('per_class.csv', 'eurosat_per_class.csv'),
                                      ('features_306.csv', 'eurosat_306_features.csv'),
                                      ('imagestats_c_sweep.csv', 'imagestats_c_sweep.csv')]:
                (output / generated).write_text((ROOT / 'results' / source).read_text())
            for family in ('frontier', 'imagestats'):
                with (ROOT / 'results' / f'{family}_fractions.csv').open() as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    row['parameters'] = row.pop('learned_parameters')
                    row['C'] = row.pop('regularization_C')
                write_csv(output / f'{family}_fractions.csv', rows)
            check_results(output, True)
            for field, value in [('seed_0_test_accuracy', 'nan'), ('n_train', '100'),
                                 ('parameters', '306'), ('C', '3')]:
                changed = [{**row} for row in rows]
                changed[0][field] = value
                write_csv(output / 'imagestats_fractions.csv', changed)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    check_results(output, True)

    def test_cpu_cli_exposes_current_refit_without_importing_torch(self):
        completed = subprocess.run(
            [sys.executable, str(ROOT / 'reproduce.py'), '--help'],
            check=True, capture_output=True, text=True,
        )
        self.assertIn('--refit', completed.stdout)
        self.assertNotIn('--refit-306', completed.stdout)
        subprocess.run(
            [sys.executable, '-c', 'import reproduce, sys; assert "torch" not in sys.modules'],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )


if __name__ == '__main__':
    unittest.main()
