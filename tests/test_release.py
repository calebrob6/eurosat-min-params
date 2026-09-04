"""Small, data-free checks for the public EuroSAT reproduction contract."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduce import check_results
from src.data import iter_images
from src.features import patch_features
from src.frontier import CORE_CONFIG, RECIPE, frontier_features
from src.linmodel import predict, predict_reference_class, to_reference_class

ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
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
        with np.load(ROOT / 'submissions/14_reference_class_96/model.npz') as model:
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


if __name__ == '__main__':
    unittest.main()
