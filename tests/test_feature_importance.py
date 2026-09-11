"""Full-pool feature-importance experiment tests."""

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.feature_importance.run import (
    fit_classifier,
    frontier33_indices,
    recursive_elimination,
    run_experiment,
    select_regularization,
)
from src.feature_pool import (
    FULL_POOL_SIZE,
    HISTORICAL_POOL_SIZE,
    full_pool_features,
    historical_pool_features,
)
from src.frontier import POOL_INDICES


class FullPoolTests(unittest.TestCase):
    def test_full_pool_extends_historical_pool_with_all_region_shapes(self) -> None:
        images = (
            np.random.default_rng(31)
            .uniform(1, 1000, (2, 13, 64, 64))
            .astype(np.float32)
        )
        historical, historical_names, historical_families = historical_pool_features(
            images
        )
        full, names, families = full_pool_features(images)

        self.assertEqual(full.shape, (2, FULL_POOL_SIZE))
        np.testing.assert_array_equal(full[:, :HISTORICAL_POOL_SIZE], historical)
        self.assertEqual(names[:HISTORICAL_POOL_SIZE], historical_names)
        self.assertEqual(families[:HISTORICAL_POOL_SIZE], historical_families)
        self.assertEqual(names[381], 'tail_aniso_low_ndvi')
        self.assertEqual(
            names[HISTORICAL_POOL_SIZE:],
            [
                'tail_aniso_low_pan',
                'tail_spread_low_pan',
                'tail_aniso_high_pan',
                'tail_spread_high_pan',
                'tail_aniso_low_ndvi',
                'tail_spread_low_ndvi',
                'tail_aniso_high_ndvi',
                'tail_spread_high_ndvi',
                'tail_aniso_low_ndbi',
                'tail_spread_low_ndbi',
                'tail_aniso_high_ndbi',
                'tail_spread_high_ndbi',
            ],
        )
        self.assertEqual(families[HISTORICAL_POOL_SIZE:], ['region_shape'] * 12)
        np.testing.assert_array_equal(
            frontier33_indices(names)[:32], np.asarray(POOL_INDICES)
        )
        self.assertEqual(frontier33_indices(names)[-1], 381)


class EliminationTests(unittest.TestCase):
    @staticmethod
    def fixture() -> tuple[dict, list[str], list[str]]:
        rng = np.random.default_rng(9)
        weights = rng.normal(size=(3, 12))
        datasets = {}
        for split, size in (('train', 360), ('val', 120), ('test', 120)):
            features = rng.normal(size=(size, 12))
            labels = np.argmax(
                features @ weights.T + rng.normal(scale=0.6, size=(size, 3)), axis=1
            )
            datasets[split] = {'features': features, 'labels': labels}
        return datasets, [f'f{i}' for i in range(12)], ['fixture'] * 12

    def test_standardized_importance_is_invariant_to_input_units(self) -> None:
        datasets, _, _ = self.fixture()
        train = datasets['train']
        indices = np.arange(train['features'].shape[1])
        scales = np.geomspace(0.01, 100, len(indices))
        first = fit_classifier(train['features'], train['labels'], indices, 1.0, 1000)
        second = fit_classifier(
            train['features'] * scales, train['labels'], indices, 1.0, 1000
        )
        np.testing.assert_allclose(
            first.importances, second.importances, rtol=1e-10, atol=1e-10
        )

    def test_recursive_elimination_removes_exact_batches_and_ranks_every_feature(
        self,
    ) -> None:
        datasets, names, families = self.fixture()
        full, _ = select_regularization(
            datasets['train'], datasets['val'], np.arange(12), (0.3, 1.0), max_iter=1000
        )
        expected_first_drop = set(
            np.lexsort((full.feature_indices, full.importances))[:5].tolist()
        )

        scores, importances, removals, sweeps = recursive_elimination(
            datasets,
            names,
            families,
            ['zero', 'one', 'two'],
            c_grid=(0.3, 1.0),
            step=5,
            minimum_features=2,
            max_iter=1000,
        )

        self.assertEqual([row['features_kept'] for row in scores], [12, 7, 2])
        self.assertEqual(len(importances), 12)
        self.assertEqual(
            sorted(row['importance_rank'] for row in importances), list(range(1, 13))
        )
        self.assertEqual(len(removals), 12)
        self.assertEqual(
            [
                sum(row['removal_round'] == round_number for row in removals)
                for round_number in (1, 2)
            ],
            [5, 5],
        )
        self.assertEqual(sum(row['final_kept'] for row in removals), 2)
        self.assertEqual(
            {row['pool_index'] for row in removals if row['removal_round'] == 1},
            expected_first_drop,
        )
        self.assertEqual(sorted(row['pool_index'] for row in removals), list(range(12)))
        self.assertEqual(len(sweeps), 6)
        self.assertEqual(
            [
                sum(row['selected'] for row in sweeps if row['features_kept'] == count)
                for count in (12, 7, 2)
            ],
            [1, 1, 1],
        )

    def test_end_to_end_writes_tables_plots_and_summary(self) -> None:
        datasets, names, families = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            run_experiment(
                datasets,
                {'names': names, 'families': families},
                output,
                c_grid=(0.3, 1.0),
                step=5,
                minimum_features=2,
                max_iter=1000,
                reference_indices=np.array([0, 1, 2]),
            )
            expected = {
                'c_sweep.csv',
                'feature_importances.csv',
                'elimination_order.csv',
                'scores.csv',
                'frontier33.csv',
                'frontier33_c_sweep.csv',
                'frontier33_features.csv',
                'score_by_features.png',
                'top_feature_importances.png',
                'summary.json',
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            self.assertTrue(
                all((output / name).stat().st_size > 0 for name in expected)
            )
            with (output / 'c_sweep.csv').open() as handle:
                sweep_rows = list(csv.DictReader(handle))
            self.assertEqual(len(sweep_rows), 6)
            self.assertEqual(
                [
                    sum(
                        row['selected'] == 'True'
                        for row in sweep_rows
                        if int(row['features_kept']) == count
                    )
                    for count in (12, 7, 2)
                ],
                [1, 1, 1],
            )
            with (output / 'frontier33.csv').open() as handle:
                reference = next(csv.DictReader(handle))
            self.assertEqual(reference['features_kept'], '3')
            self.assertIn(reference['C'], {'0.3', '1.0'})


if __name__ == '__main__':
    unittest.main()
