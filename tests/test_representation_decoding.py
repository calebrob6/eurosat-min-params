"""Small, data-free numerical checks for directional representation decoding."""

from dataclasses import asdict
from unittest import mock
import unittest

import numpy as np

from experiments.representation_overlap.decoding import (
    RidgeMap,
    bootstrap_r2,
    fit_target_pca,
    linear_cka,
    paired_r2,
    pca_diagnostics,
    prepare_ridge_source,
    score_predictions,
    select_ridge,
)


class RidgeTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(21)
        self.xtr = self.rng.normal(size=(90, 5)) * [1, 2, 3, 4, 5] + [3, 1, -2, 5, 0]
        self.xva = self.rng.normal(size=(40, 5)) * [2, 1, 4, 3, 6] - 2
        self.coef = self.rng.normal(size=(5, 3))
        self.intercept = np.array([3.0, -4.0, 8.0])
        self.ytr = self.xtr @ self.coef + self.intercept
        self.yva = self.xva @ self.coef + self.intercept

    def test_affine_recovery_and_raw_unit_serialization(self):
        for scoring in ('native', 'macro'):
            with self.subTest(scoring=scoring):
                fitted, sweep = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [0], scoring=scoring)
                np.testing.assert_allclose(fitted.coef, self.coef, atol=1e-12)
                np.testing.assert_allclose(fitted.intercept, self.intercept, atol=1e-12)
                heldout = self.rng.normal(size=(15, 5)) * 3 + 12
                np.testing.assert_allclose(fitted.predict(heldout), heldout @ self.coef + self.intercept, atol=1e-12)
                self.assertEqual(set(asdict(fitted)), {'coef', 'intercept', 'alpha', 'source_constant'})
                restored = RidgeMap(**asdict(fitted))
                np.testing.assert_array_equal(restored.predict(heldout), fitted.predict(heldout))
                self.assertTrue(sweep[0]['selected_on_boundary'])
                self.assertEqual(sweep[0]['boundary'], 'only')

    def test_mean_loss_scaling_matches_normal_equation(self):
        alpha = 0.7
        fitted, _ = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [alpha])
        mean, scale = self.xtr.mean(0), self.xtr.std(0)
        design = (self.xtr - mean) / scale
        centered_target = self.ytr - self.ytr.mean(0)
        expected = np.linalg.solve(design.T @ design + len(design) * alpha * np.eye(5),
                                   design.T @ centered_target)
        np.testing.assert_allclose(fitted.coef, expected / scale[:, None], atol=1e-12)
        np.testing.assert_allclose(fitted.intercept, self.ytr.mean(0) - mean @ fitted.coef, atol=1e-12)
        standardized_residual = centered_target - design @ expected
        np.testing.assert_allclose(design.T @ standardized_residual, len(design) * alpha * expected, atol=1e-11)
        self.assertGreater(np.linalg.norm(design.T @ standardized_residual), 1)

    def test_preprocessing_is_train_only_and_exact_constants_are_ignored(self):
        xtr = np.column_stack([self.xtr, np.full(len(self.xtr), 0.1)])
        xva = np.column_stack([self.xva, np.linspace(-1000, 1000, len(self.xva))])
        fitted, sweep = select_ridge(xtr, self.ytr, xva, self.yva, [0.3])
        expected, _ = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [0.3])
        np.testing.assert_array_equal(fitted.source_constant, [False] * 5 + [True])
        np.testing.assert_array_equal(fitted.coef[-1], 0)
        np.testing.assert_allclose(fitted.coef[:-1], expected.coef, atol=1e-12)
        np.testing.assert_allclose(fitted.intercept, expected.intercept, atol=1e-12)
        np.testing.assert_allclose(fitted.predict(xva), expected.predict(self.xva), atol=1e-12)
        self.assertEqual(sweep[0]['source_constant_changed_in_validation'], 1)
        changed, _ = select_ridge(xtr, self.ytr, xva * 100, self.yva - 1000, [0.3])
        np.testing.assert_array_equal(changed.coef, fitted.coef)
        np.testing.assert_array_equal(changed.intercept, fitted.intercept)

    def test_near_constant_is_not_removed(self):
        xtr = (1 + np.arange(20) * 1e-12)[:, None]
        xva = xtr[::-1]
        fitted, _ = select_ridge(xtr, np.arange(20)[:, None], xva, np.arange(20)[::-1, None], [0])
        self.assertFalse(fitted.source_constant[0])
        self.assertGreater(abs(fitted.coef[0, 0]), 1e10)

    def test_all_source_constants_stronger_tie_and_boundary(self):
        xtr, xva = np.full((8, 2), 0.1), np.ones((4, 2))
        ytr, yva = np.arange(8)[:, None], np.arange(4)[:, None]
        fitted, sweep = select_ridge(xtr, ytr, xva, yva, [0, 0.2, 10, 1, 10])
        self.assertEqual(fitted.alpha, 10)
        self.assertEqual([row['alpha'] for row in sweep], [10, 1, 0.2, 0])
        self.assertEqual(sum(row['selected'] for row in sweep), 1)
        self.assertTrue(sweep[0]['selected_on_boundary'])
        self.assertEqual(sweep[0]['boundary'], 'upper')
        np.testing.assert_array_equal(fitted.coef, 0)
        np.testing.assert_array_equal(fitted.predict(xva), np.full((4, 1), 3.5))

    def test_selection_uses_macro_not_native_target_variance(self):
        x = np.array([-1, 1, -1, 1], dtype=float)[:, None]
        train_target = np.column_stack([100 * x[:, 0], x[:, 0]])
        val_target = np.column_stack([100 * x[:, 0], 0.1 * x[:, 0]])
        native, native_sweep = select_ridge(x, train_target, x, val_target, [0, 9, 99])
        macro, macro_sweep = select_ridge(x, train_target, x, val_target, [0, 9, 99], scoring='macro')
        self.assertEqual(native.alpha, 0)
        self.assertEqual(macro.alpha, 9)
        for scoring, fitted, sweep in (('native', native, native_sweep), ('macro', macro, macro_sweep)):
            row = next(row for row in sweep if row['selected'])
            summary, _ = score_predictions(val_target, fitted.predict(x))
            self.assertAlmostEqual(row['validation_score'], summary[f'{scoring}_r2'], places=12)
        self.assertFalse(next(row for row in macro_sweep if row['selected'])['selected_on_boundary'])

    def test_macro_handles_constant_train_and_validation_targets(self):
        ytr = np.column_stack([self.ytr, np.full(len(self.ytr), 0.1)])
        yva = np.column_stack([self.yva, np.full(len(self.yva), 0.1)])
        fitted, _ = select_ridge(self.xtr, ytr, self.xva, yva, [0, 0.1], scoring='macro')
        np.testing.assert_array_equal(fitted.coef[:, -1], 0)
        np.testing.assert_array_equal(fitted.predict(self.xva)[:, -1], 0.1)
        summary, _ = score_predictions(yva, fitted.predict(self.xva))
        self.assertEqual(summary['undefined_features'], 1)
        self.assertAlmostEqual(summary['macro_r2'], 1)

    def test_one_decomposition_for_whole_positive_grid(self):
        original = np.linalg.eigh
        with mock.patch('numpy.linalg.eigh', wraps=original) as decomposition:
            fitted, sweep = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [0.001, 0.01, 0.1, 1, 10])
        self.assertEqual(decomposition.call_count, 1)
        self.assertEqual(len(sweep), 5)
        self.assertEqual(fitted.alpha, 0.001)
        for row in sweep:
            fixed, _ = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [row['alpha']])
            expected, _ = score_predictions(self.yva, fixed.predict(self.xva))
            self.assertAlmostEqual(row['validation_score'], expected['native_r2'], places=11)

    def test_prepared_source_reuses_decomposition_across_targets_and_shuffles(self):
        grid = np.logspace(-8, 4, 13)
        targets = [
            (self.ytr, self.yva, 'native'),
            (self.ytr[:, :1] * 5, self.yva[:, :1] * 5, 'macro'),
            (self.ytr[self.rng.permutation(len(self.ytr))],
             self.yva[self.rng.permutation(len(self.yva))], 'macro'),
        ]
        original = np.linalg.eigh
        cached = []
        with mock.patch('numpy.linalg.eigh', wraps=original) as decomposition:
            source = prepare_ridge_source(self.xtr)
            for ytr, yva, scoring in targets:
                cached.append(select_ridge(self.xtr, ytr, self.xva, yva, grid,
                                           scoring=scoring, source=source))
                if len(cached) == 1:
                    validation_cache = source._validation
                else:
                    self.assertIs(source._validation, validation_cache)
        self.assertEqual(decomposition.call_count, 1)
        for (ytr, yva, scoring), (actual, sweep) in zip(targets, cached, strict=True):
            expected, expected_sweep = select_ridge(self.xtr, ytr, self.xva, yva, grid, scoring=scoring)
            np.testing.assert_array_equal(actual.coef, expected.coef)
            np.testing.assert_array_equal(actual.intercept, expected.intercept)
            self.assertEqual(actual.alpha, expected.alpha)
            self.assertEqual(sweep, expected_sweep)

    def test_prepared_source_rejects_changed_training_and_refreshes_validation(self):
        source = prepare_ridge_source(self.xtr)
        for changed in (self.xtr[::-1], self.xtr + 0.001, self.xtr[:, ::-1]):
            with self.subTest(shape=changed.shape), self.assertRaisesRegex(ValueError, 'does not match'):
                select_ridge(changed, self.ytr, self.xva, self.yva, [1], source=source)
        initial, _ = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [1], source=source)
        validation_cache = source._validation
        refreshed, sweep = select_ridge(self.xtr, self.ytr, self.xva * 2, self.yva, [1], source=source)
        expected, expected_sweep = select_ridge(self.xtr, self.ytr, self.xva * 2, self.yva, [1])
        self.assertIsNot(source._validation, validation_cache)
        np.testing.assert_array_equal(initial.coef, refreshed.coef)
        np.testing.assert_array_equal(refreshed.predict(self.xva), expected.predict(self.xva))
        self.assertEqual(sweep, expected_sweep)
        with self.assertRaisesRegex(ValueError, 'prepare_ridge_source'):
            select_ridge(self.xtr, self.ytr, self.xva, self.yva, [1], source=object())
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            prepare_ridge_source(self.xtr * np.nan)

    def test_prepared_source_zero_alpha_uses_one_lazy_svd(self):
        original_eigh, original_svd = np.linalg.eigh, np.linalg.svd
        with mock.patch('numpy.linalg.eigh', wraps=original_eigh) as eigen_decomposition:
            with mock.patch('numpy.linalg.svd', wraps=original_svd) as singular_decomposition:
                source = prepare_ridge_source(self.xtr)
                select_ridge(self.xtr, self.ytr, self.xva, self.yva, [1], source=source)
                validation_cache = source._validation
                fitted, sweep = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [0, 0.1], source=source)
                self.assertIsNot(source._validation, validation_cache)
                select_ridge(self.xtr, self.ytr * 2, self.xva, self.yva * 2, [0], source=source)
        self.assertEqual(eigen_decomposition.call_count, 1)
        self.assertEqual(singular_decomposition.call_count, 1)
        expected, expected_sweep = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [0, 0.1])
        np.testing.assert_array_equal(fitted.coef, expected.coef)
        self.assertEqual(sweep, expected_sweep)

    def test_prepared_source_constants_and_wide_designs(self):
        for train, validation in ((np.full((8, 3), 0.1), self.rng.normal(size=(4, 3))),
                                  (self.rng.normal(size=(5, 9)), self.rng.normal(size=(4, 9)))):
            with self.subTest(shape=train.shape):
                target = self.rng.normal(size=(len(train), 2))
                val_target = self.rng.normal(size=(len(validation), 2))
                source = prepare_ridge_source(train)
                actual, sweep = select_ridge(train, target, validation, val_target, [0, 0.1, 100], source=source)
                expected, expected_sweep = select_ridge(train, target, validation, val_target, [0, 0.1, 100])
                np.testing.assert_array_equal(actual.coef, expected.coef)
                self.assertEqual(sweep, expected_sweep)
                actual.source_constant[:] = False
                np.testing.assert_array_equal(source.source_constant, np.all(train == train[0], axis=0))

    def test_negative_control_can_select_strongest_regularization(self):
        source_values = np.array([-1.0, 1.0, -1.0, 1.0])[:, None]
        source = prepare_ridge_source(source_values)
        fitted, sweep = select_ridge(source_values, source_values, source_values, -source_values,
                                    np.logspace(-8, 4, 13), source=source)
        self.assertEqual(fitted.alpha, 1e4)
        selected = next(row for row in sweep if row['selected'])
        self.assertEqual(selected['boundary'], 'upper')
        self.assertTrue(selected['selected_on_boundary'])
        self.assertLess(selected['validation_score'], 0)

    def test_singular_and_wide_designs(self):
        for xtr, xva in ((self.xtr[:8], self.xva[:4]),
                         (self.xtr[:8].T, self.xva[:8].T)):
            xtr = np.column_stack([xtr, xtr[:, 0], xtr[:, 1] + xtr[:, 0]])
            xva = np.column_stack([xva, xva[:, 0], xva[:, 1] + xva[:, 0]])
            target = self.rng.normal(size=(len(xtr), 2))
            val_target = self.rng.normal(size=(len(xva), 2))
            with self.subTest(shape=xtr.shape):
                original = np.linalg.svd
                with mock.patch('numpy.linalg.svd', wraps=original) as decomposition:
                    fitted, _ = select_ridge(xtr, target, xva, val_target, [0])
                self.assertEqual(decomposition.call_count, 1)
                mean, scale = xtr.mean(0), xtr.std(0)
                standardized_coef = np.linalg.lstsq((xtr - mean) / scale, target - target.mean(0), rcond=None)[0]
                expected = ((xva - mean) / scale) @ standardized_coef + target.mean(0)
                np.testing.assert_allclose(fitted.predict(xva), expected, atol=1e-11)
                positive, _ = select_ridge(xtr, target, xva, val_target, [0.5])
                self.assertTrue(np.isfinite(positive.predict(xva)).all())

    def test_invalid_ridge_inputs(self):
        for grid in ([], [-1], [np.nan], [np.inf], [[1]]):
            with self.subTest(grid=grid), self.assertRaisesRegex(ValueError, 'alphas'):
                select_ridge(self.xtr, self.ytr, self.xva, self.yva, grid)
        for position in range(4):
            values = [self.xtr.copy(), self.ytr.copy(), self.xva.copy(), self.yva.copy()]
            values[position][0, 0] = np.nan
            with self.subTest(position=position), self.assertRaisesRegex(ValueError, 'non-finite'):
                select_ridge(*values, [1])
        for scoring in ('native', 'macro'):
            with self.subTest(scoring=scoring), self.assertRaisesRegex(ValueError, 'undefined whole validation'):
                select_ridge(self.xtr, self.ytr, self.xva, np.full(self.yva.shape, 0.1), [1], scoring=scoring)
        with self.assertRaisesRegex(ValueError, 'scoring'):
            select_ridge(self.xtr, self.ytr, self.xva, self.yva, [1], scoring='correlation')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            select_ridge(self.xtr, self.ytr, self.xva[:, :-1], self.yva, [1])
        with self.assertRaisesRegex(ValueError, 'row counts'):
            select_ridge(self.xtr, self.ytr[:-1], self.xva, self.yva, [1])
        fitted, _ = select_ridge(self.xtr, self.ytr, self.xva, self.yva, [1])
        with self.assertRaisesRegex(ValueError, 'source columns'):
            fitted.predict(self.xva[:, :1])
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            fitted.predict(self.xva * np.nan)


class DirectionalityTests(unittest.TestCase):
    def test_h_contains_a_and_b_but_z_only_a(self):
        rng = np.random.default_rng(7)
        representations = []
        for rows in (160, 80, 60):
            value = rng.normal(size=(rows, 2))
            value -= value.mean(0)
            orthogonal, _ = np.linalg.qr(value)
            representations.append(orthogonal * np.sqrt(rows))
        train, validation, heldout = representations
        h_to_z, _ = select_ridge(train, train[:, :1], validation, validation[:, :1], [0])
        z_to_h, _ = select_ridge(train[:, :1], train, validation[:, :1], validation, [0], scoring='macro')
        forward, _ = score_predictions(heldout[:, :1], h_to_z.predict(heldout))
        reverse, per_feature = score_predictions(heldout, z_to_h.predict(heldout[:, :1]))
        self.assertAlmostEqual(forward['native_r2'], 1)
        np.testing.assert_allclose(per_feature, [1, 0], atol=1e-12)
        self.assertAlmostEqual(reverse['native_r2'], 0.5)

    def test_independent_data_has_no_recoverable_signal(self):
        rng = np.random.default_rng(32)
        source, target = rng.normal(size=(1500, 4)), rng.normal(size=(1500, 3))
        fitted, _ = select_ridge(source[:300], target[:300], source[300:500], target[300:500], [0, 0.1, 1, 100])
        summary, _ = score_predictions(target[500:], fitted.predict(source[500:]))
        self.assertLess(summary['native_r2'], 0.02)
        self.assertGreater(summary['native_r2'], -0.1)

    def test_affine_residual_concatenation_is_a_reparameterization(self):
        rng = np.random.default_rng(9)
        z, h = rng.normal(size=(40, 4)), rng.normal(size=(40, 3))
        decoder_coef, decoder_intercept = rng.normal(size=(4, 3)), rng.normal(size=3)
        residual = h - z @ decoder_coef - decoder_intercept
        z_coef, h_coef, intercept = rng.normal(size=(4, 2)), rng.normal(size=(3, 2)), rng.normal(size=2)
        ordinary = z @ z_coef + h @ h_coef + intercept
        reparameterized = z @ (z_coef + decoder_coef @ h_coef) + residual @ h_coef + intercept + decoder_intercept @ h_coef
        np.testing.assert_allclose(ordinary, reparameterized, atol=1e-12)
        ordinary_design = np.column_stack([np.ones(len(z)), z, h])
        residual_design = np.column_stack([np.ones(len(z)), z, residual])
        np.testing.assert_allclose(ordinary_design @ np.linalg.pinv(ordinary_design),
                                   residual_design @ np.linalg.pinv(residual_design), atol=1e-12)


class ScoreTests(unittest.TestCase):
    def test_negative_constant_and_native_aggregate_scores(self):
        actual = np.column_stack([np.arange(4), np.arange(4) * 10, np.full(4, 0.1)])
        predicted = actual.copy()
        predicted[:, 0] = 10
        predicted[:, 2] = 2.1
        summary, per_feature = score_predictions(actual, predicted)
        self.assertAlmostEqual(per_feature[0], 1 - 294 / 5)
        self.assertEqual(per_feature[1], 1)
        self.assertTrue(np.isnan(per_feature[2]))
        self.assertEqual(summary['defined_features'], 2)
        self.assertEqual(summary['undefined_features'], 1)
        self.assertAlmostEqual(summary['native_r2'], 1 - (294 + 16) / 505)
        self.assertAlmostEqual(summary['macro_r2'], np.mean(per_feature[:2]))
        self.assertAlmostEqual(summary['median_r2'], np.median(per_feature[:2]))
        self.assertAlmostEqual(summary['q25_r2'], np.quantile(per_feature[:2], 0.25))
        self.assertAlmostEqual(summary['q75_r2'], np.quantile(per_feature[:2], 0.75))
        for key in ('fraction_r2_gt_05', 'fraction_r2_gt_08', 'fraction_r2_gt_09'):
            self.assertEqual(summary[key], 0.5)

    def test_all_constant_and_single_row_scores_are_undefined(self):
        for actual in (np.full((30, 2), 0.1), np.array([[1.0, 3.0]])):
            for prediction in (actual, actual + 2):
                summary, per_feature = score_predictions(actual, prediction)
                self.assertEqual(summary['defined_features'], 0)
                self.assertEqual(summary['undefined_features'], 2)
                self.assertTrue(np.isnan(per_feature).all())
                self.assertTrue(np.isnan(summary['native_r2']))
                self.assertTrue(np.isnan(summary['macro_r2']))
                self.assertTrue(np.isnan(summary['fraction_r2_gt_05']))

    def test_evaluation_mean_not_training_mean_or_prediction_correlation(self):
        actual = np.arange(8, dtype=float)[:, None] + 100
        prediction = actual - 100
        summary, _ = score_predictions(actual, prediction)
        self.assertAlmostEqual(summary['native_r2'], 1 - 80000 / 42)
        self.assertAlmostEqual(np.corrcoef(actual[:, 0], prediction[:, 0])[0, 1], 1)

    def test_native_orthogonal_invariance_and_nonuniform_rescaling(self):
        rng = np.random.default_rng(51)
        actual = rng.normal(size=(80, 3))
        prediction = actual + rng.normal(size=actual.shape) * [0.2, 1, 3]
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        original, _ = score_predictions(actual, prediction)
        rotated, _ = score_predictions(4 * actual @ rotation + 3, 4 * prediction @ rotation + 3)
        self.assertAlmostEqual(original['native_r2'], rotated['native_r2'], places=12)
        rescaled, _ = score_predictions(actual * [1, 2, 10], prediction * [1, 2, 10])
        self.assertNotAlmostEqual(original['native_r2'], rescaled['native_r2'])
        self.assertAlmostEqual(original['macro_r2'], rescaled['macro_r2'], places=12)

    def test_shape_and_nonfinite_rejection(self):
        for actual, prediction in ((np.ones((2, 1)), np.ones((2, 2))),
                                   (np.ones(2), np.ones(2)), (np.ones((0, 2)), np.ones((0, 2))),
                                   (np.array([[np.inf]]), np.ones((1, 1))),
                                   (np.ones((2, 1)), np.full((2, 1), np.nan)),
                                   (np.ones((2, 1), dtype=complex), np.ones((2, 1)))):
            with self.subTest(shape=actual.shape), self.assertRaises(ValueError):
                score_predictions(actual, prediction)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        self.actual = rng.normal(size=(35, 4)) + [100, -300, 40, 1]
        self.baseline = self.actual + rng.normal(size=self.actual.shape)
        self.candidate = self.actual + rng.normal(size=self.actual.shape) * 0.4

    def test_bootstrap_matches_direct_row_resampling(self):
        n_boot, seed = 75, 19
        expected = []
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            indices = rng.integers(0, len(self.actual), size=len(self.actual))
            summary, _ = score_predictions(self.actual[indices], self.baseline[indices])
            expected.append(summary['native_r2'])
        result = bootstrap_r2(self.actual, self.baseline, n_boot=n_boot, seed=seed)
        np.testing.assert_allclose([result['native_ci_low'], result['native_ci_high']],
                                   np.quantile(expected, [0.025, 0.975]), atol=1e-12)
        self.assertEqual(result, bootstrap_r2(self.actual, self.baseline, n_boot=n_boot, seed=seed))
        self.assertEqual(result['bootstrap_defined'], n_boot)
        self.assertEqual(result['bootstrap_undefined'], 0)

    def test_paired_bootstrap_sign_pairing_and_identical_predictions(self):
        n_boot, seed = 73, 22
        expected = []
        rng = np.random.default_rng(seed)
        for _ in range(n_boot):
            indices = rng.integers(0, len(self.actual), size=len(self.actual))
            baseline, _ = score_predictions(self.actual[indices], self.baseline[indices])
            candidate, _ = score_predictions(self.actual[indices], self.candidate[indices])
            expected.append(candidate['native_r2'] - baseline['native_r2'])
        result = paired_r2(self.actual, self.baseline, self.candidate, n_boot=n_boot, seed=seed)
        reverse = paired_r2(self.actual, self.candidate, self.baseline, n_boot=n_boot, seed=seed)
        np.testing.assert_allclose([result['delta_ci_low'], result['delta_ci_high']],
                                   np.quantile(expected, [0.025, 0.975]), atol=1e-12)
        baseline, _ = score_predictions(self.actual, self.baseline)
        candidate, _ = score_predictions(self.actual, self.candidate)
        self.assertAlmostEqual(result['delta_r2'], candidate['native_r2'] - baseline['native_r2'])
        self.assertAlmostEqual(reverse['delta_r2'], -result['delta_r2'])
        self.assertAlmostEqual(reverse['delta_ci_low'], -result['delta_ci_high'])
        self.assertAlmostEqual(reverse['delta_ci_high'], -result['delta_ci_low'])
        identical = paired_r2(self.actual, self.baseline, self.baseline, n_boot=n_boot, seed=seed)
        self.assertEqual([identical[key] for key in ('delta_r2', 'delta_ci_low', 'delta_ci_high')], [0, 0, 0])

    def test_degenerate_resamples_are_counted_not_forced_finite(self):
        actual = np.array([[0.0], [1.0]])
        result = bootstrap_r2(actual, actual, n_boot=200, seed=4)
        rng = np.random.default_rng(4)
        rows = rng.integers(0, 2, size=(200, 2))
        self.assertEqual(result['bootstrap_undefined'], int(np.sum(rows[:, 0] == rows[:, 1])))
        self.assertEqual(result['native_ci_low'], 1)
        self.assertEqual(result['native_ci_high'], 1)
        constants = np.full((10, 3), 0.1)
        result = bootstrap_r2(constants, constants, n_boot=10)
        self.assertTrue(np.isnan(result['native_ci_low']))
        self.assertTrue(np.isnan(result['native_ci_high']))
        self.assertEqual(result['bootstrap_defined'], 0)
        self.assertEqual(result['bootstrap_undefined'], 10)
        difference = paired_r2(constants, constants, constants, n_boot=10)
        self.assertTrue(np.isnan(difference['delta_r2']))
        self.assertTrue(np.isnan(difference['delta_ci_low']))

    def test_bootstrap_validation(self):
        for count in (0, -1, 1.5, True):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, 'n_boot'):
                bootstrap_r2(self.actual, self.baseline, n_boot=count)
        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            paired_r2(self.actual, self.baseline, self.candidate[:, :1])


class GeometryTests(unittest.TestCase):
    def test_cka_orthogonal_affine_and_paired_permutation_invariance(self):
        rng = np.random.default_rng(20)
        first, second = rng.normal(size=(240, 4)), rng.normal(size=(240, 7))
        qfirst, _ = np.linalg.qr(rng.normal(size=(4, 4)))
        qsecond, _ = np.linalg.qr(rng.normal(size=(7, 7)))
        permutation = rng.permutation(len(first))
        expected = linear_cka(first, second)
        self.assertAlmostEqual(linear_cka(first @ qfirst * 3 + 2, second @ qsecond * 0.1 - 20), expected, places=12)
        self.assertAlmostEqual(linear_cka(first[permutation], second[permutation]), expected, places=12)
        self.assertAlmostEqual(linear_cka(first, first @ qfirst), 1, places=12)
        self.assertAlmostEqual(linear_cka(first, second), linear_cka(second, first), places=12)

    def test_cka_shuffle_null_and_constant_inputs(self):
        rng = np.random.default_rng(25)
        value = rng.normal(size=(500, 4))
        shuffled = linear_cka(value, value[rng.permutation(len(value))])
        self.assertGreater(shuffled, 0)
        self.assertLess(shuffled, 0.1)
        self.assertTrue(np.isnan(linear_cka(value, np.full((len(value), 3), 0.1))))
        with self.assertRaisesRegex(ValueError, 'same number'):
            linear_cka(value, value[:-1])

    def test_cka_matches_small_sample_gram_reference(self):
        rng = np.random.default_rng(62)
        first, second = rng.normal(size=(8, 4)), rng.normal(size=(8, 3))
        first -= first.mean(0)
        second -= second.mean(0)
        gram_first, gram_second = first @ first.T, second @ second.T
        expected = np.sum(gram_first * gram_second) / np.linalg.norm(gram_first) / np.linalg.norm(gram_second)
        self.assertAlmostEqual(linear_cka(first, second), expected, places=12)


class PCATests(unittest.TestCase):
    def setUp(self):
        self.mean = np.array([10.0, -20.0, 3.0])
        self.train = np.concatenate([np.diag([9.0, 3.0, 1.0]), -np.diag([9.0, 3.0, 1.0])]) + self.mean
        rng = np.random.default_rng(14)
        self.actual = rng.normal(size=(25, 3)) * [1, 4, 8] + [20, -30, -3]
        self.prediction = self.actual + rng.normal(size=(25, 3)) * 2

    def test_pca_matches_explicit_train_basis_reconstructions(self):
        rows = pca_diagnostics(self.train, self.actual, self.prediction, [0, 1, 2, 3])
        variance = np.r_[0.0, np.cumsum([81.0, 9.0, 1.0])] / 91
        for row in rows:
            rank = row['rank']
            oracle = np.broadcast_to(self.mean, self.actual.shape).copy()
            predicted = oracle.copy()
            oracle[:, :rank] = self.actual[:, :rank]
            predicted[:, :rank] = self.prediction[:, :rank]
            oracle_score, _ = score_predictions(self.actual, oracle)
            prediction_score, _ = score_predictions(self.actual, predicted)
            self.assertAlmostEqual(row['training_variance_fraction'], variance[rank], places=12)
            self.assertAlmostEqual(row['oracle_reconstruction_r2'], oracle_score['native_r2'], places=12)
            self.assertAlmostEqual(row['prediction_reconstruction_r2'], prediction_score['native_r2'], places=12)
            self.assertLessEqual(row['prediction_reconstruction_r2'], row['oracle_reconstruction_r2'] + 1e-12)
        self.assertAlmostEqual(rows[-1]['oracle_reconstruction_r2'], 1)
        oracle_scores = [row['oracle_reconstruction_r2'] for row in rows]
        self.assertTrue(np.all(np.diff(oracle_scores) >= -1e-12))

    def test_prediction_rank_curve_can_worsen(self):
        actual = np.array([[-3.0, -1, 0], [3.0, 1, 0], [-3.0, 1, 0], [3.0, -1, 0]]) + self.mean
        predicted = actual.copy()
        predicted[:, 1] = self.mean[1] + 20 * (actual[:, 1] - self.mean[1])
        rows = pca_diagnostics(self.train, actual, predicted, [1, 2])
        self.assertGreater(rows[1]['oracle_reconstruction_r2'], rows[0]['oracle_reconstruction_r2'])
        self.assertLess(rows[1]['prediction_reconstruction_r2'], rows[0]['prediction_reconstruction_r2'])
        self.assertLess(rows[1]['prediction_reconstruction_r2'], 0)

    def test_train_pca_is_not_a_heldout_optimal_subspace_or_decoder_ceiling(self):
        actual = np.array([[0, -10, 0], [0, 10, 0], [0, -20, 0], [0, 20, 0]]) + self.mean
        row = pca_diagnostics(self.train, actual, actual, [1])[0]
        self.assertAlmostEqual(row['oracle_reconstruction_r2'], 0)
        unrestricted, _ = score_predictions(actual, actual)
        self.assertEqual(unrestricted['native_r2'], 1)
        self.assertGreater(unrestricted['native_r2'], row['oracle_reconstruction_r2'])

    def test_reusable_pca_does_not_refit_on_evaluation(self):
        original = np.linalg.eigh
        with mock.patch('numpy.linalg.eigh', wraps=original) as decomposition:
            fitted = fit_target_pca(self.train)
            first = fitted.diagnostics(self.actual, self.prediction, [1, 2])
            second = fitted.diagnostics(self.actual * 100, self.prediction * 100, [1, 2])
        self.assertEqual(decomposition.call_count, 1)
        self.assertEqual([row['training_variance_fraction'] for row in first],
                         [row['training_variance_fraction'] for row in second])
        np.testing.assert_array_equal(fitted.mean, self.mean)

    def test_pca_wide_constant_and_invalid_cases(self):
        rng = np.random.default_rng(5)
        train, actual = rng.normal(size=(3, 8)), rng.normal(size=(4, 8))
        rows = pca_diagnostics(train, actual, actual, [0, 1, 3])
        self.assertAlmostEqual(rows[-1]['training_variance_fraction'], 1)
        constant = np.full((5, 3), 0.1)
        rows = pca_diagnostics(constant, constant, constant, [0, 3])
        for row in rows:
            self.assertTrue(np.isnan(row['training_variance_fraction']))
            self.assertTrue(np.isnan(row['oracle_reconstruction_r2']))
            self.assertTrue(np.isnan(row['prediction_reconstruction_r2']))
        self.assertEqual(pca_diagnostics(self.train, self.actual, self.prediction, []), [])
        for ranks in ([-1], [4], [1.5], [True]):
            with self.subTest(ranks=ranks), self.assertRaisesRegex(ValueError, 'ranks'):
                pca_diagnostics(self.train, self.actual, self.prediction, ranks)
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            pca_diagnostics(train, self.actual, self.prediction, [1])
        with self.assertRaisesRegex(ValueError, 'at least two'):
            fit_target_pca(self.train[:1])


if __name__ == '__main__':
    unittest.main()
