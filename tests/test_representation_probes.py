"""Small CPU-only checks for controlled probes; no EuroSAT data are accessed."""
from dataclasses import asdict
from io import BytesIO
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from torch.optim.optimizer import (
    _global_optimizer_post_hooks, register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torchgeo_bench.linear import LogisticRegression

from experiments.representation_overlap.probes import (
    Probe, classification_metrics, holm_adjust, paired_metrics, select_probe,
)


class ProbeSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        rng = np.random.default_rng(12)
        self.x = rng.normal(size=(160, 3)) * [1, 4, 9] + [7, -3, 11]
        self.h = rng.normal(size=(160, 2)) * [12, 0.3] + [-4, 8]
        self.y = (self.x[:, 0] - 7 + self.h[:, 1] - 8 > 0).astype(np.int64)
        self.options = {'device': 'cpu', 'max_iter': 120, 'tol': 1e-5, 'seed': 3}

    def select(self, x=None, h=None, y=None, cs=(0.3, 3), betas=(0, 1), **kwargs):
        x = self.x if x is None else x
        y = self.y if y is None else y
        extra = {} if h is None else {'extra_train': h[:100], 'extra_val': h[100:]}
        return select_probe(x[:100], x[100:], y[:100], y[100:], cs, betas,
                            **extra, **{**self.options, **kwargs})

    def test_normalization_train_only_and_folded_original_units(self):
        x = np.column_stack((self.x, np.full(len(self.x), 0.1)))
        h = np.column_stack((self.h, np.full(len(self.h), -13.0)))
        x[100:, 3], h[100:, 2] = 999, 32
        for standardize in (False, True):
            with self.subTest(standardize=standardize):
                fitted, matrices = [], []

                class CapturingLogistic(LogisticRegression):
                    def fit(self, features, labels):
                        result = super().fit(features, labels)
                        fitted.append(self)
                        matrices.append(features.cpu().numpy().copy())
                        return result

                with patch('torchgeo_bench.linear.LogisticRegression', CapturingLogistic):
                    probe, sweep = self.select(x, h, cs=(1,), betas=(0.25,),
                                               primary_standardize=standardize)
                mean, std = x[:100, :3].mean(0), x[:100, :3].std(0)
                p_scale = std * np.sqrt(3) if standardize else np.sqrt(np.square(std).sum())
                e_mean, e_scale = h[:100, :2].mean(0), h[:100, :2].std(0) * np.sqrt(2)
                expected_train = np.column_stack(((x[:100, :3] - mean) / p_scale,
                                                  0.25 * (h[:100, :2] - e_mean) / e_scale))
                expected_val = np.column_stack(((x[100:, :3] - mean) / p_scale,
                                                0.25 * (h[100:, :2] - e_mean) / e_scale))
                np.testing.assert_allclose(matrices[-1], expected_train, rtol=1e-6, atol=1e-7)
                expected_probs = fitted[-1].predict_proba(torch.tensor(expected_val, dtype=torch.float32))
                np.testing.assert_allclose(probe.predict_proba(x[100:], h[100:]),
                                           expected_probs, rtol=2e-5, atol=1e-7)
                np.testing.assert_array_equal(probe.coef[:, [3, 6]], 0)
                self.assertEqual(probe.coef.shape, (2, 7))
                self.assertEqual(probe.primary_dim, 4)
                info = sweep[0]['preprocessing']
                self.assertEqual(info['primary']['constant_columns'], [3])
                self.assertEqual(info['primary']['constant_columns_vary_on_val'], [3])
                self.assertEqual(info['extra']['constant_columns'], [2])
                self.assertEqual(info['extra']['constant_columns_vary_on_val'], [2])

    def test_native_primary_uniform_scaling_rotation_and_translation(self):
        baseline, _ = self.select(cs=(1,), betas=(0,))
        rotation = np.linalg.qr(np.random.default_rng(2).normal(size=(3, 3)))[0]
        changed = 7 * self.x @ rotation + [10, -9, 4]
        transformed, _ = self.select(changed, cs=(1,), betas=(0,))
        np.testing.assert_allclose(
            baseline.predict_proba(self.x[100:]), transformed.predict_proba(changed[100:]),
            atol=3e-4, rtol=3e-4,
        )

    def test_standardized_blocks_coordinate_units_do_not_change_predictions(self):
        baseline, _ = self.select(h=self.h, cs=(1,), betas=(1,), primary_standardize=True)
        x, h = self.x * [-2, 0.3, 8] + [9, 4, -30], self.h * [0.01, -50] + [2, 30]
        transformed, _ = self.select(x, h, cs=(1,), betas=(1,), primary_standardize=True)
        np.testing.assert_allclose(
            baseline.predict_proba(self.x[100:], self.h[100:]),
            transformed.predict_proba(x[100:], h[100:]), atol=3e-5, rtol=3e-5,
        )

    def test_genuinely_added_label_signal(self):
        rng = np.random.default_rng(45)
        x = rng.normal(size=(500, 3))
        h = rng.normal(size=(500, 2))
        y = (h[:, 0] > 0).astype(np.int64)
        baseline, _ = self.select(x, y=y, cs=(0.1, 10), betas=(0,))
        combined, sweep = self.select(x, h, y, cs=(0.1, 10), betas=(0, 1))
        base_score = classification_metrics(y[100:], baseline.predict_proba(x[100:]))
        combined_score = classification_metrics(y[100:], combined.predict_proba(x[100:], h[100:]))
        self.assertGreater(combined_score['accuracy'], 0.94)
        self.assertGreater(combined_score['accuracy'] - base_score['accuracy'], 0.3)
        self.assertEqual(combined.beta, 1)
        self.assertEqual(len(sweep), 4)

    def test_redundant_feature_is_regularization_reparameterization(self):
        x = np.linspace(-3, 3, 160)[:, None]
        y = (x[:, 0] > 0).astype(np.int64)
        permutation = np.random.default_rng(5).permutation(len(y))
        x, y = x[permutation], y[permutation]
        # Two identical normalized coordinates halve the minimum-norm penalty.
        baseline, _ = self.select(x, y=y, cs=(2,), betas=(0,))
        duplicate, _ = self.select(x, 12 * x + 17, y, cs=(1,), betas=(1,))
        np.testing.assert_allclose(
            baseline.predict_proba(x[100:]), duplicate.predict_proba(x[100:], 12 * x[100:] + 17),
            rtol=1e-3, atol=1e-4,
        )

    def test_beta_zero_is_identical_baseline_and_empty_extra_supported(self):
        baseline, baseline_sweep = self.select(betas=(0,))
        nested, nested_sweep = self.select(h=self.h, betas=(0,))
        empty, _ = self.select(h=np.empty((len(self.x), 0)), betas=(0,))
        np.testing.assert_array_equal(baseline.coef, nested.coef[:, :3])
        np.testing.assert_array_equal(baseline.intercept, nested.intercept)
        np.testing.assert_array_equal(nested.coef[:, 3:], 0)
        np.testing.assert_array_equal(baseline.coef, empty.coef)
        np.testing.assert_array_equal(baseline.predict_proba(self.x[100:]),
                                      nested.predict_proba(self.x[100:]))
        np.testing.assert_array_equal(nested.predict_proba(self.x[100:], self.h[100:] * 999),
                                      baseline.predict_proba(self.x[100:]))
        self.assertEqual([r['n_iter'] for r in baseline_sweep], [r['n_iter'] for r in nested_sweep])

    def test_class_order_and_npz_round_trip(self):
        y = np.choose(self.y, [19, 4])
        probe, _ = self.select(h=self.h, y=y, cs=(1,), betas=(1,))
        np.testing.assert_array_equal(probe.classes, [4, 19])
        encoded, _ = self.select(h=self.h, y=(y == 19).astype(int), cs=(1,), betas=(1,))
        np.testing.assert_array_equal(probe.coef, encoded.coef)
        with BytesIO() as buffer:
            np.savez(buffer, **asdict(probe))
            buffer.seek(0)
            with np.load(buffer, allow_pickle=False) as archive:
                fields = {key: archive[key].item() if archive[key].ndim == 0 else archive[key]
                          for key in archive.files}
        restored = Probe(**fields)
        np.testing.assert_array_equal(restored.predict_proba(self.x[100:], self.h[100:]),
                                      probe.predict_proba(self.x[100:], self.h[100:]))

    def test_multiclass_probabilities_and_constant_only_intercept(self):
        y = np.digitize(self.x[:, 0], [6.5, 7.5]).astype(np.int64)
        probe, _ = self.select(self.x[:, :1], y=y, cs=(10,), betas=(0,))
        probabilities = probe.predict_proba(self.x[100:, :1])
        self.assertEqual(probabilities.shape, (60, 3))
        np.testing.assert_array_equal(probe.classes, [0, 1, 2])
        self.assertGreater(classification_metrics(y[100:], probabilities)['accuracy'], 0.8)
        x = np.full((len(y), 2), [0.1, 3.0])
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Initializing zero-element tensors is a no-op')
            constant, sweep = self.select(x, y=y, cs=(1,), betas=(0,), tol=1e-7)
        np.testing.assert_array_equal(constant.coef, 0)
        np.testing.assert_allclose(
            constant.predict_proba(x[100:]),
            np.broadcast_to(np.bincount(y[:100]) / 100, (60, 3)),
            atol=1e-4, rtol=1e-4,
        )
        self.assertEqual(sweep[0]['preprocessing']['primary']['active_dim'], 0)

    def test_gpu_request_never_silently_falls_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torchgeo_bench.linear.LogisticRegression') as estimator:
                with self.assertRaisesRegex(RuntimeError, 'refusing CPU fallback'):
                    self.select(device='cuda:0')
                estimator.assert_not_called()
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=1):
            with self.assertRaisesRegex(ValueError, 'index 1 is unavailable'):
                self.select(device='cuda:1')
        with self.assertRaisesRegex(ValueError, 'device must be cpu'):
            self.select(device='meta')

    def test_solver_precision_seed_and_iteration_budget_retry(self):
        calls = []
        tf32_during_fit = []
        old_tf32 = torch.backends.cuda.matmul.allow_tf32

        class RetryLogistic(LogisticRegression):
            def __init__(self, **kwargs):
                calls.append(kwargs)
                super().__init__(**kwargs)

            def fit(self, features, labels):
                tf32_during_fit.append(torch.backends.cuda.matmul.allow_tf32)
                return super().fit(features, labels)

        def report_boundary(optimizer, args, kwargs):
            group = optimizer.param_groups[0]
            state = optimizer.state[group['params'][0]]
            state['n_iter'] = group['max_iter'] if len(calls) == 1 else 7
            state['func_evals'] = 65 if len(calls) == 1 else 9

        with register_optimizer_step_post_hook(report_boundary):
            with patch('torchgeo_bench.linear.LogisticRegression', RetryLogistic):
                probe, sweep = self.select(cs=(1,), betas=(0,), max_iter=60)
        self.assertEqual([call['max_iter'] for call in calls], [60, 120])
        self.assertTrue(all(call['random_state'] == 3 and call['use_tf32'] is False for call in calls))
        self.assertEqual(tf32_during_fit, [False, False])
        self.assertEqual(torch.backends.cuda.matmul.allow_tf32, old_tf32)
        self.assertEqual(probe.n_iter, 7)
        self.assertTrue(sweep[0]['retried'])
        self.assertEqual(sweep[0]['attempts'], [
            {'max_iter': 60, 'max_eval': 75, 'n_iter': 60, 'func_evals': 65,
             'termination_status': 'budget_exhausted', 'iteration_budget_exhausted': True,
             'evaluation_budget_exhausted': False, 'tolerance_grad': 1e-7,
             'tolerance_change': self.options['tol'] * 0.1},
            {'max_iter': 120, 'max_eval': 150, 'n_iter': 7, 'func_evals': 9,
             'termination_status': 'stopped_before_budgets', 'iteration_budget_exhausted': False,
             'evaluation_budget_exhausted': False, 'tolerance_grad': 1e-7,
             'tolerance_change': self.options['tol'] * 0.1},
        ])

    def test_evaluation_budget_exhaustion_before_iteration_cap_retries(self):
        budgets = []

        def limit_first_evaluation_budget(optimizer, args, kwargs):
            group = optimizer.param_groups[0]
            budgets.append((group['max_iter'], group['max_eval']))
            if len(budgets) == 1:
                group['max_eval'] = 2

        with register_optimizer_step_pre_hook(limit_first_evaluation_budget):
            probe, sweep = self.select(cs=(1,), betas=(0,), max_iter=60)
        self.assertEqual(budgets, [(60, 75), (120, 150)])
        first, second = sweep[0]['attempts']
        self.assertEqual(first['max_eval'], 2)
        self.assertGreaterEqual(first['func_evals'], 2)
        self.assertLess(first['n_iter'], first['max_iter'])
        self.assertFalse(first['iteration_budget_exhausted'])
        self.assertTrue(first['evaluation_budget_exhausted'])
        self.assertEqual(first['termination_status'], 'budget_exhausted')
        self.assertEqual(second['termination_status'], 'stopped_before_budgets')
        self.assertEqual(sweep[0]['func_evals'], second['func_evals'])
        self.assertEqual(probe.n_iter, second['n_iter'])
        self.assertTrue(sweep[0]['retried'])
        self.assertNotIn('converged', sweep[0])

    def test_second_evaluation_budget_exhaustion_fails(self):
        budgets = []

        def limit_evaluations(optimizer, args, kwargs):
            group = optimizer.param_groups[0]
            budgets.append(group['max_iter'])
            group['max_eval'] = 2

        hooks_before = dict(_global_optimizer_post_hooks)
        with register_optimizer_step_pre_hook(limit_evaluations):
            with self.assertRaisesRegex(RuntimeError, 'optimizer budget exhausted after retry.*function evaluations'):
                self.select(cs=(1,), betas=(0,), max_iter=60)
        self.assertEqual(budgets, [60, 120])
        self.assertEqual(dict(_global_optimizer_post_hooks), hooks_before)

    def test_second_iteration_cap_and_nonfinite_coefficients_fail(self):
        def report_iteration_cap(optimizer, args, kwargs):
            group = optimizer.param_groups[0]
            optimizer.state[group['params'][0]]['n_iter'] = group['max_iter']

        with register_optimizer_step_post_hook(report_iteration_cap):
            with self.assertRaisesRegex(RuntimeError, 'optimizer budget exhausted after retry'):
                self.select(cs=(1,), betas=(0,))

        class NonfiniteLogistic(LogisticRegression):
            @property
            def coef_(self):
                return super().coef_ * np.nan

        with patch('torchgeo_bench.linear.LogisticRegression', NonfiniteLogistic):
            with self.assertRaisesRegex(ValueError, 'non-finite fitted coefficients'):
                self.select(cs=(1,), betas=(0,))

    def test_optimizer_hook_cleanup_on_success_and_solver_failure(self):
        hooks_before = dict(_global_optimizer_post_hooks)
        self.select(cs=(1,), betas=(0,))
        self.assertEqual(dict(_global_optimizer_post_hooks), hooks_before)
        with patch.object(LogisticRegression, 'fit', side_effect=RuntimeError('synthetic fit failure')):
            with self.assertRaisesRegex(RuntimeError, 'synthetic fit failure'):
                self.select(cs=(1,), betas=(0,))
        self.assertEqual(dict(_global_optimizer_post_hooks), hooks_before)

        def invalid_state(optimizer, args, kwargs):
            group = optimizer.param_groups[0]
            optimizer.state[group['params'][0]]['func_evals'] = -1

        with register_optimizer_step_post_hook(invalid_state):
            with self.assertRaisesRegex(ValueError, 'LBFGS func_evals'):
                self.select(cs=(1,), betas=(0,))
        self.assertEqual(dict(_global_optimizer_post_hooks), hooks_before)

    def test_default_iteration_and_evaluation_budgets(self):
        _, sweep = select_probe(self.x[:100], self.x[100:], self.y[:100], self.y[100:],
                                (1,), (0,), device='cpu')
        self.assertEqual(sweep[0]['max_iter'], 8000)
        self.assertEqual(sweep[0]['max_eval'], 10000)
        self.assertEqual(sweep[0]['termination_status'], 'stopped_before_budgets')

    def test_selection_ties_and_boundary_diagnostics(self):
        # No extra block: all betas are equivalent, including the baseline listed last.
        with patch('experiments.representation_overlap.probes.classification_metrics',
                   return_value={'accuracy': 0.5, 'log_loss': 1.0}):
            probe, sweep = self.select(cs=(3, 0.1), betas=(4, 1, 0))
        self.assertEqual((probe.C, probe.beta), (0.1, 0))
        self.assertEqual(sum(record['selected'] for record in sweep), 1)
        chosen = next(record for record in sweep if record['selected'])
        self.assertTrue(chosen['c_at_lower_boundary'])
        self.assertFalse(chosen['c_at_upper_boundary'])
        self.assertTrue(all(record['termination_status'] == 'stopped_before_budgets' for record in sweep))
        self.assertTrue(all('preprocessing' in record and 'attempts' in record for record in sweep))

    def test_accuracy_then_log_loss_precedes_regularization_ties(self):
        for scores, expected_c in [
            ([{'accuracy': 0.9, 'log_loss': 0.8}, {'accuracy': 0.8, 'log_loss': 0.1}], 0.1),
            ([{'accuracy': 0.9, 'log_loss': 0.8}, {'accuracy': 0.9, 'log_loss': 0.1}], 3),
        ]:
            with self.subTest(expected_c=expected_c):
                with patch('experiments.representation_overlap.probes.classification_metrics', side_effect=scores):
                    probe, _ = self.select(cs=(0.1, 3), betas=(0,))
                self.assertEqual(probe.C, expected_c)

    def test_selection_rejects_invalid_inputs(self):
        for options in ({'cs': ()}, {'cs': (0,)}, {'cs': (1, 1)}, {'betas': (-1,)},
                        {'betas': (np.nan,)}, {'tol': 0}, {'max_iter': 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.select(**options)
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            self.select(x=self.x * np.nan)
        with self.assertRaisesRegex(ValueError, 'both be supplied'):
            self.select(extra_train=self.h[:100])
        unseen = self.y.copy()
        unseen[-1] = 9
        with self.assertRaisesRegex(ValueError, 'every validation class'):
            self.select(y=unseen)


class ProbeMetricsTests(unittest.TestCase):
    def test_stable_softmax_and_input_validation(self):
        probe = Probe(np.array([[1000., 0], [-1000., 0]]), np.array([500., -500.]),
                      np.array([8, 3]), 1, 0, 5, 2)
        probabilities = probe.predict_proba(np.array([[1, 0], [-1, 0], [0, 0]]))
        self.assertEqual(probabilities.dtype, np.float64)
        np.testing.assert_array_equal(probabilities.argmax(1), [0, 1, 0])
        np.testing.assert_array_equal(probabilities.sum(1), 1)
        for invalid in (np.ones((3, 1)), np.full((3, 2), np.inf), np.ones(2)):
            with self.assertRaises(ValueError):
                probe.predict_proba(invalid)
        with self.assertRaisesRegex(ValueError, 'zero extra-feature weights'):
            Probe(np.ones((2, 2)), np.zeros(2), np.arange(2), 1, 0, 1, 1)
        combined = Probe(np.ones((2, 2)), np.zeros(2), np.arange(2), 1, 1, 1, 1)
        with self.assertRaisesRegex(ValueError, 'extra must be provided'):
            combined.predict_proba(np.ones((3, 1)))

    def test_classification_counts_and_log_loss(self):
        probabilities = np.array([[0.8, 0.2], [0.3, 0.7], [0.4, 0.6]])
        result = classification_metrics(np.array([0, 1, 0]), probabilities)
        self.assertEqual((result['accuracy'], result['correct'], result['errors']), (2 / 3, 2, 1))
        self.assertAlmostEqual(result['log_loss'], -np.log([0.8, 0.7, 0.4]).mean())
        zero_loss = classification_metrics(np.array([0]), np.array([[0., 1.]]))['log_loss']
        self.assertEqual(zero_loss, -np.log(np.finfo(np.float64).eps))

    def test_invalid_probability_matrices_are_not_repaired(self):
        invalid = (np.array([[0.4, 0.4]]), np.array([[-0.1, 1.1]]),
                   np.array([[np.nan, 0.5]]), np.array([[np.inf, 0.5]]),
                   np.ones((1, 1)), np.empty((0, 2)), np.array([0.4, 0.6]))
        for probabilities in invalid:
            with self.subTest(probabilities=probabilities):
                with self.assertRaises(ValueError):
                    classification_metrics(np.array([0]), probabilities)
                with self.assertRaises(ValueError):
                    paired_metrics(np.array([0]), np.array([[0.5, 0.5]]), probabilities)
        for labels in (np.array([-1]), np.array([2]), np.array([0.0]), np.array([[0]])):
            with self.assertRaises(ValueError):
                classification_metrics(labels, np.array([[0.5, 0.5]]))
        with self.assertRaisesRegex(ValueError, 'identical shapes'):
            paired_metrics(np.array([0]), np.array([[0.5, 0.5]]), np.array([[0.2, 0.3, 0.5]]))

    def test_paired_intervals_reuse_deterministic_sample_indices(self):
        y = np.zeros(20, dtype=np.int64)
        baseline_correct = np.array([True] * 8 + [False] * 12)
        combined_correct = np.array([True] * 6 + [False] * 2 + [True] * 10 + [False] * 2)
        baseline = np.column_stack((np.where(baseline_correct, 0.85, 0.2),
                                    np.where(baseline_correct, 0.15, 0.8)))
        combined = np.column_stack((np.where(combined_correct, 0.9, 0.3),
                                    np.where(combined_correct, 0.1, 0.7)))
        result = paired_metrics(y, baseline, combined, n_boot=99, seed=42)
        self.assertEqual(result, paired_metrics(y, baseline, combined, n_boot=99, seed=42))
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(y), size=(99, len(y)))
        delta = combined_correct.astype(float) - baseline_correct
        loss_delta = np.log(combined[:, 0]) - np.log(baseline[:, 0])
        expected_accuracy = np.percentile(delta[indices].mean(1), [2.5, 97.5])
        expected_loss = np.percentile(loss_delta[indices].mean(1), [2.5, 97.5])
        np.testing.assert_array_equal([result['accuracy_ci_low'], result['accuracy_ci_high']], expected_accuracy)
        np.testing.assert_allclose([result['log_loss_ci_low'], result['log_loss_ci_high']], expected_loss)
        self.assertEqual([result[key] for key in ('gained_correct', 'lost_correct', 'both_correct', 'both_wrong')],
                         [10, 2, 6, 2])
        self.assertAlmostEqual(result['accuracy_delta'], 0.4)
        self.assertAlmostEqual(result['relative_error_reduction'], 8 / 12)
        self.assertAlmostEqual(result['mcnemar_pvalue'], 2 * (1 + 12 + 66) / 2 ** 12)
        self.assertAlmostEqual(result['log_loss_improvement'], loss_delta.mean())
        swapped = paired_metrics(y, combined, baseline, n_boot=99, seed=42)
        self.assertEqual(swapped['mcnemar_pvalue'], result['mcnemar_pvalue'])
        self.assertAlmostEqual(swapped['accuracy_ci_low'], -result['accuracy_ci_high'])
        self.assertAlmostEqual(swapped['log_loss_ci_high'], -result['log_loss_ci_low'])

    def test_no_discordance_and_undefined_error_reduction(self):
        y = np.array([0, 1, 0, 1])
        probabilities = np.eye(2)[y]
        result = paired_metrics(y, probabilities, probabilities, n_boot=9)
        for name in ('accuracy_delta', 'accuracy_ci_low', 'accuracy_ci_high',
                     'log_loss_improvement', 'log_loss_ci_low', 'log_loss_ci_high'):
            self.assertEqual(result[name], 0)
        self.assertEqual(result['mcnemar_pvalue'], 1)
        self.assertTrue(np.isnan(result['relative_error_reduction']))
        with self.assertRaisesRegex(ValueError, 'n_boot'):
            paired_metrics(y, probabilities, probabilities, n_boot=0)

    def test_holm_adjustment_is_stable_and_in_original_order(self):
        values = np.array([0.03, 0.001, 0.02, 0.02, 0.7, 0.2, 0.5, 0.9, 1., 0.4])
        adjusted = holm_adjust(values)
        np.testing.assert_allclose(adjusted, [0.21, 0.01, 0.18, 0.18, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(holm_adjust(values), adjusted)
        np.testing.assert_array_equal(holm_adjust([]), [])
        np.testing.assert_array_equal(holm_adjust([0, 0, 1]), [0, 0, 1])
        for invalid in ([np.nan], [-0.1], [1.1], [[0.2]]):
            with self.assertRaises(ValueError):
                holm_adjust(invalid)


if __name__ == '__main__':
    unittest.main()
