"""Train-only linear probes and paired, held-out classification summaries.

The pinned TorchGeo-bench solver minimizes mean cross-entropy plus
``||W||**2 / (2 * C * n_train)`` in float32. Stored affine heads and their
predictions use float64 and the original, unnormalized feature coordinates.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest


def _matrix(value, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or not len(array) or array.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must be a nonempty, two-dimensional real numeric matrix')
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name} contains non-finite values')
    return array


def _labels(value, size: int, name: str = 'labels') -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or len(array) != size or array.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array with {size} entries')
    return array


def _integer(value, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _probabilities(value) -> np.ndarray:
    probabilities = _matrix(value, 'probabilities')
    if probabilities.shape[1] < 2:
        raise ValueError('probabilities must contain at least two class columns')
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError('probabilities must lie in [0, 1]')
    if not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-8):
        raise ValueError('probability rows must sum to one')
    return probabilities


def _outcomes(y, probabilities) -> tuple[np.ndarray, np.ndarray]:
    probabilities = _probabilities(probabilities)
    y = _labels(y, len(probabilities))
    if np.any((y < 0) | (y >= probabilities.shape[1])):
        raise ValueError('labels must index the probability columns')
    correct = probabilities.argmax(axis=1) == y
    # Legitimate zero probabilities have finite loss under float64 epsilon clipping.
    losses = -np.log(np.maximum(probabilities[np.arange(len(y)), y], np.finfo(np.float64).eps))
    return correct, losses


@dataclass
class Probe:
    coef: np.ndarray
    intercept: np.ndarray
    classes: np.ndarray
    C: float
    beta: float
    n_iter: int
    primary_dim: int

    def __post_init__(self) -> None:
        self.coef = _matrix(self.coef, 'coef')
        self.intercept = np.asarray(self.intercept, dtype=np.float64)
        self.classes = _labels(self.classes, len(self.coef), 'classes')
        if len(self.classes) < 2 or len(np.unique(self.classes)) != len(self.classes):
            raise ValueError('classes must contain at least two distinct integer labels')
        if self.intercept.shape != (len(self.classes),) or not np.isfinite(self.intercept).all():
            raise ValueError('intercept must contain one finite value per class')
        self.primary_dim = _integer(self.primary_dim, 'primary_dim', 0)
        self.n_iter = _integer(self.n_iter, 'n_iter', 0)
        if self.primary_dim > self.coef.shape[1]:
            raise ValueError('primary_dim exceeds the stored feature width')
        self.C, self.beta = float(self.C), float(self.beta)
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError('C must be finite and positive')
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError('beta must be finite and nonnegative')
        if self.beta == 0 and np.any(self.coef[:, self.primary_dim:] != 0):
            raise ValueError('a beta=0 baseline must have zero extra-feature weights')

    def predict_proba(self, primary, extra=None) -> np.ndarray:
        """Return probabilities in ``classes`` order, without refitting preprocessing.

        Extra inputs can be omitted for a beta-zero head, whose extra weights
        are exactly zero. Supplied inputs are always checked, even for baseline heads.
        """
        primary = _matrix(primary, 'primary')
        if primary.shape[1] != self.primary_dim:
            raise ValueError(f'primary must have {self.primary_dim} columns')
        extra_dim = self.coef.shape[1] - self.primary_dim
        if extra is None:
            if extra_dim and self.beta != 0:
                raise ValueError(f'extra must be provided with {extra_dim} columns')
        else:
            extra = _matrix(extra, 'extra')
            if extra.shape != (len(primary), extra_dim):
                raise ValueError(f'extra must have shape {(len(primary), extra_dim)}')
        logits = primary @ self.coef[:, :self.primary_dim].T + self.intercept
        if extra is not None and self.beta != 0:
            logits += extra @ self.coef[:, self.primary_dim:].T
        if not np.isfinite(logits).all():
            raise ValueError('probe logits contain non-finite values')
        with np.errstate(over='ignore', under='ignore'):
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return _probabilities(probabilities)


def _prepare_block(train: np.ndarray, val: np.ndarray, standardize: bool) -> tuple:
    active = np.any(train != train[0], axis=0)
    mean = train.mean(axis=0)
    mean[~active] = train[0, ~active]
    centered = train[:, active] - mean[active]
    std = np.std(centered, axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('feature centering/scaling is not finite or nonconstant variance underflowed')
    scale = float(np.sqrt(active.sum()) if standardize else np.linalg.norm(std))
    if not np.isfinite(scale) or (active.any() and scale <= 0):
        raise ValueError('training block variance is not finite and positive')
    multiplier = np.zeros(train.shape[1], dtype=np.float64)
    if active.any():
        multiplier[active] = (1 / std if standardize else 1) / scale
    transformed = centered * multiplier[active]
    if not np.isfinite(multiplier).all() or not np.isfinite(transformed).all():
        raise ValueError('normalized features contain non-finite values')
    diagnostics = {
        'standardize': bool(standardize),
        'active_dim': int(active.sum()),
        'constant_columns': np.flatnonzero(~active).tolist(),
        'constant_columns_vary_on_val': np.flatnonzero(
            ~active & np.any(val != train[0], axis=0)
        ).tolist(),
        'block_scale': scale,
    }
    return transformed, mean, multiplier, active, diagnostics


def _device(device):
    import torch

    requested = torch.device(device)
    if requested.type not in ('cpu', 'cuda'):
        raise ValueError('probe device must be cpu or an explicit CUDA device')
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(f'CUDA device {requested} requested but CUDA is unavailable; refusing CPU fallback')
        index = torch.cuda.current_device() if requested.index is None else requested.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index {index} is unavailable ({torch.cuda.device_count()} visible devices)')
        requested = torch.device('cuda', index)
    else:
        if requested.index not in (None, 0):
            raise ValueError('CPU device index must be absent or zero')
        requested = torch.device('cpu')
    return requested


@contextmanager
def _without_tf32():
    import torch

    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _fit_with_budget_diagnostics(estimator, features, labels) -> dict:
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook

    observed = []

    def capture(optimizer, args, kwargs):
        if not isinstance(optimizer, torch.optim.LBFGS) or estimator._model is None:
            return
        parameters = [parameter for group in optimizer.param_groups for parameter in group['params']]
        if {id(parameter) for parameter in parameters} != {
            id(parameter) for parameter in estimator._model.parameters()
        }:
            return
        if len(optimizer.param_groups) != 1:
            raise RuntimeError('expected one pinned LBFGS parameter group')
        group = optimizer.param_groups[0]
        state = optimizer.state[parameters[0]]
        n_iter = _integer(state['n_iter'], 'LBFGS n_iter', 0)
        func_evals = _integer(state['func_evals'], 'LBFGS func_evals', 1)
        max_iter = _integer(group['max_iter'], 'LBFGS max_iter', 1)
        max_eval = _integer(group['max_eval'], 'LBFGS max_eval', 1)
        iteration_exhausted, evaluation_exhausted = n_iter >= max_iter, func_evals >= max_eval
        observed.append({
            'n_iter': n_iter, 'func_evals': func_evals, 'max_iter': max_iter, 'max_eval': max_eval,
            'iteration_budget_exhausted': iteration_exhausted,
            'evaluation_budget_exhausted': evaluation_exhausted,
            'termination_status': ('budget_exhausted' if iteration_exhausted or evaluation_exhausted
                                   else 'stopped_before_budgets'),
            'tolerance_grad': float(group['tolerance_grad']),
            'tolerance_change': float(group['tolerance_change']),
        })

    # The pinned estimator does not expose its optimizer. Observe only its own
    # parameters through a temporary global hook, without altering the solver.
    hook = register_optimizer_step_post_hook(capture)
    try:
        with _without_tf32():
            estimator.fit(features, labels)
    finally:
        hook.remove()
    if len(observed) != 1:
        raise RuntimeError('expected exactly one observed pinned LBFGS step')
    if _integer(estimator.n_iter_, 'estimator.n_iter_', 0) != observed[0]['n_iter']:
        raise RuntimeError('estimator iteration count differs from the observed LBFGS state')
    return observed[0]


def _grid(values, name: str, positive: bool) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a nonempty, finite one-dimensional grid')
    if np.any(array <= 0 if positive else array < 0) or len(np.unique(array)) != len(array):
        raise ValueError(f'{name} must contain unique {"positive" if positive else "nonnegative"} values')
    return array.tolist()


def select_probe(
    primary_train, primary_val, y_train, y_val, c_values, betas, *,
    extra_train=None, extra_val=None, primary_standardize=False,
    device='cuda:0', max_iter=8000, tol=1e-6, seed=0,
) -> tuple[Probe, list[dict]]:
    """Select on validation accuracy, loss, smaller C, then baseline-first beta order.

    Constant train columns are removed from optimization and folded back as zeros.
    Either exhausted iteration/evaluation budget triggers one restart with twice
    the requested iterations; a second exhausted budget raises. Stopping before
    both budgets does not identify which numerical stopping criterion was met.
    Grid boundaries are recorded, never extended. No test inputs are accepted.
    """
    import torch
    from torchgeo_bench.linear import LogisticRegression

    train, val = _matrix(primary_train, 'primary_train'), _matrix(primary_val, 'primary_val')
    if train.shape[1] != val.shape[1]:
        raise ValueError('primary train/validation feature widths differ')
    y_train, y_val = _labels(y_train, len(train), 'y_train'), _labels(y_val, len(val), 'y_val')
    classes, encoded_train = np.unique(y_train, return_inverse=True)
    if len(classes) < 2 or not np.isin(y_val, classes).all():
        raise ValueError('train must contain at least two classes and every validation class')
    encoded_val = np.searchsorted(classes, y_val)
    if (extra_train is None) != (extra_val is None):
        raise ValueError('extra_train and extra_val must either both be supplied or both be absent')
    if extra_train is None:
        extra_train, extra_val = np.empty((len(train), 0)), np.empty((len(val), 0))
    extra_train, extra_val = _matrix(extra_train, 'extra_train'), _matrix(extra_val, 'extra_val')
    if (len(extra_train) != len(train) or len(extra_val) != len(val)
            or extra_train.shape[1] != extra_val.shape[1]):
        raise ValueError('extra features must match sample counts and train/validation widths')
    cs, betas = _grid(c_values, 'c_values', True), _grid(betas, 'betas', False)
    betas = ([0.0] if 0.0 in betas else []) + [beta for beta in betas if beta != 0]
    max_iter, seed = _integer(max_iter, 'max_iter', 1), _integer(seed, 'seed', 0)
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError('tol must be finite and positive')
    if not isinstance(primary_standardize, (bool, np.bool_)):
        raise ValueError('primary_standardize must be a boolean')
    requested = _device(device)
    primary, p_mean, p_multiplier, p_active, p_info = _prepare_block(train, val, primary_standardize)
    extra, e_mean, e_multiplier, e_active, e_info = _prepare_block(extra_train, extra_val, True)
    primary_tensor = torch.as_tensor(primary, dtype=torch.float32, device=requested)
    extra_tensor = torch.as_tensor(extra, dtype=torch.float32, device=requested)
    labels_tensor = torch.as_tensor(encoded_train, dtype=torch.long, device=requested)
    if not torch.isfinite(primary_tensor).all().item() or not torch.isfinite(extra_tensor).all().item():
        raise ValueError('normalized features cannot be represented in solver float32 precision')
    records, best, best_key = [], None, None
    for beta_index, beta in enumerate(betas):
        # Do not append even zero columns at beta=0: this is the identical baseline fit.
        features = (torch.cat((primary_tensor, beta * extra_tensor), dim=1)
                    if beta != 0 and e_active.any() else primary_tensor)
        if not torch.isfinite(features).all().item():
            raise ValueError(f'beta={beta} creates non-finite solver features')
        for c in cs:
            attempts = []
            for limit in (max_iter, 2 * max_iter):
                estimator = LogisticRegression(
                    C=c, max_iter=limit, tol=tol, solver='lbfgs',
                    random_state=seed, device=requested, use_tf32=False,
                )
                if estimator.device != requested:
                    raise RuntimeError('estimator changed the requested device; refusing silent fallback')
                attempt = _fit_with_budget_diagnostics(estimator, features, labels_tensor)
                coefficients = np.asarray(estimator.coef_, dtype=np.float64)
                intercept = np.asarray(estimator.intercept_, dtype=np.float64)
                if (coefficients.shape != (len(classes), features.shape[1])
                        or intercept.shape != (len(classes),)
                        or not np.isfinite(coefficients).all() or not np.isfinite(intercept).all()):
                    raise ValueError(f'C={c}, beta={beta}: invalid or non-finite fitted coefficients')
                if not np.array_equal(estimator.classes_, np.arange(len(classes))):
                    raise ValueError('estimator returned an unexpected class order')
                n_iter = attempt['n_iter']
                attempts.append(attempt)
                if attempt['termination_status'] == 'stopped_before_budgets':
                    break
            else:
                raise RuntimeError(
                    f'C={c}, beta={beta}: optimizer budget exhausted after retry '
                    f'(iterations {n_iter}/{attempt["max_iter"]}; function evaluations '
                    f'{attempt["func_evals"]}/{attempt["max_eval"]}); refusing a budget-exhausted result'
                )
            coef = np.zeros((len(classes), train.shape[1] + extra_train.shape[1]), dtype=np.float64)
            coef[:, np.flatnonzero(p_active)] = coefficients[:, :p_active.sum()] * p_multiplier[p_active]
            if beta != 0 and e_active.any():
                coef[:, train.shape[1] + np.flatnonzero(e_active)] = (
                    coefficients[:, p_active.sum():] * e_multiplier[e_active] * beta
                )
            bias = intercept - coef[:, :train.shape[1]] @ p_mean - coef[:, train.shape[1]:] @ e_mean
            probe = Probe(coef, bias, classes.copy(), c, beta, n_iter, train.shape[1])
            metrics = classification_metrics(encoded_val, probe.predict_proba(val, extra_val))
            key = (-metrics['accuracy'], metrics['log_loss'], c, beta_index)
            record = {
                'C': c, 'beta': beta, 'val_accuracy': metrics['accuracy'],
                'val_log_loss': metrics['log_loss'], **attempt,
                'retried': len(attempts) > 1, 'attempts': attempts,
                'c_at_lower_boundary': c == min(cs), 'c_at_upper_boundary': c == max(cs),
                'beta_at_upper_boundary': beta != 0 and beta == max(betas),
                'baseline': beta == 0, 'beta_order': beta_index, 'selected': False,
                'device': str(requested), 'solver': 'lbfgs', 'tol': float(tol), 'seed': seed,
                'training_dtype': 'float32', 'prediction_dtype': 'float64', 'use_tf32': False,
                'preprocessing': {'primary': p_info, 'extra': e_info},
            }
            records.append(record)
            if best_key is None or key < best_key:
                best, best_key = probe, key
                best_index = len(records) - 1
    records[best_index]['selected'] = True
    return best, records


def classification_metrics(y, probabilities) -> dict:
    """Score integer class-index labels; zero probabilities are epsilon-clipped."""
    correct, losses = _outcomes(y, probabilities)
    count = int(correct.sum())
    return {'accuracy': float(correct.mean()), 'correct': count,
            'errors': len(correct) - count, 'log_loss': float(losses.mean())}


def paired_metrics(y, baseline_probs, combined_probs, *, n_boot=200, seed=0) -> dict:
    """Paired 95% percentile intervals conditional on the two fixed fitted heads."""
    baseline, baseline_loss = _outcomes(y, baseline_probs)
    combined, combined_loss = _outcomes(y, combined_probs)
    if np.shape(baseline_probs) != np.shape(combined_probs):
        raise ValueError('paired probability matrices must have identical shapes and class order')
    n_boot, seed = _integer(n_boot, 'n_boot', 1), _integer(seed, 'seed', 0)
    accuracy_delta = combined.astype(np.float64) - baseline
    loss_delta = baseline_loss - combined_loss
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, 2), dtype=np.float64)
    for i in range(n_boot):
        indices = rng.integers(0, len(baseline), size=len(baseline))
        draws[i] = accuracy_delta[indices].mean(), loss_delta[indices].mean()
    lower, upper = np.percentile(draws, [2.5, 97.5], axis=0)
    gained, lost = int((~baseline & combined).sum()), int((baseline & ~combined).sum())
    discordant, baseline_errors = gained + lost, int((~baseline).sum())
    return {
        'accuracy_delta': float(combined.mean() - baseline.mean()),
        'accuracy_ci_low': float(lower[0]), 'accuracy_ci_high': float(upper[0]),
        'log_loss_improvement': float(baseline_loss.mean() - combined_loss.mean()),
        'log_loss_ci_low': float(lower[1]), 'log_loss_ci_high': float(upper[1]),
        'gained_correct': gained, 'lost_correct': lost,
        'both_correct': int((baseline & combined).sum()), 'both_wrong': int((~baseline & ~combined).sum()),
        'mcnemar_pvalue': float(binomtest(gained, discordant, 0.5).pvalue) if discordant else 1.0,
        'relative_error_reduction': (gained - lost) / baseline_errors if baseline_errors else np.nan,
    }


def holm_adjust(pvalues) -> np.ndarray:
    """Holm family-wise adjustment in original order, preserving stable ties."""
    values = np.asarray(pvalues, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError('pvalues must be a finite one-dimensional array in [0, 1]')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted
