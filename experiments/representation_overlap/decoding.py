"""Train-fitted affine decoding and held-out representation diagnostics.

R-squared uses the measured split's mean only in its scoring denominator.
Bootstrap intervals condition on the fitted predictions; they do not include
training variation, prior feature discovery, or geographic dependence.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
from numbers import Integral

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
        raise ValueError(f'{name} must be a real numeric matrix')
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 2 or not all(value.shape):
        raise ValueError(f'{name} must be a nonempty two-dimensional matrix')
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains non-finite values')
    return value


def _prediction_pair(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = _matrix(y_true, 'y_true')
    predicted = _matrix(y_pred, 'y_pred')
    if actual.shape != predicted.shape:
        raise ValueError('y_true and y_pred must have identical shapes')
    return actual, predicted


def _center(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Subtracting an observed origin first makes exact constants exactly zero.
    shifted = value - value[0]
    offset = shifted.mean(axis=0)
    return shifted - offset, value[0] + offset


def _squared_norm(value: np.ndarray, axis: int | None = None) -> np.ndarray:
    return np.sum(np.square(value), axis=axis)


def _target_sst(value: np.ndarray) -> np.ndarray:
    centered, _ = _center(value)
    return _squared_norm(centered, axis=0)


def _fingerprint(value: np.ndarray) -> str:
    digest = sha256(str(value.shape).encode('ascii'))
    digest.update(memoryview(np.ascontiguousarray(value)).cast('B'))
    return digest.hexdigest()


@dataclass
class RidgeSource:
    """Reusable train-only source factorization returned by prepare_ridge_source.

    Reuse across targets, target permutations, and regularization grids, but
    not across changed source rows. The most recent validation-source products
    are cached independently of all targets. This mutable in-memory cache is
    not a fitted decoder or a checkpoint.
    """

    mean: np.ndarray
    scale: np.ndarray
    source_constant: np.ndarray
    eigenvalues: np.ndarray
    basis: np.ndarray
    _training: np.ndarray = field(repr=False)
    _fingerprint: str = field(repr=False)
    _svd: bool = field(default=False, repr=False)
    _validation: tuple[str, np.ndarray, np.ndarray] | None = field(default=None, repr=False)

    def _use_svd(self) -> None:
        if self._svd:
            return
        left, singular, right = np.linalg.svd(self._training, full_matrices=False)
        keep = singular > np.finfo(np.float64).eps * max(self._training.shape) * singular[0]
        self.eigenvalues = np.square(singular[keep])
        self.basis = right[keep].T
        self._training = left[:, keep]
        self._svd = True
        self._validation = None

    def _target_cross(self, target: np.ndarray) -> np.ndarray:
        projected = self._training.T @ target
        if self._svd:
            return np.sqrt(self.eigenvalues)[:, None] * projected
        return self.basis.T @ projected

    def _validation_statistics(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        key = _fingerprint(value)
        if self._validation is None or self._validation[0] != key:
            standardized = (value[:, ~self.source_constant] - self.mean[~self.source_constant]) / self.scale
            projected = standardized @ self.basis
            self._validation = (key, projected, projected.T @ projected)
        return self._validation[1], self._validation[2]


def _prepare_ridge_source(train: np.ndarray, *, use_svd: bool) -> RidgeSource:
    constant = np.all(train == train[0], axis=0)
    centered, mean = _center(train)
    scale = np.sqrt(_squared_norm(centered[:, ~constant], axis=0) / len(train))
    if np.any(scale == 0) or not np.isfinite(scale).all():
        raise ValueError('nonconstant source scaling is outside float64 numerical range')
    design = centered[:, ~constant] / scale
    source = RidgeSource(mean, scale, constant, np.empty(0), np.empty((0, 0)),
                         design, _fingerprint(train))
    if not design.shape[1]:
        source._svd = True
    elif use_svd or design.shape[1] > design.shape[0]:
        source._use_svd()
    else:
        eigenvalues, source.basis = np.linalg.eigh(design.T @ design)
        source.eigenvalues = np.maximum(eigenvalues, 0)
    return source


def prepare_ridge_source(x_train: np.ndarray) -> RidgeSource:
    """Prepare a source once for repeated ``select_ridge(..., source=source)``.

    This source-only operation fits means/scales and a shared spectrum without
    seeing validation or target data. Exact float64-content fingerprints reject
    reuse with different training values or row order. Tall designs use a
    covariance eigensolve for positive-alpha grids; the first zero-alpha request
    replaces it with one cached SVD for stable pseudoinverse handling. Wide
    designs use an SVD from the start.
    """
    return _prepare_ridge_source(_matrix(x_train, 'x_train'), use_svd=False)


@dataclass
class RidgeMap:
    """An affine map in ORIGINAL source and target units, with no saved scaler."""

    coef: np.ndarray
    intercept: np.ndarray
    alpha: float
    source_constant: np.ndarray

    def __post_init__(self) -> None:
        self.coef = _matrix(self.coef, 'coef')
        self.intercept = np.asarray(self.intercept, dtype=np.float64)
        self.source_constant = np.asarray(self.source_constant)
        if self.intercept.shape != (self.coef.shape[1],):
            raise ValueError('intercept must have one entry per output')
        if not np.isfinite(self.intercept).all():
            raise ValueError('intercept contains non-finite values')
        if (self.source_constant.shape != (self.coef.shape[0],)
                or self.source_constant.dtype != np.dtype(bool)):
            raise ValueError('source_constant must be a boolean mask of the input columns')
        if np.any(self.coef[self.source_constant] != 0):
            raise ValueError('training-constant source columns must have zero coefficients')
        self.alpha = float(self.alpha)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError('alpha must be finite and nonnegative')

    def predict(self, x: np.ndarray) -> np.ndarray:
        source = _matrix(x, 'x')
        if source.shape[1] != self.coef.shape[0]:
            raise ValueError('x has the wrong number of source columns')
        return source @ self.coef + self.intercept


def score_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[dict, np.ndarray]:
    """Return native aggregate R2 and unweighted, defined-coordinate summaries.

    Constant evaluation coordinates have NaN per-coordinate R2, even if predicted
    perfectly. Their squared errors still contribute to the native aggregate.
    Fractions use only defined coordinates; no negative score is clipped.
    """
    actual, predicted = _prediction_pair(y_true, y_pred)
    sst = _target_sst(actual)
    sse = _squared_norm(actual - predicted, axis=0)
    defined = sst > 0
    per_feature = np.full(actual.shape[1], np.nan)
    per_feature[defined] = 1 - sse[defined] / sst[defined]
    values = per_feature[defined]
    total_sst = float(sst.sum())
    quartiles = np.quantile(values, [0.25, 0.5, 0.75]) if values.size else [np.nan] * 3
    summary = {
        'native_r2': float(1 - sse.sum() / total_sst) if total_sst > 0 else float('nan'),
        'macro_r2': float(values.mean()) if values.size else float('nan'),
        'median_r2': float(quartiles[1]),
        'q25_r2': float(quartiles[0]),
        'q75_r2': float(quartiles[2]),
        'defined_features': int(defined.sum()),
        'undefined_features': int((~defined).sum()),
        **{
            f'fraction_r2_gt_{suffix}': float(np.mean(values > threshold)) if values.size else float('nan')
            for suffix, threshold in (('05', 0.5), ('08', 0.8), ('09', 0.9))
        },
    }
    return summary, per_feature


def select_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alphas: Iterable[float],
    *,
    scoring: str = 'native',
    source: RidgeSource | None = None,
) -> tuple[RidgeMap, list[dict]]:
    """Select a train-only affine ridge map by validation R2.

    On train-standardized nonconstant sources, the objective is
    ``||Y - X W||_F**2 / n_train + alpha * ||W||_F**2`` with an
    unpenalized intercept. For macro selection, Y is train-standardized too.
    Independent output solves make this target scaling cancel when folding
    back to raw units; macro validation weighting, not target units, determines
    selection. Native selection retains native target units.

    A single spectrum is reused across the grid, along with validation
    quadratic sufficient statistics. Zero-alpha or wide designs use an SVD;
    positive-alpha tall designs use the smaller source covariance. Scores
    tied within 1e-12 relative/absolute tolerance favor larger alpha. The
    descending sweep records selection and boundary diagnostics, without
    extending the grid or accessing test data. Boundary selections, including
    strongest-regularization null controls, are outcomes rather than failures.
    An optional prepared source reuses the factorization and source-only
    validation products across target sets; its training fingerprint must match.
    """
    xtr, ytr = _matrix(x_train, 'x_train'), _matrix(y_train, 'y_train')
    xva, yva = _matrix(x_val, 'x_val'), _matrix(y_val, 'y_val')
    if xtr.shape[0] != ytr.shape[0] or xva.shape[0] != yva.shape[0]:
        raise ValueError('source and target row counts must match within each split')
    if xtr.shape[1] != xva.shape[1] or ytr.shape[1] != yva.shape[1]:
        raise ValueError('train and validation dimensions must match')
    if scoring not in ('native', 'macro'):
        raise ValueError("scoring must be 'native' or 'macro'")
    grid = np.asarray(list(alphas), dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or (grid < 0).any():
        raise ValueError('alphas must be a nonempty grid of finite nonnegative values')
    grid = np.unique(grid)[::-1]
    sst = _target_sst(yva)
    if not np.any(sst > 0):
        raise ValueError(f'undefined whole validation {scoring} R2: all targets are constant')
    weights = np.full(ytr.shape[1], 1 / sst.sum())
    if scoring == 'macro':
        weights.fill(0)
        defined = sst > 0
        weights[defined] = 1 / (defined.sum() * sst[defined])

    if source is None:
        source = _prepare_ridge_source(xtr, use_svd=grid[-1] == 0)
    elif not isinstance(source, RidgeSource):
        raise ValueError('source must be returned by prepare_ridge_source')
    elif source._fingerprint != _fingerprint(xtr):
        raise ValueError('prepared source does not match x_train values and row order')
    if grid[-1] == 0:
        source._use_svd()
    source_constant = source.source_constant
    active = ~source_constant
    xmean, scale = source.mean, source.scale
    eigenvalues, basis = source.eigenvalues, source.basis
    target, ymean = _center(ytr)
    target_scale = np.ones(ytr.shape[1])
    if scoring == 'macro':
        target_scale = np.sqrt(_squared_norm(target, axis=0) / len(ytr))
        target_scale[target_scale == 0] = 1
        target = target / target_scale

    cross = source._target_cross(target) * target_scale

    validation_basis, validation_gram = source._validation_statistics(xva)
    weighted_cross = cross * np.sqrt(weights)
    weighted_target = (yva - ymean) * np.sqrt(weights)
    constant = float(_squared_norm(weighted_target))
    linear = np.sum(weighted_cross * (validation_basis.T @ weighted_target), axis=1)
    quadratic = validation_gram * (weighted_cross @ weighted_cross.T)
    best_score = -np.inf
    best_alpha = float(grid[0])
    sweep = []
    changed_constants = int(np.any(xva[:, source_constant] != xtr[0, source_constant], axis=0).sum())
    for value in grid:
        alpha = float(value)
        inverse = 1 / (eigenvalues + len(xtr) * alpha)
        loss = constant - 2 * linear @ inverse + inverse @ quadratic @ inverse
        # The quadratic is a squared norm; roundoff can make exact recovery < 0.
        score = float(1 - max(float(loss), 0.0))
        if not np.isfinite(score):
            raise ValueError(f'non-finite validation score at alpha={alpha:g}')
        if score > best_score and not np.isclose(score, best_score, rtol=1e-12, atol=1e-12):
            best_score, best_alpha = score, alpha
        boundary = 'only' if len(grid) == 1 else 'upper' if value == grid[0] else 'lower' if value == grid[-1] else None
        sweep.append({
            'alpha': alpha, 'validation_score': score, 'scoring': scoring,
            'boundary': boundary,
            'source_constant_changed_in_validation': changed_constants,
        })

    coef = np.zeros((xtr.shape[1], ytr.shape[1]))
    coef[active] = (basis @ (cross / (eigenvalues[:, None] + len(xtr) * best_alpha))) / scale[:, None]
    fitted = RidgeMap(coef, ymean - xmean @ coef, best_alpha, source_constant.copy())
    for row in sweep:
        row['selected'] = row['alpha'] == best_alpha
        row['selected_on_boundary'] = row['selected'] and row['boundary'] is not None
    return fitted, sweep


def _bootstrap_statistics(
    actual: np.ndarray,
    row_values: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(n_boot, bool) or not isinstance(n_boot, Integral) or n_boot < 1:
        raise ValueError('n_boot must be a positive integer')
    centered, _ = _center(actual)
    row_norm = _squared_norm(centered, axis=1)
    numerator = np.empty(n_boot)
    denominator = np.empty(n_boot)
    rng = np.random.default_rng(seed)
    size = len(actual)
    # Bounded batches use row multiplicities, never a (bootstrap, row, feature) tensor.
    for start in range(0, n_boot, 32):
        stop = min(start + 32, n_boot)
        indices = rng.integers(0, size, size=(stop - start, size))
        counts = np.array([np.bincount(row, minlength=size) for row in indices], dtype=np.float64)
        second_moment = counts @ row_norm
        summed = counts @ centered
        totals = second_moment - _squared_norm(summed, axis=1) / size
        unstable = totals <= 64 * np.finfo(np.float64).eps * second_moment
        for index in np.flatnonzero(unstable):
            present = counts[index] > 0
            sample = actual[present] - actual[present][0]
            sample -= (counts[index, present] @ sample) / size
            totals[index] = counts[index, present] @ _squared_norm(sample, axis=1)
        numerator[start:stop] = counts @ row_values
        denominator[start:stop] = totals
    return numerator, denominator


def _interval(values: np.ndarray, prefix: str) -> dict:
    defined = np.isfinite(values)
    bounds = np.quantile(values[defined], [0.025, 0.975]) if defined.any() else [np.nan, np.nan]
    return {
        f'{prefix}_ci_low': float(bounds[0]),
        f'{prefix}_ci_high': float(bounds[1]),
        'bootstrap_defined': int(defined.sum()),
        'bootstrap_undefined': int((~defined).sum()),
    }


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Row-bootstrap native R2 with each resample's own target-mean denominator.

    Return 95% percentile intervals. Undefined constant resamples are excluded
    with explicit counts; an entirely undefined interval is NaN.
    """
    actual, predicted = _prediction_pair(y_true, y_pred)
    losses = _squared_norm(actual - predicted, axis=1)
    numerator, denominator = _bootstrap_statistics(actual, losses, n_boot, seed)
    values = np.full(n_boot, np.nan)
    defined = denominator > 0
    values[defined] = 1 - numerator[defined] / denominator[defined]
    return _interval(values, 'native')


def paired_r2(
    y_true: np.ndarray,
    baseline_prediction: np.ndarray,
    candidate_prediction: np.ndarray,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict:
    """Paired native R2 difference: candidate minus baseline (e.g. H377 - H33)."""
    actual, baseline = _prediction_pair(y_true, baseline_prediction)
    _, candidate = _prediction_pair(actual, candidate_prediction)
    improvement = _squared_norm(actual - baseline, axis=1) - _squared_norm(actual - candidate, axis=1)
    numerator, denominator = _bootstrap_statistics(actual, improvement, n_boot, seed)
    values = np.full(n_boot, np.nan)
    defined = denominator > 0
    values[defined] = numerator[defined] / denominator[defined]
    total_sst = float(_target_sst(actual).sum())
    return {
        'delta_r2': float(improvement.sum() / total_sst) if total_sst > 0 else float('nan'),
        **_interval(values, 'delta'),
    }


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA from feature products, without sample Gram matrices."""
    first, second = _matrix(x, 'x'), _matrix(y, 'y')
    if len(first) != len(second):
        raise ValueError('x and y must have the same number of rows')
    first, _ = _center(first)
    second, _ = _center(second)
    first_scale, second_scale = np.max(np.abs(first)), np.max(np.abs(second))
    if first_scale == 0 or second_scale == 0:
        return float('nan')
    first, second = first / first_scale, second / second_scale
    numerator = float(_squared_norm(first.T @ second))
    denominator = np.linalg.norm(first.T @ first) * np.linalg.norm(second.T @ second)
    return float(numerator / denominator)


@dataclass
class TargetPCA:
    """Reusable train-fitted target PCA; components are ordered column vectors."""

    mean: np.ndarray
    components: np.ndarray
    variance: np.ndarray

    def diagnostics(
        self,
        actual_target: np.ndarray,
        predicted_target: np.ndarray,
        ranks: Iterable[int],
    ) -> list[dict]:
        """Compare actual and predicted reconstructions in the same fixed basis.

        The oracle uses actual evaluation coordinates. It is a compression
        reference, not a held-out upper bound for an unrestricted decoder.
        Prediction quality need not improve monotonically with rank.
        """
        actual, predicted = _prediction_pair(actual_target, predicted_target)
        if actual.shape[1] != self.mean.size:
            raise ValueError('evaluation target dimensions do not match the training PCA')
        ranks = list(ranks)
        if any(isinstance(rank, bool) or not isinstance(rank, Integral)
               or rank < 0 or rank > self.components.shape[1] for rank in ranks):
            raise ValueError('PCA ranks must be integers between zero and min(train rows, target columns)')
        if not ranks:
            return []
        basis = self.components[:, :max(ranks)]
        centered = actual - self.mean
        actual_coordinates = centered @ basis
        predicted_coordinates = (predicted - self.mean) @ basis
        energy = np.r_[0.0, np.cumsum(_squared_norm(actual_coordinates, axis=0))]
        errors = np.r_[0.0, np.cumsum(_squared_norm(actual_coordinates - predicted_coordinates, axis=0))]
        training = np.r_[0.0, np.cumsum(self.variance)]
        total_training = float(training[-1])
        total_energy = float(_squared_norm(centered))
        denominator = float(_target_sst(actual).sum())
        result = []
        for rank in ranks:
            oracle_error = max(total_energy - float(energy[rank]), 0.0)
            result.append({
                'rank': int(rank),
                'training_variance_fraction': float(training[rank] / total_training) if total_training > 0 else float('nan'),
                'oracle_reconstruction_r2': 1 - oracle_error / denominator if denominator > 0 else float('nan'),
                'prediction_reconstruction_r2': 1 - (oracle_error + float(errors[rank])) / denominator if denominator > 0 else float('nan'),
            })
        return result


def fit_target_pca(train_target: np.ndarray) -> TargetPCA:
    """Fit once on train; reuse ``.diagnostics`` for multiple fixed predictors."""
    train = _matrix(train_target, 'train_target')
    if len(train) < 2:
        raise ValueError('PCA requires at least two training rows')
    centered, mean = _center(train)
    if train.shape[0] < train.shape[1]:
        _, singular, right = np.linalg.svd(centered, full_matrices=False)
        return TargetPCA(mean, right.T, np.square(singular) / (len(train) - 1))
    variance, components = np.linalg.eigh(centered.T @ centered / (len(train) - 1))
    return TargetPCA(mean, components[:, ::-1], np.maximum(variance[::-1], 0))


def pca_diagnostics(
    train_target: np.ndarray,
    actual_target: np.ndarray,
    predicted_target: np.ndarray,
    ranks: Iterable[int],
) -> list[dict]:
    """Convenience API for a train-only PCA reconstruction reference."""
    return fit_target_pca(train_target).diagnostics(actual_target, predicted_target, ranks)
