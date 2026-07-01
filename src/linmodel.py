"""Tiny linear classifier utilities with standardiser folding.

We train a logistic-regression classifier on standardised features for good
optimisation, then *fold* the standardiser into the linear weights so the
deployed model is a single affine map on the raw (parameter-free) features:

    z = (x - mu) / sigma ,  logits = W z + b
      = (W / sigma) x + (b - sum(W mu / sigma))
      = W_eff x + b_eff

The deployed parameter count is therefore exactly ``n_classes * (F + 1)`` --
no separate scaler, no feature-extractor parameters.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


def fit_folded_logreg(
    x: np.ndarray,
    y: np.ndarray,
    feature_idx: np.ndarray | None = None,
    C: float = 10.0,
    max_iter: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a standardised logreg and fold the scaler into the weights.

    Args:
        x: (N, F_all) raw feature matrix.
        y: (N,) integer labels.
        feature_idx: indices of columns to use (defaults to all).
        C: inverse L2 strength.
        max_iter: solver iterations.

    Returns:
        (W_eff [K, k], b_eff [K], feature_idx [k]) operating on the *raw*
        selected features: ``logits = x[:, feature_idx] @ W_eff.T + b_eff``.
    """
    if feature_idx is None:
        feature_idx = np.arange(x.shape[1])
    feature_idx = np.asarray(feature_idx)
    xs = x[:, feature_idx]
    sc = StandardScaler().fit(xs)
    clf = LogisticRegression(max_iter=max_iter, C=C).fit(sc.transform(xs), y)
    mu, sigma = sc.mean_, sc.scale_
    w_eff = clf.coef_ / sigma[None, :]
    b_eff = clf.intercept_ - (clf.coef_ * (mu / sigma)[None, :]).sum(axis=1)
    return w_eff.astype(np.float32), b_eff.astype(np.float32), feature_idx


def predict(x: np.ndarray, w_eff: np.ndarray, b_eff: np.ndarray,
            feature_idx: np.ndarray) -> np.ndarray:
    """Predict class indices from raw features using folded weights."""
    logits = x[:, feature_idx] @ w_eff.T + b_eff
    return logits.argmax(1)


def num_params(w_eff: np.ndarray, b_eff: np.ndarray) -> int:
    """Deployed parameter count of the folded linear model."""
    return int(w_eff.size + b_eff.size)


def l1_rank(x: np.ndarray, y: np.ndarray, C: float = 0.05,
            max_iter: int = 2000) -> np.ndarray:
    """Rank feature columns by max |coef| across classes from an L1 logreg.

    Uses the ``liblinear`` L1 solver wrapped in an explicit one-vs-rest scheme,
    which is dramatically faster than ``saga`` for this problem size and gives an
    equivalent ranking.  (Newer scikit-learn no longer lets ``liblinear`` do
    multiclass implicitly, so the OvR wrapper is now required.)
    """
    sc = StandardScaler().fit(x)
    base = LogisticRegression(
        penalty='l1', solver='liblinear', C=C, max_iter=max_iter,
    )
    clf = OneVsRestClassifier(base).fit(sc.transform(x), y)
    coef = np.vstack([est.coef_.ravel() for est in clf.estimators_])  # (K, F)
    imp = np.abs(coef).max(0)
    return np.argsort(-imp)
