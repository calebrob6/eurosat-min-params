"""Logistic regression with the standardizer folded into the weights.

Training uses standardized features; deployment is one affine map on the raw
features, ``logits = x @ W.T + b``. Subtracting one class's row from every row
leaves predictions unchanged, so a K-class head needs only ``(K - 1)(F + 1)``
stored values.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def fit_folded_logreg(
    x: np.ndarray, y: np.ndarray, C: float = 10.0, max_iter: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Fit on standardized features and return (W, b) that act on raw features."""
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=max_iter, C=C).fit(scaler.transform(x), y)
    w = clf.coef_ / scaler.scale_
    b = clf.intercept_ - (clf.coef_ * (scaler.mean_ / scaler.scale_)).sum(1)
    return w.astype(np.float32), b.astype(np.float32)


def to_reference_class(w: np.ndarray, b: np.ndarray, ref: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Shift the head so class ``ref`` has zero weights and bias (same predictions)."""
    return w - w[ref], b - b[ref]


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (x @ w.T + b).argmax(1)


def stored_parameters(w: np.ndarray, b: np.ndarray) -> int:
    """Values a reference-class head must store: its nonzero rows."""
    rows = np.any(w != 0, axis=1) | (b != 0)
    return int(rows.sum() * (w.shape[1] + 1))
