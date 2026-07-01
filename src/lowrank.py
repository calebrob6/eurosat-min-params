"""Reduced-rank linear classifier with standardiser folding.

A plain multinomial-logistic classifier on ``F`` features has a ``K x F`` weight
matrix -> ``K*(F+1)`` parameters.  But the class-discriminative subspace is at
most ``(K-1)``-dimensional, so that weight matrix is inherently low rank.  We
factor it explicitly:

    logits = (x_s @ A) @ C + d ,   A: F x r ,  C: r x K ,  d: K

with ``x_s`` the standardised features.  There is **no nonlinearity**, so the
model is still a *linear* classifier (``W = A @ C`` has rank <= r).  It does not
*add* capacity like an MLP head; it *constrains* capacity (rank regularisation),
so it should not overfit the way the MLP/quadratic heads did.  Trained by
cross-entropy + weight decay this is reduced-rank multinomial logistic
regression.

Folding the standardiser (``x_s = (x - mu) / sigma``) into ``A`` gives a deployed
model operating on the *raw* (parameter-free) selected features:

    logits = (x[:, idx] @ A_eff) @ C + d_eff

so the deployed parameter count is exactly ``k*r + r*K + K`` -- no separate
scaler and no feature-extractor parameters.
"""

from __future__ import annotations

import numpy as np

try:  # torch is only needed for *training*; deploy/predict is pure numpy.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None


def fit_lowrank(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    feature_idx: np.ndarray | None = None,
    weight_decay: float = 1e-3,
    epochs: int = 3000,
    lr: float = 0.05,
    seed: int = 0,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a reduced-rank linear classifier and fold the scaler into ``A``.

    Args:
        x: (N, F_all) raw feature matrix.
        y: (N,) integer labels in ``[0, K)``.
        rank: bottleneck rank ``r`` (factorises the ``k x K`` weight matrix).
        feature_idx: columns of ``x`` to use (defaults to all).
        weight_decay: L2 penalty on ``A`` and ``C`` (rank + shrinkage reg).
        epochs: full-batch optimisation steps.
        lr: Adam learning rate.
        seed: torch RNG seed for the weight init (results are saved, so this
            only affects which optimum we land in).
        device: 'cuda'/'cpu'; defaults to cuda if available.

    Returns:
        (A_eff [k, r], C [r, K], d_eff [K], feature_idx [k]) operating on the
        *raw* selected features: ``logits = (x[:, idx] @ A_eff) @ C + d_eff``.
    """
    if torch is None:  # pragma: no cover
        raise RuntimeError('fit_lowrank requires torch')
    if feature_idx is None:
        feature_idx = np.arange(x.shape[1])
    feature_idx = np.asarray(feature_idx)
    xs_raw = x[:, feature_idx].astype(np.float64)
    mu = xs_raw.mean(0)
    sigma = xs_raw.std(0)
    sigma[sigma == 0] = 1.0
    xs = (xs_raw - mu) / sigma
    k = xs.shape[1]
    K = int(y.max()) + 1

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed)
    xt = torch.tensor(xs, dtype=torch.float32, device=device)
    yt = torch.tensor(y.astype(np.int64), device=device)

    A = nn.Parameter(torch.randn(k, rank, device=device) * (1.0 / np.sqrt(k)))
    C = nn.Parameter(torch.randn(rank, K, device=device) * (1.0 / np.sqrt(rank)))
    d = nn.Parameter(torch.zeros(K, device=device))
    opt = torch.optim.Adam([A, C, d], lr=lr, weight_decay=weight_decay)
    lossfn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        logits = (xt @ A) @ C + d
        loss = lossfn(logits, yt)
        loss.backward()
        opt.step()

    A_np = A.detach().cpu().numpy().astype(np.float64)
    C_np = C.detach().cpu().numpy().astype(np.float64)
    d_np = d.detach().cpu().numpy().astype(np.float64)

    # Fold the standardiser into A: x_s = (x - mu)/sigma, so
    #   z = x_s @ A = x @ (A / sigma[:,None]) - (mu/sigma) @ A
    A_eff = A_np / sigma[:, None]
    o = -(mu / sigma) @ A_np            # (r,)
    d_eff = o @ C_np + d_np             # fold the projection offset into d
    return (
        A_eff.astype(np.float32),
        C_np.astype(np.float32),
        d_eff.astype(np.float32),
        feature_idx,
    )


def fit_lowrank_svd(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    feature_idx: np.ndarray | None = None,
    C: float = 10.0,
    max_iter: int = 4000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Optimal reduced-rank linear classifier via SVD of a full linear solution.

    A full multinomial-logistic weight matrix ``W`` (``K x k``) is only defined up
    to adding a per-feature constant across classes (softmax is shift-invariant),
    so after centering ``W`` over the class axis its effective rank is at most
    ``K-1``.  We fit a well-optimised full linear model, center it, and keep the
    top-``rank`` singular directions -- the best rank-``r`` approximation of the
    linear logit function.  At ``rank == K-1`` this is loss-less (recovers the
    full linear model); smaller ``rank`` drops the least-discriminative
    directions.  The standardiser is folded into ``A`` exactly as in
    :func:`fit_lowrank`, so params == ``k*r + r*K + K``.
    """
    if feature_idx is None:
        feature_idx = np.arange(x.shape[1])
    feature_idx = np.asarray(feature_idx)
    xs_raw = x[:, feature_idx].astype(np.float64)
    mu = xs_raw.mean(0)
    sigma = xs_raw.std(0)
    sigma[sigma == 0] = 1.0
    xs = (xs_raw - mu) / sigma

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=max_iter, C=C).fit(xs, y)
    W = clf.coef_.astype(np.float64)          # (K, k)
    b = clf.intercept_.astype(np.float64)     # (K,)
    # Remove the softmax null direction so the effective rank is <= K-1.
    Wc = W - W.mean(axis=0, keepdims=True)
    bc = b - b.mean()

    U, S, Vt = np.linalg.svd(Wc, full_matrices=False)  # U (K,m) S(m,) Vt(m,k)
    r = min(rank, S.shape[0])
    A = Vt[:r].T                               # (k, r)
    Cmat = U[:, :r] * S[:r]                     # (K, r); Wc_r = Cmat @ A.T
    C_out = Cmat.T                              # (r, K), logits = (xs@A)@C_out + bc

    # Fold the standardiser into A (see fit_lowrank).
    A_eff = A / sigma[:, None]
    o = -(mu / sigma) @ A
    d_eff = o @ C_out + bc
    return (
        A_eff.astype(np.float32),
        C_out.astype(np.float32),
        d_eff.astype(np.float32),
        feature_idx,
    )


def predict_lowrank(
    x: np.ndarray, a_eff: np.ndarray, c: np.ndarray, d_eff: np.ndarray,
    feature_idx: np.ndarray,
) -> np.ndarray:
    """Predict class indices from raw features using the folded factors."""
    z = x[:, feature_idx] @ a_eff
    logits = z @ c + d_eff
    return logits.argmax(1)


def num_params_lowrank(a_eff: np.ndarray, c: np.ndarray, d_eff: np.ndarray) -> int:
    """Deployed parameter count of the folded reduced-rank model."""
    return int(a_eff.size + c.size + d_eff.size)
