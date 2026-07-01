"""Tiny MLP head on fixed (parameter-free) features, with standardiser folding.

A linear classifier spends ``10 * (F + 1)`` parameters -- 10 per feature.  A
small ``F -> H -> 10`` MLP can reach the same accuracy from *fewer* features
because the ReLU nonlinearity separates the crop/vegetation classes that a
linear boundary confuses.  As long as ``H`` is modest and ``F`` is small the
total head is cheaper than the linear model at equal accuracy.

Just like :mod:`src.linmodel`, we standardise the features for optimisation and
then *fold* the standardiser into the first linear layer, so the deployed model
operates directly on the raw (zero-parameter) features:

    z = (x - mu) / sigma
    h = relu(W1 z + b1) = relu((W1/sigma) x + (b1 - W1 (mu/sigma)))
    logits = W2 h + b2

The deployed parameter count is exactly ``F*H + H + H*K + K`` -- no separate
scaler, no feature-extractor parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # torch is only needed for *training*; deploy/predict is pure numpy.
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


@dataclass
class FoldedMLP:
    """Deployed tiny MLP operating on raw selected features (numpy only)."""

    w1: np.ndarray  # (H, F)
    b1: np.ndarray  # (H,)
    w2: np.ndarray  # (K, H)
    b2: np.ndarray  # (K,)
    feature_idx: np.ndarray  # (F,)

    def logits(self, x: np.ndarray) -> np.ndarray:
        xs = x[:, self.feature_idx]
        h = np.maximum(xs @ self.w1.T + self.b1, 0.0)
        return h @ self.w2.T + self.b2

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.logits(x).argmax(1)

    @property
    def num_params(self) -> int:
        return int(self.w1.size + self.b1.size + self.w2.size + self.b2.size)


def fit_folded_mlp(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    feature_idx: np.ndarray,
    hidden: int = 12,
    epochs: int = 400,
    lr: float = 3e-3,
    weight_decay: float = 1e-3,
    seed: int = 0,
    device: str = 'cuda',
    num_classes: int = 10,
) -> tuple[FoldedMLP, float, float]:
    """Train a tiny ``F -> H -> K`` MLP on standardised selected features.

    Model selection is done on the validation split: the epoch with the best
    val accuracy is kept.  Returns ``(folded_model, best_val_acc, train_acc)``.
    """
    assert torch is not None, 'torch required for training'
    torch.manual_seed(seed)
    np.random.seed(seed)

    fi = np.asarray(feature_idx)
    xs_tr = x_tr[:, fi].astype(np.float32)
    xs_va = x_va[:, fi].astype(np.float32)
    mu = xs_tr.mean(0)
    sigma = xs_tr.std(0) + 1e-6
    ztr = (xs_tr - mu) / sigma
    zva = (xs_va - mu) / sigma

    dev = device if (device == 'cpu' or torch.cuda.is_available()) else 'cpu'
    Xtr = torch.from_numpy(ztr).to(dev)
    Ytr = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
    Xva = torch.from_numpy(zva).to(dev)

    F = len(fi)
    net = nn.Sequential(
        nn.Linear(F, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)
    ).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()

    best_val = -1.0
    best_state = None
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xtr), Ytr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            va = (net(Xva).argmax(1).cpu().numpy() == y_va).mean()
        if va > best_val:
            best_val = float(va)
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    net.load_state_dict(best_state)
    with torch.no_grad():
        tr_acc = float((net(Xtr).argmax(1).cpu().numpy() == y_tr).mean())

    # Extract weights and fold the standardiser into layer 1.
    w1 = net[0].weight.detach().cpu().numpy().astype(np.float64)  # (H, F)
    b1 = net[0].bias.detach().cpu().numpy().astype(np.float64)    # (H,)
    w2 = net[2].weight.detach().cpu().numpy().astype(np.float32)  # (K, H)
    b2 = net[2].bias.detach().cpu().numpy().astype(np.float32)    # (K,)

    inv = 1.0 / sigma
    w1_eff = (w1 * inv[None, :]).astype(np.float32)
    b1_eff = (b1 - (w1 * (mu * inv)[None, :]).sum(1)).astype(np.float32)

    model = FoldedMLP(w1_eff, b1_eff, w2, b2, fi.astype(np.int64))
    return model, best_val, tr_acc
