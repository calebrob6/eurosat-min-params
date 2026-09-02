#!/usr/bin/env python
"""Shared GPU helpers for the RESISC45 minimum-parameter search.

RESISC45 has 45 classes, so a conventional reference-class affine head costs
``44 * (k + 1)`` stored values and only 22 features fit inside a 1,024-parameter
budget.  Factorising the head as ``x @ A @ C + d`` (a *linear* reduced-rank map,
no nonlinearity) costs ``k*r + 44*(r + 1)`` instead, which trades rank for
features at a fixed budget.  Both fitters standardise during optimisation and
fold the standardiser into the deployed weights, matching ``src.linmodel``.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '8')

import numpy as np
import torch

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'cache')
NUM_CLASSES = 45
SPLITS = ('train', 'val', 'test')


def load_pool(name: str = 'rgb_pool') -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[str]]:
    """Load a cached RESISC45 feature pool plus labels for all three splits."""
    x = {s: np.load(os.path.join(CACHE_DIR, f'resisc45_{s}_{name}.npy')) for s in SPLITS}
    y = {s: np.load(os.path.join(CACHE_DIR, f'resisc45_{s}_y.npy')) for s in SPLITS}
    names_path = os.path.join(CACHE_DIR, f'resisc45_{name}_names.txt')
    names = open(names_path).read().split() if os.path.exists(names_path) else []
    return x, y, names


def standardise(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the (mu, sigma) used to standardise a feature matrix."""
    mu = train.mean(0).astype(np.float64)
    sigma = train.std(0).astype(np.float64)
    sigma[sigma < 1e-12] = 1.0
    return mu, sigma


def _to_device(a: np.ndarray, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype, device=device)


def _fit_logreg_std(
    xs: torch.Tensor, yt: torch.Tensor, C: float, steps: int, n_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-batch LBFGS logistic regression on already-standardised features."""
    n, k = xs.shape
    w = torch.zeros(n_classes, k, device=xs.device, requires_grad=True)
    b = torch.zeros(n_classes, device=xs.device, requires_grad=True)
    l2 = 1.0 / (2.0 * C * n)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(xs @ w.T + b, yt) + l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def fit_logreg_gpu(
    x: np.ndarray,
    y: np.ndarray,
    C: float = 10.0,
    steps: int = 300,
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch multinomial logistic regression with a folded scaler.

    Uses scikit-learn's objective ``C * sum_i loss_i + 0.5 * ||W||^2`` so the
    selected ``C`` transfers between the two implementations.

    Returns:
        ``(w_eff [K, k], b_eff [K])`` acting on the raw features.
    """
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    w, b = _fit_logreg_std(xs, yt, C, steps, int(y.max()) + 1)
    w_np = w.double().cpu().numpy()
    b_np = b.double().cpu().numpy()
    return w_np / sigma[None, :], b_np - (w_np * (mu / sigma)[None, :]).sum(1)


def fit_lowrank_gpu(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    weight_decay: float = 1e-3,
    epochs: int = 3000,
    lr: float = 0.03,
    device: str = 'cuda',
    seed: int = 0,
    init_C: float | None = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit ``logits = x_s @ A @ Cm + d`` and fold the scaler into ``A``.

    The factorised objective is non-convex, so the default initialisation is the
    rank-``r`` truncated SVD of a converged full-rank solution; Adam then refines
    it.  Set ``init_C=None`` for a random initialisation.

    Returns:
        ``(a_eff [k, r], c [r, K], d_eff [K])`` acting on the raw features.
    """
    torch.manual_seed(seed)
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    if init_C is not None:
        w0, b0 = _fit_logreg_std(xs, yt, init_C, 300, n_classes)
        u, s, vh = torch.linalg.svd(w0.double(), full_matrices=False)
        root = torch.sqrt(s[:rank])
        a = (vh[:rank].T * root).float().contiguous()
        c = (u[:, :rank] * root).T.float().contiguous()
        d = b0.clone()
    else:
        a = torch.randn(k, rank, device=device) / np.sqrt(k)
        c = torch.randn(rank, n_classes, device=device) / np.sqrt(rank)
        d = torch.zeros(n_classes, device=device)
    for t in (a, c, d):
        t.requires_grad_(True)
    opt = torch.optim.Adam([a, c, d], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy((xs @ a) @ c + d, yt)
        loss = loss + weight_decay * ((a * a).sum() + (c * c).sum())
        loss.backward()
        opt.step()
        sched.step()
    a_np = a.detach().double().cpu().numpy()
    c_np = c.detach().double().cpu().numpy()
    d_np = d.detach().double().cpu().numpy()
    return a_np / sigma[:, None], c_np, d_np - (mu / sigma) @ a_np @ c_np


def accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((pred == y).mean())


def logreg_params(k: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a reference-class affine head."""
    return (n_classes - 1) * (k + 1)


def lowrank_params(k: int, rank: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a reference-class reduced-rank head."""
    return k * rank + (n_classes - 1) * (rank + 1)


def fit_mlp_gpu(
    x: np.ndarray,
    y: np.ndarray,
    hidden: int,
    weight_decay: float = 1e-4,
    epochs: int = 3000,
    lr: float = 0.03,
    device: str = 'cuda',
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit a one-hidden-layer ReLU head with the scaler folded into layer one.

    Returns:
        ``(w1_eff [k, h], b1_eff [h], w2 [h, K], b2 [K])`` acting on raw
        features: ``logits = relu(x @ w1_eff + b1_eff) @ w2 + b2``.
    """
    torch.manual_seed(seed)
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    w1 = (torch.randn(k, hidden, device=device) / np.sqrt(k)).requires_grad_(True)
    b1 = torch.zeros(hidden, device=device, requires_grad=True)
    w2 = (torch.randn(hidden, n_classes, device=device) / np.sqrt(hidden)).requires_grad_(True)
    b2 = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam([w1, b1, w2, b2], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = torch.relu(xs @ w1 + b1) @ w2 + b2
        loss = torch.nn.functional.cross_entropy(logits, yt)
        loss = loss + weight_decay * ((w1 * w1).sum() + (w2 * w2).sum())
        loss.backward()
        opt.step()
        sched.step()
    w1_np = w1.detach().double().cpu().numpy()
    b1_np = b1.detach().double().cpu().numpy()
    return (
        w1_np / sigma[:, None],
        b1_np - (mu / sigma) @ w1_np,
        w2.detach().double().cpu().numpy(),
        b2.detach().double().cpu().numpy(),
    )


def mlp_params(k: int, hidden: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a one-hidden-layer ReLU head with a reference class."""
    return k * hidden + hidden + (n_classes - 1) * (hidden + 1)


def predict_mlp(x: np.ndarray, w1, b1, w2, b2) -> np.ndarray:
    return (np.maximum(x @ w1 + b1, 0.0) @ w2 + b2).argmax(1)


def group_lasso_rank(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 1e-3,
    epochs: int = 1500,
    lr: float = 0.05,
    device: str = 'cuda',
    rank: int | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank features by an L2,1 group penalty on their weight column.

    With ``rank=None`` the penalty is applied to a full multinomial weight
    matrix; with an integer ``rank`` it is applied to the rows of the ``k x r``
    projection of the reduced-rank head, which selects features for exactly the
    structure that will be deployed.  Optimised by proximal Adam.

    Returns:
        ``(order, norms)`` with ``order`` descending by group norm.
    """
    torch.manual_seed(seed)
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    if rank is None:
        a = torch.zeros(k, n_classes, device=device, requires_grad=True)
        params = [a]
        c = None
    else:
        a = (torch.randn(k, rank, device=device) / np.sqrt(k)).requires_grad_(True)
        c = (torch.randn(rank, n_classes, device=device) / np.sqrt(rank)).requires_grad_(True)
        params = [a, c]
    d = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.Adam(params + [d], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        logits = xs @ a + d if c is None else (xs @ a) @ c + d
        torch.nn.functional.cross_entropy(logits, yt).backward()
        opt.step()
        with torch.no_grad():  # proximal step for the row-wise L2,1 penalty
            step = lam * opt.param_groups[0]['lr']
            norm = a.norm(dim=1, keepdim=True)
            a.mul_(torch.clamp(1.0 - step / (norm + 1e-12), min=0.0))
        sched.step()
    norms = a.detach().norm(dim=1).cpu().numpy()
    return np.argsort(-norms), norms


def fit_sparse_lowrank_gpu(
    x: np.ndarray,
    y: np.ndarray,
    rank: int,
    nonzeros: int,
    weight_decay: float = 1e-4,
    epochs: int = 1500,
    rounds: int = 5,
    lr: float = 0.03,
    device: str = 'cuda',
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduced-rank head whose projection ``A`` keeps only ``nonzeros`` entries.

    The dense ``k x r`` projection is the expensive part of the budget, yet a
    rank-``r`` head fitted on the *whole* pool is far more accurate than one
    fitted on the few features a dense projection can afford.  Iterative
    magnitude pruning lets each of the ``r`` compound features draw on its own
    subset of the pool, so the stored values are ``nonzeros + 44 * (rank + 1)``
    while the union of used pool columns stays large.

    Returns:
        ``(a_eff [k, r], c [r, K], d_eff [K])`` acting on the raw features;
        ``a_eff`` has at most ``nonzeros`` nonzero entries.
    """
    torch.manual_seed(seed)
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    a = (torch.randn(k, rank, device=device) / np.sqrt(k)).requires_grad_(True)
    c = (torch.randn(rank, n_classes, device=device) / np.sqrt(rank)).requires_grad_(True)
    d = torch.zeros(n_classes, device=device, requires_grad=True)
    mask = torch.ones(k, rank, device=device)
    schedule = np.geomspace(k * rank, nonzeros, rounds + 1)[1:]
    for round_index in range(rounds + 1):
        opt = torch.optim.Adam([a, c, d], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            logits = (xs @ (a * mask)) @ c + d
            loss = torch.nn.functional.cross_entropy(logits, yt)
            loss = loss + weight_decay * (((a * mask) ** 2).sum() + (c * c).sum())
            loss.backward()
            opt.step()
            sched.step()
        if round_index == rounds:
            break
        keep = int(schedule[round_index])
        with torch.no_grad():
            score = (a * mask).abs().reshape(-1)
            cutoff = torch.topk(score, keep).values[-1]
            mask = (score >= cutoff).float().reshape(k, rank)
    a_np = (a * mask).detach().double().cpu().numpy()
    c_np = c.detach().double().cpu().numpy()
    d_np = d.detach().double().cpu().numpy()
    return a_np / sigma[:, None], c_np, d_np - (mu / sigma) @ a_np @ c_np


def sparse_lowrank_params(nonzeros: int, rank: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a sparse-projection reduced-rank reference-class head."""
    return nonzeros + (n_classes - 1) * (rank + 1)


def fit_sparse_logreg_gpu(
    x: np.ndarray,
    y: np.ndarray,
    nonzeros: int,
    weight_decay: float = 1e-4,
    epochs: int = 1500,
    rounds: int = 5,
    lr: float = 0.03,
    device: str = 'cuda',
    ref: int = 0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-rank affine head pruned to ``nonzeros`` reference-class weights.

    Pruning is applied to the reference-class form ``W - W[ref]`` that is
    actually stored, so the kept-weight count equals the deployed count.

    Returns:
        ``(w_eff [K, k], b_eff [K])`` acting on the raw features.
    """
    torch.manual_seed(seed)
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    rows = [i for i in range(n_classes) if i != ref]
    w = torch.zeros(n_classes - 1, k, device=device, requires_grad=True)
    b = torch.zeros(n_classes - 1, device=device, requires_grad=True)
    mask = torch.ones(n_classes - 1, k, device=device)
    schedule = np.geomspace((n_classes - 1) * k, nonzeros, rounds + 1)[1:]
    zero = torch.zeros(1, k, device=device)
    for round_index in range(rounds + 1):
        opt = torch.optim.Adam([w, b], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        for _ in range(epochs):
            opt.zero_grad(set_to_none=True)
            rest = xs @ (w * mask).T + b
            logits = torch.cat((torch.zeros_like(rest[:, :1]), rest), dim=1)
            loss = torch.nn.functional.cross_entropy(logits, yt)
            loss = loss + weight_decay * ((w * mask) ** 2).sum()
            loss.backward()
            opt.step()
            sched.step()
        if round_index == rounds:
            break
        keep = int(schedule[round_index])
        with torch.no_grad():
            score = (w * mask).abs().reshape(-1)
            cutoff = torch.topk(score, keep).values[-1]
            mask = (score >= cutoff).float().reshape(n_classes - 1, k)
    w_np = (w * mask).detach().double().cpu().numpy()
    b_np = b.detach().double().cpu().numpy()
    w_full = np.insert(w_np, ref, 0.0, axis=0)
    b_full = np.insert(b_np, ref, 0.0)
    return w_full / sigma[None, :], b_full - (w_full * (mu / sigma)[None, :]).sum(1)


def sparse_logreg_params(nonzeros: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a sparse reference-class affine head."""
    return nonzeros + (n_classes - 1)


def _fit_masked_ref_logreg(
    xs: torch.Tensor,
    yt: torch.Tensor,
    mask: torch.Tensor | None,
    C: float,
    steps: int,
    n_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convex reference-class logistic regression on a *fixed* support mask.

    The reference row is class 0 and is held at zero, so the stored rows are
    ``mask.sum()`` weights plus ``n_classes - 1`` biases.  With the support
    fixed the problem is convex, so LBFGS reaches the same optimum every time.
    """
    n, k = xs.shape
    w = torch.zeros(n_classes - 1, k, device=xs.device, requires_grad=True)
    b = torch.zeros(n_classes - 1, device=xs.device, requires_grad=True)
    l2 = 1.0 / (2.0 * C * n)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, history_size=20,
                            tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        wm = w if mask is None else w * mask
        rest = xs @ wm.T + b
        logits = torch.cat((torch.zeros_like(rest[:, :1]), rest), dim=1)
        loss = torch.nn.functional.cross_entropy(logits, yt) + l2 * (wm * wm).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        wm = w.detach() if mask is None else w.detach() * mask
    return wm, b.detach()


def _fold_ref(w: torch.Tensor, b: torch.Tensor, mu, sigma, ref: int = 0):
    """Insert the zero reference row and fold the standardiser into the head."""
    w_np = w.double().cpu().numpy()
    b_np = b.double().cpu().numpy()
    w_full = np.insert(w_np, ref, 0.0, axis=0)
    b_full = np.insert(b_np, ref, 0.0)
    return w_full / sigma[None, :], b_full - (w_full * (mu / sigma)[None, :]).sum(1)


def _support_clusters(
    profile: np.ndarray, n_groups: int, per_group_k: int, iters: int = 25, seed: int = 0
) -> np.ndarray:
    """Partition rows so that each group is well covered by one shared support.

    ``profile`` holds one unit-norm nonnegative row per non-reference class
    describing how strongly that class uses each pool column.  The objective is
    the total retained energy ``sum_c sum_{f in S_g(c)} profile[c, f]^2`` where
    each group's support ``S_g`` is its own top-``per_group_k`` columns; the
    alternating updates below are the k-means algorithm for that objective.
    """
    rows, k = profile.shape
    if n_groups >= rows:
        return np.arange(rows)
    rng = np.random.default_rng(seed)
    energy = profile ** 2
    centres = [int(rng.integers(rows))]
    for _ in range(n_groups - 1):  # k-means++ seeding under cosine distance
        sim = energy @ energy[centres].T
        d = 1.0 - sim.max(1)
        d = np.clip(d, 0.0, None)
        if d.sum() <= 0:
            centres.append(int(rng.integers(rows)))
        else:
            centres.append(int(rng.choice(rows, p=d / d.sum())))
    groups = np.argmax(energy @ energy[centres].T, axis=1)
    for _ in range(iters):
        supports = np.zeros((n_groups, k), dtype=bool)
        for g in range(n_groups):
            members = energy[groups == g]
            score = members.sum(0) if len(members) else energy.sum(0)
            supports[g, np.argpartition(-score, per_group_k - 1)[:per_group_k]] = True
        retained = energy @ supports.T.astype(np.float64)
        new_groups = retained.argmax(1)
        for g in range(n_groups):  # keep every group nonempty
            if not (new_groups == g).any():
                new_groups[int(retained[:, g].argmax())] = g
        if (new_groups == groups).all():
            break
        groups = new_groups
    return groups


def _block_mask(w: torch.Tensor, groups: np.ndarray, keep: int, n_groups: int) -> torch.Tensor:
    """Keep each group's top-``keep`` columns, scored by its members' energy."""
    rows, k = w.shape
    mask = torch.zeros(rows, k, device=w.device)
    energy = (w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)) ** 2
    gt = torch.as_tensor(groups, device=w.device)
    for g in range(n_groups):
        member = torch.nonzero(gt == g, as_tuple=True)[0]
        if member.numel() == 0:
            continue
        idx = torch.topk(energy[member].sum(0), min(keep, k)).indices
        mask[member[:, None], idx[None, :]] = 1.0
    return mask


def fit_block_sparse_logreg_gpu(
    x: np.ndarray,
    y: np.ndarray,
    n_groups: int,
    per_group_k: int,
    C: float = 10.0,
    steps: int = 300,
    rounds: int = 6,
    device: str = 'cuda',
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reference-class head whose 44 rows share only ``n_groups`` feature lists.

    A fully unstructured sparse head needs one ``(class, column)`` index per
    stored weight, which is a far larger deployment artefact than the single
    feature list every dense result in this project carries.  Restricting the
    support to ``n_groups`` shared lists of ``per_group_k`` columns keeps the
    stored-value count at ``44 * (per_group_k + 1)`` while shrinking the index
    pattern to ``n_groups * per_group_k`` column ids plus 44 group labels.
    ``n_groups=1`` is ordinary feature selection and ``n_groups=44`` is the
    unstructured head with an equal per-class allocation.

    Returns:
        ``(w_eff [K, k], b_eff [K], groups [K-1], mask [K-1, k])``.
    """
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    k = xs.shape[1]
    n_classes = int(y.max()) + 1
    w, b = _fit_masked_ref_logreg(xs, yt, None, C, steps, n_classes)
    profile = (w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)).abs().double().cpu().numpy()
    groups = _support_clusters(profile, n_groups, per_group_k, seed=seed)
    mask = None
    for keep in np.geomspace(k, per_group_k, rounds + 1)[1:]:
        mask = _block_mask(w, groups, int(round(keep)), n_groups)
        w, b = _fit_masked_ref_logreg(xs, yt, mask, C, steps, n_classes)
    w_eff, b_eff = _fold_ref(w, b, mu, sigma)
    return w_eff, b_eff, groups, mask.cpu().numpy()


def refit_masked_ref_logreg_gpu(
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    C: float = 10.0,
    steps: int = 300,
    device: str = 'cuda',
) -> tuple[np.ndarray, np.ndarray]:
    """Refit a reference-class head on a fixed support mask at a given ``C``."""
    mu, sigma = standardise(x)
    xs = _to_device((x - mu) / sigma, device)
    yt = _to_device(y, device, torch.long)
    mt = _to_device(mask, device)
    w, b = _fit_masked_ref_logreg(xs, yt, mt, C, steps, int(y.max()) + 1)
    return _fold_ref(w, b, mu, sigma)


def block_sparse_params(per_group_k: int, n_classes: int = NUM_CLASSES) -> int:
    """Stored values of a block-sparse reference-class head."""
    return (n_classes - 1) * (per_group_k + 1)
