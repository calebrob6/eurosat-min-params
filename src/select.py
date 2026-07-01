"""Backward-greedy feature elimination for minimal-parameter linear models.

The L1-rank ``top-k`` order (see :func:`src.linmodel.l1_rank`) is only a *proxy*
for the best k-feature subset: it ranks features once by their L1 coefficient
magnitude and never reconsiders.  Backward-greedy elimination instead starts from
a larger known-good set (e.g. the L1 top-42) and repeatedly removes the single
feature whose removal *least* hurts (or most helps) a cross-validated accuracy,
re-fitting after every drop.  Because it re-evaluates the whole remaining set at
each step it finds markedly smaller subsets that still clear the accuracy bar --
on EuroSAT it reaches the 0.940 honest floor at far fewer features than L1-rank.

Honesty note: the greedy is guided by CV over ``select_seeds``.  That CV is
therefore *optimistically biased* for the chosen subset (features were picked to
maximise it).  Always confirm the resulting subset on an independent estimate --
a disjoint set of CV shuffle seeds and/or the held-out val split -- before
trusting the floor.  :func:`backward_eliminate` returns the full trace so the
caller can apply such a rule.

The candidate evaluations at each step fan out over a ``ProcessPoolExecutor``
with BLAS threads left to the caller to pin (set ``OMP_NUM_THREADS=1`` etc. for
speed, as elsewhere in this project).
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from .linmodel import fit_folded_logreg, predict

# worker-process globals (populated by the pool initializer to avoid pickling the
# full feature matrix on every candidate evaluation)
_X: np.ndarray | None = None
_Y: np.ndarray | None = None


def mean_cv(x: np.ndarray, y: np.ndarray, idx: np.ndarray,
            seeds, folds: int = 5, C: float = 10.0) -> float:
    """Mean StratifiedKFold accuracy of a folded-logreg on features ``idx``.

    Averaged over ``folds``-fold CV repeated for each shuffle seed in ``seeds``
    (train split only).  This is the de-noised selection metric used throughout
    the project.
    """
    vals = []
    for sd in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=sd)
        accs = [accuracy_score(y[te], predict(x[te],
                *fit_folded_logreg(x[tr], y[tr], feature_idx=idx, C=C)))
                for tr, te in skf.split(x, y)]
        vals.append(np.mean(accs))
    return float(np.mean(vals))


def _init(x, y):
    global _X, _Y
    _X, _Y = x, y


def _drop_score(args):
    idx, j, seeds, folds, C = args
    return j, mean_cv(_X, _Y, np.delete(idx, j), seeds, folds, C)


def backward_eliminate(
    x: np.ndarray,
    y: np.ndarray,
    init_idx: np.ndarray,
    target_k: int,
    *,
    select_seeds=range(10),
    folds: int = 5,
    C: float = 10.0,
    workers: int | None = None,
):
    """Greedy backward elimination from ``init_idx`` down to ``target_k`` features.

    At each step the feature whose removal MAXIMISES the mean CV over
    ``select_seeds`` is dropped (ties broken by lowest index).  The process is
    deterministic given ``(x, y, init_idx, select_seeds, folds, C)``.

    Args:
        x: (N, F_all) raw feature matrix (the full pool).
        y: (N,) integer labels.
        init_idx: indices into the pool to start from (e.g. the L1 top-42).
        target_k: number of features to stop at.
        select_seeds: CV shuffle seeds guiding the greedy removals.
        folds, C: CV folds and logreg inverse-L2 strength.
        workers: process-pool size (defaults to min(16, cpu-2)).

    Returns:
        (feature_idx [target_k] into the pool, trace) where ``trace`` is a list of
        ``(k, select_cv)`` from ``len(init_idx)`` down to ``target_k``.
    """
    idx = np.asarray(init_idx).copy()
    seeds = list(select_seeds)
    workers = workers or min(16, (os.cpu_count() or 4) - 2)
    trace: list[tuple[int, float]] = []
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init, initargs=(x, y)) as ex:
        while True:
            trace.append((len(idx), mean_cv(x, y, idx, seeds, folds, C)))
            if len(idx) <= target_k:
                break
            args = [(idx, j, seeds, folds, C) for j in range(len(idx))]
            results = list(ex.map(_drop_score, args))
            best_j, _ = max(results, key=lambda t: t[1])
            idx = np.delete(idx, best_j)
    return idx, trace
