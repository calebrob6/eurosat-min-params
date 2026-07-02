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


def _subset_score(args):
    """Evaluate mean CV of an explicit feature subset (used by SFBS).

    ``tag`` is any hashable the caller uses to identify the candidate (e.g. the
    feature being dropped or added); it is returned unchanged alongside the CV.
    """
    tag, subset, seeds, folds, C = args
    return tag, mean_cv(_X, _Y, np.asarray(subset), seeds, folds, C)


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
    record: bool = False,
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
        record: also return every intermediate subset (to confirm a whole k
            neighbourhood from a single descent instead of one descent per k).

    Returns:
        ``(feature_idx [target_k], trace)`` where ``trace`` is a list of
        ``(k, select_cv)`` from ``len(init_idx)`` down to ``target_k``.  If
        ``record`` is set, also returns a third value ``subsets``: a dict mapping
        each visited ``k`` to the ``np.ndarray`` of feature indices at that k.
    """
    idx = np.asarray(init_idx).copy()
    seeds = list(select_seeds)
    workers = workers or min(16, (os.cpu_count() or 4) - 2)
    trace: list[tuple[int, float]] = []
    subsets: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init, initargs=(x, y)) as ex:
        while True:
            if record:
                subsets[len(idx)] = idx.copy()
            trace.append((len(idx), mean_cv(x, y, idx, seeds, folds, C)))
            if len(idx) <= target_k:
                break
            args = [(idx, j, seeds, folds, C) for j in range(len(idx))]
            results = list(ex.map(_drop_score, args))
            best_j, _ = max(results, key=lambda t: t[1])
            idx = np.delete(idx, best_j)
    if record:
        return idx, trace, subsets
    return idx, trace


def floating_backward(
    x: np.ndarray,
    y: np.ndarray,
    universe: np.ndarray,
    target_k: int,
    *,
    select_seeds=range(10),
    folds: int = 5,
    C: float = 10.0,
    workers: int | None = None,
    log=None,
):
    """Sequential Floating Backward Selection (SFBS) over a fixed candidate pool.

    Pure backward elimination (:func:`backward_eliminate`) is one-directional: a
    feature dropped early can never come back, so the size-k subset it lands on is
    *nested* inside the size-(k+1) subset.  SFBS (Pudil et al. 1994) relaxes this:
    after each backward removal it performs *conditional forward* steps -- adding
    back the most useful excluded feature as long as doing so beats the best
    subset of that larger size found so far.  This lets it escape the nesting and
    reach a genuinely better subset at each size, at the cost of extra CV
    evaluations.

    The candidate ``universe`` is fixed (e.g. the L1 top-N): the search only ever
    considers features in it, so a comparison against ``backward_eliminate`` on
    the same universe isolates the value of the floating (re-addition) step.

    Honesty note: SFBS optimises the ``select_seeds`` CV *harder* than plain
    backward, so its selection CV is even more optimistically biased.  Confirm the
    returned per-size subsets on disjoint verify seeds / held-out data before
    trusting any floor (see :func:`mean_cv` and the experiment scripts).

    Args:
        x: (N, F_all) raw feature matrix (the full pool).
        y: (N,) integer labels.
        universe: candidate feature indices (into the pool) to search within.
        target_k: smallest subset size to float down to.
        select_seeds, folds, C: CV configuration guiding every decision.
        workers: process-pool size (defaults to min(16, cpu-2)).
        log: optional callable(str) for progress messages.

    Returns:
        (best, trace) where ``best`` maps subset size k -> (select_cv, idx array)
        for the best size-k subset SFBS found, and ``trace`` is the ordered list
        of ``(op, k, select_cv)`` decisions ('-' backward, '+' floating add).
    """
    universe = np.asarray(universe)
    seeds = list(select_seeds)
    workers = workers or min(16, (os.cpu_count() or 4) - 2)
    uni_set = set(int(v) for v in universe)

    def _log(m):
        if log is not None:
            log(m)

    cur = list(int(v) for v in universe)  # current subset (feature ids)
    best: dict[int, tuple[float, np.ndarray]] = {}
    trace: list[tuple[str, int, float]] = []

    with ProcessPoolExecutor(max_workers=workers,
                             initializer=_init, initargs=(x, y)) as ex:
        cv0 = mean_cv(x, y, np.asarray(cur), seeds, folds, C)
        best[len(cur)] = (cv0, np.asarray(cur))
        trace.append(('0', len(cur), cv0))
        _log(f"start k={len(cur)} selCV={cv0:.4f}")

        while len(cur) > target_k:
            # --- backward step: drop the feature whose removal maximises CV ---
            args = [(f, [c for c in cur if c != f], seeds, folds, C) for f in cur]
            res = dict(ex.map(_subset_score, args))
            drop_f = max(res, key=res.get)
            cur = [c for c in cur if c != drop_f]
            k = len(cur)
            cv = res[drop_f]
            if k not in best or cv > best[k][0]:
                best[k] = (cv, np.asarray(cur))
            trace.append(('-', k, cv))
            _log(f"- drop {drop_f} -> k={k} selCV={cv:.4f}")

            # --- conditional floating: add back excluded features while it helps ---
            while len(cur) < len(universe):
                excluded = [f for f in uni_set if f not in cur]
                args = [(f, cur + [f], seeds, folds, C) for f in excluded]
                res = dict(ex.map(_subset_score, args))
                add_f = max(res, key=res.get)
                add_cv = res[add_f]
                k1 = len(cur) + 1
                # only re-add if it strictly beats the best subset of that size
                # (guarantees progress / prevents cycling)
                if k1 in best and add_cv <= best[k1][0] + 1e-9:
                    break
                cur = cur + [add_f]
                best[k1] = (add_cv, np.asarray(cur))
                trace.append(('+', k1, add_cv))
                _log(f"+ add  {add_f} -> k={k1} selCV={add_cv:.4f}")

    return best, trace
