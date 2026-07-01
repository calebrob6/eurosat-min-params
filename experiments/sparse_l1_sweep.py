#!/usr/bin/env python
"""Element-wise weight sparsity via L1, as a parameter-reduction lever.

Every prior submission selects whole *features* (columns of the 10xF weight
matrix): dropping a feature zeros all 10 class-weights for it at once.  A feature
useful for only a few classes still pays for 10 weights.  This probe instead lets
individual ``W[c, f]`` go to zero independently, via a per-class (one-vs-rest) L1
logistic regression, and counts the honest deployed params as
``nnz(W_folded) + nnz(b) = nnz(W) + 10``.

Convention/honesty: prior submissions count ``10*(k+1)`` and treat *which* k
features are used as structural (free).  By the same convention a sparse W's
*pattern* is structural and only the non-zero *values* are parameters -- so
``nnz(W)+10`` is the consistent count (the standard neural-network pruning
convention).  Folding the StandardScaler into the weights preserves zeros.

Result (see experiments/sparse_l1_sweep_result.txt): OvR-L1 needs ~553 non-zero
weights (563 params) to reach val>=0.940 -- more than 2x the dense-backward k=25
frontier (260 params).  Element-wise sparsity is far *less* parameter-efficient
than column (feature) selection here; the masked-multinomial retrain
(experiments/masked_imp_cv.py) confirms the same verdict honestly on CV.

Uses liblinear (fast, per-class) rather than saga multinomial L1, which is
impractically slow on this shared machine (>9 min for a single full-305 fit).
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import time
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

BASE = os.path.join(os.path.dirname(__file__), '..', 'data', 'cache')


def load(split):
    return (np.load(os.path.join(BASE, f'{split}_feat_o6.npy')),
            np.load(os.path.join(BASE, f'{split}_y.npy')))


def coefs(clf):
    return np.vstack([e.coef_.ravel() for e in clf.estimators_])


def main():
    grid = os.environ.get('C_GRID')
    Cs = [float(c) for c in grid.split(',')] if grid else \
        [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.4]

    Xtr, ytr = load('train')
    Xva, yva = load('val')
    Xte, yte = load('test')
    sc = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    print('reference: sub09 dense-backward k=25 = 260 params, val 0.9404 '
          'test 0.9443\n')
    print(f'{"C":>7} {"nnzW":>5} {"params":>6} {"feat":>4} '
          f'{"val":>7} {"test":>7}  t')
    for C in Cs:
        t = time.time()
        base = LogisticRegression(penalty='l1', solver='liblinear', C=C,
                                  max_iter=3000, random_state=0)
        clf = OneVsRestClassifier(base, n_jobs=1).fit(Ztr, ytr)
        W = coefs(clf)
        intr = np.array([e.intercept_[0] for e in clf.estimators_])
        nnz = int((W != 0).sum())
        nf = int((np.abs(W).sum(0) > 0).sum())

        def pred(Z):
            return np.argmax(Z @ W.T + intr, 1)

        va = accuracy_score(yva, pred(Zva))
        te = accuracy_score(yte, pred(Zte))
        print(f'{C:>7} {nnz:>5} {nnz + 10:>6} {nf:>4} '
              f'{va:>7.4f} {te:>7.4f}  {time.time() - t:.0f}s', flush=True)


if __name__ == '__main__':
    main()
