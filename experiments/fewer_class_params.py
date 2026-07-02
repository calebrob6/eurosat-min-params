#!/usr/bin/env python
"""Cut the 10x per-class multiplier in the linear head.

A K-class softmax is shift-invariant: adding a constant to every class logit
leaves argmax (and probabilities) unchanged. So one class is a free REFERENCE
(fixed 0 logit) and a 10-class linear head needs only (K-1)*(F+1) = 9*(F+1)
stored parameters, NOT 10*(F+1) -- with IDENTICAL predictions. Every submission
so far over-counted the head by exactly (F+1).

Then probe going BELOW 9 output dims via error-correcting output codes (ECOC):
b binary base-learners + a FIXED (free, seeded) code matrix -> a rank-b head with
params b*(F+1). iter-3 found all 9 softmax discriminant dims are needed, so we
expect accuracy to fall off below b=9 -- this quantifies the cliff.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')

import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402

SUB11 = [58, 65, 68, 75, 86, 92, 93, 145, 203, 273, 275, 287, 291, 294, 297, 298, 310, 316]


def pool(split):
    parts = [np.load(os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')),
             np.load(os.path.join(CACHE_DIR, f'{split}_linefam_line.npy')),
             np.load(os.path.join(CACHE_DIR, f'{split}_gs2fam_corn2.npy'))]
    return np.concatenate(parts, 1).astype(np.float32)


def main():
    idx = np.array(SUB11)
    ftr, fva, fte = pool('train')[:, idx], pool('val')[:, idx], pool('test')[:, idx]
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    F = ftr.shape[1]
    print(f'submission-11 features: F={F}')

    # ---- baseline: full 10-class folded head (current convention) ----------
    w, b, fi = fit_folded_logreg(ftr, ytr, np.arange(F), C=10.0)
    base_va = accuracy_score(yva, predict(fva, w, b, fi))
    base_te = accuracy_score(yte, predict(fte, w, b, fi))
    print(f'\n[full 10-class]     params={10*(F+1):>4}  val={base_va:.4f} test={base_te:.4f}')

    # ---- reference-class reparameterization (LOSSLESS) ---------------------
    # subtract a reference class's logits from all -> ref row becomes 0, drop it.
    r = 0
    Wr = np.delete(w - w[r], r, axis=0)          # (9, F)
    br = np.delete(b - b[r], r)                   # (9,)
    def predict_ref(x):
        l9 = x @ Wr.T + br                        # 9 non-ref logits
        full = np.concatenate([l9[:, :r], np.zeros((len(x), 1)), l9[:, r:]], 1)
        return full.argmax(1)
    rva = accuracy_score(yva, predict_ref(fva))
    rte = accuracy_score(yte, predict_ref(fte))
    same = np.array_equal(predict_ref(fte), predict(fte, w, b, fi))
    print(f'[reference-class 9] params={9*(F+1):>4}  val={rva:.4f} test={rte:.4f}  '
          f'identical_to_full={same}')

    # ---- ECOC: b binary learners + FIXED random code (rank-b, params b*(F+1))
    sc = StandardScaler().fit(ftr)
    Ztr, Zva, Zte = sc.transform(ftr), sc.transform(fva), sc.transform(fte)
    print('\nECOC (fixed seeded code, best of 40 seeds on val):')
    print(f'{"b":>2} {"params":>6} {"val":>7} {"test":>7}')
    rng_seeds = range(40)
    for bbits in (4, 5, 6, 7, 8, 9, 10):
        best = None
        for s in rng_seeds:
            rs = np.random.RandomState(s)
            # random +/-1 code with distinct, non-constant columns
            M = rs.choice([-1, 1], size=(10, bbits))
            if len(set(map(tuple, M))) < 10 or any(len(set(M[:, j])) == 1 for j in range(bbits)):
                continue
            D = np.zeros((3, len(Ztr) if False else 0))  # placeholder
            # train one binary logreg per bit, collect decision functions
            dv = {'va': np.zeros((len(Zva), bbits)), 'te': np.zeros((len(Zte), bbits))}
            ok = True
            for j in range(bbits):
                tgt = M[ytr, j]
                if len(set(tgt)) < 2:
                    ok = False; break
                clf = LogisticRegression(max_iter=2000, C=10.0).fit(Ztr, tgt)
                dv['va'][:, j] = clf.decision_function(Zva)
                dv['te'][:, j] = clf.decision_function(Zte)
            if not ok:
                continue
            pv = (dv['va'] @ M.T).argmax(1)      # loss-based (linear) decoding
            va = accuracy_score(yva, pv)
            if best is None or va > best[0]:
                pt = (dv['te'] @ M.T).argmax(1)
                best = (va, accuracy_score(yte, pt))
        if best:
            print(f'{bbits:>2} {bbits*(F+1):>6} {best[0]:>7.4f} {best[1]:>7.4f}'
                  f'{"  >=94" if best[1] >= .94 else ""}')


if __name__ == '__main__':
    main()
