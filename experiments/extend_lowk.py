#!/usr/bin/env python
"""Fast low-k extension of the pure backward-greedy descent (k=28 -> STOP).

Submissions 07/08 ran ``backward_select.py`` only down to STOP=28, so the honest
floor was *reported* as k=28 -- but k=27/26/25 were never measured.  Re-running
the full k=42 descent just to see those rows is wasteful: backward-greedy is
deterministic, so the k=42->28 path is identical to what the existing
``backward_select_*_result.txt`` files already contain.  This script instead
*resumes* the descent from the known k=28 subset and continues to STOP, computing
the honest checks (VERIFY-CV + val + test) at every new k.

Honesty protocol is unchanged from ``backward_select.py``:
  * SELECT seeds drive the greedy drop.
  * VERIFY seeds (disjoint) give an unbiased CV of each resulting subset.
  * val (held out) is the third independent check.  test drives no decision.
The honest floor is the smallest k with BOTH verCV>=0.940 AND val>=0.940.

Env: SEL0/VER0/NSEED (seed partition), STOP (default 24), START28 selects the
k=28 subset for that partition.  Writes ``extend_lowk_<tag>_result.txt``.
"""
from __future__ import annotations

import os

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.cache import CACHE_DIR  # noqa: E402
from src.linmodel import fit_folded_logreg, predict  # noqa: E402
from src.select import _init, _subset_score, mean_cv  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

# The deterministic k=28 pure-backward subsets, copied verbatim from the existing
# result files (identical method, one per seed partition):
#   partition 0  -> experiments/backward_select_result.txt
#   partition 20 -> experiments/backward_select_o6_cfm_result.txt
K28 = {
    0: [291, 94, 285, 145, 297, 181, 93, 92, 111, 294, 195, 275, 33, 287, 58,
        68, 196, 203, 188, 75, 86, 91, 95, 273, 300, 304, 65, 295],
    20: [291, 94, 285, 297, 181, 93, 144, 92, 111, 294, 195, 275, 278, 287, 58,
         68, 196, 203, 188, 75, 86, 91, 95, 273, 300, 304, 65, 295],
}

_SEL0 = int(os.environ.get('SEL0', '0'))
_VER0 = int(os.environ.get('VER0', '10'))
_NSEED = int(os.environ.get('NSEED', '10'))
SELECT_SEEDS = list(range(_SEL0, _SEL0 + _NSEED))
VERIFY_SEEDS = list(range(_VER0, _VER0 + _NSEED))
STOP = int(os.environ.get('STOP', '24'))
FOLDS = 5
# C defaults to 10.0 (the value the K28 subsets below were derived with); set the
# C env var to re-run the low-k drops under a different inverse-L2 strength while
# keeping the same verified k=28 starting subset (iteration-10 C-per-k probe).
C = float(os.environ.get('C', '10.0'))
THRESH = 0.940
WORKERS = int(os.environ.get('WORKERS', '18'))


def load_pool(split):
    return np.load(os.path.join(CACHE_DIR, f'{split}_feat_o6.npy')).astype(np.float32)


def main():
    ytr = np.load(os.path.join(CACHE_DIR, 'train_y.npy'))
    yva = np.load(os.path.join(CACHE_DIR, 'val_y.npy'))
    yte = np.load(os.path.join(CACHE_DIR, 'test_y.npy'))
    ftr, fva, fte = load_pool('train'), load_pool('val'), load_pool('test')

    cur = list(K28[_SEL0])
    assert len(cur) == 28, len(cur)

    _ctag = '' if C == 10.0 else f'_C{C:g}'
    outpath = os.path.join(os.path.dirname(__file__),
                           f'extend_lowk_s{_SEL0}{_ctag}_result.txt')
    out = open(outpath, 'w')

    def emit(msg):
        print(msg, flush=True)
        out.write(msg + '\n')
        out.flush()

    emit(f'EXTEND o6 dim={ftr.shape[1]} from k=28 to STOP={STOP} C={C:g} '
         f'SELECT={SELECT_SEEDS[0]}..{SELECT_SEEDS[-1]} '
         f'VERIFY={VERIFY_SEEDS[0]}..{VERIFY_SEEDS[-1]}')
    emit(f'{"k":>4} {"params":>7} {"selCV":>7} {"verCV":>7} {"val":>7} {"test":>7}'
         f'  dropped')

    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_init, initargs=(ftr, ytr)) as ex:
        while True:
            idx = np.asarray(cur)
            sel = mean_cv(ftr, ytr, idx, SELECT_SEEDS, FOLDS, C)
            ver = mean_cv(ftr, ytr, idx, VERIFY_SEEDS, FOLDS, C)
            w, b, fi = fit_folded_logreg(ftr, ytr, feature_idx=idx, C=C)
            va = accuracy_score(yva, predict(fva, w, b, fi))
            te = accuracy_score(yte, predict(fte, w, b, fi))
            rows.append((len(cur), 10 * (len(cur) + 1), sel, ver, va, te, list(cur)))
            emit(f'{len(cur):>4} {10*(len(cur)+1):>7} {sel:>7.4f} {ver:>7.4f} '
                 f'{va:>7.4f} {te:>7.4f}')
            if len(cur) <= STOP:
                break
            # drop the feature whose removal maximises SELECT-CV
            args = [(f, [c for c in cur if c != f], SELECT_SEEDS, FOLDS, C)
                    for f in cur]
            res = dict(ex.map(_subset_score, args))
            drop_f = max(res, key=res.get)
            cur = [c for c in cur if c != drop_f]
            out.write(f'    (dropped {drop_f})\n')
            out.flush()

    honest = [r for r in rows if r[3] >= THRESH and r[4] >= THRESH]
    emit('\n########## HONEST FLOOR (verCV>=0.940 AND val>=0.940) ##########')
    if honest:
        b0 = min(honest, key=lambda r: r[0])
        emit(f'k={b0[0]} params={b0[1]} selCV={b0[2]:.4f} verCV={b0[3]:.4f} '
             f'val={b0[4]:.4f} test={b0[5]:.4f}')
        emit('feature_idx=' + ','.join(map(str, b0[6])))
    else:
        emit('none cleared both verCV>=0.940 and val>=0.940')
    emit('ALL_DONE')
    out.close()


if __name__ == '__main__':
    main()
