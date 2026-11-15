"""Fast probability of backtest overfitting (CSCV) using per-block sufficient statistics.

Numerically identical to :func:`wfo.stats.pbo_cscv` but sums block-level first and
second moments instead of re-indexing the full return matrix for every split, which
makes S = 16 (12,870 splits) cheap for long daily samples.
"""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats as st


def pbo_cscv_fast(M: np.ndarray, S: int = 16) -> dict:
    n, K = M.shape
    S = S - (S % 2)
    edges = np.linspace(0, n, S + 1).astype(int)
    cnt = np.diff(edges).astype(float)
    s1 = np.array([M[edges[i]:edges[i + 1]].sum(0) for i in range(S)])
    s2 = np.array([(M[edges[i]:edges[i + 1]] ** 2).sum(0) for i in range(S)])
    tot1, tot2, totn = s1.sum(0), s2.sum(0), cnt.sum()

    def sharpe(a1, a2, m):
        mu = a1 / m
        var = a2 / m - mu * mu
        return mu / np.sqrt(np.maximum(var, 1e-300))

    logits, best_idx = [], []
    for comb in combinations(range(S), S // 2):
        c = list(comb)
        i1, i2, m = s1[c].sum(0), s2[c].sum(0), cnt[c].sum()
        sr_is = sharpe(i1, i2, m)
        sr_oos = sharpe(tot1 - i1, tot2 - i2, totn - m)
        b = int(np.argmax(sr_is))
        rank = st.rankdata(sr_oos)[b] / (K + 1)
        logits.append(math.log(rank / (1 - rank)))
        best_idx.append(b)
    logits = np.array(logits)
    return dict(pbo=float(np.mean(logits <= 0)), logits=logits, best=np.array(best_idx))
