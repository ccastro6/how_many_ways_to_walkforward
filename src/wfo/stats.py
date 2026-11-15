"""Inference tools for dependent data and multiple testing."""
from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy import stats as st

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------- HAC t-stats
def nw_lag(n: int) -> int:
    """Newey-West (1994) rule-of-thumb bandwidth floor(4 (n/100)^(2/9))."""
    return int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def nw_se(x: np.ndarray, lag: int | None = None) -> np.ndarray:
    """HAC standard error of the mean of each row of ``x`` (K, n), Bartlett kernel."""
    x = np.atleast_2d(x)
    n = x.shape[1]
    L = nw_lag(n) if lag is None else lag
    e = x - x.mean(axis=1, keepdims=True)
    lrv = (e * e).sum(axis=1) / n
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)
        lrv += 2.0 * w * (e[:, k:] * e[:, :-k]).sum(axis=1) / n
    return np.sqrt(np.maximum(lrv, 1e-300) / n)


def nw_tstat(x: np.ndarray, lag: int | None = None) -> np.ndarray:
    x = np.atleast_2d(x)
    return x.mean(axis=1) / nw_se(x, lag)


# ---------------------------------------------------------- stationary bootstrap
def stationary_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano (1994) stationary-bootstrap indices with circular wrapping."""
    p = 1.0 / mean_block
    new = rng.random(n) < p
    new[0] = True
    starts = rng.integers(n, size=n)
    pos = np.arange(n)
    first = np.maximum.accumulate(np.where(new, pos, 0))   # index where the current block began
    return (starts[first] + (pos - first)) % n


def block_length(x: np.ndarray) -> float:
    """Politis-White (2004) / Patton-Politis-White (2009) block length for the stationary bootstrap."""
    from arch.bootstrap import optimal_block_length
    b = float(optimal_block_length(np.asarray(x)).iloc[0]["stationary"])
    return float(min(max(b, 1.0), len(x) / 10))


def boot_stats(d: np.ndarray, se: np.ndarray, idx: np.ndarray):
    """Recentred bootstrap t-statistics for each row given one index draw.

    Returns (t_star per row, max over rows, mean over rows).
    """
    db = d[:, idx]
    t_star = (db.mean(axis=1) - d.mean(axis=1)) / se
    return t_star, t_star.max(), t_star.mean()


# ------------------------------------------------------------- Romano-Wolf
def stepm_pvalues(d: np.ndarray, B: int = 999, mean_block: float | None = None,
                  rng: np.random.Generator | None = None) -> dict:
    """Romano-Wolf (2005) step-down max-t adjusted p-values (one-sided, H0: mean <= 0).

    Also returns the White (2000) Reality-Check-style p-value of the maximum and the
    bootstrap p-value of the average t-statistic across rows.
    """
    rng = np.random.default_rng() if rng is None else rng
    K, n = d.shape
    se = nw_se(d)
    t = d.mean(axis=1) / se
    b = block_length(d.mean(axis=0)) if mean_block is None else mean_block
    T_star = np.empty((B, K))
    for i in range(B):
        T_star[i] = boot_stats(d, se, stationary_indices(n, b, rng))[0]
    order = np.argsort(-t)
    adj = np.empty(K)
    running = 0.0
    for r, k in enumerate(order):
        remaining = order[r:]
        pv = (np.sum(T_star[:, remaining].max(axis=1) >= t[k]) + 1) / (B + 1)
        running = max(running, pv)
        adj[k] = running
    p_max = (np.sum(T_star.max(axis=1) >= t.max()) + 1) / (B + 1)
    p_avg = (np.sum(T_star.mean(axis=1) >= t.mean()) + 1) / (B + 1)
    p_raw = 1 - st.norm.cdf(t)
    return dict(t=t, p_raw=p_raw, p_stepm=adj, p_max=p_max, p_avg=p_avg, block=b)


# ------------------------------------------------ dependence between designs
def effective_counts(lam: np.ndarray) -> dict:
    """Effective numbers of series from the eigenvalues ``lam`` of a K x K correlation matrix.

    n_eff         participation ratio of the spectrum, (sum lambda)^2 / sum lambda^2
    n_eff_liji    Li and Ji (2005): sum over eigenvalues of 1{lambda >= 1} + (lambda - floor(lambda))
    n_eff_galwey  Galwey (2009): (sum sqrt(lambda))^2 / sum lambda
    n_eff_nyholt  Nyholt (2004): 1 + (K - 1) (1 - Var(lambda) / K)
    Every count equals K for uncorrelated series and 1 for perfectly correlated ones.
    """
    lam = np.clip(np.asarray(lam, dtype=float), 0.0, None)
    K = lam.shape[0]
    if K < 2:
        return dict(n_eff=1.0, n_eff_liji=1.0, n_eff_galwey=1.0, n_eff_nyholt=1.0)
    lam_r = np.round(lam, 10)                   # so that an eigenvalue of exactly 1 counts once
    return dict(n_eff=float(lam.sum() ** 2 / np.sum(lam ** 2)),
                n_eff_liji=float(np.sum((lam_r >= 1.0) + (lam_r - np.floor(lam_r)))),
                n_eff_galwey=float(np.sqrt(lam).sum() ** 2 / lam.sum()),
                n_eff_nyholt=float(1.0 + (K - 1.0) * (1.0 - np.var(lam, ddof=1) / K)))


def effective_number(d: np.ndarray) -> dict:
    """Dependence summary of K series (rows of ``d``): mean pairwise correlation and the
    effective numbers of series of :func:`effective_counts`. ``n_eff`` is the participation
    ratio of the correlation matrix's spectrum, (sum lambda)^2 / sum lambda^2."""
    d = np.atleast_2d(d)
    K = d.shape[0]
    if K < 2:
        return dict(mean_corr=1.0, **effective_counts(np.ones(1)))
    sd = d.std(axis=1)
    keep = sd > 0
    C = np.corrcoef(d[keep]) if keep.sum() > 1 else np.ones((1, 1))
    C = np.nan_to_num(C)
    iu = np.triu_indices(C.shape[0], 1)
    lam = np.linalg.eigvalsh(C)
    return dict(mean_corr=float(C[iu].mean()) if len(iu[0]) else 1.0, **effective_counts(lam))


def independent_tests(fpr: float, alpha: float = 0.05) -> float:
    """Number of independent tests at level ``alpha`` whose maximum rejects with probability ``fpr``."""
    if fpr >= 1.0:                    # every replication rejects (a strong alternative): no finite count
        return float("inf")
    return float(math.log(1.0 - fpr) / math.log(1.0 - alpha))


# ---------------------------------------------------- Deflated Sharpe ratio
def deflated_sharpe(x: np.ndarray, sr_trials: np.ndarray, n_trials: float | None = None) -> float:
    """Bailey and Lopez de Prado (2014) DSR of return series ``x`` given the
    per-period Sharpe ratios of all trials (``sr_trials``). ``n_trials`` is the number of
    independent trials N (default: the number of trials supplied). Returns a probability."""
    x = np.asarray(x)
    n = x.shape[0]
    sr = x.mean() / x.std(ddof=1)
    g3 = st.skew(x)
    g4 = st.kurtosis(x, fisher=False)
    N = len(sr_trials) if n_trials is None else float(n_trials)
    v = np.var(sr_trials, ddof=1) if len(sr_trials) > 1 else 0.0
    if N > 1:
        sr0 = math.sqrt(v) * ((1 - EULER_GAMMA) * st.norm.ppf(1 - 1 / N)
                              + EULER_GAMMA * st.norm.ppf(1 - 1 / (N * math.e)))
    else:
        sr0 = 0.0
    den = math.sqrt(max(1 - g3 * sr + (g4 - 1) / 4 * sr * sr, 1e-12))
    return float(st.norm.cdf((sr - sr0) * math.sqrt(n - 1) / den))


# ------------------------------------------------------------------- PBO
def pbo_cscv(M: np.ndarray, S: int = 16) -> dict:
    """Probability of backtest overfitting via CSCV (Bailey et al. 2017).

    ``M`` is (n, K): n time periods, K candidate configurations. The sample is cut
    into S blocks; for every half/half split the best in-sample configuration's
    out-of-sample rank is recorded.
    """
    n, K = M.shape
    S = S - (S % 2)
    edges = np.linspace(0, n, S + 1).astype(int)
    parts = [np.arange(edges[i], edges[i + 1]) for i in range(S)]
    logits = []
    for comb in combinations(range(S), S // 2):
        is_idx = np.concatenate([parts[i] for i in comb])
        oos_idx = np.concatenate([parts[i] for i in range(S) if i not in comb])
        sr_is = M[is_idx].mean(0) / M[is_idx].std(0)
        sr_oos = M[oos_idx].mean(0) / M[oos_idx].std(0)
        best = int(np.argmax(sr_is))
        rank = (st.rankdata(sr_oos)[best]) / (K + 1)
        logits.append(math.log(rank / (1 - rank)))
    logits = np.array(logits)
    return dict(pbo=float(np.mean(logits <= 0)), logits=logits)
