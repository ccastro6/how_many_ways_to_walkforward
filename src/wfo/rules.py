"""Technical trading-rule universe: four of the five rule families of Sullivan,
Timmermann and White (1999), on closing prices.

    filter rules (Alexander 1961)              93 rules
    moving-average crossovers                 116 rules
    support and resistance (range breakout)    70 rules
    channel breakouts                          72 rules
                                              351 rules in total

The fifth family of Sullivan et al., on-balance volume, needs volume data, which the
index series used here do not have. Within each family the parameter grid is a subset of
theirs, chosen to span the same ranges with a few hundred rules rather than several
thousand. Positions are +1 (long), -1 (short) or 0 (flat).

Every position is causal: the position held over day t+1 is decided with prices up to
and including day t. ``BURN_IN`` is the first day on which every rule has a defined
signal (the longest look-back is 250 days).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

# ------------------------------------------------------------------ parameter grids
FILTER_X = (0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15)
FILTER_Y = (0.005, 0.01, 0.02)          # liquidation band, used only when y < x
FILTER_C = (5, 10, 25)                  # holding period in days
FILTER_E = (5, 10, 20)                  # window for the reference extremum

MA_FAST = (1, 2, 5, 10, 20)
MA_SLOW = (20, 50, 100, 150, 200, 250)
MA_BAND = (0.005, 0.01, 0.02)

SR_N = (5, 10, 15, 20, 25, 50, 100, 150, 200, 250)
SR_BAND = (0.005, 0.01, 0.02)
SR_C = (5, 10, 25)

CB_N = (5, 10, 20, 50, 100, 200)
CB_X = (0.02, 0.05, 0.10, 0.15)
CB_C = (5, 10, 25)

#: first day on which every rule has a defined signal
BURN_IN = max(max(MA_SLOW), max(SR_N), max(CB_N))


@dataclass(frozen=True)
class Rule:
    family: str      # "filter", "ma", "sr", "cb"
    name: str
    params: tuple


def _rules() -> list[Rule]:
    out = []
    for x in FILTER_X:
        out.append(Rule("filter", f"F({x:g})", (x, 0.0, 0, 0)))
    for x in FILTER_X:
        for y in FILTER_Y:
            if y < x:
                out.append(Rule("filter", f"F({x:g},y={y:g})", (x, y, 0, 0)))
    for x in FILTER_X:
        for c in FILTER_C:
            out.append(Rule("filter", f"F({x:g},c={c})", (x, 0.0, c, 0)))
    for x in FILTER_X:
        for e in FILTER_E:
            out.append(Rule("filter", f"F({x:g},e={e})", (x, 0.0, 0, e)))
    for n in MA_FAST:
        for m in MA_SLOW:
            if n < m:
                out.append(Rule("ma", f"MA({n},{m})", (n, m, 0.0)))
    for b in MA_BAND:
        for n in MA_FAST:
            for m in MA_SLOW:
                if n < m:
                    out.append(Rule("ma", f"MA({n},{m},b={b:g})", (n, m, b)))
    for n in SR_N:
        out.append(Rule("sr", f"SR({n})", (n, 0.0, 0)))
    for b in SR_BAND:
        for n in SR_N:
            out.append(Rule("sr", f"SR({n},b={b:g})", (n, b, 0)))
    for c in SR_C:
        for n in SR_N:
            out.append(Rule("sr", f"SR({n},c={c})", (n, 0.0, c)))
    for n in CB_N:
        for x in CB_X:
            for c in CB_C:
                out.append(Rule("cb", f"CB({n},{x:g},c={c})", (n, x, c)))
    return out


RULES: list[Rule] = _rules()
N_RULES = len(RULES)
FAMILY_NAMES = {"filter": "Filter", "ma": "Moving average", "sr": "Support and resistance",
                "cb": "Channel breakout"}


def rule_names() -> list[str]:
    return [r.name for r in RULES]


def family_counts() -> dict[str, int]:
    out: dict[str, int] = {}
    for r in RULES:
        out[r.family] = out.get(r.family, 0) + 1
    return out


# ------------------------------------------------------------------ kernels
@njit(cache=True)
def _filter(lp, x, y, c, e):
    """Filter rule on log prices.

    Buy when the price has risen x above the reference low, go short when it has
    fallen x below the reference high. Without options the position is reversed at
    each signal and the reference extremes run from the last signal. ``y`` (0 < y < x)
    liquidates a position after an adverse move of y, after which the next signal
    needs a full move of x. ``c`` holds each position for c days, ignoring signals,
    then goes flat. ``e`` takes the reference extremes over the previous e days.
    """
    T = lp.shape[0]
    pos = np.zeros(T)
    state = 0
    hi = lp[0]
    lo = lp[0]
    hold = 0
    for t in range(T):
        p = lp[t]
        if c > 0 and hold > 0:
            hold -= 1
            if hold == 0:
                state = 0
                hi = p
                lo = p
            pos[t] = state
            continue
        if e > 0:
            if t < e:
                pos[t] = 0
                continue
            hi = lp[t - e]
            lo = lp[t - e]
            for k in range(t - e + 1, t):
                if lp[k] > hi:
                    hi = lp[k]
                if lp[k] < lo:
                    lo = lp[k]
        entered = False
        if state == 1:
            if e == 0 and p > hi:
                hi = p
            if y > 0.0 and p <= hi - y:
                state = 0
                lo = p
            elif p <= hi - x:
                state = -1
                lo = p
                entered = True
        elif state == -1:
            if e == 0 and p < lo:
                lo = p
            if y > 0.0 and p >= lo + y:
                state = 0
                hi = p
            elif p >= lo + x:
                state = 1
                hi = p
                entered = True
        else:
            if e == 0:
                if p > hi:
                    hi = p
                if p < lo:
                    lo = p
            if p >= lo + x:
                state = 1
                hi = p
                entered = True
            elif p <= hi - x:
                state = -1
                lo = p
                entered = True
        if entered and c > 0:
            hold = c
        pos[t] = state
    return pos


@njit(cache=True)
def _hold(sig, c):
    """Take each non-zero signal when not already holding; hold it c days, then go flat."""
    T = sig.shape[0]
    pos = np.zeros(T)
    state = 0
    hold = 0
    for t in range(T):
        if hold > 0:
            hold -= 1
            if hold == 0:
                state = 0
        if hold == 0 and sig[t] != 0:
            state = int(sig[t])
            hold = c
        pos[t] = state
    return pos


@njit(cache=True)
def _ffill(sig):
    """Hold the last non-zero signal until the opposite one (flat before the first)."""
    T = sig.shape[0]
    pos = np.zeros(T)
    state = 0
    for t in range(T):
        if sig[t] != 0:
            state = int(sig[t])
        pos[t] = state
    return pos


def _sma(x: np.ndarray, w: int) -> np.ndarray:
    if w == 1:
        return x.astype(float).copy()
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    out = np.full(x.shape[0], np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def positions(price: np.ndarray) -> np.ndarray:
    """Return a (T, 351) array of positions in {-1, 0, +1} decided at each close.

    Days before a rule's signal is defined carry position 0.
    """
    price = np.asarray(price, dtype=float)
    T = price.shape[0]
    lp = np.log(price)
    s = pd.Series(price)
    prev = s.shift(1)
    hi_n = {n: prev.rolling(n, min_periods=n).max().to_numpy() for n in set(SR_N) | set(CB_N)}
    lo_n = {n: prev.rolling(n, min_periods=n).min().to_numpy() for n in set(SR_N) | set(CB_N)}
    smas = {w: _sma(price, w) for w in set(MA_FAST) | set(MA_SLOW)}
    cols = []
    for r in RULES:
        if r.family == "filter":
            x, y, c, e = r.params
            cols.append(_filter(lp, float(x), float(y), int(c), int(e)))
        elif r.family == "ma":
            n, m, b = r.params
            f, sl = smas[n], smas[m]
            with np.errstate(invalid="ignore"):
                p = np.where(f > sl * (1.0 + b), 1.0, np.where(f < sl * (1.0 - b), -1.0, 0.0))
            p[np.isnan(sl)] = 0.0
            cols.append(p)
        elif r.family == "sr":
            n, b, c = r.params
            hi, lo = hi_n[n], lo_n[n]
            with np.errstate(invalid="ignore"):
                sig = np.where(price > hi * (1.0 + b), 1.0, np.where(price < lo * (1.0 - b), -1.0, 0.0))
            sig[np.isnan(hi)] = 0.0
            cols.append(_hold(sig, int(c)) if c > 0 else _ffill(sig))
        else:  # channel breakout
            n, x, c = r.params
            hi, lo = hi_n[n], lo_n[n]
            with np.errstate(invalid="ignore"):
                channel = hi <= lo * (1.0 + x)
                sig = np.where(channel & (price > hi), 1.0, np.where(channel & (price < lo), -1.0, 0.0))
            sig[np.isnan(hi)] = 0.0
            cols.append(_hold(sig, int(c)))
    pos = np.column_stack(cols)
    assert pos.shape == (T, N_RULES)
    return pos


def rule_returns(price: np.ndarray, ret: np.ndarray):
    """Daily simple returns of each rule: position at close t-1 times return on day t.

    ``ret[t]`` is the simple return from close t-1 to close t (ret[0] unused).
    """
    pos = positions(price)
    R = np.zeros_like(pos)
    R[1:] = pos[:-1] * ret[1:, None]
    return R, pos
