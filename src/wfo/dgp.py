"""Data-generating processes for the Monte Carlo study.

Null models have no predictable mean, so no trading rule has genuine skill:
    iid_t    : i.i.d. Student-t returns, zero mean
    garch_t  : zero-mean GARCH(1,1) with standardised Student-t shocks
Alternatives plant a known, modest amount of predictability:
    ar1      : r_t = phi * r_{t-1} + e_t, e_t GARCH-t   (short-horizon momentum)
    regime   : r_t = s_t * mu + e_t, s_t a persistent +/-1 Markov chain (trends)

Returns are simple returns. A fat-tailed, highly persistent GARCH process can produce a shock
below -100%, which no price can follow, so every shock is truncated at +/- ``MAX_ABS_SHOCK``.
The truncation is symmetric, which keeps the zero mean, and it enters the variance recursion,
so the simulated path is the one an observer would have seen. It binds on a handful of days
in a few simulated markets (the count is stored with the Monte Carlo output).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GarchParams:
    omega: float
    alpha: float
    beta: float
    nu: float

    @property
    def uncond_var(self) -> float:
        return self.omega / (1.0 - self.alpha - self.beta)


#: largest absolute daily shock; keeps simulated prices positive (see the module docstring)
MAX_ABS_SHOCK = 0.5

#: default calibration; replaced by the US market fit in experiments/calibrate_garch.py
DEFAULT_GARCH = GarchParams(omega=1.6e-6, alpha=0.10, beta=0.885, nu=6.0)


def std_t(rng: np.random.Generator, nu: float, n: int) -> np.ndarray:
    """Student-t draws scaled to unit variance."""
    return rng.standard_t(nu, size=n) * np.sqrt((nu - 2.0) / nu)


def garch_noise(rng: np.random.Generator, n: int, p: GarchParams, burn: int = 500) -> np.ndarray:
    z = std_t(rng, p.nu, n + burn)
    e = np.empty(n + burn)
    h = p.uncond_var
    for t in range(n + burn):
        e[t] = min(max(np.sqrt(h) * z[t], -MAX_ABS_SHOCK), MAX_ABS_SHOCK)
        h = p.omega + p.alpha * e[t] ** 2 + p.beta * h
    return e[burn:]


def simulate(model: str, n: int, rng: np.random.Generator, garch: GarchParams = DEFAULT_GARCH,
             phi: float = 0.0, mu: float = 0.0, duration: float = 126.0, drift: float = 0.0) -> np.ndarray:
    """Simulate ``n`` daily simple returns from ``model``.

    ``drift`` is a constant daily premium added to the GARCH null (``garch_t`` only); it is zero
    in the paper's nulls and positive in the equity-premium check."""
    if model == "iid_t":
        return np.clip(std_t(rng, garch.nu, n) * np.sqrt(garch.uncond_var), -MAX_ABS_SHOCK, MAX_ABS_SHOCK)
    e = garch_noise(rng, n, garch)
    if model == "garch_t":
        return e + drift
    if model == "ar1":
        r = np.empty(n)
        r[0] = e[0]
        for t in range(1, n):
            r[t] = phi * r[t - 1] + e[t]
        return r
    if model == "regime":
        stay = 1.0 - 1.0 / duration
        flips = rng.random(n) > stay
        s = np.where(np.cumsum(flips) % 2 == 0, 1.0, -1.0) * (1 if rng.random() < 0.5 else -1)
        return s * mu + e
    raise ValueError(model)


def prices(ret: np.ndarray) -> np.ndarray:
    return 100.0 * np.cumprod(1.0 + ret)
