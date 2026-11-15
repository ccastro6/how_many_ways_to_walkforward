"""Walk-forward selection: in each training window pick the rule with the best
in-sample Sharpe ratio, then hold that rule through the next test block."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .splits import WFConfig, blocks

#: Sharpe ratios are rounded to this many decimals before the arg-max, so that rules
#: with identical returns over a window tie exactly (floating-point noise in the
#: cumulative sums is of order 1e-13) and the tie is broken by rule order.
TIE_DECIMALS = 10


@dataclass
class WFResult:
    oos: np.ndarray        # (n_oos,) out-of-sample strategy returns
    held: np.ndarray       # (n_oos,) position held on each out-of-sample day
    is_mean: float         # average in-sample daily return of the selected rules
    oos_mean: float        # average out-of-sample daily return
    n_blocks: int
    chosen: np.ndarray     # index of the rule chosen in each block
    n_tied: int            # number of training windows in which the best Sharpe ratio was tied


class WindowStats:
    """O(1) mean and Sharpe ratio of every rule over any window, via cumulative sums.

    The sums are taken on returns centred at each rule's full-sample mean (the shifted-data
    form of the one-pass variance), so the subtraction in the variance loses no precision.
    """

    def __init__(self, R: np.ndarray):
        self.R = R
        self.c = R.mean(axis=0)
        Rc = R - self.c
        z = np.zeros((1, R.shape[1]))
        self.S1 = np.vstack([z, np.cumsum(Rc, axis=0)])
        self.S2 = np.vstack([z, np.cumsum(Rc * Rc, axis=0)])

    def mean_sharpe(self, a: int, b: int):
        n = b - a
        mc = (self.S1[b] - self.S1[a]) / n          # mean of the centred returns
        v = (self.S2[b] - self.S2[a]) / n - mc * mc
        m = mc + self.c
        sd = np.sqrt(np.maximum(v, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = np.where(sd > 1e-12, m / sd, 0.0)
        return m, sh


def select(sh: np.ndarray) -> tuple[int, bool]:
    """Index of the best Sharpe ratio (first in rule order among ties) and whether it was tied."""
    r = np.round(sh, TIE_DECIMALS)
    j = int(np.argmax(r))
    return j, bool(np.sum(r == r[j]) > 1)


def walk_forward(ws: WindowStats, pos: np.ndarray, cfg: WFConfig,
                 start: int, end: int, origin: int) -> WFResult:
    R = ws.R
    oos, held, chosen = [], [], []
    is_means = []
    n = 0
    n_tied = 0
    for blk in blocks(cfg, start, end, origin):
        m, sh = ws.mean_sharpe(blk.train_start, blk.train_end)
        j, tied = select(sh)
        n_tied += tied
        chosen.append(j)
        is_means.append(m[j])
        oos.append(R[blk.test_start:blk.test_end, j])
        held.append(pos[blk.test_start - 1:blk.test_end - 1, j])
        n += 1
    oos = np.concatenate(oos)
    return WFResult(oos=oos, held=np.concatenate(held), is_mean=float(np.mean(is_means)),
                    oos_mean=float(oos.mean()), n_blocks=n, chosen=np.array(chosen), n_tied=n_tied)


def run_grid(R: np.ndarray, pos: np.ndarray, grid: list[WFConfig],
             start: int, end: int, origin: int):
    """Run every configuration; returns (K, n_oos) OOS returns, held positions and IS means."""
    ws = WindowStats(R)
    res = [walk_forward(ws, pos, c, start, end, origin) for c in grid]
    return (np.vstack([r.oos for r in res]), np.vstack([r.held for r in res]),
            np.array([r.is_mean for r in res]), res)


def turnover(held: np.ndarray, first_prev: np.ndarray | None = None) -> np.ndarray:
    """Absolute position change per day for each row of ``held`` (K, n)."""
    prev = np.zeros((held.shape[0], 1)) if first_prev is None else first_prev[:, None]
    return np.abs(np.diff(np.hstack([prev, held]), axis=1))
