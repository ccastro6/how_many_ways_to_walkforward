"""Walk-forward split generator.

A single parameterised family covers the rolling, growing-rolling and
expanding schemes used in walk-forward practice, with two requirements:

1. Test blocks are contiguous, non-overlapping and tile the evaluation span
   exactly, so every out-of-sample day is scored once and only once.
2. All configurations share the same evaluation span [start, end), so their
   out-of-sample records are directly comparable.

Notation (all in trading days)
------------------------------
start : first out-of-sample day (common to every configuration)
end   : one past the last day of data
h     : test-block length (the "step"); the last block may be shorter
L0    : training length for the first block
G     : growth of the training window per block (0 = rolling,
        0 < G < h = growing-rolling, G = h = expanding from the first block)
g     : gap ("embargo") between the end of training and the start of testing
origin: earliest day any training window may use (data burn-in)

For block i = 0, 1, ...:
    test_i  = [start + i*h, min(start + (i+1)*h, end))
    train_i = [max(origin, test_start - g - (L0 + G*i)), test_start - g)
An expanding scheme is obtained with ``expanding=True``, which fixes the
training start at ``origin``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterator


@dataclass(frozen=True)
class WFConfig:
    """One walk-forward design."""

    h: int                 # test-block length
    L0: int | None         # initial training length (None for expanding)
    G: int = 0             # growth per block
    g: int = 0             # gap between training and test
    expanding: bool = False

    @property
    def scheme(self) -> str:
        if self.expanding:
            return "expanding"
        return "rolling" if self.G == 0 else "growing"

    @property
    def label(self) -> str:
        if self.expanding:
            train = "EXP"
        else:
            train = f"R{self.L0}" if self.G == 0 else f"G{self.L0}+{self.G}"
        return f"{train}|h{self.h}|g{self.g}"

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update(scheme=self.scheme, label=self.label)
        return d


@dataclass(frozen=True)
class Block:
    train_start: int
    train_end: int   # exclusive
    test_start: int
    test_end: int    # exclusive


def blocks(cfg: WFConfig, start: int, end: int, origin: int = 0) -> Iterator[Block]:
    """Yield the walk-forward blocks of ``cfg`` over the span [start, end)."""
    if cfg.h <= 0:
        raise ValueError("h must be positive")
    if cfg.g < 0 or cfg.G < 0:
        raise ValueError("g and G must be non-negative")
    if not cfg.expanding and (cfg.L0 is None or cfg.L0 <= 0):
        raise ValueError("rolling/growing schemes need L0 > 0")
    i = 0
    t = start
    while t < end:
        test_end = min(t + cfg.h, end)
        train_end = t - cfg.g
        if cfg.expanding:
            train_start = origin
        else:
            train_start = max(origin, train_end - (cfg.L0 + cfg.G * i))
        if train_end - train_start < 2:
            raise ValueError(f"training window too short in block {i} of {cfg.label}")
        yield Block(train_start, train_end, t, test_end)
        t = test_end
        i += 1


def default_grid(h_values=(21, 63, 126, 252), L_values=(252, 504, 1008), g_values=(0, 21)):
    """The 56-configuration design space used in the paper.

    For each test length h and gap g: three rolling windows, three
    growing-rolling windows (growth G = h/2 per block) and one expanding window.
    """
    grid = []
    for h in h_values:
        for g in g_values:
            for L in L_values:
                grid.append(WFConfig(h=h, L0=L, G=0, g=g))
            for L in L_values:
                grid.append(WFConfig(h=h, L0=L, G=h // 2, g=g))
            grid.append(WFConfig(h=h, L0=None, g=g, expanding=True))
    return grid
