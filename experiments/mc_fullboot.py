"""Check the warp-speed shortcut against a full bootstrap.

For a smaller number of replications, every replication gets B bootstrap draws
and its own critical values for the max-t and average-t tests over all 56 designs,
against buy-and-hold and against cash.

Usage: python experiments/mc_fullboot.py --dgp garch_t --reps 400 --B 299
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from mc_run import GRID, START, END, T_DAYS, load_garch  # noqa: E402
from wfo.rules import rule_returns, BURN_IN  # noqa: E402
from wfo.backtest import run_grid  # noqa: E402
from wfo.dgp import simulate, prices  # noqa: E402
from wfo.stats import nw_se, stationary_indices, block_length  # noqa: E402

B = 299


def one_rep(args):
    dgp, kw, seed_seq, garch, B = args
    rng = np.random.default_rng(seed_seq)
    r = simulate(dgp, T_DAYS, rng, garch=garch, **kw)
    R, pos = rule_returns(prices(r), r)
    oos, _, _, _ = run_grid(R, pos, GRID, START, END, BURN_IN)
    d = oos - r[START:END][None, :]
    se = nw_se(d)
    t = d.mean(1) / se
    b = block_length(d.mean(0))
    m = d.mean(1)
    mx, av = np.empty(B), np.empty(B)
    for i in range(B):
        ts = (d[:, stationary_indices(d.shape[1], b, rng)].mean(1) - m) / se
        mx[i], av[i] = ts.max(), ts.mean()
    # the same two tests with cash as the benchmark; drawn after the buy-and-hold resamples,
    # so that those are the same with and without this block
    se_c = nw_se(oos)
    tc = oos.mean(1) / se_c
    b_c = block_length(oos.mean(0))
    m_c = oos.mean(1)
    mxc, avc = np.empty(B), np.empty(B)
    for i in range(B):
        ts = (oos[:, stationary_indices(oos.shape[1], b_c, rng)].mean(1) - m_c) / se_c
        mxc[i], avc[i] = ts.max(), ts.mean()
    return dict(max_t=float(t.max()), avg_t=float(t.mean()),
                rej_max=bool(t.max() > np.quantile(mx, 0.95)),
                rej_avg=bool(t.mean() > np.quantile(av, 0.95)),
                p_max=float((np.sum(mx >= t.max()) + 1) / (B + 1)),
                p_avg=float((np.sum(av >= t.mean()) + 1) / (B + 1)),
                rej_max_cash=bool(tc.max() > np.quantile(mxc, 0.95)),
                rej_avg_cash=bool(tc.mean() > np.quantile(avc, 0.95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", default="garch_t")
    ap.add_argument("--mu-annual", type=float, default=0.0)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--B", type=int, default=B)
    ap.add_argument("--seed", type=int, default=777)
    a = ap.parse_args()
    garch, gsrc = load_garch()
    kw, param = {}, 0.0
    if a.dgp == "regime":
        kw, param = dict(mu=a.mu_annual / 252.0), a.mu_annual
    elif a.dgp == "ar1":
        kw, param = dict(phi=a.phi), a.phi
    seeds = np.random.SeedSequence(a.seed).spawn(a.reps)
    with Pool(os.cpu_count()) as pool:
        out = pool.map(one_rep, [(a.dgp, kw, s, garch, a.B) for s in seeds], chunksize=2)
    res = dict(dgp=a.dgp, param=param, reps=a.reps, B=a.B, garch_source=gsrc,
               reject_max=float(np.mean([o["rej_max"] for o in out])),
               reject_avg=float(np.mean([o["rej_avg"] for o in out])),
               reject_max_cash=float(np.mean([o["rej_max_cash"] for o in out])),
               reject_avg_cash=float(np.mean([o["rej_avg_cash"] for o in out])))
    od = ROOT / "results" / "mc"
    od.mkdir(parents=True, exist_ok=True)
    (od / f"fullboot_{a.dgp}_p{param:g}_s{a.seed}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
