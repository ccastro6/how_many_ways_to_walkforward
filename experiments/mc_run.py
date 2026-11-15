"""Monte Carlo driver: one DGP setting, many replications, all 56 walk-forward designs.

Each replication
  1. simulates T daily returns from the DGP,
  2. computes the 351 rule return series,
  3. runs walk-forward selection for every design over the common evaluation span,
  4. tests d_t = r_strategy - r_buy&hold with a Newey-West t-statistic (H0: mean <= 0),
  5. draws ONE joint stationary-bootstrap resample (warp-speed method of
     Giacomini, Politis and White 2013) and stores the recentred bootstrap t* of
     every design, so design-robust tests can be evaluated in aggregation,
  6. stores the classic full-sample "best of 351 rules" statistic as a baseline,
  7. stores the dependence between designs within the replication: the mean pairwise
     correlation of the 56 difference series d_k,t and four effective numbers of designs
     (participation ratio, Li-Ji, Galwey and Nyholt counts of the correlation matrix spectrum),
  8. repeats steps 4 and 5 with cash as the benchmark: the test is on the strategy's own
     excess return (H0: mean <= 0), with its own joint bootstrap draw.

``--dgp garch_premium`` is the GARCH null with a constant annual premium added (``--premium-annual``);
it is used once, to show what an equity premium does to the test against buy-and-hold.

Usage:
  python experiments/mc_run.py --dgp garch_t --reps 2000 --seed 1
  python experiments/mc_run.py --dgp regime --mu-annual 0.10 --reps 1000 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfo.splits import default_grid  # noqa: E402
from wfo.rules import rule_returns, BURN_IN, N_RULES  # noqa: E402
from wfo.backtest import run_grid  # noqa: E402
from wfo.stats import effective_number  # noqa: E402
from wfo.dgp import simulate, prices, GarchParams, DEFAULT_GARCH, MAX_ABS_SHOCK  # noqa: E402
from wfo.stats import nw_se, stationary_indices, block_length  # noqa: E402

T_DAYS = 5292                     # 21 years of 252 days
GRID = default_grid()
MAX_L = max(c.L0 or 0 for c in GRID)
MAX_G = max(c.g for c in GRID)
START = BURN_IN + MAX_L + MAX_G   # common first out-of-sample day
END = T_DAYS

CALIB = ROOT / "results" / "calibration" / "garch_us_market.json"


def load_garch() -> tuple[GarchParams, str]:
    if CALIB.exists():
        p = json.loads(CALIB.read_text())
        return GarchParams(p["omega"], p["alpha"], p["beta"], p["nu"]), "us_market_fit"
    return DEFAULT_GARCH, "default"


def one_rep(args):
    dgp, kw, seed_seq, garch = args
    rng = np.random.default_rng(seed_seq)
    r = simulate("garch_t" if dgp == "garch_premium" else dgp, T_DAYS, rng, garch=garch, **kw)
    R, pos = rule_returns(prices(r), r)
    oos, held, is_mean, res = run_grid(R, pos, GRID, START, END, BURN_IN)
    bh = r[START:END]
    d = oos - bh[None, :]
    se = nw_se(d)
    t = d.mean(1) / se
    # one joint bootstrap draw (same indices for every design and every rule)
    b = block_length(d.mean(0))
    idx = stationary_indices(d.shape[1], b, rng)
    t_star = (d[:, idx].mean(1) - d.mean(1)) / se
    # baseline: best of the 351 rules over the whole evaluation span (no walk-forward)
    dr = (R[START:END] - bh[:, None]).T
    se_r = nw_se(dr)
    t_r = dr.mean(1) / se_r
    t_r_star = (dr[:, idx].mean(1) - dr.mean(1)) / se_r
    sd = oos.std(1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sharpe_oos = np.where(sd > 0, oos.mean(1) / sd, 0.0)
    rows = dict(
        t=t, t_star=t_star, is_mean=is_mean, oos_mean=oos.mean(1),
        sharpe_oos=sharpe_oos,
        d_sharpe=d.mean(1) / d.std(1, ddof=1),
        d_skew=st.skew(d, axis=1), d_kurt=st.kurtosis(d, axis=1, fisher=False),
        n_tied=np.array([x.n_tied for x in res]), distinct_rules=np.array([len(np.unique(x.chosen)) for x in res]),
    )
    dep = effective_number(d)
    # the same test with cash as the benchmark: the strategy's own excess return, with its own
    # joint bootstrap draw (taken last, so the buy-and-hold statistics do not depend on it)
    se_c = nw_se(oos)
    b_c = block_length(oos.mean(0))
    idx_c = stationary_indices(oos.shape[1], b_c, rng)
    rows["t_cash"] = oos.mean(1) / se_c
    rows["t_cash_star"] = (oos[:, idx_c].mean(1) - oos.mean(1)) / se_c
    dep_c = effective_number(oos)
    rule = dict(max_t_rules=float(t_r.max()), max_tstar_rules=float(t_r_star.max()),
                best_rule=int(np.argmax(t_r)), block=b,
                mean_corr_d=dep["mean_corr"], n_eff_d=dep["n_eff"], n_eff_liji_d=dep["n_eff_liji"],
                n_eff_galwey_d=dep["n_eff_galwey"], n_eff_nyholt_d=dep["n_eff_nyholt"],
                mean_corr_cash=dep_c["mean_corr"], n_eff_cash=dep_c["n_eff"], block_cash=b_c,
                max_abs_ret=float(np.abs(r).max()),
                n_capped=int(np.sum(np.abs(r) >= MAX_ABS_SHOCK - 1e-12)), n_rules=N_RULES)
    return rows, rule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True, choices=["iid_t", "garch_t", "garch_premium", "ar1", "regime"])
    ap.add_argument("--premium-annual", type=float, default=0.06)
    ap.add_argument("--phi", type=float, default=0.0)
    ap.add_argument("--mu-annual", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=126.0)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=str(ROOT / "results" / "mc"))
    a = ap.parse_args()

    garch, gsrc = load_garch()
    kw = {}
    param = 0.0
    if a.dgp == "ar1":
        kw = dict(phi=a.phi); param = a.phi
    elif a.dgp == "regime":
        kw = dict(mu=a.mu_annual / 252.0, duration=a.duration); param = a.mu_annual
    elif a.dgp == "garch_premium":
        kw = dict(drift=a.premium_annual / 252.0); param = a.premium_annual
    seeds = np.random.SeedSequence(a.seed).spawn(a.reps)
    t0 = time.time()
    with Pool(a.workers) as pool:
        out = pool.map(one_rep, [(a.dgp, kw, s, garch) for s in seeds], chunksize=4)
    el = time.time() - t0

    cfg = pd.DataFrame([c.as_dict() for c in GRID])
    frames = []
    for rep, (rows, _) in enumerate(out):
        f = cfg[["label", "scheme", "h", "L0", "G", "g"]].copy()
        for k, v in rows.items():
            f[k] = v
        f.insert(0, "rep", rep)
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df.insert(0, "param", param)
    df.insert(0, "dgp", a.dgp)
    rules = pd.DataFrame([r for _, r in out])
    rules.insert(0, "rep", range(len(out)))
    rules.insert(0, "param", param)
    rules.insert(0, "dgp", a.dgp)

    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = f"{a.dgp}_p{param:g}_s{a.seed}"
    df.to_csv(Path(a.out) / f"designs_{tag}.csv.gz", index=False, float_format="%.6g")
    rules.to_csv(Path(a.out) / f"rules_{tag}.csv.gz", index=False, float_format="%.6g")
    meta = dict(dgp=a.dgp, param=param, reps=a.reps, seed=a.seed, T=T_DAYS, start=START,
                end=END, n_oos=END - START, n_designs=len(GRID), n_rules=N_RULES, garch=garch.__dict__,
                garch_source=gsrc, max_abs_shock=MAX_ABS_SHOCK, seconds=round(el, 1), **{k: v for k, v in kw.items()})
    (Path(a.out) / f"meta_{tag}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
