"""Empirical illustration: the same 351 rules under all 56 walk-forward designs.

For every series and every design in ``default_grid()`` we run the walk-forward
meta-strategy (in each training window pick the rule with the highest in-sample
Sharpe ratio, hold it through the next test block) and evaluate it against
buy-and-hold on a common out-of-sample span.

Returns are excess returns over the one-month T-bill: a long position earns
r - rf, a short earns -(r - rf), flat earns 0. The test statistic for design k is
the Newey-West t-statistic of d_k,t = (strategy excess - buy-and-hold excess),
one-sided (H0: E[d] <= 0). Net results charge ``COST`` per unit of turnover.

Across designs we report: the naive best-of-56 p-value, Bonferroni, the
Romano-Wolf / White max-t bootstrap p-value, the average-t (design-robust)
bootstrap p-value, the dependence between designs (mean pairwise correlation of
the 56 difference series and the eigenvalue-based effective number of designs),
the Deflated Sharpe Ratio of the best design with N = 56 and with N = the
effective number, and the probability of backtest overfitting (CSCV, S = 16).

One-day execution lag. Every series is analysed a second time with all positions taken one
day later (a signal at the close of day t is traded at the close of day t+1), in the training
windows as well as in the test blocks. Stale prices in an index create first-order
autocorrelation that a rule trading at the same close appears to capture; a one-day lag removes
that channel (Bessembinder and Chan 1998). The lagged results are the rows with
``costs`` equal to ``gross_lag`` and ``net_lag``.

Outputs (results/empirical/):
    designs_<sample>.csv     one row per design
    summary.csv              one row per sample and cost treatment (gross, net, gross_lag, net_lag)
    curves_<sample>.csv      month-end cumulative log excess return per design
    eras.csv                 per-design t-statistics within three OOS eras (US)
    classic.csv              full-sample best-of-351 rule per sample (first-generation snooping)
    rule_selection.csv       how often each rule is selected, over all training windows of all designs
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wfo.backtest import run_grid, turnover  # noqa: E402
from wfo.pbo_fast import pbo_cscv_fast  # noqa: E402
from wfo.rules import BURN_IN, RULES, rule_names, rule_returns  # noqa: E402
from wfo.splits import default_grid  # noqa: E402
from wfo.stats import deflated_sharpe, effective_number, nw_tstat, stepm_pvalues  # noqa: E402

OUT = ROOT / "results" / "empirical"
COST = 0.0010           # 10 bp per unit of position change (one-way)
B = 1999                # bootstrap draws
SEED = 20261107
SEED_LAG = 20261110     # separate stream for the one-day-lag analysis, so the main results do not depend on it
START = BURN_IN + 1008 + 21   # first OOS day: every design has a full window behind it

SAMPLES = {
    # name: (file, first date, last date, has_rf)
    "us_1926": ("us_market", None, None, True),
    "developed": ("developed", None, None, True),
    "north_america": ("north_america", None, None, True),
    "europe": ("europe", None, None, True),
    "japan": ("japan", None, None, True),
    "asia_pac_ex_japan": ("asia_pac_ex_japan", None, None, True),
    "sp500_2016": ("sp500_price", None, None, False),
}
ERAS = [("1930-1959", "1900-01-01", "1959-12-31"),
        ("1960-1989", "1960-01-01", "1989-12-31"),
        ("1990-2026", "1990-01-01", "2100-01-01")]


def load(name, first, last, has_rf):
    df = pd.read_csv(ROOT / "data" / "processed" / f"{name}.csv")
    if first:
        df = df[df["date"] >= first]
    if last:
        df = df[df["date"] <= last]
    df = df.reset_index(drop=True)
    rf = df["rf"].to_numpy() if has_rf else np.zeros(len(df))
    return df["date"].to_numpy(), df["price"].to_numpy(), df["ret"].to_numpy(), rf


def ann(x):
    return 252.0 * np.mean(x)


def sharpe_ann(x):
    s = np.std(x, ddof=1)
    return float(np.sqrt(252.0) * np.mean(x) / s) if s > 0 else 0.0


def lagged(pos, ex):
    """Positions and rule returns when every signal is executed one day later."""
    pos_l = np.vstack([np.zeros((1, pos.shape[1])), pos[:-1]])
    R_l = np.zeros_like(pos_l)
    R_l[1:] = pos_l[:-1] * ex[1:, None]
    return R_l, pos_l


def analyse(sample, rng, rng_lag):
    file, first, last, has_rf = SAMPLES[sample]
    dates, price, ret, rf = load(file, first, last, has_rf)
    ex = ret - rf
    R, pos = rule_returns(price, ex)
    grid = default_grid()
    n = len(ret)
    oos, held, is_mean, res = run_grid(R, pos, grid, START, n, BURN_IN)
    bh = ex[START:n]
    cost = COST * turnover(held)
    d_gross = oos - bh[None, :]
    d_net = d_gross - cost
    oos_dates = dates[START:n]

    rows = []
    t_g, t_n = nw_tstat(d_gross), nw_tstat(d_net)
    for k, cfg in enumerate(grid):
        rows.append(dict(
            sample=sample, **cfg.as_dict(), n_blocks=res[k].n_blocks,
            is_ann=252 * is_mean[k], oos_ann=ann(oos[k]), oos_sharpe=sharpe_ann(oos[k]),
            bh_ann=ann(bh), bh_sharpe=sharpe_ann(bh),
            d_ann=ann(d_gross[k]), t=t_g[k], p=1 - st.norm.cdf(t_g[k]),
            d_ann_net=ann(d_net[k]), t_net=t_n[k], p_net=1 - st.norm.cdf(t_n[k]),
            turnover_ann=252 * float(np.mean(turnover(held[k:k + 1]))),
            n_switch=int(np.sum(np.diff(res[k].chosen) != 0)),
            distinct_rules=int(len(np.unique(res[k].chosen))),
            n_tied=res[k].n_tied,
            top_rule=rule_names()[int(np.bincount(res[k].chosen).argmax())],
            top_rule_share=float(np.bincount(res[k].chosen).max() / res[k].n_blocks),
        ))
    des = pd.DataFrame(rows)
    # rule selection over all training windows of all designs
    names = rule_names()
    allc = np.concatenate([x.chosen for x in res])
    cnt = np.bincount(allc, minlength=len(names))
    sel = pd.DataFrame(dict(sample=sample, rule=names, family=[q.family for q in RULES], windows=cnt,
                            share=cnt / cnt.sum())).sort_values("windows", ascending=False)

    summ = []
    for tag, d, t in (("gross", d_gross, t_g), ("net", d_net, t_n)):
        sm = stepm_pvalues(d, B=B, rng=rng)
        K = d.shape[0]
        kbest = int(np.argmax(t))
        sr_trials = d.mean(1) / d.std(1, ddof=1)
        dep = effective_number(d)
        dsr = deflated_sharpe(d[kbest], sr_trials)
        dsr_eff = deflated_sharpe(d[kbest], sr_trials, n_trials=dep["n_eff"])
        pbo = pbo_cscv_fast(d.T, S=16)
        summ.append(dict(
            sample=sample, costs=tag, first_oos=str(oos_dates[0]), last_oos=str(oos_dates[-1]),
            n_oos=d.shape[1], K=K, best_label=grid[kbest].label, best_t=float(t[kbest]),
            p_naive_best=float(1 - st.norm.cdf(t[kbest])),
            p_bonferroni=float(min(1.0, K * (1 - st.norm.cdf(t[kbest])))),
            p_maxt=float(sm["p_max"]), p_avgt=float(sm["p_avg"]), mean_t=float(np.mean(t)),
            median_t=float(np.median(t)), min_t=float(np.min(t)),
            n_sig_raw=int(np.sum(t > st.norm.ppf(0.95))), n_sig_stepm=int(np.sum(sm["p_stepm"] < 0.05)),
            frac_beat_bh=float(np.mean(d.mean(1) > 0)), dsr_best=float(dsr), dsr_best_eff=float(dsr_eff),
            mean_corr_d=dep["mean_corr"], n_eff_d=dep["n_eff"], pbo=pbo["pbo"],
            block=float(sm["block"]), bh_ann=ann(bh), bh_sharpe=sharpe_ann(bh),
            n_distinct_lt10=int(np.sum(des["distinct_rules"] < 10)), tied_windows=int(des["n_tied"].sum()),
            windows=int(des["n_blocks"].sum()), top_rule=sel["rule"].iloc[0], top_rule_share=float(sel["share"].iloc[0]),
            top3_share=float(sel["share"].iloc[:3].sum()),
            turnover_ann=252 * float(np.mean(turnover(held))),
        ))
        des[f"p_stepm_{tag}"] = sm["p_stepm"]

    # classic first-generation snooping: best of 351 rules on the whole OOS span, no walk-forward
    Rr = R[START:n] - bh[:, None]
    tr = nw_tstat(Rr.T)
    jb = int(np.argmax(tr))
    smr = stepm_pvalues(Rr.T, B=B, rng=rng)
    classic = dict(sample=sample, best_rule=rule_names()[jb], t=float(tr[jb]),
                   p_naive=float(1 - st.norm.cdf(tr[jb])), p_rc=float(smr["p_max"]),
                   n_rules=int(Rr.shape[1]), n_sig_raw=int(np.sum(tr > st.norm.ppf(0.95))))

    # one-day execution lag: the same designs and rules, every position taken one day later
    R_l, pos_l = lagged(pos, ex)
    oos_l, held_l, _, res_l = run_grid(R_l, pos_l, grid, START, n, BURN_IN)
    d_gross_l = oos_l - bh[None, :]
    d_net_l = d_gross_l - COST * turnover(held_l)
    for tag, d in (("gross_lag", d_gross_l), ("net_lag", d_net_l)):
        t = nw_tstat(d)
        sm = stepm_pvalues(d, B=B, rng=rng_lag)
        kbest = int(np.argmax(t))
        dep = effective_number(d)
        summ.append(dict(
            sample=sample, costs=tag, first_oos=str(oos_dates[0]), last_oos=str(oos_dates[-1]),
            n_oos=d.shape[1], K=d.shape[0], best_label=grid[kbest].label, best_t=float(t[kbest]),
            p_naive_best=float(1 - st.norm.cdf(t[kbest])),
            p_bonferroni=float(min(1.0, d.shape[0] * (1 - st.norm.cdf(t[kbest])))),
            p_maxt=float(sm["p_max"]), p_avgt=float(sm["p_avg"]), mean_t=float(np.mean(t)),
            median_t=float(np.median(t)), min_t=float(np.min(t)),
            n_sig_raw=int(np.sum(t > st.norm.ppf(0.95))), n_sig_stepm=int(np.sum(sm["p_stepm"] < 0.05)),
            frac_beat_bh=float(np.mean(d.mean(1) > 0)),
            mean_corr_d=dep["mean_corr"], n_eff_d=dep["n_eff"], pbo=pbo_cscv_fast(d.T, S=16)["pbo"],
            block=float(sm["block"]), bh_ann=ann(bh), bh_sharpe=sharpe_ann(bh),
            turnover_ann=252 * float(np.mean(turnover(held_l))),
        ))

    # month-end cumulative log excess returns (gross), for the fan chart
    month = pd.to_datetime(pd.Series(oos_dates)).dt.to_period("M").astype(str).to_numpy()
    last_in_month = np.r_[month[1:] != month[:-1], True]
    cum = np.cumsum(np.log1p(oos), axis=1)[:, last_in_month]
    cum_net = np.cumsum(np.log1p(oos - cost), axis=1)[:, last_in_month]
    cum_bh = np.cumsum(np.log1p(bh))[last_in_month]
    curves = pd.DataFrame(np.hstack([cum.T, cum_net.T]),
                          columns=[c.label for c in grid] + [c.label + " net" for c in grid])
    curves.insert(0, "buy_hold", cum_bh)
    curves.insert(0, "month", month[last_in_month])

    eras, era_summ = [], []
    if sample == "us_1926":
        to = turnover(held)
        for era, a, b in ERAS:
            m = (oos_dates >= a) & (oos_dates <= b)
            te = nw_tstat(d_gross[:, m])
            tn = nw_tstat(d_net[:, m])
            for tag, dd, tt in (("gross", d_gross[:, m], te), ("net", d_net[:, m], tn)):
                sm = stepm_pvalues(dd, B=B, rng=rng)
                era_summ.append(dict(era=era, costs=tag, n=int(m.sum()), mean_t=float(tt.mean()),
                                     min_t=float(tt.min()), max_t=float(tt.max()),
                                     n_sig_raw=int(np.sum(tt > st.norm.ppf(0.95))),
                                     n_sig_stepm=int(np.sum(sm["p_stepm"] < 0.05)),
                                     p_naive_best=float(1 - st.norm.cdf(tt.max())),
                                     p_maxt=float(sm["p_max"]), p_avgt=float(sm["p_avg"]),
                                     # cost per unit of position change at which the average design
                                     # earns exactly the buy-and-hold return, in basis points
                                     turnover_ann=252 * float(to[:, m].mean()),
                                     breakeven_bp=1e4 * float(d_gross[:, m].mean() / to[:, m].mean())))
            for k, cfg in enumerate(grid):
                eras.append(dict(era=era, label=cfg.label, scheme=cfg.scheme, h=cfg.h,
                                 t=te[k], t_net=tn[k], d_ann=ann(d_gross[k, m]), d_ann_net=ann(d_net[k, m]),
                                 n=int(m.sum())))
        # the same periods with the one-day execution lag (after the main results, on its own stream)
        to_l = turnover(held_l)
        for era, a, b in ERAS:
            m = (oos_dates >= a) & (oos_dates <= b)
            for tag, dd in (("gross_lag", d_gross_l[:, m]), ("net_lag", d_net_l[:, m])):
                tt = nw_tstat(dd)
                sm = stepm_pvalues(dd, B=B, rng=rng_lag)
                era_summ.append(dict(era=era, costs=tag, n=int(m.sum()), mean_t=float(tt.mean()),
                                     min_t=float(tt.min()), max_t=float(tt.max()),
                                     n_sig_raw=int(np.sum(tt > st.norm.ppf(0.95))),
                                     n_sig_stepm=int(np.sum(sm["p_stepm"] < 0.05)),
                                     p_naive_best=float(1 - st.norm.cdf(tt.max())),
                                     p_maxt=float(sm["p_max"]), p_avgt=float(sm["p_avg"]),
                                     turnover_ann=252 * float(to_l[:, m].mean()),
                                     breakeven_bp=1e4 * float(d_gross_l[:, m].mean() / to_l[:, m].mean())))
    if era_summ:
        pd.DataFrame(era_summ).to_csv(OUT / "eras_summary.csv", index=False, float_format="%.6g")
    return des, pd.DataFrame(summ), curves, classic, pd.DataFrame(eras), sel


def main(samples=None):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rng_lag = np.random.default_rng(SEED_LAG)
    all_summ, classics, sels = [], [], []
    for s in (samples or SAMPLES):
        t0 = time.time()
        des, summ, curves, classic, eras, sel = analyse(s, rng, rng_lag)
        sels.append(sel)
        des.to_csv(OUT / f"designs_{s}.csv", index=False, float_format="%.6g")
        curves.to_csv(OUT / f"curves_{s}.csv", index=False, float_format="%.5g")
        if len(eras):
            eras.to_csv(OUT / "eras.csv", index=False, float_format="%.6g")
        all_summ.append(summ)
        classics.append(classic)
        print(f"{s}: {time.time() - t0:.0f}s")
        print(summ[["costs", "n_oos", "best_label", "best_t", "p_naive_best", "p_maxt", "p_avgt",
                    "mean_t", "n_sig_raw", "n_eff_d", "pbo", "tied_windows", "top_rule", "top_rule_share"]].to_string(index=False))
    pd.concat(all_summ).to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    pd.concat(sels).to_csv(OUT / "rule_selection.csv", index=False, float_format="%.6g")
    pd.DataFrame(classics).to_csv(OUT / "classic.csv", index=False, float_format="%.6g")
    meta = dict(cost_one_way=COST, bootstrap_draws=B, seed=SEED, seed_lag=SEED_LAG, oos_start_index=START,
                burn_in=BURN_IN, n_designs=len(default_grid()), n_rules=len(RULES))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
