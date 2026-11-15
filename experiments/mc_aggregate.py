"""Aggregate Monte Carlo output into rejection-rate tables.

Reads results/mc/designs_*.csv.gz and rules_*.csv.gz, writes
  results/mc/summary_by_design.csv    rejection rate of each design used alone
  results/mc/summary_procedures.csv   rejection rate of each testing procedure vs K
  results/mc/summary_is_oos.csv       in-sample vs out-of-sample returns of the selected rules
  results/mc/summary_dependence.csv   correlation of the design statistics and effective numbers of designs
  results/mc/corr_t_<dgp>.csv         56 x 56 correlation matrix of the design t-statistics (nulls)
  results/mc/corr_t_cash_<dgp>.csv    the same with cash as the benchmark
  results/mc/perrep_*.csv             per-replication reject flags (K = 56)

Procedures (one-sided, nominal 5%), for a set of K designs:
  single              one design fixed in advance (averaged over the K designs)
  best_naive          report the best of K designs, compare its t to 1.645
  best_naive_sizeadj  best of K against the 95% quantile of max t under the GARCH null (K = 56 only;
                      an oracle critical value, used for size-adjusted power)
  boot_max            best of K against the bootstrap 95% quantile of max t* (Reality Check / StepM step 1)
  boot_avg            average t over the K designs against the bootstrap quantile of average t*
  bonferroni          best of K against z_(1 - 0.05/K)
  bonferroni_eff      the same with K replaced by the participation-ratio effective number of designs
                      estimated within the replication from the correlation matrix of the K series (K = 56 only)
  bonferroni_liji, bonferroni_galwey, bonferroni_nyholt
                      the same with the Li-Ji (2005), Galwey (2009) and Nyholt (2004) effective numbers
  dsr                 Deflated Sharpe Ratio of the best design with N = K trials, reject if DSR > 0.95
  dsr_eff             the same with N = the participation-ratio effective number of designs (K = 56 only)
  single_cash, best_naive_cash, boot_max_cash, boot_avg_cash
                      the first four procedures with cash as the benchmark (test on the strategy's own
                      excess return instead of its return over buy-and-hold)
Bootstrap critical values use the warp-speed method: the 95% quantile, across
replications, of the single bootstrap statistic stored for each replication.
For each K < 56, results are averaged over N_SUBSETS random subsets of the 56 designs.
"""
from __future__ import annotations

import glob
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as st

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wfo.stats import independent_tests, effective_counts  # noqa: E402

MC = ROOT / "results" / "mc"
K_VALUES = (1, 2, 4, 8, 16, 32, 56)
N_SUBSETS = 25
ALPHA = 0.05
Z = st.norm.ppf(1 - ALPHA)
EG = 0.5772156649015329


def n_oos_from_meta() -> int:
    for f in glob.glob(str(MC / "meta_*.json")):
        return int(json.loads(Path(f).read_text())["n_oos"])
    return 4013


N_OOS = n_oos_from_meta()


def dsr_vec(sr_best, skew, kurt, sr_var, N, n=N_OOS):
    """Deflated Sharpe Ratio (vectorised over replications). ``N`` may be a scalar or a vector."""
    N = np.asarray(N, dtype=float)
    sr0 = np.where(N > 1, np.sqrt(sr_var) * ((1 - EG) * st.norm.ppf(1 - 1 / np.maximum(N, 1.0 + 1e-9))
                                             + EG * st.norm.ppf(1 - 1 / (np.maximum(N, 1.0 + 1e-9) * math.e))), 0.0)
    den = np.sqrt(np.maximum(1 - skew * sr_best + (kurt - 1) / 4 * sr_best ** 2, 1e-12))
    return st.norm.cdf((sr_best - sr0) * math.sqrt(n - 1) / den)


def pivots(df: pd.DataFrame):
    labels = sorted(df.label.unique())
    return labels, {c: df.pivot(index="rep", columns="label", values=c)[labels].to_numpy()
                    for c in ("t", "t_star", "d_sharpe", "d_skew", "d_kurt", "t_cash", "t_cash_star")}


EFF_COUNTS = (("bonferroni_liji", "n_eff_liji_d"), ("bonferroni_galwey", "n_eff_galwey_d"),
              ("bonferroni_nyholt", "n_eff_nyholt_d"))


def procedures(df: pd.DataFrame, rules: pd.DataFrame, rng: np.random.Generator, crit_null: float) -> list[dict]:
    labels, piv = pivots(df)
    R = piv["t"].shape[0]
    rows = np.arange(R)
    rules = rules.sort_values("rep")
    n_eff = rules["n_eff_d"].to_numpy()
    out = []
    for K in K_VALUES:
        acc: dict[str, list] = {k: [] for k in ("single", "best_naive", "boot_max", "boot_avg", "bonferroni", "dsr",
                                                "single_cash", "best_naive_cash", "boot_max_cash", "boot_avg_cash")}
        full = K == len(labels)
        if full:
            acc.update(best_naive_sizeadj=[], bonferroni_eff=[], dsr_eff=[], **{k: [] for k, _ in EFF_COUNTS})
        n_sub = 1 if full else N_SUBSETS
        for _ in range(n_sub):
            cols = np.arange(K) if full else rng.choice(len(labels), size=K, replace=False)
            t, ts = piv["t"][:, cols], piv["t_star"][:, cols]
            mx, mxs = t.max(1), ts.max(1)
            av, avs = t.mean(1), ts.mean(1)
            acc["single"].append((t > Z).mean())
            acc["best_naive"].append((mx > Z).mean())
            acc["boot_max"].append((mx > np.quantile(mxs, 1 - ALPHA)).mean())
            acc["boot_avg"].append((av > np.quantile(avs, 1 - ALPHA)).mean())
            acc["bonferroni"].append((mx > st.norm.ppf(1 - ALPHA / K)).mean())
            tc, tcs = piv["t_cash"][:, cols], piv["t_cash_star"][:, cols]
            acc["single_cash"].append((tc > Z).mean())
            acc["best_naive_cash"].append((tc.max(1) > Z).mean())
            acc["boot_max_cash"].append((tc.max(1) > np.quantile(tcs.max(1), 1 - ALPHA)).mean())
            acc["boot_avg_cash"].append((tc.mean(1) > np.quantile(tcs.mean(1), 1 - ALPHA)).mean())
            sh = piv["d_sharpe"][:, cols]
            j = sh.argmax(1)
            var = sh.var(1, ddof=1) if K > 1 else np.zeros(R)
            sk, ku = piv["d_skew"][:, cols][rows, j], piv["d_kurt"][:, cols][rows, j]
            acc["dsr"].append((dsr_vec(sh[rows, j], sk, ku, var, K) > 1 - ALPHA).mean())
            if full:
                acc["best_naive_sizeadj"].append((mx > crit_null).mean())
                acc["bonferroni_eff"].append((mx > st.norm.ppf(1 - ALPHA / n_eff)).mean())
                acc["dsr_eff"].append((dsr_vec(sh[rows, j], sk, ku, var, n_eff) > 1 - ALPHA).mean())
                for key, col in EFF_COUNTS:
                    acc[key].append((mx > st.norm.ppf(1 - ALPHA / np.maximum(rules[col].to_numpy(), 1.0))).mean())
        for proc, v in acc.items():
            p = float(np.mean(v))
            out.append(dict(K=K, procedure=proc, reject=p, mc_se=math.sqrt(max(p * (1 - p), 1e-12) / R),
                            subsets=n_sub, reps=R))
    return out


def perrep(df: pd.DataFrame, rules: pd.DataFrame, crit_null: float) -> pd.DataFrame:
    """Per-replication reject flags over all designs (K = all)."""
    labels, p = pivots(df)
    K = len(labels)
    R = p["t"].shape[0]
    rows = np.arange(R)
    mx, mxs, av, avs = p["t"].max(1), p["t_star"].max(1), p["t"].mean(1), p["t_star"].mean(1)
    j = p["d_sharpe"].argmax(1)
    var = p["d_sharpe"].var(1, ddof=1)
    rules = rules.sort_values("rep")
    n_eff = rules["n_eff_d"].to_numpy()
    dsr = dsr_vec(p["d_sharpe"][rows, j], p["d_skew"][rows, j], p["d_kurt"][rows, j], var, K)
    dsr_e = dsr_vec(p["d_sharpe"][rows, j], p["d_skew"][rows, j], p["d_kurt"][rows, j], var, n_eff)
    return pd.DataFrame(dict(
        dgp=df.dgp.iloc[0], param=df.param.iloc[0], rep=np.arange(R), max_t=mx, avg_t=av,
        n_eff_d=n_eff, mean_corr_d=rules["mean_corr_d"].to_numpy(),
        best_naive=(mx > Z).astype(int),
        best_naive_sizeadj=(mx > crit_null).astype(int),
        boot_max=(mx > np.quantile(mxs, 1 - ALPHA)).astype(int),
        boot_avg=(av > np.quantile(avs, 1 - ALPHA)).astype(int),
        bonferroni=(mx > st.norm.ppf(1 - ALPHA / K)).astype(int),
        bonferroni_eff=(mx > st.norm.ppf(1 - ALPHA / n_eff)).astype(int),
        dsr=(dsr > 1 - ALPHA).astype(int), dsr_eff=(dsr_e > 1 - ALPHA).astype(int),
        max_t_cash=p["t_cash"].max(1), avg_t_cash=p["t_cash"].mean(1),
        best_naive_cash=(p["t_cash"].max(1) > Z).astype(int),
        boot_max_cash=(p["t_cash"].max(1) > np.quantile(p["t_cash_star"].max(1), 1 - ALPHA)).astype(int),
        boot_avg_cash=(p["t_cash"].mean(1) > np.quantile(p["t_cash_star"].mean(1), 1 - ALPHA)).astype(int),
        rules_best_naive=(rules.max_t_rules.to_numpy() > Z).astype(int),
        rules_reality_check=(rules.max_t_rules.to_numpy()
                             > np.quantile(rules.max_tstar_rules, 1 - ALPHA)).astype(int)))


def dependence(df: pd.DataFrame, rules: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Correlation of the design t-statistics across replications and the effective numbers of designs."""
    labels, p = pivots(df)
    C = np.corrcoef(p["t"].T)
    iu = np.triu_indices(len(labels), 1)
    cnt = effective_counts(np.linalg.eigvalsh(C))
    fpr = float((p["t"].max(1) > Z).mean())
    # the same summaries with cash as the benchmark
    Cc = np.corrcoef(p["t_cash"].T)
    cnt_c = effective_counts(np.linalg.eigvalsh(Cc))
    fpr_c = float((p["t_cash"].max(1) > Z).mean())
    capped = rules["n_capped"] > 0
    out = dict(mean_corr_t=float(C[iu].mean()), min_corr_t=float(C[iu].min()), max_corr_t=float(C[iu].max()),
               n_eff_t=cnt["n_eff"], n_eff_liji_t=cnt["n_eff_liji"], n_eff_galwey_t=cnt["n_eff_galwey"],
               n_eff_nyholt_t=cnt["n_eff_nyholt"],
               fpr_best_naive=fpr, n_indep_tests=independent_tests(fpr),
               mean_corr_d=float(rules["mean_corr_d"].mean()), n_eff_d_mean=float(rules["n_eff_d"].mean()),
               n_eff_d_median=float(rules["n_eff_d"].median()),
               n_eff_liji_d_mean=float(rules["n_eff_liji_d"].mean()),
               n_eff_galwey_d_mean=float(rules["n_eff_galwey_d"].mean()),
               n_eff_nyholt_d_mean=float(rules["n_eff_nyholt_d"].mean()),
               mean_corr_t_cash=float(Cc[iu].mean()), min_corr_t_cash=float(Cc[iu].min()),
               n_eff_t_cash=cnt_c["n_eff"], fpr_best_naive_cash=fpr_c, n_indep_tests_cash=independent_tests(fpr_c),
               mean_corr_cash=float(rules["mean_corr_cash"].mean()),
               tied_windows_share=float(df["n_tied"].sum() / np.ceil(N_OOS / df["h"]).sum()) if "n_tied" in df else float("nan"),
               capped_days=int(rules["n_capped"].sum()), capped_reps=int(capped.sum()),
               max_abs_ret=float(rules["max_abs_ret"].max()),
               reps=int(p["t"].shape[0]))
    cm = pd.DataFrame(C, index=labels, columns=labels)
    return out, cm, pd.DataFrame(Cc, index=labels, columns=labels)


def main():
    rng = np.random.default_rng(2026)
    files = sorted(glob.glob(str(MC / "designs_*.csv.gz")))
    # oracle critical value for the naive best-of-56 statistic, from the GARCH null
    crit_null = float("nan")
    for f in files:
        if "garch_t_p0" in f:
            df0 = pd.read_csv(f)
            _, p0 = pivots(df0)
            crit_null = float(np.quantile(p0["t"].max(1), 1 - ALPHA))
    by_design, procs, isoos, deps = [], [], [], []
    for f in files:
        df = pd.read_csv(f)
        dgp, param = df.dgp.iloc[0], float(df.param.iloc[0])
        R = df.rep.nunique()
        g = df.groupby(["label", "scheme", "h", "L0", "G", "g"], dropna=False)
        bd = g.agg(reject=("t", lambda x: float((x > Z).mean())), mean_t=("t", "mean"),
                   is_ann=("is_mean", lambda x: float(x.mean() * 252)),
                   oos_ann=("oos_mean", lambda x: float(x.mean() * 252)),
                   tied_windows=("n_tied", "mean"), distinct_rules=("distinct_rules", "mean")).reset_index()
        bd["mc_se"] = np.sqrt(bd.reject * (1 - bd.reject) / R)
        bd.insert(0, "param", param); bd.insert(0, "dgp", dgp)
        by_design.append(bd)
        rules = pd.read_csv(f.replace("designs_", "rules_"))
        for row in procedures(df, rules, rng, crit_null):
            procs.append(dict(dgp=dgp, param=param, **row))
        perrep(df, rules, crit_null).to_csv(f.replace("designs_", "perrep_"), index=False, float_format="%.6g")
        for proc, rej in (("rules_best_naive", (rules.max_t_rules > Z).mean()),
                          ("rules_reality_check", (rules.max_t_rules > np.quantile(rules.max_tstar_rules, 1 - ALPHA)).mean())):
            procs.append(dict(dgp=dgp, param=param, K=int(rules.get("n_rules", pd.Series([351])).iloc[0]) if "n_rules" in rules else 351,
                              procedure=proc, reject=float(rej),
                              mc_se=math.sqrt(rej * (1 - rej) / len(rules)), subsets=1, reps=len(rules)))
        s = df.groupby("scheme").agg(is_ann=("is_mean", lambda x: x.mean() * 252),
                                     oos_ann=("oos_mean", lambda x: x.mean() * 252)).reset_index()
        s.insert(0, "param", param); s.insert(0, "dgp", dgp)
        isoos.append(s)
        dep, cm, cmc = dependence(df, rules)
        deps.append(dict(dgp=dgp, param=param, **dep))
        if param == 0.0:
            cm.to_csv(MC / f"corr_t_{dgp}.csv", float_format="%.4f")
            cmc.to_csv(MC / f"corr_t_cash_{dgp}.csv", float_format="%.4f")
    pd.concat(by_design).to_csv(MC / "summary_by_design.csv", index=False, float_format="%.5g")
    pd.DataFrame(procs).to_csv(MC / "summary_procedures.csv", index=False, float_format="%.5g")
    pd.concat(isoos).to_csv(MC / "summary_is_oos.csv", index=False, float_format="%.5g")
    pd.DataFrame(deps).to_csv(MC / "summary_dependence.csv", index=False, float_format="%.5g")
    print(pd.DataFrame(procs).query("K in (1, 56, 351)").to_string())
    print(pd.DataFrame(deps).to_string())


if __name__ == "__main__":
    main()
