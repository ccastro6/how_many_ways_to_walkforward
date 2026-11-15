"""Write paper/numbers.tex: every number quoted in the text, read from the result files.

Nothing in the manuscript is typed by hand; if a result changes, rerun this script.
Missing inputs produce a visible placeholder (??) instead of a stale number.
"""
from __future__ import annotations

import glob
import json
import math
from statistics import NormalDist
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from wfo.rules import BURN_IN, N_RULES, family_counts  # noqa: E402

EMP, MC, CAL = ROOT / "results" / "empirical", ROOT / "results" / "mc", ROOT / "results" / "calibration"
macros: dict[str, str] = {}


def put(name, value):
    macros[name] = str(value)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pv(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


def neg(s):
    return s.replace("-", "$-$")


def fmt_sci(x):
    m, e = f"{x:.2e}".split("e")
    return rf"\ensuremath{{{float(m):.2f}\times10^{{{int(e)}}}}}"


# ------------------------------------------------------------- general
put("NumDate", date.today().strftime("%B %Y").replace(" 0", " "))
fc = family_counts()
put("NumRules", N_RULES)
put("NumRulesFilter", fc["filter"])
put("NumRulesMA", fc["ma"])
put("NumRulesSR", fc["sr"])
put("NumRulesCB", fc["cb"])
put("NumBurnIn", BURN_IN)

# calibration
g = json.loads((CAL / "garch_us_market.json").read_text())
put("NumGarchOmega", fmt_sci(g["omega"]))
put("NumGarchAlpha", f"{g['alpha']:.3f}")
put("NumGarchBeta", f"{g['beta']:.3f}")
put("NumGarchNu", f"{g['nu']:.2f}")

# ------------------------------------------------------------- Monte Carlo
MC_KEYS = ["NumNullReps", "NumAltReps", "NumT", "NumNoos", "NumPhiList", "NumMuList", "NumFullbootB", "NumFullbootReps",
           "NumFullbootMax", "NumFullbootAvg", "NumFullbootMaxIid", "NumFullbootAvgIid", "NumSubsets",
           "NumSizeSingleMin", "NumSizeSingleMax", "NumSizeSingle", "NumSizeBestTwo", "NumSizeBestEight",
           "NumSizeBestNaive", "NumSizeBestNaiveRound", "NumSizeBonf", "NumSizeDSR", "NumSizeBonfEff", "NumSizeDSREff",
           "NumSizeMaxT", "NumSizeAvgT", "NumSizeRulesNaive", "NumSizeRulesRC", "NumISann", "NumOOSann",
           "NumMeanCorrT", "NumMinCorrT", "NumNeffT", "NumNindep", "NumNindepRound", "NumMeanCorrD", "NumNeffD",
           "NumMCdistinct", "NumMCtiedShare", "NumPowerPhi", "NumPowerAvg", "NumPowerMax", "NumPowerMedian",
           "NumPowerSingleMin", "NumPowerSingleMax", "NumPowerBest", "NumPowerExpanding", "NumPowerNaiveAdj",
           "NumPowerNaive", "NumPowerBonfEff", "NumPowerDSREff", "NumPowerMu", "NumPowerAvgMu", "NumPowerMaxMu",
           "NumPowerMedianMu", "NumPowerExpandingMu", "NumPowerNaiveAdjMu", "NumPowerGapPoints", "NumPowerGapBest", "NumSizeBonfIndep", "NumFullbootSE", "NumFullbootRepsIid",
           # benchmark dependence, effective-number formulas, premium null, shock cap, full-bootstrap power
           "NumSizeSingleCash", "NumSizeBestNaiveCash", "NumSizeMaxTCash", "NumSizeAvgTCash", "NumSizeBestNaiveCashIid",
           "NumSizeMaxTCashIid", "NumSizeAvgTCashIid", "NumSizeRobustMin", "NumSizeRobustMax",
           "NumMeanCorrTCash", "NumMinCorrTCash", "NumMeanCorrCash", "NumNindepCash", "NumNindepCashRound",
           "NumNeffLiji", "NumNeffGalwey", "NumNeffNyholt", "NumNeffNyholtRound",
           "NumSizeBonfLiji", "NumSizeBonfGalwey", "NumSizeBonfNyholt",
           "NumPremium", "NumPremiumReps", "NumSizeSinglePremium", "NumSizeBestNaivePremium",
           "NumCapPct", "NumCappedDays", "NumCappedReps",
           "NumFullbootMaxCash", "NumFullbootAvgCash", "NumFullbootMaxCashIid", "NumFullbootAvgCashIid",
           "NumFullbootPowerReps", "NumFullbootPowerMax", "NumFullbootPowerAvg", "NumFullbootPowerMaxMu",
           "NumFullbootPowerAvgMu"]
for k in MC_KEYS:
    put(k, "??")

N_OOS_MC = 4013
sp_f, sd_f, dep_f = MC / "summary_procedures.csv", MC / "summary_by_design.csv", MC / "summary_dependence.csv"
if sp_f.exists() and sd_f.exists():
    sp, sd = pd.read_csv(sp_f), pd.read_csv(sd_f)
    metas = [json.loads(Path(f).read_text()) for f in glob.glob(str(MC / "meta_*.json"))]
    gm = [m for m in metas if m.get("dgp") == "garch_t"]
    am = [m for m in metas if m.get("dgp") == "ar1"]
    if gm:
        put("NumNullReps", f"{int(gm[0].get('reps', 0)):,}")
        put("NumT", f"{int(gm[0].get('T', 5292)):,}")
        put("NumNoos", f"{int(gm[0].get('n_oos', 4013)):,}")
        N_OOS_MC = int(gm[0].get("n_oos", 4013))
    if am:
        put("NumAltReps", f"{int(am[0].get('reps', 0)):,}")
    phis = sorted({float(m["param"]) for m in metas if m.get("dgp") == "ar1"})
    mus = sorted({float(m["param"]) for m in metas if m.get("dgp") == "regime"})
    if phis:
        put("NumPhiList", ", ".join(f"{p:.2f}" for p in phis))
    if mus:
        put("NumMuList", ", ".join(f"{100 * m:g}\\%" for m in mus))

    def rate(dgp, proc, K, par=0.0):
        q = sp[(sp["dgp"] == dgp) & (sp["procedure"] == proc) & (sp["K"] == K) & np.isclose(sp["param"], par)]
        return float(q["reject"].iloc[0]) if len(q) else float("nan")

    gd = sd[sd["dgp"] == "garch_t"]
    put("NumSizeSingleMin", pct(gd["reject"].min()))
    put("NumSizeSingleMax", pct(gd["reject"].max()))
    put("NumSizeSingle", pct(gd["reject"].mean()))
    put("NumSizeBestTwo", pct(rate("garch_t", "best_naive", 2)))
    put("NumSizeBestEight", pct(rate("garch_t", "best_naive", 8)))
    b56 = rate("garch_t", "best_naive", 56)
    put("NumSizeBestNaive", pct(b56))
    put("NumSizeBestNaiveRound", f"{b56 / 0.05:.1f}".rstrip("0").rstrip("."))
    put("NumSizeBonf", pct(rate("garch_t", "bonferroni", 56)))
    put("NumSizeDSR", pct(rate("garch_t", "dsr", 56)))
    put("NumSizeBonfEff", pct(rate("garch_t", "bonferroni_eff", 56)))
    put("NumSizeDSREff", pct(rate("garch_t", "dsr_eff", 56)))
    put("NumSizeMaxT", pct(rate("garch_t", "boot_max", 56)))
    put("NumSizeAvgT", pct(rate("garch_t", "boot_avg", 56)))
    put("NumSizeRulesNaive", pct(rate("garch_t", "rules_best_naive", N_RULES)))
    put("NumSizeRulesRC", pct(rate("garch_t", "rules_reality_check", N_RULES)))
    put("NumSubsets", int(sp["subsets"][sp["K"] < 56].max()) if "subsets" in sp.columns else 25)
    # the same tests with cash as the benchmark
    put("NumSizeSingleCash", pct(rate("garch_t", "single_cash", 56)))
    put("NumSizeBestNaiveCash", pct(rate("garch_t", "best_naive_cash", 56)))
    put("NumSizeMaxTCash", pct(rate("garch_t", "boot_max_cash", 56)))
    put("NumSizeAvgTCash", pct(rate("garch_t", "boot_avg_cash", 56)))
    put("NumSizeBestNaiveCashIid", pct(rate("iid_t", "best_naive_cash", 56)))
    put("NumSizeMaxTCashIid", pct(rate("iid_t", "boot_max_cash", 56)))
    put("NumSizeAvgTCashIid", pct(rate("iid_t", "boot_avg_cash", 56)))
    # range of the two design-robust tests over both benchmarks and both nulls (warp-speed estimates)
    robust = [rate(dg, pr, 56) for dg in ("garch_t", "iid_t")
              for pr in ("boot_max", "boot_avg", "boot_max_cash", "boot_avg_cash")]
    if not np.isnan(robust).any():
        put("NumSizeRobustMin", pct(min(robust)))
        put("NumSizeRobustMax", pct(max(robust)))
    # Bonferroni fed with three published eigenvalue counts, estimated within each simulated market
    put("NumSizeBonfLiji", pct(rate("garch_t", "bonferroni_liji", 56)))
    put("NumSizeBonfGalwey", pct(rate("garch_t", "bonferroni_galwey", 56)))
    put("NumSizeBonfNyholt", pct(rate("garch_t", "bonferroni_nyholt", 56)))
    # GARCH null with an equity premium, against buy-and-hold
    pm = [m for m in metas if m.get("dgp") == "garch_premium"]
    if pm:
        prem = float(pm[0]["param"])
        put("NumPremium", f"{100 * prem:g}\\%")
        put("NumPremiumReps", f"{int(pm[0]['reps']):,}")
        put("NumSizeSinglePremium", pct(rate("garch_premium", "single", 56, prem)))
        put("NumSizeBestNaivePremium", pct(rate("garch_premium", "best_naive", 56, prem)))
    if gm:
        put("NumCapPct", f"{100 * float(gm[0].get('max_abs_shock', 0.5)):g}\\%")
    put("NumMCdistinct", f"{gd['distinct_rules'].mean():.0f}")
    io_f = MC / "summary_is_oos.csv"
    if io_f.exists():
        io = pd.read_csv(io_f)
        q = io[io["dgp"] == "garch_t"]
        put("NumISann", pct(q["is_ann"].mean()))
        put("NumOOSann", pct(q["oos_ann"].mean()))
    fbs = [json.loads(Path(f).read_text()) for f in glob.glob(str(MC / "fullboot_garch_t*.json"))]
    def pool(js):
        """Pool full-bootstrap runs of one null: rejection counts summed over all replications."""
        n = sum(int(j["reps"]) for j in js)
        return (n, sum(j["reject_max"] * j["reps"] for j in js) / n, sum(j["reject_avg"] * j["reps"] for j in js) / n,
                int(js[0].get("B", 299)))

    if fbs:
        n_fb, r_max, r_avg, b_fb = pool(fbs)
        put("NumFullbootB", f"{b_fb:,}")
        put("NumFullbootReps", f"{n_fb:,}")
        put("NumFullbootMax", pct(r_max))
        put("NumFullbootAvg", pct(r_avg))
        put("NumFullbootSE", f"{100 * math.sqrt(0.05 * 0.95 / n_fb):.1f}")
    fbi = [json.loads(Path(f).read_text()) for f in glob.glob(str(MC / "fullboot_iid_t*.json"))]
    if fbi:
        n_fbi, r_max_i, r_avg_i, _ = pool(fbi)
        put("NumFullbootRepsIid", f"{n_fbi:,}")
        put("NumFullbootMaxIid", pct(r_max_i))
        put("NumFullbootAvgIid", pct(r_avg_i))

    def pool_key(js, key):
        js = [j for j in js if key in j]
        return sum(j[key] * j["reps"] for j in js) / sum(int(j["reps"]) for j in js) if js else float("nan")

    for js, sfx in ((fbs, ""), (fbi, "Iid")):
        if js and all("reject_max_cash" in j for j in js):
            put("NumFullbootMaxCash" + sfx, pct(pool_key(js, "reject_max_cash")))
            put("NumFullbootAvgCash" + sfx, pct(pool_key(js, "reject_avg_cash")))
    # full-bootstrap power at the two alternative points quoted in the text
    fpa = [json.loads(Path(f).read_text()) for f in glob.glob(str(MC / "fullboot_ar1_p0.1_*.json"))]
    fpr = [json.loads(Path(f).read_text()) for f in glob.glob(str(MC / "fullboot_regime_p0.3_*.json"))]
    if fpa and fpr:
        n_a, pa_max, pa_avg, _ = pool(fpa)
        n_r, pr_max, pr_avg, _ = pool(fpr)
        put("NumFullbootPowerReps", f"{n_a:,}" if n_a == n_r else f"{n_a:,} and {n_r:,}")
        put("NumFullbootPowerMax", pct(pa_max))
        put("NumFullbootPowerAvg", pct(pa_avg))
        put("NumFullbootPowerMaxMu", pct(pr_max))
        put("NumFullbootPowerAvgMu", pct(pr_avg))
    if dep_f.exists():
        dep = pd.read_csv(dep_f)
        q = dep[(dep["dgp"] == "garch_t")].iloc[0]
        put("NumMeanCorrT", f"{q['mean_corr_t']:.2f}")
        put("NumMinCorrT", f"{q['min_corr_t']:.2f}")
        put("NumNeffT", f"{q['n_eff_t']:.1f}")
        put("NumNindep", f"{q['n_indep_tests']:.1f}")
        put("NumNindepRound", f"{round(q['n_indep_tests']):.0f}")
        put("NumMeanCorrD", f"{q['mean_corr_d']:.2f}")
        put("NumNeffD", f"{q['n_eff_d_mean']:.1f}")
        put("NumMCtiedShare", pct(q["tied_windows_share"], 0))
        if "mean_corr_t_cash" in q.index:
            put("NumMeanCorrTCash", f"{q['mean_corr_t_cash']:.2f}")
            put("NumMinCorrTCash", f"{q['min_corr_t_cash']:.2f}")
            put("NumMeanCorrCash", f"{q['mean_corr_cash']:.2f}")
            put("NumNindepCash", f"{q['n_indep_tests_cash']:.1f}")
            put("NumNindepCashRound", f"{round(q['n_indep_tests_cash']):.0f}")
            put("NumNeffLiji", f"{q['n_eff_liji_d_mean']:.1f}")
            put("NumNeffGalwey", f"{q['n_eff_galwey_d_mean']:.1f}")
            put("NumNeffNyholt", f"{q['n_eff_nyholt_d_mean']:.1f}")
            put("NumNeffNyholtRound", f"{round(q['n_eff_nyholt_d_mean']):.0f}")
            put("NumCappedDays", f"{int(q['capped_days']):,}")
            put("NumCappedReps", f"{int(q['capped_reps']):,}")
        # Bonferroni fed with the independent-tests count, from the per-replication maxima
        pr_f = glob.glob(str(MC / "perrep_garch_t*.csv.gz"))
        if pr_f:
            pr = pd.read_csv(pr_f[0])
            z = NormalDist().inv_cdf(1 - 0.05 / float(q["n_indep_tests"]))
            put("NumSizeBonfIndep", pct((pr["max_t"] > z).mean()))

    # power
    def single(dgp, par):
        return sd[(sd["dgp"] == dgp) & np.isclose(sd["param"], par)]

    if phis and mus:
        pa = 0.10 if any(np.isclose(phis, 0.10)) else phis[len(phis) // 2]
        mb = 0.30 if any(np.isclose(mus, 0.30)) else mus[len(mus) // 2]
        s1, s2 = single("ar1", pa), single("regime", mb)
        put("NumPowerPhi", f"{pa:.2f}")
        put("NumPowerMu", f"{100 * mb:g}\\%")
        put("NumPowerAvg", pct(rate("ar1", "boot_avg", 56, pa)))
        put("NumPowerMax", pct(rate("ar1", "boot_max", 56, pa)))
        put("NumPowerNaive", pct(rate("ar1", "best_naive", 56, pa)))
        put("NumPowerNaiveAdj", pct(rate("ar1", "best_naive_sizeadj", 56, pa)))
        put("NumPowerBonfEff", pct(rate("ar1", "bonferroni_eff", 56, pa)))
        put("NumPowerDSREff", pct(rate("ar1", "dsr_eff", 56, pa)))
        put("NumPowerMedian", pct(s1["reject"].median()))
        put("NumPowerSingleMin", pct(s1["reject"].min(), 0))
        put("NumPowerSingleMax", pct(s1["reject"].max(), 0))
        put("NumPowerBest", pct(s1["reject"].max()))
        put("NumPowerExpanding", pct(s1[s1["scheme"] == "expanding"]["reject"].mean()))
        gap = s1[s1["scheme"] == "expanding"]["reject"].mean() - rate("ar1", "boot_avg", 56, pa)
        put("NumPowerGapPoints", f"{100 * gap:.0f}")
        put("NumPowerGapBest", f"{100 * (s1['reject'].max() - rate('ar1', 'boot_avg', 56, pa)):.0f}")
        put("NumPowerAvgMu", pct(rate("regime", "boot_avg", 56, mb)))
        put("NumPowerMaxMu", pct(rate("regime", "boot_max", 56, mb)))
        put("NumPowerNaiveAdjMu", pct(rate("regime", "best_naive_sizeadj", 56, mb)))
        put("NumPowerMedianMu", pct(s2["reject"].median()))
        put("NumPowerExpandingMu", pct(s2[s2["scheme"] == "expanding"]["reject"].mean()))

# ------------------------------------------------------------- empirical
s = pd.read_csv(EMP / "summary.csv")
us_g = s[(s["sample"] == "us_1926") & (s["costs"] == "gross")].iloc[0]
us_n = s[(s["sample"] == "us_1926") & (s["costs"] == "net")].iloc[0]
put("NumUSoosFirst", str(us_g["first_oos"])[:4])
put("NumUSgrossSig", int(us_g["n_sig_raw"]))
put("NumUSgrossBestT", f"{us_g['best_t']:.2f}")
put("NumUSgrossMinT", neg(f"{us_g['min_t']:.2f}"))
put("NumUSgrossPmax", pv(us_g["p_maxt"]))
put("NumUSgrossPavg", pv(us_g["p_avgt"]))
put("NumUSnetSig", int(us_n["n_sig_raw"]))
put("NumUSnetBestT", neg(f"{us_n['best_t']:.2f}"))
put("NumUSnetMeanT", neg(f"{us_n['mean_t']:.2f}"))
put("NumUSnetPmax", pv(us_n["p_maxt"]))
put("NumUSnetPavg", pv(us_n["p_avgt"]))
put("NumUSmeanCorr", f"{us_g['mean_corr_d']:.2f}")
put("NumUSneff", f"{us_g['n_eff_d']:.1f}")
put("NumUSdistinctLow", int(us_g["n_distinct_lt10"]))
put("NumUStopRule", str(us_g["top_rule"]).replace("(", "(").replace(",", ",\\allowbreak "))
put("NumUStopShare", pct(us_g["top_rule_share"], 0))
put("NumUStopThreeShare", pct(us_g["top3_share"], 0))
put("NumUStiedWindows", f"{int(us_g['tied_windows']):,}")
put("NumUSwindows", f"{int(us_g['windows']):,}")
put("NumUStiedShare", pct(us_g["tied_windows"] / us_g["windows"], 1))
d = pd.read_csv(EMP / "designs_us_1926.csv")
put("NumUSturnover", f"{d['turnover_ann'].median() / 2:.0f}")   # one switch long<->short = 2 units
put("NumUSturnoverUnits", f"{d['turnover_ann'].median():.0f}")
put("NumUSmaxTiedDesign", d.loc[d["n_tied"].idxmax(), "label"].replace("|", "/"))
put("NumUSmaxTied", int(d["n_tied"].max()))
put("NumUSmaxTiedBlocks", int(d.loc[d["n_tied"].idxmax(), "n_blocks"]))
_r = d.loc[d["n_tied"].idxmax()]
_win = ("expanding window" if _r["expanding"] else
        f"{'rolling' if _r['G'] == 0 else 'growing-rolling'} {int(_r['L0'])}-day window")
put("NumUSmaxTiedDesc", f"{_win}, {int(_r['h'])}-day test blocks, "
    + ("no gap" if int(_r["g"]) == 0 else f"{int(_r['g'])}-day gap"))
put("NumNoosShareUS", f"{100 * N_OOS_MC / int(us_g['n_oos']):.0f}")
put("NumNoosYears", f"{N_OOS_MC / 252:.0f}")
um = pd.read_csv(ROOT / "data" / "processed" / "us_market.csv")
for nm, a, b in (("NumACone", "1926", "1960"), ("NumACtwo", "1960", "1990"), ("NumACthree", "1990", "2100"),
                 ("NumACpre", "1926", "1982"), ("NumACpost", "1982", "2100")):
    x = um[(um["date"] >= a) & (um["date"] < b)]["ret"].to_numpy()
    put(nm, neg(f"{np.corrcoef(x[1:], x[:-1])[0, 1]:.2f}"))
apx = pd.read_csv(ROOT / "data" / "processed" / "asia_pac_ex_japan.csv")["ret"].to_numpy()
put("NumAPac", neg(f"{np.corrcoef(apx[1:], apx[:-1])[0, 1]:.2f}"))
ap_n = s[(s["sample"] == "asia_pac_ex_japan") & (s["costs"] == "net")].iloc[0]
ap_g = s[(s["sample"] == "asia_pac_ex_japan") & (s["costs"] == "gross")].iloc[0]
dv_g = s[(s["sample"] == "developed") & (s["costs"] == "gross")].iloc[0]
put("NumAPnetPavg", pv(ap_n["p_avgt"]))
put("NumAPnetPmax", pv(ap_n["p_maxt"]))
put("NumAPgrossSig", int(ap_g["n_sig_raw"]))
put("NumDVgrossSig", int(dv_g["n_sig_raw"]))
put("NumAPnetPBO", f"{100 * ap_n['pbo']:.0f}")
put("NumAPnetSig", int(ap_n["n_sig_raw"]))
put("NumAPnetStepm", int(ap_n["n_sig_stepm"]))
# where the US gross out-performance accrued: cumulative log excess return relative to buy-and-hold
cv = pd.read_csv(EMP / "curves_us_1926.csv").set_index("month")
gcols = [c for c in cv.columns if c != "buy_hold" and not c.endswith(" net")]
rel_g = cv[gcols].sub(cv["buy_hold"], axis=0)
rel_n = cv[[c + " net" for c in gcols]].sub(cv["buy_hold"], axis=0)
put("NumUSgrossSharePreFutures", f"{100 * rel_g.loc['1981-12'].mean() / rel_g.loc['1989-12'].mean():.0f}")
put("NumUSnetRelPreFutures", f"{rel_n.loc['1981-12'].mean():.1f}")
put("NumUSnetRelEndEighties", f"{rel_n.loc['1989-12'].mean():.1f}")
put("NumUSnetRelEnd", neg(f"{rel_n.loc[cv.index[-1]].mean():.1f}"))
put("NumUSnetPBO", f"{100 * us_n['pbo']:.0f}")
# count of international series with no significant design (gross) among single regions
zero = [smp for smp in ("north_america", "europe", "japan")
        if int(s[(s["sample"] == smp) & (s["costs"] == "gross")].iloc[0]["n_sig_raw"]) == 0]
put("NumRegionsZero", len(zero))
e = pd.read_csv(EMP / "eras_summary.csv")
e1 = e[(e["era"] == "1930-1959") & (e["costs"] == "gross")].iloc[0]
e2g = e[(e["era"] == "1960-1989") & (e["costs"] == "gross")].iloc[0]
e2n = e[(e["era"] == "1960-1989") & (e["costs"] == "net")].iloc[0]
e3 = e[(e["era"] == "1990-2026") & (e["costs"] == "gross")].iloc[0]
put("NumEraOneSig", int(e1["n_sig_raw"]))
put("NumEraOneNetSig", int(e[(e["era"] == "1930-1959") & (e["costs"] == "net")].iloc[0]["n_sig_raw"]))
put("NumEraTwoNetStepm", int(e2n["n_sig_stepm"]))
put("NumEraOnePmax", pv(e1["p_maxt"]))
put("NumEraOnePavg", pv(e1["p_avgt"]))
put("NumEraTwoSig", int(e2g["n_sig_raw"]))
put("NumEraTwoNetSig", int(e2n["n_sig_raw"]))
put("NumEraTwoNetPavg", pv(e2n["p_avgt"]))
put("NumEraThreeSig", int(e3["n_sig_raw"]))
put("NumEraThreeMaxT", neg(f"{e3['max_t']:.2f}"))
# signals executed one day late
def srow(sample, costs):
    return s[(s["sample"] == sample) & (s["costs"] == costs)].iloc[0]


lag = s[s["costs"] == "gross_lag"]
put("NumUSlagSig", int(srow("us_1926", "gross_lag")["n_sig_raw"]))
put("NumUSlagBestT", neg(f"{srow('us_1926', 'gross_lag')['best_t']:.2f}"))
put("NumAPlagSig", int(srow("asia_pac_ex_japan", "gross_lag")["n_sig_raw"]))
put("NumAPlagBestT", neg(f"{srow('asia_pac_ex_japan', 'gross_lag')['best_t']:.2f}"))
put("NumAPgrossBestT", f"{ap_g['best_t']:.2f}")
put("NumAPlagNetSig", int(srow("asia_pac_ex_japan", "net_lag")["n_sig_raw"]))
put("NumDVlagSig", int(srow("developed", "gross_lag")["n_sig_raw"]))
put("NumLagMaxSig", int(lag["n_sig_raw"].max()))            # most significant designs in any series
put("NumLagBestT", neg(f"{lag['best_t'].max():.2f}"))       # largest t-statistic in any series
put("NumLagMinPmax", pv(lag["p_maxt"].min()))
put("NumLagSeries", len(lag))
put("NumUSlagTurnover", f"{srow('us_1926', 'gross_lag')['turnover_ann'] / 2:.0f}")
e2l = e[(e["era"] == "1960-1989") & (e["costs"] == "gross_lag")].iloc[0]
put("NumEraTwoLagSig", int(e2l["n_sig_raw"]))
put("NumEraTwoLagMaxT", neg(f"{e2l['max_t']:.2f}"))
put("NumEraTwoBreakeven", f"{e2g['breakeven_bp']:.0f}")
put("NumEraTwoMaxT", f"{e2g['max_t']:.2f}")
put("NumCostBp", "10")
# first version of the study (42 moving-average and 169 channel rules; see DECISIONS.md); those result
# files were replaced when the rule universe was enlarged, so the two numbers are fixed here
put("NumOldRules", "211")
put("NumOldSizeBestNaive", "16.2")
cl = pd.read_csv(EMP / "classic.csv")
clu = cl[cl["sample"] == "us_1926"].iloc[0]
put("NumUSclassicRule", str(clu["best_rule"]))
put("NumUSclassicT", f"{clu['t']:.2f}")
put("NumUSclassicPrc", pv(clu["p_rc"]))

out = ROOT / "paper" / "numbers.tex"
out.write_text("% Generated by make_numbers.py; do not edit.\n" +
               "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in macros.items()))
missing = [k for k, v in macros.items() if v == "??"]
print(f"wrote {len(macros)} macros; missing: {missing}")
