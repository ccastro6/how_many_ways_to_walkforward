"""Calibrate the zero-mean GARCH(1,1) null to US daily market returns.

The null model of the Monte Carlo keeps the volatility clustering and fat tails of
real equity returns but has no predictable mean, so no trading rule has skill.
We fit, by Gaussian-t maximum likelihood (arch package),

    r_t = sigma_t z_t,   sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2,
    z_t ~ standardised Student-t(nu),

to daily simple returns of the US value-weighted market (Ken French), July 1990 to
the end of the data. This span matches the international series and the modern,
decimalised market; the long 1926 sample is used only in the illustration.
The S&P 500 itself is not available as a long free daily series;
the CRSP value-weighted market is its closest public proxy.

Writes results/calibration/garch_us_market.json (parameters on the decimal scale).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model

ROOT = Path(__file__).resolve().parents[1]
START = "1990-07-02"


def main():
    df = pd.read_csv(ROOT / "data/processed/us_market.csv")
    df = df[df["date"] >= START]
    r = df["ret"].to_numpy()
    am = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
    res = am.fit(disp="off")
    p = res.params
    out = dict(
        series="us_market", start=df["date"].iloc[0], end=df["date"].iloc[-1], n=int(len(r)),
        omega=float(p["omega"]) / 1e4, alpha=float(p["alpha[1]"]), beta=float(p["beta[1]"]),
        nu=float(p["nu"]),
        se=dict(omega=float(res.std_err["omega"]) / 1e4, alpha=float(res.std_err["alpha[1]"]),
                beta=float(res.std_err["beta[1]"]), nu=float(res.std_err["nu"])),
        loglik=float(res.loglikelihood),
        sample_ann_vol=float(np.sqrt(252) * r.std(ddof=1)),
    )
    out["persistence"] = out["alpha"] + out["beta"]
    out["implied_ann_vol"] = float(np.sqrt(252 * out["omega"] / (1 - out["persistence"])))
    dest = ROOT / "results/calibration/garch_us_market.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
