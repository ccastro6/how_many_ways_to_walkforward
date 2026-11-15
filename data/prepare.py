"""Turn the raw downloads into tidy daily return series.

Output: data/processed/<series>.csv with columns
    date   : trading day (YYYY-MM-DD)
    ret    : simple daily return (decimal), total return in USD
    rf     : simple daily risk-free return (decimal; 1-month T-bill based)
    price  : total-return index, 100 on the day before the first return

Series
    us_market     US value-weighted market (CRSP), Jul 1926 onward; Mkt-RF + RF
    developed     Developed markets, Jul 1990 onward
    north_america, europe, japan, asia_pac_ex_japan   (same source, Jul 1990 onward)
    sp500_price   S&P 500 price index from FRED (last 10 years, no dividends)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RAW, OUT = HERE / "raw", HERE / "processed"

KF_FILES = {
    "us_market": "F-F_Research_Data_Factors_daily.csv",
    "developed": "Developed_3_Factors_Daily.csv",
    "north_america": "North_America_3_Factors_Daily.csv",
    "europe": "Europe_3_Factors_Daily.csv",
    "japan": "Japan_3_Factors_Daily.csv",
    "asia_pac_ex_japan": "Asia_Pacific_ex_Japan_3_Factors_Daily.csv",
}


def read_kf(path: Path) -> pd.DataFrame:
    """Parse a Ken French daily CSV: keep rows whose first field is an 8-digit date."""
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 8:
            rows.append((parts[0], float(parts[1]), float(parts[-1])))
    df = pd.DataFrame(rows, columns=["date", "mkt_rf", "rf"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    bad = (df["mkt_rf"] <= -99.98) | (df["rf"] <= -99.98)
    if bad.any():
        raise ValueError(f"{path.name}: {bad.sum()} missing values")
    df["ret"] = (df["mkt_rf"] + df["rf"]) / 100.0
    df["rf"] = df["rf"] / 100.0
    return df[["date", "ret", "rf"]]


def read_fred_sp500(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = ["date", "close"]
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["ret"] = df["close"].pct_change()
    df["rf"] = np.nan
    return df.iloc[1:][["date", "ret", "rf"]].reset_index(drop=True)


def finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["price"] = 100.0 * np.cumprod(1.0 + df["ret"].to_numpy())
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def main():
    OUT.mkdir(exist_ok=True)
    summary = []
    series = {k: read_kf(RAW / v) for k, v in KF_FILES.items()}
    series["sp500_price"] = read_fred_sp500(RAW / "SP500_fred.csv")
    for name, df in series.items():
        df = finish(df)
        df.to_csv(OUT / f"{name}.csv", index=False, float_format="%.8g")
        r = df["ret"].to_numpy()
        summary.append(dict(series=name, first=df["date"].iloc[0], last=df["date"].iloc[-1], n=len(df),
                            ann_mean=252 * r.mean(), ann_vol=np.sqrt(252) * r.std(ddof=1),
                            min=r.min(), max=r.max()))
    s = pd.DataFrame(summary)
    s.to_csv(OUT / "summary.csv", index=False, float_format="%.6g")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
