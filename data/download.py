"""Download the raw data used in the empirical illustration.

All series are public:
  * Kenneth R. French Data Library (Tuck School of Business, Dartmouth):
    daily Fama-French factors for the US (from July 1926) and daily 3-factor
    files for Developed, North America, Europe, Japan and Asia Pacific ex Japan
    (from July 1990). Market return = Mkt-RF + RF.
  * FRED (Federal Reserve Bank of St. Louis), series SP500: S&P 500 price index,
    last ten years only (the licence of S&P Dow Jones Indices limits FRED to ten years).

The Data Library is revised monthly, so hashes recorded in PROVENANCE.md
identify the vintage used in the paper. Run from the ``replication`` folder:

    python data/download.py
"""
from __future__ import annotations

import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw"
KF = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FILES = {
    "F-F_Research_Data_Factors_daily_CSV.zip": KF + "F-F_Research_Data_Factors_daily_CSV.zip",
    "Developed_3_Factors_Daily_CSV.zip": KF + "Developed_3_Factors_Daily_CSV.zip",
    "North_America_3_Factors_Daily_CSV.zip": KF + "North_America_3_Factors_Daily_CSV.zip",
    "Europe_3_Factors_Daily_CSV.zip": KF + "Europe_3_Factors_Daily_CSV.zip",
    "Japan_3_Factors_Daily_CSV.zip": KF + "Japan_3_Factors_Daily_CSV.zip",
    "Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip": KF + "Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip",
    "SP500_fred.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500",
}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    log = {}
    for name, url in FILES.items():
        with urllib.request.urlopen(url, timeout=60) as r:
            b = r.read()
        (RAW / name).write_bytes(b)
        if name.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                z.extractall(RAW)
        log[name] = dict(url=url, sha256=sha256(b), bytes=len(b),
                         retrieved_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        print(f"{name:50s} {len(b):>9d} bytes  {log[name]['sha256'][:16]}")
    (RAW / "download_log.json").write_text(json.dumps(log, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
