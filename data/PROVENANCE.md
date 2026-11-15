# Data provenance

All raw files were downloaded by `data/download.py` on the date shown (UTC). The Kenneth R. French Data Library is revised monthly, so the SHA-256 hashes identify the exact vintage used in the paper (files built from the September 2026 CRSP and Bloomberg databases). `data/prepare.py` turns them into `data/processed/*.csv`.

| File | Source URL | Retrieved (UTC) | SHA-256 |
|---|---|---|---|
| `F-F_Research_Data_Factors_daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip | 2026-11-07T02:58:16+00:00 | `96c23b88e651cbfff3740a53f8225e59ee39893371e96385a1d8125ec2cd9c89` |
| `Developed_3_Factors_Daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Developed_3_Factors_Daily_CSV.zip | 2026-11-07T02:58:17+00:00 | `3a73ae9ffb33ffd53919f51dc855836967db4cc0cff6c76277d5488cfd26203d` |
| `North_America_3_Factors_Daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/North_America_3_Factors_Daily_CSV.zip | 2026-11-07T02:58:17+00:00 | `591d2b0fa3b79284aabbb216ad760e29b3bc415d497c057bc7f936faa7bdd408` |
| `Europe_3_Factors_Daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Europe_3_Factors_Daily_CSV.zip | 2026-11-07T02:58:17+00:00 | `53598ec5af305d321ee93db492a32076fc882f976e67d27eba724d0266d20ad2` |
| `Japan_3_Factors_Daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Japan_3_Factors_Daily_CSV.zip | 2026-11-07T02:58:17+00:00 | `3fd49c8815c017c0b95c0e47eb493369a1a86d38c0925985bea602237e9e8119` |
| `Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip` | https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Asia_Pacific_ex_Japan_3_Factors_Daily_CSV.zip | 2026-11-07T02:58:18+00:00 | `19ebd35c21bd81b14a09c6872b40f694c1c2395911bc10aa0e4dc682d076bbf3` |
| `SP500_fred.csv` | https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500 | 2026-11-07T02:58:18+00:00 | `e3d7566c49052717d5f4afd0dd50c0c7056f6e5859c5dc83d2f71437c17c2c77` |

## Series and definitions

- **us_market**: US value-weighted market return from CRSP (all NYSE, AMEX and NASDAQ common stocks), July 1926 to September 2026, computed as `Mkt-RF + RF` from the daily Fama-French factor file. Total return (dividends included), in percent in the source and converted to decimals.
- **developed, north_america, europe, japan, asia_pac_ex_japan**: regional value-weighted market returns in US dollars from the international daily 3-factor files (Bloomberg data), July 1990 to September 2026, again `Mkt-RF + RF` with the US one-month T-bill as RF.
- **sp500_price**: S&P 500 price index (no dividends) from FRED series SP500. FRED carries only the last ten years under its licence with S&P Dow Jones Indices, so this series covers November 2016 to November 2026 and is used only as a short recent check.

## Sources that could not be used

A long daily S&P 500 history was attempted from Stooq (`^spx`) and Yahoo Finance (`^GSPC`). Neither could be downloaded reliably at retrieval time (Stooq reset every connection and Yahoo returned HTTP 429), so neither is used. The CRSP value-weighted market is the closest public long-history proxy (the S&P 500 covers roughly 80% of its capitalisation).

## Citation

Kenneth R. French, Data Library, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html. Federal Reserve Bank of St. Louis, FRED, series SP500, https://fred.stlouisfed.org/series/SP500. S&P 500 data copyright S&P Dow Jones Indices LLC.
