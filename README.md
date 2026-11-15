# Replication package

**How Many Ways to Walk Forward? Design Snooping in Trading-Rule Backtests**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24237608.svg)](https://doi.org/10.5281/zenodo.24237608)

This repository reproduces every number, table and figure in the paper, and the paper itself
(`paper/main.pdf`).

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11+
npm install playwright@1 && npx playwright install chromium   # only if Playwright is not installed
bash run_all.sh            # full run from the stored data vintage: tests, simulations, empirics, figures, PDF
SKIP_MC=1 bash run_all.sh  # reuse the stored Monte Carlo output (a few minutes)
FETCH=1 bash run_all.sh    # download a fresh data vintage first (numbers may then differ slightly)
```

`run_all.sh` runs, in order: the test suite, `data/prepare.py` (and `data/download.py` with `FETCH=1`),
`experiments/calibrate_garch.py`, the Monte Carlo (`experiments/run_mc_all.sh`,
which ends with `experiments/mc_fullboot.py` and `experiments/mc_aggregate.py`), `experiments/empirical.py`,
`figures/build_figures.py`, `tables/build_tables.py`, `figures/render.js`, and finally
`paper/make_numbers.py` and `latexmk`. Every number quoted in the text is written into
`paper/numbers.tex` by `make_numbers.py`; none is typed by hand. The two exceptions are the two figures
of the first version of the study (211 rules), which are fixed in `make_numbers.py` and documented in
`DECISIONS.md`. `DECISIONS.md` lists, with dates, every choice that affects a result and
every change made after first results.

## Layout

| Path | Content |
|---|---|
| `src/wfo/splits.py` | The unified walk-forward family (Definition 1) and the 56-design grid |
| `src/wfo/rules.py` | The 351 rules: filter, moving-average, support-and-resistance and channel-breakout families of Sullivan, Timmermann and White (1999); causal positions (numba kernels) |
| `src/wfo/backtest.py` | Walk-forward meta-strategy (Algorithm 1), O(1) window statistics on centred returns, explicit tie rule, turnover |
| `src/wfo/dgp.py` | Null (i.i.d.-t, GARCH-t, GARCH-t with a premium) and alternative (AR(1), Markov trend) return processes; shocks truncated at +/-50% a day |
| `src/wfo/stats.py` | Newey-West t, stationary bootstrap, Politis-White block length, Romano-Wolf/Reality Check, four effective numbers of designs (participation ratio, Nyholt, Li-Ji, Galwey), DSR, PBO |
| `src/wfo/pbo_fast.py` | Fast CSCV probability of backtest overfitting (identical to the reference version) |
| `tests/` | Unit tests: tiling and no-overlap of every design, causality and behaviour of rules, window statistics and tie-breaking, bootstrap, PBO, effective numbers, shock truncation, execution lag |
| `data/` | Download and preparation scripts, raw files, processed series, `PROVENANCE.md` with hashes |
| `experiments/` | GARCH calibration, Monte Carlo drivers and aggregation (tests against buy-and-hold and against cash), empirical illustration (with and without a one-day execution lag) |
| `results/` | All outputs: `calibration/`, `mc/` (see `mc/SCHEMA.md`), `empirical/` (per-design results, dependence summaries, rule-selection counts) |
| `figures/` | `build_figures.py` writes HTML+SVG pages to `html/`; `render.js` prints them to vector PDFs in `pdf/` |
| `tables/` | `build_tables.py` writes each table as LaTeX (`tex/`, read by `paper/main.tex`) and as CSV (`csv/`) |
| `paper/` | `main.tex`, `references.bib` (Crossref-verified), `numbers.tex`, `main.pdf` |
| `DECISIONS.md` | Dated log of the choices that affect results and of the changes made after first results |

## Figures and tables

Figures are written as HTML and CSS (inline SVG for charts) and printed to PDF with headless Chromium
at their exact print size (624 px = 6.5 in, the text width), so text and lines stay vector and keep
their designed sizes. Fonts (Source Sans Pro, STIX Two Text; SIL Open Font License) are bundled in
`figures/fonts/`. Tables are native LaTeX: `tables/build_tables.py` writes one `tabular` with its notes
per table into `tables/tex/`, and `paper/main.tex` reads them with `\input`; the same cells are in
`tables/csv/`.

## Data

All data are public: the Kenneth R. French Data Library (US market since July 1926, five regions since
July 1990) and FRED (S&P 500, last ten years). See `data/PROVENANCE.md` for URLs, retrieval dates and
SHA-256 hashes. The Data Library is revised monthly; the stored raw files are the vintage used in the
paper (built from the September 2026 CRSP and Bloomberg databases).

## Computing time

On a 4-core machine: Monte Carlo about 75 minutes, empirical illustration about 6 minutes,
figures, tables and paper under 1 minute. The rule kernels are compiled with numba on first use. Seeds are fixed in every script.
