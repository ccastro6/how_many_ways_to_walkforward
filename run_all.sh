#!/usr/bin/env bash
# Reproduce every number, table and figure of the paper, then compile it.
# Run from the replication folder:  bash run_all.sh            (full run, about 90 minutes on 4 cores)
#                                   SKIP_MC=1 bash run_all.sh  (reuse results/mc, about 10 minutes)
#                                   FETCH=1 bash run_all.sh    (first download a fresh data vintage)
set -euo pipefail
cd "$(dirname "$0")"

echo "== tests";            python3 -m pytest -q
echo "== data";             if [ "${FETCH:-0}" = 1 ]; then python3 data/download.py; fi   # FETCH=1: fresh vintage
                            python3 data/prepare.py
echo "== calibration";      python3 experiments/calibrate_garch.py
if [ "${SKIP_MC:-0}" != 1 ]; then
  echo "== Monte Carlo";    bash experiments/run_mc_all.sh          # includes full-bootstrap check and aggregation
else
  echo "== MC aggregate";   python3 experiments/mc_aggregate.py
fi
echo "== empirical";        python3 experiments/empirical.py
echo "== figures/tables";   python3 figures/build_figures.py
                            python3 tables/build_tables.py
                            node figures/render.js
echo "== paper";            cd paper && python3 make_numbers.py && latexmk -pdf -interaction=nonstopmode -quiet main.tex && python3 make_submission.py
echo "done: paper/main.pdf"
