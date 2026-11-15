#!/usr/bin/env bash
# Full Monte Carlo study (about 75 minutes on 4 cores). Pass --skip-existing to resume.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=results/mc
run() {  # run unless the output already exists and --skip-existing was given
  local tag=$1; shift
  if [[ "${SKIP:-0}" == 1 && -f "$OUT/meta_${tag}.json" ]]; then echo "skip $tag"; return; fi
  python3 experiments/mc_run.py "$@" --out $OUT
}
[[ "${1:-}" == "--skip-existing" ]] && SKIP=1
run garch_t_p0_s101 --dgp garch_t --reps 5000 --seed 101
run iid_t_p0_s102   --dgp iid_t   --reps 5000 --seed 102
# GARCH null with a 6% annual premium: what an equity premium does to the test against buy-and-hold
run garch_premium_p0.06_s103 --dgp garch_premium --premium-annual 0.06 --reps 2000 --seed 103
for phi in 0.02 0.04 0.06 0.08 0.1 0.12 0.15; do
  run ar1_p${phi}_s201 --dgp ar1 --phi $phi --reps 2000 --seed 201
done
for mu in 0.1 0.2 0.3 0.4 0.5; do
  run regime_p${mu}_s301 --dgp regime --mu-annual $mu --reps 2000 --seed 301
done
# warp-speed check: full bootstrap (B = 299) for both nulls
python3 experiments/mc_fullboot.py --dgp garch_t --reps 400 --B 299 --seed 777
python3 experiments/mc_fullboot.py --dgp iid_t   --reps 400 --B 299 --seed 778
python3 experiments/mc_fullboot.py --dgp garch_t --reps 800 --B 299 --seed 779   # extra replications, pooled
python3 experiments/mc_fullboot.py --dgp iid_t   --reps 800 --B 299 --seed 780
# the same check for power, at the two alternative points quoted in the text
python3 experiments/mc_fullboot.py --dgp ar1    --phi 0.1       --reps 1000 --B 299 --seed 781
python3 experiments/mc_fullboot.py --dgp regime --mu-annual 0.3 --reps 1000 --B 299 --seed 782
python3 experiments/mc_aggregate.py
