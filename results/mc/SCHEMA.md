# Monte Carlo outputs

Produced by `experiments/run_mc_all.sh` (calls `mc_run.py`), aggregated by `mc_aggregate.py`;
`mc_fullboot.py` checks the warp-speed bootstrap shortcut.

Common setup: T = 5292 simulated days per replication; burn-in 250 (longest rule look-back); every design is
evaluated on the same out-of-sample span [1279, 5292), n = 4013 days. GARCH parameters come from
`results/calibration/garch_us_market.json` (source recorded in each `meta_*.json`).
Test statistic: Newey-West t of d_t = r_strategy,t - r_buy&hold,t, H0: E[d] <= 0, one-sided 5% (z = 1.645).
Every statistic is also computed with cash as the benchmark (the strategy's own excess return in place of d);
those columns and procedures carry the suffix `_cash`.
Simulated shocks are truncated at +/-50% a day (`max_abs_shock` in each `meta_*.json`), which keeps simulated
prices positive; `n_capped` records how often the truncation binds.

## designs_<dgp>_p<param>_s<seed>.csv.gz (one row per replication x design)
| column | meaning |
|---|---|
| dgp | iid_t, garch_t (nulls); garch_premium (GARCH null plus a constant annual premium); ar1, regime (alternatives) |
| param | phi for ar1; annual drift magnitude for regime; annual premium for garch_premium; 0 for nulls |
| rep | replication index |
| label | design id, e.g. `R504|h63|g0` (rolling 504), `G504+31|h63|g0` (growing), `EXP|h63|g21` (expanding) |
| scheme, h, L0, G, g | rolling/growing/expanding, test length, initial training length, growth per block, gap (days) |
| t | Newey-West t-statistic of d |
| t_star | recentred bootstrap t* from ONE joint stationary-bootstrap draw (same indices for all designs) |
| t_cash, t_cash_star | the same two statistics with cash as the benchmark (own block length, own draw, taken after the buy-and-hold draw) |
| is_mean | mean in-sample daily return of the rules selected in the training windows |
| oos_mean | mean out-of-sample daily return of the walk-forward strategy |
| sharpe_oos | daily Sharpe ratio of the strategy |
| d_sharpe, d_skew, d_kurt | daily Sharpe, skewness and (non-excess) kurtosis of d (DSR inputs) |
| n_tied | training windows of the design in which the best in-sample Sharpe ratio was tied (ties go to rule order) |
| distinct_rules | number of distinct rules the design selected over the span |

## rules_<...>.csv.gz (one row per replication)
`max_t_rules`: largest t among the 351 rules backtested over the whole span without walk-forward;
`max_tstar_rules`: its bootstrap counterpart (same draw); `best_rule`: index into `wfo.rules.rule_names()`;
`block`: Politis-White mean block length; `mean_corr_d`, `n_eff_d`: mean pairwise correlation of the 56 daily
series d and the participation-ratio effective number of designs within the replication;
`n_eff_liji_d`, `n_eff_galwey_d`, `n_eff_nyholt_d`: the counts of Li and Ji (2005), Galwey (2009) and Nyholt
(2004) from the same correlation matrix; `mean_corr_cash`, `n_eff_cash`, `block_cash`: mean correlation,
participation ratio and block length for the 56 strategy return series (cash benchmark);
`max_abs_ret`, `n_capped`: largest absolute simulated return and number of days on which the shock
truncation binds; `n_rules`: 351.

## perrep_<...>.csv.gz (one row per replication, all 56 designs)
Reject flags (0/1) of `best_naive`, `best_naive_sizeadj` (oracle critical value from the GARCH null),
`boot_max`, `boot_avg`, `bonferroni`, `bonferroni_eff`, `dsr`, `dsr_eff` (the `_eff` versions use the
within-replication participation ratio `n_eff_d`), `rules_best_naive`, `rules_reality_check`, the cash-benchmark
flags `best_naive_cash`, `boot_max_cash`, `boot_avg_cash`, plus `max_t`, `avg_t`, `max_t_cash`, `avg_t_cash`,
`n_eff_d`, `mean_corr_d`. Bootstrap flags use warp-speed critical values computed across
the replications of the same file.

## meta_<...>.json: run settings, GARCH parameters and source, run time.

## summary_by_design.csv
Rejection rate of each design used alone (`reject`, `mc_se`), mean t, annualised in-sample (`is_ann`) and
out-of-sample (`oos_ann`) strategy returns, mean number of tied training windows and of distinct rules selected.

## summary_procedures.csv
Rejection rate (`reject`, `mc_se`) of each procedure for K designs, K = 1, 2, 4, 8, 16, 32, 56
(averaged over 25 random subsets of designs, except K = 56): `single`, `best_naive`, `boot_max`, `boot_avg`,
`bonferroni`, `dsr`, and with cash as the benchmark `single_cash`, `best_naive_cash`, `boot_max_cash`,
`boot_avg_cash`; at K = 56 only: `best_naive_sizeadj`, `bonferroni_eff`, `dsr_eff` (participation ratio) and
`bonferroni_liji`, `bonferroni_galwey`, `bonferroni_nyholt` (Bonferroni fed with each published eigenvalue
count, estimated within each replication); and for K = 351 rules: `rules_best_naive`, `rules_reality_check`.
Under a null DGP `reject` is the false-positive rate (size); under an alternative it is power.
This file is also the FPR-vs-K curve.

## summary_is_oos.csv
In-sample vs out-of-sample annualised returns of the selected rules, by scheme.

## summary_dependence.csv (one row per DGP setting)
`mean_corr_t`, `min_corr_t`, `max_corr_t`: correlation across replications of the 56 design t-statistics;
`n_eff_t`: participation-ratio effective number of designs from that matrix (`n_eff_liji_t`, `n_eff_galwey_t`,
`n_eff_nyholt_t`: the three published counts from the same matrix); `fpr_best_naive` and
`n_indep_tests` = log(1 - FPR)/log(0.95), the number of independent 5% tests whose maximum rejects as often;
`mean_corr_d`, `n_eff_d_mean`, `n_eff_d_median`: within-replication versions from the daily d series
(`n_eff_liji_d_mean`, `n_eff_galwey_d_mean`, `n_eff_nyholt_d_mean` for the published counts);
`mean_corr_t_cash`, `min_corr_t_cash`, `n_eff_t_cash`, `fpr_best_naive_cash`, `n_indep_tests_cash`,
`mean_corr_cash`: the same summaries with cash as the benchmark;
`tied_windows_share`: share of training windows with a tied best Sharpe ratio;
`capped_days`, `capped_reps`, `max_abs_ret`: days and replications in which the shock truncation binds, and
the largest absolute simulated return.

## corr_t_<dgp>.csv and corr_t_cash_<dgp>.csv
The 56 x 56 correlation matrix of the design t-statistics under each null (rows and columns are design labels),
against buy-and-hold and against cash.

## fullboot_<...>.json
Full-bootstrap (B = 299 per replication) rejection rates of boot_max and boot_avg over all 56 designs,
against buy-and-hold (`reject_max`, `reject_avg`) and against cash (`reject_max_cash`, `reject_avg_cash`):
size under the two nulls, power under the AR(1) alternative with phi = 0.10 and the trend alternative with an
annual drift of 30%.
