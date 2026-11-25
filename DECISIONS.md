# Decision log

Every choice that affects a result in the paper, in the order it was made, and every change made after
first results had been seen. Times are UTC.

## Fixed before any result

**2026-11-07 02:53. The walk-forward family and the grid of 56 designs.**
Test-block lengths h of 21, 63, 126 and 252 days; gaps g of 0 and 21 days; seven training-window
settings (rolling 252, 504 and 1,008 days; growing-rolling from the same three lengths with growth h/2
per block; expanding). All designs are scored on the same out-of-sample days and each day is scored once.
The grid has not changed since.

**2026-11-07 02:53 to 03:06. The test and the procedures.**
Newey-West t-statistic of the strategy's return minus buy-and-hold, one-sided 5%. Across designs: naive
best, Bonferroni, Deflated Sharpe Ratio, bootstrap max-t (studentised Reality Check) and bootstrap
average-t, with a joint stationary bootstrap and the Politis-White block length. Warp-speed Monte Carlo
with a full-bootstrap check. Selection by in-sample Sharpe ratio before costs. Costs of 10 basis points
per unit of position change in the empirical section.

**2026-11-07. Null processes.**
Zero-mean GARCH(1,1) with Student-t shocks fitted to daily US market returns from July 1990 to September
2026, and i.i.d. Student-t returns with the same variance and tails. Seeds are fixed in every script.

## Changed after first results

**2026-11-07, before the first full run. Range of the alternatives widened (power only).**
A pilot with AR(1) coefficients up to 0.06 and trend drifts up to 20% a year gave power too low to
separate the procedures. The ranges were widened to 0.15 and 50%. The pilot was not stored. Size results
are not affected.

**2026-11-07. First full run, 211 rules.**
Universe: 42 moving-average crossovers and 169 channel rules. GARCH null, best of 56 designs: 16.2%;
single design 4.3% to 5.1%; bootstrap max-t 3.5% and average-t 5.2%. US data: 55 of 56 designs
significant before costs, none after costs. These result files were replaced by the next entry; the two
figures quoted in the paper (211 rules, 16.2%) are fixed in `paper/make_numbers.py`.

**2026-11-08. Rule universe replaced (351 rules).**
The universe was replaced by four of the five families of Sullivan,
Timmermann and White (1999): filter (93), moving average (116), support and resistance (70) and channel
breakout (72). The old channel rules were in fact a stochastic oscillator. Designs, tests and seeds
unchanged; every simulation and the empirical section were rerun. GARCH null, best of 56 designs: 20.4%
(was 16.2%). US data: 36 of 56 designs significant before costs (was 55), none after costs.

**2026-11-08. Tie-breaking made exact.**
In-sample Sharpe ratios are rounded to ten decimals before the arg-max so that exactly tied rules are
broken by rule order and not by floating-point noise.

**2026-11-08. More full-bootstrap replications.**
The first 400 full-bootstrap replications of the GARCH null gave 2.0% and 2.2%, which looked low. 800
more replications per null were added with new seeds and all replications are pooled (1,200 per null).
No replication was dropped.

**2026-11-10. Revision.**

1. *Cash benchmark added next to buy-and-hold.* Every Monte Carlo replication now also tests the
   strategy's own excess return (own block length, own bootstrap draw, taken after the buy-and-hold draw
   so that the buy-and-hold statistics are the same with and without it). Result: the naive best of 56
   designs rejects in 47.8% of GARCH null markets against cash, against
   20.5% against buy-and-hold; the design statistics have a mean pairwise correlation of
   0.38 against cash and 0.79 against buy-and-hold. Bootstrap max-t and average-t
   against cash: 3.2% and 6.0% (GARCH), 2.9% and
   4.8% (i.i.d.), warp-speed; full bootstrap 4.0% and
   4.5% (GARCH), 4.2% and 3.8% (i.i.d.).
2. *GARCH null with an equity premium* (6% a year, 2,000 markets, seed 103), to show
   what a premium does to the test against buy-and-hold: single design 0.3%, best of
   56 3.1%.
3. *Three published effective numbers of tests added* (Nyholt 2004, Li and Ji 2005, Galwey 2009) next to
   the participation ratio. Averages over the GARCH null: 1.6 (participation ratio),
   11.9 (Galwey), 12.1 (Li and Ji), 21.6 (Nyholt). Bonferroni fed with
   each: 14.6%, 2.9%, 2.8% and 1.4%;
   with N = 56, 0.6%.
4. *One-day execution lag in the empirical section.* Every position is entered one day after its
   signal, in the training windows and in the test blocks. The lagged runs use a separate random-number
   stream (seed 20261110) and are computed after the main results, so every earlier empirical number is
   unchanged to the last digit. Result: no design is significant before costs in any of the seven
   series (US: 36 of 56 without the lag, 0 with it; best t-statistic
   3.65 and -1.83). US 1960 to 1989: 56 of 56 without the lag,
   0 with it; break-even cost 19 basis points per unit traded.
5. *Simulated shocks truncated at 50% a day in absolute value.* The fat-tailed, highly persistent
   GARCH process produced, in a handful of the 5,000 null markets, daily returns that no price can
   follow (2 markets with a return below -100% and 38 with a return beyond 50% in absolute value). The
   truncation is symmetric, so the mean stays zero, and it enters the variance recursion. It binds on
   69 days in 38 of the 5,000 GARCH null markets. Simulating log returns was
   considered and rejected: with Student-t shocks the implied simple returns have a positive mean, which
   would break the zero-mean null. Effect on the headline numbers: best of 56 designs 20.5%
   (20.4% before), single design 4.8% (4.8%), bootstrap max-t 3.7% (3.7%) and
   average-t 5.0% (5.0%).
6. *Full-bootstrap check extended to power* at the two alternatives quoted in the text (1,000
   replications each, seeds 781 and 782): max-t and average-t reject in 42.8% and
   39.0% at an AR(1) coefficient of 0.10 (warp-speed: 41.2% and 39.1%)
   and in 38.0% and 38.6% at a trend drift of 30% (warp-speed:
   39.9% and 40.7%).
7. *Tables typeset in LaTeX* from the result files (`tables/tex/`) instead of being printed to PDF from
   HTML. No number changes.

No design, rule, test, seed or sample was changed in this revision other than as listed above.

**2026-11-24. Manuscript revised and moved to the Springer Nature template.** No code, seed, design,
rule, test or result changed, and every number in the text is unchanged.

1. The abstract and introduction state the relation to Rossi and Inoue (2012), Hansen and Timmermann
   (2012) and the Reality Check explicitly.
2. The shared-benchmark effect, previously a footnote, is a main result: Proposition 1 in Section 6.2
   gives the correlation of two strategies against cash and against buy-and-hold, with the
   equity-premium result next to it.
3. Section 3.4 describes the computation (constant-time window statistics, joint resampling,
   warp-speed Monte Carlo); the timing quoted is the stored run time of the GARCH null
   (`results/mc/meta_garch_t_p0_s101.json`, 510 seconds).
4. Explanations of basic terms were shortened.
5. The paper uses the Springer Nature class `sn-jnl` (`sn-basic` reference style). Tables are set in
   footnote size to fit its text width. `paper/make_submission.py` writes a single-file version of the
   manuscript to `paper/submission/`.

