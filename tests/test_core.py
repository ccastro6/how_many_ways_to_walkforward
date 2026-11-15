import numpy as np
import pytest

from wfo.splits import WFConfig, blocks, default_grid
from wfo.rules import positions, rule_returns, BURN_IN, rule_names, N_RULES, RULES, family_counts
from wfo.backtest import run_grid, WindowStats, select
from wfo.dgp import simulate, prices
from wfo.stats import nw_tstat, stationary_indices, pbo_cscv, deflated_sharpe


START, END, ORIGIN = 1300, 5000, BURN_IN


@pytest.mark.parametrize("cfg", default_grid())
def test_blocks_tile_span_without_overlap(cfg):
    bl = list(blocks(cfg, START, END, ORIGIN))
    days = np.concatenate([np.arange(b.test_start, b.test_end) for b in bl])
    assert days[0] == START and days[-1] == END - 1
    assert len(days) == len(set(days)) == END - START       # each day scored exactly once
    for b in bl:
        assert b.train_end + cfg.g == b.test_start               # gap respected
        assert b.train_start >= ORIGIN                          # no use of burn-in
        assert b.train_end <= b.test_start                      # training strictly before test


def test_growth_and_expanding():
    g = list(blocks(WFConfig(h=63, L0=252, G=31), START, END, ORIGIN))
    lens = [b.train_end - b.train_start for b in g[:3]]
    assert lens == [252, 283, 314]
    e = list(blocks(WFConfig(h=63, L0=None, expanding=True), START, END, ORIGIN))
    assert all(b.train_start == ORIGIN for b in e)


def test_grid_size():
    grid = default_grid()
    assert len(grid) == 56
    assert len({c.label for c in grid}) == 56


def test_rules_are_causal():
    rng = np.random.default_rng(0)
    r = simulate("iid_t", 800, rng)
    p = prices(r)
    pos = positions(p)
    p2 = p.copy()
    p2[600:] *= 1.5                      # change the future only
    pos2 = positions(p2)
    assert np.array_equal(pos[:600], pos2[:600])
    assert pos.shape[1] == len(rule_names()) == N_RULES == 351
    assert family_counts() == {'filter': 93, 'ma': 116, 'sr': 70, 'cb': 72}
    assert len(set(rule_names())) == N_RULES


def test_rule_returns_alignment():
    rng = np.random.default_rng(1)
    r = simulate("iid_t", 600, rng)
    R, pos = rule_returns(prices(r), r)
    assert np.allclose(R[1:], pos[:-1] * r[1:, None])


def test_oos_mean_zero_under_null_on_average():
    rng = np.random.default_rng(2)
    ts = []
    for _ in range(30):
        r = simulate("iid_t", 3000, rng)
        R, pos = rule_returns(prices(r), r)
        oos, held, _, _ = run_grid(R, pos, [WFConfig(h=63, L0=504)], 1300, 3000, BURN_IN)
        ts.append(nw_tstat(oos)[0])
    assert abs(np.mean(ts)) < 0.6


def test_stationary_indices_range():
    idx = stationary_indices(1000, 10.0, np.random.default_rng(3))
    assert idx.min() >= 0 and idx.max() < 1000


def test_pbo_and_dsr_ranges():
    rng = np.random.default_rng(4)
    M = rng.standard_normal((1600, 20)) * 0.01
    out = pbo_cscv(M, S=8)
    assert 0 <= out["pbo"] <= 1
    d = deflated_sharpe(M[:, 0], M.mean(0) / M.std(0))
    assert 0 <= d <= 1


def test_positions_defined_after_burn_in_and_in_range():
    rng = np.random.default_rng(5)
    r = simulate("garch_t", 2000, rng)
    pos = positions(prices(r))
    assert set(np.unique(pos)) <= {-1.0, 0.0, 1.0}
    # every rule family takes both long and short positions at some point
    fam = np.array([q.family for q in RULES])
    for f in ("filter", "ma", "sr", "cb"):
        sub = pos[BURN_IN:, fam == f]
        assert (sub == 1).any() and (sub == -1).any()


def test_filter_rule_reverses_at_x():
    # a 1% filter rule (log scale) goes long after a rise of 1% above the low, short after a fall of 1% from the high
    p = np.array([100, 100.5, 101.1, 102, 102.5, 101.5, 100.4, 100.0, 101.5])
    pos = positions(np.r_[np.full(300, 100.0), p])[-len(p):]
    j = rule_names().index("F(0.01)")
    assert list(pos[:, j]) == [0, 0, 1, 1, 1, 1, -1, -1, 1]


def test_window_stats_match_direct_computation_and_ties_go_to_rule_order():
    rng = np.random.default_rng(6)
    R = rng.standard_normal((3000, 7)) * 0.01
    R[:, 3] = R[:, 1]                      # an exact duplicate of rule 1 with a higher index
    ws = WindowStats(R)
    for a, b in ((10, 300), (1000, 2500), (0, 3000)):
        m, sh = ws.mean_sharpe(a, b)
        assert np.allclose(m, R[a:b].mean(0), atol=1e-14)
        assert np.allclose(sh, R[a:b].mean(0) / R[a:b].std(0), atol=1e-10)
        j, tied = select(sh)
        direct = np.round(R[a:b].mean(0) / R[a:b].std(0), 10)
        assert j == int(np.argmax(direct))
        if j == 1:
            assert tied and j < 3


def test_effective_counts_at_the_two_extremes():
    from wfo.stats import effective_counts, effective_number
    K = 12
    indep = effective_counts(np.ones(K))                       # identity correlation matrix
    ident = effective_counts(np.r_[float(K), np.zeros(K - 1)])  # all series identical
    for key in ("n_eff", "n_eff_liji", "n_eff_galwey", "n_eff_nyholt"):
        assert indep[key] == pytest.approx(K)
        assert ident[key] == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    common = rng.standard_normal(4000)
    d = 0.8 * common + 0.6 * rng.standard_normal((K, 4000))    # pairwise correlation 0.64
    out = effective_number(d)
    assert out["mean_corr"] == pytest.approx(0.64, abs=0.03)
    assert 1.0 < out["n_eff"] < out["n_eff_nyholt"] < K        # the counts disagree, in a known order


def test_simulated_shocks_are_truncated_and_keep_prices_positive():
    from wfo.dgp import GarchParams, MAX_ABS_SHOCK, garch_noise, std_t
    wild = GarchParams(omega=1e-4, alpha=0.5, beta=0.49, nu=2.5)     # explosive on purpose
    for model in ("garch_t", "iid_t", "ar1", "regime"):
        r = simulate(model, 5000, np.random.default_rng(11), garch=wild, phi=0.1, mu=0.001)
        if model in ("garch_t", "iid_t"):
            assert np.abs(r).max() <= MAX_ABS_SHOCK
    r = simulate("garch_t", 5000, np.random.default_rng(11), garch=wild)
    assert np.abs(r).max() == MAX_ABS_SHOCK and prices(r).min() > 0   # the cap binds and prices survive
    # when the cap does not bind the path is the plain GARCH recursion
    p = GarchParams(omega=1.6e-6, alpha=0.10, beta=0.885, nu=6.0)
    e = garch_noise(np.random.default_rng(5), 300, p, burn=50)
    z = std_t(np.random.default_rng(5), p.nu, 350)
    h, ref = p.uncond_var, np.empty(350)
    for t in range(350):
        ref[t] = np.sqrt(h) * z[t]
        h = p.omega + p.alpha * ref[t] ** 2 + p.beta * h
    assert np.abs(ref).max() < MAX_ABS_SHOCK and np.allclose(e, ref[50:])


def test_premium_shifts_the_garch_null_by_a_constant():
    a = simulate("garch_t", 500, np.random.default_rng(8))
    b = simulate("garch_t", 500, np.random.default_rng(8), drift=0.06 / 252)
    assert np.allclose(b - a, 0.06 / 252)


def test_execution_lag_shifts_every_position_by_one_day():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from empirical import lagged
    rng = np.random.default_rng(2)
    ex = rng.standard_normal(60) * 0.01
    pos = rng.choice([-1.0, 0.0, 1.0], size=(60, 5))
    R_l, pos_l = lagged(pos, ex)
    assert np.array_equal(pos_l[1:], pos[:-1]) and not pos_l[0].any()
    # the return of day t uses the position decided at the close of day t-2
    assert np.allclose(R_l[2:], pos[:-2] * ex[2:, None]) and not R_l[:2].any()
