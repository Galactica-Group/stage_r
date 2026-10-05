"""Tests for the look-3 ETF trend engine (tools/etf_trend_r3.py).

Everything is a constructed fixture with a hand-computable answer, except the
two control tests, which run the real control procedures at reduced R under a
fixed seed so they are deterministic rather than flaky.

The mutation this battery is built to catch is the rule reversed — holding the
ETF when it is BELOW its ten-month average. test_ten_month_rule_hand_checked and
test_rule_beats_timing_null_on_a_trending_fixture are the tests that fail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import etf_trend_r3 as e  # noqa: E402


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def daily_from_monthly(closes, start="2000-01"):
    """One observation per month, on the last day of the month."""
    per = pd.period_range(start=start, periods=len(closes), freq="M")
    idx = pd.DatetimeIndex([p.end_time.normalize() for p in per])
    return pd.Series(np.asarray(closes, dtype=float), index=idx)


def panel_from_monthly(spec, classes, start="2000-01", cash_closes=None, n=None):
    """spec: {ticker: [monthly closes]}. Shorter lists are back-padded with NaN."""
    n = n or max(len(v) for v in spec.values())
    prices = {}
    for t, v in spec.items():
        pad = [np.nan] * (n - len(v))
        prices[t] = daily_from_monthly(pad + list(v), start=start).dropna()
    cc = cash_closes if cash_closes is not None else [100.0] * n
    prices[e.CASH] = daily_from_monthly(cc, start=start)
    return e.build_panel(prices, classes=classes)


# ---------------------------------------------------------------------------
# 1. the window guard
# ---------------------------------------------------------------------------
def test_window_guard_filters_at_read_and_asserts_after(tmp_path):
    d = pd.date_range("2012-11-01", "2013-03-01", freq="D")
    pd.DataFrame({"ticker": "ZZZ", "date": d.astype(str),
                  "closeadj": np.arange(len(d), dtype=float)}
                 ).to_parquet(tmp_path / "etf_ZZZ.parquet")
    out, spans = e.load_fund_prices(tmp_path, ["ZZZ"], "2013-01-01")
    assert out["ZZZ"].index.max() < pd.Timestamp("2013-01-01")
    assert spans["ZZZ"][1] == "2012-12-31"
    assert spans["ZZZ"][3] == "2013-03-01", "the raw span must still be recorded"


def test_window_guard_rejects_a_vendor_span_that_never_reaches_the_edge(tmp_path):
    """A short slice must be distinguishable from a filter doing its job."""
    d = pd.date_range("2010-01-01", "2012-06-30", freq="D")
    pd.DataFrame({"ticker": "ZZZ", "date": d.astype(str),
                  "closeadj": np.ones(len(d))}).to_parquet(tmp_path / "etf_ZZZ.parquet")
    with pytest.raises(AssertionError, match="truncation"):
        e.load_fund_prices(tmp_path, ["ZZZ"], "2013-01-01")


# ---------------------------------------------------------------------------
# 2. the ten-month rule, hand checked
# ---------------------------------------------------------------------------
HAND = [100.0] * 10 + [110.0, 90.0, 105.0, 95.0]
# return month 10: window months 0-9 all 100, mean 100.0, close 100 -> NOT above
# return month 11: window months 1-10 mean 101.0, close 110.0      -> above
# return month 12: window months 2-11 mean 100.0, close  90.0      -> NOT above
# return month 13: window months 3-12 mean 100.5, close 105.0      -> above
HAND_EXPECTED = [False, True, False, True]


def test_ten_month_rule_hand_checked():
    p = panel_from_monthly({"AAA": HAND, "BBB": HAND},
                           {"ca": ["AAA"], "cb": ["BBB"]})
    assert list(p["months"].astype(str)) == ["2000-11", "2000-12", "2001-01", "2001-02"]
    for s in range(2):
        assert list(p["state_rule"][:, s]) == HAND_EXPECTED


def test_rule_is_strict_above_so_a_flat_series_is_never_in():
    p = panel_from_monthly({"AAA": [100.0] * 14, "BBB": [100.0] * 14},
                           {"ca": ["AAA"], "cb": ["BBB"]})
    assert not p["state_rule"].any(), "close == mean must not count as above"


# ---------------------------------------------------------------------------
# 3. eligibility and the class weights
# ---------------------------------------------------------------------------
def test_eligibility_begins_at_the_tenth_month_end():
    p = panel_from_monthly({"AAA": HAND, "BBB": HAND},
                           {"ca": ["AAA"], "cb": ["BBB"]})
    # first scored return month is 10, i.e. the month after the 10th month-end
    assert p["first_idx"] == 10
    assert p["valid"].all()


def test_class_weights_renormalise_as_a_class_enters():
    n = 16
    long_ = [100.0 + i for i in range(n)]
    late = [100.0 + i for i in range(12)]       # 4 months shorter -> enters later
    p = panel_from_monthly(
        {"A1": long_, "A2": long_, "B1": long_, "C1": late},
        {"ca": ["A1", "A2"], "cb": ["B1"], "cc": ["C1"]}, n=n)
    w = p["w"]
    sl = p["sleeves"]
    i = {t: sl.index(t) for t in sl}
    early = 0                                    # only ca and cb eligible
    late_i = int(np.flatnonzero(p["valid"][:, i["C1"]])[0])
    assert late_i > early
    assert p["n_present"][early] == 2 and p["n_present"][late_i] == 3
    assert w[early, i["A1"]] == pytest.approx(0.25)
    assert w[early, i["B1"]] == pytest.approx(0.5)
    assert w[early, i["C1"]] == 0.0
    assert w[late_i, i["A1"]] == pytest.approx(1 / 6)
    assert w[late_i, i["B1"]] == pytest.approx(1 / 3)
    assert w[late_i, i["C1"]] == pytest.approx(1 / 3)
    assert np.allclose(w.sum(axis=1), 1.0)


def test_scored_window_also_waits_for_the_cash_proxy():
    """Excess returns and the cash leg both need SHY, so SHY can bind instead."""
    n = 16
    rise = [100.0 + i for i in range(n)]
    cash = [np.nan] * 13 + [100.0, 100.1, 100.2]
    prices = {t: daily_from_monthly(rise) for t in ("A1", "B1")}
    prices[e.CASH] = daily_from_monthly(cash).dropna()
    p = e.build_panel(prices, classes={"ca": ["A1"], "cb": ["B1"]})
    assert p["first_two_class_month"] == "2000-11"
    assert p["first_cash_return_month"] == "2001-03"
    assert str(p["months"][0]) == "2001-03", "the later of the two must bind"


# ---------------------------------------------------------------------------
# 4. switch counting and the cost arithmetic
# ---------------------------------------------------------------------------
def flat_panel(n=14):
    return panel_from_monthly({"AAA": [100.0] * n, "BBB": [100.0] * n},
                              {"ca": ["AAA"], "cb": ["BBB"]}, n=n)


def test_switch_count_counts_state_changes_on_eligible_pairs():
    p = flat_panel()
    st = np.zeros_like(p["valid"])
    st[0] = [True, True]
    st[1] = [False, True]
    st[2] = [True, True]
    st[3] = [True, False]
    assert e.switch_count(p, st) == 3


def test_cost_arithmetic_hand_computed_on_flat_prices():
    p = flat_panel()
    st = np.array([[True, True], [False, True], [True, True], [True, True]])
    net, gross, turn = e.evaluate_books(p, st[None, :, :], e.BPS_BASE)
    assert np.allclose(gross[0], 0.0), "flat prices earn nothing"
    # m0 buy-in 1.0; m1 0.5 out of AAA and 0.5 into cash; m2 the reverse; m3 none
    assert np.allclose(turn[0], [1.0, 1.0, 1.0, 0.0])
    assert np.allclose(net[0], -5e-4 * np.array([1.0, 1.0, 1.0, 0.0]))


def test_turnover_is_aggregated_by_asset_not_by_sleeve():
    """Two sleeves swapping which of them sits in cash never trades the cash."""
    p = flat_panel()
    st = np.array([[True, False], [False, True], [False, True], [False, True]])
    _, _, turn = e.evaluate_books(p, st[None, :, :], e.BPS_BASE)
    # by asset: AAA 0.5->0, BBB 0->0.5, cash 0.5->0.5  => 1.0
    # by sleeve it would have been 2.0
    assert turn[0, 1] == pytest.approx(1.0)


def test_stress_costs_are_three_times_base():
    p = flat_panel()
    st = np.array([[True, True], [False, True], [True, True], [True, True]])
    nb, _, _ = e.evaluate_books(p, st[None, :, :], e.BPS_BASE)
    ns, _, _ = e.evaluate_books(p, st[None, :, :], e.BPS_STRESS)
    assert np.allclose(nb[0] - ns[0], 2 * 5e-4 * np.array([1.0, 1.0, 1.0, 0.0]))


# ---------------------------------------------------------------------------
# 5. the Markov generator
# ---------------------------------------------------------------------------
def test_markov_generator_reproduces_its_target_rates():
    rng = np.random.default_rng(11)
    valid = np.ones((400, 3), bool)
    rates = {"p01": np.array([0.30, 0.10, 0.70]),
             "p10": np.array([0.60, 0.05, 0.20]),
             "pi1": np.array([1 / 3, 2 / 3, 7 / 9])}
    S = e.simulate_markov(rates, valid, 300, rng)
    got01, got10 = [], []
    for r in range(S.shape[0]):
        m = e.measure_markov_rates(S[r], valid)
        got01.append(m["p01"]); got10.append(m["p10"])
    got01 = np.mean(got01, axis=0); got10 = np.mean(got10, axis=0)
    assert np.allclose(got01, rates["p01"], atol=0.02), got01
    assert np.allclose(got10, rates["p10"], atol=0.02), got10


def test_markov_states_are_zero_before_eligibility():
    rng = np.random.default_rng(3)
    valid = np.zeros((20, 2), bool)
    valid[5:, 0] = True
    valid[12:, 1] = True
    rates = {"p01": np.array([0.5, 0.5]), "p10": np.array([0.5, 0.5]),
             "pi1": np.array([0.5, 0.5])}
    S = e.simulate_markov(rates, valid, 50, rng)
    assert not S[:, :5, 0].any()
    assert not S[:, :12, 1].any()
    assert S[:, 5:, 0].any() and S[:, 12:, 1].any()


def test_degenerate_rates_reproduce_an_always_in_path():
    valid = np.ones((30, 1), bool)
    always = np.ones((30, 1), bool)
    r = e.measure_markov_rates(always, valid)
    assert r["p10"][0] == 0.0 and r["p01"][0] == 1.0 and r["pi1"][0] == 1.0
    S = e.simulate_markov(r, valid, 20, np.random.default_rng(1))
    assert S.all(), "a never-switching book's null must also never switch"


# ---------------------------------------------------------------------------
# 6. the timing p, on fixtures with known answers
# ---------------------------------------------------------------------------
def walk_panel(months=160, seed=5):
    rng = np.random.default_rng(seed)
    spec = {}
    for t in ("A1", "A2", "B1", "B2"):
        r = rng.normal(0.006, 0.045, months)
        spec[t] = list(100 * np.cumprod(1 + r))
    cash = list(100 * np.cumprod(1 + np.full(months, 0.0015)))
    return panel_from_monthly(spec, {"ca": ["A1", "A2"], "cb": ["B1", "B2"]},
                              cash_closes=cash, n=months)


def test_timing_p_is_one_when_the_book_under_test_is_buy_and_hold():
    """dSR is exactly zero and the null, matched to an always-in path, is too."""
    p = walk_panel()
    out = e.timing_p(p, p["valid"], e.BPS_BASE, 200, np.random.default_rng(0))
    assert out["dsr"] == pytest.approx(0.0, abs=1e-12)
    assert out["p"] == 1.0
    assert np.allclose(out["null"], 0.0, atol=1e-12)


def test_timing_p_is_zero_for_a_perfectly_timed_book():
    p = walk_panel()
    perfect = e.plant_states(p, 1.0, 1, np.random.default_rng(0))[0]
    out = e.timing_p(p, perfect, e.BPS_BASE, 200, np.random.default_rng(1))
    assert out["dsr"] > 1.0
    assert out["p"] == 0.0


def test_mde_is_positive_and_scales_with_the_null_spread():
    tight = np.random.default_rng(1).normal(0, 0.1, 5000)
    wide = np.random.default_rng(1).normal(0, 0.4, 5000)
    assert e.mde_from_null(tight) > 0
    assert e.mde_from_null(wide) > 3 * e.mde_from_null(tight)


# ---------------------------------------------------------------------------
# 7. the two controls, run for real at reduced R
# ---------------------------------------------------------------------------
def test_known_null_control_sits_near_its_nominal_size():
    """Random-timing books fed in as observed must reject at about the bar."""
    p = walk_panel()
    rng = np.random.default_rng(777)
    net_bh, _, _ = e.evaluate_books(p, p["valid"][None, :, :], e.BPS_BASE)
    net_bh = net_bh[0]
    rates = e.measure_markov_rates(p["state_rule"], p["valid"])
    books = e.simulate_markov(rates, p["valid"], 60, rng)
    ps = [e.timing_p(p, books[i], e.BPS_BASE, 200, rng, net_bh=net_bh)["p"]
          for i in range(books.shape[0])]
    ps = np.array(ps)
    share = float((ps <= 0.05).mean())
    assert share <= 0.20, "a known null must not reject wholesale (got %.3f)" % share
    assert 0.25 <= ps.mean() <= 0.75, "p should be roughly uniform (got %.3f)" % ps.mean()


def test_positive_control_is_detected_at_a_strong_planting():
    p = walk_panel()
    rng = np.random.default_rng(4242)
    net_bh, _, _ = e.evaluate_books(p, p["valid"][None, :, :], e.BPS_BASE)
    net_bh = net_bh[0]
    hits = 0
    for _ in range(10):
        st = e.plant_states(p, 0.80, 1, rng)[0]
        hits += e.timing_p(p, st, e.BPS_BASE, 200, rng, net_bh=net_bh)["p"] <= 0.05
    assert hits >= 8, "detection %d/10 at q=0.80" % hits


def test_the_planted_leak_is_deliberate_and_named():
    """The positive control looks at the return it is about to earn, on purpose."""
    assert "DELIBERATE LEAK" in e.plant_states.__doc__
    assert "must never be read as a result" in e.plant_states.__doc__
    p = walk_panel()
    st = e.plant_states(p, 1.0, 1, np.random.default_rng(0))[0]
    ex = p["A"][:, p["asset_of"]] - p["shy_r"][:, None]
    assert np.array_equal(st, (ex > 0) & p["valid"]), \
        "at q=1 the planted state IS the sign of the next-month excess return"


def test_plant_at_one_half_is_a_coin_flip():
    p = walk_panel()
    st = e.plant_states(p, 0.5, 200, np.random.default_rng(9))
    ex = p["A"][:, p["asset_of"]] - p["shy_r"][:, None]
    agree = (st == ((ex > 0)[None] & p["valid"][None]))[:, p["valid"]].mean()
    assert 0.47 <= agree <= 0.53, agree


# ---------------------------------------------------------------------------
# 8. the mutation target
# ---------------------------------------------------------------------------
def test_rule_beats_timing_null_on_a_trending_fixture():
    """A fixture where timing clearly helps: a long rise, then a sustained fall.

    This is the test the reversed rule (hold when BELOW the average) fails.
    """
    months = 150
    path = np.concatenate([np.full(90, 0.012), np.full(30, -0.030), np.full(30, 0.012)])
    rng = np.random.default_rng(2)
    spec = {}
    for t in ("A1", "A2", "B1", "B2"):
        spec[t] = list(100 * np.cumprod(1 + path + rng.normal(0, 0.004, months)))
    cash = list(100 * np.cumprod(1 + np.full(months, 0.0015)))
    p = panel_from_monthly(spec, {"ca": ["A1", "A2"], "cb": ["B1", "B2"]},
                           cash_closes=cash, n=months)
    out = e.timing_p(p, p["state_rule"], e.BPS_BASE, 500, np.random.default_rng(7))
    assert out["dsr"] > 0, "trend following must beat holding through this crash"
    assert out["p"] <= 0.05, "and must beat chance timing at the same turnover"


# ---------------------------------------------------------------------------
# 9. reported-not-voting interval
# ---------------------------------------------------------------------------
def test_block_bootstrap_index_is_in_range_and_blocky():
    rng = np.random.default_rng(6)
    idx = e.stationary_block_indices(120, 6.0, 400, rng)
    assert idx.shape == (400, 120)
    assert idx.min() >= 0 and idx.max() < 120
    consecutive = (idx[:, 1:] == (idx[:, :-1] + 1) % 120).mean()
    assert 0.78 <= consecutive <= 0.89, consecutive


def test_paired_bootstrap_shares_one_index_so_a_tie_has_a_zero_interval():
    rng = np.random.default_rng(8)
    x = rng.normal(0.01, 0.04, 140)
    shy = np.full(140, 0.0015)
    out = e.paired_bootstrap(x, x, shy, 6.0, 300, rng)
    assert out["dsr_ci90"] == [0.0, 0.0]
    assert out["mean_monthly_diff_point"] == pytest.approx(0.0)


def test_cash_time_by_class_reads_eligible_sleeve_months():
    p = flat_panel()
    st = np.array([[True, False], [True, False], [True, False], [True, False]])
    out = e.cash_time_by_class(p, st)
    assert out["ca"] == pytest.approx(0.0)
    assert out["cb"] == pytest.approx(1.0)
    assert out["_all"] == pytest.approx(0.5)
