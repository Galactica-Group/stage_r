"""Tests for the T-R4 engine and runner (tools/crypto_trend_r4.py, run_t_r4.py).

Fixtures only. Nothing here reads the vintage's prices, computes a figure for
either observed book, or touches the tracked ledger. Where a registered rule is
an arithmetic claim it is checked against a hand-computable answer; where it is a
structural claim (no return across a gap, turnover aggregated by asset) the test
constructs the violation and insists the engine refuses it.

The mutations this battery is built to catch: a sleeve split that lets a return
cross a gap; an eligibility clause dropped; a ranking that is not point-in-time;
a non-strict comparison in the trend rule; turnover netted per sleeve instead of
per asset; a first month that pays no buy-in; a cost charged to the signal month;
a control band or detection floor read with the wrong inequality; and an observed
stage that runs without its receipt.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import crypto_trend_r4 as c  # noqa: E402
import run_t_r4 as r  # noqa: E402


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def bars(start, n_days, base=100.0, drift=0.0, qv=1.0, drop=()):
    """Contiguous daily bars from start, optionally with offsets removed."""
    idx = pd.date_range(start, periods=n_days, freq="D", tz="UTC")
    close = base * (1.0 + drift) ** np.arange(n_days, dtype=float)
    df = pd.DataFrame({"close": close, "quote_volume": float(qv)}, index=idx)
    if drop:
        df = df.loc[~np.isin(np.arange(n_days), np.asarray(drop))]
    return df


def panel_of(frames, first="2020-11", last="2021-12"):
    recs = c.sleeve_records(frames)
    return c.build_panel(recs, pd.Period(first, freq="M"), pd.Period(last, freq="M"))


def two_symbol_frames(**kw):
    return {"AAAUSDT": bars("2020-01-01", 730, drift=0.001, qv=9.0, **kw),
            "BBBUSDT": bars("2020-01-01", 730, drift=0.0005, qv=8.0)}


def toy_panel(w, A, asset_of, n_assets, cash_j, A_sleeve=None):
    """A hand-built panel for the book arithmetic: w (M,NS), A (M,NJ)."""
    w = np.asarray(w, dtype=float)
    NS = w.shape[1]
    P = np.zeros((NS, n_assets))
    P[np.arange(NS), np.asarray(asset_of)] = 1.0
    return {"w": w, "P": P, "A": np.asarray(A, dtype=float), "cash_j": cash_j,
            "member": w > 0,
            "A_sleeve": (np.zeros_like(w) if A_sleeve is None
                         else np.asarray(A_sleeve, dtype=float))}


# ---------------------------------------------------------------------------
# 1. sleeve split
# ---------------------------------------------------------------------------
def test_sleeve_bounds_cuts_at_every_gap():
    idx = pd.DatetimeIndex(pd.to_datetime(
        ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06", "2020-01-20"],
        utc=True))
    assert c.sleeve_bounds(idx) == [(0, 2), (2, 4), (4, 5)]


def test_a_single_gap_makes_two_sleeves():
    f = {"AAAUSDT": bars("2020-01-01", 730, drift=0.001, drop=(400,))}
    recs = c.sleeve_records(f)
    assert len(recs) == 2
    assert [x["name"] for x in recs] == ["AAAUSDT#0", "AAAUSDT#1"]
    assert recs[0]["last_bar"] < recs[1]["first_bar"]


def test_contiguous_history_is_one_sleeve():
    recs = c.sleeve_records({"AAAUSDT": bars("2020-01-01", 730)})
    assert len(recs) == 1
    assert recs[0]["n_bars_total"] == 730


def test_no_return_is_formed_across_a_gap():
    f = two_symbol_frames()
    f["GAPUSDT"] = bars("2020-01-01", 730, drift=0.002, qv=7.0,
                        drop=tuple(range(320, 340)))
    p = panel_of(f)
    names = p["sleeve_names"]
    pre = [i for i, n in enumerate(names) if n == "GAPUSDT#0"]
    post = [i for i, n in enumerate(names) if n == "GAPUSDT#1"]
    assert pre and post, names
    # the two sleeves never both carry a return in the same month
    A = p["A_sleeve"]
    overlap = np.isfinite(A[:, pre[0]]) & np.isfinite(A[:, post[0]])
    assert not overlap.any()


def test_cross_gap_return_assertion_fires():
    M, NS = 4, 1
    C = np.array([[10.0], [np.nan], [12.0], [13.0]])
    IN = np.array([[True], [False], [True], [True]])
    A = np.array([[np.nan], [np.nan], [0.2], [0.08]])   # 0.2 bridges the gap
    fe = np.zeros((M, NS), bool)
    with pytest.raises(AssertionError, match="return computed across a gap"):
        c.assert_no_cross_gap_return(C, A, IN, IN, fe, ["X#0"],
                                     pd.period_range("2020-01", periods=4, freq="M"))


def test_cross_gap_guard_accepts_a_clean_panel():
    C = np.array([[10.0], [11.0], [12.0]])
    IN = np.ones((3, 1), bool)
    A = np.array([[np.nan], [0.1], [1.0 / 11.0]])
    fe = np.zeros((3, 1), bool)
    assert c.assert_no_cross_gap_return(
        C, A, IN, IN, fe, ["X#0"],
        pd.period_range("2020-01", periods=3, freq="M")) is True


def test_forced_exit_must_be_a_zero_return():
    C = np.array([[10.0], [11.0], [np.nan]])
    IN = np.array([[True], [True], [False]])
    A = np.array([[np.nan], [0.1], [0.05]])
    fe = np.array([[False], [False], [True]])
    with pytest.raises(AssertionError, match="must be a zero return"):
        c.assert_no_cross_gap_return(C, A, IN, IN, fe, ["X#0"],
                                     pd.period_range("2020-01", periods=3, freq="M"))


def test_a_sleeve_is_contiguous_in_month_space():
    recs = c.sleeve_records({"AAAUSDT": bars("2020-01-01", 730, drop=(400,))})
    for rec in recs:
        ms = rec["months"]
        assert all((ms[i + 1] - ms[i]).n == 1 for i in range(len(ms) - 1))


# ---------------------------------------------------------------------------
# 2. month-end close and the in-sleeve fallback
# ---------------------------------------------------------------------------
def test_month_end_close_is_the_bar_stamped_on_the_last_calendar_day():
    df = bars("2020-01-01", 400, base=100.0, drift=0.0)
    df = df.assign(close=np.arange(len(df), dtype=float))
    recs = c.sleeve_records({"AAAUSDT": df})
    rows = {x["month"]: x for x in recs[0]["rows"]}
    j = rows[pd.Period("2020-06", freq="M")]
    assert j["bar_date"] == pd.Timestamp("2020-06-30", tz="UTC")
    assert j["close"] == float(df.index.get_loc(pd.Timestamp("2020-06-30", tz="UTC")))


def test_month_end_close_falls_back_only_inside_the_sleeve():
    start = pd.Timestamp("2020-01-01", tz="UTC")
    drop = [(pd.Timestamp(d, tz="UTC") - start).days
            for d in pd.date_range("2020-06-26", "2020-07-02")]
    df = bars("2020-01-01", 730, drift=0.001, drop=tuple(drop))
    recs = c.sleeve_records({"GAPUSDT": df})
    assert len(recs) == 2
    pre = {x["month"]: x for x in recs[0]["rows"]}
    post = {x["month"]: x for x in recs[1]["rows"]}
    # sleeve 0's June month-end close is its own last bar, the 25th
    assert pre[pd.Period("2020-06", freq="M")]["bar_date"] == \
        pd.Timestamp("2020-06-25", tz="UTC")
    # sleeve 1 has no June month-end at all: the fallback did not reach back
    assert pd.Period("2020-06", freq="M") not in post
    assert min(post) == pd.Period("2020-07", freq="M")
    assert post[pd.Period("2020-07", freq="M")]["bar_date"] == \
        pd.Timestamp("2020-07-31", tz="UTC")


def test_a_sleeve_ending_mid_month_uses_its_last_close_for_that_month():
    df = bars("2020-01-01", 177)        # 2020-01-01 .. 2020-06-25
    recs = c.sleeve_records({"AAAUSDT": df})
    rows = {x["month"]: x for x in recs[0]["rows"]}
    assert rows[pd.Period("2020-06", freq="M")]["bar_date"] == \
        pd.Timestamp("2020-06-25", tz="UTC")


# ---------------------------------------------------------------------------
# 3. eligibility
# ---------------------------------------------------------------------------
def test_eligibility_month_end_clause_binds_alone(monkeypatch):
    monkeypatch.setattr(c, "MIN_BARS", 1)
    f = two_symbol_frames()
    f["LATEUSDT"] = bars("2020-03-01", 670, drift=0.001, qv=99.0)
    p = panel_of(f, first="2020-11", last="2021-12")
    names = p["all_sleeve_names"]
    assert "LATEUSDT#0" in names
    # LATE has 9 month-ends at 2020-11-30 and 10 at 2020-12-31
    recs = c.sleeve_records({"LATEUSDT": f["LATEUSDT"]})
    ms = recs[0]["months"]
    assert ms.index(pd.Period("2020-11", freq="M")) + 1 == 9
    assert ms.index(pd.Period("2020-12", freq="M")) + 1 == 10


def test_eligibility_ninety_bar_clause_binds_alone(monkeypatch):
    monkeypatch.setattr(c, "MIN_MONTH_ENDS", 1)
    df = bars("2020-01-01", 95)
    recs = c.sleeve_records({"AAAUSDT": df})
    rows = {x["month"]: x for x in recs[0]["rows"]}
    # at the 2020-03 month-end the sleeve has 90 bars exactly; at 2020-02, 60
    assert rows[pd.Period("2020-02", freq="M")]["n_bars"] == 60
    assert rows[pd.Period("2020-03", freq="M")]["n_bars"] == 91
    assert c.MIN_BARS == 90


def test_eligibility_requires_both_clauses_at_registered_values():
    src = (ROOT / "tools" / "crypto_trend_r4.py").read_text()
    assert "(NME[m] >= MIN_MONTH_ENDS) & (NB[m] >= MIN_BARS)" in src
    assert c.MIN_MONTH_ENDS == 10 and c.MIN_BARS == 90


def test_ten_month_ends_implies_the_ninety_bar_clause_is_satisfied():
    # contiguity makes ten consecutive month-ends at least nine whole months
    df = bars("2020-01-31", 400)
    recs = c.sleeve_records({"AAAUSDT": df})
    tenth = recs[0]["rows"][c.MIN_MONTH_ENDS - 1]
    assert tenth["n_bars"] >= c.MIN_BARS


def test_qv90_is_a_true_ninety_bar_sum_for_an_eligible_sleeve():
    df = bars("2020-01-01", 400, qv=2.0)
    recs = c.sleeve_records({"AAAUSDT": df})
    tenth = recs[0]["rows"][9]
    assert tenth["n_bars"] >= 90
    assert tenth["qv90"] == pytest.approx(2.0 * 90)


def test_qv90_never_reaches_outside_the_sleeve():
    df = bars("2020-01-01", 730, qv=5.0, drop=tuple(range(500, 520)))
    recs = c.sleeve_records({"GAPUSDT": df})
    post = recs[1]
    first = post["rows"][0]
    assert first["qv90"] == pytest.approx(5.0 * first["n_bars"])
    assert first["n_bars"] <= post["n_bars_total"]


# ---------------------------------------------------------------------------
# 4. point-in-time ranking and membership
# ---------------------------------------------------------------------------
def test_point_in_time_ranking_late_lister_enters_and_incumbent_leaves():
    f = {}
    for i in range(21):
        f["A%02dUSDT" % i] = bars("2020-01-01", 730, drift=0.001,
                                  qv=float(100 - i))
    f["LATEUSDT"] = bars("2020-03-01", 670, drift=0.001, qv=1000.0)
    p = panel_of(f, first="2020-11", last="2021-02")
    names = p["sleeve_names"]
    months = [str(m) for m in p["months"]]
    mem = p["member"]
    i_late = names.index("LATEUSDT#0")
    i_a19 = names.index("A19USDT#0")
    # A20 has the lowest volume and never reaches the top 20, so it is a
    # sleeve that never becomes a member
    assert "A20USDT#0" in p["all_sleeve_names"]
    assert "A20USDT#0" not in names

    m_nov = months.index("2020-11")
    m_dec = months.index("2020-12")
    m_jan = months.index("2021-01")
    assert int(mem[m_nov].sum()) == c.TOP_N
    # November is decided at 2020-10-31, when LATE has only 8 month-ends
    assert not mem[m_nov, i_late]
    assert mem[m_nov, i_a19]
    # December is decided at 2020-11-30, when LATE has only 9 month-ends: a
    # ranking read at the RETURN month instead would already have let it in
    assert not mem[m_dec, i_late]
    # January is decided at 2020-12-31: LATE is eligible and ranks first
    assert mem[m_jan, i_late]
    assert not mem[m_jan, i_a19]
    assert int(mem[m_jan].sum()) == c.TOP_N


def test_membership_candidates_are_taken_at_the_signal_month_end():
    src = (ROOT / "tools" / "crypto_trend_r4.py").read_text()
    assert "cand = np.flatnonzero(eligible[sig])" in src
    assert "cand = np.flatnonzero(eligible[m])" not in src


def test_membership_is_capped_at_top_n_and_equally_weighted():
    f = {}
    for i in range(25):
        f["A%02dUSDT" % i] = bars("2020-01-01", 730, drift=0.001,
                                  qv=float(100 - i))
    p = panel_of(f, first="2020-11", last="2021-12")
    n = p["n_members"]
    assert n.max() == c.TOP_N
    w = p["w"]
    for m in range(len(p["months"])):
        row = w[m][w[m] > 0]
        assert len(row) == n[m]
        assert np.allclose(row, 1.0 / n[m])
        assert w[m].sum() == pytest.approx(1.0)


def test_all_eligible_are_members_when_fewer_than_top_n():
    p = panel_of(two_symbol_frames())
    assert p["n_members"].max() == 2
    assert np.allclose(p["w"][0][p["w"][0] > 0], 0.5)


def test_ranking_ties_break_deterministically_by_name():
    f = {}
    for i in range(22):
        f["A%02dUSDT" % i] = bars("2020-01-01", 730, drift=0.001, qv=1.0)
    p = panel_of(f, first="2020-11", last="2020-12")
    names = p["sleeve_names"]
    mem = p["member"][0]
    chosen = sorted(names[i] for i in np.flatnonzero(mem))
    assert chosen == sorted(["A%02dUSDT#0" % i for i in range(20)])


# ---------------------------------------------------------------------------
# 5. the state rule and its equality edge
# ---------------------------------------------------------------------------
def test_state_rule_equality_holds_usdt():
    f = {"FLATUSDT": bars("2020-01-01", 730, drift=0.0, qv=9.0),
         "UPUSDT": bars("2020-01-01", 730, drift=0.002, qv=8.0)}
    p = panel_of(f)
    names = p["sleeve_names"]
    st = p["state_rule"]
    i_flat = names.index("FLATUSDT#0")
    i_up = names.index("UPUSDT#0")
    # a flat series sits exactly ON its own mean: strictly-above fails
    assert not st[:, i_flat].any()
    # a rising series is above it in every month
    assert st[:, i_up].all()


def test_state_rule_is_strictly_above_in_source():
    src = (ROOT / "tools" / "crypto_trend_r4.py").read_text()
    assert "(C[m] > ma)" in src
    assert "(C[m] >= ma)" not in src


def test_state_rule_uses_the_last_ten_month_end_closes_inclusive():
    src = (ROOT / "tools" / "crypto_trend_r4.py").read_text()
    assert "win = C[m - LOOKBACK + 1: m + 1, :]" in src
    assert c.LOOKBACK == 10


def test_state_is_decided_at_the_signal_month_and_earns_the_next():
    f = two_symbol_frames()
    p = panel_of(f)
    assert p["state_rule"].shape == p["member"].shape
    src = (ROOT / "tools" / "crypto_trend_r4.py").read_text()
    assert '"state_rule": state_sig[np.arange(first - 1, last)][:, keep]' in src


# ---------------------------------------------------------------------------
# 6. book arithmetic: turnover by asset, buy-in, cost timing
# ---------------------------------------------------------------------------
def test_turnover_is_aggregated_by_asset_so_offsetting_sleeves_do_not_trade():
    # two sleeves, two coins (assets 0 and 1) and USDT (asset 2)
    w = np.array([[0.5, 0.5], [0.5, 0.5]])
    A = np.zeros((2, 3))
    p = toy_panel(w, A, asset_of=[0, 1], n_assets=3, cash_j=2)
    # month 0: sleeve A in cash, sleeve B in coin. month 1: the reverse.
    st = np.array([[[False, True], [True, False]]])
    net, gross, turn = c.evaluate_books(p, st, bps=10.0)
    # the cash leg is 0.5 in both months, so it does not trade; only the two
    # coin legs do, 0.5 each way
    assert turn[0, 1] == pytest.approx(1.0)
    # per-sleeve accounting is computed here as the witness: it would charge the
    # offsetting cash legs as a round trip and bill 2.0 for the same market order
    S = st.astype(float)
    Wsleeve = np.concatenate([p["w"][None] * S, p["w"][None] * (1.0 - S)], axis=2)
    Vs = Wsleeve * 1.0
    Ds = np.zeros_like(Wsleeve)
    Ds[:, 1:, :] = Vs[:, :-1, :] / Vs[:, :-1, :].sum(axis=2, keepdims=True)
    per_sleeve = np.abs(Wsleeve - Ds).sum(axis=2)
    assert per_sleeve[0, 1] == pytest.approx(2.0)
    assert turn[0, 1] < per_sleeve[0, 1]


def test_two_sleeves_in_usdt_are_one_position():
    w = np.array([[0.5, 0.5], [0.5, 0.5]])
    A = np.zeros((2, 3))
    p = toy_panel(w, A, asset_of=[0, 1], n_assets=3, cash_j=2)
    st = np.array([[[False, False], [False, False]]])
    net, gross, turn = c.evaluate_books(p, st, bps=10.0)
    WS = (p["w"][None] * st.astype(float)) @ p["P"]
    WS[:, :, 2] += (p["w"][None] * (1 - st.astype(float))).sum(axis=2)
    assert WS[0, 0, 2] == pytest.approx(1.0)
    # month 1 trades nothing: the single cash position is already in place
    assert turn[0, 1] == pytest.approx(0.0)


def test_first_month_pays_a_full_buy_in_in_both_books():
    w = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    A = np.zeros((3, 3))
    p = toy_panel(w, A, asset_of=[0, 1], n_assets=3, cash_j=2)
    trend = np.array([[[True, True], [True, True], [True, True]]])
    bh = np.array([[[True, True], [True, True], [True, True]]])
    _, _, t_tr = c.evaluate_books(p, trend, bps=10.0)
    _, _, t_bh = c.evaluate_books(p, bh, bps=10.0)
    assert t_tr[0, 0] == pytest.approx(1.0)
    assert t_bh[0, 0] == pytest.approx(1.0)
    # and a cash-only first month is still a full buy-in of the cash position
    allcash = np.zeros((1, 3, 2), bool)
    _, _, t_c = c.evaluate_books(p, allcash, bps=10.0)
    assert t_c[0, 0] == pytest.approx(1.0)


def test_cost_is_charged_to_the_month_the_new_positions_earn():
    w = np.array([[1.0], [1.0], [1.0]])
    A = np.array([[0.10, 0.0], [0.20, 0.0], [0.30, 0.0]])
    p = toy_panel(w, A, asset_of=[0], n_assets=2, cash_j=1)
    # in the coin, then cash from month 1 onward: the switch is decided at the
    # end of month 0 and the cost lands in month 1, not month 0
    st = np.array([[[True], [False], [False]]])
    net, gross, turn = c.evaluate_books(p, st, bps=10.0)
    assert turn[0, 0] == pytest.approx(1.0)      # the buy-in
    assert turn[0, 1] == pytest.approx(2.0)      # sell the coin, buy cash
    assert turn[0, 2] == pytest.approx(0.0)      # nothing changes afterwards
    assert net[0, 1] == pytest.approx(gross[0, 1] - 10.0e-4 * 2.0)
    assert net[0, 2] == pytest.approx(gross[0, 2])


def test_weights_drift_with_realised_returns_inside_the_month():
    w = np.array([[0.5, 0.5], [0.5, 0.5]])
    A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])   # coin 0 doubles
    p = toy_panel(w, A, asset_of=[0, 1], n_assets=3, cash_j=2)
    st = np.ones((1, 2, 2), bool)
    net, gross, turn = c.evaluate_books(p, st, bps=0.0)
    assert gross[0, 0] == pytest.approx(0.5)
    # drifted weights are 2/3 and 1/3; rebalancing to 0.5/0.5 trades 1/6 each
    # side, so the traded notional is 1/3
    assert turn[0, 1] == pytest.approx(1.0 / 6.0 + 1.0 / 6.0)


def test_book_weights_must_sum_to_one():
    w = np.array([[0.4, 0.4]])
    A = np.zeros((1, 3))
    p = toy_panel(w, A, asset_of=[0, 1], n_assets=3, cash_j=2)
    with pytest.raises(AssertionError, match="weights do not sum to 1"):
        c.evaluate_books(p, np.ones((1, 1, 2), bool), bps=10.0)


def test_usdt_return_is_zero_so_cash_months_earn_nothing_gross():
    w = np.array([[1.0], [1.0]])
    A = np.array([[0.5, 0.0], [0.5, 0.0]])
    p = toy_panel(w, A, asset_of=[0], n_assets=2, cash_j=1)
    net, gross, turn = c.evaluate_books(p, np.zeros((1, 2, 1), bool), bps=0.0)
    assert gross[0].tolist() == [0.0, 0.0]


# ---------------------------------------------------------------------------
# 7. statistics
# ---------------------------------------------------------------------------
def test_annualised_sharpe_hand_computed():
    x = np.array([0.01, -0.01, 0.02, 0.00])
    mu, sd = x.mean(), x.std(ddof=1)
    assert c.annualised_sharpe(x) == pytest.approx(mu / sd * np.sqrt(12))


def test_annualised_sharpe_is_nan_on_a_constant_series():
    assert np.isnan(c.annualised_sharpe(np.array([0.01, 0.01, 0.01])))


def test_dsr_and_dmu_hand_computed():
    t = np.array([0.02, 0.01, 0.03, 0.00])
    b = np.array([0.01, 0.02, 0.01, 0.01])
    assert c.dsr(t, b) == pytest.approx(
        c.annualised_sharpe(t) - c.annualised_sharpe(b))
    assert c.dmu(t, b) == pytest.approx((t.mean() - b.mean()) * 12)


def test_dmu_sign_separates_risk_only_from_something():
    t = np.array([0.00, 0.00, 0.01, 0.00])      # lower mean, lower vol
    b = np.array([0.04, -0.03, 0.05, -0.02])
    assert c.dsr(t, b) > 0
    assert c.dmu(t, b) < 0


def test_book_metrics_hand_computed():
    net = np.array([0.10, -0.20, 0.05])
    m = c.book_metrics(net)
    cum = np.cumprod(1 + net)
    assert m["months"] == 3
    assert m["cagr"] == pytest.approx(cum[-1] ** (12 / 3) - 1)
    assert m["max_drawdown"] == pytest.approx(cum[1] / cum[0] - 1)
    assert m["mean_monthly"] == pytest.approx(net.mean())


# ---------------------------------------------------------------------------
# 8. the null: rates, degenerate cases, simulation, MDE
# ---------------------------------------------------------------------------
def test_markov_rates_measured_from_the_path():
    member = np.ones((5, 1), bool)
    states = np.array([[False], [True], [True], [False], [True]])
    rr = c.measure_markov_rates(states, member)
    # from 0: at months 0 and 3 -> next is 1 both times  => p01 = 1.0
    # from 1: at months 1, 2, 4(no next) -> 1->1, 1->0   => p10 = 0.5
    assert rr["p01"][0] == pytest.approx(1.0)
    assert rr["p10"][0] == pytest.approx(0.5)
    assert rr["pi1"][0] == pytest.approx(1.0 / 1.5)


def test_degenerate_rate_no_zero_state_gives_p01_one():
    member = np.ones((4, 1), bool)
    states = np.ones((4, 1), bool)
    rr = c.measure_markov_rates(states, member)
    assert rr["p01"][0] == pytest.approx(1.0)
    assert rr["p10"][0] == pytest.approx(0.0)


def test_degenerate_rate_no_one_state_gives_p10_zero():
    member = np.ones((4, 1), bool)
    states = np.zeros((4, 1), bool)
    rr = c.measure_markov_rates(states, member)
    assert rr["p10"][0] == pytest.approx(0.0)
    assert rr["p01"][0] == pytest.approx(0.0)
    # p01 + p10 == 0 -> stationary falls back to the empirical mean state
    assert rr["pi1"][0] == pytest.approx(0.0)


def test_degenerate_rate_sum_zero_uses_the_empirical_mean_state():
    member = np.ones((4, 1), bool)
    states = np.array([[True], [True], [False], [False]])
    rr = c.measure_markov_rates(states, member)
    # with p01 = 0 and p10 = 0 forced, pi1 is the mean state; construct that case
    rr2 = c.measure_markov_rates(states, member)
    assert 0.0 <= rr2["pi1"][0] <= 1.0
    forced = {"p01": np.array([0.0]), "p10": np.array([0.0]),
              "pi1": np.array([float(states.mean())])}
    S = c.simulate_markov(forced, member, 200, np.random.default_rng(0))
    # a frozen chain keeps its initial state for the whole path
    assert np.all(S[:, 0, 0] == S[:, -1, 0])
    assert 0.3 < S[:, 0, 0].mean() < 0.7


def test_rates_are_measured_over_member_pairs_only():
    member = np.array([[True], [False], [True], [True]])
    states = np.array([[True], [True], [False], [True]])
    rr = c.measure_markov_rates(states, member)
    # only the (2,3) pair is member-on-both: 0 -> 1
    assert rr["p01"][0] == pytest.approx(1.0)
    assert rr["p10"][0] == pytest.approx(0.0)


def test_simulated_states_are_false_where_not_a_member():
    member = np.array([[True, False], [True, True], [False, True]])
    rates = {"p01": np.array([1.0, 1.0]), "p10": np.array([0.0, 0.0]),
             "pi1": np.array([1.0, 1.0])}
    S = c.simulate_markov(rates, member, 50, np.random.default_rng(1))
    assert not S[:, ~member].any()


def test_mde_formula_is_the_registered_expression():
    rng = np.random.default_rng(3)
    d = rng.normal(size=5000)
    expected = (np.percentile(d, 95) - np.median(d)) + c.Z80 * d.std(ddof=1)
    assert c.mde_from_null(d) == pytest.approx(expected)
    assert c.Z80 == 0.8416


def test_mde_and_p_are_invariant_to_the_buy_and_hold_constant():
    rng = np.random.default_rng(4)
    s_null = rng.normal(size=4000)
    s_obs = 0.7
    s_bh = 1.234                      # the constant dSR subtracts from both sides
    p_sharpe = float((s_null >= s_obs).mean())
    p_dsr = float(((s_null - s_bh) >= (s_obs - s_bh)).mean())
    assert p_sharpe == p_dsr
    assert c.mde_from_null(s_null) == pytest.approx(c.mde_from_null(s_null - s_bh))


def test_band_ok_is_a_closed_band():
    assert c.band_ok(0.01, [0.01, 0.10])
    assert c.band_ok(0.10, [0.01, 0.10])
    assert c.band_ok(0.05, [0.01, 0.10])
    assert not c.band_ok(0.009, [0.01, 0.10])
    assert not c.band_ok(0.101, [0.01, 0.10])


def test_detection_ok_is_at_or_above_the_floor():
    assert c.detection_ok(0.80, 0.80)
    assert c.detection_ok(0.90, 0.80)
    assert not c.detection_ok(0.79, 0.80)


# ---------------------------------------------------------------------------
# 9. the positive control's leak, named
# ---------------------------------------------------------------------------
def test_positive_control_leak_reads_the_sign_of_the_return_it_is_about_to_earn():
    member = np.ones((4, 2), bool)
    A_s = np.array([[0.1, -0.1], [-0.2, 0.3], [0.05, 0.05], [-0.01, -0.02]])
    p = {"member": member, "A_sleeve": A_s}
    st = c.plant_states(p, q=1.0, R=1, rng=np.random.default_rng(0))[0]
    # at q = 1 the planted state IS the sign of that month's own return
    assert np.array_equal(st, A_s > 0)
    # that is the leak, and it is deliberate
    assert "DELIBERATE LEAK" in c.plant_states.__doc__
    assert "about to earn" in c.plant_states.__doc__


def test_positive_control_at_q_zero_is_the_exact_opposite():
    member = np.ones((3, 1), bool)
    A_s = np.array([[0.1], [-0.2], [0.3]])
    st = c.plant_states({"member": member, "A_sleeve": A_s}, q=0.0, R=1,
                        rng=np.random.default_rng(0))[0]
    assert np.array_equal(st, ~(A_s > 0))


def test_planted_states_are_false_where_not_a_member():
    member = np.array([[True], [False], [True]])
    A_s = np.array([[0.1], [0.1], [0.1]])
    st = c.plant_states({"member": member, "A_sleeve": A_s}, q=1.0, R=5,
                        rng=np.random.default_rng(0))
    assert not st[:, 1, 0].any()


# ---------------------------------------------------------------------------
# 10. universe resolution
# ---------------------------------------------------------------------------
def constructed_manifest():
    syms = {}
    for i in range(95):
        syms["TK%02dBUSDT" % i] = {"first_open_time": "2026-07-01T00:00:00+00:00",
                                   "rows": 10}
    for s in ("BFUSDUSDT", "PAXGUSDT", "WBTCUSDT", "BTCUSDT", "ETHUSDT",
              "SHIBUSDT", "ARBUSDT"):
        syms[s] = {"first_open_time": "2020-01-01T00:00:00+00:00", "rows": 100}
    return {"symbols": syms}


def constructed_reg():
    return {"universe": {"exclusions": {
        "pegged_or_metal_backed": ["BFUSDUSDT", "PAXGUSDT"],
        "wrapped_or_staked_duplicates": ["WBTCUSDT"],
        "tokenized_equities_rule": "ends in B on or after 2026-06-11, exactly 95"}}}


def test_exclusions_resolve_including_the_tokenized_equity_rule():
    u = c.resolve_universe(constructed_manifest(), constructed_reg())
    assert len(u["tokenized_equities"]) == 95
    assert u["vintage_symbols"] == 102
    assert u["excluded_union"] == 98
    assert u["pool_size"] == 4
    assert sorted(u["pool"]) == ["ARBUSDT", "BTCUSDT", "ETHUSDT", "SHIBUSDT"]


def test_tokenized_rule_ignores_a_b_base_listed_before_the_cutoff():
    # SHIB and ARB both end in B but list in 2020; neither is excluded
    u = c.resolve_universe(constructed_manifest(), constructed_reg())
    assert "SHIBUSDT" not in u["tokenized_equities"]
    assert "ARBUSDT" not in u["tokenized_equities"]


def test_tokenized_set_not_ninety_five_halts():
    man = constructed_manifest()
    del man["symbols"]["TK00BUSDT"]
    with pytest.raises(AssertionError, match="registration pins 95"):
        c.resolve_universe(man, constructed_reg())


def test_a_double_listed_symbol_is_counted_once_in_the_union():
    man = constructed_manifest()
    man["symbols"]["USDEBUSDT"] = {"first_open_time": "2026-10-07T00:00:00+00:00",
                                   "rows": 8}
    reg = constructed_reg()
    reg["universe"]["exclusions"]["pegged_or_metal_backed"].append("USDEBUSDT")
    with pytest.raises(AssertionError, match="registration pins 95"):
        c.resolve_universe(man, reg)
    del man["symbols"]["TK00BUSDT"]
    u = c.resolve_universe(man, reg)
    assert "USDEBUSDT" in u["double_listed"]
    # 95 tokenized + 3 pegged + 1 wrapped, less the one symbol on both lists
    assert u["excluded_union"] == 95 + 3 + 1 - 1


def test_registered_name_absent_from_the_vintage_is_named_not_silent():
    reg = constructed_reg()
    reg["universe"]["exclusions"]["pegged_or_metal_backed"].append("GHOSTUSDT")
    u = c.resolve_universe(constructed_manifest(), reg)
    assert u["registered_names_absent_from_vintage"] == ["GHOSTUSDT"]


def test_base_asset_requires_a_usdt_quote():
    assert c.base_asset("NVDABUSDT") == "NVDAB"
    with pytest.raises(AssertionError, match="not USDT-quoted"):
        c.base_asset("ETHBTC")


# ---------------------------------------------------------------------------
# 11. registration loading and the window
# ---------------------------------------------------------------------------
def test_load_registration_refuses_a_sha_mismatch(tmp_path):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"a": 1}))
    good = c.sha256_file(p)
    assert c.load_registration(p, good)["a"] == 1
    with pytest.raises(AssertionError, match="registration sha256 mismatch"):
        c.load_registration(p, "0" * 64)


def test_window_bounds_derive_the_exclusive_max_from_the_seal():
    reg = {"window": {"half_A_earning_months": "2018-06 .. 2021-12"}}
    first, last, cut = c.window_bounds(reg)
    assert str(first) == "2018-06"
    assert str(last) == "2021-12"
    assert cut == pd.Timestamp("2022-01-01", tz="UTC")


def test_loader_refuses_a_bar_at_or_past_the_cut(tmp_path):
    df = bars("2021-12-25", 20)
    (tmp_path).mkdir(exist_ok=True)
    df.to_parquet(tmp_path / "ZZZUSDT.parquet")
    out, spans = c.load_vintage(tmp_path, ["ZZZUSDT"],
                               pd.Timestamp("2022-01-01", tz="UTC"))
    assert out["ZZZUSDT"].index.max() < pd.Timestamp("2022-01-01", tz="UTC")
    assert len(out["ZZZUSDT"]) == 7


def test_registration_prose_pins_the_constants():
    """The few integers the seal states in prose must match the engine."""
    sealed = ROOT / "logs" / "r4" / "registration_v1.json"
    if sealed.exists():
        text = sealed.read_text()
    else:
        text = json.dumps({
            "books": {"TREND": "the mean of its last 10 month-end closes"},
            "universe": {
                "eligibility": "at least 10 month-end closes (t included) and at "
                               "least 90 daily bars ending at t",
                "ranking": "the sum of quote_volume over the 90 daily bars ending at t",
                "membership": "the top 20 by that ranking",
                "exclusions": {"tokenized_equities_rule":
                               "whose base ends in the letter B"}},
            "order_of_operations": ["+ 0.8416 * sd(null dSR)"]})
    assert "last %d month-end closes" % c.LOOKBACK in text
    assert "at least %d month-end closes" % c.MIN_MONTH_ENDS in text
    assert "at least %d daily bars" % c.MIN_BARS in text
    assert "over the %d daily bars" % c.RANK_BARS in text
    assert "the top %d" % c.TOP_N in text
    assert "%s * sd(null dSR)" % c.Z80 in text
    assert "ends in the letter %s" % c.TOKENIZED_SUFFIX in text


# ---------------------------------------------------------------------------
# 12. stage gating
# ---------------------------------------------------------------------------
def test_observed_stage_refuses_without_a_receipt(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(r, "LOGDIR", tmp_path)
    monkeypatch.setattr(r, "RECEIPT", tmp_path / "controls_pass.json")
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"a": 1}))
    rc = r.main(["--stage", "observed", "--registration", str(reg),
                 "--sha", c.sha256_file(reg)])
    assert rc == 2
    assert "REFUSED" in capsys.readouterr().out


def test_observed_stage_refuses_on_a_registration_sha_mismatch(tmp_path, monkeypatch,
                                                               capsys):
    monkeypatch.setattr(r, "LOGDIR", tmp_path)
    receipt = tmp_path / "controls_pass.json"
    monkeypatch.setattr(r, "RECEIPT", receipt)
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"a": 1}))
    receipt.write_text(json.dumps({"registration_sha": "0" * 64}))
    rc = r.main(["--stage", "observed", "--registration", str(reg),
                 "--sha", c.sha256_file(reg)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "REFUSED" in out
    assert "live file is" in out


def test_receipt_carries_the_registered_fields():
    src = (ROOT / "tools" / "run_t_r4.py").read_text()
    for field in ("registration_sha", "mde", "known_null_share", "q_chosen",
                  "detection", "utc"):
        assert '"%s"' % field in src


def test_halt_writes_the_same_figures_and_exits_non_zero():
    src = (ROOT / "tools" / "run_t_r4.py").read_text()
    assert "def halt_out(" in src
    assert "return 15" in src
    assert "RECEIPT.unlink()" in src


# ---------------------------------------------------------------------------
# 13. housekeeping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", ["crypto_trend_r4.py", "run_t_r4.py"])
def test_no_credential_is_read_by_the_new_tools(tool):
    path = ROOT / "tools" / tool
    src = path.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    code = src.replace(doc, "")
    for forbidden in ("dotenv", "load_dotenv", ".env", "API_KEY", "api_key",
                      "secret", "X-MBX-APIKEY", "requests"):
        assert forbidden not in code, (tool, forbidden)


@pytest.mark.parametrize("tool", ["crypto_trend_r4.py", "run_t_r4.py",
                                  "ledger_chain.py", "verify_ledger.py"])
def test_new_tools_import_nothing_from_the_factory_tree(tool):
    src = (ROOT / "tools" / tool).read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            for a in n.names:
                assert "Monte_Carlo" not in a.name
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or "").startswith(("src.", "1_data", "2_research"))
    assert "Monte_Carlo_Strategies" not in src
