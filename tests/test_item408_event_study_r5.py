"""Tests for the T-R5 engine and runner (tools/item408_event_study_r5.py,
tools/run_t_r5.py).

Fixtures only. Nothing here reads the frozen snapshot, forms a return of a real
event, or touches the tracked ledger. Every registered resolution clause is made
to bind ALONE: the fixture differs from a passing fixture in exactly the one
field that clause reads, so a dropped clause cannot hide behind its neighbours.
Where a rule is arithmetic it is checked against a hand-computable answer; where
it is structural (one common shift per replicate, the overlap drop, the 90 per
cent redraw) the test constructs the violation and insists the engine refuses it.

The mutations this battery is built to catch: a termination clause that also
admits modifications; a lag clause with the inequality reversed; a form filter
widened to the /A amendments; a direction rule that drops the "both" and
"neither" rows it is registered to keep; a mapping tie-break that ignores which
ticker actually trades; a completeness filter that tolerates one missing day; a
cost that is never charged; a market index that averages sub-dollar or newly
listed tickers; a null that shifts each event independently; a retention rule
read with the wrong inequality; and an observed stage that runs without its
receipt.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import item408_event_study_r5 as c  # noqa: E402
import run_t_r5 as r  # noqa: E402


# ---------------------------------------------------------------------------
# fixture helpers
# ---------------------------------------------------------------------------
def frame(start, n, rate=0.0, px=50.0, unadj=None, vol=1e6, drop=()):
    """Business-day bars with a constant log drift; `drop` removes positions."""
    idx = pd.bdate_range(start, periods=n)
    ca = px * np.exp(rate * np.arange(n, dtype=float))
    cu = ca if unadj is None else np.full(n, float(unadj))
    df = pd.DataFrame({"closeadj": ca, "closeunadj": cu,
                       "volume": np.full(n, float(vol))},
                      index=[str(d.date()) for d in idx])
    if drop:
        df = df.drop(index=[df.index[i] for i in drop])
    return df


def holed(df, positions):
    """Remove rows but remember their dates, so the CALENDAR keeps the day.

    A date no ticker trades on is not a trading day at all; a hole only exists
    relative to a calendar that still carries it.
    """
    dates = [df.index[i] for i in positions]
    return df.drop(index=dates), dates


def make_book(frames, extra_dates=()):
    """A PriceBook over the given per-ticker frames. The calendar is their union."""
    tickers = sorted(frames)
    dates = sorted({d for f in frames.values() for d in f.index} | set(extra_dates))
    pos = {d: i for i, d in enumerate(dates)}
    cal = np.array([(pd.Timestamp(d) - pd.Timestamp("1970-01-01")).days
                    for d in dates], dtype=np.int32)
    code, dayidx, ca, cu, vol = [], [], [], [], []
    for ci, t in enumerate(tickers):
        f = frames[t].sort_index()
        for d in f.index:
            code.append(ci)
            dayidx.append(pos[d])
            ca.append(float(f.at[d, "closeadj"]))
            cu.append(float(f.at[d, "closeunadj"]))
            vol.append(float(f.at[d, "volume"]))
    return c.PriceBook(np.asarray(tickers, dtype=object),
                       np.asarray(code, dtype=np.int32),
                       np.asarray(dayidx, dtype=np.int32),
                       np.asarray(ca, dtype=float),
                       np.asarray(cu, dtype=np.float32),
                       np.asarray(vol, dtype=np.float32), cal)


EV_DEFAULTS = {
    "adsh": "0000000001-23-000001", "cik": "0000000001", "form": "10-Q",
    "file_date": "2024-03-15", "period_ending": "2024-01-31",
    "context_ref": "c1", "individual_name": "A Person",
    "individual_title": "Chief Executive Officer",
    "adopted_flag": None, "terminated_flag": None,
    "adoption_date": "", "termination_date": "2024-02-01",
    "expiration_date": "", "adoption_date_raw": "",
    "termination_date_raw": "February1, 2024", "expiration_date_raw": "",
    "duration_text": "", "securities_agg_available": "",
    "rule10b51_adopted_flag": False, "rule10b51_terminated_flag": True,
    "nonrule10b51_adopted_flag": False, "nonrule10b51_terminated_flag": False,
    "raw_text": "terminated a plan for the sale of shares",
}


def ev_rows(*overrides):
    rows = []
    for i, o in enumerate(overrides):
        d = dict(EV_DEFAULTS)
        d["adsh"] = "0000000001-23-%06d" % (i + 1)
        d.update(o)
        rows.append(d)
    return pd.DataFrame(rows)


def sep_frame(pairs):
    """pairs: [(permaticker, ticker, cik_int)]"""
    return pd.DataFrame([{
        "table": "SEP", "permaticker": str(p), "ticker": t,
        "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action="
                      "getcompany&CIK=%010d" % k} for p, t, k in pairs])


def flat_market(book, value=0.0):
    return {"mean_log_return": np.full(book.L, float(value)),
            "n_tickers": np.full(book.L, 10, dtype=np.int64)}


# ---------------------------------------------------------------------------
# 1. the row clauses, each binding alone
# ---------------------------------------------------------------------------
def test_termination_flag_clause_binds_alone():
    ev = ev_rows({}, {"rule10b51_terminated_flag": False},
                 {"rule10b51_terminated_flag": None},
                 {"rule10b51_terminated_flag": False,
                  "nonrule10b51_terminated_flag": True})
    kept, n = c.resolve_rows(ev)
    assert n["rows_in_vintage"] == 4
    assert n["A_termination_flag"] == 2          # row 0 and the non-rule row
    assert set(kept["adsh"]) == {"0000000001-23-000001", "0000000001-23-000004"}


def test_adoption_flag_clause_binds_alone_a_modification_is_not_a_termination():
    ev = ev_rows({}, {"rule10b51_adopted_flag": True},
                 {"nonrule10b51_adopted_flag": True})
    kept, n = c.resolve_rows(ev)
    assert n["A_termination_flag"] == 3
    assert n["B_no_adoption_flag"] == 1
    assert n["B_removed_modifications"] == 2


def test_parseable_termination_date_clause_binds_alone():
    ev = ev_rows({}, {"termination_date": ""}, {"termination_date": None},
                 {"termination_date": "not a date"})
    kept, n = c.resolve_rows(ev)
    assert n["B_no_adoption_flag"] == 4
    assert n["C_parseable_T"] == 1


def test_lag_clause_binds_alone_a_scheduled_end_is_excluded():
    ev = ev_rows({},                                     # T before D
                 {"termination_date": "2024-03-15"},     # T == D, lag 0, kept
                 {"termination_date": "2024-03-16"})     # T after D, dropped
    kept, n = c.resolve_rows(ev)
    assert n["C_parseable_T"] == 3
    assert n["D_lag_ge_zero"] == 2
    assert sorted(kept["lag_days"]) == [0, 43]


def test_form_clause_binds_alone_amendments_are_not_the_registered_forms():
    ev = ev_rows({}, {"form": "10-K"}, {"form": "10-Q/A"}, {"form": "10-K/A"},
                 {"form": "20-F"}, {"form": "10-KT"})
    kept, n = c.resolve_rows(ev)
    assert n["D_lag_ge_zero"] == 6
    assert n["E_form_10Q_or_10K"] == 2
    assert sorted(kept["form"]) == ["10-K", "10-Q"]


@pytest.mark.parametrize("text,want", [
    ("terminated a plan to purchase shares", "purchase_only"),
    ("a plan to buy stock was terminated", "purchase_only"),
    ("terminated a plan for the sale of shares", "sale_only"),
    ("intends to sell shares", "sale_only"),
    ("shares sold under the plan", "sale_only"),
    ("a plan to purchase and to sell shares", "both"),
    ("the arrangement was terminated", "neither"),
    ("", "neither"),
    (None, "neither"),
])
def test_direction_rule_classifies_all_four_families(text, want):
    assert c.direction_class(text) == want


def test_direction_clause_excludes_purchase_only_and_keeps_both_and_neither():
    ev = ev_rows({"raw_text": "purchase plan terminated"},
                 {"raw_text": "sale plan terminated"},
                 {"raw_text": "purchase and sale plan terminated"},
                 {"raw_text": ""})
    kept, n = c.resolve_rows(ev)
    assert n["F_direction_split"] == {"purchase_only": 1, "sale_only": 1,
                                      "both": 1, "neither": 1}
    assert n["F_after_purchase_exclusion"] == 3
    assert "purchase_only" not in set(kept["direction"])


# ---------------------------------------------------------------------------
# 2. one event per filing, the calendar, mapping
# ---------------------------------------------------------------------------
def test_several_individuals_in_one_filing_are_one_event():
    ev = ev_rows({"adsh": "X-1", "individual_title": "Chief Executive Officer"},
                 {"adsh": "X-1", "individual_title": "Director"},
                 {"adsh": "X-2", "individual_title": "President"})
    ev["adsh"] = ["X-1", "X-1", "X-2"]
    book = make_book({"AAA": frame("2024-01-01", 200)})
    kept, n = c.resolve_rows(ev)
    fil, n = c.collapse_to_filings(kept, book, n)
    assert n["G_one_event_per_filing"] == 2
    row = fil.loc[fil["adsh"] == "X-1"].iloc[0]
    assert row["n_individuals"] == 2
    assert "Chief Executive Officer" in row["titles"] and "Director" in row["titles"]


def test_calendar_rolls_a_weekend_filing_date_forward():
    book = make_book({"AAA": frame("2024-03-01", 40)})
    sat, mon = "2024-03-16", "2024-03-18"
    assert sat not in [book.date_str(j) for j in range(book.L)]
    assert book.date_str(book.roll_forward(sat)) == mon
    assert book.date_str(book.roll_forward(mon)) == mon


def test_a_weekend_termination_date_rolls_forward_too():
    book = make_book({"AAA": frame("2024-03-01", 40)})
    # T on a Sunday is positioned at the next trading day, as the seal says
    assert book.date_str(book.roll_forward("2024-03-17")) == "2024-03-18"
    assert book.roll_forward("2099-01-01") is None


def test_cik_is_extracted_from_the_browse_edgar_url():
    assert c.extract_cik("https://www.sec.gov/cgi-bin/browse-edgar?action="
                         "getcompany&CIK=0001027077") == 1027077
    assert c.extract_cik("no cik here") is None
    assert c.extract_cik(None) is None


def test_mapping_tie_break_prefers_the_ticker_that_trades_then_the_longer_history():
    long_f = frame("2023-01-02", 400)
    short_f = frame("2024-01-01", 60)
    dead_f = frame("2023-01-02", 100)          # stops well before D+1
    book = make_book({"LONG": long_f, "SHORT": short_f, "DEAD": dead_f})
    d_idx = book.roll_forward("2024-03-15")
    fil = pd.DataFrame([{"adsh": "X-1", "cik": 7, "form": "10-Q",
                         "file_date": pd.Timestamp("2024-03-15"),
                         "T": pd.Timestamp("2024-02-01"),
                         "T_last": pd.Timestamp("2024-02-01"),
                         "n_individuals": 1, "titles": "Director",
                         "direction_mix": "sale_only", "D_idx": int(d_idx)}])
    sep = sep_frame([(1, "DEAD", 7), (2, "SHORT", 7), (3, "LONG", 7)])
    mp, n = c.map_issuers(fil, sep, book, {})
    assert n["H_mapping"]["multi_permaticker"] == 1
    assert mp["ticker"].tolist() == ["LONG"]       # trades on D+1 and has more rows


def test_mapping_drops_a_cik_with_no_sep_row():
    book = make_book({"AAA": frame("2024-01-01", 200)})
    fil = pd.DataFrame([{"adsh": "X-1", "cik": 99, "form": "10-Q",
                         "file_date": pd.Timestamp("2024-06-03"),
                         "T": pd.Timestamp("2024-05-01"),
                         "T_last": pd.Timestamp("2024-05-01"),
                         "n_individuals": 1, "titles": "", "direction_mix": "",
                         "D_idx": 10}])
    mp, n = c.map_issuers(fil, sep_frame([(1, "AAA", 7)]), book, {})
    assert len(mp) == 0
    assert n["H_mapping"]["dropped_no_usable_ticker"] == 1


def test_two_filings_for_one_issuer_on_one_day_collapse_to_one_event():
    book = make_book({"AAA": frame("2024-01-01", 200)})
    base = {"cik": 7, "form": "10-Q", "file_date": pd.Timestamp("2024-06-03"),
            "T": pd.Timestamp("2024-05-01"), "T_last": pd.Timestamp("2024-05-01"),
            "n_individuals": 1, "titles": "", "direction_mix": "", "D_idx": 50}
    fil = pd.DataFrame([dict(base, adsh="X-2"), dict(base, adsh="X-1")])
    mp, n = c.map_issuers(fil, sep_frame([(1, "AAA", 7)]), book, {})
    assert n["I_unique_on_permaticker_D"] == 1 and n["I_collapsed"] == 1
    assert mp["adsh"].tolist() == ["X-1"]


# ---------------------------------------------------------------------------
# 3. history, tradability, completeness
# ---------------------------------------------------------------------------
def _one_filing(book, ticker, file_date, T="2024-02-01"):
    d = book.roll_forward(file_date)
    return pd.DataFrame([{"adsh": "X-1", "cik": 7, "form": "10-Q",
                          "file_date": pd.Timestamp(file_date),
                          "T": pd.Timestamp(T), "T_last": pd.Timestamp(T),
                          "n_individuals": 1, "titles": "Director",
                          "direction_mix": "sale_only", "D_idx": int(d),
                          "permaticker": "1", "ticker": ticker}])


def test_history_clause_binds_alone_at_250_rows_before_T():
    for n_pre, kept in ((c.HISTORY_MIN_ROWS, 1), (c.HISTORY_MIN_ROWS - 1, 0)):
        f = frame("2024-02-01", 120)                     # from T forward
        pre = pd.bdate_range(end="2024-01-31", periods=n_pre)
        back = pd.DataFrame({"closeadj": 50.0, "closeunadj": 50.0,
                             "volume": 1e6},
                            index=[str(d.date()) for d in pre])
        book = make_book({"AAA": pd.concat([back, f])})
        fil = _one_filing(book, "AAA", "2024-02-05")
        res, counts = c.apply_price_filters(fil, book, {})
        assert counts["J_history_250_rows_before_T"] == kept, n_pre


def _deep_book(unadj=50.0, vol=1e6, hole_at=None):
    pre = pd.bdate_range(end="2024-01-31", periods=400)
    back = pd.DataFrame({"closeadj": 50.0, "closeunadj": unadj, "volume": vol},
                        index=[str(d.date()) for d in pre])
    full = pd.concat([back, frame("2024-02-01", 140, unadj=unadj, vol=vol)])
    holes = []
    if hole_at is not None:
        full, holes = holed(full, (hole_at,))
    return make_book({"AAA": full}, extra_dates=holes)


def test_price_floor_clause_binds_alone_on_D_plus_1():
    for px, kept in ((c.PRICE_FLOOR, 1), (c.PRICE_FLOOR - 0.01, 0)):
        book = _deep_book(unadj=px)
        res, counts = c.apply_price_filters(_one_filing(book, "AAA", "2024-02-05"),
                                           book, {})
        assert counts["J_history_250_rows_before_T"] == 1
        assert counts["K_price_floor_on_D_plus_1"] == kept, px


def test_dollar_volume_clause_binds_alone_at_the_registered_floor():
    px = 10.0
    for vol, kept in ((c.DOLLAR_VOL_FLOOR / px, 1),
                      (c.DOLLAR_VOL_FLOOR / px - 1.0, 0)):
        book = _deep_book(unadj=px, vol=vol)
        res, counts = c.apply_price_filters(_one_filing(book, "AAA", "2024-02-05"),
                                           book, {})
        assert counts["K_price_floor_on_D_plus_1"] == 1
        assert counts["K_dollar_volume_floor"] == kept, vol


def test_dollar_volume_median_reads_exactly_the_twenty_trading_days_ending_D():
    book = _deep_book(unadj=10.0, vol=1e5)
    j = book.roll_forward("2024-03-01")
    got = book.dollar_volume_median("AAA", j)
    assert got == pytest.approx(1e6)
    # the window is 20 wide, not 19 and not 21
    assert c.DOLLAR_VOL_WINDOW == 20


def test_completeness_clause_binds_alone_one_missing_day_drops_the_event():
    book = _deep_book()
    fil = _one_filing(book, "AAA", "2024-02-05")
    res, counts = c.apply_price_filters(fil, book, {})
    assert counts["L_complete_D1_to_D61"] == 1 and len(res) == 1
    d = int(fil["D_idx"].iloc[0])
    gap = _deep_book(hole_at=d + 30)           # one bar inside D+1..D+61 removed
    fil2 = _one_filing(gap, "AAA", "2024-02-05")
    assert int(fil2["D_idx"].iloc[0]) == d     # the calendar still carries the day
    res2, counts2 = c.apply_price_filters(fil2, gap, {})
    assert counts2["K_dollar_volume_floor"] == 1
    assert counts2["L_complete_D1_to_D61"] == 0


def test_complete_between_rejects_a_window_running_off_the_calendar():
    book = _deep_book()
    assert book.complete_between("AAA", 10, 70) is True
    assert book.complete_between("AAA", book.L - 5, book.L + 60) is False
    assert book.complete_between("MISSING", 10, 70) is False


# ---------------------------------------------------------------------------
# 4. the market index
# ---------------------------------------------------------------------------
def test_market_index_excludes_sub_dollar_and_short_history_tickers():
    n = 300
    good = frame("2023-01-02", n, rate=0.01, px=50.0, unadj=50.0)
    cheap = frame("2023-01-02", n, rate=0.05, px=0.50, unadj=0.50)
    young = frame("2023-01-02", n, rate=0.09, px=50.0, unadj=50.0)
    young = young.iloc[-100:]                     # only 100 rows
    book = make_book({"GOOD": good, "CHEAP": cheap, "YOUNG": young})
    m = c.build_market_index(book)
    j = book.L - 1
    assert m["n_tickers"][j] == 1
    assert m["mean_log_return"][j] == pytest.approx(0.01, abs=1e-12)
    # before the 250th row nobody contributes
    assert m["n_tickers"][c.MARKET_MIN_PRIOR_ROWS - 1] == 0
    assert m["n_tickers"][c.MARKET_MIN_PRIOR_ROWS] == 1


def test_market_index_starts_contributing_only_after_250_prior_rows():
    book = make_book({"GOOD": frame("2023-01-02", 300, rate=0.02, unadj=50.0)})
    m = c.build_market_index(book)
    first = int(np.flatnonzero(m["n_tickers"] > 0)[0])
    assert first == c.MARKET_MIN_PRIOR_ROWS


def test_abnormal_return_is_the_stock_minus_the_market():
    book = make_book({"AAA": frame("2023-01-02", 300, rate=0.03, unadj=50.0)})
    mkt = flat_market(book, 0.01)
    ab, ok = c.abnormal_series(book, "AAA", mkt)
    assert ok[1:].all() and not ok[0]
    assert ab[5] == pytest.approx(0.02, abs=1e-12)


def test_abnormal_return_is_absent_across_a_hole_in_the_history():
    f, holes = holed(frame("2023-01-02", 300, rate=0.03, unadj=50.0), (100,))
    book = make_book({"AAA": f}, extra_dates=holes)
    ab, ok = c.abnormal_series(book, "AAA", flat_market(book, 0.0))
    assert not ok[100]            # no row at all on that trading day
    assert not ok[101]            # a row, but its previous row is two days back
    assert ok[99] and ok[102]


# ---------------------------------------------------------------------------
# 5. window sums and the cost
# ---------------------------------------------------------------------------
def _vote_setup(rate=0.001, n=400):
    book = make_book({"AAA": frame("2023-01-02", n, rate=rate, unadj=50.0)})
    wb = c.WindowBook(book, ["AAA"], flat_market(book, 0.0))
    groups = np.array([0])
    d_idx = np.array([100])
    return book, wb, groups, d_idx


def test_voted_window_is_sixty_returns_from_D2_to_D61_hand_computed():
    rate = 0.001
    book, wb, g, d = _vote_setup(rate)
    s, n = wb.window(g, d + c.VOTE_FROM, d + c.VOTE_TO)
    assert int(n[0]) == c.VOTE_N == 60
    assert float(s[0]) == pytest.approx(60 * rate, abs=1e-12)


def test_the_cost_is_twenty_bps_round_trip_subtracted_once():
    rate = 0.001
    book, wb, g, d = _vote_setup(rate)
    net = c.voted_returns(wb, g, d)
    assert c.COST_FRACTION == pytest.approx(0.0020)
    assert float(net[0]) == pytest.approx(60 * rate - 0.0020, abs=1e-12)


def test_an_incomplete_voted_window_is_not_scored():
    f, holes = holed(frame("2023-01-02", 400, rate=0.001, unadj=50.0), (130,))
    book = make_book({"AAA": f}, extra_dates=holes)
    wb = c.WindowBook(book, ["AAA"], flat_market(book, 0.0))
    net = c.voted_returns(wb, np.array([0]), np.array([100]))
    assert not np.isfinite(net[0])


def test_the_twenty_day_window_is_twenty_returns():
    rate = 0.002
    book, wb, g, d = _vote_setup(rate)
    s, n = wb.window(g, d + c.VOTE_FROM, d + c.SHORT_TO)
    assert int(n[0]) == c.SHORT_N == 20
    assert float(s[0]) == pytest.approx(20 * rate, abs=1e-12)


def test_the_pre_termination_window_is_twenty_one_returns_ending_at_T():
    rate = 0.004
    book, wb, g, d = _vote_setup(rate)
    t = np.array([80])
    s, n = wb.window(g, t - (c.PRETERM_LAG - 1), t)
    assert int(n[0]) == c.PRETERM_LAG == 21
    assert float(s[0]) == pytest.approx(21 * rate, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. the common-shift null
# ---------------------------------------------------------------------------
def _null_setup(n_events=2, n=1200, rate=0.001):
    frames = {"AAA": frame("2019-01-02", n, rate=rate, unadj=50.0),
              "BBB": frame("2019-01-02", n, rate=rate / 2, unadj=50.0)}
    book = make_book(frames)
    wb = c.WindowBook(book, ["AAA", "BBB"], flat_market(book, 0.0))
    groups = np.array([0, 1])[:n_events]
    d_idx = np.array([600, 620])[:n_events]
    wb.set_blocked(groups, d_idx - 20, d_idx)
    return book, wb, groups, d_idx


def test_draw_shift_covers_both_directions_and_never_the_excluded_middle():
    rng = np.random.default_rng(0)
    ks = np.array([c.draw_shift(rng) for _ in range(4000)])
    assert (np.abs(ks) >= c.SHIFT_LO).all()
    assert (np.abs(ks) <= c.SHIFT_HI).all()
    assert (ks < 0).any() and (ks > 0).any()
    assert abs((ks < 0).mean() - 0.5) < 0.05


def test_one_common_shift_per_replicate_not_one_per_event(monkeypatch):
    book, wb, g, d = _null_setup()
    seq = [200, -300, 250]
    calls = []

    def fake(rng):
        k = seq[len(calls) % len(seq)]
        calls.append(k)
        return k

    monkeypatch.setattr(c, "draw_shift", fake)
    out = c.common_shift_null(wb, g, d, 3, np.random.default_rng(1))
    assert calls == seq                       # exactly one draw per replicate
    for i, k in enumerate(seq):
        m, ret, used = c.shift_mcar(wb, g, d, k, len(d))
        assert out["mcar"][i] == pytest.approx(m)
        assert used == len(d)                 # the SAME k moved every event


def test_an_overlapping_window_is_dropped_from_the_replicate():
    book, wb, g, d = _null_setup()
    m_far, ret_far, used_far = c.shift_mcar(wb, g, d, 300, len(d))
    assert used_far == 2 and ret_far == 1.0
    # a small shift leaves the window inside the real events' T-60..D+61 region
    wb.set_blocked(g, d - 20, d)
    m_near, ret_near, used_near = c.shift_mcar(wb, g, d, 1, len(d))
    assert used_near == 0 and ret_near == 0.0


def test_a_replicate_retaining_under_ninety_percent_is_redrawn(monkeypatch):
    book, wb, g, d = _null_setup()
    # block ONE issuer's whole calendar, so any shift drops half the events
    wb.blocked_cum[0, 1:] = np.cumsum(np.ones(wb.L, dtype=np.int32))
    bad = c.shift_mcar(wb, g, d, 200, len(d))
    assert bad[1] == pytest.approx(0.5) and bad[1] < c.RETENTION_MIN
    with pytest.raises(AssertionError, match="could not retain"):
        c.common_shift_null(wb, g, d, 1, np.random.default_rng(2), max_redraw=5)


def test_retention_is_read_as_at_or_above_ninety_percent(monkeypatch):
    book, wb, g, d = _null_setup()
    seq = [200]
    monkeypatch.setattr(c, "draw_shift", lambda rng: seq[0])
    out = c.common_shift_null(wb, g, d, 2, np.random.default_rng(3))
    assert out["redraws"] == 0
    m, ret, used = c.shift_mcar(wb, g, d, 200, len(d))
    assert ret == 1.0 >= c.RETENTION_MIN
    assert np.allclose(out["mcar"], m)


def test_a_shift_running_off_the_calendar_drops_the_event():
    book, wb, g, d = _null_setup()
    m, ret, used = c.shift_mcar(wb, g, np.array([wb.L - 10, 620]), 300, 2)
    assert used == 0 or ret < 1.0


def test_null_p_is_the_share_reaching_the_observed_value():
    null = np.array([0.0, 0.1, 0.2, 0.3])
    assert c.null_p(0.25, null) == pytest.approx(0.25)
    assert c.null_p(0.1, null) == pytest.approx(0.75)      # >= is inclusive
    assert c.null_p(1.0, null) == 0.0


def test_mde_formula_is_the_registered_expression():
    rng = np.random.default_rng(5)
    d = rng.normal(size=5000)
    want = (np.percentile(d, 95) - np.median(d)) + c.Z80 * d.std(ddof=1)
    assert c.mde_from_null(d) == pytest.approx(want)
    assert c.Z80 == 0.8416


def test_mde_is_invariant_to_a_constant_offset():
    d = np.random.default_rng(6).normal(size=2000)
    assert c.mde_from_null(d) == pytest.approx(c.mde_from_null(d + 7.5))


# ---------------------------------------------------------------------------
# 7. the control predicates and the plant
# ---------------------------------------------------------------------------
def test_band_ok_is_a_closed_band():
    assert c.band_ok(0.01, (0.01, 0.10)) and c.band_ok(0.10, (0.01, 0.10))
    assert not c.band_ok(0.009, (0.01, 0.10))
    assert not c.band_ok(0.101, (0.01, 0.10))


def test_detection_ok_is_at_or_above_the_floor():
    assert c.detection_ok(0.80, 0.80) and c.detection_ok(0.81, 0.80)
    assert not c.detection_ok(0.79, 0.80)


def test_plant_drift_is_uniform_zero_to_twice_X_with_mean_X():
    X = 0.03
    v = c.plant_drift(200000, X, np.random.default_rng(7))
    assert (v >= 0).all() and (v <= 2 * X).all()
    assert v.mean() == pytest.approx(X, abs=5e-4)


def test_X_is_calibrated_on_one_draw_to_the_registered_planted_mean():
    cal = c.calibrate_X(500, 0.03, 4242)
    assert cal["one_draw_planted_mean"] == pytest.approx(0.03, abs=1e-12)
    assert cal["X"] == pytest.approx(0.03 / cal["reference_mean_u"])


def test_the_plant_is_a_leak_it_lands_in_the_very_window_the_vote_reads():
    """Named deliberately: the plant rides the voted window, so it must be seen."""
    book, wb, g, d = _null_setup(n_events=2, n=1400)
    null = c.common_shift_null(wb, g, d, 200, np.random.default_rng(8))["mcar"]
    obs = float(c.voted_returns(wb, g, d).mean())
    p_plain = c.null_p(obs, null)
    p_planted = c.null_p(obs + 0.30, null)
    assert p_planted == 0.0
    assert p_planted < p_plain
    src = (ROOT / "tools" / "run_t_r5.py").read_text()
    assert "_internal_obs + float(drift.mean())" in src


def test_the_registration_names_the_leak_and_the_engine_repeats_it():
    doc = ast.get_docstring(ast.parse(
        (ROOT / "tools" / "item408_event_study_r5.py").read_text()))
    fn = c.plant_drift.__doc__
    assert "DELIBERATE LEAK" in fn
    assert "voted window" in fn
    assert "observed" in doc


# ---------------------------------------------------------------------------
# 8. the verdict ladder
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p,m,want", [
    (0.01, 0.020, "SOMETHING"),
    (0.05, 0.015, "SOMETHING"),
    (0.01, 0.0149, "SIGNIFICANT_IMMATERIAL"),
    (0.05, 0.0001, "SIGNIFICANT_IMMATERIAL"),
    (0.01, -0.01, "NOTHING"),
    (0.06, 0.0001, "SUGGESTIVE"),
    (0.10, 0.5, "SUGGESTIVE"),
    (0.06, -0.001, "NOTHING"),
    (0.11, 0.9, "NOTHING"),
])
def test_the_verdict_ladder_is_the_registered_one(p, m, want):
    assert c.verdict_of(p, m, 0.05) == want


# ---------------------------------------------------------------------------
# 9. registration loading, prose pins, halves
# ---------------------------------------------------------------------------
def test_load_registration_refuses_a_sha_mismatch(tmp_path):
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"a": 1}))
    assert c.load_registration(p, c.sha256_file(p))["a"] == 1
    with pytest.raises(AssertionError, match="registration sha256 mismatch"):
        c.load_registration(p, "0" * 64)


def test_half_bounds_are_read_out_of_the_seal_not_hardcoded():
    reg = {"event_definition": {"halves":
           "half A: 2023-07-01 <= D <= 2025-03-31; half B: 2025-04-01 <= D <= "
           "2026-06-30, UNOPENED; holdout: D >= 2026-07-01, SEALED"}}
    hb = c.half_bounds(reg)
    assert hb["half_A"] == (pd.Timestamp("2023-07-01"), pd.Timestamp("2025-03-31"))
    assert hb["half_B"][1] == pd.Timestamp("2026-06-30")
    assert hb["holdout"][0] == pd.Timestamp("2026-07-01")
    with pytest.raises(AssertionError, match="exactly five dates"):
        c.half_bounds({"event_definition": {"halves": "half A: 2023-07-01"}})


REG_PROSE_FALLBACK = {
    "event_definition": {
        "row_filter": "AND form in {10-Q, 10-K}",
        "direction_filter": "contains 'purchase' or 'buy' and contains none of "
                            "'sale', 'sell', 'sold'",
        "history": "at least 250 bulk_stocks rows for the ticker dated before T",
        "tradability": "closeunadj on D+1 >= 2.00 USD and the median of "
                       "closeunadj times volume over the 20 trading days ending "
                       "D >= 500000 USD",
        "completeness": "closeadj present on every trading day from D+1 to D+61 "
                        "inclusive",
        "minimum_events": "if half A holds fewer than 200 events after all "
                          "filters, HALT at resolution",
    },
    "returns": {
        "market": "closeunadj >= 1.00 USD on the prior day and at least 250 "
                  "prior rows",
        "voted_window": "the sum of the 60 abnormal daily log returns from D+2 "
                        "to D+61",
        "cost": "20 bps round trip (10 a side) subtracted from every event's "
                "voted-window return",
        "reported_windows": "the 20-day window D+2 to D+21; the pre-termination "
                            "window from the close of T-21 to the close of T; "
                            "the mean abnormal path at 5, 10, 20, 40 and 60 days",
    },
    "voting_null": {"mechanism": "k drawn uniformly from the union of [-378, "
                                 "-126] and [126, 378]; a shifted window lying "
                                 "inside T-60 to D+61 of any real event of the "
                                 "same issuer is dropped from that replicate; a "
                                 "replicate retaining fewer than 90 percent of "
                                 "events is redrawn"},
    "order_of_operations": ["+ 0.8416 * sd(null mCAR)"],
    "verdict": {"materiality": "mCAR >= 0.015 net", "SUGGESTIVE": "p in (0.05, 0.10]"},
}


def _registration_text():
    for p in (ROOT / "registrations" / "r5" / "registration_v1.json",
              ROOT / "logs" / "r5" / "registration_v1.json"):
        if p.exists():
            return p.read_text(), str(p)
    return json.dumps(REG_PROSE_FALLBACK), "fallback"


def test_registration_prose_pins_the_constants():
    """Every integer the seal states in prose must match the engine's constant."""
    text, src = _registration_text()
    assert "form in {%s, %s}" % (c.FORMS[0], c.FORMS[1]) in text
    assert "'%s' or '%s'" % (c.PURCHASE_WORDS[0], c.PURCHASE_WORDS[1]) in text
    assert "none of '%s', '%s', '%s'" % c.SALE_WORDS in text
    assert "at least %d bulk_stocks rows" % c.HISTORY_MIN_ROWS in text
    assert "closeunadj on D+1 >= %.2f USD" % c.PRICE_FLOOR in text
    assert "over the %d trading days ending D >= %d USD" % (
        c.DOLLAR_VOL_WINDOW, int(c.DOLLAR_VOL_FLOOR)) in text
    assert "from D+%d to D+%d inclusive" % (c.COMPLETE_FROM, c.COMPLETE_TO) in text
    assert "closeunadj >= %.2f USD on the prior day" % c.MARKET_PRICE_FLOOR in text
    assert "at least %d prior rows" % c.MARKET_MIN_PRIOR_ROWS in text
    assert "the sum of the %d abnormal daily log returns from D+%d to D+%d" % (
        c.VOTE_N, c.VOTE_FROM, c.VOTE_TO) in text
    assert "the %d-day window D+%d to D+%d" % (
        c.SHORT_N, c.VOTE_FROM, c.SHORT_TO) in text
    assert "close of T-%d to the close of T" % c.PRETERM_LAG in text
    assert "at %d, %d, %d, %d and %d days" % c.PATH_POINTS in text
    assert "%d bps round trip" % int(c.COST_ROUND_TRIP_BPS) in text
    assert "union of [-%d, -%d] and [%d, %d]" % (
        c.SHIFT_HI, c.SHIFT_LO, c.SHIFT_LO, c.SHIFT_HI) in text
    assert "inside T-%d to D+%d" % (c.CONTAM_BACK, c.CONTAM_FWD) in text
    assert "fewer than %d percent of events is redrawn" % int(
        100 * c.RETENTION_MIN) in text
    assert "%s * sd(null mCAR)" % c.Z80 in text
    assert "fewer than %d events" % c.MIN_EVENTS in text
    assert "mCAR >= %s net" % c.MATERIALITY in text
    assert "p in (%.2f, %.2f]" % (0.05, c.SUGGESTIVE_HI) in text


def test_the_prose_pin_binds_to_the_tracked_seal_once_it_exists():
    tracked = ROOT / "registrations" / "r5" / "registration_v1.json"
    _, src = _registration_text()
    if tracked.exists():
        assert src == str(tracked)
    else:
        assert src in ("fallback", str(ROOT / "logs" / "r5" / "registration_v1.json"))


# ---------------------------------------------------------------------------
# 10. the seal's data pins, read from the seal
# ---------------------------------------------------------------------------
def test_data_paths_and_pins_are_read_out_of_the_seal():
    reg = {"data": {
        "events": "data/edgar_item408/20261104/events.parquet sha256 %s; tracked "
                  "manifest manifests/edgar_item408_20261104.json sha256 %s"
                  % ("a" * 64, "b" * 64),
        "prices": "Sharadar snapshot_20261004p bulk_stocks.parquet sha256 %s"
                  % ("c" * 64),
        "tickers": "bulk_tickers.parquet sha256 %s" % ("d" * 64)}}
    p = r.data_paths_and_pins(reg)
    assert p["events"][1] == "a" * 64 and p["manifest"][1] == "b" * 64
    assert p["stocks"][1] == "c" * 64 and p["tickers"][1] == "d" * 64
    assert p["snapshot"] == "snapshot_20261004p"
    assert p["stocks"][0].name == "bulk_stocks.parquet"
    assert "snapshot_20261004p" in str(p["tickers"][0])


def test_a_data_pin_mismatch_is_refused(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"hello")
    with pytest.raises(AssertionError, match="sha256 mismatch"):
        r.assert_data_pins({"events": (f, "0" * 64)})


# ---------------------------------------------------------------------------
# 11. stage gating
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
    assert rc == 2 and "REFUSED" in out and "live file is" in out


def test_receipt_carries_the_registered_fields():
    src = (ROOT / "tools" / "run_t_r5.py").read_text()
    for field in ("registration_sha", "n_events", "mde", "known_null_share",
                  "X_calibrated", "detection", "utc"):
        assert '"%s"' % field in src


def test_halt_writes_the_same_figures_and_exits_non_zero():
    src = (ROOT / "tools" / "run_t_r5.py").read_text()
    assert "def halt_out(" in src
    assert "return 15" in src
    assert "RECEIPT.unlink()" in src
    assert "MIN_EVENTS" in src


def test_the_runner_halts_below_the_registered_minimum_event_count():
    """The <200 halt is reached before any null is formed."""
    src = (ROOT / "tools" / "run_t_r5.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "stage_controls")
    body = ast.dump(fn)
    i_min = body.index("MIN_EVENTS")
    i_null = body.index("common_shift_null")
    assert i_min < i_null, "the minimum-event halt must precede the null"


# ---------------------------------------------------------------------------
# 12. housekeeping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", ["item408_event_study_r5.py", "run_t_r5.py"])
def test_no_credential_is_read_by_the_new_tools(tool):
    path = ROOT / "tools" / tool
    src = path.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    code = src.replace(doc, "")
    for forbidden in ("dotenv", "load_dotenv", ".env", "API_KEY", "api_key",
                      "secret", "requests", "urllib", "NASDAQ_DATA_LINK"):
        assert forbidden not in code, (tool, forbidden)


@pytest.mark.parametrize("tool", ["item408_event_study_r5.py", "run_t_r5.py"])
def test_new_tools_import_nothing_from_the_factory_tree(tool):
    src = (ROOT / "tools" / tool).read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            for a in n.names:
                assert "Monte_Carlo" not in a.name
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or "").startswith(("src.", "1_data", "2_research"))
    assert "Monte_Carlo_Strategies" not in src


def test_the_probe_no_longer_uses_the_deprecated_utcnow():
    src = (ROOT / "tools" / "edgar_item408_probe.py").read_text()
    assert "datetime.utcnow()" not in src
    assert "datetime.now(timezone.utc)" in src
    assert "timezone" in src.split("\n\n")[0] or "import timezone" in src or \
        "from datetime import" in src
