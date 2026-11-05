"""Tests for the T-R6 engine and runner (tools/form4_event_study_r6.py,
tools/run_t_r6.py).

Fixtures only. Nothing here reads the frozen snapshot, forms a return of a real
event, or touches the tracked ledger. Every registered resolution clause is made
to bind ALONE: the fixture differs from a passing fixture in exactly the one
field that clause reads, so a dropped clause cannot hide behind its neighbours.
Where a rule is arithmetic it is checked against a hand-computable answer; where
it is structural (one permutation per replicate, the truncation clause, the
benchmark's three floors) the test constructs the violation and insists the
engine refuses it.

The seven mutants this battery is built to catch, each with its own test:
  M1 a null that SHIFTS DATES instead of permuting issuers;
  M2 a bar measured FROM ZERO instead of from the null's median;
  M3 a TRUNCATED window scored as a delisting;
  M4 a benchmark with its three prior-day floors DROPPED;
  M5 a clustering window dropped, so one issuer's whole decade is one event;
  M6 the lag clause with its inequality reversed;
  M7 a cost that is never charged.
Each mutant is exercised against the real engine, which must disagree with it.
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
import form4_event_study_r6 as c  # noqa: E402
import item408_event_study_r5 as r5  # noqa: E402
import run_t_r6 as r  # noqa: E402


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
    """Remove rows but remember their dates, so the CALENDAR keeps the day."""
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


def flat_index(book, value=0.0):
    return {"mean_log_return": np.full(book.L, float(value)),
            "n_tickers": np.full(book.L, 10, dtype=np.int64)}


PUR_DEFAULTS = {
    "accession": "0000000001-10-000001",
    "filing_date": "2010-03-15", "trans_date": "2010-03-12",
    "filing_lag_days": 3,
    "issuer_cik": "0000000001", "issuer_name": "ACME CORP",
    "owner_cik": "0000000100", "owner_name": "A Person",
    "value": 50000.0, "is_routine": False,
}


def pur_rows(*overrides):
    rows = []
    for i, o in enumerate(overrides):
        d = dict(PUR_DEFAULTS)
        d["accession"] = "0000000001-10-%06d" % (i + 1)
        d.update(o)
        rows.append(d)
    return pd.DataFrame(rows)


def sep_frame(pairs):
    """pairs: [(permaticker, ticker, cik_int)]"""
    return pd.DataFrame([{
        "table": "SEP", "permaticker": str(p), "ticker": t,
        "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action="
                      "getcompany&CIK=%010d" % k} for p, t, k in pairs])


def bday(i, start="2010-01-04"):
    return str(pd.bdate_range(start, periods=i + 1)[i].date())


# ---------------------------------------------------------------------------
# 0. the reused T-R5 scorer must share this look's window
# ---------------------------------------------------------------------------
def test_the_reused_r5_scorer_shares_every_window_constant():
    assert c.assert_shared_constants() is True
    for name, want in c.SHARED_WITH_R5.items():
        assert getattr(r5, name) == want


def test_a_divergent_r5_constant_refuses_the_import(monkeypatch):
    monkeypatch.setattr(r5, "VOTE_TO", 42)
    with pytest.raises(AssertionError, match="does not share this look"):
        c.assert_shared_constants()


def test_the_scorer_and_null_are_the_r5_objects_not_copies():
    assert c.score_pairs is r5.score_pairs
    assert c.permutation_null is r5.permutation_null
    assert c.mde_from_null is r5.mde_from_null
    assert c.plant_drift is r5.plant_drift


# ---------------------------------------------------------------------------
# 1. the purchase clauses, each binding alone
# ---------------------------------------------------------------------------
def test_lag_clause_binds_alone_at_both_ends():
    pur = pur_rows({"filing_lag_days": -1}, {"filing_lag_days": 0},
                   {"filing_lag_days": 90}, {"filing_lag_days": 91})
    kept, counts = c.resolve_purchases(pur)
    assert counts["rows_in_vintage"] == 4
    assert counts["A_lag_0_to_90"] == 2
    assert counts["A_removed_lag_negative"] == 1
    assert counts["A_removed_lag_over_90"] == 1
    assert sorted(kept["filing_lag_days"]) == [0, 90]


def test_M6_the_lag_inequality_is_not_reversed():
    """Mutant 6: a clause keeping lag > 90 instead of lag <= 90."""
    pur = pur_rows({"filing_lag_days": 3}, {"filing_lag_days": 400})
    kept, _ = c.resolve_purchases(pur)
    mutant = pur.loc[pur["filing_lag_days"] > c.LAG_MAX]
    assert list(kept["filing_lag_days"]) == [3]
    assert list(mutant["filing_lag_days"]) == [400]
    assert set(kept["filing_lag_days"]) != set(mutant["filing_lag_days"])


def test_a_missing_lag_is_dropped_and_counted():
    pur = pur_rows({"filing_lag_days": np.nan}, {})
    kept, counts = c.resolve_purchases(pur)
    assert counts["A_removed_lag_missing"] == 1
    assert len(kept) == 1


def test_routine_clause_binds_alone():
    pur = pur_rows({"is_routine": True}, {"is_routine": False})
    kept, counts = c.resolve_purchases(pur)
    assert counts["A_lag_0_to_90"] == 2
    assert counts["B_not_routine"] == 1
    assert counts["B_removed_routine"] == 1
    assert list(kept["is_routine"]) == [False]


def test_the_two_purchase_clauses_are_independent():
    pur = pur_rows({"filing_lag_days": 200, "is_routine": False},
                   {"filing_lag_days": 3, "is_routine": True},
                   {"filing_lag_days": 3, "is_routine": False})
    kept, counts = c.resolve_purchases(pur)
    assert counts["A_lag_0_to_90"] == 2 and counts["B_not_routine"] == 1
    assert len(kept) == 1


# ---------------------------------------------------------------------------
# 2. clustering
# ---------------------------------------------------------------------------
def test_an_event_opens_on_the_first_filing_day():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10)}, {"filing_date": bday(12)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 1
    assert int(ev["D_idx"].iloc[0]) == 10


def test_a_weekend_filing_date_rolls_forward_to_the_next_trading_day():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    sat = str((pd.Timestamp(bday(10)) + pd.Timedelta(days=1)).date())
    pur = pur_rows({"filing_date": sat})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert int(ev["D_idx"].iloc[0]) == 11


def test_the_cluster_absorbs_the_last_day_inside_the_window():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10)},
                   {"filing_date": bday(10 + c.CLUSTER_WINDOW - 1)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 1 and int(ev["n_purchases"].iloc[0]) == 2


def test_the_next_event_opens_only_after_the_window():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10)},
                   {"filing_date": bday(10 + c.CLUSTER_WINDOW)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 2
    assert sorted(int(x) for x in ev["D_idx"]) == [10, 10 + c.CLUSTER_WINDOW]


def test_the_second_event_is_measured_from_its_own_opening_not_the_first():
    """Three filings at 10, 69 and 128: two events, not three and not one."""
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10)}, {"filing_date": bday(69)},
                   {"filing_date": bday(128)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert sorted(int(x) for x in ev["D_idx"]) == [10, 128]
    assert sorted(int(x) for x in ev["n_purchases"]) == [1, 2]


def test_M5_dropping_the_clustering_window_merges_distinct_episodes():
    """Mutant 5: one event per issuer, ignoring the 60-day window."""
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10)},
                   {"filing_date": bday(10 + c.CLUSTER_WINDOW)},
                   {"filing_date": bday(10 + 2 * c.CLUSTER_WINDOW)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    mutant_n_events = kept["issuer_cik"].nunique()
    assert len(ev) == 3 and mutant_n_events == 1


def test_different_issuers_never_share_an_event():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows({"filing_date": bday(10), "issuer_cik": "0000000001"},
                   {"filing_date": bday(11), "issuer_cik": "0000000002"})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 2
    assert set(ev["issuer_cik"]) == {"0000000001", "0000000002"}


def test_the_insider_count_and_total_value_are_the_clusters():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows(
        {"filing_date": bday(10), "owner_cik": "A", "value": 1000.0},
        {"filing_date": bday(11), "owner_cik": "A", "value": 2000.0},
        {"filing_date": bday(12), "owner_cik": "B", "value": 4000.0})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 1
    assert int(ev["n_insiders"].iloc[0]) == 2
    assert int(ev["n_purchases"].iloc[0]) == 3
    assert float(ev["total_value"].iloc[0]) == pytest.approx(7000.0)


def test_T_is_the_transaction_date_of_the_opening_filing():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    pur = pur_rows(
        {"filing_date": bday(10), "trans_date": bday(8)},
        {"filing_date": bday(10), "trans_date": bday(6)},
        {"filing_date": bday(20), "trans_date": bday(1)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert len(ev) == 1
    assert str(pd.Timestamp(ev["T"].iloc[0]).date()) == bday(6)
    assert str(pd.Timestamp(ev["T_last"].iloc[0]).date()) == bday(8)


def test_a_filing_past_the_calendars_end_is_dropped_and_counted():
    book = make_book({"AAA": frame("2010-01-04", 50)})
    pur = pur_rows({"filing_date": "2099-01-04"}, {"filing_date": bday(5)})
    kept, counts = c.resolve_purchases(pur)
    ev = c.cluster_events(kept, book, counts)
    assert counts["C_dropped_off_calendar"] == 1
    assert counts["C_filing_day_on_the_calendar"] == 1
    assert len(ev) == 1


# ---------------------------------------------------------------------------
# 3. issuer mapping
# ---------------------------------------------------------------------------
def test_mapping_tie_break_prefers_the_ticker_that_trades_then_the_history():
    long_ = frame("2010-01-04", 200)
    short = frame("2010-07-01", 60)
    dead, _ = holed(frame("2010-01-04", 200), range(100, 200))
    book = make_book({"LONG": long_, "SHORT": short, "DEAD": dead})
    d_idx = 150
    ev = pd.DataFrame([{"issuer_cik": "0000000007", "D_idx": d_idx,
                        "n_insiders": 1, "n_purchases": 1, "total_value": 1.0,
                        "T": pd.Timestamp("2010-01-05"),
                        "T_last": pd.Timestamp("2010-01-05"),
                        "issuer_name": "X", "first_accession": "a"}])
    sep = sep_frame([(1, "DEAD", 7), (2, "LONG", 7), (3, "SHORT", 7)])
    mp, counts = c.map_issuers(ev, sep, book, {})
    assert len(mp) == 1
    assert mp["ticker"].iloc[0] == "LONG"
    assert counts["E_mapping"]["multi_permaticker"] == 1
    assert counts["E_mapping"]["multi_resolved_by_row_count"] == 1


def test_mapping_drops_a_cik_with_no_sep_row():
    book = make_book({"AAA": frame("2010-01-04", 200)})
    ev = pd.DataFrame([{"issuer_cik": "0000000099", "D_idx": 50,
                        "n_insiders": 1, "n_purchases": 1, "total_value": 1.0,
                        "T": pd.Timestamp("2010-01-05"),
                        "T_last": pd.Timestamp("2010-01-05"),
                        "issuer_name": "X", "first_accession": "a"}])
    mp, counts = c.map_issuers(ev, sep_frame([(1, "AAA", 7)]), book, {})
    assert len(mp) == 0
    assert counts["E_mapping"]["dropped_no_usable_ticker"] == 1


def test_mapping_ignores_rows_that_are_not_the_sep_table():
    book = make_book({"AAA": frame("2010-01-04", 200)})
    sep = sep_frame([(1, "AAA", 7)])
    sep.loc[0, "table"] = "SF1"
    ev = pd.DataFrame([{"issuer_cik": "0000000007", "D_idx": 50,
                        "n_insiders": 1, "n_purchases": 1, "total_value": 1.0,
                        "T": pd.Timestamp("2010-01-05"),
                        "T_last": pd.Timestamp("2010-01-05"),
                        "issuer_name": "X", "first_accession": "a"}])
    mp, counts = c.map_issuers(ev, sep, book, {})
    assert len(mp) == 0 and counts["E_mapping"]["dropped_no_usable_ticker"] == 1


def test_cik_is_extracted_from_the_browse_edgar_url():
    assert c.extract_cik("https://x?action=getcompany&CIK=0000320193") == 320193
    assert c.extract_cik("CIK=12345&type=4") == 12345
    assert c.extract_cik("no cik here") is None
    assert c.extract_cik(None) is None


def test_duplicate_permaticker_date_pairs_are_counted_not_hidden():
    book = make_book({"AAA": frame("2010-01-04", 200)})
    rows = [{"issuer_cik": "0000000007", "D_idx": 50, "n_insiders": 1,
             "n_purchases": 1, "total_value": 1.0,
             "T": pd.Timestamp("2010-01-05"), "T_last": pd.Timestamp("2010-01-05"),
             "issuer_name": "X", "first_accession": "a"},
            {"issuer_cik": "0000000008", "D_idx": 50, "n_insiders": 1,
             "n_purchases": 1, "total_value": 1.0,
             "T": pd.Timestamp("2010-01-05"), "T_last": pd.Timestamp("2010-01-05"),
             "issuer_name": "X", "first_accession": "b"}]
    sep = sep_frame([(1, "AAA", 7), (1, "AAA", 8)])
    mp, counts = c.map_issuers(pd.DataFrame(rows), sep, book, {})
    assert len(mp) == 2
    assert counts["E_duplicate_permaticker_D_pairs"] == 1


# ---------------------------------------------------------------------------
# 4. history and tradability, each binding alone
# ---------------------------------------------------------------------------
def _mp_row(ticker, d_idx, **kw):
    row = {"issuer_cik": "0000000007", "cik": 7, "D_idx": int(d_idx),
           "n_insiders": 1, "n_purchases": 1, "total_value": 1.0,
           "T": pd.Timestamp("2010-01-05"), "T_last": pd.Timestamp("2010-01-05"),
           "issuer_name": "X", "first_accession": "a",
           "permaticker": "1", "ticker": ticker}
    row.update(kw)
    return row


def test_history_clause_binds_alone_at_250_rows_before_D():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    mp = pd.DataFrame([_mp_row("AAA", 250), _mp_row("AAA", 249)])
    res, counts = c.apply_price_filters(mp, book, {})
    assert counts["F_history_250_rows_before_D"] == 1
    assert set(res["D_idx"]) == {250}


def test_price_floor_clause_binds_alone_on_D_plus_1():
    good = frame("2010-01-04", 400, unadj=2.00)
    bad = frame("2010-01-04", 400, unadj=1.99)
    book = make_book({"GOOD": good, "BAD": bad})
    mp = pd.DataFrame([_mp_row("GOOD", 300), _mp_row("BAD", 300)])
    res, counts = c.apply_price_filters(mp, book, {})
    assert counts["F_history_250_rows_before_D"] == 2
    assert counts["G_price_floor_on_D_plus_1"] == 1
    assert set(res["ticker"]) == {"GOOD"}


def test_dollar_volume_clause_binds_alone_at_the_registered_floor():
    at = frame("2010-01-04", 400, unadj=10.0, vol=c.DOLLAR_VOL_FLOOR / 10.0)
    below = frame("2010-01-04", 400, unadj=10.0,
                  vol=(c.DOLLAR_VOL_FLOOR / 10.0) - 1.0)
    book = make_book({"ATFLOOR": at, "BELOW": below})
    mp = pd.DataFrame([_mp_row("ATFLOOR", 300), _mp_row("BELOW", 300)])
    res, counts = c.apply_price_filters(mp, book, {})
    assert counts["G_price_floor_on_D_plus_1"] == 2
    assert counts["G_dollar_volume_floor"] == 1
    assert set(res["ticker"]) == {"ATFLOOR"}
    assert float(res["dv_median_at_D"].iloc[0]) == pytest.approx(
        c.DOLLAR_VOL_FLOOR)


def test_the_dollar_volume_median_reads_the_twenty_trading_days_ending_D():
    """Exactly positions D-19..D: one day further on and the window has moved."""
    f = frame("2010-01-04", 400, unadj=1.0, vol=1.0)
    f.loc[f.index[281:301], "volume"] = 1e9
    book = make_book({"AAA": f})
    assert book.dollar_volume_median("AAA", 300, 20) == pytest.approx(1e9)
    assert book.dollar_volume_median("AAA", 280, 20) == pytest.approx(1.0)
    # at D=310 the window holds ten loud days and ten quiet ones
    assert book.dollar_volume_median("AAA", 310, 20) == pytest.approx(
        (1e9 + 1.0) / 2.0)


# ---------------------------------------------------------------------------
# 5. completeness, four ways, with the truncation clause
# ---------------------------------------------------------------------------
def test_a_window_that_fits_entirely_is_complete_and_exits_at_D61():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    assert c.classify_window(book, "AAA", 100) == ("complete", 161)


def test_a_delisted_event_is_scored_to_its_last_close():
    f = frame("2010-01-04", 400)
    short = f.iloc[:140]
    book = make_book({"AAA": short, "BBB": frame("2010-01-04", 400)})
    status, exit_ = c.classify_window(book, "AAA", 100)
    assert status == "delisted" and exit_ == 139


def test_M3_a_last_row_at_the_snapshot_edge_is_truncated_not_delisted():
    """Mutant 3: scoring a truncated window as though the position had closed."""
    f = frame("2010-01-04", 400)
    stopper = f.iloc[:140]
    keeper = f.iloc[:143]            # the calendar ends at position 142
    book = make_book({"AAA": stopper, "BBB": keeper})
    assert book.L == 143
    status, exit_ = c.classify_window(book, "AAA", 100)
    assert status == "truncated" and exit_ == -1
    mutant = ("delisted", 139)
    assert (status, exit_) != mutant


def test_the_same_last_row_far_from_the_edge_is_a_delisting():
    """The truncation clause reads the SNAPSHOT's end, nothing about the stock."""
    f = frame("2010-01-04", 400)
    book = make_book({"AAA": f.iloc[:140], "BBB": f.iloc[:160]})
    assert c.classify_window(book, "AAA", 100) == ("delisted", 139)


def test_the_truncation_edge_is_exactly_five_trading_days():
    f = frame("2010-01-04", 400)
    at_edge = make_book({"AAA": f.iloc[:140],
                         "BBB": f.iloc[:140 + c.TRUNCATION_EDGE]})
    assert c.classify_window(at_edge, "AAA", 100)[0] == "delisted"
    inside = make_book({"AAA": f.iloc[:140],
                        "BBB": f.iloc[:140 + c.TRUNCATION_EDGE - 1]})
    assert c.classify_window(inside, "AAA", 100)[0] == "truncated"


def test_the_five_return_floor_drops_a_too_short_delisting():
    f = frame("2010-01-04", 400)
    book = make_book({"AAA": f.iloc[:105], "BBB": f.iloc[:200]})
    assert c.classify_window(book, "AAA", 100)[0] == "too_short"
    book2 = make_book({"AAA": f.iloc[:107], "BBB": f.iloc[:200]})
    assert c.classify_window(book2, "AAA", 100)[0] == "delisted"


def test_a_hole_while_the_ticker_continues_drops_the_event():
    f = frame("2010-01-04", 400)
    holedf, _ = holed(f, [120])
    book = make_book({"AAA": holedf, "BBB": f})
    assert c.classify_window(book, "AAA", 100) == ("hole", -1)


def test_a_hole_before_an_in_window_delisting_is_still_a_hole():
    f = frame("2010-01-04", 400)
    stop = f.iloc[:140]
    stop2, _ = holed(stop, [120])
    book = make_book({"AAA": stop2, "BBB": f.iloc[:200]})
    assert c.classify_window(book, "AAA", 100) == ("hole", -1)


def test_no_row_on_the_entry_day_cannot_open_the_position():
    f = frame("2010-01-04", 400)
    holedf, _ = holed(f, [101])
    book = make_book({"AAA": holedf, "BBB": f})
    assert c.classify_window(book, "AAA", 100) == ("no_entry", -1)


def test_an_absent_ticker_is_its_own_outcome():
    book = make_book({"AAA": frame("2010-01-04", 400)})
    assert c.classify_window(book, "ZZZ", 100) == ("absent", -1)


def test_every_completeness_outcome_is_counted_separately():
    f = frame("2010-01-04", 400)
    holedf, _ = holed(f, [320])
    book = make_book({"FULL": f, "STOP": f.iloc[:340], "HOLE": holedf,
                      "EDGE": f.iloc[:398]})
    mp = pd.DataFrame([_mp_row("FULL", 300), _mp_row("STOP", 280),
                       _mp_row("HOLE", 300), _mp_row("EDGE", 340)])
    res, counts = c.apply_price_filters(mp, book, {})
    ws = counts["H_window_status"]
    assert ws["complete"] == 1 and ws["delisted"] == 1 and ws["hole"] == 1
    assert ws["truncated"] == 1
    assert counts["H_complete_or_delisted"] == 2 == len(res)
    assert counts["H_truncated_at_snapshot_edge"] == 1


def test_classify_window_and_score_pairs_agree_on_the_exit():
    f = frame("2010-01-04", 400, rate=0.001)
    book = make_book({"AAA": f.iloc[:140], "BBB": f})
    status, exit_ = c.classify_window(book, "AAA", 100)
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    net, n, ok, ex = c.score_pairs(wb, np.array([0]), np.array([100]))
    assert status == "delisted"
    assert int(ex[0]) == exit_ and bool(ok[0])


# ---------------------------------------------------------------------------
# 6. the investable benchmark
# ---------------------------------------------------------------------------
def test_the_rolling_dollar_volume_median_is_the_twenty_day_median():
    f = frame("2010-01-04", 60, unadj=1.0, vol=1.0)
    f.loc[f.index[30:50], "volume"] = 7.0
    book = make_book({"AAA": f})
    dv = c.rolling_dollar_volume_median(book)
    assert dv[49] == pytest.approx(7.0)
    assert dv[0] == pytest.approx(1.0)
    assert dv[39] == pytest.approx(4.0)


def test_the_rolling_median_never_reads_across_a_ticker_boundary():
    a = frame("2010-01-04", 30, unadj=1.0, vol=1.0)
    b = frame("2010-01-04", 30, unadj=1.0, vol=1000.0)
    book = make_book({"AAA": a, "BBB": b})
    dv = c.rolling_dollar_volume_median(book)
    assert dv[30] == pytest.approx(1000.0)
    assert dv[29] == pytest.approx(1.0)


def test_the_benchmark_admits_only_tickers_clearing_all_three_prior_day_floors():
    n = 400
    ok = frame("2010-01-04", n, rate=0.01, unadj=10.0, vol=1e6)
    cheap = frame("2010-01-04", n, rate=0.50, unadj=1.00, vol=1e6)
    thin = frame("2010-01-04", n, rate=0.50, unadj=10.0, vol=1.0)
    young = frame(bday(300), 90, rate=0.50, unadj=10.0, vol=1e6)
    book = make_book({"OK": ok, "CHEAP": cheap, "THIN": thin, "YOUNG": young})
    dv = c.rolling_dollar_volume_median(book)
    idx = c.build_investable_index(book, dv)
    j = 350
    assert int(idx["n_tickers"][j]) == 1
    assert idx["mean_log_return"][j] == pytest.approx(0.01)


def test_M4_dropping_the_floors_lets_the_excluded_names_into_the_benchmark():
    """Mutant 4: a benchmark with no prior-day floors at all."""
    n = 400
    ok = frame("2010-01-04", n, rate=0.01, unadj=10.0, vol=1e6)
    thin = frame("2010-01-04", n, rate=0.50, unadj=10.0, vol=1.0)
    book = make_book({"OK": ok, "THIN": thin})
    dv = c.rolling_dollar_volume_median(book)
    real = c.build_investable_index(book, dv)
    nofloor = c.build_investable_index(
        book, np.full(len(dv), c.BENCH_DV_FLOOR + 1.0))
    j = 350
    assert int(real["n_tickers"][j]) == 1
    assert int(nofloor["n_tickers"][j]) == 2
    assert real["mean_log_return"][j] != pytest.approx(
        nofloor["mean_log_return"][j])


def test_the_benchmark_starts_contributing_only_after_250_prior_rows():
    n = 400
    a = frame("2010-01-04", n, rate=0.02, unadj=10.0, vol=1e6)
    book = make_book({"AAA": a})
    dv = c.rolling_dollar_volume_median(book)
    idx = c.build_investable_index(book, dv)
    assert int(idx["n_tickers"][c.BENCH_MIN_PRIOR_ROWS]) == 0
    assert int(idx["n_tickers"][c.BENCH_MIN_PRIOR_ROWS + 1]) == 1


def test_the_benchmark_skips_a_return_formed_across_a_hole():
    n = 400
    a = frame("2010-01-04", n, rate=0.02, unadj=10.0, vol=1e6)
    a2, _ = holed(a, [300])
    b = frame("2010-01-04", n, rate=0.02, unadj=10.0, vol=1e6)
    book = make_book({"AAA": a2, "BBB": b})
    dv = c.rolling_dollar_volume_median(book)
    idx = c.build_investable_index(book, dv)
    assert int(idx["n_tickers"][301]) == 1
    assert int(idx["n_tickers"][305]) == 2


def test_index_members_on_names_the_tickers_the_benchmark_averages():
    n = 400
    ok = frame("2010-01-04", n, rate=0.01, unadj=10.0, vol=1e6)
    cheap = frame("2010-01-04", n, rate=0.01, unadj=1.00, vol=1e6)
    book = make_book({"CHEAP": cheap, "OK": ok})
    dv = c.rolling_dollar_volume_median(book)
    assert c.index_members_on(book, dv, 350) == ["OK"]


# ---------------------------------------------------------------------------
# 7. abnormal returns, window sums and the cost
# ---------------------------------------------------------------------------
def test_abnormal_return_is_the_stock_minus_the_benchmark():
    book = make_book({"AAA": frame("2010-01-04", 50, rate=0.01)})
    ab, ok = c.abnormal_series(book, "AAA", flat_index(book, 0.004))
    assert ok[1:].all() and not ok[0]
    assert ab[5] == pytest.approx(0.006)


def test_abnormal_return_is_absent_across_a_hole_in_the_history():
    f = frame("2010-01-04", 50, rate=0.01)
    holedf, dates = holed(f, [20])
    book = make_book({"AAA": holedf}, extra_dates=dates)
    ab, ok = c.abnormal_series(book, "AAA", flat_index(book))
    assert not ok[20] and not ok[21]
    assert ok[22]


def test_the_voted_window_is_sixty_returns_from_D2_to_D61_hand_computed():
    rate = 0.001
    book = make_book({"AAA": frame("2010-01-04", 400, rate=rate)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    net, n, ok, ex = c.score_pairs(wb, np.array([0]), np.array([100]))
    assert int(n[0]) == c.VOTE_N == 60
    assert float(net[0]) == pytest.approx(60 * rate - c.COST_FRACTION, abs=1e-12)


def test_M7_the_cost_is_twenty_bps_charged_once():
    """Mutant 7: a vote that never charges the round trip."""
    rate = 0.001
    book = make_book({"AAA": frame("2010-01-04", 400, rate=rate)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    net = float(c.score_pairs(wb, np.array([0]), np.array([100]))[0][0])
    gross = net + c.COST_FRACTION
    assert c.COST_FRACTION == pytest.approx(0.0020)
    assert gross - net == pytest.approx(0.0020)
    assert gross == pytest.approx(60 * rate)


def test_the_twenty_day_window_is_twenty_returns():
    rate = 0.001
    book = make_book({"AAA": frame("2010-01-04", 400, rate=rate)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    g, d = np.array([0]), np.array([100])
    s, n = wb.window(g, d + c.VOTE_FROM, d + c.SHORT_TO)
    assert int(n[0]) == c.SHORT_N == 20
    assert float(s[0]) == pytest.approx(20 * rate)


def test_the_pre_purchase_window_is_twenty_one_returns_ending_at_T():
    rate = 0.001
    book = make_book({"AAA": frame("2010-01-04", 400, rate=rate)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    t = np.array([100])
    s, n = wb.window(np.array([0]), t - (c.PREPURCHASE_LAG - 1), t)
    assert int(n[0]) == c.PREPURCHASE_LAG == 21
    assert float(s[0]) == pytest.approx(21 * rate)


def test_the_path_points_are_k_returns_after_D_plus_1():
    rate = 0.001
    book = make_book({"AAA": frame("2010-01-04", 400, rate=rate)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    g, d = np.array([0]), np.array([100])
    for k in c.PATH_POINTS:
        s, n = wb.window(g, d + c.VOTE_FROM, d + 1 + k)
        assert int(n[0]) == k
        assert float(s[0]) == pytest.approx(k * rate)


def test_an_incomplete_voted_window_is_not_scored():
    f = frame("2010-01-04", 400)
    holedf, _ = holed(f, [120])
    book = make_book({"AAA": holedf, "BBB": f})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    net, n, ok, ex = c.score_pairs(wb, np.array([0]), np.array([100]))
    assert not bool(ok[0]) and not np.isfinite(net[0])


def test_the_window_book_counters_are_narrow_but_exact():
    book = make_book({"AAA": frame("2010-01-04", 400, rate=0.001)})
    wb = c.WindowBook(book, ["AAA"], flat_index(book))
    assert wb.cum_ok.dtype == np.int16 and wb.cum_row.dtype == np.int16
    assert wb.cum_ab.dtype == np.float64
    assert int(wb.cum_row[0, book.L]) == 400
    assert int(wb.cum_ok[0, book.L]) == 399


# ---------------------------------------------------------------------------
# 8. the null, the distance and contamination
# ---------------------------------------------------------------------------
def _two_issuer_book(rate_a=0.002, rate_b=-0.001, n=400):
    book = make_book({"AAA": frame("2010-01-04", n, rate=rate_a),
                      "BBB": frame("2010-01-04", n, rate=rate_b)})
    wb = c.WindowBook(book, ["AAA", "BBB"], flat_index(book))
    return book, wb


def test_M1_the_null_permutes_issuers_and_leaves_every_date_unchanged(monkeypatch):
    """Mutant 1: a null that shifts dates instead of permuting issuers."""
    book, wb = _two_issuer_book()
    groups = np.array([0, 1])
    d_idx = np.array([100, 200])
    seen = {}

    real = c.score_pairs

    def spy(w, g, d):
        seen["g"] = np.asarray(g).copy()
        seen["d"] = np.asarray(d).copy()
        return real(w, g, d)

    monkeypatch.setattr(r5, "score_pairs", spy)
    c.permuted_mcar(wb, groups, d_idx, np.array([1, 0]))
    assert list(seen["d"]) == [100, 200], "a date moved: this is a shift null"
    assert list(seen["g"]) == [1, 0], "the issuers were not permuted"


def test_the_permutation_is_a_bijection():
    rng = np.random.default_rng(1)
    s = c.draw_permutation(rng, 50)
    assert sorted(s.tolist()) == list(range(50))


def test_one_permutation_per_replicate_not_one_per_event(monkeypatch):
    book, wb = _two_issuer_book()
    calls = {"n": 0}
    real = c.draw_permutation

    def counted(rng, n):
        calls["n"] += 1
        return real(rng, n)

    monkeypatch.setattr(r5, "draw_permutation", counted)
    c.permutation_null(wb, np.array([0, 1]), np.array([100, 200]), 7,
                       np.random.default_rng(0))
    assert calls["n"] == 7


def test_the_distance_is_mCAR_minus_the_null_median():
    null = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
    assert c.null_median(null) == pytest.approx(0.003)
    assert c.distance_of(0.010, null) == pytest.approx(0.007)


def test_M2_the_bar_is_measured_from_the_null_median_not_from_zero():
    """Mutant 2: adjudicating mCAR itself against the materiality bar."""
    null = np.array([0.008, 0.009, 0.010, 0.011, 0.012])
    mcar = 0.012
    dist = c.distance_of(mcar, null)
    assert dist == pytest.approx(0.002)
    assert c.verdict_of(0.01, dist, 0.05) == "SIGNIFICANT_IMMATERIAL"
    mutant = c.verdict_of(0.01, mcar, 0.05)
    assert mutant == "SOMETHING"
    assert c.verdict_of(0.01, dist, 0.05) != mutant


def test_the_distance_is_invariant_to_a_constant_drift_in_both():
    null = np.array([0.001, 0.002, 0.003])
    a = c.distance_of(0.010, null)
    b = c.distance_of(0.110, null + 0.100)
    assert a == pytest.approx(b)


def test_contamination_is_counted_and_never_dropped():
    book, wb = _two_issuer_book()
    groups = np.array([0, 1])
    d_idx = np.array([100, 200])
    wb.set_real_voted(groups, d_idx, np.array([161, 261]))
    m, ret, used, con = c.permuted_mcar(wb, groups, d_idx, np.array([0, 1]))
    assert used == 2 and ret == pytest.approx(1.0)
    assert con == pytest.approx(1.0)
    m2, ret2, used2, con2 = c.permuted_mcar(wb, groups, d_idx, np.array([1, 0]))
    assert used2 == 2 and con2 == pytest.approx(0.0)


def test_the_corrected_distance_divides_by_one_minus_the_contaminated_share():
    assert c.corrected_distance(0.004, 0.20) == pytest.approx(0.005)
    assert c.corrected_distance(0.004, 0.0) == pytest.approx(0.004)
    assert c.corrected_distance(0.004, 1.0) is None


def test_a_pair_the_issuer_cannot_cover_is_dropped_and_counted():
    f = frame("2010-01-04", 400, rate=0.001)
    book = make_book({"AAA": f, "SHORT": f.iloc[:103]})
    wb = c.WindowBook(book, ["AAA", "SHORT"], flat_index(book))
    m, ret, used, con = c.permuted_mcar(wb, np.array([0, 1]),
                                        np.array([100, 300]), np.array([1, 0]))
    assert used == 1 and ret == pytest.approx(0.5)


def test_mde_formula_is_the_registered_expression():
    d = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    want = (np.percentile(d, 95) - np.median(d)) + c.Z80 * d.std(ddof=1)
    assert c.mde_from_null(d) == pytest.approx(want)


def test_band_ok_is_a_closed_band():
    assert c.band_ok(0.01, (0.01, 0.10)) and c.band_ok(0.10, (0.01, 0.10))
    assert not c.band_ok(0.009, (0.01, 0.10))
    assert not c.band_ok(0.101, (0.01, 0.10))


def test_detection_ok_is_at_or_above_the_floor():
    assert c.detection_ok(0.80, 0.80) and c.detection_ok(0.81, 0.80)
    assert not c.detection_ok(0.79, 0.80)


def test_plant_drift_is_uniform_zero_to_twice_X():
    d = c.plant_drift(200000, 0.01, np.random.default_rng(3))
    assert d.min() >= 0.0 and d.max() <= 0.02
    assert d.mean() == pytest.approx(0.01, abs=5e-5)


def test_X_is_calibrated_on_one_draw_to_the_registered_planted_mean():
    cal = c.calibrate_X(5000, 0.01, 4242)
    assert cal["one_draw_planted_mean"] == pytest.approx(0.01, abs=1e-12)


def test_the_plant_rides_the_real_pairs_and_never_the_permuted_ones():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage_controls")
    body = ast.dump(fn)
    assert "plant_drift" in body
    assert "plant" not in ast.dump(next(
        n for n in ast.walk(ast.parse((ROOT / "tools" /
                                       "form4_event_study_r6.py").read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == "distance_of"))


# ---------------------------------------------------------------------------
# 9. the verdict ladder, on the distance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p,dist,want", [
    (0.01, 0.020, "SOMETHING"),
    (0.05, 0.010, "SOMETHING"),
    (0.05, 0.0099, "SIGNIFICANT_IMMATERIAL"),
    (0.01, 0.0, "NOTHING"),
    (0.01, -0.01, "NOTHING"),
    (0.06, 0.0001, "SUGGESTIVE"),
    (0.10, 0.030, "SUGGESTIVE"),
    (0.101, 0.030, "NOTHING"),
    (0.06, -0.001, "NOTHING"),
])
def test_the_verdict_ladder_is_the_registered_one(p, dist, want):
    assert c.verdict_of(p, dist, 0.05) == want


# ---------------------------------------------------------------------------
# 10. descriptives
# ---------------------------------------------------------------------------
def test_tercile_labels_are_equal_sized_and_ordered():
    lab = c.tercile_labels([9.0, 1.0, 5.0, 2.0, 8.0, 4.0])
    assert sorted(np.bincount(lab).tolist()) == [2, 2, 2]
    assert lab[1] == 0 and lab[0] == 2


def test_subgroup_distances_are_the_registered_subgroups():
    half = pd.DataFrame({
        "n_insiders": [1, 2, 3, 1],
        "total_value": [50.0, 200000.0, 100000.0, 1e6],
        "dv_median_at_D": [1.0, 2.0, 3.0, 4.0]})
    net = np.array([0.01, 0.02, 0.03, 0.04])
    out = c.subgroup_distances(half, net, 0.005)
    assert out["insiders_ge_2"]["n"] == 2
    assert out["insiders_ge_2"]["mCAR_net"] == pytest.approx(0.025)
    assert out["insiders_ge_2"]["distance"] == pytest.approx(0.020)
    assert out["value_gt_100000"]["n"] == 2
    assert set(out) >= {"liquidity_tercile_low", "liquidity_tercile_mid",
                        "liquidity_tercile_high"}


def test_events_by_year_carries_the_pooled_distance_by_year():
    out = c.events_by_year(["2006-03-01", "2006-09-01", "2007-02-01"],
                           [0.01, 0.03, 0.05], 0.01)
    assert out["2006"]["n"] == 2
    assert out["2006"]["mCAR_net"] == pytest.approx(0.02)
    assert out["2006"]["distance"] == pytest.approx(0.01)
    assert out["2007"]["distance"] == pytest.approx(0.04)


def test_the_descriptives_are_formed_only_inside_the_observed_stage():
    """The subgroup and tercile reads must not exist before the pass receipt."""
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    tree = ast.parse(src)
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("subgroup_distances", "events_by_year"):
        assert name in ast.dump(fns["stage_observed"]), name
        for other in ("prepare", "stage_controls", "stage_resolve",
                      "build_window_book", "halt_out"):
            assert name not in ast.dump(fns[other]), (name, other)


def test_the_resolve_stage_forms_no_window_sum_and_no_null():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage_resolve")
    body = ast.dump(fn)
    for forbidden in ("score_pairs", "permutation_null", "voted_returns",
                      "WindowBook", "build_window_book"):
        assert forbidden not in body, forbidden


# ---------------------------------------------------------------------------
# 11. the seal: loading, halves, data pins, prose
# ---------------------------------------------------------------------------
def test_load_registration_refuses_a_sha_mismatch(tmp_path):
    f = tmp_path / "reg.json"
    f.write_text(json.dumps({"a": 1}))
    with pytest.raises(AssertionError, match="sha256 mismatch"):
        c.load_registration(f, "0" * 64)
    reg = c.load_registration(f, c.sha256_file(f))
    assert reg["_sha256"] == c.sha256_file(f)


def test_half_bounds_are_read_out_of_the_seal_not_hardcoded():
    reg = {"event_definition": {"halves":
           "half A: 2001-01-01 <= D <= 2002-12-31; half B: 2003-01-01 <= D <= "
           "2004-12-31; holdout: D >= 2005-01-01"}}
    hb = c.half_bounds(reg)
    assert str(hb["half_A"][0].date()) == "2001-01-01"
    assert str(hb["half_B"][1].date()) == "2004-12-31"
    assert str(hb["holdout"][0].date()) == "2005-01-01"


def test_split_halves_cuts_on_D_not_on_the_transaction_date():
    res = pd.DataFrame({"D_date": ["2015-12-31", "2016-01-01", "2023-01-02"],
                        "D_idx": [1, 2, 3]})
    reg = {"event_definition": {"halves":
           "half A: 2006-01-01 <= D <= 2015-12-31; half B: 2016-01-01 <= D <= "
           "2022-12-31; holdout: D >= 2023-01-01"}}
    h = c.split_halves(res, reg)
    assert len(h["half_A"]) == 1 and len(h["half_B"]) == 1
    assert len(h["holdout"]) == 1


def test_manifest_recorded_sha_reads_the_manifests_own_record(tmp_path):
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sha256": {"purchases.parquet": "a" * 64}}))
    assert c.manifest_recorded_sha(m, "purchases.parquet") == "a" * 64
    with pytest.raises(AssertionError, match="records no sha256"):
        c.manifest_recorded_sha(m, "other.parquet")


def test_data_paths_and_pins_take_the_parquet_sha_from_the_manifest():
    reg = {"data": {
        "events": "data/form4/20261105/purchases.parquet, sha256 as recorded in "
                  "manifests/form4_20261105.json (sha256 %s)"
                  % c.sha256_file(ROOT / "manifests" / "form4_20261105.json"),
        "prices": "Sharadar snapshot_20261004p bulk_stocks.parquet sha256 %s"
                  % ("c" * 64),
        "tickers": "bulk_tickers.parquet sha256 %s" % ("d" * 64)}}
    p = r.data_paths_and_pins(reg)
    assert p["snapshot"] == "snapshot_20261004p"
    assert p["events"][0].name == "purchases.parquet"
    assert p["events"][1] == c.manifest_recorded_sha(
        ROOT / "manifests" / "form4_20261105.json", "purchases.parquet")
    assert p["stocks"][1] == "c" * 64 and p["tickers"][1] == "d" * 64


def test_a_data_pin_mismatch_is_refused(tmp_path):
    f = tmp_path / "x.parquet"
    f.write_bytes(b"hello")
    with pytest.raises(AssertionError, match="sha256 mismatch"):
        r.assert_data_pins({"manifest": (f, "0" * 64)})


SEAL = ROOT / "registrations" / "r6" / "registration_v1.json"
SEAL_LOG = ROOT / "logs" / "r6" / "registration_v1.json"


def _seal_text():
    for p in (SEAL, SEAL_LOG):
        if p.exists():
            return p.read_text(), str(p)
    return None, None


def test_registration_prose_pins_the_constants():
    """Every integer the seal states in prose must match the engine's constant."""
    text, src = _seal_text()
    if text is None:
        pytest.skip("the seal is written in phase 2")
    assert "filing_lag_days between %d and %d inclusive" % (
        c.LAG_MIN, c.LAG_MAX) in text
    assert "is_routine false" in text
    assert "within the following %d trading days" % c.CLUSTER_WINDOW in text
    assert "at least %d bulk_stocks rows" % c.HISTORY_MIN_ROWS in text
    assert "closeunadj on D+1 >= %.2f USD" % c.PRICE_FLOOR in text
    assert "over the %d trading days ending D >= %d USD" % (
        c.DOLLAR_VOL_WINDOW, int(c.DOLLAR_VOL_FLOOR)) in text
    assert "on every trading day D+%d..D+%d" % (
        c.COMPLETE_FROM, c.COMPLETE_TO) in text
    assert "at least %d trading days before the snapshot's last date" % (
        c.TRUNCATION_EDGE) in text
    assert "within %d trading days of the snapshot's end is TRUNCATED" % (
        c.TRUNCATION_EDGE) in text
    assert "at least %d returns exist" % c.MIN_RETURNS in text
    assert "at least %d tickers have a bulk_stocks row" % (
        c.CALENDAR_MIN_TICKERS) in text
    assert "fewer than %d half-A events" % c.MIN_EVENTS in text
    assert "closeunadj >= %.2f USD, at least %d prior rows and a %d-day median " \
           "closeunadj times volume >= %d USD" % (
               c.BENCH_PRICE_FLOOR, c.BENCH_MIN_PRIOR_ROWS, c.BENCH_DV_WINDOW,
               int(c.BENCH_DV_FLOOR)) in text
    assert "exit at the close of D+%d" % c.VOTE_TO in text
    assert "the sum of the abnormal daily log returns from D+%d to the exit" % (
        c.VOTE_FROM) in text
    assert "D+%d..D+%d" % (c.VOTE_FROM, c.SHORT_TO) in text
    assert "at %d, %d, %d, %d and %d days after D+1" % c.PATH_POINTS in text
    assert "the pre-purchase window T-%d..T" % c.PREPURCHASE_LAG in text
    assert "%d bps round trip" % int(c.COST_ROUND_TRIP_BPS) in text
    assert "under %d percent retention is redrawn" % int(
        100 * c.RETENTION_MIN) in text
    assert "%s * sd(null mCAR)" % c.Z80 in text
    assert "distance >= %.3f net" % c.MATERIALITY in text
    assert "p in (%.2f, %.2f]" % (0.05, c.SUGGESTIVE_HI) in text
    assert "%s or more insiders" % (
        {1: "one", 2: "two", 3: "three"}[c.INSIDER_SUBGROUP]) in text
    assert "total value over %d USD" % int(c.VALUE_SUBGROUP) in text
    assert "KEPT and counted as contaminated" in text


def test_the_prose_pin_binds_to_the_tracked_seal_once_it_exists():
    text, src = _seal_text()
    if text is None:
        pytest.skip("the seal is written in phase 2")
    if SEAL.exists():
        assert src == str(SEAL)


def test_the_seal_names_the_distance_as_the_adjudicated_quantity():
    text, _ = _seal_text()
    if text is None:
        pytest.skip("the seal is written in phase 2")
    reg = json.loads(text)
    assert reg["statistic"]["distance"].startswith(
        "mCAR minus the median of the null's mCAR")
    assert reg["voting_null"]["name"] == "issuer permutation at fixed dates"
    assert int(reg["voting_null"]["R"]) == 2000
    assert int(reg["voting_null"]["seed"]) == 20261106
    assert c.calendar_min_tickers(reg) == c.CALENDAR_MIN_TICKERS
    assert any(x["id"] == "RULE-BAR-FROM-NULL"
               for x in reg["interpretation_rulings"])


def test_the_seal_carries_the_registered_control_parameters():
    text, _ = _seal_text()
    if text is None:
        pytest.skip("the seal is written in phase 2")
    reg = json.loads(text)
    kn = reg["controls"]["known_null"]
    pc = reg["controls"]["positive_control"]
    assert (kn["N"], kn["R_each"], kn["seed"], kn["band"]) == (
        200, 500, 777, [0.01, 0.10])
    assert (pc["N_draws"], pc["R_each"], pc["seed"]) == (50, 500, 4242)
    assert pc["target_mCAR"] == 0.01 and pc["detection_floor"] == 0.80
    assert reg["ledger"]["batch_id"] == "T-R6"


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


def test_observed_stage_refuses_on_a_registration_sha_mismatch(tmp_path,
                                                               monkeypatch,
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
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    for field in ("registration_sha", "null_family", "n_events", "mde",
                  "known_null_share", "X_calibrated", "detection", "utc"):
        assert '"%s"' % field in src


def test_halt_writes_the_same_figures_and_exits_non_zero():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    assert "def halt_out(" in src
    assert "return 15" in src
    assert "RECEIPT.unlink()" in src
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "halt_out")
    assert "null_family" in ast.dump(fn), "a halt must name its null family"


def test_the_runner_halts_below_the_registered_minimum_event_count():
    """The <5000 halt is reached before any null is formed."""
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage_controls")
    body = ast.dump(fn)
    i_min = body.index("MIN_EVENTS")
    i_null = body.index("permutation_null")
    assert i_min < i_null, "the minimum-event halt must precede the null"
    assert c.MIN_EVENTS == 5000


def test_the_truncation_clause_is_asserted_out_of_reach_for_permuted_pairs():
    assert r.assert_truncation_unreachable(np.array([10, 20]), 200)[
        "margin_trading_days"] == 200 - 1 - c.TRUNCATION_EDGE - (20 + c.VOTE_TO)
    with pytest.raises(AssertionError, match="truncation edge"):
        r.assert_truncation_unreachable(np.array([140]), 200)


def test_the_ledger_row_carries_the_distance_in_bps_not_the_mcar():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    assert '"%.4f" % (1e4 * dist)' in src
    assert "1e4 * mcar_net" not in src
    assert "ledger_bps_column" in src
    assert "carries the DISTANCE in bps" in src


# ---------------------------------------------------------------------------
# 13. housekeeping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("tool", ["form4_event_study_r6.py", "run_t_r6.py"])
def test_no_credential_is_read_by_the_new_tools(tool):
    path = ROOT / "tools" / tool
    src = path.read_text()
    doc = ast.get_docstring(ast.parse(src)) or ""
    code = src.replace(doc, "")
    for forbidden in ("dotenv", "load_dotenv", ".env", "API_KEY", "api_key",
                      "secret", "requests", "urllib", "NASDAQ_DATA_LINK"):
        assert forbidden not in code, (tool, forbidden)


@pytest.mark.parametrize("tool", ["form4_event_study_r6.py", "run_t_r6.py"])
def test_new_tools_import_nothing_from_the_factory_tree(tool):
    src = (ROOT / "tools" / tool).read_text()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            for a in n.names:
                assert "Monte_Carlo" not in a.name
        if isinstance(n, ast.ImportFrom):
            assert not (n.module or "").startswith(("src.", "1_data", "2_research"))
    assert "Monte_Carlo_Strategies" not in src


def test_the_r5_engine_and_runner_are_not_modified_by_this_look():
    """R6 reuses R5; the sprint's binding rule is that it never edits it."""
    for name in ("item408_event_study_r5.py", "run_t_r5.py"):
        src = (ROOT / "tools" / name).read_text()
        assert "form4" not in src, name
        assert "T-R6" not in src, name


# ---------------------------------------------------------------------------
# 14. the step-3 halt: diagnostics that spend no look
# ---------------------------------------------------------------------------
def _span_book():
    """Two issuers, one of which stops long before the later dates."""
    f = frame("2010-01-04", 400, rate=0.001)
    book = make_book({"LONG": f, "STOP": f.iloc[:200]})
    wb = c.WindowBook(book, ["LONG", "STOP"], flat_index(book))
    return book, wb


def test_the_identity_pairing_always_retains_every_resolved_event():
    """Each real event is covered by its OWN issuer: the scorer's own floor."""
    book, wb = _span_book()
    groups = np.array([0, 1])
    d_idx = np.array([300, 100])
    prof = c.retention_profile(wb, groups, d_idx, 20, np.random.default_rng(0))
    assert prof["identity_retention"] == pytest.approx(1.0)
    assert prof["identity_used"] == 2


def test_retention_profile_reports_the_shortfall_against_the_sealed_rule():
    book, wb = _span_book()
    groups = np.array([0, 1])
    d_idx = np.array([300, 100])
    prof = c.retention_profile(wb, groups, d_idx, 40, np.random.default_rng(1))
    assert prof["required"] == c.RETENTION_MIN == 0.90
    assert prof["min"] <= 0.5 and prof["max"] <= 1.0
    assert "0.90" in prof["share_at_or_above"]
    assert isinstance(prof["reachable"], bool)


def test_a_reachable_retention_is_reported_as_reachable():
    book = make_book({"AAA": frame("2010-01-04", 400, rate=0.001),
                      "BBB": frame("2010-01-04", 400, rate=0.002)})
    wb = c.WindowBook(book, ["AAA", "BBB"], flat_index(book))
    prof = c.retention_profile(wb, np.array([0, 1]), np.array([100, 200]), 10,
                               np.random.default_rng(2))
    assert prof["mean"] == pytest.approx(1.0) and prof["reachable"]


def test_coverage_density_is_the_exact_ceiling_a_bijection_converges_to():
    book, wb = _span_book()
    groups = np.array([0, 1])
    d_idx = np.array([300, 100])
    cov = c.coverage_density(wb, groups, d_idx)
    assert cov["distinct_issuers"] == 2 and cov["distinct_dates"] == 2
    # LONG covers both dates, STOP covers only the early one: 3 of 4 pairs
    assert cov["pair_density_unweighted"] == pytest.approx(0.75)
    assert cov["event_weighted_expected_retention"] == pytest.approx(0.75)
    assert cov["issuers_covering_every_date"] == 1


def test_the_coverage_ceiling_matches_the_measured_retention_mean():
    book, wb = _span_book()
    groups = np.array([0, 0, 1, 1])
    d_idx = np.array([300, 100, 300, 100])
    cov = c.coverage_density(wb, groups, d_idx)
    prof = c.retention_profile(wb, groups, d_idx, 300, np.random.default_rng(3))
    assert prof["mean"] == pytest.approx(
        cov["event_weighted_expected_retention"], abs=0.03)


def test_issuer_span_profile_names_the_listings_and_delistings():
    book, wb = _span_book()
    sp = c.issuer_span_profile(book, wb, np.array([0, 1]), np.array([100, 300]))
    assert sp["issuers"] == 2
    assert sp["delisting_before_the_latest_date_plus_window"] == 1
    assert sp["spanning_the_whole_window"] == 1
    assert sp["date_range"][0] == bday(100) and sp["date_range"][1] == bday(300)


def test_the_diagnostic_forms_no_observed_figure():
    """Nothing in the halt diagnostic touches the voted mean."""
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef)
              and n.name == "retention_halt_diagnostic")
    body = ast.dump(fn)
    for forbidden in ("voted_returns", "mcar_net", "distance_of", "null_p",
                      "mde_from_null", "subgroup_distances"):
        assert forbidden not in body, forbidden
    for name in ("retention_profile", "coverage_density",
                 "issuer_span_profile"):
        assert name in body, name


def test_a_step3_null_failure_routes_through_the_halt_not_a_traceback():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage_controls")
    handlers = [h for n in ast.walk(fn) if isinstance(n, ast.Try)
                for h in n.handlers]
    assert handlers, "step 3 must not let the null's refusal escape"
    dumped = " ".join(ast.dump(h) for h in handlers)
    assert "halt_out" in dumped and "retention_halt_diagnostic" in dumped
    assert "STEP 3" in dumped


def test_the_step3_halt_precedes_the_known_null_and_the_plant():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "stage_controls")
    body = ast.dump(fn)
    assert body.index("STEP 3") < body.index("plant_drift")
    assert body.index("retention_halt_diagnostic") < body.index("plant_drift")


def test_the_halt_payload_carries_its_diagnostic():
    src = (ROOT / "tools" / "run_t_r6.py").read_text()
    fn = next(n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.FunctionDef) and n.name == "halt_out")
    assert "value='diagnostic'" in ast.dump(fn)
    assert "diagnostic" in [a.arg for a in fn.args.args + fn.args.kwonlyargs] \
        or any(a.arg == "diagnostic" for a in fn.args.args)


def test_the_runner_never_relaxes_the_sealed_retention_rule():
    """A halt explains the shortfall; it must not lower the bar it failed."""
    for name in ("run_t_r6.py", "form4_event_study_r6.py"):
        src = (ROOT / "tools" / name).read_text()
        tree = ast.parse(src)
        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        assert t.id != "RETENTION_MIN" or name == \
                            "form4_event_study_r6.py"
                    if isinstance(t, ast.Attribute):
                        assert t.attr != "RETENTION_MIN", name
    assert c.RETENTION_MIN == r5.RETENTION_MIN == 0.90
    assert "RETENTION_MIN" in c.SHARED_WITH_R5
