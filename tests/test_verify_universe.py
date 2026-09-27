"""
================================================================================
Script:       test_verify_universe.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the universe-verification reducers: the S&P membership walk, the
              delisted-name presence check, and the master composition summary.
Entry Point:  No — pytest
Dependencies: stdlib (importlib, pathlib); pandas; the verification tool
Last Sprint:  Sprint R0-C — Paid tier verified; full-history universe (2026-09-27)
================================================================================

The live run returned a perfect reconciliation — 114 of 114 quarterly steps exact
— and a perfect presence check. A measurement that can only return "perfect" is
worth nothing, so each reducer here is shown detecting a constructed break as
well as passing a clean case.

Falsifiable throughout: each test fails on a constructed violation of the
property it names.
================================================================================
"""

import importlib.util
import pathlib

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "verify_universe.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_universe", TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VU = _load()


def _sp500(rows):
    return pd.DataFrame(rows, columns=["date", "action", "ticker"])


# ---------------------------------------------------------------------------
# reconcile_sp500 — must be able to fail
# ---------------------------------------------------------------------------

def test_a_consistent_membership_history_reconciles_exactly():
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2020-03-31", "historical", "BBB"),
        ("2020-05-01", "added", "CCC"),
        ("2020-05-01", "removed", "BBB"),
        ("2020-06-30", "historical", "AAA"),
        ("2020-06-30", "historical", "CCC"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["steps"] == 1
    assert out["steps_exact"] == 1
    assert out["total_extras"] == 0
    assert out["total_missing"] == 0


def test_a_missing_addition_event_shows_as_missing():
    """The snapshot gains a name the events never announced — the walk must
    report it as missing rather than quietly agreeing."""
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2020-06-30", "historical", "AAA"),
        ("2020-06-30", "historical", "CCC"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["total_missing"] == 1
    assert out["total_extras"] == 0
    assert out["steps_exact"] == 0


def test_a_missing_removal_event_shows_as_extra():
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2020-03-31", "historical", "BBB"),
        ("2020-06-30", "historical", "AAA"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["total_extras"] == 1
    assert out["total_missing"] == 0


def test_an_event_outside_the_step_window_is_not_applied_early():
    """An addition dated after the closing snapshot belongs to the next step.
    Applying it early would manufacture an agreement."""
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2020-07-15", "added", "CCC"),
        ("2020-06-30", "historical", "AAA"),
        ("2020-06-30", "historical", "CCC"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["total_missing"] == 1


def test_an_event_on_the_closing_snapshot_date_is_applied():
    """The window is exclusive at the open edge and inclusive at the close, so an
    event stamped on the closing date counts toward that step."""
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2020-06-30", "added", "CCC"),
        ("2020-06-30", "historical", "AAA"),
        ("2020-06-30", "historical", "CCC"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["steps_exact"] == 1


def test_the_current_snapshot_participates_as_a_membership_date():
    """'current' is the live membership list; excluding it would drop the final
    step, which is the one a live universe depends on."""
    df = _sp500([
        ("2020-03-31", "historical", "AAA"),
        ("2026-09-27", "current", "AAA"),
    ])
    out = VU.reconcile_sp500(df)
    assert out["snapshot_dates"] == 2
    assert out["snapshot_date_last"] == "2026-09-27"


def test_median_membership_is_reported():
    df = _sp500([("2020-03-31", "historical", "A%d" % i) for i in range(5)]
                + [("2020-06-30", "historical", "A%d" % i) for i in range(5)])
    assert VU.reconcile_sp500(df)["median_membership"] == 5


def test_no_snapshots_yields_no_steps_rather_than_an_exception():
    assert VU.reconcile_sp500(_sp500([("2020-01-01", "added", "AAA")]))["steps"] == 0


# ---------------------------------------------------------------------------
# delisted_presence_check — must be able to report absence
# ---------------------------------------------------------------------------

def _master(rows):
    return pd.DataFrame(rows, columns=["ticker", "permaticker", "isdelisted",
                                       "lastpricedate"])


def test_a_present_dead_name_reads_present_at_death():
    m = _master([("DEAD", "1", "Y", "2009-03-04")])
    out = VU.delisted_presence_check(m, {"DEAD": "2009-03-04"}, n=5)
    assert out["present_at_death"] == 1
    assert out["absent"] == 0


def test_a_dropped_dead_name_reads_absent():
    """The survivorship test that matters: a vendor that purged delisted names
    must be caught, not reported clean."""
    m = _master([("DEAD", "1", "Y", "2009-03-04")])
    out = VU.delisted_presence_check(m, {}, n=5)
    assert out["absent"] == 1
    assert out["present"] == 0


def test_a_truncated_history_reads_off_date_not_present_at_death():
    """Rows exist but stop years early — present, but not present at death."""
    m = _master([("DEAD", "1", "Y", "2009-03-04")])
    out = VU.delisted_presence_check(m, {"DEAD": "2005-01-03"}, n=5)
    assert out["present"] == 1
    assert out["present_at_death"] == 0
    assert out["detail"][0]["verdict"] == "PRESENT_BUT_OFF_DATE"


def test_a_few_sessions_short_still_counts_as_present_at_death():
    m = _master([("DEAD", "1", "Y", "2009-03-06")])
    out = VU.delisted_presence_check(m, {"DEAD": "2009-03-04"}, n=5)
    assert out["present_at_death"] == 1


def test_living_names_are_not_sampled():
    """Sampling a live ticker would prove nothing about survivorship."""
    m = _master([("ALIVE", "1", "N", "2026-09-25"),
                 ("DEAD", "2", "Y", "2009-03-04")])
    out = VU.delisted_presence_check(m, {"ALIVE": "2026-09-25",
                                         "DEAD": "2009-03-04"}, n=5)
    assert [r["ticker"] for r in out["detail"]] == ["DEAD"]


def test_recent_deaths_are_not_sampled():
    """A name that died last month could survive a purge by luck, so the sample
    is drawn from names dead well before the snapshot."""
    m = _master([("RECENT", "1", "Y", "2024-06-01"),
                 ("OLD", "2", "Y", "2009-03-04")])
    out = VU.delisted_presence_check(m, {"RECENT": "2024-06-01",
                                         "OLD": "2009-03-04"}, n=5)
    assert [r["ticker"] for r in out["detail"]] == ["OLD"]


# ---------------------------------------------------------------------------
# summarise_master — composition and key uniqueness
# ---------------------------------------------------------------------------

def _full_master(rows):
    return pd.DataFrame(rows, columns=[
        "table", "permaticker", "ticker", "isdelisted",
        "firstpricedate", "lastpricedate"])


def test_master_splits_by_vendor_table_and_isolates_the_price_slice():
    m = _full_master([
        ("SEP", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
        ("SEP", "2", "BBB", "Y", "1999-01-04", "2009-03-04"),
        ("SF1", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
    ])
    out = VU.summarise_master(m)
    assert out["rows"] == 3
    assert out["price_table_rows"] == 2
    assert out["price_table_active"] == 1
    assert out["price_table_delisted"] == 1
    assert out["rows_by_table"] == {"SEP": 2, "SF1": 1}


def test_permaticker_uniqueness_is_judged_within_the_price_table():
    """permaticker repeats across vendor tables by design, so a whole-file
    uniqueness test would always fail and tell us nothing."""
    m = _full_master([
        ("SEP", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
        ("SF1", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
    ])
    out = VU.summarise_master(m)
    assert out["permaticker_unique_within_price_table"] is True
    assert out["permaticker_distinct_all_tables"] == 1


def test_a_duplicated_permaticker_inside_the_price_table_is_reported():
    m = _full_master([
        ("SEP", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
        ("SEP", "1", "AAB", "N", "1998-01-02", "2026-09-25"),
    ])
    out = VU.summarise_master(m)
    assert out["permaticker_unique_within_price_table"] is False
    assert out["permaticker_duplicate_rows_price_table"] == 1


def test_blank_price_dates_are_counted_as_absent_not_as_a_date():
    m = _full_master([
        ("SEP", "1", "AAA", "N", "1998-01-02", "2026-09-25"),
        ("SEP", "2", "BBB", "N", "", ""),
    ])
    out = VU.summarise_master(m)
    assert out["firstpricedate_present"] == 1
    assert out["firstpricedate_present_pct"] == 50.0
    assert out["firstpricedate_min"] == "1998-01-02"


# ---------------------------------------------------------------------------
# year_of
# ---------------------------------------------------------------------------

def test_year_extraction_is_the_leading_four_characters():
    s = pd.Series(["1997-12-31", "2026-09-25"])
    assert list(VU.year_of(s)) == ["1997", "2026"]
