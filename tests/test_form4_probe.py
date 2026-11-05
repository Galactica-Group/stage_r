"""
================================================================================
Script:       test_form4_probe.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Fixture-only battery on tools/form4_probe.py: the purchase filter,
              the relationship join, the dollar value, the filing lag, the
              trading-day clustering, the routine-purchase test, the Sharadar
              CIK join, the manifest, and the static guards (no .env, no factory
              import).
Entry Point:  No — pytest
Dependencies: stdlib (ast, json, pathlib, sys, zipfile), numpy, pandas; the tool
Last Sprint:  Sprint R6-0 — Form 4 bulk dataset and inventory (2026-11-05)
================================================================================

Every assertion is falsifiable on a constructed violation of the property it
names. No test reads a downloaded archive, a vendor price, or the network: the
archive-shaped fixtures are built in the test from the headers the 2024q1
archive publishes, so a header change breaks the fixture rather than passing
silently.

No test forms a return, a price change or an abnormal return. The only price
figure any test forms is shares times price, which is the size of a purchase as
filed, not a statistic about it.
================================================================================
"""

from __future__ import annotations

import ast
import io
import json
import pathlib
import re
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL = REPO / "tools" / "form4_probe.py"
FACTORY_DIR_NAME = "Monte_Carlo_Strategies"
ALLOWED_THIRD_PARTY = {"requests", "pandas", "pyarrow", "numpy"}
ALLOWED_LOCAL = {"edgar_item408_probe"}

sys.path.insert(0, str(REPO / "tools"))
import form4_probe as F  # noqa: E402


# The headers the 2024q1 archive publishes, used verbatim so a vendor schema
# change breaks the fixtures rather than passing silently.
SUBMISSION_COLS = (
    "ACCESSION_NUMBER FILING_DATE PERIOD_OF_REPORT DATE_OF_ORIG_SUB "
    "NO_SECURITIES_OWNED NOT_SUBJECT_SEC16 FORM3_HOLDINGS_REPORTED "
    "FORM4_TRANS_REPORTED DOCUMENT_TYPE ISSUERCIK ISSUERNAME "
    "ISSUERTRADINGSYMBOL REMARKS AFF10B5ONE"
).split()
OWNER_COLS = (
    "ACCESSION_NUMBER RPTOWNERCIK RPTOWNERNAME RPTOWNER_RELATIONSHIP "
    "RPTOWNER_TITLE RPTOWNER_TXT RPTOWNER_STREET1 RPTOWNER_STREET2 "
    "RPTOWNER_CITY RPTOWNER_STATE RPTOWNER_ZIPCODE RPTOWNER_STATE_DESC "
    "FILE_NUMBER"
).split()
NONDERIV_COLS = (
    "ACCESSION_NUMBER NONDERIV_TRANS_SK SECURITY_TITLE SECURITY_TITLE_FN "
    "TRANS_DATE TRANS_DATE_FN DEEMED_EXECUTION_DATE DEEMED_EXECUTION_DATE_FN "
    "TRANS_FORM_TYPE TRANS_CODE EQUITY_SWAP_INVOLVED EQUITY_SWAP_TRANS_CD_FN "
    "TRANS_TIMELINESS TRANS_TIMELINESS_FN TRANS_SHARES TRANS_SHARES_FN "
    "TRANS_PRICEPERSHARE TRANS_PRICEPERSHARE_FN TRANS_ACQUIRED_DISP_CD "
    "TRANS_ACQUIRED_DISP_CD_FN SHRS_OWND_FOLWNG_TRANS SHRS_OWND_FOLWNG_TRANS_FN "
    "VALU_OWND_FOLWNG_TRANS VALU_OWND_FOLWNG_TRANS_FN "
    "DIRECT_INDIRECT_OWNERSHIP DIRECT_INDIRECT_OWNERSHIP_FN "
    "NATURE_OF_OWNERSHIP NATURE_OF_OWNERSHIP_FN"
).split()


def _row(cols, **kw):
    r = {c: "" for c in cols}
    unknown = set(kw) - set(cols)
    assert not unknown, "fixture names a column the archive has not: %s" % unknown
    r.update({k: str(v) for k, v in kw.items()})
    return r


def _tsv(cols, rows):
    out = io.StringIO()
    out.write("\t".join(cols) + "\n")
    for r in rows:
        out.write("\t".join(r[c] for c in cols) + "\n")
    return out.getvalue().encode("utf-8")


def build_archive(path, submissions, owners, transactions):
    """An archive-shaped zip with the three members the tool reads."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(F.MEMBER_SUBMISSION, _tsv(SUBMISSION_COLS, submissions))
        z.writestr(F.MEMBER_OWNER, _tsv(OWNER_COLS, owners))
        z.writestr(F.MEMBER_NONDERIV, _tsv(NONDERIV_COLS, transactions))
    return path


# ---------------------------------------------------------------------------
# archive naming
# ---------------------------------------------------------------------------

def test_archive_names_parse_and_sort_oldest_first():
    assert F.parse_archive_name("2006q1_form345.zip") == (2006, 1)
    assert F.parse_archive_name("2026q3_form345.zip") == (2026, 3)
    assert F.parse_archive_name("2024q1_notes.zip") is None
    assert F.parse_archive_name("2024q5_form345.zip") is None
    names = ["2007q1_form345.zip", "2006q4_form345.zip", "2006q1_form345.zip"]
    assert sorted(names, key=F.archive_sort_key) == [
        "2006q1_form345.zip", "2006q4_form345.zip", "2007q1_form345.zip"]


def test_archive_links_are_extracted_and_foreign_zips_ignored():
    html = (
        '<a href="/files/structureddata/data/insider-transactions-data-sets/'
        '2006q1_form345.zip">2006 Q1 345</a>'
        '<a href="/files/dera/data/x/2024q1_notes.zip">notes</a>'
        '<a href="/files/datastandardsinnovation/data/'
        'insider-transactions-data-sets/2026q3_form345.zip">2026 Q3 345</a>'
        '<a href="/about/anything">nav</a>'
    )
    arcs = F.extract_archive_links(html)
    assert [a["name"] for a in arcs] == [
        "2006q1_form345.zip", "2026q3_form345.zip"]
    assert "datastandardsinnovation" in arcs[-1]["href"]


def test_archive_url_refuses_a_non_sec_host():
    assert F.archive_url("/files/x/2006q1_form345.zip").startswith(
        "https://www.sec.gov/")
    assert F.archive_url("https://www.sec.gov/a.zip") == "https://www.sec.gov/a.zip"
    with pytest.raises(F.StopRun):
        F.archive_url("https://example.com/2006q1_form345.zip")
    with pytest.raises(F.StopRun):
        F.archive_url("files/x.zip")


# ---------------------------------------------------------------------------
# the purchase filter
# ---------------------------------------------------------------------------

def _sub(doc_type="4", **kw):
    return _row(SUBMISSION_COLS, DOCUMENT_TYPE=doc_type, **kw)


def _trn(code="P", ad="A", price="10.5", shares="100", **kw):
    return _row(NONDERIV_COLS, TRANS_CODE=code, TRANS_ACQUIRED_DISP_CD=ad,
                TRANS_PRICEPERSHARE=price, TRANS_SHARES=shares, **kw)


def test_the_purchase_filter_keeps_only_form_4_open_market_priced_acquisitions():
    assert F.keep_purchase(_sub("4"), _trn()) is True
    # Form 3, Form 5 and every amendment are out.
    for dt in ("3", "5", "4/A", "3/A", "5/A", ""):
        assert F.keep_purchase(_sub(dt), _trn()) is False, dt
    # Only code P is an open-market purchase.
    for code in ("A", "S", "F", "M", "G", "D", "J", "C", "X", ""):
        assert F.keep_purchase(_sub(), _trn(code=code)) is False, code
    # Only the acquired side.
    for ad in ("D", ""):
        assert F.keep_purchase(_sub(), _trn(ad=ad)) is False, ad
    # A price that is blank, zero or negative is dropped, never imputed.
    for price in ("", "0", "0.00", "-1", "abc", "nan"):
        assert F.keep_purchase(_sub(), _trn(price=price)) is False, price
    assert F.keep_purchase(_sub(), _trn(price="0.0001")) is True


def test_is_form4_is_exact_and_not_a_prefix_test():
    assert F.is_form4("4") is True
    assert F.is_form4(" 4 ") is True
    assert F.is_form4("4/A") is False
    assert F.is_form4("45") is False


# ---------------------------------------------------------------------------
# the relationship join
# ---------------------------------------------------------------------------

def test_relationship_flags_are_independent_and_a_row_may_carry_several():
    assert F.parse_relationship("Director") == {
        "director": True, "officer": False,
        "ten_percent_owner": False, "other": False}
    assert F.parse_relationship("Officer") == {
        "director": False, "officer": True,
        "ten_percent_owner": False, "other": False}
    assert F.parse_relationship("TenPercentOwner") == {
        "director": False, "officer": False,
        "ten_percent_owner": True, "other": False}
    assert F.parse_relationship("Other") == {
        "director": False, "officer": False,
        "ten_percent_owner": False, "other": True}
    # The field is a comma-joined set, so several hats at once.
    assert F.parse_relationship("Director,Officer") == {
        "director": True, "officer": True,
        "ten_percent_owner": False, "other": False}
    assert F.parse_relationship(
        "Director,Officer,TenPercentOwner,Other") == {
            k: True for k in F.RELATIONSHIPS}
    # Blank and unreadable resolve to all False, never to a default hat.
    for bad in ("", None, "Trustee"):
        assert F.parse_relationship(bad) == {k: False for k in F.RELATIONSHIPS}


def test_a_ten_percent_owner_is_not_read_as_an_officer():
    """Non-vacuity: the flags do not bleed into one another."""
    flags = F.parse_relationship("TenPercentOwner")
    assert flags["officer"] is False and flags["director"] is False


# ---------------------------------------------------------------------------
# value, lag, flag
# ---------------------------------------------------------------------------

def test_the_dollar_value_is_shares_times_price():
    assert F.dollar_value("100", "10.5") == pytest.approx(1050.0)
    assert F.dollar_value("1,500", "2.25") == pytest.approx(3375.0)
    assert F.dollar_value("", "10") is None
    assert F.dollar_value("100", "") is None
    assert F.dollar_value("100", "abc") is None


def test_the_filing_lag_is_calendar_days_filing_minus_transaction():
    assert F.filing_lag_days("05-JAN-2024", "03-JAN-2024") == 2
    # Across a weekend the calendar count is larger than the business count.
    assert F.filing_lag_days("08-JAN-2024", "04-JAN-2024") == 4
    assert F.filing_lag_days("2024-01-05", "2024-01-05") == 0
    # A filing stamped before its transaction is reported negative, not clipped.
    assert F.filing_lag_days("03-JAN-2024", "05-JAN-2024") == -2
    assert F.filing_lag_days("05-JAN-2024", "") is None
    assert F.filing_lag_days("", "05-JAN-2024") is None


def test_the_10b5_1_flag_reads_both_spellings_and_blank_is_not_false():
    for t in ("1", "true", "TRUE", "True", " 1 "):
        assert F.normalise_flag(t) is True, t
    for f in ("0", "false", "FALSE", " 0 "):
        assert F.normalise_flag(f) is False, f
    for n in ("", None, "   ", "maybe"):
        assert F.normalise_flag(n) is None, n


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------

def _calendar(start="2024-01-02", n=120):
    days = pd.bdate_range(start, periods=n)
    return days.values.astype("datetime64[D]")


def test_clustering_opens_on_the_first_filing_day_and_merges_inside_the_window():
    cal = _calendar()
    rows = [
        # issuer A: three filings inside 20 trading days -> one event.
        {"issuer_cik": "A", "filing_date": "2024-01-02", "owner_cik": "o1", "value": 1000.0},
        {"issuer_cik": "A", "filing_date": "2024-01-10", "owner_cik": "o2", "value": 2000.0},
        {"issuer_cik": "A", "filing_date": "2024-01-25", "owner_cik": "o1", "value": 500.0},
        # a fourth beyond the window opens the next event.
        {"issuer_cik": "A", "filing_date": "2024-03-01", "owner_cik": "o3", "value": 10.0},
        # issuer B is clustered independently.
        {"issuer_cik": "B", "filing_date": "2024-01-03", "owner_cik": "o9", "value": 7.0},
    ]
    ev = F.cluster_events(pd.DataFrame(rows), 20, cal)
    a = ev[ev["issuer_cik"] == "A"].sort_values("event_start").reset_index(drop=True)
    assert len(a) == 2
    assert a.loc[0, "n_transactions"] == 3
    assert a.loc[0, "n_insiders"] == 2            # the multi-insider count
    assert a.loc[0, "total_value"] == pytest.approx(3500.0)
    assert str(a.loc[0, "event_start"])[:10] == "2024-01-02"
    assert a.loc[1, "n_transactions"] == 1
    assert len(ev[ev["issuer_cik"] == "B"]) == 1


def test_a_wider_window_merges_what_a_narrower_one_splits():
    """Non-vacuity: the window argument is load-bearing, not decorative."""
    cal = _calendar()
    rows = [
        {"issuer_cik": "A", "filing_date": "2024-01-02", "owner_cik": "o1", "value": 1.0},
        {"issuer_cik": "A", "filing_date": "2024-02-15", "owner_cik": "o2", "value": 1.0},
    ]
    df = pd.DataFrame(rows)
    assert len(F.cluster_events(df, 20, cal)) == 2
    assert len(F.cluster_events(df, 60, cal)) == 1


def test_a_filing_on_a_non_trading_day_lands_on_the_next_trading_day():
    cal = _calendar()
    # 2024-01-06 is a Saturday; it must share a window with the Monday.
    rows = [
        {"issuer_cik": "A", "filing_date": "2024-01-06", "owner_cik": "o1", "value": 1.0},
        {"issuer_cik": "A", "filing_date": "2024-01-08", "owner_cik": "o2", "value": 1.0},
    ]
    ev = F.cluster_events(pd.DataFrame(rows), 20, cal)
    assert len(ev) == 1 and ev.loc[0, "n_insiders"] == 2


def test_clustering_an_empty_table_returns_an_empty_frame_not_an_error():
    cal = _calendar()
    ev = F.cluster_events(
        pd.DataFrame(columns=["issuer_cik", "filing_date", "owner_cik", "value"]),
        20, cal)
    assert len(ev) == 0 and "n_insiders" in ev.columns


def test_the_trading_day_calendar_drops_dates_below_the_ticker_floor():
    day = np.array(
        ["2024-01-02"] * 3 + ["2024-01-03"] * 1 + ["2024-01-04"] * 3,
        dtype="datetime64[D]")
    cal, stats = F.trading_day_calendar(day, min_tickers=3)
    assert list(cal.astype(str)) == ["2024-01-02", "2024-01-04"]
    assert stats["dates_excluded"] == 1 and stats["trading_days"] == 2
    with pytest.raises(AssertionError):
        F.trading_day_calendar(day, min_tickers=99)


# ---------------------------------------------------------------------------
# the routine-purchase test
# ---------------------------------------------------------------------------

def test_the_routine_test_binds_on_a_constructed_three_year_history():
    rows = []
    for y in (2017, 2018, 2019, 2020):
        rows.append({"owner_cik": "o1", "issuer_cik": "A",
                     "trans_date": "%d-03-15" % y})
    # A pair with only two prior years in the same month is not routine.
    for y in (2019, 2020):
        rows.append({"owner_cik": "o2", "issuer_cik": "B",
                     "trans_date": "%d-03-15" % y})
    # A pair with three prior years in a different month is not routine.
    for y in (2017, 2018, 2019):
        rows.append({"owner_cik": "o3", "issuer_cik": "C",
                     "trans_date": "%d-04-15" % y})
    rows.append({"owner_cik": "o3", "issuer_cik": "C", "trans_date": "2020-03-15"})
    out = F.routine_pairs(pd.DataFrame(rows))
    flag = {(r["owner_cik"], r["trans_date"]): r["is_routine"]
            for r in out.to_dict("records")}
    assert flag[("o1", "2020-03-15")] is True
    assert flag[("o1", "2019-03-15")] is False     # only two prior years in table
    assert flag[("o2", "2020-03-15")] is False
    assert flag[("o3", "2020-03-15")] is False     # prior years are a different month
    assert flag[("o3", "2019-04-15")] is False


def test_the_routine_test_is_owner_and_issuer_specific():
    """Non-vacuity: the same months at a different issuer do not make it routine."""
    rows = [{"owner_cik": "o1", "issuer_cik": "A", "trans_date": "%d-03-15" % y}
            for y in (2017, 2018, 2019)]
    rows.append({"owner_cik": "o1", "issuer_cik": "Z", "trans_date": "2020-03-15"})
    out = F.routine_pairs(pd.DataFrame(rows))
    z = out[out["issuer_cik"] == "Z"]
    assert bool(z["is_routine"].iloc[0]) is False


# ---------------------------------------------------------------------------
# the Sharadar CIK join
# ---------------------------------------------------------------------------

def _tickers_frame():
    return pd.DataFrame(
        {
            "table": ["SEP", "SEP", "SEP", "SFP"],
            "permaticker": [111, 222, 333, 444],
            "ticker": ["AAA", "BBB", "BBB2", "FFF"],
            "secfilings": [
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000123",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000456",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000456",
                "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000789",
            ],
            "relatedtickers": [None, None, None, None],
        }
    )


def test_the_cik_join_matches_pads_and_flags_multi_mapped_ciks():
    cmap = F.build_cik_map(_tickers_frame())
    hits, multi = F.match_issuer_cik("123", cmap)
    assert [h["ticker"] for h in hits] == ["AAA"] and multi is False
    # Two permatickers behind one CIK is a multi-mapping, reported not resolved.
    hits, multi = F.match_issuer_cik("0000000456", cmap)
    assert multi is True and len(hits) == 2
    # A non-SEP table row is not in the map at all.
    assert F.match_issuer_cik("789", cmap) == ([], False)
    assert F.match_issuer_cik("999", cmap) == ([], False)
    assert F.match_issuer_cik(None, cmap) == ([], False)


def test_cik_padding_is_ten_digits():
    assert F.pad_cik("123") == "0000000123"
    assert F.pad_cik(123) == "0000000123"
    assert F.pad_cik("0000000123") == "0000000123"
    assert F.pad_cik("") is None and F.pad_cik(None) is None


def test_the_proposed_halves_partition_filing_dates_without_overlap():
    assert F.half_of("2006-01-01") == "A"
    assert F.half_of("2015-12-31") == "A"
    assert F.half_of("2016-01-01") == "B"
    assert F.half_of("2022-12-31") == "B"
    assert F.half_of("2023-01-01") == "holdout"
    assert F.half_of("2005-12-31") is None
    assert F.half_of("") is None


# ---------------------------------------------------------------------------
# archive reading, end to end on a fixture
# ---------------------------------------------------------------------------

def test_an_archive_fixture_yields_the_rows_the_filter_describes(tmp_path):
    subs = [
        _sub("4", ACCESSION_NUMBER="A1", FILING_DATE="05-JAN-2024",
             ISSUERCIK="123", ISSUERNAME="Alpha Inc", ISSUERTRADINGSYMBOL="AAA",
             AFF10B5ONE="0"),
        _sub("4", ACCESSION_NUMBER="A2", FILING_DATE="06-JAN-2024",
             ISSUERCIK="456", ISSUERNAME="Beta Inc", ISSUERTRADINGSYMBOL="BBB",
             AFF10B5ONE="true"),
        _sub("4/A", ACCESSION_NUMBER="A3", FILING_DATE="07-JAN-2024",
             ISSUERCIK="123", AFF10B5ONE="0"),
        _sub("3", ACCESSION_NUMBER="A4", FILING_DATE="08-JAN-2024",
             ISSUERCIK="123", AFF10B5ONE=""),
        # a Form 4 whose only owner is a ten percent owner: filtered out
        _sub("4", ACCESSION_NUMBER="A5", FILING_DATE="09-JAN-2024",
             ISSUERCIK="789", AFF10B5ONE="0"),
        # a Form 4 filed before the box existed: the flag is blank, not false
        _sub("4", ACCESSION_NUMBER="A6", FILING_DATE="10-JAN-2024",
             ISSUERCIK="123", ISSUERNAME="Alpha Inc",
             ISSUERTRADINGSYMBOL="AAA", AFF10B5ONE=""),
    ]
    owners = [
        _row(OWNER_COLS, ACCESSION_NUMBER="A1", RPTOWNERCIK="900",
             RPTOWNERNAME="Doe John", RPTOWNER_RELATIONSHIP="Director",
             RPTOWNER_TITLE=""),
        # a joint filing: the primary owner is the lowest CIK
        _row(OWNER_COLS, ACCESSION_NUMBER="A2", RPTOWNERCIK="800",
             RPTOWNERNAME="Roe Jane", RPTOWNER_RELATIONSHIP="Officer",
             RPTOWNER_TITLE="CFO"),
        _row(OWNER_COLS, ACCESSION_NUMBER="A2", RPTOWNERCIK="700",
             RPTOWNERNAME="Coe Sam",
             RPTOWNER_RELATIONSHIP="Director,TenPercentOwner"),
        _row(OWNER_COLS, ACCESSION_NUMBER="A3", RPTOWNERCIK="900",
             RPTOWNER_RELATIONSHIP="Director"),
        _row(OWNER_COLS, ACCESSION_NUMBER="A4", RPTOWNERCIK="900",
             RPTOWNER_RELATIONSHIP="Director"),
        _row(OWNER_COLS, ACCESSION_NUMBER="A5", RPTOWNERCIK="600",
             RPTOWNER_RELATIONSHIP="TenPercentOwner"),
        _row(OWNER_COLS, ACCESSION_NUMBER="A6", RPTOWNERCIK="500",
             RPTOWNER_RELATIONSHIP="Officer,Other", RPTOWNER_TITLE="CEO"),
    ]
    trans = [
        _trn(ACCESSION_NUMBER="A1", NONDERIV_TRANS_SK="1",
             TRANS_DATE="03-JAN-2024", shares="100", price="10.5",
             DIRECT_INDIRECT_OWNERSHIP="D", SECURITY_TITLE="Common Stock"),
        _trn(ACCESSION_NUMBER="A1", NONDERIV_TRANS_SK="2",
             TRANS_DATE="03-JAN-2024", code="S", shares="50", price="11"),
        _trn(ACCESSION_NUMBER="A2", NONDERIV_TRANS_SK="3",
             TRANS_DATE="05-JAN-2024", shares="10", price="2",
             DIRECT_INDIRECT_OWNERSHIP="I"),
        _trn(ACCESSION_NUMBER="A2", NONDERIV_TRANS_SK="4",
             TRANS_DATE="05-JAN-2024", shares="10", price=""),
        _trn(ACCESSION_NUMBER="A3", NONDERIV_TRANS_SK="5",
             TRANS_DATE="05-JAN-2024"),
        _trn(ACCESSION_NUMBER="A4", NONDERIV_TRANS_SK="6",
             TRANS_DATE="05-JAN-2024"),
        _trn(ACCESSION_NUMBER="A5", NONDERIV_TRANS_SK="7",
             TRANS_DATE="05-JAN-2024"),
        _trn(ACCESSION_NUMBER="A6", NONDERIV_TRANS_SK="8",
             TRANS_DATE="08-JAN-2024", shares="5", price="3"),
    ]
    path = build_archive(tmp_path / "2024q1_form345.zip", subs, owners, trans)
    p, o, cov = F.purchases_from_archive(str(path), "2024q1_form345.zip")

    assert sorted(p["trans_sk"]) == ["1", "3", "8"]
    a1 = p[p["trans_sk"] == "1"].iloc[0]
    assert a1["issuer_cik"] == "0000000123"
    assert a1["issuer_name"] == "Alpha Inc" and a1["issuer_symbol"] == "AAA"
    assert a1["owner_cik"] == "0000000900" and a1["owner_name"] == "Doe John"
    assert bool(a1["director"]) and not bool(a1["officer"])
    assert a1["value"] == pytest.approx(1050.0)
    assert a1["filing_lag_days"] == 2
    assert a1["aff10b5one_as_filed"] == "0"
    assert a1["aff10b5one"] is not None and bool(a1["aff10b5one"]) is False
    assert a1["direct_indirect"] == "D"

    a2 = p[p["trans_sk"] == "3"].iloc[0]
    assert a2["owner_cik"] == "0000000700"          # lowest CIK on the filing
    assert a2["owner_count"] == 2
    assert bool(a2["any_officer"]) and bool(a2["any_director"])
    assert bool(a2["any_ten_percent_owner"])
    assert not bool(a2["officer"])                  # the primary owner is not one
    assert bool(a2["aff10b5one"]) is True

    # The blank box is carried as blank: a pre-2023 filing is not a "false".
    a6 = p[p["trans_sk"] == "8"].iloc[0]
    assert a6["aff10b5one_as_filed"] == "" and a6["aff10b5one"] is None
    assert bool(a6["officer"]) and bool(a6["other"])
    assert a2["direct_indirect"] == "I"

    assert cov["form4_submissions"] == 4
    assert cov["code_p_acquired"] == 5       # A1, A2 x2, A5, A6 (A3/A4 not Form 4)
    assert cov["code_p_acquired_priced"] == 4
    assert cov["dropped_no_director_or_officer"] == 1
    assert cov["purchases_kept"] == 3
    assert set(o["accession"]) == {"A1", "A2", "A5", "A6"}


def test_a_fixture_naming_a_column_the_archive_has_not_is_rejected():
    """Non-vacuity: the fixtures are bound to the published headers."""
    with pytest.raises(AssertionError):
        _row(SUBMISSION_COLS, NOT_A_REAL_COLUMN="x")


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------

def test_the_manifest_round_trips_sorted_and_newline_terminated(tmp_path):
    dest = tmp_path / "sub" / "manifest.json"
    man = {"vintage": "v1", "rows": {"purchases": 3}, "archives": 1}
    F.write_manifest(str(dest), man)
    raw = dest.read_bytes()
    assert raw.endswith(b"\n")
    assert json.loads(raw) == man
    keys = list(json.loads(raw).keys())
    assert keys == sorted(keys)
    assert F.load_manifest(str(dest)) == man
    assert F.load_manifest(str(tmp_path / "absent.json")) == {}


def test_the_tracked_manifest_path_is_under_manifests_and_names_the_vintage():
    p = pathlib.Path(F.tracked_manifest_path("20261105"))
    assert p.parent == REPO / "manifests"
    assert p.name == "form4_20261105.json"


# ---------------------------------------------------------------------------
# static guards
# ---------------------------------------------------------------------------

def _tree():
    return ast.parse(TOOL.read_text(encoding="utf-8"))


def _top_level_imports(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
            elif node.level:
                mods.add("__RELATIVE__")
    return mods


def _docstring_nodes(tree):
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                out.add(id(body[0].value))
    return out


def test_imports_are_stdlib_allowed_third_party_or_the_shared_probe():
    mods = _top_level_imports(_tree())
    assert "__RELATIVE__" not in mods
    stdlib = set(sys.stdlib_module_names)
    offenders = {
        m for m in mods
        if m not in stdlib and m not in ALLOWED_THIRD_PARTY
        and m not in ALLOWED_LOCAL
    }
    assert not offenders, "disallowed imports: %s" % sorted(offenders)


def test_the_tool_names_no_factory_path_and_imports_nothing_from_it():
    src = TOOL.read_text(encoding="utf-8")
    assert FACTORY_DIR_NAME not in src
    for mod in sys.modules:
        if FACTORY_DIR_NAME.lower() in mod.lower():
            pytest.fail("a factory module is importable in this process: %s" % mod)
    assert FACTORY_DIR_NAME not in str(pathlib.Path(F.__file__).resolve())


def test_the_tool_opens_no_dotenv_and_reads_no_credential():
    """Prose may say the tool opens no .env; the code may not name one."""
    tree = _tree()
    docs = _docstring_nodes(tree)
    literals = [
        n.value.lower() for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docs
    ]
    identifiers = [
        (n.id if isinstance(n, ast.Name) else n.attr).lower()
        for n in ast.walk(tree) if isinstance(n, (ast.Name, ast.Attribute))
    ]
    for token in (".env", "dotenv", "api_key", "apikey", "authorization",
                  "bearer", "secret", "password"):
        hits = [s for s in literals if token in s]
        assert not hits, "credential-shaped literal: %s -> %s" % (token, hits)
    for token in ("api_key", "apikey", "authorization", "bearer", "secret",
                  "password"):
        flat = token.replace("_", "")
        bad = [i for i in identifiers if i.replace("_", "") == flat]
        assert not bad, "credential-shaped identifier: %s" % token
    assert "dotenv" not in identifiers


def test_the_credential_scan_bites_on_a_constructed_violation():
    """Non-vacuity: the scan flags an offending literal outside a docstring."""
    tree = ast.parse('"""a docstring naming .env is fine"""\nx = "my api_key"\n')
    docs = _docstring_nodes(tree)
    literals = [
        n.value.lower() for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
        and id(n) not in docs
    ]
    assert any("api_key" in s for s in literals)


def test_every_sec_request_goes_through_the_six_per_second_limiter():
    assert F.MAX_REQUESTS_PER_SECOND == 6
    src = TOOL.read_text(encoding="utf-8")
    assert "RateLimiter(max_per_sec=max_per_sec)" in src
    # The only URL constants are sec.gov.
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            v = node.value
            if re.match(r"^https?://\S", v):
                assert "sec.gov" in v, "non-sec.gov URL constant: %s" % v


def test_the_tool_forms_no_return_or_abnormal_return():
    """The no-look guard, as a static property of the tool's own source."""
    src = TOOL.read_text(encoding="utf-8").lower()
    for token in ("abnormal_return", "car_", "log_return", "pct_change",
                  "cumret", "excess_return"):
        assert token not in src, "price-statistic token in the tool: %s" % token
