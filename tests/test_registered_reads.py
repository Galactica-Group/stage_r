"""
================================================================================
Script:       test_registered_reads.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the Sprint R0-D registered-read emissions against a synthetic
              snapshot pair carrying one known injection per read.
Entry Point:  No — pytest
Dependencies: tools/sharadar_snapshot_diff.py, pandas
Last Sprint:  Sprint R0-D — Snapshot 20261004p stability diff (2026-10-04)
================================================================================

Every number asserted here is one a wrong implementation would get wrong: the
fixture injects a counted, dated change of each class, so each assertion fails
if the emission it covers is dropped, mis-split or mis-dated. The two revision
rules are discriminated by one row (DELTA, a settled `close`-only move) that the
tool's own rule calls an ADJUSTMENT and the brief's wider gloss calls a REVISION.
================================================================================
"""

import importlib.util
import json
import os
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "diff_tool", ROOT / "tools" / "sharadar_snapshot_diff.py")
DIFF = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DIFF)

FETCH_A = "2026-09-27T07:00:00Z"
FETCH_B = "2026-10-04T10:52:00Z"
ONLYB_CUTOFF = "2026-09-25"
REVISION_CUTOFF = "2026-09-01"
ACT_LO, ACT_HI = "2026-09-25", "2026-10-02"

OLD_BAR = "2020-05-05"      # settled, before the revision cutoff
SETTLED = "2026-09-10"      # settled, after the revision cutoff
TAIL_BAR = "2026-09-24"     # inside the 7-day tail window at fetch A

# Injected values are deliberately unround: a magnitude distribution legitimately
# publishes ratios and differences, so a round decoy like 0.5 would collide with
# a sanctioned statistic and make the licence test fire on nothing.
REV_CLOSEUNADJ = 7919.3701
REV_CLOSE = 8831.4129
REV_VOLUME = 987654321
DIV_BETA = 0.487113
DIV_ALPHA = 0.613397

PRICE_TICKERS = ["ALPHA", "BETA", "GAMMA", "DELTA", "EPS", "ZETA",
                 "OLDROW", "EXIST"]
READS = {"onlyb_cutoff_date": ONLYB_CUTOFF,
         "revision_cutoff_date": REVISION_CUTOFF,
         "action_window_start": ACT_LO,
         "action_window_end": ACT_HI}


def _bar(ticker, date, c=100.0, vol=1000, wm="2026-09-26"):
    return {"ticker": ticker, "date": date, "open": c - 1, "high": c + 1,
            "low": c - 2, "close": c, "volume": vol, "closeadj": c * 0.9,
            "closeunadj": c, "lastupdated": wm}


def _stocks_a():
    rows = []
    for t in PRICE_TICKERS:
        for d in (OLD_BAR, SETTLED, TAIL_BAR):
            rows.append(_bar(t, d))
    return pd.DataFrame(rows)


def _stocks_b(a):
    b = a.copy()

    def at(ticker, date):
        m = (b["ticker"] == ticker) & (b["date"] == date)
        assert m.sum() == 1, (ticker, date)
        return b.index[m][0]

    # ADJUSTMENT, explained by a dividend inside the window.
    i = at("ALPHA", SETTLED)
    b.loc[i, ["closeadj", "lastupdated"]] = [b.loc[i, "closeadj"] * 1.5, "2026-10-03"]
    # ADJUSTMENT, with no action inside the window: the unexplained case.
    i = at("BETA", SETTLED)
    b.loc[i, ["closeadj", "lastupdated"]] = [b.loc[i, "closeadj"] * 1.5, "2026-10-03"]
    # REVISION on a bar dated before the revision cutoff.
    i = at("GAMMA", OLD_BAR)
    b.loc[i, ["closeunadj", "lastupdated"]] = [REV_CLOSEUNADJ, "2026-10-03"]
    # Settled `close`-only move: the rule discriminator, explained by a split.
    i = at("DELTA", SETTLED)
    b.loc[i, ["close", "lastupdated"]] = [REV_CLOSE, "2026-10-03"]
    # Value moved and the watermark did NOT advance: R4's first-order finding.
    i = at("EPS", SETTLED)
    b.loc[i, "volume"] = REV_VOLUME
    # Watermark advanced with no value change: benign.
    b.loc[at("ZETA", SETTLED), "lastupdated"] = "2026-10-03"

    # One historical row disappears (R1).
    b = b.drop(index=at("OLDROW", OLD_BAR)).reset_index(drop=True)

    added = [
        _bar("EXIST", "2019-03-04"),        # backfill onto an existing name
        # Dated exactly ON the cutoff: R2 reads "after 2026-09-25", so this is a
        # BACKFILL. It is the only row that distinguishes > from >=.
        _bar("EXIST", ONLYB_CUTOFF),
        _bar("NEWCO", "2018-07-02"),        # backfill onto a name new to coverage
        _bar("NEWCO", "2026-09-30"),        # new session, new name
        _bar("ALPHA", "2026-09-28"),        # new session
        _bar("ALPHA", "2026-10-01"),        # new session
    ]
    return pd.concat([b, pd.DataFrame(added)], ignore_index=True)


def _actions_a():
    return pd.DataFrame([
        # A dividend on BETA five days BEFORE the window opens: it must not
        # explain BETA's adjustment, which is what makes the bound load-bearing.
        {"date": "2026-09-20", "action": "dividend", "ticker": "BETA",
         "name": "Beta Inc", "value": DIV_BETA},
        {"date": "2026-09-26", "action": "announcement", "ticker": "OMEGA",
         "name": "Omega Inc", "value": None},
        {"date": "2019-01-07", "action": "split", "ticker": "EXIST",
         "name": "Exist Inc", "value": 2.0},
    ])


def _actions_b(a):
    # The OMEGA announcement is withdrawn; three actions land inside the window.
    b = a[a["ticker"] != "OMEGA"].copy()
    return pd.concat([b, pd.DataFrame([
        {"date": "2026-09-28", "action": "dividend", "ticker": "ALPHA",
         "name": "Alpha Inc", "value": DIV_ALPHA},
        {"date": "2026-09-29", "action": "split", "ticker": "DELTA",
         "name": "Delta Inc", "value": 2.0},
        # In the window but on an UNCHANGED ticker: counted in the window, never
        # counted as an explanation.
        {"date": "2026-09-30", "action": "dividend", "ticker": "ZETA",
         "name": "Zeta Inc", "value": 0.1},
    ])], ignore_index=True)


def _tickers_a():
    rows = []
    for n, t in enumerate(PRICE_TICKERS, start=1):
        rows.append({"permaticker": 1000 + n, "ticker": t, "name": "%s Inc" % t,
                     "isdelisted": "Y" if t == "OLDROW" else "N",
                     "exchange": "NYSE", "lastpricedate": "2026-09-25",
                     "lastupdated": "2026-09-26"})
    return pd.DataFrame(rows)


def _tickers_b(a):
    b = a.copy()
    live = b.index[b["ticker"].isin(["ALPHA", "BETA"])]
    b.loc[live, "lastpricedate"] = "2026-10-02"
    b.loc[live, "lastupdated"] = "2026-10-03"
    b = pd.concat([b, pd.DataFrame([
        {"permaticker": 9999, "ticker": "NEWCO", "name": "Newco Inc",
         "isdelisted": "N", "exchange": "NASDAQ",
         "lastpricedate": "2026-10-02", "lastupdated": "2026-10-03"}])],
        ignore_index=True)
    return b


def _sp500_a():
    return pd.DataFrame([
        {"date": "2024-01-02", "action": "added", "ticker": "ALPHA",
         "name": "Alpha Inc", "contraticker": None, "contraname": None,
         "note": None},
        {"date": "2024-01-02", "action": "removed", "ticker": "OLDROW",
         "name": "Oldrow Inc", "contraticker": None, "contraname": None,
         "note": None},
    ])


def _sp500_b(a):
    return pd.concat([a, pd.DataFrame([
        {"date": "2026-09-29", "action": "added", "ticker": "NEWCO",
         "name": "Newco Inc", "contraticker": None, "contraname": None,
         "note": None},
        {"date": "2026-09-29", "action": "removed", "ticker": "ZETA",
         "name": "Zeta Inc", "contraticker": None, "contraname": None,
         "note": None},
    ])], ignore_index=True)


def _write(root, label, frames, fetched_at):
    snap = pathlib.Path(root) / "data" / "sharadar" / ("snapshot_%s" % label)
    snap.mkdir(parents=True, exist_ok=True)
    manifest = []
    for table, df in frames.items():
        fn = "bulk_%s.parquet" % table
        df.to_parquet(snap / fn, index=False)
        manifest.append({"name": "bulk_%s" % table, "columns": list(df.columns),
                         "rows": int(len(df)), "parquet_file": fn,
                         "fetched_at_utc": fetched_at})
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    sa, aa, ta, pa = _stocks_a(), _actions_a(), _tickers_a(), _sp500_a()
    frames_a = {"stocks": sa, "actions": aa, "tickers": ta, "sp500": pa}
    frames_b = {"stocks": _stocks_b(sa), "actions": _actions_b(aa),
                "tickers": _tickers_b(ta), "sp500": _sp500_b(pa)}
    root = tmp_path_factory.mktemp("r0d")
    _write(root, "A", frames_a, FETCH_A)
    _write(root, "B", frames_b, FETCH_B)
    plain, jp, mp, xp = DIFF.run(str(root), "A", "B", reads=READS)
    chunked, _j, _m, _x = DIFF.run(str(root), "A", "B", chunk_by_ticker=4,
                                   reads=READS)
    return {"root": str(root), "report": plain, "chunked": chunked,
            "json_path": jp, "md_path": mp}


def _rd(pair, table, block, chunked=False):
    rep = pair["chunked"] if chunked else pair["report"]
    return rep["tables"][table]["registered_reads"][block]


# --- R1 / R2: membership -----------------------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r1_only_a_rows_are_counted_by_their_year(pair, chunked):
    m = _rd(pair, "stocks", "membership", chunked)
    assert m["only_in_a_rows"] == 1
    assert m["only_in_a_by_year"] == {"2020": 1}


@pytest.mark.parametrize("chunked", [False, True])
def test_r2_new_sessions_are_split_from_backfill(pair, chunked):
    m = _rd(pair, "stocks", "membership", chunked)
    assert m["only_in_b_rows"] == 6
    assert m["after_cutoff_rows"] == 3
    assert m["backfill_rows"] == 3
    assert m["cutoff_date"] == ONLYB_CUTOFF


@pytest.mark.parametrize("chunked", [False, True])
def test_r2_new_sessions_are_dated(pair, chunked):
    m = _rd(pair, "stocks", "membership", chunked)
    assert m["after_cutoff_by_date"] == {"2026-09-28": 1, "2026-09-30": 1,
                                         "2026-10-01": 1}


@pytest.mark.parametrize("chunked", [False, True])
def test_r2_backfill_splits_new_listing_from_existing_name(pair, chunked):
    m = _rd(pair, "stocks", "membership", chunked)
    assert m["backfill_by_year"] == {"2018": 1, "2019": 1, "2026": 1}
    assert m["backfill_new_listing_rows"] == 1
    assert m["backfill_existing_name_rows"] == 2
    assert m["backfill_new_listing_tickers"] == 1
    assert m["backfill_existing_name_tickers"] == 1


# --- R3: revision rows, both rules ------------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r2_a_row_dated_on_the_cutoff_is_a_backfill_not_a_new_session(pair, chunked):
    """R2 reads "after 2026-09-25", so the cutoff day itself is already served."""
    m = _rd(pair, "stocks", "membership", chunked)
    assert ONLYB_CUTOFF not in m["after_cutoff_by_date"]
    assert m["backfill_by_year"].get("2026") == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_r3_value_changed_row_count_excludes_the_watermark(pair, chunked):
    r = _rd(pair, "stocks", "revision", chunked)
    assert r["value_changed_rows"] == 5
    assert DIFF.WATERMARK_FIELD not in r["fields_compared"]


@pytest.mark.parametrize("chunked", [False, True])
def test_r3_tool_rule_counts_only_closeunadj_and_volume_as_revisions(pair, chunked):
    r = _rd(pair, "stocks", "revision", chunked)["rules"]["rule_tool"]
    assert r["revision_rows"] == 2
    assert r["adjustment_only_rows"] == 3
    assert r["revision_rows_before_cutoff"] == 1
    assert r["revision_before_cutoff_by_year"] == {"2020": 1}
    assert r["revision_rows_on_or_after_cutoff"] == 1
    assert r["revision_on_or_after_by_date"] == {SETTLED: 1}


@pytest.mark.parametrize("chunked", [False, True])
def test_r3_brief_gloss_rule_additionally_counts_a_settled_close_move(pair, chunked):
    r = _rd(pair, "stocks", "revision", chunked)["rules"]["rule_brief_gloss"]
    assert r["revision_rows"] == 3
    assert r["adjustment_only_rows"] == 2
    assert r["revision_rows_before_cutoff"] == 1
    assert r["revision_rows_on_or_after_cutoff"] == 2


def test_r3_the_two_rules_are_not_the_same_rule(pair):
    rules = _rd(pair, "stocks", "revision")["rules"]
    assert (rules["rule_tool"]["revision_rows"]
            != rules["rule_brief_gloss"]["revision_rows"])


# --- R3: adjustment explanation ---------------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r3_adjustment_tickers_are_explained_by_an_action_in_the_window(pair, chunked):
    x = _rd(pair, "stocks", "adjustment_explanation", chunked)
    assert x["adjustment_tickers"] == 3
    assert x["explained"] == 2
    assert x["explained_by_type"] == {"dividend": 1, "split": 1}
    assert x["window"] == [ACT_LO, ACT_HI]


@pytest.mark.parametrize("chunked", [False, True])
def test_r3_an_action_outside_the_window_does_not_explain(pair, chunked):
    x = _rd(pair, "stocks", "adjustment_explanation", chunked)
    assert x["unexplained"] == 1
    assert x["unexplained_tickers"] == ["BETA"]


def test_r3_window_actions_on_unchanged_tickers_never_count_as_explanations(pair):
    x = _rd(pair, "stocks", "adjustment_explanation")
    assert x["actions_in_window"] == 3
    assert x["actions_in_window_on_adjustment_tickers"] == 2


# --- R4: watermark integrity -------------------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r4_a_value_change_without_watermark_advance_is_counted(pair, chunked):
    w = _rd(pair, "stocks", "watermark", chunked)
    assert w["value_changed_rows"] == 5
    assert w["value_changed_with_watermark_advance"] == 4
    assert w["value_changed_without_watermark_advance"] == 1


@pytest.mark.parametrize("chunked", [False, True])
def test_r4_watermark_movement_without_a_value_change_is_counted_separately(pair, chunked):
    w = _rd(pair, "stocks", "watermark", chunked)
    assert w["watermark_moved_without_value_change"] == 1
    assert w["watermark_advanced_without_value_change"] == 1


# --- R5: tickers, delisted vs active ----------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r5_a_delisted_record_is_shown_unchanged(pair, chunked):
    d = _rd(pair, "tickers", "delisted_split", chunked)
    assert d["delisted_rows"] == 1
    assert d["delisted_rows_changed_any_field"] == 0
    assert d["delisted_changed_by_field"] == {}


@pytest.mark.parametrize("chunked", [False, True])
def test_r5_active_record_changes_are_tallied_by_field(pair, chunked):
    d = _rd(pair, "tickers", "delisted_split", chunked)
    assert d["active_rows_changed_any_field"] == 2
    assert d["active_changed_by_field"] == {"lastpricedate": 2, "lastupdated": 2}


@pytest.mark.parametrize("chunked", [False, True])
def test_r5_a_new_listing_appears_only_on_the_b_side(pair, chunked):
    rep = pair["chunked"] if chunked else pair["report"]
    e = rep["tables"]["tickers"]
    assert e["rows_only_in_a"] == 0
    assert e["rows_only_in_b"] == 1


# --- R6: event tables by type ------------------------------------------------

@pytest.mark.parametrize("chunked", [False, True])
def test_r6_actions_membership_is_tallied_by_action_type(pair, chunked):
    t = _rd(pair, "actions", "by_type", chunked)
    assert t["only_in_b_by_type"] == {"dividend": 2, "split": 1}
    assert t["only_in_a_by_type"] == {"announcement": 1}


@pytest.mark.parametrize("chunked", [False, True])
def test_r6_sp500_adds_and_removes_are_separated(pair, chunked):
    t = _rd(pair, "sp500", "by_type", chunked)
    assert t["only_in_b_by_type"] == {"added": 1, "removed": 1}
    assert t["only_in_a_by_type"] == {}


# --- chunk-invariance and the licence rule ----------------------------------

def test_chunked_reads_equal_unchunked_reads(pair):
    for table in ("stocks", "tickers", "actions", "sp500"):
        a = pair["report"]["tables"][table].get("registered_reads")
        b = pair["chunked"]["tables"][table].get("registered_reads")
        assert a == b, table


def test_tracked_reports_carry_no_vendor_value(pair):
    """No injected price, volume or dividend amount reaches a tracked artifact."""
    blob = (pathlib.Path(pair["json_path"]).read_text()
            + pathlib.Path(pair["md_path"]).read_text())
    for forbidden in (REV_CLOSEUNADJ, REV_CLOSE, REV_VOLUME, DIV_BETA, DIV_ALPHA):
        assert str(forbidden) not in blob, forbidden
