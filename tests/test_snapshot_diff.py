"""
================================================================================
Script:       test_snapshot_diff.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Falsifiable tests for tools/sharadar_snapshot_diff.py: the self-diff
              null, a synthetic pair with injected changes of each class, and the
              licence property that tracked reports carry no vendor values.
Entry Point:  No — pytest
Dependencies: stdlib, pandas, pytest; imports the diff tool by path.
Last Sprint:  Sprint R0-B-0 — Sharadar Snapshot Diff Tool (2026-09-17)
================================================================================

Every assertion here names an exact count. A test that would pass on a tool that
did nothing is not a test: the synthetic pair injects one change of each class
and demands that class, that field and that age bucket back.
================================================================================
"""

import importlib.util
import json
import os
import pathlib
import shutil
import subprocess

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL_PATH = REPO / "tools" / "sharadar_snapshot_diff.py"
REAL_LABEL = "20260915"
REAL_SNAP = REPO / "data" / "sharadar" / ("snapshot_%s" % REAL_LABEL)

# Distinctive injected values: implausible as a derived statistic, so the licence
# test can assert their absence from the tracked reports without ambiguity.
REV_NEW_CLOSEUNADJ = 1234.5678
TAIL_NEW_CLOSE = 8765.4321
ADJ_FACTOR = 1.5
FETCH_A = "2026-09-15T20:30:00Z"
FETCH_B = "2026-09-22T20:30:00Z"


def _load_tool():
    spec = importlib.util.spec_from_file_location("sharadar_snapshot_diff", TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIFF = _load_tool()

pytestmark = pytest.mark.skipif(
    not REAL_SNAP.exists(), reason="snapshot %s absent" % REAL_LABEL)


# ----------------------------------------------------------------------------
# Self-diff: the tool's null.
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def self_diff(tmp_path_factory):
    root = tmp_path_factory.mktemp("selfdiff")
    copy_label = "20260915COPY"
    for label in (REAL_LABEL, copy_label):
        dest = root / "data" / "sharadar" / ("snapshot_%s" % label)
        dest.mkdir(parents=True)
        shutil.copy2(REAL_SNAP / "manifest.json", dest / "manifest.json")
        for pq in REAL_SNAP.glob("*.parquet"):
            shutil.copy2(pq, dest / pq.name)
    report, _json_path, _md_path, _ex = DIFF.run(str(root), REAL_LABEL, copy_label)
    return report


def test_self_diff_finds_no_membership_change(self_diff):
    """A snapshot against a copy of itself must move no row count anywhere."""
    assert self_diff["tables_only_in_a"] == []
    assert self_diff["tables_only_in_b"] == []
    compared = 0
    for table, entry in self_diff["tables"].items():
        if entry.get("status"):
            continue
        compared += 1
        assert entry["rows_only_in_a"] == 0, table
        assert entry["rows_only_in_b"] == 0, table
        assert entry["rows_in_both"] > 0, table
    assert compared == 14, "expected the 14 vendor tables, got %d" % compared


def test_self_diff_finds_no_field_change(self_diff):
    for table, entry in self_diff["tables"].items():
        if entry.get("status"):
            continue
        assert entry["rows_in_both_differing_vendor_fields"] == 0, table
        assert entry["rows_in_both_differing_any_shared_field"] == 0, table
        assert entry["watermark_only_changes"] == 0, table
        assert not [f for f, n in entry["per_field_changed"].items() if n], table


def test_self_diff_classifies_every_price_ticker_unchanged(self_diff):
    for table in DIFF.PRICE_TABLES:
        pa = self_diff["tables"][table]["price_analysis"]
        assert pa["tickers_compared"] > 0, table
        assert set(pa["class_tally"]) == {"UNCHANGED"}, (table, pa["class_tally"])
        assert pa["revision_tickers"] == {}, table
        for field, fd in pa["fields"].items():
            assert fd["changed"] == 0, (table, field)


def test_self_diff_writes_no_examples(self_diff):
    assert self_diff["examples_file"]["rows"] == 0


# ----------------------------------------------------------------------------
# Synthetic pair: one injected change of each class, built from a real slice.
# ----------------------------------------------------------------------------

def _write_snapshot(root, label, stocks, fetched_at):
    snap = pathlib.Path(root) / "data" / "sharadar" / ("snapshot_%s" % label)
    snap.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REAL_SNAP / "desc_all.parquet", snap / "desc_all.parquet")
    stocks.to_parquet(snap / "stocks_syn.parquet", index=False)
    desc = pd.read_parquet(snap / "desc_all.parquet")
    manifest = [
        {"name": "desc_all", "columns": list(desc.columns), "rows": int(len(desc)),
         "parquet_file": "desc_all.parquet", "fetched_at_utc": fetched_at},
        {"name": "stocks_syn", "columns": list(stocks.columns), "rows": int(len(stocks)),
         "parquet_file": "stocks_syn.parquet", "fetched_at_utc": fetched_at},
    ]
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    src = pd.read_parquet(REAL_SNAP / "stocks_5y_dow30.parquet")
    tickers = sorted(src["ticker"].unique())[:6]
    base = src[src["ticker"].isin(tickers)].copy().reset_index(drop=True)
    t_adj, t_rev, t_tail, t_row, t_wm, t_none = tickers

    fetched = pd.Timestamp(FETCH_A).tz_localize(None)
    ages = (fetched - pd.to_datetime(base["date"])).dt.days

    b = base.copy()
    injected = {}

    # (i) ADJUSTMENT: adjusted series moves, as-traded series does not.
    adj_mask = b["ticker"] == t_adj
    injected["adj_old_close"] = str(b.loc[adj_mask, "close"].iloc[0])
    b.loc[adj_mask, "close"] = b.loc[adj_mask, "close"] * ADJ_FACTOR
    b.loc[adj_mask, "closeadj"] = b.loc[adj_mask, "closeadj"] * ADJ_FACTOR
    n_adj_bars = int(adj_mask.sum())

    # (ii) REVISION: unadjusted close moves on one settled bar ~380 days old.
    rev_pool = b.index[(b["ticker"] == t_rev) & (ages > 366) & (ages < 396)]
    assert len(rev_pool), "no bar in the 366-396 day band for the revision injection"
    rev_idx = rev_pool[len(rev_pool) // 2]
    rev_age = int(ages.loc[rev_idx])
    injected["rev_old"] = str(b.loc[rev_idx, "closeunadj"])
    b.loc[rev_idx, "closeunadj"] = REV_NEW_CLOSEUNADJ

    # (iii) TAIL: a change confined to an unsettled bar.
    tail_pool = b.index[(b["ticker"] == t_tail) & (ages <= DIFF.TAIL_DAYS)]
    assert len(tail_pool), "no bar inside the tail window"
    tail_idx = tail_pool[-1]
    injected["tail_old"] = str(b.loc[tail_idx, "close"])
    b.loc[tail_idx, "close"] = TAIL_NEW_CLOSE

    # (iv) one row removed and one row added.
    row_idx = b.index[b["ticker"] == t_row][0]
    b = b.drop(index=row_idx)
    added = base[base["ticker"] == t_row].iloc[-1].copy()
    added["date"] = "2026-09-14"
    b = pd.concat([b, pd.DataFrame([added])], ignore_index=True)

    # (v) watermark only.
    wm_mask = b["ticker"] == t_wm
    n_wm_bars = int(wm_mask.sum())
    b.loc[wm_mask, "lastupdated"] = "2026-09-22"

    root = tmp_path_factory.mktemp("synthetic")
    _write_snapshot(root, "A", base, FETCH_A)
    _write_snapshot(root, "B", b, FETCH_B)
    report, json_path, md_path, ex_path = DIFF.run(str(root), "A", "B")
    return {
        "root": str(root), "report": report, "json_path": json_path,
        "md_path": md_path, "ex_path": ex_path, "injected": injected,
        "tickers": {"adj": t_adj, "rev": t_rev, "tail": t_tail,
                    "row": t_row, "wm": t_wm, "none": t_none},
        "n_adj_bars": n_adj_bars, "n_wm_bars": n_wm_bars, "rev_age": rev_age,
    }


def _stocks(synthetic):
    return synthetic["report"]["tables"]["stocks"]


def test_synthetic_row_membership_is_exact(synthetic):
    e = _stocks(synthetic)
    assert e["rows_only_in_a"] == 1
    assert e["rows_only_in_b"] == 1


def test_synthetic_every_ticker_gets_the_class_it_was_built_for(synthetic):
    """Each injection must produce its own class on its own ticker, and no other."""
    pa = _stocks(synthetic)["price_analysis"]
    cls = pa["classes_by_ticker"]
    tk = synthetic["tickers"]
    assert cls[tk["adj"]] == "ADJUSTMENT"
    assert cls[tk["rev"]] == "REVISION"
    assert cls[tk["tail"]] == "TAIL"
    assert cls[tk["row"]] == "UNCHANGED"
    assert cls[tk["wm"]] == "UNCHANGED"
    assert cls[tk["none"]] == "UNCHANGED"


def test_written_report_omits_the_per_ticker_class_map(synthetic):
    """classes_by_ticker is an in-memory convenience; it must not reach disk."""
    written = json.loads(pathlib.Path(synthetic["json_path"]).read_text())
    pa = written["tables"]["stocks"]["price_analysis"]
    assert "classes_by_ticker" not in pa
    assert pa["class_tally"] == _stocks(synthetic)["price_analysis"]["class_tally"]


def test_synthetic_class_tally_is_exact(synthetic):
    pa = _stocks(synthetic)["price_analysis"]
    assert pa["class_tally"] == {"ADJUSTMENT": 1, "REVISION": 1, "TAIL": 1, "UNCHANGED": 3}
    assert pa["tickers_compared"] == 6


def test_synthetic_revision_ticker_names_field_and_age_bucket(synthetic):
    pa = _stocks(synthetic)["price_analysis"]
    rev = pa["revision_tickers"]
    assert list(rev) == [synthetic["tickers"]["rev"]]
    detail = rev[synthetic["tickers"]["rev"]]
    assert detail["fields_on_settled_bars"] == ["closeunadj"]
    assert detail["bars_by_age_bucket"] == {">365": {"closeunadj": 1}}
    assert synthetic["rev_age"] > 365


def test_synthetic_per_field_counts_are_exact(synthetic):
    pa = _stocks(synthetic)["price_analysis"]
    n_adj = synthetic["n_adj_bars"]
    assert pa["fields"]["close"]["changed"] == n_adj + 1      # adjustment bars + the tail bar
    assert pa["fields"]["closeadj"]["changed"] == n_adj
    assert pa["fields"]["closeunadj"]["changed"] == 1
    assert pa["fields"]["volume"]["changed"] == 0
    for field in ("open", "high", "low"):
        assert pa["fields"][field]["changed"] == 0, field


def test_synthetic_tail_change_lands_in_the_tail_bucket(synthetic):
    pa = _stocks(synthetic)["price_analysis"]
    buckets = pa["fields"]["close"]["by_age_bucket"]
    assert buckets["<=7"]["changed"] >= 1
    assert pa["fields"]["closeunadj"]["by_age_bucket"][">365"]["changed"] == 1
    assert pa["fields"]["closeunadj"]["by_age_bucket"]["<=7"]["changed"] == 0
    assert pa["fields"]["closeunadj"]["by_age_bucket"]["31-365"]["changed"] == 0


def test_synthetic_watermark_moves_but_no_class_does(synthetic):
    e = _stocks(synthetic)
    assert e["per_field_changed"]["lastupdated"] == synthetic["n_wm_bars"]
    assert e["watermark_only_changes"] == synthetic["n_wm_bars"]
    pa = e["price_analysis"]
    # the watermark ticker stays UNCHANGED: lastupdated never enters classification
    assert pa["class_tally"].get("UNCHANGED") == 3
    assert synthetic["tickers"]["wm"] not in pa["revision_tickers"]


def test_synthetic_differing_rows_exclude_the_watermark(synthetic):
    e = _stocks(synthetic)
    n = synthetic["n_adj_bars"] + 2      # adjustment bars, the tail bar, the revised bar
    assert e["rows_in_both_differing_vendor_fields"] == n
    assert e["rows_in_both_differing_any_shared_field"] == n + synthetic["n_wm_bars"]


# ----------------------------------------------------------------------------
# Licence property: tracked reports carry statistics, never vendor values.
# ----------------------------------------------------------------------------

def test_tracked_reports_contain_no_injected_price_value(synthetic):
    text = (pathlib.Path(synthetic["json_path"]).read_text()
            + pathlib.Path(synthetic["md_path"]).read_text())
    forbidden = list(synthetic["injected"].values()) + [
        str(REV_NEW_CLOSEUNADJ), str(TAIL_NEW_CLOSE)]
    hits = [v for v in forbidden if v and v in text]
    assert not hits, "vendor value(s) leaked into a tracked report: %s" % hits


def test_examples_file_holds_the_values_and_is_capped(synthetic):
    df = pd.read_csv(synthetic["ex_path"])
    assert len(df) <= DIFF.EXAMPLE_ROW_CAP
    assert list(df.columns) == ["ticker", "date", "field", "old", "new"]
    assert synthetic["report"]["examples_file"]["rows"] == len(df)


def test_examples_file_path_is_gitignored():
    """The examples file must sit outside the tracked tree — proven by git.

    R0-C moved data/ onto the external volume behind a symlink, and git refuses
    to resolve a path that lies beyond one. The property is asserted at the
    symlink instead, which is strictly stronger: the whole data entry is
    ignored, so nothing beneath it can be tracked.
    """
    rel = "data/sharadar/diff_examples_20260915_vs_20260922.csv"
    root = rel.split("/")[0]
    rc = subprocess.run(["git", "check-ignore", "-q", root], cwd=str(REPO)).returncode
    assert rc == 0, "%s is not gitignored, so %s could be tracked" % (root, rel)
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=str(REPO),
                             capture_output=True, text=True, check=True).stdout
    under = [f for f in tracked.split("\0") if f.startswith(root + "/")]
    assert not under, "tracked file(s) live under %s/: %s" % (root, under[:5])


def test_reports_directory_is_tracked_not_ignored():
    rel = "reports/sharadar_stability_20260915_vs_20260922.json"
    rc = subprocess.run(["git", "check-ignore", "-q", rel], cwd=str(REPO)).returncode
    assert rc == 1, "%s is gitignored; the report would never be committed" % rel
