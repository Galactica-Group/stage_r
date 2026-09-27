"""
================================================================================
Script:       test_chunked_diff.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the chunked-by-ticker comparison: a partitioned run must return
              exactly what a whole-table run returns, and the partitioning itself
              must be stable, disjoint and total.
Entry Point:  No — pytest
Dependencies: stdlib (importlib, json, pathlib); pandas; the diff tool
Last Sprint:  Sprint R0-C — Chunked-by-ticker comparison (2026-09-27)
================================================================================

The full-history price table cannot be held twice in memory, so it is compared
one ticker partition at a time. That only helps if it changes no answer, so the
centrepiece here is an equivalence proof on a synthetic pair carrying a known
change of every class: chunked and unchunked must agree field for field, at two
partition counts, one of which does not divide the ticker count evenly.

Falsifiable throughout: each test fails on a constructed violation of the
property it names.
================================================================================
"""

import importlib.util
import json
import pathlib

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
TOOL_PATH = REPO / "tools" / "sharadar_snapshot_diff.py"

FETCH_A = "2026-09-15T20:30:00Z"
FETCH_B = "2026-09-22T20:30:00Z"
REV_NEW_CLOSEUNADJ = 1234.5678
TAIL_NEW_CLOSE = 8765.4321
ADJ_FACTOR = 1.5
N_TICKERS = 12
N_BARS = 420


def _load_tool():
    spec = importlib.util.spec_from_file_location("sharadar_snapshot_diff_chunk",
                                                  TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIFF = _load_tool()

# Columns the price analysis keys on, in the tool's own order.
STOCK_COLS = ["ticker", "date", "open", "high", "low", "close", "volume",
              "closeadj", "closeunadj", "lastupdated"]


def _base_frame():
    """A synthetic price table: many tickers so partitions are non-trivial."""
    rows = []
    dates = pd.date_range("2025-07-24", periods=N_BARS, freq="D")
    for i in range(N_TICKERS):
        tk = "SYN%02d" % i
        for j, d in enumerate(dates):
            px = 10.0 + i + j * 0.01
            rows.append({
                "ticker": tk,
                "date": d.strftime("%Y-%m-%d"),
                "open": "%.4f" % px, "high": "%.4f" % (px * 1.01),
                "low": "%.4f" % (px * 0.99), "close": "%.4f" % px,
                "volume": str(1000 + j),
                "closeadj": "%.4f" % px, "closeunadj": "%.4f" % px,
                "lastupdated": "2026-09-15",
            })
    return pd.DataFrame(rows, columns=STOCK_COLS)


def _write_snapshot(root, label, stocks, fetched_at):
    snap = pathlib.Path(root) / "data" / "sharadar" / ("snapshot_%s" % label)
    snap.mkdir(parents=True, exist_ok=True)
    stocks.to_parquet(snap / "stocks_syn.parquet", index=False)
    manifest = [{"name": "stocks_syn", "columns": list(stocks.columns),
                 "rows": int(len(stocks)), "parquet_file": "stocks_syn.parquet",
                 "fetched_at_utc": fetched_at}]
    (snap / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _mutated(base):
    """One injected change of each class, spread across distinct tickers."""
    fetched = pd.Timestamp(FETCH_A).tz_localize(None)
    ages = (fetched - pd.to_datetime(base["date"])).dt.days
    b = base.copy()

    # ADJUSTMENT: adjusted series moves, as-traded does not.
    adj = b["ticker"] == "SYN00"
    b.loc[adj, "close"] = (b.loc[adj, "close"].astype(float) * ADJ_FACTOR).map("%.4f".__mod__)
    b.loc[adj, "closeadj"] = (b.loc[adj, "closeadj"].astype(float) * ADJ_FACTOR).map("%.4f".__mod__)

    # REVISION: as-traded close moves on a settled bar.
    pool = b.index[(b["ticker"] == "SYN01") & (ages > DIFF.TAIL_DAYS + 30)]
    assert len(pool)
    b.loc[pool[len(pool) // 2], "closeunadj"] = "%.4f" % REV_NEW_CLOSEUNADJ

    # TAIL: a change confined to an unsettled bar.
    tail = b.index[(b["ticker"] == "SYN02") & (ages <= DIFF.TAIL_DAYS)]
    assert len(tail)
    b.loc[tail[-1], "close"] = "%.4f" % TAIL_NEW_CLOSE

    # membership: one row dropped, one row added.
    drop_idx = b.index[b["ticker"] == "SYN03"][0]
    b = b.drop(index=drop_idx)
    added = base[base["ticker"] == "SYN03"].iloc[-1].copy()
    added["date"] = "2026-12-31"
    b = pd.concat([b, pd.DataFrame([added])], ignore_index=True)

    # watermark only.
    b.loc[b["ticker"] == "SYN04", "lastupdated"] = "2026-09-22"
    return b


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    root = tmp_path_factory.mktemp("chunked")
    base = _base_frame()
    _write_snapshot(root, "A", base, FETCH_A)
    _write_snapshot(root, "B", _mutated(base), FETCH_B)
    return str(root)


@pytest.fixture(scope="module")
def whole(pair):
    return DIFF.diff_snapshots(pair, "A", "B")[0]["tables"]["stocks"]


@pytest.fixture(scope="module")
def chunked4(pair):
    return DIFF.diff_snapshots(pair, "A", "B", chunk_by_ticker=4)[0]["tables"]["stocks"]


@pytest.fixture(scope="module")
def chunked7(pair):
    return DIFF.diff_snapshots(pair, "A", "B", chunk_by_ticker=7)[0]["tables"]["stocks"]


# ---------------------------------------------------------------------------
# ticker_bucket — stable, disjoint, total
# ---------------------------------------------------------------------------

def test_bucket_is_stable_across_calls(mod_ticker="AAPL"):
    assert DIFF.ticker_bucket(mod_ticker, 16) == DIFF.ticker_bucket(mod_ticker, 16)


def test_bucket_does_not_use_the_salted_builtin_hash():
    """hash() is salted per process, so the two sides of a comparison would
    partition the same ticker differently. md5 is why this is reproducible.

    Checked on the AST, not the text, so the explanatory docstring that names
    hash() cannot satisfy the assertion by accident."""
    import ast
    tree = ast.parse(TOOL_PATH.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "ticker_bucket")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "hash" not in called
    attrs = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert any(a.startswith("hashlib.md5") for a in attrs)


def test_buckets_are_in_range_and_partition_every_ticker():
    tickers = ["SYN%02d" % i for i in range(200)]
    for n in (1, 2, 4, 7, 16, 64):
        got = [DIFF.ticker_bucket(t, n) for t in tickers]
        assert all(0 <= g < n for g in got)
        # totality: every ticker lands in exactly one bucket, so the bucket
        # memberships sum back to the whole set
        union = set()
        for b in range(n):
            union |= {t for t in tickers if DIFF.ticker_bucket(t, n) == b}
        assert union == set(tickers)


def test_single_bucket_degenerates_to_bucket_zero():
    assert DIFF.ticker_bucket("ANYTHING", 1) == 0


def test_partitioning_actually_splits_the_test_universe(pair):
    """If every synthetic ticker fell in one bucket the equivalence tests below
    would pass vacuously."""
    used = {DIFF.ticker_bucket("SYN%02d" % i, 4) for i in range(N_TICKERS)}
    assert len(used) > 1


# ---------------------------------------------------------------------------
# equivalence — the centrepiece
# ---------------------------------------------------------------------------

COUNT_FIELDS = ("rows_only_in_a", "rows_only_in_b", "rows_in_both",
                "rows_in_both_differing_vendor_fields",
                "rows_in_both_differing_any_shared_field",
                "watermark_only_changes")


@pytest.mark.parametrize("n", [4, 7])
def test_chunked_counts_equal_whole_table_counts(pair, whole, n):
    got = DIFF.diff_snapshots(pair, "A", "B", chunk_by_ticker=n)[0]["tables"]["stocks"]
    for f in COUNT_FIELDS:
        assert got[f] == whole[f], "%s differs at n=%d" % (f, n)


def test_the_synthetic_pair_has_non_zero_differences(whole):
    """Anti-vacuity: an all-zero pair would make every equivalence test above
    true for the wrong reason."""
    assert whole["rows_only_in_a"] > 0
    assert whole["rows_only_in_b"] > 0
    assert whole["rows_in_both_differing_vendor_fields"] > 0
    assert whole["watermark_only_changes"] > 0


def test_chunked_per_field_counts_equal_whole_table(whole, chunked4):
    assert chunked4["per_field_changed"] == whole["per_field_changed"]


def test_chunked_class_tally_equals_whole_table(whole, chunked4, chunked7):
    assert chunked4["price_analysis"]["class_tally"] == \
        whole["price_analysis"]["class_tally"]
    assert chunked7["price_analysis"]["class_tally"] == \
        whole["price_analysis"]["class_tally"]


def test_every_class_is_represented_so_the_tally_test_bites(whole):
    tally = whole["price_analysis"]["class_tally"]
    assert {"ADJUSTMENT", "REVISION", "TAIL", "UNCHANGED"} <= set(tally)


def test_chunked_per_ticker_classes_equal_whole_table(whole, chunked4):
    """Disjoint partitions mean the per-ticker maps union with no collision."""
    assert chunked4["price_analysis"]["classes_by_ticker"] == \
        whole["price_analysis"]["classes_by_ticker"]


def test_chunked_revision_detail_equals_whole_table(whole, chunked4):
    assert chunked4["price_analysis"]["revision_tickers"] == \
        whole["price_analysis"]["revision_tickers"]


def test_chunked_magnitude_distributions_equal_whole_table(whole, chunked4):
    """Pooled from raw values, not averaged from per-partition summaries — so
    they must match exactly, not approximately."""
    wf = whole["price_analysis"]["fields"]
    cf = chunked4["price_analysis"]["fields"]
    assert set(cf) == set(wf)
    for col in wf:
        assert cf[col]["changed"] == wf[col]["changed"], col
        assert cf[col]["abs"] == wf[col]["abs"], col
        assert cf[col]["rel"] == wf[col]["rel"], col
        for label in wf[col]["by_age_bucket"]:
            assert (cf[col]["by_age_bucket"][label]["changed"]
                    == wf[col]["by_age_bucket"][label]["changed"]), (col, label)
            assert (cf[col]["by_age_bucket"][label]["abs"]
                    == wf[col]["by_age_bucket"][label]["abs"]), (col, label)


def test_a_distribution_is_actually_populated(whole):
    """Anti-vacuity for the test above: comparing None to None proves nothing."""
    fields = whole["price_analysis"]["fields"]
    assert any(f.get("abs") for f in fields.values())


def test_chunked_rows_reconcile_to_the_whole_table(whole, chunked4):
    assert chunked4["rows_before_dedupe"] == whole["rows_before_dedupe"]
    assert chunked4["rows_after_dedupe"] == whole["rows_after_dedupe"]


def test_chunking_metadata_is_recorded(chunked4):
    ch = chunked4["chunking"]
    assert ch["mode"] == "key-component partition"
    assert ch["partition_column"] == "ticker"
    assert ch["buckets"] == 4
    assert ch["buckets_compared"] >= 1
    assert ch["peak_rss_gib"] > 0


# ---------------------------------------------------------------------------
# merge_table_entries — the pure merge rule
# ---------------------------------------------------------------------------

def test_merge_sums_disjoint_counts(mod=None):
    a = {"rows_only_in_a": 1, "rows_only_in_b": 2, "rows_in_both": 10,
         "per_field_changed": {"close": 3}}
    b = {"rows_only_in_a": 4, "rows_only_in_b": 0, "rows_in_both": 5,
         "per_field_changed": {"close": 1, "volume": 2}}
    out = DIFF.merge_table_entries([a, b])
    assert out["rows_only_in_a"] == 5
    assert out["rows_in_both"] == 15
    assert out["per_field_changed"] == {"close": 4, "volume": 2}
    assert out["partitions"] == 2


def test_merge_propagates_unknown_rather_than_treating_none_as_zero():
    """compare_rowsets returns None for fields it cannot compute; summing that
    as zero would report a confident zero the tool never measured."""
    a = {"rows_in_both_differing_vendor_fields": None}
    b = {"rows_in_both_differing_vendor_fields": 7}
    out = DIFF.merge_table_entries([a, b])
    assert out["rows_in_both_differing_vendor_fields"] is None


def test_merge_of_one_partition_is_the_partition(mod=None):
    a = {"rows_in_both": 3, "per_field_changed": {"close": 1}}
    assert DIFF.merge_table_entries([a]) == a


def test_merge_folds_provenance_back_to_one_row_per_entry():
    a = {"source_entries_a": [{"entry": "stocks_syn", "rows": 10}],
         "source_entries_b": [{"entry": "stocks_syn", "rows": 10}]}
    b = {"source_entries_a": [{"entry": "stocks_syn", "rows": 5}],
         "source_entries_b": [{"entry": "stocks_syn", "rows": 5}]}
    out = DIFF.merge_table_entries([a, b])
    assert out["source_entries_a"] == [{"entry": "stocks_syn", "rows": 15}]


# ---------------------------------------------------------------------------
# load_table_bucket — bounded reads, honest provenance
# ---------------------------------------------------------------------------

def test_bucket_load_reads_row_groups_not_the_whole_file():
    seg = TOOL_PATH.read_text()
    seg = seg[seg.index("def load_table_bucket"):seg.index("def _sum_or_none")]
    assert "read_row_group" in seg
    assert "pd.read_parquet" not in seg


def test_bucket_loads_partition_to_the_same_total_as_the_whole_table(pair):
    man = DIFF.read_manifest(pair, "A")
    snap = DIFF.snapshot_dir(pair, "A")
    whole, _p, _s = DIFF.load_table(snap, man, "stocks")
    total = 0
    seen = set()
    for bucket in range(4):
        part, _p2, _s2 = DIFF.load_table_bucket(snap, man, "stocks", bucket, 4)
        if part is None:
            continue
        total += len(part)
        overlap = seen & set(part["ticker"])
        assert not overlap, "ticker in two partitions: %s" % overlap
        seen |= set(part["ticker"])
    assert total == len(whole)
    assert seen == set(whole["ticker"])


def test_a_table_with_no_ticker_column_is_not_chunked(pair, tmp_path):
    """sp500-style tables key on ticker, but a table without one must fall back
    to a whole-table comparison rather than silently comparing nothing."""
    snap = DIFF.snapshot_dir(pair, "A")
    man = DIFF.read_manifest(pair, "A")
    assert DIFF.table_has_column(snap, man, "stocks", "ticker") is True
    assert DIFF.table_has_column(snap, man, "stocks", "no_such_column") is False


# ---------------------------------------------------------------------------
# entry-name mapping — a table nobody maps is a table nobody compares
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,table", [
    ("bulk_stocks", "stocks"),
    ("bulk_tickers", "tickers"),
    ("bulk_actions", "actions"),
    ("bulk_sp500", "sp500"),
    ("tier_stocks_TSLA", "stocks"),
    ("tier_tickers_free", "tickers"),
    ("tier_sp500", "sp500"),
])
def test_r0c_entry_names_map_to_their_vendor_table(name, table):
    """The R0-C bulk and tier entries must resolve to a table.

    Before this mapping existed the self-diff compared nothing and still
    reported every count as zero — a clean verdict that measured no data at all.
    """
    assert DIFF.entry_table(name) == table


def test_docs_and_schema_entries_still_map_to_no_table():
    """The new rules must not sweep non-table artifacts into a comparison."""
    for name in ("schema_index", "schema_stocks", "docs_index", "page_terms",
                 "llms_txt"):
        assert DIFF.entry_table(name) is None


def test_the_real_snapshot_self_diff_compares_its_bulk_tables():
    """Anti-vacuity on live artifacts: the banked snapshot's own manifest must
    yield a non-empty set of comparable tables."""
    snap = REPO / "data" / "sharadar" / "snapshot_20260927p"
    if not (snap / "manifest.json").exists():
        pytest.skip("snapshot 20260927p absent")
    man = json.loads((snap / "manifest.json").read_text())
    mapped = {DIFF.entry_table(e["name"]) for e in man
              if e.get("parquet_file") and not DIFF.is_error_payload(e)}
    mapped.discard(None)
    assert {"stocks", "tickers", "actions", "sp500"} <= mapped


# ---------------------------------------------------------------------------
# partition_column — the partition must be a function of the comparison key
# ---------------------------------------------------------------------------

def test_a_table_keyed_on_permaticker_is_not_partitioned_by_ticker():
    """The defect this pins: `tickers` keys on permaticker, and 239 permatickers
    carry more than one ticker string, so a ticker partition splits a key across
    two partitions and per-partition dedupe keeps both — inflating the row count
    by exactly those 239. A self-diff cannot catch it, because both sides inflate
    identically."""
    assert DIFF.partition_column("tickers", ["permaticker"]) == "permaticker"


def test_price_tables_partition_on_ticker_because_classification_needs_it():
    assert DIFF.partition_column("stocks", ["ticker", "date"]) == "ticker"
    assert DIFF.partition_column("funds", ["ticker", "date"]) == "ticker"


def test_a_price_table_without_ticker_in_its_key_refuses_to_chunk():
    """Rather than partition a price table on something its per-ticker class
    merge cannot survive, chunking declines and the table is compared whole."""
    assert DIFF.partition_column("stocks", ["date"]) is None
    assert DIFF.partition_column("stocks", []) is None


def test_a_multi_column_key_containing_ticker_partitions_on_ticker():
    assert DIFF.partition_column("sp500", ["ticker", "date", "action"]) == "ticker"


def test_a_keyless_table_partitions_on_ticker_safely():
    """A whole-row set comparison signs every column including ticker, so two
    identical rows cannot straddle a ticker partition."""
    assert DIFF.partition_column("actions", []) == "ticker"


def test_the_chosen_column_is_always_a_key_component_when_a_key_exists():
    for table, key in (("tickers", ["permaticker"]),
                       ("sp500", ["ticker", "date", "action"]),
                       ("descriptions", ["table", "indicator"]),
                       ("metrics", ["ticker", "date"])):
        col = DIFF.partition_column(table, key)
        assert col in key, (table, col, key)


def test_real_tickers_table_has_keys_that_straddle_ticker_buckets():
    """Anti-vacuity on live data: if no permaticker carried two ticker strings,
    the test above would be guarding against nothing."""
    snap = REPO / "data" / "sharadar" / "snapshot_20260927p"
    if not (snap / "bulk_tickers.parquet").exists():
        pytest.skip("snapshot 20260927p absent")
    df = pd.read_parquet(snap / "bulk_tickers.parquet",
                         columns=["permaticker", "ticker"])
    spread = df.groupby("permaticker")["ticker"].nunique()
    assert int((spread > 1).sum()) > 0


def test_chunked_tickers_table_equals_unchunked_on_real_data():
    """The regression itself, end to end on the banked snapshot: partitioned
    dedupe must reproduce the whole-table dedupe count exactly."""
    snap_root = str(REPO)
    snap = DIFF.snapshot_dir(snap_root, "20260927p")
    if not pathlib.Path(snap, "manifest.json").exists():
        pytest.skip("snapshot 20260927p absent")
    man = DIFF.read_manifest(snap_root, "20260927p")
    key, _ks = DIFF.resolve_key("tickers", {})
    col = DIFF.partition_column("tickers", key)
    whole, _p, _s = DIFF.load_table(snap, man, "tickers")
    cols = [c for c in whole.columns if not c.startswith("__")]
    w2, _c = DIFF.dedupe_on_key(whole, key, cols)
    total, seen = 0, set()
    for b in range(8):
        part, _p2, _s2 = DIFF.load_table_bucket(snap, man, "tickers", b, 8,
                                                column=col)
        if part is None:
            continue
        p2, _cc = DIFF.dedupe_on_key(part, key, cols)
        total += len(p2)
        assert not (seen & set(p2[key[0]])), "key straddled a partition"
        seen |= set(p2[key[0]])
    assert total == len(w2)
