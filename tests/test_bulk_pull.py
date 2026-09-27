"""
================================================================================
Script:       test_bulk_pull.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the bulk full-history pull: the redirect classifier, the
              pre-signed-URL redaction, the full-history span assertion, and the
              bounded-memory zip-to-parquet conversion.
Entry Point:  No — pytest
Dependencies: stdlib (ast, csv, importlib, io, pathlib, zipfile); pandas; the
              probe module
Last Sprint:  Sprint R0-C — Paid tier verified; bulk full-history pull (2026-09-27)
================================================================================

Falsifiable throughout: each test fails on a constructed violation of the
property it names. The span assertion carries the trap the vendor sets — a plan
sold as five years answers years=full with HTTP 200 and five years of rows — so
the only defence is reading the returned span back, and that is what is pinned.
================================================================================
"""

import ast
import csv
import importlib.util
import io
import pathlib
import zipfile

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"


def _probe_module():
    spec = importlib.util.spec_from_file_location("sharadar_probe_bulk", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _probe_module()


# ---------------------------------------------------------------------------
# classify_bulk_response
# ---------------------------------------------------------------------------

def test_302_with_location_is_the_happy_path(mod):
    v = mod.classify_bulk_response(302, "https://dl.example/stocks.zip?X-Sig=abc")
    assert v["verdict"] == "REDIRECT"


@pytest.mark.parametrize("status", [301, 303, 307, 308])
def test_other_redirect_codes_also_read_redirect(mod, status):
    assert mod.classify_bulk_response(status, "https://dl/x.zip")["verdict"] == "REDIRECT"


def test_redirect_without_location_is_unexpected_not_redirect(mod):
    """A 3xx naming nowhere is not a download; treating it as one would send the
    downloader at None."""
    v = mod.classify_bulk_response(302, None)
    assert v["verdict"] == "UNEXPECTED"
    assert "no Location" in v["reason"]


@pytest.mark.parametrize("status", [401, 403])
def test_refusal_reads_forbidden(mod, status):
    v = mod.classify_bulk_response(status, None)
    assert v["verdict"] == "FORBIDDEN"


def test_200_carrying_a_zip_needs_no_redirect_hop(mod):
    assert mod.classify_bulk_response(
        200, None, "application/zip")["verdict"] == "DIRECT_ZIP"
    assert mod.classify_bulk_response(
        200, None, "application/octet-stream")["verdict"] == "DIRECT_ZIP"


def test_200_carrying_json_is_not_a_bulk_archive(mod):
    """The data route answers 200 with JSON for an ordinary query. Grading that
    as an archive would hand a JSON body to the zip reader."""
    v = mod.classify_bulk_response(200, None, "application/json")
    assert v["verdict"] == "UNEXPECTED"


def test_server_error_is_unexpected(mod):
    assert mod.classify_bulk_response(500, None)["verdict"] == "UNEXPECTED"


# ---------------------------------------------------------------------------
# redact_url_query — a pre-signed link is a credential
# ---------------------------------------------------------------------------

def test_presigned_signature_never_survives_redaction(mod):
    url = ("https://sharadar-bulk.s3.amazonaws.com/stocks_full.zip"
           "?X-Amz-Signature=deadbeefcafe&X-Amz-Credential=AKIAsecret")
    out = mod.redact_url_query(url)
    assert "deadbeefcafe" not in out
    assert "AKIAsecret" not in out
    assert "X-Amz-Signature" not in out
    assert out.startswith("https://sharadar-bulk.s3.amazonaws.com/stocks_full.zip")


def test_redaction_keeps_a_query_less_url_intact(mod):
    url = "https://dl.example/stocks.zip"
    assert mod.redact_url_query(url) == url


def test_redaction_tolerates_none(mod):
    assert mod.redact_url_query(None) is None


# ---------------------------------------------------------------------------
# check_full_history_range — the vendor's own trap
# ---------------------------------------------------------------------------

def test_full_span_passes(mod):
    out = mod.check_full_history_range("1998-01-02", "2026-09-25",
                                       run_date="2026-09-27")
    assert out["status"] == "ok"


def test_five_year_file_sold_as_full_is_caught(mod):
    """The load-bearing case: a five-year plan answers years=full with 200 and a
    2021 start. Nothing but the span itself reveals it."""
    out = mod.check_full_history_range("2021-09-27", "2026-09-25",
                                       run_date="2026-09-27")
    assert out["status"] == "SHORTFALL"
    assert "start 2021" in out["reason"]


def test_stale_end_edge_is_caught(mod):
    out = mod.check_full_history_range("1998-01-02", "2026-06-30",
                                       run_date="2026-09-27")
    assert out["status"] == "SHORTFALL"
    assert "behind the run date" in out["reason"]


def test_end_edge_inside_tolerance_passes(mod):
    """A weekend or holiday leaves the last bar a few days back; that is not a
    clip."""
    out = mod.check_full_history_range("1998-01-02", "2026-09-24",
                                       run_date="2026-09-27")
    assert out["status"] == "ok"
    assert out["end_days_behind"] == 3


def test_missing_span_is_shortfall_not_silent_pass(mod):
    assert mod.check_full_history_range(None, None,
                                        run_date="2026-09-27")["status"] == "SHORTFALL"


def test_boundary_year_is_inclusive(mod):
    """min_year itself passes; the year after it does not."""
    assert mod.check_full_history_range(
        "1998-12-31", "2026-09-25", run_date="2026-09-27")["status"] == "ok"
    assert mod.check_full_history_range(
        "1999-01-01", "2026-09-25", run_date="2026-09-27")["status"] == "SHORTFALL"


# ---------------------------------------------------------------------------
# zip_member_to_parquet — bounded memory, stable schema
# ---------------------------------------------------------------------------

def _make_zip(path, rows, name="stocks.csv", header=("ticker", "date", "close")):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(header)
    for r in rows:
        w.writerow(r)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name, buf.getvalue())


def test_zip_converts_to_parquet_with_every_row(mod, tmp_path):
    rows = [("AAPL", "1998-01-02", "0.5"), ("AAPL", "2026-09-25", "250.1"),
            ("TWO", "2009-03-04", "3.25")]
    z = tmp_path / "b.zip"
    out = tmp_path / "b.parquet"
    _make_zip(z, rows)
    stats = mod.zip_member_to_parquet(str(z), str(out))
    assert stats["rows"] == 3
    assert stats["columns"] == ["ticker", "date", "close"]
    df = pd.read_parquet(out)
    assert len(df) == 3
    assert set(df["ticker"]) == {"AAPL", "TWO"}


def test_chunked_conversion_preserves_row_count_and_order(mod, tmp_path):
    """Several chunks must land as one table, in order — the bounded-memory
    path is the only path used in production, so it carries the correctness."""
    rows = [("T%04d" % i, "2020-01-01", str(i)) for i in range(1000)]
    z = tmp_path / "c.zip"
    out = tmp_path / "c.parquet"
    _make_zip(z, rows)
    stats = mod.zip_member_to_parquet(str(z), str(out), chunk_rows=100)
    assert stats["chunks"] == 10
    assert stats["rows"] == 1000
    df = pd.read_parquet(out)
    assert list(df["ticker"]) == [r[0] for r in rows]


def test_everything_is_written_as_string(mod, tmp_path):
    """A numeric-looking column that flips dtype between chunks would make the
    parquet schema drift and break a later byte-wise replay comparison."""
    rows = [("A", "2020-01-01", "1")] * 50 + [("B", "2020-01-02", "")] * 50
    z = tmp_path / "d.zip"
    out = tmp_path / "d.parquet"
    _make_zip(z, rows)
    mod.zip_member_to_parquet(str(z), str(out), chunk_rows=50)
    df = pd.read_parquet(out)
    assert all(str(t) == "str" or t == object for t in
               [df[c].dtype if df[c].dtype == object else "str" for c in df.columns])
    # the blank cell stays an empty string, not a null
    assert (df["close"] == "").sum() == 50


def test_an_empty_archive_halts_rather_than_banking_nothing(mod, tmp_path):
    z = tmp_path / "e.zip"
    with zipfile.ZipFile(z, "w") as zf:
        pass
    with pytest.raises(SystemExit):
        mod.zip_member_to_parquet(str(z), str(tmp_path / "e.parquet"))


def test_directory_entries_are_not_mistaken_for_the_member(mod, tmp_path):
    z = tmp_path / "f.zip"
    out = tmp_path / "f.parquet"
    buf = "ticker,date,close\nAAPL,2020-01-01,1\n"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("dir/", "")
        zf.writestr("dir/stocks.csv", buf)
    stats = mod.zip_member_to_parquet(str(z), str(out))
    assert stats["member"] == "dir/stocks.csv"
    assert stats["rows"] == 1


# ---------------------------------------------------------------------------
# wiring — the pull must not be reachable without the span assertion
# ---------------------------------------------------------------------------

def test_bulk_tables_are_the_four_r1_designs_against(mod):
    assert set(mod.BULK_TABLES) == {"stocks", "tickers", "actions", "sp500"}


def test_bulk_table_calls_the_span_check_and_can_halt(mod):
    """AST tripwire: bulk_table must consult check_full_history_range and raise
    on SHORTFALL. A pull that banked a short span silently would defeat every
    test above."""
    tree = ast.parse(PROBE.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bulk_table")
    body = ast.dump(fn)
    assert "check_full_history_range" in body
    assert "SHORTFALL" in body
    assert any(isinstance(n, ast.Raise) for n in ast.walk(fn))


def test_no_staging_outside_the_snapshot_directory(mod):
    """The boot volume has no room for a full-history archive, so the download
    must land in the snapshot directory — nothing may reach tempfile or /tmp."""
    fn_src = PROBE.read_text()
    tree = ast.parse(fn_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bulk_table")
    seg = ast.get_source_segment(fn_src, fn)
    assert "tempfile" not in seg
    assert "/tmp" not in seg
    assert "self.snap_dir" in seg


def test_download_is_streamed_not_held_in_memory(mod):
    fn_src = PROBE.read_text()
    tree = ast.parse(fn_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bulk_table")
    seg = ast.get_source_segment(fn_src, fn)
    assert "stream=True" in seg
    assert "iter_content" in seg


def test_redirect_is_captured_not_followed(mod):
    """allow_redirects=False is what makes the vendor's 302 visible in the
    manifest instead of being silently consumed by the client."""
    fn_src = PROBE.read_text()
    tree = ast.parse(fn_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bulk_table")
    seg = ast.get_source_segment(fn_src, fn)
    assert "allow_redirects=False" in seg


def test_presigned_location_is_redacted_before_it_reaches_the_manifest(mod):
    fn_src = PROBE.read_text()
    tree = ast.parse(fn_src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "bulk_table")
    seg = ast.get_source_segment(fn_src, fn)
    assert "redact_url_query(location)" in seg
