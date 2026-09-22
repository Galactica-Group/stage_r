"""
================================================================================
Script:       test_probe_replay.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin probe v1.2 --replay: a manifest is re-issued entry for entry,
              in order, with the same names, endpoints and params; derived rows
              are rebuilt rather than fetched; an occupied label is refused.
Entry Point:  No — pytest
Dependencies: stdlib (importlib, json, pathlib, sys); pandas
Last Sprint:  Sprint R0-B — Snapshot 2 by replay (2026-09-22)
================================================================================

The HTTP layer is mocked at requests.get, so budget accounting, redaction,
parsing, the date-range assertion and manifest writing all run for real; only
the network is absent.
================================================================================
"""

import importlib.util
import json
import pathlib
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"

CAPTURE_KEYS = {
    "name", "url_redacted", "params", "http_status", "rows", "columns",
    "raw_file", "raw_sha256", "raw_bytes", "parquet_file", "parquet_sha256",
    "fetched_at_utc", "date_range_check",
}
DERIVED_KEYS = {
    "name", "derived_from", "url_redacted", "params", "http_status", "rows",
    "columns", "raw_file", "raw_sha256", "raw_bytes", "parquet_file",
    "parquet_sha256", "fetched_at_utc",
}


def _load(tmp_root):
    """A fresh probe module whose every write lands under tmp_root."""
    spec = importlib.util.spec_from_file_location("sharadar_probe_replay", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.REPO_ROOT = str(tmp_root)
    mod.SLEEP_BETWEEN_S = 0.0
    return mod


class FakeResp:
    def __init__(self, url, params, status, body, ctype):
        qs = "&".join("%s=%s" % (k, v) for k, v in (params or {}).items())
        self.url = url + ("?" + qs if qs else "")
        self.status_code = status
        self.text = body
        self.content = body.encode()
        self.headers = {"Content-Type": ctype}

    def json(self):
        return json.loads(self.text)


# Three entries: a page with no key, a keyed table, and a row derived from it.
SOURCE_MANIFEST = [
    {
        "name": "page_terms",
        "url_redacted": "https://sharadar.com/terms",
        "params": {},
        "http_status": 200,
        "rows": None,
        "parquet_file": None,
    },
    {
        "name": "stocks_w_2024",
        "url_redacted": ("https://api.sharadar.com/v1.0/data/stocks"
                         "?from=2024-01-01&to=2024-12-31&limit=10000&format=json"
                         "&api_key=<REDACTED>"),
        "params": {"from": "2024-01-01", "to": "2024-12-31",
                   "limit": 10000, "format": "json"},
        "http_status": 200,
        "rows": 2,
        "parquet_file": "stocks_w_2024.parquet",
    },
    {
        "name": "stocks_combined",
        "derived_from": ["stocks_w_2024"],
        "url_redacted": None,
        "params": {"note": "one window, deduped on (ticker,date)"},
        "http_status": None,
        "rows": 2,
        "parquet_file": "stocks_dow30.parquet",
    },
]

TABLE_BODY = json.dumps({"count": 2, "data": [
    {"ticker": "AAPL", "date": "2024-01-02", "close": "1"},
    {"ticker": "AAPL", "date": "2024-12-31", "close": "2"},
]})


@pytest.fixture
def replayed(tmp_path, monkeypatch):
    """Run a full replay of SOURCE_MANIFEST with the network mocked out."""
    mod = _load(tmp_path)
    calls = []

    def fake_get(url, params=None, timeout=None, headers=None):
        calls.append({"url": url, "params": dict(params or {})})
        if "api.sharadar.com" in url:
            return FakeResp(url, params, 200, TABLE_BODY, "application/json")
        return FakeResp(url, params, 200, "<html>terms</html>", "text/html")

    monkeypatch.setattr(mod.requests, "get", fake_get)
    p = mod.Probe("testlabel", "SECRETKEY", "https://api.sharadar.com/v1.0")
    summary = p.replay_manifest(SOURCE_MANIFEST)
    manifest = json.loads(pathlib.Path(p.manifest_path).read_text())
    return mod, p, calls, summary, manifest


# ---------------------------------------------------------------------------
# The replay contract
# ---------------------------------------------------------------------------

def test_replay_issues_the_same_names_in_the_same_order(replayed):
    _mod, _p, _calls, _summary, manifest = replayed
    assert [e["name"] for e in manifest] == [e["name"] for e in SOURCE_MANIFEST]


def test_replay_sends_the_same_params_on_every_captured_entry(replayed):
    _mod, _p, calls, _summary, _manifest = replayed
    sent = [{k: v for k, v in c["params"].items() if k != "api_key"} for c in calls]
    expected = [e["params"] for e in SOURCE_MANIFEST if e["url_redacted"] is not None]
    assert sent == expected
    # teeth: at least one entry carries params, so equality is not vacuous
    assert any(s for s in sent), "fixture must include a params-bearing entry"


def test_replay_hits_the_recorded_endpoint_with_the_query_stripped(replayed):
    _mod, _p, calls, _summary, _manifest = replayed
    assert [c["url"] for c in calls] == [
        "https://sharadar.com/terms",
        "https://api.sharadar.com/v1.0/data/stocks",
    ]
    assert all("?" not in c["url"] for c in calls)


def test_replay_reproduces_the_recorded_auth_choice(replayed):
    _mod, _p, calls, _summary, _manifest = replayed
    assert "api_key" not in calls[0]["params"], "a page fetch must stay unauthenticated"
    assert calls[1]["params"].get("api_key") == "SECRETKEY", "a table fetch must send the key"


def test_replay_writes_a_manifest_of_the_same_shape(replayed):
    _mod, _p, _calls, _summary, manifest = replayed
    assert set(manifest[0]) == CAPTURE_KEYS
    assert set(manifest[1]) == CAPTURE_KEYS
    assert set(manifest[2]) == DERIVED_KEYS


def test_replay_spends_one_request_per_captured_entry_only(replayed):
    _mod, p, calls, _summary, _manifest = replayed
    assert len(calls) == 2, "the derived row must not be fetched"
    assert p.budget_used() == 2


def test_derived_row_is_rebuilt_from_its_named_source(replayed):
    _mod, p, _calls, _summary, manifest = replayed
    derived = manifest[2]
    assert derived["derived_from"] == ["stocks_w_2024"]
    assert derived["http_status"] is None and derived["raw_file"] is None
    assert derived["parquet_file"] == "stocks_dow30.parquet"
    df = pd.read_parquet(pathlib.Path(p.snap_dir) / derived["parquet_file"])
    assert len(df) == 2 and derived["rows"] == 2
    assert {"ticker", "date", "close"} <= set(df.columns)


def test_summary_carries_both_the_old_and_the_new_status(replayed):
    _mod, _p, _calls, summary, _manifest = replayed
    assert [s["kind"] for s in summary] == ["capture", "capture", "derived"]
    assert summary[1]["http_status"] == 200
    assert summary[1]["was_status"] == 200 and summary[1]["was_rows"] == 2


def test_an_error_payload_entry_is_replayed_not_skipped(tmp_path, monkeypatch):
    """A 403 in the source manifest is re-issued; its status is the measurement."""
    mod = _load(tmp_path)
    src = [{"name": "restrict_daily_TSLA",
            "url_redacted": ("https://api.sharadar.com/v1.0/data/daily"
                             "?ticker=TSLA&api_key=<REDACTED>"),
            "params": {"ticker": "TSLA"}, "http_status": 403, "rows": 1,
            "parquet_file": "restrict_daily_TSLA.parquet"}]
    seen = []

    def fake_get(url, params=None, timeout=None, headers=None):
        seen.append(url)
        return FakeResp(url, params, 403,
                        json.dumps([{"error": "forbidden", "description": "upgrade"}]),
                        "application/json")

    monkeypatch.setattr(mod.requests, "get", fake_get)
    p = mod.Probe("errlabel", "SECRETKEY", "https://api.sharadar.com/v1.0")
    summary = p.replay_manifest(src)
    assert seen == ["https://api.sharadar.com/v1.0/data/daily"]
    assert summary[0]["http_status"] == 403 and summary[0]["was_status"] == 403


# ---------------------------------------------------------------------------
# Refusal to overwrite, and the pure helpers
# ---------------------------------------------------------------------------

def test_replay_refuses_a_label_whose_snapshot_directory_exists(tmp_path):
    mod = _load(tmp_path)
    (tmp_path / "data" / "sharadar" / "snapshot_taken").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        mod.assert_label_is_new(str(tmp_path), "taken")
    assert "REFUSED" in str(exc.value)


def test_replay_refuses_a_label_whose_budget_file_exists(tmp_path):
    mod = _load(tmp_path)
    (tmp_path / "logs").mkdir(parents=True)
    (tmp_path / "logs" / "budget_spent.json").write_text("{}")
    with pytest.raises(SystemExit) as exc:
        mod.assert_label_is_new(str(tmp_path), "spent")
    assert "REFUSED" in str(exc.value)


def test_a_virgin_label_is_allowed(tmp_path):
    mod = _load(tmp_path)
    assert mod.assert_label_is_new(str(tmp_path), "fresh") is None


def test_endpoint_of_strips_the_query_and_keeps_the_table_path(tmp_path):
    mod = _load(tmp_path)
    assert mod.endpoint_of(
        "https://api.sharadar.com/v1.0/data/stocks?limit=5&api_key=<REDACTED>"
    ) == "https://api.sharadar.com/v1.0/data/stocks"
    assert mod.endpoint_of("https://sharadar.com/terms") == "https://sharadar.com/terms"


def test_replay_used_auth_reads_the_recorded_query(tmp_path):
    mod = _load(tmp_path)
    assert mod.replay_used_auth("https://x/y?a=1&api_key=<REDACTED>") is True
    assert mod.replay_used_auth("https://x/y?a=1") is False


def test_is_derived_entry_separates_computed_rows_from_fetched_ones(tmp_path):
    mod = _load(tmp_path)
    assert mod.is_derived_entry({"url_redacted": None}) is True
    assert mod.is_derived_entry({"url_redacted": "https://x", "derived_from": ["a"]}) is True
    assert mod.is_derived_entry({"url_redacted": "https://x"}) is False


def test_a_derived_row_with_a_missing_source_halts(tmp_path, monkeypatch):
    mod = _load(tmp_path)
    p = mod.Probe("orphan", "SECRETKEY", "https://api.sharadar.com/v1.0")
    with pytest.raises(SystemExit) as exc:
        p.replay_derived({"name": "combined", "derived_from": ["never_fetched"],
                          "params": {}, "parquet_file": "combined.parquet"})
    assert "HALT" in str(exc.value)
