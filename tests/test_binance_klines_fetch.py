"""Tests for the Binance spot 1d fetch tool (tools/binance_klines_fetch.py).

No network. Every kline is a constructed twelve-field fixture with a
hand-checkable answer, and the exchangeInfo fixture carries one symbol per
exclusion class so the universe rule is pinned class by class.

The mutations this battery is built to catch are: a universe rule that drops a
filter step or a name exclusion; a parser that keeps the unclosed day; a re-run
that re-fetches already-banked symbols; and a parquet writer whose bytes are not
a function of the frame alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import binance_klines_fetch as b  # noqa: E402


DAY_MS = 86_400_000


def kline(open_ms, close=100.0, volume=7.0):
    """One twelve-field 1d kline as the venue serves it: strings for decimals."""
    return [
        open_ms, "%.8f" % close, "%.8f" % (close + 1), "%.8f" % (close - 1),
        "%.8f" % close, "%.8f" % volume, open_ms + DAY_MS - 1,
        "%.8f" % (volume * close), 11, "%.8f" % (volume / 2),
        "%.8f" % (volume * close / 2), "0",
    ]


def exchange_info_fixture():
    """One symbol per outcome: kept, and every exclusion class."""
    def s(sym, base, quote="USDT", status="TRADING", spot=True):
        return {"symbol": sym, "baseAsset": base, "quoteAsset": quote,
                "status": status, "isSpotTradingAllowed": spot}
    return {"symbols": [
        s("BTCUSDT", "BTC"),                      # kept
        s("ETHUSDT", "ETH"),                      # kept
        s("ETHBTC", "ETH", quote="BTC"),          # dropped: quote not USDT
        s("DEADUSDT", "DEAD", status="BREAK"),    # dropped: not TRADING
        s("NOSPOTUSDT", "NOSPOT", spot=False),    # dropped: spot not allowed
        s("USDCUSDT", "USDC"),                    # excluded: stable base
        s("EURUSDT", "EUR"),                      # excluded: fiat base
        s("BTCUPUSDT", "BTCUP"),                  # excluded: UP
        s("BTCDOWNUSDT", "BTCDOWN"),              # excluded: DOWN
        s("ETHBULLUSDT", "ETHBULL"),              # excluded: BULL
        s("ETHBEARUSDT", "ETHBEAR"),              # excluded: BEAR
    ]}


# --------------------------------------------------------------- 1. schema
def test_schema_parsing_types_and_values():
    raw = [kline(0, close=100.0, volume=7.0), kline(DAY_MS, close=200.0)]
    df = b.parse_klines(raw, now_ms=10 * DAY_MS)
    assert list(df.columns) == b.OUT_COLUMNS
    assert len(df) == 2
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("1970-01-01", tz="UTC")
    assert df.index[1] == pd.Timestamp("1970-01-02", tz="UTC")
    assert df["close"].dtype == "float64"
    assert df["trades"].dtype == "int64"
    assert df["close_time"].dtype == "int64"
    assert df["open"].iloc[0] == pytest.approx(100.0)
    assert df["high"].iloc[0] == pytest.approx(101.0)
    assert df["low"].iloc[0] == pytest.approx(99.0)
    assert df["volume"].iloc[0] == pytest.approx(7.0)
    assert df["quote_volume"].iloc[0] == pytest.approx(700.0)
    assert df["taker_buy_base"].iloc[0] == pytest.approx(3.5)
    assert "ignore" not in df.columns


def test_parse_klines_empty_input_keeps_the_schema():
    df = b.parse_klines([], now_ms=10 * DAY_MS)
    assert len(df) == 0
    assert list(df.columns) == b.OUT_COLUMNS


def test_parse_klines_dedupes_and_sorts():
    raw = [kline(DAY_MS), kline(0), kline(DAY_MS)]
    df = b.parse_klines(raw, now_ms=10 * DAY_MS)
    assert len(df) == 2
    assert df.index.is_monotonic_increasing


# ------------------------------------------------------- 2. universe filter
def test_symbol_filter_every_exclusion_class():
    kept, excluded, counts = b.filter_symbols(exchange_info_fixture())
    assert kept == ["BTCUSDT", "ETHUSDT"]
    assert counts["exchange_info_total"] == 11
    assert counts["quote_usdt"] == 10
    assert counts["status_trading"] == 9
    assert counts["spot_allowed"] == 8
    assert counts["kept"] == 2
    assert counts["excluded_by_name"] == 6
    by_sym = {e["symbol"]: e["reason"] for e in excluded}
    assert set(by_sym) == {"USDCUSDT", "EURUSDT", "BTCUPUSDT", "BTCDOWNUSDT",
                           "ETHBULLUSDT", "ETHBEARUSDT"}
    assert "stable/fiat" in by_sym["USDCUSDT"]
    assert "stable/fiat" in by_sym["EURUSDT"]
    assert by_sym["BTCUPUSDT"].endswith("(leveraged-token naming)")
    assert "UP" in by_sym["BTCUPUSDT"]
    assert "DOWN" in by_sym["BTCDOWNUSDT"]
    assert "BULL" in by_sym["ETHBULLUSDT"]
    assert "BEAR" in by_sym["ETHBEARUSDT"]


def test_exclusion_reason_is_none_for_a_plain_base():
    assert b.exclusion_reason("BTC") is None
    assert b.exclusion_reason("SOL") is None


def test_excluded_list_is_sorted_by_symbol():
    _, excluded, _ = b.filter_symbols(exchange_info_fixture())
    assert [e["symbol"] for e in excluded] == sorted(e["symbol"] for e in excluded)


# ------------------------------------------------------ 3. unclosed-day drop
def test_unclosed_day_is_dropped():
    # three days; now sits inside day 3, whose close_time has not passed
    raw = [kline(0), kline(DAY_MS), kline(2 * DAY_MS)]
    now = 2 * DAY_MS + 3_600_000           # one hour into day 3
    df = b.parse_klines(raw, now_ms=now)
    assert len(df) == 2
    assert df.index[-1] == pd.Timestamp("1970-01-02", tz="UTC")


def test_closed_day_is_kept_at_the_exact_boundary():
    raw = [kline(0)]
    # close_time is 23:59:59.999; one millisecond later the bar is closed
    df = b.parse_klines(raw, now_ms=DAY_MS)
    assert len(df) == 1
    # and at its own close_time it is not yet closed
    assert len(b.parse_klines(raw, now_ms=DAY_MS - 1)) == 0


# --------------------------------------------------- 4. manifest skip on re-run
def test_manifest_skip_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "DATA", tmp_path / "binance_spot_1d")
    man = {"vintage": "20260101", "symbols": {
        "BTCUSDT": {"rows": 10, "sha256": "x"},
        "ETHUSDT": {"rows": 10, "sha256": "y"}}, "failures": {}}
    p = b.save_manifest("20260101", man)
    assert p.exists()
    back = b.load_manifest("20260101")
    assert back["symbols"].keys() == {"BTCUSDT", "ETHUSDT"}
    done = b.completed_symbols(back)
    kept = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert [s for s in kept if s not in done] == ["SOLUSDT"]


def test_completed_symbols_on_absent_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "DATA", tmp_path / "binance_spot_1d")
    assert b.load_manifest("20990101") is None
    assert b.completed_symbols(None) == set()


def test_failed_symbol_is_not_treated_as_banked(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "DATA", tmp_path / "binance_spot_1d")
    b.save_manifest("20260101", {"symbols": {}, "failures": {
        "SOLUSDT": {"last_status": 429}}})
    done = b.completed_symbols(b.load_manifest("20260101"))
    assert "SOLUSDT" not in done


# ------------------------------------------------- 5. parquet sha256 stability
def test_parquet_sha256_stable_across_two_writes(tmp_path):
    df = b.parse_klines([kline(i * DAY_MS, close=100.0 + i) for i in range(5)],
                        now_ms=10 * DAY_MS)
    a = b.write_parquet(df, tmp_path / "a.parquet")
    c = b.write_parquet(df, tmp_path / "b.parquet")
    assert a == c
    assert len(a) == 64


def test_parquet_sha256_changes_when_a_value_changes(tmp_path):
    df = b.parse_klines([kline(i * DAY_MS) for i in range(5)],
                        now_ms=10 * DAY_MS)
    a = b.write_parquet(df, tmp_path / "a.parquet")
    df2 = df.copy()
    df2.iloc[0, df2.columns.get_loc("close")] = 1.0
    c = b.write_parquet(df2, tmp_path / "c.parquet")
    assert a != c


def test_parquet_roundtrip_preserves_index_and_dtypes(tmp_path):
    df = b.parse_klines([kline(i * DAY_MS) for i in range(3)],
                        now_ms=10 * DAY_MS)
    b.write_parquet(df, tmp_path / "a.parquet")
    back = pd.read_parquet(tmp_path / "a.parquet")
    assert back.index.name == "open_time"
    assert str(back.index.tz) == "UTC"
    pd.testing.assert_frame_equal(df, back)


# ------------------------------------------------------------- housekeeping
def test_no_credential_is_read_anywhere_in_the_tool():
    # the module docstring is prose and says the tool reads no .env; the check
    # is on executable code, so strip the docstring before scanning
    import ast
    path = ROOT / "tools" / "binance_klines_fetch.py"
    src = path.read_text()
    tree = ast.parse(src)
    doc = ast.get_docstring(tree) or ""
    code = src.replace(doc, "")
    for forbidden in ("dotenv", ".env", "API_KEY", "api_key", "X-MBX-APIKEY",
                      "secret", "load_dotenv"):
        assert forbidden not in code, forbidden


def test_manifest_is_written_as_sorted_json(tmp_path, monkeypatch):
    monkeypatch.setattr(b, "DATA", tmp_path / "binance_spot_1d")
    p = b.save_manifest("20260101", {"b": 1, "a": 2})
    assert json.loads(p.read_text()) == {"b": 1, "a": 2}
    assert p.read_text().index('"a"') < p.read_text().index('"b"')
