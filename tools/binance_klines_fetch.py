"""
================================================================================
Script:       binance_klines_fetch.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Fetch the full Binance spot daily (1d) kline history for every
              USDT-quoted trading pair that survives a fixed exclusion rule, and
              bank it as a sha256-manifested vintage.
Entry Point:  Yes — python tools/binance_klines_fetch.py --vintage <YYYYMMDD>
Dependencies: stdlib, requests, pandas, pyarrow. Imports nothing from any other
              repository.
Last Sprint:  Sprint R4-0 — Crypto trend: data vintage and inventory (2026-10-15)
================================================================================

Public endpoints only. No API key is read, sent, or required; no .env is loaded.

The universe is resolved from /api/v3/exchangeInfo and narrowed by one fixed
rule: quote asset USDT, status TRADING, spot trading allowed, then two name-based
exclusions — a list of stable-coin and fiat base assets, and any base ending in
UP, DOWN, BULL or BEAR (Binance's leveraged-token naming). The rule is applied in
code and echoed into the manifest, so the universe is reproducible from the
artifact rather than from this docstring.

Each symbol is paged forward from startTime=0 at limit=1000 until a page comes
back short. A kline whose close_time has not yet passed is the unclosed UTC day
and is dropped, so a vintage never contains a partial bar.

The manifest is rewritten after every symbol, and a re-run under the same vintage
label reads it and skips the symbols already banked. An interrupted fetch
therefore resumes rather than restarting, and a completed fetch is a no-op.
================================================================================
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "binance_spot_1d"

DEFAULT_HOST = "https://api.binance.com"
FALLBACK_HOST = "https://data-api.binance.vision"

# Base assets excluded as stable-coin or fiat legs (a USDT pair against them is a
# currency quote, not a crypto asset).
STABLE_FIAT_BASES = frozenset({
    "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDP", "PAX", "USDS", "USD1",
    "PYUSD", "XUSD", "EUR", "EURI", "AEUR", "GBP", "AUD", "BRL", "TRY", "RUB",
    "UAH", "NGN", "ZAR", "JPY", "ARS",
})

# Binance leveraged-token suffixes.
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

FLOAT_FIELDS = ["open", "high", "low", "close", "volume", "quote_volume",
                "taker_buy_base", "taker_buy_quote"]

OUT_COLUMNS = ["open", "high", "low", "close", "volume", "close_time",
               "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]

SLEEP_SECONDS = 0.150
MAX_ATTEMPTS = 3
PAGE_LIMIT = 1000


def log(msg):
    print("[%s] %s" % (datetime.now(timezone.utc).strftime("%H:%M:%S"), msg),
          flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ universe
def exclusion_reason(base):
    """The reason this base asset is excluded, or None if it is kept."""
    if base in STABLE_FIAT_BASES:
        return "base in stable/fiat exclusion list"
    for suf in ("DOWN", "BULL", "BEAR", "UP"):
        if base.endswith(suf):
            return "base ends in %s (leveraged-token naming)" % suf
    return None


def filter_symbols(exchange_info):
    """Apply the fixed universe rule to an exchangeInfo payload.

    Returns (kept, excluded, counts) where kept is a sorted list of symbols,
    excluded is a list of {symbol, base, reason} for symbols that passed the
    venue filters but failed a name exclusion, and counts records the population
    after each filter step.
    """
    syms = exchange_info.get("symbols", [])
    counts = {"exchange_info_total": len(syms)}

    step = [s for s in syms if s.get("quoteAsset") == "USDT"]
    counts["quote_usdt"] = len(step)

    step = [s for s in step if s.get("status") == "TRADING"]
    counts["status_trading"] = len(step)

    step = [s for s in step if s.get("isSpotTradingAllowed") is True]
    counts["spot_allowed"] = len(step)

    kept, excluded = [], []
    for s in step:
        base = s.get("baseAsset", "")
        reason = exclusion_reason(base)
        if reason is None:
            kept.append(s["symbol"])
        else:
            excluded.append({"symbol": s["symbol"], "base": base,
                             "reason": reason})
    counts["kept"] = len(kept)
    counts["excluded_by_name"] = len(excluded)
    return sorted(kept), sorted(excluded, key=lambda e: e["symbol"]), counts


# ------------------------------------------------------------------- parsing
def parse_klines(raw, now_ms):
    """Twelve-field klines to a typed, UTC-indexed frame.

    Any kline whose close_time is at or beyond now_ms is still forming (the
    unclosed UTC day) and is dropped. An empty input yields an empty frame with
    the full schema, so callers never special-case it.
    """
    df = pd.DataFrame(list(raw), columns=KLINE_COLUMNS)
    if len(df):
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        df["trades"] = df["trades"].astype("int64")
        for c in FLOAT_FIELDS:
            df[c] = df[c].astype("float64")
        df = df[df["close_time"] < int(now_ms)]
    else:
        df["open_time"] = df["open_time"].astype("int64")
        df["close_time"] = df["close_time"].astype("int64")
        df["trades"] = df["trades"].astype("int64")
        for c in FLOAT_FIELDS:
            df[c] = df[c].astype("float64")
    df = df.drop(columns=["ignore"])
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    idx.name = "open_time"
    df = df.drop(columns=["open_time"])
    df.index = idx
    df = df[OUT_COLUMNS]
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def write_parquet(df, path):
    """Write the frame and return its sha256. Deterministic for a given frame."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)
    return sha256_file(path)


# ------------------------------------------------------------------ manifest
def manifest_path(vintage):
    return DATA / vintage / "manifest.json"


def load_manifest(vintage):
    p = manifest_path(vintage)
    if p.exists():
        return json.loads(p.read_text())
    return None


def completed_symbols(manifest):
    """Symbols already banked in this manifest (skipped on a re-run)."""
    if not manifest:
        return set()
    return set(manifest.get("symbols", {}).keys())


def save_manifest(vintage, manifest):
    p = manifest_path(vintage)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return p


# --------------------------------------------------------------------- fetch
def _get(session, url, params, label):
    """One GET with 429/418 backoff. Returns (payload, status) or (None, status)."""
    status = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        r = session.get(url, params=params, timeout=60)
        status = r.status_code
        if status == 200:
            return r.json(), status
        if status in (429, 418):
            wait = float(r.headers.get("Retry-After", 2 * attempt))
            log("%s: HTTP %d, honouring backoff %.1fs (attempt %d/%d)"
                % (label, status, wait, attempt, MAX_ATTEMPTS))
            time.sleep(wait)
            continue
        log("%s: HTTP %d (attempt %d/%d) %s"
            % (label, status, attempt, MAX_ATTEMPTS, r.text[:200]))
        time.sleep(1.0 * attempt)
    return None, status


def fetch_symbol(session, host, symbol, now_ms):
    """Page 1d klines forward from startTime=0. Returns (frame, calls, status)."""
    url = host.rstrip("/") + "/api/v3/klines"
    rows, calls, start = [], 0, 0
    while True:
        payload, status = _get(session, url,
                               {"symbol": symbol, "interval": "1d",
                                "limit": PAGE_LIMIT, "startTime": start},
                               "%s@%d" % (symbol, start))
        calls += 1
        if payload is None:
            return None, calls, status
        rows.extend(payload)
        time.sleep(SLEEP_SECONDS)
        if len(payload) < PAGE_LIMIT:
            break
        start = int(payload[-1][0]) + 1
    return parse_klines(rows, now_ms), calls, 200


def fetch_exchange_info(session, host):
    payload, status = _get(session, host.rstrip("/") + "/api/v3/exchangeInfo",
                           {}, "exchangeInfo")
    return payload, status


# ---------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description="Binance spot 1d kline vintage")
    ap.add_argument("--vintage",
                    default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    ap.add_argument("--host", default=DEFAULT_HOST)
    ap.add_argument("--symbols", default=None,
                    help="comma-separated subset; default is the whole rule")
    args = ap.parse_args(argv)

    import requests
    session = requests.Session()
    t0 = time.time()

    info, status = fetch_exchange_info(session, args.host)
    if info is None and args.host != FALLBACK_HOST:
        log("exchangeInfo refused on %s (HTTP %s); trying %s"
            % (args.host, status, FALLBACK_HOST))
        args.host = FALLBACK_HOST
        info, status = fetch_exchange_info(session, args.host)
    if info is None:
        log("FATAL: exchangeInfo unavailable (HTTP %s)" % status)
        return 2
    calls = 1

    kept, excluded, counts = filter_symbols(info)
    log("universe: %s" % json.dumps(counts))
    if args.symbols:
        subset = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
        kept = [s for s in kept if s in subset]
        log("subset requested: %d of %d kept symbols" % (len(kept), len(subset)))

    now_ms = int(info.get("serverTime") or time.time() * 1000)

    man = load_manifest(args.vintage) or {
        "vintage": args.vintage,
        "tool": "binance_klines_fetch.py",
        "tool_version": VERSION,
        "host": args.host,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "exclusions_applied": {
            "stable_fiat_bases": sorted(STABLE_FIAT_BASES),
            "leveraged_suffixes": list(LEVERAGED_SUFFIXES),
            "excluded_symbols": excluded,
        },
        "universe_counts": counts,
        "symbols": {},
        "failures": {},
    }
    man["last_run_at"] = datetime.now(timezone.utc).isoformat()
    done = completed_symbols(man)
    todo = [s for s in kept if s not in done]
    log("vintage %s: %d banked, %d to fetch" % (args.vintage, len(done), len(todo)))

    for i, sym in enumerate(todo, start=1):
        df, c, st = fetch_symbol(session, args.host, sym, now_ms)
        calls += c
        if df is None:
            man["failures"][sym] = {"last_status": st, "attempts": MAX_ATTEMPTS}
            log("%4d/%4d %-14s FAILED (HTTP %s)" % (i, len(todo), sym, st))
        else:
            p = DATA / args.vintage / ("%s.parquet" % sym)
            sha = write_parquet(df, p)
            man["symbols"][sym] = {
                "rows": int(len(df)),
                "first_open_time": (df.index[0].isoformat() if len(df) else None),
                "last_open_time": (df.index[-1].isoformat() if len(df) else None),
                "sha256": sha,
            }
            man["failures"].pop(sym, None)
            log("%4d/%4d %-14s rows %5d  %s..%s"
                % (i, len(todo), sym, len(df),
                   man["symbols"][sym]["first_open_time"],
                   man["symbols"][sym]["last_open_time"]))
        save_manifest(args.vintage, man)

    man["calls_made_this_run"] = calls
    man["wall_seconds_this_run"] = round(time.time() - t0, 1)
    mp = save_manifest(args.vintage, man)
    log("manifest %s sha %s" % (mp, sha256_file(mp)))
    log("DONE symbols %d failures %d calls %d wall %.1fs"
        % (len(man["symbols"]), len(man["failures"]), calls, time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
