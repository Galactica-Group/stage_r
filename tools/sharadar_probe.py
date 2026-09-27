"""
================================================================================
Script:       sharadar_probe.py
Version:      1.6.0
Phase:        Stage R — External Data Evaluation
Purpose:      Read-only probe of the Sharadar free tier: surface discovery, table
              sampling, and snapshot capture with sha256 manifesting.
Entry Point:  Yes — python tools/sharadar_probe.py --snapshot <label> <command>
Dependencies: stdlib, requests, pandas, pyarrow. Imports nothing from any other
              repository.
Last Sprint:  Sprint R0-C — Paid tier verified; bulk full-history pull (2026-09-27)
================================================================================

The API key is read at runtime from a .env-style file whose path is supplied by
--env-path; it is never written to disk, logs, or stdout. Every HTTP request is
budget-counted (shared across invocations of one snapshot label), rate-limited,
logged with the key redacted, and saved raw beside a parquet rendering.

Every request that names a date window has its returned span checked against the
window it asked for: the vendor clips a long request to about a year and returns
HTTP 200, so a shortfall halts the probe with both ranges printed rather than
banking a silently partial snapshot.

--replay re-issues an earlier snapshot's manifest under a new label: the same
entries, in the same order, with the same names, endpoints and params, so the two
snapshots differ only by the day they were fetched. A manifest entry that records
no URL is a derived row, not a request; it is rebuilt from its named sources
instead of being fetched. A vendor error payload is replayed like any other entry,
because its status is itself part of what the comparison measures.

A replayed manifest may contain a request the vendor already clipped, so in replay
mode the date assertion is adjudicated against the span the source snapshot itself
received: reproducing a known clip is recorded and allowed through, while a span
shorter than last time's is a regression and halts. The requested-window check is
written to the manifest either way, so the clip never stops being visible.

The vendor serves a rolling window, so the oldest bars age out between snapshots.
--replay-allow-rolloff states how many days of start-edge loss that accounts for
— the calendar days between the two fetches. A start edge that recedes by no more
than that is retention, recorded as ROLLOFF; further than that, or any loss at the
end edge, is still a regression and still halts.
================================================================================
"""

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlsplit, urlunsplit

import pandas as pd
import requests

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQUEST_BUDGET = 80
SLEEP_BETWEEN_S = 1.0
RETRY_WAIT_S = 30.0
KEY_PARAM = "api_key"
RATE_HEADER_HINTS = ("ratelimit", "rate-limit", "retry-after", "x-rate")
# A requested window edge may legitimately fall on a non-trading day; anything
# beyond this many calendar days short of the request is a clip, not a weekend.
MAX_EDGE_GAP_DAYS = 5
DATE_PARAM_FROM = "from"
DATE_PARAM_TO = "to"


def check_date_range(params, df):
    """Compare a request's date window against the span actually returned.

    Returns a dict with status "ok", "not_applicable" or "SHORTFALL". Pure:
    takes the request params and the parsed frame, touches nothing else.
    """
    params = params or {}
    want_from = params.get(DATE_PARAM_FROM)
    want_to = params.get(DATE_PARAM_TO)
    if not want_from and not want_to:
        return {"status": "not_applicable", "reason": "no date window requested"}
    if df is None or "date" not in getattr(df, "columns", []):
        return {"status": "not_applicable", "reason": "no date column returned"}
    dates = pd.to_datetime(pd.Series(df["date"]).astype(str), errors="coerce").dropna()
    if not len(dates):
        return {
            "status": "SHORTFALL",
            "requested": [want_from, want_to],
            "returned": [None, None],
            "reason": "window requested but no dated rows returned",
        }
    got_from, got_to = dates.min(), dates.max()
    gaps = {}
    if want_from:
        gaps["start_days_short"] = int((got_from - pd.to_datetime(want_from)).days)
    if want_to:
        gaps["end_days_short"] = int((pd.to_datetime(want_to) - got_to).days)
    short = [k for k, v in gaps.items() if v > MAX_EDGE_GAP_DAYS]
    out = {
        "requested": [want_from, want_to],
        "returned": [got_from.strftime("%Y-%m-%d"), got_to.strftime("%Y-%m-%d")],
        "gaps_days": gaps,
        "tolerance_days": MAX_EDGE_GAP_DAYS,
    }
    out["status"] = "SHORTFALL" if short else "ok"
    if short:
        out["reason"] = "returned span falls short at: %s" % ",".join(sorted(short))
    return out


# ---------------------------------------------------------------------------
# Bulk downloads (paid tier). GET /data/{table}?years=full answers 302 to a
# pre-signed zip. The redirect is captured deliberately rather than followed
# automatically, so the manifest records the vendor's own answer to the bulk
# request separately from the bytes that answer points at.
# ---------------------------------------------------------------------------
BULK_TABLES = ("stocks", "tickers", "actions", "sp500")
BULK_YEARS = "full"
# Tables the vendor publishes only as a whole-table snapshot, so a full-history
# request is answered by the same file a 5- or 10-year request would get.
BULK_SNAPSHOT_ONLY = ("tickers", "descriptions")
# Full history must reach at least this far back for the stocks table; Sharadar
# documents nearly 30 years and R1 designs against a pre-2000 start.
FULL_HISTORY_MIN_YEAR = 1998
# Rows per parquet write. Bounded so a multi-gigabyte table converts in fixed
# memory instead of being materialised whole.
BULK_CHUNK_ROWS = 250_000


def redact_url_query(url):
    """Drop a URL's query string. Pre-signed bulk links carry a signature that
    is a credential in its own right, so only the path is ever recorded."""
    if not url:
        return url
    parts = urlsplit(str(url))
    if not parts.query:
        return url
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "<REDACTED_QUERY>", ""))


def classify_bulk_response(status, location, content_type=""):
    """Judge a bulk request's first response. Pure: no I/O.

    REDIRECT   — 3xx naming a download location, the documented happy path.
    FORBIDDEN  — 401/403; this plan does not carry the table.
    DIRECT_ZIP — 200 already carrying the archive, no redirect hop needed.
    UNEXPECTED — anything else, including a 3xx with no location.
    """
    ct = (content_type or "").lower()
    if status in (301, 302, 303, 307, 308):
        if location:
            return {"verdict": "REDIRECT", "reason": "%d to a download location" % status}
        return {"verdict": "UNEXPECTED", "reason": "%d with no Location header" % status}
    if status in (401, 403):
        return {"verdict": "FORBIDDEN", "reason": "%d — table not on this plan" % status}
    if status == 200 and ("zip" in ct or "octet-stream" in ct):
        return {"verdict": "DIRECT_ZIP", "reason": "200 carrying %s" % ct}
    return {"verdict": "UNEXPECTED", "reason": "status %d ct=%s" % (status, ct)}


def check_full_history_range(earliest, latest, run_date, min_year=FULL_HISTORY_MIN_YEAR,
                            tol_days=MAX_EDGE_GAP_DAYS):
    """Assert a full-history pull actually spans full history. Pure.

    A bulk file matches the purchased history, so a plan sold as five years
    answers years=full with five years and HTTP 200. The only way to tell is to
    read the span back: the start must reach min_year or earlier and the end must
    sit within tol_days of the run date.
    """
    out = {"earliest": earliest, "latest": latest, "run_date": run_date,
           "min_year": min_year, "tolerance_days": tol_days}
    if not earliest or not latest:
        out["status"] = "SHORTFALL"
        out["reason"] = "no dated rows to check"
        return out
    first = pd.to_datetime(earliest)
    last = pd.to_datetime(latest)
    end_gap = int((pd.to_datetime(run_date) - last).days)
    out["start_year"] = int(first.year)
    out["end_days_behind"] = end_gap
    fails = []
    if first.year > min_year:
        fails.append("start %d is later than %d" % (first.year, min_year))
    if end_gap > tol_days:
        fails.append("end %s is %d days behind the run date" % (latest, end_gap))
    out["status"] = "SHORTFALL" if fails else "ok"
    if fails:
        out["reason"] = "; ".join(fails)
    return out


def zip_member_to_parquet(zip_path, out_path, chunk_rows=BULK_CHUNK_ROWS):
    """Convert the single CSV inside a bulk zip to parquet in bounded memory.

    Everything is written as string, matching the probe's existing convention,
    so the schema cannot drift between chunks and a snapshot compares byte-wise
    against its replay. Returns a stats dict; raises if the zip holds no CSV.
    """
    import zipfile
    import pyarrow as pa
    import pyarrow.parquet as pq

    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        csvs = [n for n in members if n.lower().endswith((".csv", ".txt"))] or members
        if not csvs:
            raise SystemExit("HALT: bulk archive %s holds no member" % zip_path)
        member = csvs[0]
        rows = 0
        chunks = 0
        columns = None
        writer = None
        try:
            with zf.open(member) as fh:
                reader = pd.read_csv(fh, dtype=str, keep_default_na=False,
                                     chunksize=chunk_rows, low_memory=False)
                for chunk in reader:
                    if columns is None:
                        columns = list(chunk.columns)
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(out_path, table.schema,
                                                  compression="snappy")
                    writer.write_table(table)
                    rows += len(chunk)
                    chunks += 1
        finally:
            if writer is not None:
                writer.close()
    return {"member": member, "members": members, "rows": rows,
            "chunks": chunks, "columns": columns or []}


# The three requests that separate a paid key from a free one. Each carries the
# verdict R0-A/R0-B recorded for it under the free key, so a rerun is judged
# against measured history rather than an assumption.
TIER_PROBES = (
    {"name": "tier_tickers_free", "path": "data/tickers",
     "params": {"limit": 5, "format": "json"},
     "paid_gated": False, "free_tier_observed": "200 (R0-A/R0-B)"},
    {"name": "tier_stocks_TSLA", "path": "data/stocks",
     "params": {"ticker": "TSLA", "limit": 5, "format": "json"},
     "paid_gated": True, "free_tier_observed": "403 Exceeds free tier (R0-A/R0-B)"},
    {"name": "tier_sp500", "path": "data/sp500",
     "params": {"limit": 5, "format": "json"},
     "paid_gated": True, "free_tier_observed": "200 (R0-B manifest: probe5_sp500, 5 rows)"},
)
# Response headers worth surfacing when asking what plan a key is on.
PLAN_HEADER_HINTS = ("plan", "tier", "subscription", "quota", "entitle", "limit",
                     "credit", "ratelimit", "rate-limit")


def plan_headers(headers):
    """The subset of response headers that could name a plan, tier or quota."""
    return {k: v for k, v in dict(headers or {}).items()
            if any(h in k.lower() for h in PLAN_HEADER_HINTS)}


def classify_tier(results):
    """Judge whether a key is on the paid tier from the tier-probe results.

    results: list of dicts with keys name, paid_gated, http_status, rows.
    Pure: no I/O, no network. Verdict is one of PAID, FREE, BASELINE_FAILED,
    INDETERMINATE. A 403 on any paid-gated probe is FREE and must halt; the
    baseline failing means the key or endpoint is wrong, not that the tier is.
    """
    by_name = {r["name"]: r for r in results}
    baseline = [r for r in results if not r.get("paid_gated")]
    gated = [r for r in results if r.get("paid_gated")]
    if not gated:
        return {"verdict": "INDETERMINATE", "reason": "no paid-gated probe ran",
                "probes": by_name}
    if any(r.get("http_status") != 200 for r in baseline):
        bad = [r["name"] for r in baseline if r.get("http_status") != 200]
        return {"verdict": "BASELINE_FAILED",
                "reason": "free-tier baseline did not return 200: %s" % ",".join(bad),
                "probes": by_name}
    forbidden = [r["name"] for r in gated if r.get("http_status") == 403]
    if forbidden:
        return {"verdict": "FREE",
                "reason": "paid-gated endpoint(s) still 403: %s" % ",".join(forbidden),
                "probes": by_name}
    unhealthy = [r["name"] for r in gated
                 if r.get("http_status") != 200 or not (r.get("rows") or 0) > 0]
    if unhealthy:
        return {"verdict": "INDETERMINATE",
                "reason": "paid-gated endpoint(s) not 200-with-rows: %s" % ",".join(unhealthy),
                "probes": by_name}
    return {"verdict": "PAID",
            "reason": "every paid-gated endpoint returned 200 with rows",
            "probes": by_name}


def endpoint_of(url):
    """The request URL with its query string removed.

    A manifest records the resolved URL (key redacted), which carries the command
    and the table in its path. Replaying that endpoint reproduces the original
    request more exactly than reconstructing a command name would.
    """
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def replay_used_auth(url):
    """True when the recorded request carried the key parameter."""
    return KEY_PARAM in parse_qs(urlsplit(url).query, keep_blank_values=True)


def span_of_parquet(path):
    """The date span a banked capture actually returned, or None if undated."""
    if not path or not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if "date" not in df.columns:
        return None
    dates = pd.to_datetime(pd.Series(df["date"]).astype(str), errors="coerce").dropna()
    if not len(dates):
        return None
    return [dates.min().strftime("%Y-%m-%d"), dates.max().strftime("%Y-%m-%d")]


def classify_replay_shortfall(check, source_span, tolerance=MAX_EDGE_GAP_DAYS,
                              rolloff_days=0):
    """Judge a replay shortfall against what the source snapshot itself received.

    Halting again on a clip the source already carried measures nothing; a span
    shorter than last time's is the regression worth stopping for. rolloff_days
    budgets the start edge for the vendor's rolling retention window. Returns a
    dict whose "verdict" is "ok", "REPRODUCED_CLIP", "ROLLOFF" or "REGRESSION".
    """
    if check.get("status") != "SHORTFALL":
        return {"verdict": "ok"}
    got = check.get("returned") or [None, None]
    if not source_span:
        return {"verdict": "REGRESSION",
                "reason": "shortfall with no source span to compare against",
                "replay_span": list(got)}
    if got[0] is None or got[1] is None:
        return {"verdict": "REGRESSION", "reason": "no dated rows returned",
                "source_span": list(source_span), "replay_span": list(got)}
    start_short = int((pd.to_datetime(got[0]) - pd.to_datetime(source_span[0])).days)
    end_short = int((pd.to_datetime(source_span[1]) - pd.to_datetime(got[1])).days)
    start_budget = max(tolerance, rolloff_days)
    worse = [k for k, v, lim in (("start", start_short, start_budget),
                                 ("end", end_short, tolerance)) if v > lim]
    out = {
        "source_span": list(source_span),
        "replay_span": list(got),
        "start_days_short_vs_source": start_short,
        "end_days_short_vs_source": end_short,
        "tolerance_days": tolerance,
        "rolloff_days_allowed": rolloff_days,
    }
    if worse:
        out["verdict"] = "REGRESSION"
        out["reason"] = "replay span is shorter than the source at: %s" % ",".join(worse)
    elif start_short > tolerance:
        out["verdict"] = "ROLLOFF"
        out["reason"] = ("start edge receded %d day(s), within the %d day(s) of "
                         "retention allowed" % (start_short, rolloff_days))
    else:
        out["verdict"] = "REPRODUCED_CLIP"
    return out


def is_derived_entry(entry):
    """A derived row is computed from other entries, never fetched."""
    return entry.get("url_redacted") is None or entry.get("derived_from") is not None


def assert_label_is_new(root, label):
    """Refuse to replay into a label that already has anything on disk."""
    snap = os.path.join(root, "data", "sharadar", "snapshot_%s" % label)
    budget = os.path.join(root, "logs", "budget_%s.json" % label)
    for path, what in ((snap, "snapshot directory"), (budget, "budget file")):
        if os.path.exists(path):
            raise SystemExit(
                "REFUSED: %s already exists for label %s (%s); "
                "a replay must write a virgin label" % (what, label, path))


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_api_key(env_path, var_name):
    """Read one KEY=VALUE from a .env-style file given by path. Never logged."""
    with open(env_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(var_name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("key variable not found in supplied env path")


class Probe:
    def __init__(self, label, api_key, base_url):
        self.label = label
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.snap_dir = os.path.join(REPO_ROOT, "data", "sharadar", "snapshot_%s" % label)
        self.raw_dir = os.path.join(self.snap_dir, "raw")
        self.log_path = os.path.join(REPO_ROOT, "logs", "sharadar_probe_%s.log" % label)
        self.budget_path = os.path.join(REPO_ROOT, "logs", "budget_%s.json" % label)
        self.manifest_path = os.path.join(self.snap_dir, "manifest.json")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    # ---- budget -----------------------------------------------------------
    def _budget(self):
        if os.path.exists(self.budget_path):
            with open(self.budget_path) as fh:
                return json.load(fh)
        return {"used": 0}

    def _spend(self):
        b = self._budget()
        if b["used"] >= REQUEST_BUDGET:
            raise SystemExit("BUDGET EXHAUSTED: %d/%d requests used" % (b["used"], REQUEST_BUDGET))
        b["used"] += 1
        with open(self.budget_path, "w") as fh:
            json.dump(b, fh)
        return b["used"]

    def budget_used(self):
        return self._budget()["used"]

    # ---- logging ----------------------------------------------------------
    def redact(self, text):
        if not text:
            return text
        out = text.replace(self.api_key, "<REDACTED>")
        out = re.sub(r"(api_key|apikey|api-key|token)=[^&\s]+", r"\1=<REDACTED>", out, flags=re.I)
        return out

    def log(self, msg):
        line = "%s %s" % (utc_now(), self.redact(str(msg)))
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        print(line)

    # ---- http -------------------------------------------------------------
    def request(self, url, params=None, auth=True, tag="misc"):
        params = dict(params or {})
        if auth:
            params[KEY_PARAM] = self.api_key
        n = self._spend()
        attempt = 0
        while True:
            attempt += 1
            t0 = time.time()
            try:
                resp = requests.get(url, params=params, timeout=60,
                                    headers={"User-Agent": "StageR-probe/1.0"})
            except Exception as exc:
                self.log("REQ#%d %s GET %s EXC %s" % (n, tag, self.redact(url), type(exc).__name__))
                time.sleep(SLEEP_BETWEEN_S)
                return None
            ms = int((time.time() - t0) * 1000)
            rl = {k: v for k, v in resp.headers.items()
                  if any(h in k.lower() for h in RATE_HEADER_HINTS)}
            self.log("REQ#%d %s GET %s -> %d %dms %dB ct=%s rl=%s"
                     % (n, tag, self.redact(resp.url), resp.status_code, ms,
                        len(resp.content), resp.headers.get("Content-Type", ""), json.dumps(rl)))
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                if attempt == 1:
                    self.log("REQ#%d retrying after %ds" % (n, RETRY_WAIT_S))
                    time.sleep(RETRY_WAIT_S)
                    n = self._spend()
                    continue
                self.log("REQ#%d FAILED after retry, moving on" % n)
            time.sleep(SLEEP_BETWEEN_S)
            return resp

    # ---- persistence ------------------------------------------------------
    def save_raw(self, name, content_bytes, ext):
        path = os.path.join(self.raw_dir, "%s.%s" % (name, ext))
        with open(path, "wb") as fh:
            fh.write(content_bytes)
        return path

    def to_frame(self, resp):
        """Parse a response into a DataFrame; returns (df, ext) or (None, ext)."""
        ct = (resp.headers.get("Content-Type") or "").lower()
        text = resp.text
        if "json" in ct or text.lstrip()[:1] in "[{":
            try:
                obj = resp.json()
            except Exception:
                return None, "json"
            df = None
            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                for key in ("data", "rows", "results", "datatable", "records"):
                    if key in obj:
                        sub = obj[key]
                        if isinstance(sub, dict) and "data" in sub:
                            cols = [c["name"] for c in sub.get("columns", [])] or None
                            df = pd.DataFrame(sub["data"], columns=cols)
                        elif isinstance(sub, list):
                            df = pd.DataFrame(sub)
                        break
                if df is None:
                    df = pd.DataFrame([obj])
            return df, "json"
        if "csv" in ct or ("," in text[:200] and "\n" in text[:2000] and "<html" not in text[:200].lower()):
            try:
                return pd.read_csv(io.StringIO(text)), "csv"
            except Exception:
                return None, "csv"
        return None, "html" if "<html" in text[:400].lower() else "txt"

    def manifest_add(self, entry):
        items = []
        if os.path.exists(self.manifest_path):
            with open(self.manifest_path) as fh:
                items = json.load(fh)
        items.append(entry)
        with open(self.manifest_path, "w") as fh:
            json.dump(items, fh, indent=2)

    def capture(self, name, url, params, tag=None, auth=True, write_parquet=True,
                halt_on_shortfall=True):
        resp = self.request(url, params=params, auth=auth, tag=tag or name)
        if resp is None:
            return None, None
        df, ext = self.to_frame(resp)
        check = check_date_range(params, df)
        raw_path = self.save_raw(name, resp.content, ext)
        pq_path = None
        pq_sha = None
        if write_parquet and df is not None and len(df.columns):
            pq_path = os.path.join(self.snap_dir, "%s.parquet" % name)
            try:
                df.astype(str).to_parquet(pq_path, index=False)
                pq_sha = sha256_file(pq_path)
            except Exception as exc:
                self.log("parquet write failed for %s: %s" % (name, type(exc).__name__))
                pq_path = None
        self.manifest_add({
            "name": name,
            "url_redacted": self.redact(resp.url),
            "params": {k: v for k, v in (params or {}).items()},
            "http_status": resp.status_code,
            "rows": int(len(df)) if df is not None else None,
            "columns": list(df.columns) if df is not None else None,
            "raw_file": os.path.relpath(raw_path, self.snap_dir),
            "raw_sha256": sha256_file(raw_path),
            "raw_bytes": len(resp.content),
            "parquet_file": os.path.relpath(pq_path, self.snap_dir) if pq_path else None,
            "parquet_sha256": pq_sha,
            "fetched_at_utc": utc_now(),
            "date_range_check": check,
        })
        if check.get("status") == "SHORTFALL" and halt_on_shortfall:
            self.log("DATE RANGE SHORTFALL on %s: requested %s..%s, returned %s..%s (%s)" % (
                name, check["requested"][0], check["requested"][1],
                check["returned"][0], check["returned"][1], check.get("reason")))
            print("HALT: date range shortfall on %s" % name)
            print("  requested: %s .. %s" % (check["requested"][0], check["requested"][1]))
            print("  returned:  %s .. %s" % (check["returned"][0], check["returned"][1]))
            print("  gaps_days: %s (tolerance %s)" % (check.get("gaps_days"),
                                                      check.get("tolerance_days")))
            raise SystemExit(2)
        return resp, df

    # ---- tier check -------------------------------------------------------
    def tier_check(self):
        """Issue the tier probes and return (verdict_dict, per-probe detail).

        Every probe is captured and manifested like any other request, so the
        evidence for the verdict is banked rather than merely printed.
        """
        results = []
        detail = []
        for spec in TIER_PROBES:
            url = "%s/%s" % (self.base_url, spec["path"])
            resp, df = self.capture(spec["name"], url, dict(spec["params"]),
                                    tag=spec["name"])
            status = None if resp is None else resp.status_code
            rows = None if df is None else int(len(df))
            body = "" if resp is None else self.redact(resp.text[:400])
            results.append({"name": spec["name"], "paid_gated": spec["paid_gated"],
                            "http_status": status, "rows": rows})
            detail.append({
                "name": spec["name"],
                "url": "%s?%s" % (url, "&".join("%s=%s" % kv for kv in spec["params"].items())),
                "paid_gated": spec["paid_gated"],
                "free_tier_observed": spec["free_tier_observed"],
                "http_status": status,
                "rows": rows,
                "columns": None if df is None else list(df.columns),
                "body_first_400": body,
                "plan_headers": {} if resp is None else plan_headers(resp.headers),
                "all_headers": {} if resp is None else dict(resp.headers),
            })
        return classify_tier(results), detail

    # ---- bulk downloads ---------------------------------------------------
    def bulk_table(self, table, years=BULK_YEARS, run_date=None,
                   assert_full_history=False):
        """Pull one whole table over the bulk route and bank it.

        Two hops: the data route answers 302 with a pre-signed location, which is
        then streamed straight into the snapshot directory — no staging anywhere
        else, because the snapshot lives on the external volume and the boot disk
        has no room for a full-history archive. The archive is banked exactly as
        delivered; the parquet beside it is a rendering, not the record.
        """
        name = "bulk_%s" % table
        url = "%s/data/%s" % (self.base_url, table)
        params = {"years": years}
        n = self._spend()
        t0 = time.time()
        resp = requests.get(url, params=dict(params, **{KEY_PARAM: self.api_key}),
                            timeout=120, allow_redirects=False,
                            headers={"User-Agent": "StageR-probe/1.0"})
        location = resp.headers.get("Location")
        ct = resp.headers.get("Content-Type", "")
        verdict = classify_bulk_response(resp.status_code, location, ct)
        self.log("BULK#%d %s GET %s?years=%s -> %d %s ct=%s loc=%s"
                 % (n, table, self.redact(url), years, resp.status_code,
                    verdict["verdict"], ct, redact_url_query(location)))

        entry = {
            "name": name,
            "url_redacted": self.redact(resp.url),
            "params": dict(params),
            "http_status": resp.status_code,
            "bulk_verdict": verdict["verdict"],
            "bulk_verdict_reason": verdict["reason"],
            "redirect_location_redacted": redact_url_query(location),
            "snapshot_only_table": table in BULK_SNAPSHOT_ONLY,
            "fetched_at_utc": utc_now(),
        }

        if verdict["verdict"] == "FORBIDDEN":
            body = self.redact(resp.text[:400])
            self.log("BULK %s REFUSED: %s" % (table, body))
            entry.update({"rows": None, "columns": None, "raw_file": None,
                          "raw_sha256": None, "raw_bytes": None,
                          "parquet_file": None, "parquet_sha256": None,
                          "error_body_first_400": body})
            self.manifest_add(entry)
            return entry
        if verdict["verdict"] == "UNEXPECTED":
            body = self.redact(resp.text[:400])
            self.log("BULK %s UNEXPECTED: %s" % (table, verdict["reason"]))
            entry.update({"rows": None, "columns": None, "raw_file": None,
                          "raw_sha256": None, "raw_bytes": None,
                          "parquet_file": None, "parquet_sha256": None,
                          "error_body_first_400": body})
            self.manifest_add(entry)
            return entry

        raw_rel = os.path.join("raw", "%s.zip" % name)
        raw_path = os.path.join(self.snap_dir, raw_rel)
        if verdict["verdict"] == "DIRECT_ZIP":
            with open(raw_path, "wb") as fh:
                fh.write(resp.content)
        else:
            n2 = self._spend()
            with requests.get(location, stream=True, timeout=600) as dl:
                dl.raise_for_status()
                with open(raw_path, "wb") as fh:
                    for chunk in dl.iter_content(1 << 22):
                        if chunk:
                            fh.write(chunk)
            self.log("BULK#%d %s downloaded %d bytes from the pre-signed location"
                     % (n2, table, os.path.getsize(raw_path)))
        time.sleep(SLEEP_BETWEEN_S)

        pq_rel = "%s.parquet" % name
        stats = zip_member_to_parquet(raw_path, os.path.join(self.snap_dir, pq_rel))
        secs = time.time() - t0
        self.log("BULK %s -> %s rows=%d chunks=%d zip=%dB parquet=%dB %.1fs"
                 % (table, stats["member"], stats["rows"], stats["chunks"],
                    os.path.getsize(raw_path),
                    os.path.getsize(os.path.join(self.snap_dir, pq_rel)), secs))

        span = self.parquet_date_span(pq_rel)
        check = {"status": "not_applicable", "reason": "no full-history assertion asked"}
        if assert_full_history:
            check = check_full_history_range(span[0], span[1],
                                             run_date or utc_now()[:10])
        entry.update({
            "rows": stats["rows"],
            "columns": stats["columns"],
            "zip_member": stats["member"],
            "zip_members": stats["members"],
            "raw_file": raw_rel,
            "raw_sha256": sha256_file(raw_path),
            "raw_bytes": os.path.getsize(raw_path),
            "parquet_file": pq_rel,
            "parquet_sha256": sha256_file(os.path.join(self.snap_dir, pq_rel)),
            "date_span": list(span),
            "full_history_check": check,
            "wall_seconds": round(secs, 1),
        })
        self.manifest_add(entry)
        if check.get("status") == "SHORTFALL":
            print("HALT: %s is not full history — %s" % (table, check.get("reason")))
            raise SystemExit(2)
        return entry

    def parquet_date_span(self, rel):
        """Earliest and latest date in a banked parquet, read one column only."""
        import pyarrow.parquet as pq
        path = os.path.join(self.snap_dir, rel)
        cols = pq.ParquetFile(path).schema_arrow.names
        date_col = "date" if "date" in cols else None
        if date_col is None:
            return (None, None)
        s = pd.to_datetime(
            pq.read_table(path, columns=[date_col])[date_col].to_pandas(),
            errors="coerce").dropna()
        if not len(s):
            return (None, None)
        return (s.min().strftime("%Y-%m-%d"), s.max().strftime("%Y-%m-%d"))

    # ---- replay -----------------------------------------------------------
    def _manifest_parquets(self):
        """name -> parquet_file for what this snapshot has written so far."""
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path) as fh:
            return {e["name"]: e.get("parquet_file") for e in json.load(fh)}

    def replay_derived(self, entry):
        """Rebuild a derived row from the entries it names. Spends no budget."""
        written = self._manifest_parquets()
        frames = []
        for src in entry.get("derived_from") or []:
            rel = written.get(src) or ("%s.parquet" % src)
            path = os.path.join(self.snap_dir, rel)
            if not os.path.exists(path):
                raise SystemExit(
                    "HALT: derived entry %s needs %s, which this replay did not produce"
                    % (entry["name"], rel))
            frames.append(pd.read_parquet(path))
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        key = [c for c in ("ticker", "date") if c in df.columns]
        if len(key) == 2:
            df = df.drop_duplicates(subset=key, keep="first")
        rel = entry.get("parquet_file") or ("%s.parquet" % entry["name"])
        pq_path = os.path.join(self.snap_dir, rel)
        df.astype(str).to_parquet(pq_path, index=False)
        self.manifest_add({
            "name": entry["name"],
            "derived_from": list(entry.get("derived_from") or []),
            "url_redacted": None,
            "params": dict(entry.get("params") or {}),
            "http_status": None,
            "rows": int(len(df)),
            "columns": list(df.columns),
            "raw_file": None,
            "raw_sha256": None,
            "raw_bytes": None,
            "parquet_file": rel,
            "parquet_sha256": sha256_file(pq_path),
            "fetched_at_utc": None,
        })
        self.log("DERIVED %s from %s -> %d rows"
                 % (entry["name"], ",".join(entry.get("derived_from") or []), len(df)))
        return df

    def replay_manifest(self, entries, source_dir=None, resume=False, rolloff_days=0):
        """Re-issue every entry of a manifest, in order, under this label."""
        summary = []
        done = set(self._manifest_parquets()) if resume else set()
        if done:
            self.log("RESUME: %d entries already present, skipping them" % len(done))
        for i, entry in enumerate(entries):
            name = entry["name"]
            params = dict(entry.get("params") or {})
            if name in done:
                summary.append({"name": name, "kind": "skipped (already present)",
                                "http_status": None, "rows": None,
                                "was_status": entry.get("http_status"),
                                "was_rows": entry.get("rows")})
                continue
            if is_derived_entry(entry):
                df = self.replay_derived(entry)
                summary.append({"name": name, "kind": "derived", "http_status": None,
                                "rows": int(len(df)), "was_status": entry.get("http_status"),
                                "was_rows": entry.get("rows")})
                continue
            url = endpoint_of(entry["url_redacted"])
            auth = replay_used_auth(entry["url_redacted"])
            self.log("REPLAY %d/%d %s -> %s (auth=%s)"
                     % (i + 1, len(entries), name, url, auth))
            resp, df = self.capture(name, url, params, tag=name, auth=auth,
                                    halt_on_shortfall=False)
            row = {
                "name": name, "kind": "capture",
                "http_status": None if resp is None else resp.status_code,
                "rows": None if df is None else int(len(df)),
                "was_status": entry.get("http_status"), "was_rows": entry.get("rows"),
            }
            check = check_date_range(params, df)
            if check.get("status") == "SHORTFALL":
                src = os.path.join(source_dir, entry["parquet_file"]) \
                    if source_dir and entry.get("parquet_file") else None
                verdict = classify_replay_shortfall(check, span_of_parquet(src),
                                                    rolloff_days=rolloff_days)
                row["shortfall"] = verdict
                if verdict["verdict"] == "REGRESSION":
                    self.log("REPLAY REGRESSION on %s: %s" % (name, json.dumps(verdict)))
                    print("HALT: replay span regressed on %s" % name)
                    print("  source span: %s" % verdict.get("source_span"))
                    print("  replay span: %s" % verdict.get("replay_span"))
                    print("  %s" % verdict.get("reason"))
                    raise SystemExit(2)
                self.log("%s on %s: source %s, replay %s - allowed through"
                         % (verdict["verdict"], name, verdict.get("source_span"),
                            verdict.get("replay_span")))
            summary.append(row)
        return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--env-path", required=True)
    ap.add_argument("--env-var", default="SHARADAR_API_KEY")
    ap.add_argument("--base-url", default="https://api.sharadar.com/v1.0")
    ap.add_argument("--cmd", default=None)
    ap.add_argument("--replay", default=None,
                    help="path to a manifest.json to re-issue under --snapshot")
    ap.add_argument("--replay-source-dir", default=None,
                    help="snapshot directory the manifest came from; enables "
                         "adjudicating a shortfall against the span it received")
    ap.add_argument("--replay-allow-rolloff", type=int, default=0,
                    help="days of start-edge loss attributable to the vendor's "
                         "rolling window; normally the calendar days between fetches")
    ap.add_argument("--replay-resume", action="store_true",
                    help="continue a replay into an existing label, skipping "
                         "entries already recorded in its manifest")
    ap.add_argument("--url", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--noauth", action="store_true")
    ap.add_argument("--bulk-tables", default=",".join(BULK_TABLES),
                    help="comma-separated tables to pull over the bulk route")
    ap.add_argument("--bulk-years", default=BULK_YEARS)
    ap.add_argument("--assert-full-history", default="stocks",
                    help="comma-separated tables whose returned span must reach "
                         "full history; a short span halts")
    ap.add_argument("--run-date", default=None,
                    help="date the full-history end edge is measured against "
                         "(default: today, UTC)")
    args = ap.parse_args()

    if bool(args.cmd) == bool(args.replay):
        raise SystemExit("supply exactly one of --cmd or --replay")

    key = read_api_key(args.env_path, args.env_var)

    if args.replay:
        if not args.replay_resume:
            assert_label_is_new(REPO_ROOT, args.snapshot)
        with open(args.replay, "r", encoding="utf-8") as fh:
            entries = json.load(fh)
        p = Probe(args.snapshot, key, args.base_url)
        p.log("REPLAY of %s (%d entries) into snapshot %s"
              % (os.path.basename(args.replay), len(entries), args.snapshot))
        summary = p.replay_manifest(entries, source_dir=args.replay_source_dir,
                                    resume=args.replay_resume,
                                    rolloff_days=args.replay_allow_rolloff)
        report = {
            "replayed_from": os.path.abspath(args.replay),
            "replayed_from_sha256": sha256_file(args.replay),
            "snapshot": args.snapshot,
            "replay_source_dir": args.replay_source_dir,
            "resumed": bool(args.replay_resume),
            "rolloff_days_allowed": args.replay_allow_rolloff,
            "entries": len(entries),
            "requests_used": p.budget_used(),
            "request_budget": REQUEST_BUDGET,
            "completed_at_utc": utc_now(),
            "captures": summary,
        }
        out = os.path.join(REPO_ROOT, "logs", "replay_report_%s.json" % args.snapshot)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print("REPLAY COMPLETE: %d entries, %d/%d requests used"
              % (len(entries), p.budget_used(), REQUEST_BUDGET))
        print("report %s" % os.path.relpath(out, REPO_ROOT))
        return

    p = Probe(args.snapshot, key, args.base_url)

    if args.cmd == "tier-check":
        verdict, detail = p.tier_check()
        report = {
            "snapshot": args.snapshot,
            "base_url": p.base_url,
            "api_key_length": len(key),
            "checked_at_utc": utc_now(),
            "verdict": verdict["verdict"],
            "reason": verdict["reason"],
            "probes": detail,
            "requests_used": p.budget_used(),
            "request_budget": REQUEST_BUDGET,
        }
        out = os.path.join(REPO_ROOT, "logs", "tier_check_%s.json" % args.snapshot)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        for d in detail:
            print("%-20s gated=%-5s -> %s rows=%s  (free tier: %s)"
                  % (d["name"], d["paid_gated"], d["http_status"], d["rows"],
                     d["free_tier_observed"]))
            if d["http_status"] != 200:
                print("    BODY: %s" % d["body_first_400"])
            if d["plan_headers"]:
                print("    PLAN HEADERS: %s" % json.dumps(d["plan_headers"]))
        print("VERDICT: %s - %s" % (verdict["verdict"], verdict["reason"]))
        print("report %s" % os.path.relpath(out, REPO_ROOT))
        if verdict["verdict"] != "PAID":
            raise SystemExit(3)
        return

    if args.cmd == "bulk":
        tables = [t.strip() for t in args.bulk_tables.split(",") if t.strip()]
        must_span = {t.strip() for t in (args.assert_full_history or "").split(",") if t.strip()}
        run_date = args.run_date or utc_now()[:10]
        p.log("BULK PULL %s years=%s run_date=%s (full-history asserted on: %s)"
              % (",".join(tables), args.bulk_years, run_date,
                 ",".join(sorted(must_span)) or "none"))
        results = []
        for table in tables:
            results.append(p.bulk_table(table, years=args.bulk_years,
                                        run_date=run_date,
                                        assert_full_history=table in must_span))
        report = {
            "snapshot": args.snapshot,
            "years": args.bulk_years,
            "run_date": run_date,
            "tables": tables,
            "full_history_asserted": sorted(must_span),
            "completed_at_utc": utc_now(),
            "requests_used": p.budget_used(),
            "request_budget": REQUEST_BUDGET,
            "entries": results,
        }
        out = os.path.join(REPO_ROOT, "logs", "bulk_report_%s.json" % args.snapshot)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        for e in results:
            print("%-16s %-11s status=%s rows=%s zip=%sB span=%s full_history=%s"
                  % (e["name"], e["bulk_verdict"], e["http_status"], e.get("rows"),
                     e.get("raw_bytes"), e.get("date_span"),
                     (e.get("full_history_check") or {}).get("status")))
        refused = [e["name"] for e in results if e["bulk_verdict"] != "REDIRECT"
                   and e["bulk_verdict"] != "DIRECT_ZIP"]
        if refused:
            print("FINDING: bulk refused for %s" % ",".join(refused))
        print("report %s" % os.path.relpath(out, REPO_ROOT))
        return

    if args.cmd == "budget":
        print(json.dumps({"used": p.budget_used(), "budget": REQUEST_BUDGET}))
        return

    if args.cmd == "schema":
        target = "%s/schema" % p.base_url if not args.table else "%s/schema/%s" % (p.base_url, args.table)
        nm = args.name or ("schema_index" if not args.table else "schema_%s" % args.table)
        resp, df = p.capture(nm, target, json.loads(args.params), auth=False)
        if resp is not None:
            print("STATUS", resp.status_code, "BYTES", len(resp.content))
        return

    if args.cmd == "table":
        target = "%s/data/%s" % (p.base_url, args.table)
        nm = args.name or ("table_%s" % args.table)
        resp, df = p.capture(nm, target, json.loads(args.params), auth=not args.noauth)
        if resp is not None:
            print("STATUS", resp.status_code, "BYTES", len(resp.content))
            if df is not None:
                print("ROWS", len(df))
                print("COLS", list(df.columns))
            else:
                print("BODY", p.redact(resp.text[:400]))
        return

    if args.cmd == "geturl":
        resp, df = p.capture(args.name, args.url, json.loads(args.params),
                             auth=not args.noauth)
        if resp is not None:
            print("STATUS", resp.status_code, "BYTES", len(resp.content))
            if df is not None:
                print("ROWS", len(df))
                print("COLS", list(df.columns))
        return

    raise SystemExit("unknown cmd")


if __name__ == "__main__":
    main()
