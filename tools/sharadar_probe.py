"""
================================================================================
Script:       sharadar_probe.py
Version:      1.1.0
Phase:        Stage R — External Data Evaluation
Purpose:      Read-only probe of the Sharadar free tier: surface discovery, table
              sampling, and snapshot capture with sha256 manifesting.
Entry Point:  Yes — python tools/sharadar_probe.py --snapshot <label> <command>
Dependencies: stdlib, requests, pandas, pyarrow. Imports nothing from any other
              repository.
Last Sprint:  Sprint R0-B-0 — Snapshot Diff Tool; probe date-range assert (2026-09-17)
================================================================================

The API key is read at runtime from a .env-style file whose path is supplied by
--env-path; it is never written to disk, logs, or stdout. Every HTTP request is
budget-counted (shared across invocations of one snapshot label), rate-limited,
logged with the key redacted, and saved raw beside a parquet rendering.

Every request that names a date window has its returned span checked against the
window it asked for: the vendor clips a long request to about a year and returns
HTTP 200, so a shortfall halts the probe with both ranges printed rather than
banking a silently partial snapshot.
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

    def capture(self, name, url, params, tag=None, auth=True, write_parquet=True):
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
        if check.get("status") == "SHORTFALL":
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--env-path", required=True)
    ap.add_argument("--env-var", default="SHARADAR_API_KEY")
    ap.add_argument("--base-url", default="https://api.sharadar.com/v1.0")
    ap.add_argument("--cmd", required=True)
    ap.add_argument("--url", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--table", default=None)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--noauth", action="store_true")
    args = ap.parse_args()

    key = read_api_key(args.env_path, args.env_var)
    p = Probe(args.snapshot, key, args.base_url)

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
