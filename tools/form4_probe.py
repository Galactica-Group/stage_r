"""
================================================================================
Script:       form4_probe.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Build the table of open-market purchases by officers and directors
              from the SEC's Insider Transactions (Form 3/4/5) bulk data sets,
              and report counts, dates, sizes and matching only.
Entry Point:  Yes — python tools/form4_probe.py {archives,fetch,build,join,report}
Dependencies: stdlib, requests, pandas, pyarrow, numpy. Shares the rate-limited
              session, contact resolver and manifest helpers of
              edgar_item408_probe by import. Imports nothing from any other
              repository.
Last Sprint:  Sprint R6-0 — Form 4 bulk dataset and inventory (2026-11-05)
================================================================================

No credential is read, sent, or required; no .env file is opened. The contact
string the SEC requires in the User-Agent is resolved by the shared
read_contact, never printed, and reported by byte length only. Every request to
a sec.gov host goes through one sliding-window limiter capped at six per second
with a single worker; gzip is accepted; 429/403 and the transient 5xx family are
retried three times.

A purchase is a transaction that clears every clause of keep_purchase: the
submission is a Form 4 and not an amendment, the transaction sits in
NONDERIV_TRANS, its TRANS_CODE is P, its acquired/disposed code is A, and its
price per share parses to a positive number. Code P is the open-market purchase
code; A alone would also admit grants and exercises, and P alone would admit the
rare disposal-side P row. A price that is blank, zero or negative is dropped
rather than imputed: a purchase with no price has no size, and this tool forms no
size it cannot read off the filing.

RPTOWNER_RELATIONSHIP is a comma-joined set, not a single value: "Director,
Officer,TenPercentOwner" is one owner wearing three hats. The four flags are
resolved separately and a row may carry several. The filing-level gate is that
at least one reporting owner on the accession is a Director or an Officer; the
owner columns name the primary owner (the lowest reporting-owner CIK on the
accession, a deterministic choice on a joint filing) and owners.parquet keeps
one row per reporting owner per accession, so the full structure of a joint
filing is recoverable from the artifact.

AFF10B5ONE, at submission level, is the only Rule 10b5-1 field the data sets
carry. It is stored as filed and normalised separately: the dataset writes both
"1"/"0" and "true"/"false" in different vintages, and blank where the filer left
the box untouched. No plan adoption date is a structured field anywhere in these
archives; it appears, when at all, in free-text remarks and footnotes, which
this tool stores nothing of and classifies nothing from.

This tool forms no return, no price change and no abnormal return. Where a price
row is consulted it is counted, never read: presence on a date is a count.
================================================================================
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from edgar_item408_probe import (  # noqa: E402
    RateLimiter,
    SecSession,
    StopRun,
    build_cik_map,
    cik_from_secfilings,
    load_manifest,
    read_contact,
    say,
    sha256_bytes,
    sha256_file,
    write_manifest,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Found by walking sec.gov from /data-research to the DERA data library to
# /data-research/sec-markets-data/insider-transactions-data-sets. Archives are
# named <YYYY>q<Q>_form345.zip and are served from one of two /files prefixes.
INSIDER_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/"
    "insider-transactions-data-sets"
)
SEC_HOST = "https://www.sec.gov"

# The brief's fair-access cap for this sprint.
MAX_REQUESTS_PER_SECOND = 6

ARCHIVE_RE = re.compile(r"^(\d{4})q([1-4])_form345\.zip$")

# The member files this tool reads. The rest of the archive (footnotes,
# signatures, holdings, derivative rows) is left unread.
MEMBER_SUBMISSION = "SUBMISSION.tsv"
MEMBER_OWNER = "REPORTINGOWNER.tsv"
MEMBER_NONDERIV = "NONDERIV_TRANS.tsv"

# The open-market purchase code and the acquired side.
PURCHASE_CODE = "P"
ACQUIRED_CODE = "A"

# The four relationship hats, spelled as the data sets spell them.
RELATIONSHIPS = ("director", "officer", "ten_percent_owner", "other")
_REL_PATTERNS = (
    ("director", re.compile(r"DIRECTOR")),
    ("officer", re.compile(r"OFFICER")),
    ("ten_percent_owner", re.compile(r"TENPERCENTOWNER")),
    ("other", re.compile(r"OTHER")),
)

TRUE_TOKENS = ("1", "true", "y", "yes")
FALSE_TOKENS = ("0", "false", "n", "no")

# The Sharadar calendar floor, the same one the R5 look used.
CALENDAR_MIN_TICKERS = 1000

# The halves this sprint proposes. Counts only; nothing is scored here.
HALF_A = ("2006-01-01", "2015-12-31")
HALF_B = ("2016-01-01", "2022-12-31")
HOLDOUT_START = "2023-01-01"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def raw_dir(root=None):
    return os.path.join(root or REPO_ROOT, "data", "form4", "raw")


def vintage_dir(vintage, root=None):
    return os.path.join(root or REPO_ROOT, "data", "form4", str(vintage))


def manifest_path(vintage, root=None):
    return os.path.join(vintage_dir(vintage, root), "manifest.json")


def tracked_manifest_path(vintage, root=None):
    return os.path.join(root or REPO_ROOT, "manifests", "form4_%s.json" % vintage)


def fetch_manifest_path(root=None):
    return os.path.join(raw_dir(root), "fetch_manifest.json")


def log_path(name, root=None):
    return os.path.join(root or REPO_ROOT, "logs", "r6", name)


def session(max_per_sec=MAX_REQUESTS_PER_SECOND):
    contact, source = read_contact()
    say("contact: source %s, byte length %d" % (source, len(contact.encode())))
    return SecSession(contact, limiter=RateLimiter(max_per_sec=max_per_sec))


# ---------------------------------------------------------------------------
# Pure helpers: archive names
# ---------------------------------------------------------------------------

def parse_archive_name(name):
    """Return (year, quarter) for an insider-transactions archive name, else None."""
    m = ARCHIVE_RE.match(str(name).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def archive_sort_key(name):
    yq = parse_archive_name(name)
    if yq is None:
        raise ValueError("not an insider-transactions archive name: %r" % (name,))
    return yq


def extract_archive_links(html):
    """Every insider-transactions archive on the page, oldest first.

    Returns a list of {"name", "href"}. Only .zip hrefs whose basename matches
    the archive naming are kept, so navigation and taxonomy links are ignored.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    seen = {}
    for m in re.finditer(r'href="([^"]+\.zip)"', html, re.I):
        href = m.group(1).strip()
        name = href.rsplit("/", 1)[-1]
        if parse_archive_name(name) is None:
            continue
        seen.setdefault(name, href)
    return [
        {"name": n, "href": seen[n]} for n in sorted(seen, key=archive_sort_key)
    ]


def archive_url(href):
    """Absolute URL for an archive href, which the page gives host-relative."""
    s = str(href).strip()
    if s.startswith("http://") or s.startswith("https://"):
        if not s.startswith(SEC_HOST):
            raise StopRun("refusing a non-sec.gov archive URL")
        return s
    if not s.startswith("/"):
        raise StopRun("archive href is neither absolute nor host-relative")
    return SEC_HOST + s


# ---------------------------------------------------------------------------
# Pure helpers: fields
# ---------------------------------------------------------------------------

def normalise_flag(value):
    """True, False or None for a data-set boolean written in any of its spellings.

    The archives write "1"/"0" in some vintages and "true"/"false" in others,
    and leave the field blank where the filer touched nothing. Blank and
    anything unrecognised return None: an unreadable flag is not a False.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if s == "" or s in ("nan", "none"):
        return None
    if s in TRUE_TOKENS:
        return True
    if s in FALSE_TOKENS:
        return False
    return None


def parse_relationship(text):
    """The four relationship flags for one comma-joined RPTOWNER_RELATIONSHIP.

    A reporting owner may wear several hats at once, so the flags are resolved
    independently and more than one may be True. An empty or unreadable field
    returns all False.
    """
    s = "" if text is None else str(text).upper()
    s = re.sub(r"[^A-Z]", "", s)
    return {key: bool(pat.search(s)) for key, pat in _REL_PATTERNS}


def is_form4(document_type):
    """True only for a Form 4 that is not an amendment.

    Form 3 is an initial statement of holdings and Form 5 an annual catch-up;
    neither reports an open-market purchase at the moment it happened. A 4/A
    restates a filing already in the table, so counting it would double it.
    """
    return str(document_type).strip() == "4"


def parse_number(value):
    """A positive, finite float, or None. Commas and blanks are handled."""
    if value is None:
        return None
    s = str(value).strip().replace(",", "")
    if s == "":
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    if not np.isfinite(x):
        return None
    return x


def dollar_value(shares, price):
    """Shares times price per share, or None when either side is unreadable."""
    s = parse_number(shares)
    p = parse_number(price)
    if s is None or p is None:
        return None
    return s * p


def parse_date(value):
    """A date from the data sets' DD-MON-YYYY or an ISO date, else None."""
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none"):
        return None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%B-%Y", "%m/%d/%Y"):
        try:
            return pd.to_datetime(s, format=fmt).date()
        except (ValueError, TypeError):
            continue
    try:
        ts = pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        return None
    if ts is None or pd.isna(ts):
        return None
    return ts.date()


def filing_lag_days(filing_date, trans_date):
    """Filing date minus transaction date, in calendar days, or None.

    Calendar days, not trading days: the two-business-day rule is a deadline,
    and the lag reported here is the raw distance so a late filer is visible.
    """
    f = parse_date(filing_date)
    t = parse_date(trans_date)
    if f is None or t is None:
        return None
    return (f - t).days


def keep_purchase(submission_row, trans_row):
    """True when this non-derivative transaction is an open-market purchase.

    Every clause is load-bearing: Form 4 and not an amendment, code P, the
    acquired side, and a price that parses positive.
    """
    if not is_form4(submission_row.get("DOCUMENT_TYPE")):
        return False
    if str(trans_row.get("TRANS_CODE", "")).strip().upper() != PURCHASE_CODE:
        return False
    if str(trans_row.get("TRANS_ACQUIRED_DISP_CD", "")).strip().upper() != ACQUIRED_CODE:
        return False
    price = parse_number(trans_row.get("TRANS_PRICEPERSHARE"))
    if price is None or price <= 0:
        return False
    return True


def pad_cik(value):
    """A CIK as ten digits, or None."""
    if value is None:
        return None
    digits = re.sub(r"\D", "", str(value))
    if digits == "":
        return None
    return digits.zfill(10)


# ---------------------------------------------------------------------------
# Pure helpers: clustering, routine purchases
# ---------------------------------------------------------------------------

def calendar_index(calendar, days):
    """Position of each day on the trading calendar; -1 off-calendar.

    A filing day that is not a trading day is placed at the next trading day on
    or after it, so a Friday-evening filing and the Monday it reaches the market
    share one window.
    """
    cal = np.asarray(calendar)
    d = np.asarray(days)
    j = np.searchsorted(cal, d, "left")
    out = np.where(j < len(cal), j, -1)
    return out


def cluster_events(purchases, window_trading_days, calendar):
    """Group an issuer's purchases into events of at most N trading days.

    The first filing day of an issuer opens an event; every later purchase whose
    filing day sits within `window_trading_days` trading days of that opening
    day joins it; the first purchase beyond the window opens the next event.
    Returns a DataFrame with one row per event.

    Grouping on the filing day, not the transaction day, is deliberate: the
    filing is when the market could have known.
    """
    cols = ["issuer_cik", "filing_date", "owner_cik", "value"]
    for c in cols:
        if c not in purchases.columns:
            raise ValueError("purchases needs column %r" % c)
    if len(purchases) == 0:
        return pd.DataFrame(
            columns=[
                "issuer_cik", "event_start", "event_end", "n_transactions",
                "n_insiders", "total_value", "n_routine", "all_routine",
            ]
        )
    df = purchases.copy()
    df["_day"] = pd.to_datetime(df["filing_date"]).values.astype("datetime64[D]")
    df["_pos"] = calendar_index(calendar, df["_day"].values)
    df = df[df["_pos"] >= 0].sort_values(["issuer_cik", "_pos"], kind="mergesort")
    if "is_routine" not in df.columns:
        df["is_routine"] = False

    rows = []
    for issuer, g in df.groupby("issuer_cik", sort=True):
        open_pos = None
        bucket = []
        for rec in g.to_dict("records"):
            if open_pos is None or rec["_pos"] - open_pos >= int(window_trading_days):
                if bucket:
                    rows.append(_event_row(issuer, bucket))
                open_pos = rec["_pos"]
                bucket = [rec]
            else:
                bucket.append(rec)
        if bucket:
            rows.append(_event_row(issuer, bucket))
    return pd.DataFrame(rows)


def _event_row(issuer, bucket):
    days = [r["_day"] for r in bucket]
    vals = [r["value"] for r in bucket if r["value"] is not None and np.isfinite(r["value"])]
    routine = [bool(r.get("is_routine")) for r in bucket]
    return {
        "issuer_cik": issuer,
        "event_start": min(days),
        "event_end": max(days),
        "n_transactions": len(bucket),
        "n_insiders": len({r["owner_cik"] for r in bucket}),
        "total_value": float(sum(vals)) if vals else 0.0,
        "n_routine": int(sum(routine)),
        "all_routine": bool(routine) and all(routine),
    }


def routine_pairs(purchases, years_back=3):
    """Flag each purchase whose owner-issuer pair bought in the same calendar
    month in each of the `years_back` prior years.

    The test is a membership test on (owner, issuer, year, month) keys, so it
    looks only backwards and needs the prior years to be in the table: a
    purchase in the first three years of the vintage can never be routine, and
    is reported as such rather than imputed.
    """
    df = purchases.copy()
    if len(df) == 0:
        df["is_routine"] = pd.Series(dtype=bool)
        return df
    d = pd.to_datetime(df["trans_date"])
    df["_y"] = d.dt.year.astype("Int64")
    df["_m"] = d.dt.month.astype("Int64")
    keys = set(
        zip(df["owner_cik"], df["issuer_cik"], df["_y"].tolist(), df["_m"].tolist())
    )
    flags = []
    for owner, issuer, y, m in zip(
        df["owner_cik"], df["issuer_cik"], df["_y"].tolist(), df["_m"].tolist()
    ):
        if y is pd.NA or m is pd.NA or y is None or m is None:
            flags.append(False)
            continue
        flags.append(
            all((owner, issuer, int(y) - k, int(m)) in keys
                for k in range(1, int(years_back) + 1))
        )
    df["is_routine"] = flags
    return df.drop(columns=["_y", "_m"])


def trading_day_calendar(day, min_tickers=CALENDAR_MIN_TICKERS):
    """The dates carrying at least `min_tickers` rows, and the count statistics.

    The same floor the R5 look used: a date only a handful of tickers trade on
    is not a trading day, and counting a window across it would shift every
    window by a day the market never had.
    """
    day = np.asarray(day)
    uniq, cnt = np.unique(day, return_counts=True)
    cal = uniq[cnt >= int(min_tickers)]
    if len(cal) == 0:
        raise AssertionError(
            "no date in the snapshot carries %d tickers" % int(min_tickers))
    stats = {
        "min_tickers": int(min_tickers),
        "dates_in_snapshot": int(len(uniq)),
        "trading_days": int(len(cal)),
        "dates_excluded": int(len(uniq) - len(cal)),
    }
    return cal, stats


def match_issuer_cik(cik, cik_map):
    """(hits, multi_mapped) for one issuer CIK against the Sharadar CIK map."""
    key = pad_cik(cik)
    if key is None:
        return [], False
    hits = cik_map.get(key) or []
    multi = len({h["permaticker"] for h in hits}) > 1
    return hits, multi


def half_of(filing_date, half_a=HALF_A, half_b=HALF_B, holdout_start=HOLDOUT_START):
    """"A", "B", "holdout" or None for a filing date under the proposed halves."""
    d = parse_date(filing_date)
    if d is None:
        return None
    if date.fromisoformat(half_a[0]) <= d <= date.fromisoformat(half_a[1]):
        return "A"
    if date.fromisoformat(half_b[0]) <= d <= date.fromisoformat(half_b[1]):
        return "B"
    if d >= date.fromisoformat(holdout_start):
        return "holdout"
    return None


# ---------------------------------------------------------------------------
# Archive reading
# ---------------------------------------------------------------------------

def read_member(zf, name):
    """One member of an archive as all-string columns, blanks kept as blanks."""
    with zf.open(name) as fh:
        return pd.read_csv(
            io.BytesIO(fh.read()), sep="\t", dtype=str, keep_default_na=False,
            low_memory=False, on_bad_lines="warn",
        )


def purchases_from_archive(path, archive_name=None):
    """(purchases, owners, coverage) for one insider-transactions archive.

    purchases: one row per kept open-market purchase transaction.
    owners:    one row per reporting owner per kept accession.
    coverage:  the counts this archive contributed, for the coverage table.
    """
    name = archive_name or os.path.basename(path)
    with zipfile.ZipFile(path) as zf:
        sub = read_member(zf, MEMBER_SUBMISSION)
        own = read_member(zf, MEMBER_OWNER)
        nd = read_member(zf, MEMBER_NONDERIV)

    cov = {
        "archive": name,
        "submissions": int(len(sub)),
        "form4_submissions": int(sum(is_form4(x) for x in sub["DOCUMENT_TYPE"])),
        "nonderiv_rows": int(len(nd)),
        "reporting_owner_rows": int(len(own)),
    }

    sub4 = sub[[is_form4(x) for x in sub["DOCUMENT_TYPE"]]].copy()
    submeta = {
        r["ACCESSION_NUMBER"]: r for r in sub4.to_dict("records")
    }
    nd4 = nd[nd["ACCESSION_NUMBER"].isin(submeta)].copy()
    cov["nonderiv_rows_form4"] = int(len(nd4))

    kept_rows = []
    code_p_acquired = 0
    for t in nd4.to_dict("records"):
        s = submeta[t["ACCESSION_NUMBER"]]
        if (str(t.get("TRANS_CODE", "")).strip().upper() == PURCHASE_CODE
                and str(t.get("TRANS_ACQUIRED_DISP_CD", "")).strip().upper() == ACQUIRED_CODE):
            code_p_acquired += 1
        if keep_purchase(s, t):
            kept_rows.append((s, t))
    cov["code_p_acquired"] = code_p_acquired
    cov["code_p_acquired_priced"] = len(kept_rows)

    # Reporting owners for the accessions that survived the price clause.
    acc = {s["ACCESSION_NUMBER"] for s, _t in kept_rows}
    ow = own[own["ACCESSION_NUMBER"].isin(acc)].copy()
    owner_rows = []
    by_acc = collections.defaultdict(list)
    for r in ow.to_dict("records"):
        flags = parse_relationship(r.get("RPTOWNER_RELATIONSHIP"))
        rec = {
            "archive": name,
            "accession": r["ACCESSION_NUMBER"],
            "owner_cik": pad_cik(r.get("RPTOWNERCIK")),
            "owner_name": r.get("RPTOWNERNAME") or None,
            "relationship_as_filed": r.get("RPTOWNER_RELATIONSHIP") or None,
            "title": r.get("RPTOWNER_TITLE") or None,
        }
        rec.update(flags)
        owner_rows.append(rec)
        by_acc[r["ACCESSION_NUMBER"]].append(rec)

    purch_rows = []
    dropped_no_dir_or_off = 0
    for s, t in kept_rows:
        owners_here = by_acc.get(s["ACCESSION_NUMBER"], [])
        if not any(o["director"] or o["officer"] for o in owners_here):
            dropped_no_dir_or_off += 1
            continue
        primary = sorted(
            owners_here, key=lambda o: (o["owner_cik"] or "9" * 10))[0]
        shares = parse_number(t.get("TRANS_SHARES"))
        price = parse_number(t.get("TRANS_PRICEPERSHARE"))
        fdate = parse_date(s.get("FILING_DATE"))
        tdate = parse_date(t.get("TRANS_DATE"))
        purch_rows.append(
            {
                "accession": s["ACCESSION_NUMBER"],
                "trans_sk": t.get("NONDERIV_TRANS_SK") or None,
                "filing_date": fdate,
                "trans_date": tdate,
                "filing_lag_days": filing_lag_days(
                    s.get("FILING_DATE"), t.get("TRANS_DATE")),
                "issuer_cik": pad_cik(s.get("ISSUERCIK")),
                "issuer_name": s.get("ISSUERNAME") or None,
                "issuer_symbol": s.get("ISSUERTRADINGSYMBOL") or None,
                "owner_cik": primary["owner_cik"],
                "owner_name": primary["owner_name"],
                "relationship_as_filed": primary["relationship_as_filed"],
                "title": primary["title"],
                "director": primary["director"],
                "officer": primary["officer"],
                "ten_percent_owner": primary["ten_percent_owner"],
                "other": primary["other"],
                "any_director": any(o["director"] for o in owners_here),
                "any_officer": any(o["officer"] for o in owners_here),
                "any_ten_percent_owner": any(
                    o["ten_percent_owner"] for o in owners_here),
                "owner_count": len(owners_here),
                "security_title": t.get("SECURITY_TITLE") or None,
                "direct_indirect": (
                    t.get("DIRECT_INDIRECT_OWNERSHIP") or None),
                "shares": shares,
                "price": price,
                "value": dollar_value(t.get("TRANS_SHARES"),
                                      t.get("TRANS_PRICEPERSHARE")),
                "aff10b5one_as_filed": s.get("AFF10B5ONE"),
                "aff10b5one": normalise_flag(s.get("AFF10B5ONE")),
                "archive": name,
            }
        )
    cov["dropped_no_director_or_officer"] = dropped_no_dir_or_off
    cov["purchases_kept"] = len(purch_rows)
    cov["owner_rows_kept"] = len(owner_rows)
    return pd.DataFrame(purch_rows), pd.DataFrame(owner_rows), cov


# ---------------------------------------------------------------------------
# archives
# ---------------------------------------------------------------------------

def cmd_archives(args):
    sess = session()
    html = sess.get(INSIDER_PAGE)
    arcs = extract_archive_links(html)
    if not arcs:
        raise StopRun("no insider-transactions archive found on %s" % INSIDER_PAGE)
    out = {
        "page": INSIDER_PAGE,
        "count": len(arcs),
        "first": arcs[0]["name"],
        "last": arcs[-1]["name"],
        "archives": arcs,
        "page_sha256": sha256_bytes(html),
    }
    dest = args.out or log_path("archives.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
        fh.write("\n")
    say("archives %d  first %s  last %s" % (len(arcs), out["first"], out["last"]))
    say("wrote %s" % dest)
    say("DONE archives calls=%d bytes=%d" % (sess.calls, sess.bytes))
    return 0


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    with open(args.archives, "r", encoding="utf-8") as fh:
        arcs = json.load(fh)["archives"]
    dest_dir = raw_dir()
    os.makedirs(dest_dir, exist_ok=True)
    man_path = fetch_manifest_path()
    man = load_manifest(man_path)
    files = man.setdefault("files", {})
    sess = session()
    t0 = time.time()
    fetched = skipped = failed = 0
    for a in arcs:
        name = a["name"]
        dest = os.path.join(dest_dir, name)
        rec = files.get(name)
        if rec and os.path.isfile(dest) and rec.get("bytes") == os.path.getsize(dest):
            skipped += 1
            continue
        try:
            body = sess.get(archive_url(a["href"]))
        except StopRun as exc:
            say("FAILED %-22s %s" % (name, exc))
            failed += 1
            continue
        tmp = dest + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(body)
        os.replace(tmp, dest)
        files[name] = {
            "bytes": len(body),
            "sha256": sha256_bytes(body),
            "href": a["href"],
        }
        write_manifest(man_path, man)
        fetched += 1
        say("got %-22s %12d bytes  %s" % (name, len(body), files[name]["sha256"][:16]))
    man["archives"] = len(arcs)
    man["total_bytes"] = sum(v["bytes"] for v in files.values())
    write_manifest(man_path, man)
    say(
        "DONE fetch archives=%d fetched=%d skipped=%d failed=%d bytes=%d wall=%.0fs"
        % (len(arcs), fetched, skipped, failed, man["total_bytes"], time.time() - t0)
    )
    return 1 if failed else 0


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def cmd_build(args):
    with open(args.archives, "r", encoding="utf-8") as fh:
        arcs = json.load(fh)["archives"]
    vdir = vintage_dir(args.vintage)
    os.makedirs(vdir, exist_ok=True)
    t0 = time.time()
    P, O, C = [], [], []
    for a in arcs:
        path = os.path.join(raw_dir(), a["name"])
        if not os.path.isfile(path):
            raise StopRun("archive not fetched: %s" % a["name"])
        p, o, cov = purchases_from_archive(path, a["name"])
        P.append(p)
        O.append(o)
        C.append(cov)
        say("built %-22s purchases %7d  owners %7d" % (a["name"], len(p), len(o)))
    purchases = pd.concat([d for d in P if len(d)], ignore_index=True)
    owners = pd.concat([d for d in O if len(d)], ignore_index=True)
    coverage = pd.DataFrame(C)

    purchases = routine_pairs(purchases)
    for c in ("filing_date", "trans_date"):
        purchases[c] = pd.to_datetime(purchases[c])

    p_path = os.path.join(vdir, "purchases.parquet")
    o_path = os.path.join(vdir, "owners.parquet")
    c_path = os.path.join(vdir, "coverage.parquet")
    purchases.to_parquet(p_path, index=False)
    owners.to_parquet(o_path, index=False)
    coverage.to_parquet(c_path, index=False)

    man = {
        "vintage": args.vintage,
        "built_utc": pd.Timestamp.utcnow().isoformat(),
        "tool_version": "1.0.0",
        "archives": len(arcs),
        "rows": {
            "purchases": int(len(purchases)),
            "owners": int(len(owners)),
            "coverage": int(len(coverage)),
        },
        "sha256": {
            "purchases.parquet": sha256_file(p_path),
            "owners.parquet": sha256_file(o_path),
            "coverage.parquet": sha256_file(c_path),
        },
        "per_archive": {c["archive"]: c for c in C},
        "fetch_manifest_sha256": (
            sha256_file(fetch_manifest_path())
            if os.path.isfile(fetch_manifest_path()) else None),
    }
    write_manifest(manifest_path(args.vintage), man)
    tracked = tracked_manifest_path(args.vintage)
    write_manifest(tracked, man)
    say("purchases %d  owners %d  archives %d  wall %.0fs"
        % (len(purchases), len(owners), len(arcs), time.time() - t0))
    say("manifest %s sha256 %s" % (tracked, sha256_file(tracked)))
    return 0


# ---------------------------------------------------------------------------
# join and report
# ---------------------------------------------------------------------------

def load_purchases(vintage):
    return pd.read_parquet(os.path.join(vintage_dir(vintage), "purchases.parquet"))


def cmd_join(args):
    purchases = load_purchases(args.vintage)
    tickers = pd.read_parquet(args.tickers)
    cmap = build_cik_map(tickers)
    matched, multi = [], set()
    for cik in purchases["issuer_cik"]:
        hits, is_multi = match_issuer_cik(cik, cmap)
        if is_multi:
            multi.add(pad_cik(cik))
        matched.append(bool(hits))
    purchases["sharadar_matched"] = matched
    say("purchases                 %d" % len(purchases))
    say("matched to Sharadar       %d (%.4f)"
        % (int(sum(matched)), float(np.mean(matched)) if len(matched) else 0.0))
    say("unmatched issuer CIKs     %d"
        % purchases.loc[~purchases["sharadar_matched"], "issuer_cik"].nunique())
    say("multi-mapped CIKs         %d" % len(multi))
    return 0


def cmd_report(args):
    """Deferred to the sprint's reporting driver; counts only, never a return."""
    say("report: use the sprint driver; this sub-command forms no statistic")
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("archives", help="locate and list the bulk archives")
    a.add_argument("--out", default=None)
    a.set_defaults(fn=cmd_archives)

    f = sub.add_parser("fetch", help="download every archive, resumable")
    f.add_argument("--archives", required=True)
    f.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("build", help="build purchases.parquet and owners.parquet")
    b.add_argument("--archives", required=True)
    b.add_argument("--vintage", required=True)
    b.set_defaults(fn=cmd_build)

    j = sub.add_parser("join", help="join issuers to Sharadar; presence counts only")
    j.add_argument("--vintage", required=True)
    j.add_argument("--tickers", required=True)
    j.set_defaults(fn=cmd_join)

    r = sub.add_parser("report", help="counts, dates and sizes only")
    r.add_argument("--vintage", required=True)
    r.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except StopRun as exc:
        say("STOP %s" % exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
