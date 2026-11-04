"""
================================================================================
Script:       edgar_item408_probe.py
Version:      1.1.1
Phase:        Stage R — External Data Evaluation
Purpose:      Build the event table of Rule 10b5-1 trading-arrangement adoptions
              and terminations disclosed under Item 408(a) of Regulation S-K in
              10-Q and 10-K filings, from EDGAR full-text search plus the
              Inline XBRL (ecd:*) tags in the primary document.
Entry Point:  Yes — python tools/edgar_item408_probe.py {count,fetch,parse,join,report,fsn}
Dependencies: stdlib, requests, pandas, pyarrow. Imports nothing from any other
              repository.
Last Sprint:  Sprint R5-1 — bulk dataset route (2026-11-04)
================================================================================

No credential is read, sent, or required; no .env file is opened. The only
identifier sent is the contact string the SEC requires in the User-Agent, read
from the environment variable EDGAR_CONTACT if set, otherwise from the file
$HOME/.config/stage_r/edgar_contact. It is a public contact identifier, not a
secret, but it is never printed, logged, or written into any artifact: the tool
reports its byte length only.

Fair access is enforced in one place. Every request to a sec.gov host passes
through a sliding-window rate limiter capped at 8 requests per second with a
single worker, and gzip is accepted. A 429 or 403 is retried up to three times
with exponential backoff; after that the run stops and reports the status and
the URL path, never the header.

The event table is keyed by (adsh, individual). The key is the ecd:IndividualAxis
dimension member, NOT the raw contextRef: a filer tags an individual's duration
facts and instant facts in different contexts (the aggregate-securities amount
is an instant fact), so contextRef over-splits one arrangement into several
rows. Resolving the context's dimension merges them. When a context carries no
IndividualAxis member the contextRef is used, which keeps the entity-level
summary flags as their own row. Flags are resolved from
the ixt format attribute first and from the element text only as a fallback;
anything that resolves to neither true nor false is recorded UNRESOLVED rather
than guessed. The smallest enclosing table or paragraph is stored verbatim
alongside each row so that sell-versus-purchase classification can be done
later from the artifact; this tool stores that text and classifies nothing.
================================================================================
"""

from __future__ import annotations

import argparse
import csv
import collections
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
import zipfile

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FTS_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
# The Financial Statement and Notes data sets, found by walking sec.gov from
# /data-research to the DERA data library to
# /data-research/financial-statement-notes-data-sets. Archives are named
# <YYYY>q<Q>_notes.zip through 2025q3 and <YYYY>_<MM>_notes.zip thereafter.
FSN_BASE = "https://www.sec.gov/files/dera/data/financial-statement-notes-data-sets"
CONTACT_ENV = "EDGAR_CONTACT"
CONTACT_FILE = os.path.join("~", ".config", "stage_r", "edgar_contact")

MAX_REQUESTS_PER_SECOND = 8
# 429 and 403 are the fair-access responses the SEC documents. The transient
# 5xx family is retried on the same budget: a single 500 from the search
# endpoint is a server hiccup, not a verdict on the request.
RETRY_STATUSES = (429, 403, 500, 502, 503, 504)
MAX_RETRIES = 3
FTS_PAGE_SIZE = 100
# Elasticsearch from+size ceiling behind EDGAR full-text search.
FTS_RESULT_CEILING = 10000

# The primary-document file types we keep. Amendments (10-Q/A, 10-K/A) and
# exhibits (EX-*) are counted and set aside, not parsed.
PRIMARY_FILE_TYPES = ("10-Q", "10-K")

# The Item 408 / Rule 10b5-1 Inline XBRL family, as published in the SEC's
# Insider Trading Arrangements and Policies taxonomy.
ECD_FIELDS = (
    "ecd:TrdArrIndName",
    "ecd:TrdArrIndTitle",
    "ecd:TrdArrAdoptedFlag",
    "ecd:TrdArrTerminatedFlag",
    "ecd:TrdArrAdoptionDate",
    "ecd:TrdArrTerminationDate",
    "ecd:TrdArrDuration",
    "ecd:TrdArrSecuritiesAggAvailAmt",
    "ecd:TrdArrExpirationDate",
    "ecd:Rule10b51ArrAdoptedFlag",
    "ecd:Rule10b51ArrTrmntdFlag",
    "ecd:NonRule10b51ArrAdoptedFlag",
    "ecd:NonRule10b51ArrTrmntdFlag",
)

UNRESOLVED = "UNRESOLVED"


class StopRun(Exception):
    """A condition the sprint brief says must stop the run, not be worked around."""


# ---------------------------------------------------------------------------
# Contact string
# ---------------------------------------------------------------------------

def read_contact(env=None, file_path=None):
    """Return the SEC contact string from the environment or the config file.

    Raises StopRun if neither source exists or the value fails the shape check.
    The value is returned, never printed; callers report its byte length only.
    """
    env = os.environ if env is None else env
    raw = env.get(CONTACT_ENV)
    source = "env:%s" % CONTACT_ENV
    if raw is None or not raw.strip():
        path = os.path.expanduser(CONTACT_FILE if file_path is None else file_path)
        if not (os.path.isfile(path) and os.path.getsize(path) > 0):
            raise StopRun(
                "no SEC contact string: %s unset and %s absent or empty"
                % (CONTACT_ENV, CONTACT_FILE)
            )
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        source = "file:%s" % CONTACT_FILE
    value = raw.rstrip("\n")
    if len(value) < 8 or "@" not in value:
        raise StopRun(
            "SEC contact string failed the shape check (length >= 8 and '@' required)"
        )
    return value, source


# ---------------------------------------------------------------------------
# Fair-access rate limiting
# ---------------------------------------------------------------------------

class RateLimiter:
    """Sliding-window limiter: never more than max_per_sec calls in any second.

    time_fn and sleep_fn are injected so the window can be driven by a fake
    clock under test.
    """

    def __init__(self, max_per_sec=MAX_REQUESTS_PER_SECOND, time_fn=None, sleep_fn=None):
        self.max_per_sec = max_per_sec
        self._time = time.monotonic if time_fn is None else time_fn
        self._sleep = time.sleep if sleep_fn is None else sleep_fn
        self._stamps = collections.deque()

    def acquire(self):
        while True:
            now = self._time()
            while self._stamps and now - self._stamps[0] >= 1.0:
                self._stamps.popleft()
            if len(self._stamps) < self.max_per_sec:
                self._stamps.append(now)
                return now
            wait = 1.0 - (now - self._stamps[0])
            self._sleep(max(wait, 0.0))


class SecSession:
    """One shared rate-limited session for every sec.gov request."""

    def __init__(self, contact, limiter=None, session=None, sleep_fn=None):
        import requests

        self._contact = contact
        self.limiter = RateLimiter() if limiter is None else limiter
        self.session = requests.Session() if session is None else session
        self._sleep = time.sleep if sleep_fn is None else sleep_fn
        self.calls = 0
        self.bytes = 0

    def _headers(self):
        return {
            "User-Agent": self._contact,
            "Accept-Encoding": "gzip, deflate",
        }

    def get(self, url, params=None, expect_json=False):
        attempt = 0
        while True:
            self.limiter.acquire()
            self.calls += 1
            resp = self.session.get(
                url, params=params, headers=self._headers(), timeout=60
            )
            if resp.status_code == 200:
                self.bytes += len(resp.content)
                return resp.json() if expect_json else resp.content
            if resp.status_code in RETRY_STATUSES and attempt < MAX_RETRIES:
                attempt += 1
                self._sleep(2.0 ** attempt + random.random())
                continue
            raise StopRun(
                "HTTP %d after %d retries on path %s"
                % (resp.status_code, attempt, url_path(resp.url))
            )


def url_path(url):
    """The path of a URL, for error reporting. Never includes headers."""
    s = str(url)
    if "://" not in s:
        return s.split("?")[0]
    return "/" + s.split("://", 1)[1].split("/", 1)[1].split("?")[0] if "/" in s.split("://", 1)[1] else "/"


# ---------------------------------------------------------------------------
# Pure helpers: identifiers, URLs, windows
# ---------------------------------------------------------------------------

def normalise_adsh(value):
    """Return an accession number as 18 digits with dashes: NNNNNNNNNN-NN-NNNNNN."""
    digits = re.sub(r"\D", "", str(value))
    if len(digits) != 18:
        raise ValueError("accession number must carry 18 digits, got %r" % (value,))
    return "%s-%s-%s" % (digits[:10], digits[10:12], digits[12:])


def adsh_nodash(value):
    """Return an accession number as 18 bare digits."""
    return normalise_adsh(value).replace("-", "")


def build_archive_url(cik, adsh, filename):
    """The EDGAR archive URL of one document inside one filing.

    The CIK is used in its integer form, which is how the Archives tree is laid
    out; a zero-padded 10-digit CIK is accepted and normalised.
    """
    cik_int = int(str(cik).strip().lstrip("0") or "0")
    name = str(filename).strip().lstrip("/")
    if not name:
        raise ValueError("filename required")
    return "%s/%d/%s/%s" % (ARCHIVE_BASE, cik_int, adsh_nodash(adsh), name)


def cik_from_secfilings(url):
    """Extract the 10-digit CIK from a Sharadar secfilings browse-edgar URL.

    Returns None when the field is absent or carries no CIK= parameter.
    """
    if url is None:
        return None
    s = str(url)
    m = re.search(r"CIK=(\d+)", s)
    if not m:
        return None
    return m.group(1).zfill(10)


def month_windows(start, end):
    """Inclusive calendar-month [startdt, enddt] windows covering start..end."""
    s = date.fromisoformat(str(start))
    e = date.fromisoformat(str(end))
    if e < s:
        raise ValueError("end before start")
    out = []
    y, m = s.year, s.month
    while True:
        first = date(y, m, 1)
        nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
        last = nxt - timedelta(days=1)
        lo = max(first, s)
        hi = min(last, e)
        if lo <= hi:
            out.append((lo.isoformat(), hi.isoformat()))
        if nxt > e:
            break
        y, m = nxt.year, nxt.month
    return out


def fts_params(q, forms, startdt, enddt, frm=0, size=FTS_PAGE_SIZE):
    """Query-string parameters for one EDGAR full-text search page."""
    return {
        "q": q,
        "forms": ",".join(forms) if not isinstance(forms, str) else forms,
        "dateRange": "custom",
        "startdt": startdt,
        "enddt": enddt,
        "from": frm,
        "size": size,
    }


def page_offsets(total, size=FTS_PAGE_SIZE, ceiling=FTS_RESULT_CEILING):
    """from= offsets needed to walk `total` hits at `size` per page.

    Offsets stop at the Elasticsearch from+size ceiling behind the endpoint;
    a caller that gets fewer offsets than total/size implies has hit the cap.
    """
    if total <= 0:
        return []
    reachable = min(int(total), int(ceiling))
    return [f for f in range(0, reachable, int(size))]


def parse_fts_page(payload):
    """Return (total, hits) from one full-text search response.

    Each hit is a flat dict: adsh, filename, cik, form, file_type, file_date,
    period_ending, display_name.
    """
    hits_block = (payload or {}).get("hits", {}) or {}
    total_block = hits_block.get("total", {}) or {}
    total = int(total_block.get("value", 0))
    out = []
    for hit in hits_block.get("hits", []) or []:
        src = hit.get("_source", {}) or {}
        hid = str(hit.get("_id", ""))
        adsh_part, _, filename = hid.partition(":")
        ciks = src.get("ciks") or []
        out.append(
            {
                "adsh": normalise_adsh(adsh_part) if adsh_part else None,
                "filename": filename or None,
                "cik": str(ciks[0]).zfill(10) if ciks else None,
                "form": src.get("form"),
                "file_type": src.get("file_type"),
                "file_date": src.get("file_date"),
                "period_ending": src.get("period_ending"),
                "display_name": (src.get("display_names") or [None])[0],
            }
        )
    return total, out


def keep_primary(hits, primary_types=PRIMARY_FILE_TYPES):
    """Split hits into primary documents and everything set aside.

    Returns (primary, aside) where aside counts amendments and exhibits by
    file_type so the discard is reported rather than silent.
    """
    primary, aside = [], collections.Counter()
    for h in hits:
        if h.get("file_type") in primary_types:
            primary.append(h)
        else:
            aside[h.get("file_type")] += 1
    return primary, aside


# ---------------------------------------------------------------------------
# Inline XBRL extraction
# ---------------------------------------------------------------------------

_TRUE_TEXTS = {"true", "yes", "y", "x", "☒", "☑", "[x]", "(x)"}
_FALSE_TEXTS = {"false", "no", "n", "☐", "[ ]", "()", "[]"}


def resolve_flag(fmt, text):
    """Resolve an iXBRL boolean to True, False, or UNRESOLVED.

    The ixt format attribute is authoritative when present; the element text is
    only a fallback. Anything that is neither true nor false is UNRESOLVED,
    never guessed.
    """
    f = (fmt or "").strip().lower()
    if f:
        tail = f.split(":")[-1]
        if "fixed-true" in tail or tail == "booleantrue":
            return True
        if "fixed-false" in tail or tail == "booleanfalse":
            return False
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "", t)
    if t in _TRUE_TEXTS:
        return True
    if t in _FALSE_TEXTS:
        return False
    return UNRESOLVED


_MONTHS = {
    m.lower(): i
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}
for _k in list(_MONTHS):
    _MONTHS[_k[:3]] = _MONTHS[_k]


def parse_ix_date(fmt, text):
    """Parse an iXBRL date to an ISO string, or None when unparseable.

    Covers the ixt:date-* families actually used in Item 408 disclosure:
    month-name day year, numeric month/day/year, and ISO.
    """
    t = re.sub(r"\s+", " ", (text or "")).strip().strip(".")
    if not t:
        return None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        y, mo, d = (int(x) for x in m.groups())
        return _safe_iso(y, mo, d)
    # The bulk dataset ships month-name dates with the space collapsed out
    # ("December19, 2025"), so the separator is optional, not required.
    m = re.fullmatch(r"([A-Za-z]{3,9})\.?,?\s*(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", t)
    if m:
        mo = _MONTHS.get(m.group(1).lower())
        if mo:
            return _safe_iso(int(m.group(3)), mo, int(m.group(2)))
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})", t)
    if m:
        mo = _MONTHS.get(m.group(2).lower())
        if mo:
            return _safe_iso(int(m.group(3)), mo, int(m.group(1)))
    m = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", t)
    if m:
        mo, d, y = (int(x) for x in m.groups())
        if y < 100:
            y += 2000
        return _safe_iso(y, mo, d)
    return None


def _safe_iso(y, mo, d):
    try:
        return date(y, mo, d).isoformat()
    except ValueError:
        return None


class _IxParser(HTMLParser):
    """Collect ix:nonNumeric / ix:nonFraction facts whose name is in the ecd namespace.

    Each fact carries its contextRef, format, raw text, and the raw text of the
    smallest enclosing table or paragraph.
    """

    CONTAINERS = ("table", "p")
    FACT_TAGS = ("ix:nonnumeric", "ix:nonfraction")

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.facts = []
        self._open_facts = []
        self._container_stack = []
        self._container_text = {}
        self._next_cid = 0

    # -- containers ------------------------------------------------------
    def _open_container(self, tag):
        if tag == "p" and self._container_stack and self._container_stack[-1][0] == "p":
            # HTML allows an unclosed <p>; a new one implicitly closes it.
            self._container_stack.pop()
        self._next_cid += 1
        cid = self._next_cid
        self._container_stack.append((tag, cid))
        self._container_text[cid] = []

    def _close_container(self, tag):
        for i in range(len(self._container_stack) - 1, -1, -1):
            if self._container_stack[i][0] == tag:
                del self._container_stack[i:]
                return

    def _current_cid(self):
        return self._container_stack[-1][1] if self._container_stack else None

    # -- parser hooks ----------------------------------------------------
    def handle_starttag(self, tag, attrs):
        if tag in self.CONTAINERS:
            self._open_container(tag)
            return
        if tag in self.FACT_TAGS:
            a = dict(attrs)
            name = a.get("name")
            if name and name.startswith("ecd:"):
                self._open_facts.append(
                    {
                        "tag": tag,
                        "name": name,
                        "context": a.get("contextref"),
                        "format": a.get("format"),
                        "sign": a.get("sign"),
                        "scale": a.get("scale"),
                        "nil": a.get("xsi:nil"),
                        "container": self._current_cid(),
                        "chunks": [],
                    }
                )

    def handle_startendtag(self, tag, attrs):
        if tag in self.FACT_TAGS:
            a = dict(attrs)
            name = a.get("name")
            if name and name.startswith("ecd:"):
                self.facts.append(
                    {
                        "name": name,
                        "context": a.get("contextref"),
                        "format": a.get("format"),
                        "sign": a.get("sign"),
                        "scale": a.get("scale"),
                        "nil": a.get("xsi:nil"),
                        "container": self._current_cid(),
                        "text": "",
                    }
                )

    def handle_endtag(self, tag):
        if self._open_facts and self._open_facts[-1]["tag"] == tag:
            f = self._open_facts.pop()
            text = re.sub(r"\s+", " ", "".join(f["chunks"])).strip()
            self.facts.append(
                {
                    "name": f["name"],
                    "context": f["context"],
                    "format": f["format"],
                    "sign": f["sign"],
                    "scale": f["scale"],
                    "nil": f["nil"],
                    "container": f["container"],
                    "text": text,
                }
            )
            return
        if tag in self.CONTAINERS:
            self._close_container(tag)

    def handle_data(self, data):
        for f in self._open_facts:
            f["chunks"].append(data)
        for _tag, cid in self._container_stack:
            self._container_text[cid].append(data)

    # -- result ----------------------------------------------------------
    def container_text(self, cid):
        if cid is None:
            return ""
        return re.sub(r"\s+", " ", "".join(self._container_text.get(cid, []))).strip()


def extract_ecd_facts(html):
    """Return the list of ecd:* iXBRL facts found in a document.

    Each fact: name, context, format, text, container_text.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    p = _IxParser()
    p.feed(html)
    p.close()
    out = []
    for f in p.facts:
        out.append(
            {
                "name": f["name"],
                "context": f["context"],
                "format": f["format"],
                "text": f["text"],
                "container_text": p.container_text(f["container"]),
            }
        )
    return out


def ecd_names(html):
    """The distinct ecd:* iXBRL names present in a document, sorted."""
    return sorted({f["name"] for f in extract_ecd_facts(html)})


_CONTEXT_RE = re.compile(
    r"<(?:\w+:)?context\b[^>]*\bid=\"([^\"]+)\"(.*?)</(?:\w+:)?context>",
    re.S | re.I,
)
_MEMBER_RE = re.compile(
    r"<(?:\w+:)?explicitMember\b[^>]*\bdimension=\"([^\"]+)\"[^>]*>(.*?)"
    r"</(?:\w+:)?explicitMember>",
    re.S | re.I,
)
INDIVIDUAL_AXIS = "IndividualAxis"
INDIVIDUAL_DIM = "Individual"


def _norm_axis(axis):
    a = _localname(axis)
    return a[:-4] if a.endswith("Axis") else a


def _norm_member(member):
    v = _localname(member)
    return v[:-6] if v.endswith("Member") else v


def canonical_dim_key(dims):
    """A source-independent key for a dimension set.

    The HTML path reads "IndividualAxis" -> "SanjayMehrotraMember" from the
    document's context blocks; the bulk dataset writes "Individual=SanjayMehrotra".
    Stripping the Axis and Member suffixes makes the two spell the same key, so
    the routes are comparable fact by fact. An empty set returns "".
    """
    if not dims:
        return ""
    pairs = sorted((_norm_axis(a), _norm_member(mv)) for a, mv in dims.items())
    return ";".join("%s=%s" % (a, mv) for a, mv in pairs)


def parse_canonical_dim_key(key):
    """Inverse of canonical_dim_key for a key this module produced."""
    out = {}
    for part in str(key or "").split(";"):
        if "=" in part:
            a, _, mv = part.partition("=")
            out[a] = mv
    return out


_META_COLUMNS = ("adsh", "cik", "form", "file_date", "period_ending", "context_ref")


def apply_individual_inheritance(rows):
    """Push individual-level facts down onto that individual's arrangement rows.

    A filer may tag a person's name, title and dates against the Individual
    axis alone while tagging each separate trading arrangement's amount against
    (Individual, TradingArr). The arrangements are the arrangements, so they are
    the rows; the individual-level facts describe the person and are inherited
    by each of them, never overwriting a value the arrangement states itself.
    The now-redundant individual-only row is dropped. A person with no
    arrangement-level row keeps their individual-level row unchanged.
    """
    groups = {}
    for r in rows:
        dims = parse_canonical_dim_key(r.get("context_ref"))
        ind = dims.get(INDIVIDUAL_DIM)
        if not ind:
            continue
        slot = groups.setdefault(ind, {"base": None, "arr": []})
        if len(dims) == 1:
            if slot["base"] is None:
                slot["base"] = r
        else:
            slot["arr"].append(r)

    dropped = set()
    for _ind, slot in groups.items():
        base, arr = slot["base"], slot["arr"]
        if base is None or not arr:
            continue
        for a in arr:
            for col, val in base.items():
                if col in _META_COLUMNS:
                    continue
                if val in (None, "") or a.get(col) not in (None, ""):
                    continue
                a[col] = val
        dropped.add(id(base))
    return [r for r in rows if id(r) not in dropped]



def _localname(qname):
    return str(qname).split(":")[-1].strip()


def extract_context_dimensions(html):
    """Map each XBRL context id to its explicit dimension members.

    Returns {context_id: {axis_localname: member_localname}}. Contexts with no
    segment map to an empty dict. Namespace prefixes vary between filers, so
    both the element names and the dimension/member values are matched on their
    local names.
    """
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    out = {}
    for mt in _CONTEXT_RE.finditer(html):
        cid, body = mt.group(1), mt.group(2)
        dims = {}
        for mm in _MEMBER_RE.finditer(body):
            axis = _localname(mm.group(1))
            member = _localname(re.sub(r"<[^>]+>", "", mm.group(2)))
            if axis:
                dims[axis] = member
        out[cid] = dims
    return out


def individual_key(context_ref, ctx_dims):
    """The row key for a fact: its full canonical dimension set, else contextRef.

    Keying on the dimension set merges the duration-context and instant-context
    facts a filer tags for one arrangement, while keeping two arrangements of
    the same person apart - a distinction that keying on the individual alone
    destroys. A context with no dimensions (the entity-level summary flags)
    keeps its own contextRef, so it stays a row of its own.
    """
    if not ctx_dims:
        return context_ref
    dims = ctx_dims.get(context_ref) or {}
    key = canonical_dim_key(dims)
    return key if key else context_ref


def facts_to_rows(facts, ctx_dims=None):
    """Group ecd facts into one row per contextRef.

    An Item 408 disclosure tags one individual per context, so the grouping is
    the filing's own. Flags are resolved, dates parsed, text kept raw.
    """
    by_ctx = collections.OrderedDict()
    for f in facts:
        ctx = f.get("context")
        if ctx is None:
            continue
        by_ctx.setdefault(individual_key(ctx, ctx_dims), []).append(f)

    rows = []
    for ctx, group in by_ctx.items():
        row = {"context_ref": ctx}
        # "Smallest enclosing" is per fact; for the row we keep the shortest
        # non-empty block any of its facts sat in.
        containers = []
        for f in group:
            name = f["name"]
            text = f["text"]
            if f.get("container_text"):
                containers.append(f["container_text"])
            if name.endswith("Flag"):
                row[name] = resolve_flag(f.get("format"), text)
            elif name.endswith("Date"):
                row[name] = parse_ix_date(f.get("format"), text)
                row[name + "_raw"] = text
            else:
                row[name] = text
        row["container_text"] = min(containers, key=len) if containers else ""
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(path):
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def write_manifest(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)
    return path


def already_fetched(manifest, adsh):
    """True when this filing's document is already banked under this vintage."""
    return adsh in (manifest.get("docs") or {})


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def vintage_dir(vintage, root=None):
    return os.path.join(root or REPO_ROOT, "data", "edgar_item408", str(vintage))


def docs_dir(vintage, root=None):
    return os.path.join(vintage_dir(vintage, root), "docs")


def manifest_path(vintage, root=None):
    return os.path.join(vintage_dir(vintage, root), "manifest.json")


def tracked_manifest_path(vintage, root=None):
    return os.path.join(root or REPO_ROOT, "manifests", "edgar_item408_%s.json" % vintage)


def union_path(vintage, root=None):
    return os.path.join(vintage_dir(vintage, root), "union.json")


def log_path(name, root=None):
    return os.path.join(root or REPO_ROOT, "logs", "r5", name)


def say(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------

def fts_total(sess, q, forms, startdt, enddt):
    payload = sess.get(
        FTS_URL, params=fts_params(q, forms, startdt, enddt, 0, FTS_PAGE_SIZE),
        expect_json=True,
    )
    total, hits = parse_fts_page(payload)
    return total, hits, payload


def fts_collect(sess, q, forms, startdt, enddt):
    """Every hit for one query window, paged to the endpoint's ceiling."""
    total, hits, _ = fts_total(sess, q, forms, startdt, enddt)
    out = list(hits)
    for frm in page_offsets(total, FTS_PAGE_SIZE)[1:]:
        payload = sess.get(
            FTS_URL, params=fts_params(q, forms, startdt, enddt, frm, FTS_PAGE_SIZE),
            expect_json=True,
        )
        _t, page = parse_fts_page(payload)
        if not page:
            break
        out.extend(page)
    return total, out


def _write_union(args, template, terms, forms, per_month, aside_total, union):
    """Write (or rewrite) the union checkpoint. Called after every month."""
    payload = {
        "query_template": template,
        "terms": terms,
        "forms": forms,
        "start": args.start,
        "end": args.end,
        "per_month": per_month,
        "aside_by_file_type": dict(aside_total),
        "union_size": len(union),
        "filings": sorted(union.values(), key=lambda h: (h["file_date"] or "", h["adsh"])),
    }
    os.makedirs(vintage_dir(args.vintage), exist_ok=True)
    return write_manifest(union_path(args.vintage), payload)


def cmd_count(args):
    contact, source = read_contact()
    say("contact: source %s, byte length %d" % (source, len(contact.encode())))
    sess = SecSession(contact)
    forms = ["10-Q", "10-K"]

    if args.variants:
        say("")
        say("--- 2.2 variant probe, month %s ---" % args.variants)
        start, end = month_bounds(args.variants)
        variants = {
            "A": '"10b5-1"',
            "B": '"10b5-1" terminated',
            "C": '"10b5-1" AND terminated',
            "D": '"10b5-1" OR terminated',
        }
        results = {}
        sample = {}
        keys_report = None
        for label in ("A", "B", "C", "D"):
            total, hits, payload = fts_total(sess, variants[label], forms, start, end)
            results[label] = total
            sample[label] = [h["adsh"] + ":" + (h["filename"] or "") for h in hits[:2]]
            if keys_report is None and hits:
                keys_report = {
                    "top_level_keys": sorted(payload.keys()),
                    "hits_keys": sorted(payload.get("hits", {}).keys()),
                    "source_keys": sorted(
                        (payload["hits"]["hits"][0].get("_source") or {}).keys()
                    ),
                }
            say("variant %s  q=%-28s total %d" % (label, variants[label], total))
        say("")
        say("FTS top-level keys: %s" % keys_report["top_level_keys"])
        say("FTS hits keys:      %s" % keys_report["hits_keys"])
        say("FTS _source keys:   %s" % keys_report["source_keys"])
        for label in ("A", "B", "C", "D"):
            say("sample %s: %s" % (label, sample[label]))
        narrowing = [
            L for L in ("B", "C")
            if 0 < results[L] < results["A"]
        ]
        say("")
        say("narrowing variants (0 < total < A): %s" % (narrowing or "NONE"))
        if not narrowing:
            raise StopRun(
                "neither B nor C narrows: A=%d B=%d C=%d D=%d"
                % (results["A"], results["B"], results["C"], results["D"])
            )
        say("AND form chosen: variant %s" % narrowing[0])
        out = {
            "variant_totals": results,
            "samples": sample,
            "fts_keys": keys_report,
            "and_form": narrowing[0],
            "month": args.variants,
        }
        os.makedirs(os.path.dirname(log_path("x")), exist_ok=True)
        with open(log_path("variants.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        say("wrote %s" % log_path("variants.json"))
        say("calls %d  bytes %d" % (sess.calls, sess.bytes))
        return 0

    # Full-range union over the three term spellings.
    terms = ["terminated", "termination", "terminate"]
    template = args.and_template
    windows = month_windows(args.start, args.end)
    per_month = []
    union = {}
    aside_total = collections.Counter()

    # Month-level resume: a transient endpoint failure costs one month, not the
    # whole range. The checkpoint is only reused when the query shape matches.
    # A month is reusable only when the query shape AND that month's exact
    # window match: a clipped first or last month is not the same measurement
    # as the whole month.
    done_windows = set()
    ckpt = load_manifest(union_path(args.vintage))
    shape_now = {"query_template": template, "terms": terms, "forms": forms}
    if ckpt and all(ckpt.get(k) == v for k, v in shape_now.items()):
        wanted = {(lo, hi) for (lo, hi) in windows}
        for r in ckpt.get("per_month") or []:
            w = (r.get("startdt"), r.get("enddt"))
            if w in wanted:
                per_month.append(r)
                done_windows.add(w)
        if done_windows:
            keep = {m for (m, _h) in ((r["month"], None) for r in per_month)}
            for h in ckpt.get("filings") or []:
                if (h.get("file_date") or "")[:7] in keep:
                    union[h["adsh"]] = h
            aside_total.update(ckpt.get("aside_by_file_type") or {})
            say("resuming: %d months already counted, union %d"
                % (len(done_windows), len(union)))

    for (lo, hi) in windows:
        if (lo, hi) in done_windows:
            continue
        a_total, _h, _p = fts_total(sess, '"10b5-1"', forms, lo, hi)
        month_union = {}
        month_aside = collections.Counter()
        for term in terms:
            q = template.replace("{term}", term)
            total, hits = fts_collect(sess, q, forms, lo, hi)
            primary, aside = keep_primary(hits)
            month_aside.update(aside)
            for h in primary:
                month_union.setdefault(h["adsh"], h)
        aside_total.update(month_aside)
        union.update(month_union)
        per_month.append(
            {
                "month": lo[:7],
                "startdt": lo,
                "enddt": hi,
                "variant_A_total": a_total,
                "union_filings": len(month_union),
                "aside": sum(month_aside.values()),
            }
        )
        say(
            "%s  A=%-6d union=%-5d aside=%-6d  running union=%d"
            % (lo[:7], a_total, len(month_union), sum(month_aside.values()), len(union))
        )
        _write_union(args, template, terms, forms, per_month, aside_total, union)

    per_month.sort(key=lambda r: r["month"])
    say("")
    say("MONTH    VARIANT_A  UNION  ASIDE")
    for r in per_month:
        say("%-8s %9d %6d %6d" % (r["month"], r["variant_A_total"], r["union_filings"], r["aside"]))
    say("")
    say("union filings: %d" % len(union))
    say("set aside (non-primary file_type) hits: %d" % sum(aside_total.values()))
    for k, v in aside_total.most_common(12):
        say("  aside %-12s %d" % (k, v))
    say("calls %d  bytes %d" % (sess.calls, sess.bytes))

    _write_union(args, template, terms, forms, per_month, aside_total, union)
    say("wrote %s" % union_path(args.vintage))

    if len(union) > args.union_cap:
        raise StopRun(
            "union %d exceeds the %d-filing cap; stopping before any document fetch"
            % (len(union), args.union_cap)
        )
    return 0


def month_bounds(ym):
    y, m = (int(x) for x in str(ym).split("-")[:2])
    first = date(y, m, 1)
    nxt = date(y + 1, 1, 1) if m == 12 else date(y, m + 1, 1)
    return first.isoformat(), (nxt - timedelta(days=1)).isoformat()


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def cmd_fetch(args):
    contact, source = read_contact()
    say("contact: source %s, byte length %d" % (source, len(contact.encode())))
    with open(union_path(args.vintage), "r", encoding="utf-8") as fh:
        union = json.load(fh)
    filings = union["filings"]
    if args.sample:
        rnd = random.Random(args.seed)
        filings = rnd.sample(filings, min(args.sample, len(filings)))
        say("sampling %d filings at seed %d" % (len(filings), args.seed))

    os.makedirs(docs_dir(args.vintage), exist_ok=True)
    mpath = manifest_path(args.vintage)
    manifest = load_manifest(mpath)
    manifest.setdefault("vintage", str(args.vintage))
    manifest.setdefault("docs", {})
    manifest.setdefault("failures", {})

    sess = SecSession(contact)
    t0 = time.time()
    skipped = fetched = failed = 0
    for i, f in enumerate(filings, start=1):
        adsh = f["adsh"]
        if already_fetched(manifest, adsh):
            skipped += 1
            continue
        url = build_archive_url(f["cik"], adsh, f["filename"])
        try:
            body = sess.get(url)
        except StopRun as exc:
            manifest["failures"][adsh] = str(exc)
            failed += 1
            say("FAIL %s %s" % (adsh, exc))
            if failed > args.max_failures:
                raise
            continue
        out = os.path.join(docs_dir(args.vintage), "%s.htm.gz" % adsh)
        with gzip.open(out, "wb") as gz:
            gz.write(body)
        manifest["docs"][adsh] = {
            "filename": f["filename"],
            "cik": f["cik"],
            "form": f["form"],
            "file_date": f["file_date"],
            "period_ending": f["period_ending"],
            "bytes": len(body),
            "sha256": sha256_bytes(body),
        }
        fetched += 1
        if fetched % 100 == 0:
            write_manifest(mpath, manifest)
            say(
                "  %d/%d fetched=%d skipped=%d failed=%d  %.0fs"
                % (i, len(filings), fetched, skipped, failed, time.time() - t0)
            )
    manifest["fetch"] = {
        "calls": sess.calls,
        "bytes": sess.bytes,
        "fetched": fetched,
        "skipped": skipped,
        "failed": failed,
        "wall_s": round(time.time() - t0, 1),
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_manifest(mpath, manifest)
    say(
        "DONE fetch calls=%d bytes=%d fetched=%d skipped=%d failed=%d wall=%.0fs"
        % (sess.calls, sess.bytes, fetched, skipped, failed, time.time() - t0)
    )
    return 0


def read_doc(vintage, adsh, root=None):
    path = os.path.join(docs_dir(vintage, root), "%s.htm.gz" % adsh)
    with gzip.open(path, "rb") as gz:
        return gz.read()


# ---------------------------------------------------------------------------
# parse
# ---------------------------------------------------------------------------

EVENT_COLUMNS = [
    "adsh", "cik", "form", "file_date", "period_ending", "context_ref",
    "individual_name", "individual_title",
    "adopted_flag", "terminated_flag",
    "adoption_date", "termination_date", "expiration_date",
    "adoption_date_raw", "termination_date_raw", "expiration_date_raw",
    "duration_text", "securities_agg_available",
    "rule10b51_adopted_flag", "rule10b51_terminated_flag",
    "nonrule10b51_adopted_flag", "nonrule10b51_terminated_flag",
    "raw_text",
]

_FIELD_MAP = {
    "ecd:TrdArrIndName": "individual_name",
    "ecd:TrdArrIndTitle": "individual_title",
    "ecd:TrdArrAdoptedFlag": "adopted_flag",
    "ecd:TrdArrTerminatedFlag": "terminated_flag",
    "ecd:TrdArrAdoptionDate": "adoption_date",
    "ecd:TrdArrTerminationDate": "termination_date",
    "ecd:TrdArrExpirationDate": "expiration_date",
    "ecd:TrdArrDuration": "duration_text",
    "ecd:TrdArrSecuritiesAggAvailAmt": "securities_agg_available",
    "ecd:Rule10b51ArrAdoptedFlag": "rule10b51_adopted_flag",
    "ecd:Rule10b51ArrTrmntdFlag": "rule10b51_terminated_flag",
    "ecd:NonRule10b51ArrAdoptedFlag": "nonrule10b51_adopted_flag",
    "ecd:NonRule10b51ArrTrmntdFlag": "nonrule10b51_terminated_flag",
}


def event_rows_for_filing(meta, html):
    """Rows for one filing: one per (adsh, individual). Empty when untagged."""
    facts = extract_ecd_facts(html)
    ctx_dims = extract_context_dimensions(html)
    out = []
    for grouped in facts_to_rows(facts, ctx_dims):
        row = {c: None for c in EVENT_COLUMNS}
        row["adsh"] = meta["adsh"]
        row["cik"] = meta["cik"]
        row["form"] = meta["form"]
        row["file_date"] = meta["file_date"]
        row["period_ending"] = meta["period_ending"]
        row["context_ref"] = grouped["context_ref"]
        row["raw_text"] = grouped.get("container_text") or ""
        for ecd, col in _FIELD_MAP.items():
            if ecd in grouped:
                row[col] = grouped[ecd]
            raw_key = ecd + "_raw"
            if raw_key in grouped and col.endswith("_date"):
                row[col + "_raw"] = grouped[raw_key]
        out.append(row)
    return apply_individual_inheritance(out)


def cmd_parse(args):
    mpath = manifest_path(args.vintage)
    manifest = load_manifest(mpath)
    docs = manifest.get("docs") or {}
    if not docs:
        raise StopRun("no fetched documents under vintage %s" % args.vintage)

    rows = []
    untagged = []
    for n, (adsh, meta) in enumerate(sorted(docs.items()), start=1):
        meta = dict(meta, adsh=adsh)
        html = read_doc(args.vintage, adsh)
        got = event_rows_for_filing(meta, html)
        if not got:
            untagged.append(
                {"adsh": adsh, "form": meta.get("form"), "file_date": meta.get("file_date")}
            )
        rows.extend(got)
        if n % 500 == 0:
            say("  parsed %d/%d filings, %d rows" % (n, len(docs), len(rows)))

    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    for c in df.columns:
        df[c] = df[c].astype("object")
    ev_path = os.path.join(vintage_dir(args.vintage), "events.parquet")
    df.to_parquet(ev_path, index=False)

    ut_path = os.path.join(vintage_dir(args.vintage), "untagged.csv")
    with open(ut_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["adsh", "form", "file_date"])
        w.writeheader()
        for r in untagged:
            w.writerow(r)

    flag_counts = {}
    for col in (
        "adopted_flag", "terminated_flag",
        "rule10b51_adopted_flag", "rule10b51_terminated_flag",
        "nonrule10b51_adopted_flag", "nonrule10b51_terminated_flag",
    ):
        flag_counts[col] = {
            str(k): int(v) for k, v in df[col].astype(str).value_counts().items()
        }

    manifest["parse"] = {
        "filings_parsed": len(docs),
        "filings_untagged": len(untagged),
        "filings_tagged": len(docs) - len(untagged),
        "rows": int(len(df)),
        "flag_counts": flag_counts,
        "events_parquet_sha256": sha256_file(ev_path),
        "untagged_csv_sha256": sha256_file(ut_path),
        "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    write_manifest(mpath, manifest)
    tracked = tracked_manifest_path(args.vintage)
    write_manifest(tracked, manifest)
    say("rows %d  untagged filings %d" % (len(df), len(untagged)))
    say("events.parquet sha256 %s" % manifest["parse"]["events_parquet_sha256"])
    say("manifest %s" % mpath)
    say("tracked  %s" % tracked)
    return 0


# ---------------------------------------------------------------------------
# fsn: the Financial Statement and Notes bulk dataset route
# ---------------------------------------------------------------------------

# Bare tag names as they appear in the bulk dataset (no namespace prefix).
_FSN_FIELD_MAP = {_k.split(":")[1]: _v for _k, _v in _FIELD_MAP.items()}
FSN_TEXT_BLOCK_TAG = "MtrlTermsOfTrdArrTextBlock"


def is_item408_fact(tag, version):
    """True for an Item 408 trading-arrangement fact in the ecd taxonomy.

    The version gate matters: a filer may define a custom tag of the same name
    in its own namespace, and that is not the SEC concept.
    """
    if not str(version or "").startswith("ecd"):
        return False
    t = str(tag or "")
    return t.startswith("TrdArr") or "10b51" in t


def is_item408_payload(tag, version):
    """is_item408_fact, plus the narrative text block the pivot stores.

    MtrlTermsOfTrdArrTextBlock does not START with TrdArr, so it falls outside
    the counting predicate above (which is kept exactly as specified, so the
    reported tag counts mean what they say). The pivot needs it, so the read
    path uses this wider predicate.
    """
    if is_item408_fact(tag, version):
        return True
    return str(version or "").startswith("ecd") and str(tag or "") == FSN_TEXT_BLOCK_TAG


def parse_dim_segments(segments):
    """Parse a dim.tsv segments string into {axis: member}.

    The format is a semicolon-terminated list of Axis=Member pairs, e.g.
    "Individual=DavidRanhoff;". An empty string is the entity-level context.
    """
    out = {}
    for part in str(segments or "").split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        axis, _, member = part.partition("=")
        axis = axis.strip()
        if axis:
            out[axis] = member.strip()
    return out


def fsn_individual_key(dimh, segments):
    """Row key for a bulk fact: its full canonical dimension set, else the hash.

    Same key as the HTML path produces, so the two routes are comparable.
    """
    key = canonical_dim_key(parse_dim_segments(segments))
    return key if key else dimh


def fsn_url(name):
    return "%s/%s" % (FSN_BASE, name)


def fsn_dir(root=None):
    return os.path.join(root or REPO_ROOT, "data", "edgar_item408", "fsn")


def _tsv_rows(zf, member):
    """Stream one TSV member as dicts, skipping short lines."""
    with zf.open(member) as fh:
        t = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
        hdr = t.readline().rstrip("\r\n").split("\t")
        width = len(hdr)
        for line in t:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < width:
                continue
            yield dict(zip(hdr, parts))


def read_fsn_item408(zip_path):
    """Extract the Item 408 facts, their submissions, and their dimensions.

    Returns (facts, subs, dims). Only the dimension hashes actually used by an
    Item 408 fact are resolved, so dim.tsv is read once and mostly discarded.
    """
    facts = []
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        for member, numeric in (("txt.tsv", False), ("num.tsv", True)):
            if member not in names:
                continue
            for r in _tsv_rows(zf, member):
                if not is_item408_payload(r.get("tag"), r.get("version")):
                    continue
                facts.append(
                    {
                        "adsh": r["adsh"], "tag": r["tag"], "version": r["version"],
                        "ddate": r.get("ddate"), "dimh": r.get("dimh"),
                        "value": r.get("value"), "numeric": numeric,
                    }
                )
        wanted_adsh = {f["adsh"] for f in facts}
        wanted_dimh = {f["dimh"] for f in facts}
        subs = {}
        if "sub.tsv" in names:
            for r in _tsv_rows(zf, "sub.tsv"):
                if r["adsh"] in wanted_adsh:
                    subs[r["adsh"]] = r
        dims = {}
        if "dim.tsv" in names:
            for r in _tsv_rows(zf, "dim.tsv"):
                if r.get("dimhash") in wanted_dimh:
                    dims[r["dimhash"]] = r.get("segments", "")
    return facts, subs, dims


def _iso_from_fsn(value):
    v = str(value or "").strip()
    if re.fullmatch(r"\d{8}", v):
        return _safe_iso(int(v[:4]), int(v[4:6]), int(v[6:]))
    return None


def pivot_fsn_rows(facts, subs, dims):
    """Pivot bulk facts into the R5-0 row schema, one row per (adsh, individual).

    context_ref holds the row key - the Individual dimension member, or the
    dimension hash for an entity-level row - so the schema is unchanged while
    the key is the semantically correct one.
    """
    grouped = collections.OrderedDict()
    for f in facts:
        key = (f["adsh"], fsn_individual_key(f["dimh"], dims.get(f["dimh"], "")))
        grouped.setdefault(key, []).append(f)

    rows = []
    for (adsh, key), group in grouped.items():
        sub = subs.get(adsh, {})
        row = {c: None for c in EVENT_COLUMNS}
        row["adsh"] = adsh
        row["cik"] = str(sub.get("cik", "")).zfill(10) if sub.get("cik") else None
        row["form"] = sub.get("form")
        row["file_date"] = _iso_from_fsn(sub.get("filed"))
        row["period_ending"] = _iso_from_fsn(sub.get("period"))
        row["context_ref"] = key
        row["raw_text"] = ""
        for f in group:
            tag, value = f["tag"], f["value"]
            if tag == FSN_TEXT_BLOCK_TAG:
                row["raw_text"] = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", value or "")).strip()
                continue
            col = _FSN_FIELD_MAP.get(tag)
            if col is None:
                continue
            if col.endswith("_flag"):
                row[col] = resolve_flag(None, value)
            elif col.endswith("_date"):
                row[col] = parse_ix_date(None, value)
                row[col + "_raw"] = value
            else:
                row[col] = value
        rows.append(row)
    out = []
    for _adsh, group in collections.OrderedDict(
        (a, [r for r in rows if r["adsh"] == a])
        for a in collections.OrderedDict((r["adsh"], None) for r in rows)
    ).items():
        out.extend(apply_individual_inheritance(group))
    return out


def cmd_fsn(args):
    """Download bulk archives and/or build events.parquet from them."""
    os.makedirs(fsn_dir(), exist_ok=True)
    mpath = os.path.join(fsn_dir(), "archives.json")
    manifest = load_manifest(mpath)
    manifest.setdefault("archives", {})

    if args.download:
        import requests

        contact, source = read_contact()
        say("contact: source %s, byte length %d" % (source, len(contact.encode())))
        limiter = RateLimiter(max_per_sec=args.rate)
        for name in args.archives:
            dest = os.path.join(fsn_dir(), name)
            rec = manifest["archives"].get(name)
            if rec and os.path.isfile(dest) and rec.get("bytes") == os.path.getsize(dest):
                say("skip %s (already banked)" % name)
                continue
            limiter.acquire()
            t0 = time.time()
            h = hashlib.sha256()
            n = 0
            with requests.get(
                fsn_url(name),
                headers={"User-Agent": contact, "Accept-Encoding": "gzip, deflate"},
                stream=True, timeout=600,
            ) as resp:
                if resp.status_code != 200:
                    manifest["archives"][name] = {"status": resp.status_code}
                    write_manifest(mpath, manifest)
                    say("ABSENT %s HTTP %d" % (name, resp.status_code))
                    continue
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(1 << 20):
                        fh.write(chunk)
                        h.update(chunk)
                        n += len(chunk)
            manifest["archives"][name] = {
                "bytes": n, "sha256": h.hexdigest(), "status": 200,
                "wall_s": round(time.time() - t0, 1),
            }
            write_manifest(mpath, manifest)
            say("got %-22s %12d bytes  %s  %.0fs" % (name, n, h.hexdigest()[:16], time.time() - t0))

    if not args.build:
        return 0

    all_rows = []
    coverage = []
    for name in args.archives:
        path = os.path.join(fsn_dir(), name)
        if not os.path.isfile(path):
            coverage.append({"archive": name, "present": False})
            say("missing %s" % name)
            continue
        facts, subs, dims = read_fsn_item408(path)
        rows = pivot_fsn_rows(facts, subs, dims)
        forms = collections.Counter(r["form"] for r in rows)
        coverage.append(
            {
                "archive": name, "present": True,
                "item408_facts": len(facts),
                "filings_with_any_item408_fact": len({f["adsh"] for f in facts}),
                "rows": len(rows),
                "rows_by_form": dict(forms),
            }
        )
        all_rows.extend(rows)
        say("%-22s facts %7d  filings %6d  rows %7d"
            % (name, len(facts), len({f["adsh"] for f in facts}), len(rows)))

    df = pd.DataFrame(all_rows, columns=EVENT_COLUMNS)
    for c in df.columns:
        df[c] = df[c].astype("object")
    os.makedirs(vintage_dir(args.vintage), exist_ok=True)
    ev = os.path.join(vintage_dir(args.vintage), "events.parquet")
    df.to_parquet(ev, index=False)

    vman = load_manifest(manifest_path(args.vintage))
    vman.update(
        {
            "vintage": str(args.vintage),
            "route": "bulk_dataset_fsn",
            "archives": {k: v for k, v in manifest["archives"].items() if k in args.archives},
            "coverage": coverage,
            "rows": int(len(df)),
            "events_parquet_sha256": sha256_file(ev),
            "finished_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    )
    write_manifest(manifest_path(args.vintage), vman)
    write_manifest(tracked_manifest_path(args.vintage), vman)
    say("rows %d" % len(df))
    say("events.parquet sha256 %s" % vman["events_parquet_sha256"])
    say("manifest %s" % manifest_path(args.vintage))
    say("tracked  %s" % tracked_manifest_path(args.vintage))
    return 0


# ---------------------------------------------------------------------------
# report  (counts and dates only — no price-based statistic)
# ---------------------------------------------------------------------------

def disclosure_lag_days(file_date, termination_date):
    """file_date minus termination_date in calendar days, or None."""
    if not file_date or not termination_date:
        return None
    try:
        a = date.fromisoformat(str(file_date))
        b = date.fromisoformat(str(termination_date))
    except ValueError:
        return None
    return (a - b).days


def quantiles(values, points=(0.10, 0.25, 0.50, 0.75, 0.90)):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {}
    s = pd.Series(vals, dtype="float64")
    out = {"count": len(vals), "min": int(s.min()), "max": int(s.max())}
    for p in points:
        out["p%d" % round(p * 100)] = float(s.quantile(p))
    return out


TITLE_BUCKETS = ("Director", "Chief", "President", "Officer", "Trust")


def title_mix(titles):
    """Substring counts over title text. A count, not a classification."""
    c = collections.Counter()
    for t in titles:
        s = str(t or "")
        hit = False
        for b in TITLE_BUCKETS:
            if b.lower() in s.lower():
                c[b] += 1
                hit = True
        if not hit:
            c["other"] += 1
    return c


def cmd_report(args):
    ev_path = os.path.join(vintage_dir(args.vintage), "events.parquet")
    df = pd.read_parquet(ev_path)
    manifest = load_manifest(manifest_path(args.vintage))
    say("=== 4.1 FILINGS AND ROWS ===")
    union = load_manifest(union_path(args.vintage))
    fetch = manifest.get("fetch", {})
    parse = manifest.get("parse", {})
    say("union filings              %s" % union.get("union_size"))
    say("fetched                    %s" % fetch.get("fetched"))
    say("failed                     %s" % fetch.get("failed"))
    say("filings parsed             %s" % parse.get("filings_parsed"))
    say("filings untagged           %s" % parse.get("filings_untagged"))
    say("filings tagged             %s" % parse.get("filings_tagged"))
    say("rows total                 %d" % len(df))

    adopted = df["adopted_flag"] == True          # noqa: E712
    terminated = df["terminated_flag"] == True    # noqa: E712
    say("rows adopted flag true     %d" % int(adopted.sum()))
    say("rows terminated flag true  %d" % int(terminated.sum()))
    say("rows both true             %d" % int((adopted & terminated).sum()))
    say("rows both false            %d" % int(((df["adopted_flag"] == False) & (df["terminated_flag"] == False)).sum()))  # noqa: E712
    unres = ((df["adopted_flag"] == UNRESOLVED) | (df["terminated_flag"] == UNRESOLVED))
    say("rows UNRESOLVED flags      %d" % int(unres.sum()))

    term = df[terminated].copy()
    say("distinct CIKs with >=1 termination  %d" % term["cik"].nunique())

    say("")
    say("=== 4.2 TERMINATIONS BY MONTH ===")
    say("filing month:")
    fm = term["file_date"].astype(str).str.slice(0, 7).value_counts().sort_index()
    for k, v in fm.items():
        say("  %s  %d" % (k, v))
    say("termination-date month:")
    tm = term["termination_date"].astype(str).str.slice(0, 7)
    tm = tm[tm.str.match(r"\d{4}-\d{2}")].value_counts().sort_index()
    for k, v in tm.items():
        say("  %s  %d" % (k, v))

    lags = [disclosure_lag_days(a, b) for a, b in zip(term["file_date"], term["termination_date"])]
    q = quantiles(lags)
    say("")
    say("disclosure lag, file_date minus termination date, calendar days:")
    for k in ("count", "min", "p10", "p25", "p50", "p75", "p90", "max"):
        if k in q:
            say("  %-6s %s" % (k, q[k]))
    say("  missing or unparseable termination date  %d" % sum(1 for x in lags if x is None))

    say("")
    say("=== 4.3 TITLE MIX AND FLAG MIX (terminations) ===")
    for k, v in sorted(title_mix(term["individual_title"]).items(), key=lambda kv: -kv[1]):
        say("  title contains %-9s %d" % (k, v))
    for col in ("rule10b51_terminated_flag", "nonrule10b51_terminated_flag",
                "rule10b51_adopted_flag", "nonrule10b51_adopted_flag"):
        vc = term[col].astype(str).value_counts()
        say("  %-30s %s" % (col, dict(vc)))
    agg = term["securities_agg_available"].astype(str).str.strip()
    say("  aggregate securities present   %d" % int((agg.ne("") & agg.ne("None")).sum()))
    say("  aggregate securities absent    %d" % int((agg.eq("") | agg.eq("None")).sum()))
    return 0


# ---------------------------------------------------------------------------
# join  (presence counts only — no price value enters any computation)
# ---------------------------------------------------------------------------

def build_cik_map(tickers_df):
    """SEP-table rows keyed by the CIK in secfilings -> [(permaticker, ticker)]."""
    sep = tickers_df[tickers_df["table"] == "SEP"]
    out = collections.defaultdict(list)
    for permaticker, ticker, secf, rel in zip(
        sep["permaticker"], sep["ticker"], sep["secfilings"], sep.get("relatedtickers", [None] * len(sep))
    ):
        cik = cik_from_secfilings(secf)
        if cik is None:
            continue
        out[cik].append({"permaticker": permaticker, "ticker": ticker, "relatedtickers": rel})
    return out


def cmd_join(args):
    ev_path = os.path.join(vintage_dir(args.vintage), "events.parquet")
    df = pd.read_parquet(ev_path)
    term = df[df["terminated_flag"] == True].copy()  # noqa: E712

    tickers = pd.read_parquet(args.tickers)
    cmap = build_cik_map(tickers)

    matched_rows = []
    unmatched = 0
    multi = set()
    for _i, r in term.iterrows():
        hits = cmap.get(str(r["cik"]).zfill(10)) if r["cik"] else None
        if not hits:
            unmatched += 1
            continue
        if len({h["permaticker"] for h in hits}) > 1:
            multi.add(str(r["cik"]).zfill(10))
        h = hits[0]
        matched_rows.append(
            {
                "adsh": r["adsh"], "cik": r["cik"], "file_date": r["file_date"],
                "termination_date": r["termination_date"],
                "permaticker": h["permaticker"], "ticker": h["ticker"],
                "relatedtickers": h["relatedtickers"],
            }
        )
    m = pd.DataFrame(matched_rows)
    say("=== 4.4 SHARADAR JOIN ===")
    say("terminations                %d" % len(term))
    say("matched to a SEP permaticker %d" % len(m))
    say("unmatched                   %d" % unmatched)
    say("CIKs mapping to >1 permaticker %d" % len(multi))
    if len(m):
        rel = m["relatedtickers"].astype(str).str.strip()
        say("matched rows whose permaticker has non-empty relatedtickers %d"
            % int((rel.ne("") & rel.ne("None") & rel.ne("nan")).sum()))
        say("distinct matched permatickers %d" % m["permaticker"].nunique())

    # Presence counts only. The price frame is read for (ticker, date) membership
    # and row counts; no price value enters any computation.
    say("")
    say("reading price index for presence counts (ticker, date only)")
    import pyarrow.parquet as pq

    want = set(m["ticker"].dropna().astype(str)) if len(m) else set()
    pf = pq.ParquetFile(args.stocks)
    by_ticker = collections.defaultdict(list)
    for i in range(pf.num_row_groups):
        t = pf.read_row_group(i, columns=["ticker", "date"])
        tk = t.column("ticker").to_pylist()
        dt = t.column("date").to_pylist()
        for a, b in zip(tk, dt):
            if a in want:
                by_ticker[a].append(b)
    for k in by_ticker:
        by_ticker[k] = sorted(by_ticker[k])

    def present_within(ticker, anchor, days=5):
        if not ticker or not anchor:
            return False
        try:
            a = date.fromisoformat(str(anchor))
        except ValueError:
            return False
        lo, hi = a, a + timedelta(days=days)
        for d in by_ticker.get(str(ticker), ()):
            dd = d if isinstance(d, date) else date.fromisoformat(str(d))
            if lo <= dd <= hi:
                return True
        return False

    def rows_before(ticker, anchor):
        if not ticker or not anchor:
            return 0
        try:
            a = date.fromisoformat(str(anchor))
        except ValueError:
            return 0
        n = 0
        for d in by_ticker.get(str(ticker), ()):
            dd = d if isinstance(d, date) else date.fromisoformat(str(d))
            if dd < a:
                n += 1
        return n

    if len(m):
        near_term = sum(present_within(r.ticker, r.termination_date) for r in m.itertuples())
        near_file = sum(present_within(r.ticker, r.file_date) for r in m.itertuples())
        deep = sum(1 for r in m.itertuples() if rows_before(r.ticker, r.termination_date) >= 250)
        say("matched terminations with a price row on or within 5 days after the termination date %d"
            % near_term)
        say("matched terminations with a price row on or within 5 days after the file date        %d"
            % near_file)
        say("matched terminations whose ticker has >= 250 price rows dated before the termination date %d (%.3f)"
            % (deep, deep / len(m)))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(description="EDGAR Item 408 10b5-1 event table")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", help="size the problem; write the union filing list")
    c.add_argument("--vintage", required=True)
    c.add_argument("--start", default="2023-04-01")
    c.add_argument("--end", default="2026-10-31")
    c.add_argument("--variants", default=None, help="run the four-variant probe on one YYYY-MM")
    c.add_argument("--and-template", default='"10b5-1" {term}')
    c.add_argument("--union-cap", type=int, default=12000)
    c.set_defaults(func=cmd_count)

    f = sub.add_parser("fetch", help="fetch primary documents for the union")
    f.add_argument("--vintage", required=True)
    f.add_argument("--sample", type=int, default=0)
    f.add_argument("--seed", type=int, default=20261017)
    f.add_argument("--max-failures", type=int, default=50)
    f.set_defaults(func=cmd_fetch)

    p = sub.add_parser("parse", help="parse ecd:* facts into events.parquet")
    p.add_argument("--vintage", required=True)
    p.set_defaults(func=cmd_parse)

    r = sub.add_parser("report", help="counts and dates only")
    r.add_argument("--vintage", required=True)
    r.set_defaults(func=cmd_report)

    n = sub.add_parser("fsn", help="bulk Financial Statement and Notes dataset route")
    n.add_argument("--vintage", required=True)
    n.add_argument("--archives", nargs="+", required=True)
    n.add_argument("--download", action="store_true")
    n.add_argument("--build", action="store_true")
    n.add_argument("--rate", type=int, default=6)
    n.set_defaults(func=cmd_fsn)

    j = sub.add_parser("join", help="join issuers to the Sharadar panel; presence counts only")
    j.add_argument("--vintage", required=True)
    j.add_argument("--tickers", required=True)
    j.add_argument("--stocks", required=True)
    j.set_defaults(func=cmd_join)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StopRun as exc:
        say("STOP: %s" % exc)
        return 3


if __name__ == "__main__":
    sys.exit(main())
