"""
================================================================================
Script:       item408_event_study_r5.py
Version:      1.1.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R5 engine: Item 408 Rule 10b5-1 plan TERMINATIONS as an
              event study on the public disclosure date, with the ISSUER
              PERMUTATION null at fixed dates and both registered controls.
Entry Point:  No — imported by tools/run_t_r5.py
Dependencies: stdlib, numpy, pandas, pyarrow. Imports nothing from any other
              repository; in particular it imports nothing from the factory tree.
Last Sprint:  Sprint R5-3 — T-R5 v2: permutation null, re-seal, controls,
              half-A read (2026-11-05)
================================================================================

Every parameter comes from the sealed registration JSON, whose sha256 must match
the one passed in; load_registration refuses otherwise. The integers the
registration states in prose rather than in a numeric field (the 60-day voted
window, the 20 bps round trip, the two price floors, the 250-row history bar,
the 1000-ticker calendar floor, the 5-return delisting floor, the 90 per cent
retention rule, the materiality bar, the 200-event minimum, the MDE z) are
declared here and pinned against that prose by a test, so the code and the seal
cannot drift apart silently.

A DISCLOSURE is the unit, not an individual and not a plan. Several officers
terminating in one 10-Q are one event, because one filing is one piece of public
news. D is the filing date rolled forward to a trading day; the trade the vote
prices is an entry at the close of D+1 and an exit at the close of D+61, so it
reads nothing a reader of the filing could not have read.

THE CALENDAR is not every date in the snapshot. Sharadar carries sparse dates
that only a handful of tickers trade on; treating one as a trading day shifts
every window that straddles it. A trading day is a date on which at least
CALENDAR_MIN_TICKERS tickers have a row, and rows on any other date are dropped.

A STOCK THAT STOPS TRADING inside the window is not a hole. Its position ends at
its last close, which is the return a holder realised; the event is scored to
that close provided at least MIN_DELIST_RETURNS daily returns exist. A missing
trading day while the ticker goes on trading IS a hole, and drops the event.

THE NULL permutes issuers across fixed dates. Each replicate draws a uniformly
random bijection of the n events and scores issuer sigma(i) at date D_i, so
every disclosure date stays exactly where it is and the calendar clustering of
filings is preserved exactly. A pair the issuer cannot cover is dropped and
counted; a pair landing on the issuer's own real voted window is KEPT and
counted as contaminated, because contamination can only pull the null toward a
real effect — it costs power and cannot manufacture a finding.

The clause counts are the honest part of resolution: every filter reports how
many rows it removed, in the registered order, before any return is formed.

ONE DISCLOSED INTERNAL USE OF THE OBSERVED SET. The registered positive control
plants a drift on the real events' own voted-window returns, so forming the
planted mean necessarily forms the observed mean inside it. That figure is used
and discarded: it is returned by no public function of this module, printed by
no log line and written into no receipt. The same accommodation R4 made for the
buy-and-hold Sharpe it needed to calibrate q.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pac
import pyarrow.parquet as pq

VERSION = "1.1.0"

# --- stated in the registration as prose; pinned by test_registration_prose_pins
FORMS = ("10-Q", "10-K")              # "form in {10-Q, 10-K}"
PURCHASE_WORDS = ("purchase", "buy")  # "contains 'purchase' or 'buy'"
SALE_WORDS = ("sale", "sell", "sold")  # "none of 'sale', 'sell', 'sold'"
HISTORY_MIN_ROWS = 250                # "at least 250 bulk_stocks rows"
PRICE_FLOOR = 2.00                    # "closeunadj on D+1 >= 2.00 USD"
DOLLAR_VOL_FLOOR = 500000.0           # "... >= 500000 USD"
DOLLAR_VOL_WINDOW = 20                # "over the 20 trading days ending D"
COMPLETE_FROM = 1                     # "from D+1 to D+61 inclusive"
COMPLETE_TO = 61
CALENDAR_MIN_TICKERS = 1000           # "at least 1000 tickers have a
                                      #  bulk_stocks row"
MIN_DELIST_RETURNS = 5                # "provided at least 5 daily returns exist"
MARKET_PRICE_FLOOR = 1.00             # "closeunadj >= 1.00 USD on the prior day"
MARKET_MIN_PRIOR_ROWS = 250           # "at least 250 prior rows"
VOTE_FROM = 2                         # "the abnormal daily log returns from D+2
VOTE_TO = 61                          #  to the exit", capped at D+61
VOTE_N = 60                           # a complete window's return count
SHORT_TO = 21                         # "the 20-day window D+2 to D+21"
SHORT_N = 20
PRETERM_LAG = 21                      # "from the close of T-21 to the close of T"
PATH_POINTS = (5, 10, 20, 40, 60)     # "at 5, 10, 20, 40 and 60 days after D+1"
COST_ROUND_TRIP_BPS = 20.0            # "20 bps round trip (10 a side)"
RETENTION_MIN = 0.90                  # "fewer than 90 percent of events is redrawn"
Z80 = 0.8416                          # "+ 0.8416 * sd(null mCAR)"
MIN_EVENTS = 200                      # "fewer than 200 events after all filters"
MATERIALITY = 0.015                   # "mCAR >= 0.015 net"
SUGGESTIVE_HI = 0.10                  # "p in (0.05, 0.10]"

COST_FRACTION = COST_ROUND_TRIP_BPS * 1e-4

TITLE_PATTERNS = (
    ("chief_executive", ("chief executive", "ceo")),
    ("chief_financial", ("chief financial", "cfo")),
    ("other_chief", ("chief ",)),
    ("chair", ("chair",)),
    ("director", ("director",)),
    ("president", ("president",)),
    ("vice_president", ("vice president", " vp", "vp ")),
    ("officer", ("officer",)),
)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_registration(path, expected_sha):
    """Load the seal and refuse any file whose sha256 is not the one passed in."""
    path = Path(path)
    got = sha256_file(path)
    if got != expected_sha:
        raise AssertionError(
            "registration sha256 mismatch: file %s has %s, --sha said %s"
            % (path, got, expected_sha))
    reg = json.loads(path.read_text())
    reg["_sha256"] = got
    reg["_path"] = str(path)
    return reg


def half_bounds(reg):
    """The three date regimes, read out of the registration's halves prose."""
    s = reg["event_definition"]["halves"]
    d = re.findall(r"\d{4}-\d{2}-\d{2}", s)
    if len(d) != 5:
        raise AssertionError(
            "the halves prose must carry exactly five dates, found %d: %r"
            % (len(d), d))
    return {
        "half_A": (pd.Timestamp(d[0]), pd.Timestamp(d[1])),
        "half_B": (pd.Timestamp(d[2]), pd.Timestamp(d[3])),
        "holdout": (pd.Timestamp(d[4]), None),
    }


def calendar_min_tickers(reg):
    """The calendar's ticker floor, read out of the seal's own calendar prose.

    The seal states it in prose rather than a numeric field, so it is parsed
    here and the parsed value is what load_prices is driven with; the module
    constant is pinned against the same prose by a test.
    """
    s = reg["data"]["calendar"]
    m = re.search(r"at least (\d+) tickers have a bulk_stocks row", s)
    if not m:
        raise AssertionError(
            "the calendar prose must state 'at least <n> tickers have a "
            "bulk_stocks row', found: %r" % s)
    return int(m.group(1))


# ---------------------------------------------------------------------------
# Prices: one compact column-store over the frozen Sharadar snapshot
# ---------------------------------------------------------------------------
def _col_f8(path, name):
    t = pq.read_table(path, columns=[name]).column(name).combine_chunks()
    return pac.cast(t, "float64").to_numpy(zero_copy_only=False)


def trading_day_calendar(day, min_tickers=CALENDAR_MIN_TICKERS):
    """The calendar is the dates carrying at least `min_tickers` rows.

    Sharadar's snapshot holds sparse dates that only a handful of tickers trade
    on. A date like that is not a trading day, and a row on it is dropped: if it
    were kept, every window straddling it would be shifted by a day that the
    market never had. Returns (cal, keep, stats).
    """
    day = np.asarray(day)
    uniq, cnt = np.unique(day, return_counts=True)
    cal = uniq[cnt >= int(min_tickers)]
    if len(cal) == 0:
        raise AssertionError(
            "no date in the snapshot carries %d tickers" % int(min_tickers))
    j = np.searchsorted(cal, day, "left")
    keep = (j < len(cal)) & (cal[np.minimum(j, len(cal) - 1)] == day)
    stats = {
        "min_tickers": int(min_tickers),
        "dates_in_snapshot": int(len(uniq)),
        "trading_days": int(len(cal)),
        "dates_excluded": int(len(uniq) - len(cal)),
        "rows_on_excluded_dates": int((~keep).sum()),
        "tickers_on_the_thinnest_trading_day": int(cnt[cnt >= int(min_tickers)].min()),
        "tickers_on_the_fattest_excluded_date": (
            int(cnt[cnt < int(min_tickers)].max())
            if (cnt < int(min_tickers)).any() else 0),
    }
    return cal, keep, stats


class PriceBook:
    """Sharadar SEP, sorted by (ticker, date), plus the trading-day calendar.

    The calendar is the set of dates on which at least CALENDAR_MIN_TICKERS
    tickers have a row, as the v2 seal says, so D+k and T-k are positions in
    that array and nothing else. Rows on excluded dates are not in the book.
    """

    def __init__(self, vocab, code, dayidx, closeadj, closeunadj, volume, cal,
                 cal_stats=None):
        self.vocab = vocab
        self.code = code
        self.dayidx = dayidx
        self.closeadj = closeadj
        self.closeunadj = closeunadj
        self.volume = volume
        self.cal = cal                      # int32 days since epoch, sorted
        self.L = len(cal)
        self.cal_stats = cal_stats or {}
        self.index_of = {t: i for i, t in enumerate(vocab)}
        n = len(vocab)
        self.start = np.searchsorted(code, np.arange(n, dtype=np.int32), "left")
        self.stop = np.searchsorted(code, np.arange(n, dtype=np.int32), "right")

    # -- calendar -------------------------------------------------------
    def roll_forward(self, ts):
        """Index of the first trading day at or after ts, or None past the end."""
        d = np.int32((pd.Timestamp(ts) - pd.Timestamp("1970-01-01")).days)
        j = int(np.searchsorted(self.cal, d, "left"))
        return None if j >= self.L else j

    def date_str(self, j):
        return str(pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(self.cal[j])))[:10]

    # -- per-ticker -----------------------------------------------------
    def slice_of(self, ticker):
        c = self.index_of.get(ticker)
        if c is None:
            return None
        a, b = int(self.start[c]), int(self.stop[c])
        return None if b <= a else (a, b)

    def rows_before(self, ticker, ts):
        """Rows for this ticker dated strictly before ts. 0 if the ticker is absent."""
        sl = self.slice_of(ticker)
        if sl is None:
            return 0
        a, b = sl
        d = np.int32((pd.Timestamp(ts) - pd.Timestamp("1970-01-01")).days)
        j = int(np.searchsorted(self.cal, d, "left"))
        return int(np.searchsorted(self.dayidx[a:b], j, "left"))

    def rows_before_idx(self, ticker, j):
        """Rows for this ticker strictly before trading-day index j."""
        sl = self.slice_of(ticker)
        if sl is None:
            return 0
        a, b = sl
        return int(np.searchsorted(self.dayidx[a:b], j, "left"))

    def has_row_on(self, ticker, j):
        sl = self.slice_of(ticker)
        if sl is None:
            return False
        a, b = sl
        k = int(np.searchsorted(self.dayidx[a:b], j, "left"))
        return bool(k < (b - a) and self.dayidx[a + k] == j)

    def last_row_idx(self, ticker):
        """Trading-day index of the ticker's LAST row, or -1 if it has none."""
        sl = self.slice_of(ticker)
        if sl is None:
            return -1
        a, b = sl
        return int(self.dayidx[b - 1])

    def value_on(self, ticker, j, field):
        sl = self.slice_of(ticker)
        if sl is None:
            return np.nan
        a, b = sl
        k = int(np.searchsorted(self.dayidx[a:b], j, "left"))
        if not (k < (b - a) and self.dayidx[a + k] == j):
            return np.nan
        return float(getattr(self, field)[a + k])

    def dollar_volume_median(self, ticker, j_end, window=DOLLAR_VOL_WINDOW):
        """Median of closeunadj * volume over the `window` trading days ending j_end."""
        sl = self.slice_of(ticker)
        if sl is None:
            return np.nan
        a, b = sl
        lo = j_end - window + 1
        d = self.dayidx[a:b]
        i0 = int(np.searchsorted(d, lo, "left"))
        i1 = int(np.searchsorted(d, j_end, "right"))
        if i1 <= i0:
            return np.nan
        v = (self.closeunadj[a + i0:a + i1].astype(np.float64)
             * self.volume[a + i0:a + i1].astype(np.float64))
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else np.nan

    def complete_between(self, ticker, j0, j1):
        """True iff closeadj is present and finite on EVERY trading day j0..j1."""
        sl = self.slice_of(ticker)
        if sl is None:
            return False
        if j0 < 0 or j1 >= self.L or j1 < j0:
            return False
        a, b = sl
        d = self.dayidx[a:b]
        i0 = int(np.searchsorted(d, j0, "left"))
        i1 = int(np.searchsorted(d, j1, "right"))
        want = j1 - j0 + 1
        if (i1 - i0) != want:
            return False
        if not np.array_equal(d[i0:i1], np.arange(j0, j1 + 1)):
            return False
        return bool(np.isfinite(self.closeadj[a + i0:a + i1]).all()
                    and (self.closeadj[a + i0:a + i1] > 0).all())


def load_prices(path, min_tickers=CALENDAR_MIN_TICKERS):
    """Build the PriceBook from bulk_stocks.parquet under the v2 calendar.

    Five columns, cast at read. Rows on dates carrying fewer than `min_tickers`
    tickers are dropped before anything else is computed, so they can never
    become a window position.
    """
    tk = pq.read_table(path, columns=["ticker"], read_dictionary=["ticker"])
    ca = tk.column("ticker").combine_chunks()
    code = ca.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    vocab = np.asarray(ca.dictionary.to_pylist(), dtype=object)
    del tk, ca

    dt = pq.read_table(path, columns=["date"]).column("date").combine_chunks()
    day = pac.cast(pac.cast(dt, "date32"), "int32").to_numpy(
        zero_copy_only=False).astype(np.int32)
    del dt

    cal, keep, cal_stats = trading_day_calendar(day, min_tickers)
    dayidx = np.searchsorted(cal, day[keep]).astype(np.int32)
    code = code[keep]
    del day

    closeadj = _col_f8(path, "closeadj")[keep]
    closeunadj = _col_f8(path, "closeunadj")[keep].astype(np.float32)
    volume = _col_f8(path, "volume")[keep].astype(np.float32)
    del keep

    key = code.astype(np.int64) * np.int64(len(cal)) + dayidx
    order = np.argsort(key, kind="stable")
    del key
    return PriceBook(vocab, code[order], dayidx[order], closeadj[order],
                     closeunadj[order], volume[order], cal, cal_stats)


# ---------------------------------------------------------------------------
# The equal-weight market index
# ---------------------------------------------------------------------------
def build_market_index(book):
    """Equal-weight mean daily log return across every eligible ticker-day.

    A ticker's return on trading day j enters only if its previous ROW is the
    previous trading day, its closeunadj on that previous day clears the floor,
    and it already has at least MARKET_MIN_PRIOR_ROWS rows behind it. Not an
    event statistic: it is built from the whole snapshot under the v2 calendar
    and knows nothing of any disclosure.
    """
    L = book.L
    s = np.zeros(L)
    n = np.zeros(L, dtype=np.int64)
    for c in range(len(book.vocab)):
        a, b = int(book.start[c]), int(book.stop[c])
        if b - a < MARKET_MIN_PRIOR_ROWS + 1:
            continue
        d = book.dayidx[a:b]
        ca = book.closeadj[a:b]
        cu = book.closeunadj[a:b]
        adjacent = (d[1:] - d[:-1]) == 1
        prior_ok = cu[:-1] >= MARKET_PRICE_FLOOR
        enough = np.arange(1, b - a) >= MARKET_MIN_PRIOR_ROWS
        pos = (ca[1:] > 0) & (ca[:-1] > 0) & np.isfinite(ca[1:]) & np.isfinite(ca[:-1])
        keep = adjacent & prior_ok & enough & pos
        if not keep.any():
            continue
        r = np.log(ca[1:][keep] / ca[:-1][keep])
        j = d[1:][keep]
        np.add.at(s, j, r)
        np.add.at(n, j, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return {"mean_log_return": m, "n_tickers": n}


def market_frame(book, market):
    return pd.DataFrame({
        "date": [book.date_str(j) for j in range(book.L)],
        "mean_log_return": market["mean_log_return"],
        "n_tickers": market["n_tickers"].astype(np.int64),
    })


# ---------------------------------------------------------------------------
# Event resolution — one clause at a time, each with its own count
# ---------------------------------------------------------------------------
def is_true(v):
    """The flag columns carry True, False and None; only True is a flag."""
    return v is True


def parse_tagged_date(s):
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return pd.NaT
    return pd.to_datetime(str(s).strip(), format="%Y-%m-%d", errors="coerce")


def direction_class(text):
    """purchase_only | sale_only | both | neither, on the lower-cased text block."""
    t = "" if text is None else str(text).lower()
    p = any(w in t for w in PURCHASE_WORDS)
    s = any(w in t for w in SALE_WORDS)
    if p and not s:
        return "purchase_only"
    if s and not p:
        return "sale_only"
    return "both" if (p and s) else "neither"


def extract_cik(secfilings):
    """The CIK out of a browse-edgar URL, as an int. None when there is none."""
    if secfilings is None:
        return None
    m = re.search(r"CIK=0*(\d+)", str(secfilings))
    return int(m.group(1)) if m else None


def resolve_rows(ev):
    """Apply the registered ROW clauses in order. Returns (kept, counts)."""
    counts = {"rows_in_vintage": int(len(ev))}

    term = (ev["rule10b51_terminated_flag"].map(is_true).astype(bool)
            | ev["nonrule10b51_terminated_flag"].map(is_true).astype(bool))
    m = term
    counts["A_termination_flag"] = int(m.sum())

    adopt = (ev["rule10b51_adopted_flag"].map(is_true).astype(bool)
             | ev["nonrule10b51_adopted_flag"].map(is_true).astype(bool))
    m = m & ~adopt
    counts["B_no_adoption_flag"] = int(m.sum())
    counts["B_removed_modifications"] = counts["A_termination_flag"] - int(m.sum())

    T = ev["termination_date"].map(parse_tagged_date)
    m = m & T.notna()
    counts["C_parseable_T"] = int(m.sum())

    F = pd.to_datetime(ev["file_date"], format="%Y-%m-%d", errors="coerce")
    lag = (F - T).dt.days
    m = m & (lag >= 0)
    counts["D_lag_ge_zero"] = int(m.sum())

    m = m & ev["form"].isin(list(FORMS))
    counts["E_form_10Q_or_10K"] = int(m.sum())

    cls = ev["raw_text"].map(direction_class)
    split = cls[m].value_counts().to_dict()
    counts["F_direction_split"] = {k: int(split.get(k, 0)) for k in
                                   ("purchase_only", "sale_only", "both", "neither")}
    m = m & (cls != "purchase_only")
    counts["F_after_purchase_exclusion"] = int(m.sum())

    kept = ev.loc[m].copy()
    kept["T"] = T[m]
    kept["F"] = F[m]
    kept["lag_days"] = lag[m]
    kept["direction"] = cls[m]
    return kept, counts


def collapse_to_filings(kept, book, counts):
    """One event per filing: several individuals in one 10-Q are one disclosure."""
    rows = []
    for adsh, g in kept.groupby("adsh", sort=True):
        f = g["F"].iloc[0]
        d = book.roll_forward(f)
        titles = " | ".join(sorted({str(x) for x in g["individual_title"]
                                    if x is not None and str(x).strip()}))
        rows.append({
            "adsh": adsh, "cik": int(str(g["cik"].iloc[0])),
            "form": g["form"].iloc[0], "file_date": f,
            "T": g["T"].min(), "T_last": g["T"].max(),
            "n_individuals": int(len(g)),
            "titles": titles,
            "direction_mix": ",".join(sorted(set(g["direction"]))),
            "D_idx": (-1 if d is None else int(d)),
        })
    fil = pd.DataFrame(rows)
    counts["G_one_event_per_filing"] = int(len(fil))
    fil = fil.loc[fil["D_idx"] >= 0].copy()
    counts["G_filing_date_on_the_calendar"] = int(len(fil))
    return fil, counts


def map_issuers(fil, sep, book, counts):
    """CIK -> the SEP permaticker whose ticker actually trades around D."""
    sep = sep.loc[sep["table"] == "SEP"].copy()
    sep["cik_int"] = sep["secfilings"].map(extract_cik)
    by_cik = {}
    for cik, g in sep.dropna(subset=["cik_int"]).groupby("cik_int", sort=False):
        by_cik[int(cik)] = list(zip(g["permaticker"].astype(str),
                                    g["ticker"].astype(str)))

    out, matched, multi, dropped, resolved_multi = [], 0, 0, 0, 0
    for r in fil.to_dict("records"):
        cands = by_cik.get(r["cik"], [])
        if not cands:
            dropped += 1
            continue
        if len(cands) > 1:
            multi += 1
            alive = [c for c in cands
                     if book.has_row_on(c[1], r["D_idx"] + 1)]
            if len(alive) > 1:
                alive = sorted(
                    alive,
                    key=lambda c: (-book.rows_before_idx(c[1], r["D_idx"]), c[1]))
                resolved_multi += 1
            if not alive:
                dropped += 1
                continue
            pick = alive[0]
        else:
            pick = cands[0]
        rr = dict(r)
        rr["permaticker"], rr["ticker"] = pick
        out.append(rr)
        matched += 1
    mp = pd.DataFrame(out)
    counts["H_mapping"] = {"matched": int(matched), "multi_permaticker": int(multi),
                           "multi_resolved_by_row_count": int(resolved_multi),
                           "dropped_no_usable_ticker": int(dropped)}
    if len(mp):
        before = len(mp)
        mp = mp.sort_values(["permaticker", "D_idx", "adsh"])
        mp = mp.drop_duplicates(subset=["permaticker", "D_idx"], keep="first")
        counts["I_unique_on_permaticker_D"] = int(len(mp))
        counts["I_collapsed"] = int(before - len(mp))
    else:
        counts["I_unique_on_permaticker_D"] = 0
        counts["I_collapsed"] = 0
    return mp.reset_index(drop=True), counts
def classify_window(book, ticker, d_idx):
    """COMPLETE, DELISTED-IN-WINDOW, or a drop reason, plus the exit index.

    The registered completeness rule in three outcomes:
      complete  — closeadj on every trading day D+1..D+61; exit at D+61.
      delisted  — the ticker's LAST row falls inside D+2..D+61, every day from
                  D+1 to it is present, and at least MIN_DELIST_RETURNS daily
                  returns exist; exit at that last close, which is the return a
                  holder realised.
      hole      — a trading day is missing while the ticker goes on trading, or
                  a day is missing before an in-window delisting.
      no_entry  — no row on D+1, so the position cannot be opened.
      too_short — the ticker stops too soon to yield MIN_DELIST_RETURNS returns.
      absent    — the ticker has no rows at all.
    Returns (status, exit_idx); exit_idx is -1 on every drop.
    """
    entry = d_idx + COMPLETE_FROM
    cap = d_idx + COMPLETE_TO
    if book.slice_of(ticker) is None:
        return "absent", -1
    last = book.last_row_idx(ticker)
    if last < 0:
        return "absent", -1
    if entry < 0 or entry >= book.L:
        return "no_entry", -1
    if not book.has_row_on(ticker, entry):
        return "no_entry", -1
    exit_ = min(cap, last)
    if exit_ >= book.L:
        return "too_short", -1
    if (exit_ - entry) < MIN_DELIST_RETURNS:
        return "too_short", -1
    if not book.complete_between(ticker, entry, exit_):
        return "hole", -1
    return ("complete" if last >= cap else "delisted"), int(exit_)


def apply_price_filters(mp, book, counts):
    """History, tradability and completeness, each counted separately.

    Completeness is now three-way: a complete window, an in-window delisting
    scored to its last close, or a drop (hole, no entry, too short, absent).
    """
    keep = []
    n_hist = n_price = n_dv = 0
    comp = {"complete": 0, "delisted": 0, "hole": 0, "no_entry": 0,
            "too_short": 0, "absent": 0}
    for r in mp.to_dict("records"):
        if book.rows_before(r["ticker"], r["T"]) < HISTORY_MIN_ROWS:
            continue
        n_hist += 1
        j1 = r["D_idx"] + 1
        px = book.value_on(r["ticker"], j1, "closeunadj")
        if not (np.isfinite(px) and px >= PRICE_FLOOR):
            continue
        n_price += 1
        dv = book.dollar_volume_median(r["ticker"], r["D_idx"])
        if not (np.isfinite(dv) and dv >= DOLLAR_VOL_FLOOR):
            continue
        n_dv += 1
        status, exit_idx = classify_window(book, r["ticker"], r["D_idx"])
        comp[status] += 1
        if status not in ("complete", "delisted"):
            continue
        rr = dict(r)
        rr["window_status"] = status
        rr["exit_idx"] = int(exit_idx)
        rr["n_returns"] = int(exit_idx - (r["D_idx"] + COMPLETE_FROM))
        keep.append(rr)
    counts["J_history_250_rows_before_T"] = int(n_hist)
    counts["K_price_floor_on_D_plus_1"] = int(n_price)
    counts["K_dollar_volume_floor"] = int(n_dv)
    counts["L_window_status"] = {k: int(v) for k, v in comp.items()}
    counts["L_complete_or_delisted"] = int(comp["complete"] + comp["delisted"])
    counts["L_complete_D1_to_D61"] = int(comp["complete"])
    counts["L_delisted_in_window"] = int(comp["delisted"])
    counts["L_hole"] = int(comp["hole"])
    return pd.DataFrame(keep).reset_index(drop=True), counts


def resolve_events(ev, sep, book):
    """The whole registered resolution, in clause order, counts at every step."""
    kept, counts = resolve_rows(ev)
    fil, counts = collapse_to_filings(kept, book, counts)
    mp, counts = map_issuers(fil, sep, book, counts)
    res, counts = apply_price_filters(mp, book, counts)
    return res, counts


def split_halves(res, reg):
    """Half A, half B and the holdout by D's calendar date. Half A is this look."""
    hb = half_bounds(reg)
    d = pd.to_datetime(res["file_date"])
    a_lo, a_hi = hb["half_A"]
    b_lo, b_hi = hb["half_B"]
    h_lo, _ = hb["holdout"]
    return {
        "half_A": res.loc[(d >= a_lo) & (d <= a_hi)].reset_index(drop=True),
        "half_B": res.loc[(d >= b_lo) & (d <= b_hi)].reset_index(drop=True),
        "holdout": res.loc[d >= h_lo].reset_index(drop=True),
    }


# ---------------------------------------------------------------------------
# Abnormal returns and window sums
# ---------------------------------------------------------------------------
def abnormal_series(book, ticker, market):
    """(ab, ok) over the whole calendar: stock log return minus the market's.

    A return exists on trading day j only when the ticker has rows on j and on
    j-1, so nothing is ever formed across a hole in a ticker's history.
    """
    L = book.L
    ab = np.zeros(L)
    ok = np.zeros(L, dtype=bool)
    sl = book.slice_of(ticker)
    if sl is None:
        return ab, ok
    a, b = sl
    d = book.dayidx[a:b]
    ca = book.closeadj[a:b]
    if b - a < 2:
        return ab, ok
    adjacent = (d[1:] - d[:-1]) == 1
    pos = (ca[1:] > 0) & (ca[:-1] > 0) & np.isfinite(ca[1:]) & np.isfinite(ca[:-1])
    j = d[1:]
    mkt = market["mean_log_return"][j]
    good = adjacent & pos & np.isfinite(mkt)
    if good.any():
        r = np.log(ca[1:][good] / ca[:-1][good]) - mkt[good]
        ab[j[good]] = r
        ok[j[good]] = True
    return ab, ok


class WindowBook:
    """Cumulative abnormal returns per event ticker, so any window is O(1).

    Also carries each issuer's row presence and its last row, because the v2
    exit is min(D+61, that last row) and a permuted pair can only be scored on
    an issuer whose history actually covers the date it is handed.
    """

    def __init__(self, book, tickers, market):
        self.tickers = list(dict.fromkeys(tickers))
        self.gidx = {t: i for i, t in enumerate(self.tickers)}
        L = book.L
        self.L = L
        G = len(self.tickers)
        self.cum_ab = np.zeros((G, L + 1))
        self.cum_ok = np.zeros((G, L + 1), dtype=np.int32)
        self.cum_row = np.zeros((G, L + 1), dtype=np.int32)
        self.last_idx = np.full(G, -1, dtype=np.int64)
        for t in self.tickers:
            g = self.gidx[t]
            ab, ok = abnormal_series(book, t, market)
            self.cum_ab[g, 1:] = np.cumsum(ab)
            self.cum_ok[g, 1:] = np.cumsum(ok.astype(np.int32))
            pres = np.zeros(L, dtype=np.int32)
            sl = book.slice_of(t)
            if sl is not None:
                a, b = sl
                pres[book.dayidx[a:b]] = 1
                self.last_idx[g] = int(book.dayidx[b - 1])
            self.cum_row[g, 1:] = np.cumsum(pres)
        self.real_vote_cum = np.zeros((G, L + 1), dtype=np.int32)

    def window(self, groups, a, b):
        """Sum and availability of the abnormal returns over a..b inclusive."""
        s = self.cum_ab[groups, b + 1] - self.cum_ab[groups, a]
        n = self.cum_ok[groups, b + 1] - self.cum_ok[groups, a]
        return s, n

    def row_on(self, groups, j):
        """Row presence, elementwise, for (group, trading-day) pairs."""
        return (self.cum_row[groups, j + 1] - self.cum_row[groups, j]) == 1

    def set_real_voted(self, groups, d_idx, exit_idx):
        """Mark each real event's own voted window D+2..exit, per issuer group.

        Used only to COUNT contamination in the permutation null. Nothing is
        dropped on its account: see RULE-NO-DROP.
        """
        G, L = len(self.tickers), self.L
        blk = np.zeros((G, L), dtype=bool)
        for g, di, ei in zip(groups, d_idx, exit_idx):
            lo = max(0, int(di) + VOTE_FROM)
            hi = min(L - 1, int(ei))
            if hi >= lo:
                blk[g, lo:hi + 1] = True
        self.real_vote_cum[:, 1:] = np.cumsum(blk.astype(np.int32), axis=1)

    def contaminated(self, groups, a, b):
        """True where a..b touches that issuer's own real voted window."""
        return (self.real_vote_cum[groups, b + 1]
                - self.real_vote_cum[groups, a]) > 0


def score_pairs(wb, groups, d_idx):
    """Score (issuer, date) pairs under the registered window, cost and rules.

    The one scoring path, used by the observed read and by every null replicate,
    so the null cannot be scored under rules the observation was not. Entry at
    the close of d+1, exit at the close of min(d+61, the issuer's last row). A
    pair is not covered when the issuer has no row on d+1, when fewer than
    MIN_DELIST_RETURNS returns are available, or when a return inside the window
    is missing. Returns (net, n_returns, ok, exit_idx).
    """
    g = np.asarray(groups, dtype=np.int64)
    d = np.asarray(d_idx, dtype=np.int64)
    L = wb.L
    entry = d + COMPLETE_FROM
    cap = d + VOTE_TO
    last = wb.last_idx[g]
    exit_ = np.minimum(cap, last)
    n_exp = exit_ - entry
    ok = ((last >= 0) & (entry >= 0) & (entry < L) & (exit_ >= 0) & (exit_ < L)
          & (n_exp >= MIN_DELIST_RETURNS))
    e_safe = np.clip(entry, 0, L - 1)
    ok &= wb.row_on(g, e_safe)
    a = np.clip(entry + 1, 0, L - 1)
    b = np.clip(exit_, 0, L - 1)
    s, n = wb.window(g, a, b)
    ok &= (n == n_exp)
    net = np.where(ok, s - COST_FRACTION, np.nan)
    return net, n.astype(np.int64), ok, exit_


def voted_returns(wb, groups, d_idx):
    """Net voted-window abnormal return per event, NaN where not covered."""
    return score_pairs(wb, groups, d_idx)[0]


# ---------------------------------------------------------------------------
# The voting null: issuer permutation at fixed dates
# ---------------------------------------------------------------------------
def draw_permutation(rng, n):
    """One uniformly random bijection of range(n). Every date stays put."""
    return rng.permutation(int(n))


def permuted_mcar(wb, groups, d_idx, sigma):
    """mCAR of one replicate: issuer sigma(i) scored at date D_i.

    Returns (mcar, retained_fraction, n_used, contaminated_share). A pair the
    issuer cannot cover is dropped and counted; a pair landing on that issuer's
    own real voted window is KEPT and counted as contaminated.
    """
    g = np.asarray(groups, dtype=np.int64)[np.asarray(sigma, dtype=np.int64)]
    d = np.asarray(d_idx, dtype=np.int64)
    n_events = len(d)
    net, _, ok, exit_ = score_pairs(wb, g, d)
    used = int(ok.sum())
    if used == 0:
        return np.nan, 0.0, 0, 0.0
    con = wb.contaminated(g[ok], d[ok] + VOTE_FROM, exit_[ok])
    return (float(net[ok].mean()), used / float(n_events), used,
            float(con.mean()))


def permutation_null(wb, groups, d_idx, R, rng, max_redraw=200):
    """R replicate mCARs, one permutation per replicate, retention enforced."""
    n_events = len(d_idx)
    out = np.empty(R)
    ret_v = np.empty(R)
    con_v = np.empty(R)
    redraws = 0
    for i in range(R):
        for _ in range(max_redraw):
            sigma = draw_permutation(rng, n_events)
            m, ret, _, con = permuted_mcar(wb, groups, d_idx, sigma)
            if np.isfinite(m) and ret >= RETENTION_MIN:
                out[i], ret_v[i], con_v[i] = m, ret, con
                break
            redraws += 1
        else:
            raise AssertionError(
                "a replicate could not retain %.0f%% of events in %d redraws"
                % (100 * RETENTION_MIN, max_redraw))
    return {"mcar": out, "retention": ret_v, "contaminated": con_v,
            "redraws": int(redraws)}


def null_p(observed_mcar, null_mcar):
    """p = the share of replicates whose mCAR reaches the observed one."""
    d = np.asarray(null_mcar, dtype=float)
    d = d[np.isfinite(d)]
    return float((d >= float(observed_mcar)).mean())


def mde_from_null(null_mcar, z=Z80):
    """(p95 - median) + z * sd. The registered expression, nothing else."""
    d = np.asarray(null_mcar, dtype=float)
    d = d[np.isfinite(d)]
    return float((np.percentile(d, 95) - np.median(d)) + z * d.std(ddof=1))


def band_ok(share, band):
    """The known-null control passes only INSIDE the closed band [lo, hi]."""
    lo, hi = float(band[0]), float(band[1])
    return bool(lo <= float(share) <= hi)


def detection_ok(detection, floor):
    """The positive control passes at or above its floor."""
    return bool(float(detection) >= float(floor))


# ---------------------------------------------------------------------------
# The positive control's plant — the leak is deliberate
# ---------------------------------------------------------------------------
def plant_drift(n, X, rng):
    """One drift per event from Uniform(0, 2X).

    DELIBERATE LEAK. The plant is added to the very voted window the vote reads,
    on the REAL pairs only, which is the point: it proves the instrument can see
    an effect of the registered size. It is never a result.
    """
    return rng.uniform(0.0, 2.0 * float(X), size=int(n))


def calibrate_X(n, target, seed):
    """X such that ONE reference draw of Uniform(0,2) has planted mean = target."""
    u = np.random.default_rng(int(seed)).uniform(0.0, 2.0, size=int(n))
    mu = float(u.mean())
    if mu <= 0:
        raise AssertionError("the reference draw has a non-positive mean")
    X = float(target) / mu
    return {"X": X, "reference_mean_u": mu,
            "one_draw_planted_mean": float((X * u).mean()),
            "planted_mean_at_X_equals_target": float(float(target) * mu)}


# ---------------------------------------------------------------------------
# Descriptives
# ---------------------------------------------------------------------------
def first_event_mcar(net, tickers, d_idx):
    """mCAR over the first event of each issuer. Reported, never adjudicated.

    An issuer with several terminations in half A contributes once, at its
    earliest disclosure, so the statistic cannot be tilted by one busy filer.
    """
    net = np.asarray(net, dtype=float)
    order = np.argsort(np.asarray(d_idx, dtype=np.int64), kind="stable")
    seen, pick = set(), []
    for i in order:
        t = tickers[i]
        if t in seen:
            continue
        seen.add(t)
        if np.isfinite(net[i]):
            pick.append(net[i])
    if not pick:
        return {"mean": None, "n_issuers": 0}
    return {"mean": float(np.mean(pick)), "n_issuers": int(len(pick))}


def title_mix(titles):
    out = {}
    low = [str(t).lower() for t in titles]
    for name, pats in TITLE_PATTERNS:
        out[name] = int(sum(any(p in t for p in pats) for t in low))
    out["no_title_text"] = int(sum(len(t.strip()) == 0 for t in low))
    return out


def events_by_month(file_dates):
    s = pd.to_datetime(pd.Series(list(file_dates))).dt.strftime("%Y-%m")
    return {k: int(v) for k, v in sorted(s.value_counts().items())}


def verdict_of(p, mcar, bar, materiality=MATERIALITY, suggestive_hi=SUGGESTIVE_HI):
    """Exactly the four outcomes the registration names, in its own order."""
    p, mcar, bar = float(p), float(mcar), float(bar)
    if p <= bar and mcar >= materiality:
        return "SOMETHING"
    if p <= bar and 0.0 < mcar < materiality:
        return "SIGNIFICANT_IMMATERIAL"
    if bar < p <= suggestive_hi and mcar > 0.0:
        return "SUGGESTIVE"
    return "NOTHING"
