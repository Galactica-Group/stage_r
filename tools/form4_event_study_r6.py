"""
================================================================================
Script:       form4_event_study_r6.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R6 engine: insider open-market PURCHASES disclosed on Form 4
              as an event study on the public filing date, with the issuer
              permutation null at fixed dates and both registered controls.
Entry Point:  No — imported by tools/run_t_r6.py
Dependencies: stdlib, numpy, pandas, pyarrow, tools/item408_event_study_r5.py.
              Imports nothing from any other repository; in particular it
              imports nothing from the factory tree.
Last Sprint:  Sprint R6-1 — T-R6 insider open-market purchases: seal, controls,
              half-A read (2026-11-05)
================================================================================

Every parameter comes from the sealed registration JSON, whose sha256 must match
the one passed in; load_registration refuses otherwise. The integers the seal
states in prose rather than in a numeric field are declared here as module
constants and pinned against that prose by a test, so the code and the seal
cannot drift apart silently.

WHAT IS REUSED AND WHAT IS OWNED. The certified T-R5 machinery is imported
UNCHANGED and the R5 module is never edited: the O(1) window scorer, the issuer
permutation null and its retention rule, the MDE expression, the band and
detection predicates, the plant and its calibration, the price column-store and
its trading-day calendar, and the abnormal-return series. What this look owns is
everything Form 4 specific: the purchase filter, the per-issuer clustering, the
INVESTABLE benchmark, the completeness rule with its truncation clause, the
subgroup and by-year descriptives, and the DISTANCE statistic.

Because r5.score_pairs reads R5's own window constants, assert_shared_constants
refuses to import this module at all if any of them differs from this look's.

AN EVENT IS AN ISSUER'S PURCHASE EPISODE, not a filing and not an insider. The
first qualifying filing day of an issuer opens it and it absorbs every later
qualifying purchase filed inside the following CLUSTER_WINDOW trading days, so
several officers buying through one week are one piece of news. D is the opening
day; the trade the vote prices enters at the close of D+1, so it reads nothing a
reader of the filings could not have read.

THE BENCHMARK IS INVESTABLE. A long position in an insider-bought small cap is
not fairly judged against an index that includes names a buyer could not hold,
so the benchmark averages only tickers clearing the same three floors the event
side clears, measured on the prior trading day.

A STOCK THAT STOPS TRADING inside the window is not a hole: the position ends at
its last close, which is the return a holder realised. A last row at the
snapshot's own edge is TRUNCATION, not a delisting, and drops — the difference
matters because a truncated window would be scored as though the position had
been closed when in fact the vintage simply ends.

THE DISTANCE IS THE QUANTITY ADJUDICATED. mCAR alone carries the drift of this
issuer set at these dates; the null's median carries exactly that drift, so the
bar is measured on mCAR minus the null median and never from zero.

ONE DISCLOSED INTERNAL USE OF THE OBSERVED SET. The registered positive control
plants a drift on the real events' own voted-window returns, so forming the
planted mean necessarily forms the observed mean inside it. That figure is used
and discarded: returned by no public function, printed by no log line, written
into no receipt. The same accommodation R4 and R5 made.
================================================================================
"""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import item408_event_study_r5 as r5

VERSION = "1.0.0"

# --- stated in the registration as prose; pinned by test_registration_prose_pins
LAG_MIN = 0                     # "filing_lag_days between 0 and 90 inclusive"
LAG_MAX = 90
CLUSTER_WINDOW = 60             # "within the following 60 trading days"
HISTORY_MIN_ROWS = 250          # "at least 250 bulk_stocks rows"
PRICE_FLOOR = 2.00              # "closeunadj on D+1 >= 2.00 USD"
DOLLAR_VOL_FLOOR = 500000.0     # "... >= 500000 USD"
DOLLAR_VOL_WINDOW = 20          # "over the 20 trading days ending D"
COMPLETE_FROM = 1               # "on every trading day D+1..D+61"
COMPLETE_TO = 61
CALENDAR_MIN_TICKERS = 1000     # "at least 1000 tickers have a bulk_stocks row"
MIN_RETURNS = 5                 # "provided at least 5 returns exist"
TRUNCATION_EDGE = 5             # "within 5 trading days of the snapshot's end"
BENCH_PRICE_FLOOR = 2.00        # "closeunadj >= 2.00 USD"
BENCH_MIN_PRIOR_ROWS = 250      # "at least 250 prior rows"
BENCH_DV_FLOOR = 500000.0       # "median closeunadj times volume >= 500000 USD"
BENCH_DV_WINDOW = 20            # "a 20-day median closeunadj times volume"
VOTE_FROM = 2                   # "the abnormal daily log returns from D+2"
VOTE_TO = 61                    # "exit at the close of D+61", the cap
VOTE_N = 60                     # a complete window's return count
SHORT_TO = 21                   # "D+2..D+21"
SHORT_N = 20
PREPURCHASE_LAG = 21            # "the pre-purchase window T-21..T"
PATH_POINTS = (5, 10, 20, 40, 60)   # "at 5, 10, 20, 40 and 60 days after D+1"
COST_ROUND_TRIP_BPS = 20.0      # "20 bps round trip"
RETENTION_MIN = 0.90            # "a replicate under 90 percent retention"
Z80 = 0.8416                    # "+ 0.8416 * sd(null mCAR)"
MIN_EVENTS = 5000               # "fewer than 5000 half-A events"
MATERIALITY = 0.010             # "distance >= 0.010 net"
SUGGESTIVE_HI = 0.10            # "p in (0.05, 0.10]"
INSIDER_SUBGROUP = 2            # "two or more insiders"
VALUE_SUBGROUP = 100000.0       # "total value over 100000 USD"
N_TERCILES = 3                  # "liquidity terciles"

COST_FRACTION = COST_ROUND_TRIP_BPS * 1e-4

# The R5 constants r5.score_pairs and r5.permutation_null actually read. This
# look reuses that code unchanged, so a divergence would silently score the
# vote under another look's window. Checked at import, not hoped for.
SHARED_WITH_R5 = {
    "COMPLETE_FROM": COMPLETE_FROM,
    "VOTE_FROM": VOTE_FROM,
    "VOTE_TO": VOTE_TO,
    "MIN_DELIST_RETURNS": MIN_RETURNS,
    "COST_FRACTION": COST_FRACTION,
    "RETENTION_MIN": RETENTION_MIN,
    "Z80": Z80,
    "CALENDAR_MIN_TICKERS": CALENDAR_MIN_TICKERS,
}


def assert_shared_constants():
    """Refuse to run if any reused R5 constant differs from this look's."""
    bad = []
    for name, want in SHARED_WITH_R5.items():
        got = getattr(r5, name)
        if got != want:
            bad.append("%s: R5 has %r, T-R6 needs %r" % (name, got, want))
    if bad:
        raise AssertionError(
            "the reused T-R5 scorer does not share this look's window: "
            + "; ".join(bad))
    return True


assert_shared_constants()

# ---------------------------------------------------------------------------
# Imported UNCHANGED from the certified T-R5 engine
# ---------------------------------------------------------------------------
sha256_file = r5.sha256_file
load_registration = r5.load_registration
calendar_min_tickers = r5.calendar_min_tickers
half_bounds = r5.half_bounds
PriceBook = r5.PriceBook
load_prices = r5.load_prices
trading_day_calendar = r5.trading_day_calendar
abnormal_series = r5.abnormal_series
score_pairs = r5.score_pairs
voted_returns = r5.voted_returns
draw_permutation = r5.draw_permutation
permuted_mcar = r5.permuted_mcar
permutation_null = r5.permutation_null
null_p = r5.null_p
mde_from_null = r5.mde_from_null
band_ok = r5.band_ok
detection_ok = r5.detection_ok
plant_drift = r5.plant_drift
calibrate_X = r5.calibrate_X


# ---------------------------------------------------------------------------
# The window book: the R5 cumulative scorer with narrower counters
# ---------------------------------------------------------------------------
class WindowBook(r5.WindowBook):
    """r5.WindowBook's interface and arithmetic, with int16 counter planes.

    The counters are counts of trading days, so they cannot exceed the calendar
    length; at int32 the four planes cost more than this machine has to spare
    beside the price store, and nothing about the scorer reads their dtype. The
    cumulative abnormal returns stay float64, because that plane is the one
    every window sum is differenced out of.
    """

    def __init__(self, book, tickers, market):
        self.tickers = list(dict.fromkeys(tickers))
        self.gidx = {t: i for i, t in enumerate(self.tickers)}
        L = int(book.L)
        if L >= np.iinfo(np.int16).max:
            raise AssertionError(
                "the calendar has %d trading days, too many for int16 counters"
                % L)
        self.L = L
        G = len(self.tickers)
        self.cum_ab = np.zeros((G, L + 1))
        self.cum_ok = np.zeros((G, L + 1), dtype=np.int16)
        self.cum_row = np.zeros((G, L + 1), dtype=np.int16)
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
        self.real_vote_cum = np.zeros((G, L + 1), dtype=np.int16)


# ---------------------------------------------------------------------------
# The seal's data pins
# ---------------------------------------------------------------------------
HEX64 = re.compile(r"\b([0-9a-f]{64})\b")


def manifest_recorded_sha(manifest_path, filename):
    """The sha256 the tracked manifest records for one of its own files."""
    man = json.loads(Path(manifest_path).read_text())
    shas = man.get("sha256") or {}
    if filename not in shas:
        raise AssertionError(
            "manifest %s records no sha256 for %r" % (manifest_path, filename))
    return str(shas[filename])


# ---------------------------------------------------------------------------
# The rolling dollar-volume median, over TRADING-DAY positions
# ---------------------------------------------------------------------------
def rolling_dollar_volume_median(book, window=BENCH_DV_WINDOW, chunk=1_000_000):
    """Per row: the median of closeunadj * volume over the `window` trading
    days ENDING on that row's own trading day.

    The window is `window` positions on the calendar, not `window` rows, which
    is the same definition r5.PriceBook.dollar_volume_median uses on the event
    side; a ticker with a gap therefore medians over however many rows sit in
    those positions. At most `window` rows can sit in `window` positions, so the
    whole snapshot is done in one vectorised sweep: the lower bound of each
    row's window is found once by searchsorted over the global (ticker, day)
    key, and the up-to-`window` lagged values are stacked and nan-medianed.
    """
    L = int(book.L)
    code = book.code.astype(np.int64)
    dayidx = book.dayidx.astype(np.int64)
    n = len(code)
    out = np.full(n, np.nan)
    if n == 0:
        return out
    key = code * np.int64(L) + dayidx
    lo_key = code * np.int64(L) + np.maximum(dayidx - (int(window) - 1), 0)
    lo_idx = np.searchsorted(key, lo_key, "left")
    dv = np.multiply(book.closeunadj, book.volume, dtype=np.float64)
    idx = np.arange(n, dtype=np.int64)
    for s in range(0, n, int(chunk)):
        e = min(n, s + int(chunk))
        m = e - s
        M = np.full((m, int(window)), np.nan)
        here = idx[s:e]
        lo_here = lo_idx[s:e]
        for k in range(int(window)):
            src = here - k
            good = src >= lo_here
            if not good.any():
                continue
            M[good, k] = dv[src[good]]
        with np.errstate(invalid="ignore"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[s:e] = np.nanmedian(M, axis=1)
        del M
    return out


# ---------------------------------------------------------------------------
# The investable equal-weight benchmark
# ---------------------------------------------------------------------------
def build_investable_index(book, dvmed):
    """Equal-weight mean daily log return across INVESTABLE ticker-days.

    A ticker's return on trading day j enters only when its previous ROW is the
    previous trading day and, on that previous day, all three floors hold: the
    unadjusted close clears BENCH_PRICE_FLOOR, at least BENCH_MIN_PRIOR_ROWS
    rows sit strictly before it, and its rolling dollar-volume median clears
    BENCH_DV_FLOOR. Not an event statistic: built from the whole snapshot, it
    knows nothing of any filing.
    """
    L = int(book.L)
    s = np.zeros(L)
    cnt = np.zeros(L, dtype=np.int64)
    dvmed = np.asarray(dvmed, dtype=np.float64)
    for c in range(len(book.vocab)):
        a, b = int(book.start[c]), int(book.stop[c])
        if b - a < BENCH_MIN_PRIOR_ROWS + 2:
            continue
        d = book.dayidx[a:b]
        ca = book.closeadj[a:b]
        cu = book.closeunadj[a:b]
        dvm = dvmed[a:b]
        adjacent = (d[1:] - d[:-1]) == 1
        prior_px = cu[:-1] >= BENCH_PRICE_FLOOR
        # rows strictly before the PRIOR row: its within-ticker index.
        prior_rows = np.arange(0, b - a - 1) >= BENCH_MIN_PRIOR_ROWS
        prior_dv = dvm[:-1] >= BENCH_DV_FLOOR
        pos = ((ca[1:] > 0) & (ca[:-1] > 0)
               & np.isfinite(ca[1:]) & np.isfinite(ca[:-1]))
        keep = adjacent & prior_px & prior_rows & prior_dv & pos
        if not keep.any():
            continue
        r = np.log(ca[1:][keep] / ca[:-1][keep])
        j = d[1:][keep]
        np.add.at(s, j, r)
        np.add.at(cnt, j, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
    return {"mean_log_return": m, "n_tickers": cnt}


def index_frame(book, index):
    return pd.DataFrame({
        "date": [book.date_str(j) for j in range(book.L)],
        "mean_log_return": index["mean_log_return"],
        "n_tickers": index["n_tickers"].astype(np.int64),
    })


def index_members_on(book, dvmed, j):
    """The tickers the investable index averages on trading day j. Counts only."""
    out = []
    dvmed = np.asarray(dvmed, dtype=np.float64)
    for c in range(len(book.vocab)):
        a, b = int(book.start[c]), int(book.stop[c])
        if b - a < BENCH_MIN_PRIOR_ROWS + 2:
            continue
        d = book.dayidx[a:b]
        i = int(np.searchsorted(d, j, "left"))
        if not (i < (b - a) and d[i] == j) or i == 0:
            continue
        if d[i] - d[i - 1] != 1:
            continue
        if not (book.closeunadj[a + i - 1] >= BENCH_PRICE_FLOOR):
            continue
        if (i - 1) < BENCH_MIN_PRIOR_ROWS:
            continue
        if not (dvmed[a + i - 1] >= BENCH_DV_FLOOR):
            continue
        ca = book.closeadj
        if not (ca[a + i] > 0 and ca[a + i - 1] > 0):
            continue
        out.append(str(book.vocab[c]))
    return out


# ---------------------------------------------------------------------------
# Event resolution — one clause at a time, each with its own count
# ---------------------------------------------------------------------------
def roll_forward_many(cal, days):
    """Position of the first trading day at or after each day; -1 past the end."""
    cal = np.asarray(cal, dtype=np.int64)
    d = np.asarray(days, dtype=np.int64)
    j = np.searchsorted(cal, d, "left")
    return np.where(j < len(cal), j, -1).astype(np.int64)


def days_since_epoch(series):
    return (pd.to_datetime(pd.Series(series)).values.astype("datetime64[D]")
            .astype("int64"))


def resolve_purchases(pur):
    """The registered PURCHASE clauses, in order, each with its own count."""
    counts = {"rows_in_vintage": int(len(pur))}
    lag = pd.to_numeric(pur["filing_lag_days"], errors="coerce")
    m = lag.notna() & (lag >= LAG_MIN) & (lag <= LAG_MAX)
    counts["A_lag_0_to_90"] = int(m.sum())
    counts["A_removed_lag_negative"] = int((lag.notna() & (lag < LAG_MIN)).sum())
    counts["A_removed_lag_over_90"] = int((lag.notna() & (lag > LAG_MAX)).sum())
    counts["A_removed_lag_missing"] = int(lag.isna().sum())
    routine = pur["is_routine"].fillna(False).astype(bool)
    m = m & ~routine
    counts["B_not_routine"] = int(m.sum())
    counts["B_removed_routine"] = int(counts["A_lag_0_to_90"] - m.sum())
    kept = pur.loc[m].copy()
    return kept, counts


def cluster_events(kept, book, counts):
    """One event per issuer purchase episode of at most CLUSTER_WINDOW days.

    The first qualifying filing day of an issuer opens an event; every later
    qualifying purchase whose filing day sits strictly inside the following
    CLUSTER_WINDOW trading days joins it; the first purchase at or beyond that
    offset opens the next event, so the next event can only open after the
    window. Grouping on the FILING day, not the transaction day, is the whole
    point: the filing is when the market could have known.
    """
    if len(kept) == 0:
        counts["C_filing_day_on_the_calendar"] = 0
        counts["C_dropped_off_calendar"] = 0
        counts["D_events"] = 0
        return pd.DataFrame(columns=["issuer_cik", "D_idx", "n_insiders",
                                     "n_purchases", "total_value", "T",
                                     "T_last", "issuer_name", "first_accession"])
    df = kept.copy()
    df["_pos"] = roll_forward_many(book.cal, days_since_epoch(df["filing_date"]))
    on = df["_pos"] >= 0
    counts["C_filing_day_on_the_calendar"] = int(on.sum())
    counts["C_dropped_off_calendar"] = int((~on).sum())
    df = df.loc[on].sort_values(["issuer_cik", "_pos", "trans_date"],
                                kind="mergesort").reset_index(drop=True)

    iss = df["issuer_cik"].astype(str).to_numpy()
    p = df["_pos"].to_numpy(dtype=np.int64)
    n = len(p)
    ev = np.empty(n, dtype=np.int64)
    open_pos = np.empty(n, dtype=np.int64)
    cur = -1
    opened = -(10 ** 9)
    for i in range(n):
        if i == 0 or iss[i] != iss[i - 1] or (p[i] - opened) >= CLUSTER_WINDOW:
            cur += 1
            opened = p[i]
        ev[i] = cur
        open_pos[i] = opened
    df["_ev"] = ev
    df["_open"] = open_pos

    val = pd.to_numeric(df["value"], errors="coerce")
    df["_val"] = val.where(np.isfinite(val), 0.0)
    g = df.groupby("_ev", sort=True)
    out = pd.DataFrame({
        "issuer_cik": g["issuer_cik"].first().astype(str),
        "D_idx": g["_open"].first().astype(np.int64),
        "n_insiders": g["owner_cik"].nunique().astype(np.int64),
        "n_purchases": g.size().astype(np.int64),
        "total_value": g["_val"].sum().astype(float),
        "T_last": pd.to_datetime(g["trans_date"].max()),
        "issuer_name": g["issuer_name"].first(),
        "first_accession": g["accession"].first(),
    })
    first = df.loc[df["_pos"] == df["_open"]]
    out["T"] = pd.to_datetime(first.groupby("_ev")["trans_date"].min())
    out = out.reset_index(drop=True)
    counts["D_events"] = int(len(out))
    return out


def extract_cik(secfilings):
    """The CIK out of a browse-edgar URL, as an int. None when there is none."""
    return r5.extract_cik(secfilings)


def map_issuers(ev, sep, book, counts):
    """Issuer CIK -> the SEP permaticker whose ticker actually trades around D.

    With several permatickers behind one CIK the registered tie-break is the one
    whose ticker has a row on D+1, then the most rows before D; with none, the
    event drops.
    """
    sep = sep.loc[sep["table"] == "SEP"].copy()
    sep["cik_int"] = sep["secfilings"].map(extract_cik)
    by_cik = {}
    for cik, g in sep.dropna(subset=["cik_int"]).groupby("cik_int", sort=False):
        by_cik[int(cik)] = list(zip(g["permaticker"].astype(str),
                                    g["ticker"].astype(str)))
    out, matched, multi, dropped, resolved_multi = [], 0, 0, 0, 0
    for r in ev.to_dict("records"):
        try:
            cik = int(str(r["issuer_cik"]).strip())
        except (TypeError, ValueError):
            dropped += 1
            continue
        cands = by_cik.get(cik, [])
        if not cands:
            dropped += 1
            continue
        if len(cands) > 1:
            multi += 1
            alive = [c for c in cands
                     if book.has_row_on(c[1], int(r["D_idx"]) + 1)]
            if len(alive) > 1:
                alive = sorted(
                    alive,
                    key=lambda c: (-book.rows_before_idx(c[1], int(r["D_idx"])),
                                   c[1]))
                resolved_multi += 1
            if not alive:
                dropped += 1
                continue
            pick = alive[0]
        else:
            pick = cands[0]
        rr = dict(r)
        rr["cik"] = cik
        rr["permaticker"], rr["ticker"] = pick
        out.append(rr)
        matched += 1
    counts["E_mapping"] = {"matched": int(matched),
                           "multi_permaticker": int(multi),
                           "multi_resolved_by_row_count": int(resolved_multi),
                           "dropped_no_usable_ticker": int(dropped)}
    mp = pd.DataFrame(out)
    if len(mp):
        dup = int(mp.duplicated(subset=["permaticker", "D_idx"]).sum())
    else:
        dup = 0
    counts["E_duplicate_permaticker_D_pairs"] = dup
    return mp.reset_index(drop=True), counts


def classify_window(book, ticker, d_idx, edge=None):
    """COMPLETE, DELISTED-IN-WINDOW, TRUNCATED or a drop reason, plus the exit.

    The registered completeness rule, in its own words and its own order:
      complete  — closeadj on every trading day D+1..D+61; exit at D+61.
      truncated — the last row falls inside the window but within
                  TRUNCATION_EDGE trading days of the snapshot's end, so the
                  vintage stops, not the stock. Drops.
      too_short — fewer than MIN_RETURNS returns are available. Drops.
      hole      — a trading day is missing while the ticker goes on trading,
                  or before an in-window delisting. Drops.
      delisted  — the last row falls inside D+2..D+61, far enough from the
                  snapshot's end, with every day present and at least
                  MIN_RETURNS returns; exit at that last close.
      no_entry  — no row on D+1, so the position cannot be opened. Drops.
      absent    — the ticker has no rows at all. Drops.
    Returns (status, exit_idx); exit_idx is -1 on every drop.
    """
    L = int(book.L)
    last_cal = (L - 1) if edge is None else int(edge)
    entry = int(d_idx) + COMPLETE_FROM
    cap = int(d_idx) + COMPLETE_TO
    if book.slice_of(ticker) is None:
        return "absent", -1
    last = book.last_row_idx(ticker)
    if last < 0:
        return "absent", -1
    if entry < 0 or entry >= L:
        return "no_entry", -1
    if not book.has_row_on(ticker, entry):
        return "no_entry", -1
    if last >= cap:
        if not book.complete_between(ticker, entry, cap):
            return "hole", -1
        return "complete", int(cap)
    if last > (last_cal - TRUNCATION_EDGE):
        return "truncated", -1
    if (last - entry) < MIN_RETURNS:
        return "too_short", -1
    if not book.complete_between(ticker, entry, last):
        return "hole", -1
    return "delisted", int(last)


def apply_price_filters(mp, book, counts, edge=None):
    """History, tradability and completeness, each counted separately."""
    keep = []
    n_hist = n_price = n_dv = 0
    comp = {"complete": 0, "delisted": 0, "truncated": 0, "hole": 0,
            "no_entry": 0, "too_short": 0, "absent": 0}
    if len(mp) == 0:
        counts["F_history_250_rows_before_D"] = 0
        counts["G_price_floor_on_D_plus_1"] = 0
        counts["G_dollar_volume_floor"] = 0
        counts["H_window_status"] = comp
        counts["H_complete_or_delisted"] = 0
        return pd.DataFrame(columns=list(mp.columns) + [
            "window_status", "exit_idx", "n_returns", "dv_median_at_D",
            "px_on_D_plus_1"]), counts
    for r in mp.to_dict("records"):
        d_idx = int(r["D_idx"])
        if book.rows_before_idx(r["ticker"], d_idx) < HISTORY_MIN_ROWS:
            continue
        n_hist += 1
        px = book.value_on(r["ticker"], d_idx + 1, "closeunadj")
        if not (np.isfinite(px) and px >= PRICE_FLOOR):
            continue
        n_price += 1
        dv = book.dollar_volume_median(r["ticker"], d_idx, DOLLAR_VOL_WINDOW)
        if not (np.isfinite(dv) and dv >= DOLLAR_VOL_FLOOR):
            continue
        n_dv += 1
        status, exit_idx = classify_window(book, r["ticker"], d_idx, edge)
        comp[status] += 1
        if status not in ("complete", "delisted"):
            continue
        rr = dict(r)
        rr["window_status"] = status
        rr["exit_idx"] = int(exit_idx)
        rr["n_returns"] = int(exit_idx - (d_idx + COMPLETE_FROM))
        rr["dv_median_at_D"] = float(dv)
        rr["px_on_D_plus_1"] = float(px)
        keep.append(rr)
    counts["F_history_250_rows_before_D"] = int(n_hist)
    counts["G_price_floor_on_D_plus_1"] = int(n_price)
    counts["G_dollar_volume_floor"] = int(n_dv)
    counts["H_window_status"] = {k: int(v) for k, v in comp.items()}
    counts["H_complete_D1_to_D61"] = int(comp["complete"])
    counts["H_delisted_in_window"] = int(comp["delisted"])
    counts["H_truncated_at_snapshot_edge"] = int(comp["truncated"])
    counts["H_hole"] = int(comp["hole"])
    counts["H_complete_or_delisted"] = int(comp["complete"] + comp["delisted"])
    return pd.DataFrame(keep).reset_index(drop=True), counts


def resolve_events(pur, sep, book, edge=None):
    """The whole registered resolution, in clause order, counts at every step."""
    kept, counts = resolve_purchases(pur)
    ev = cluster_events(kept, book, counts)
    mp, counts = map_issuers(ev, sep, book, counts)
    res, counts = apply_price_filters(mp, book, counts, edge)
    return res, counts


def split_halves(res, reg):
    """Half A, half B and the holdout by D's own calendar date."""
    hb = half_bounds(reg)
    if len(res) == 0:
        return {k: res for k in ("half_A", "half_B", "holdout")}
    d = pd.to_datetime(res["D_date"])
    a_lo, a_hi = hb["half_A"]
    b_lo, b_hi = hb["half_B"]
    h_lo, _ = hb["holdout"]
    return {
        "half_A": res.loc[(d >= a_lo) & (d <= a_hi)].reset_index(drop=True),
        "half_B": res.loc[(d >= b_lo) & (d <= b_hi)].reset_index(drop=True),
        "holdout": res.loc[d >= h_lo].reset_index(drop=True),
    }


def with_D_dates(res, book):
    """Attach D's calendar date, which is what the halves are cut on."""
    if len(res) == 0:
        out = res.copy()
        out["D_date"] = pd.Series(dtype="object")
        return out
    out = res.copy()
    out["D_date"] = [book.date_str(int(j)) for j in out["D_idx"]]
    return out


# ---------------------------------------------------------------------------
# The statistic: the DISTANCE from the null's median
# ---------------------------------------------------------------------------
def null_median(null_mcar):
    d = np.asarray(null_mcar, dtype=float)
    return float(np.median(d[np.isfinite(d)]))


def distance_of(observed_mcar, null_mcar):
    """mCAR minus the median of the null's mCAR. The adjudicated quantity.

    Never measured from zero: the null's centre carries the drift of this issuer
    set at these dates, which is precisely what the vote must look past.
    """
    return float(observed_mcar) - null_median(null_mcar)


def corrected_distance(distance, contaminated_share):
    """distance / (1 - the null's mean contaminated share). Reported, not voted."""
    s = float(contaminated_share)
    if not (0.0 <= s < 1.0):
        return None
    return float(distance) / (1.0 - s)


def verdict_of(p, distance, bar, materiality=MATERIALITY,
               suggestive_hi=SUGGESTIVE_HI):
    """Exactly the four outcomes the registration names, on the DISTANCE."""
    p, distance, bar = float(p), float(distance), float(bar)
    if p <= bar and distance >= materiality:
        return "SOMETHING"
    if p <= bar and 0.0 < distance < materiality:
        return "SIGNIFICANT_IMMATERIAL"
    if bar < p <= suggestive_hi and distance > 0.0:
        return "SUGGESTIVE"
    return "NOTHING"


# ---------------------------------------------------------------------------
# Descriptives — formed only inside the observed stage
# ---------------------------------------------------------------------------
def events_by_year(d_dates, net, nullmed):
    """Events per calendar year of D, with that year's pooled distance."""
    y = pd.to_datetime(pd.Series(list(d_dates))).dt.year
    net = pd.Series(np.asarray(net, dtype=float))
    out = {}
    for yy in sorted(y.unique()):
        m = (y == yy).to_numpy()
        v = net[m]
        out[str(int(yy))] = {
            "n": int(m.sum()),
            "mCAR_net": float(v.mean()),
            "distance": float(v.mean() - nullmed),
        }
    return out


def tercile_labels(values, k=N_TERCILES):
    """Tercile membership by rank; ties broken by position, equal-sized groups."""
    v = np.asarray(values, dtype=float)
    n = len(v)
    lab = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return lab
    order = np.argsort(v, kind="stable")
    edges = [int(round(i * n / float(k))) for i in range(k + 1)]
    for i in range(k):
        lab[order[edges[i]:edges[i + 1]]] = i
    return lab


def subgroup_distances(half, net, nullmed):
    """The registered subgroups, REPORTED and never voted."""
    net = np.asarray(net, dtype=float)
    out = {}
    m = half["n_insiders"].to_numpy(dtype=np.int64) >= INSIDER_SUBGROUP
    out["insiders_ge_%d" % INSIDER_SUBGROUP] = _cell(net, m, nullmed)
    out["insiders_lt_%d" % INSIDER_SUBGROUP] = _cell(net, ~m, nullmed)
    v = half["total_value"].to_numpy(dtype=float) > VALUE_SUBGROUP
    out["value_gt_%d" % int(VALUE_SUBGROUP)] = _cell(net, v, nullmed)
    out["value_le_%d" % int(VALUE_SUBGROUP)] = _cell(net, ~v, nullmed)
    lab = tercile_labels(half["dv_median_at_D"].to_numpy(dtype=float))
    names = ("liquidity_tercile_low", "liquidity_tercile_mid",
             "liquidity_tercile_high")
    for i, nm in enumerate(names[:N_TERCILES]):
        out[nm] = _cell(net, lab == i, nullmed)
    return out


def _cell(net, mask, nullmed):
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return {"n": 0, "mCAR_net": None, "distance": None}
    v = net[mask]
    return {"n": int(mask.sum()), "mCAR_net": float(v.mean()),
            "distance": float(v.mean() - nullmed)}
