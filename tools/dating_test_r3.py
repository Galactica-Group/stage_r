"""
================================================================================
Script:       dating_test_r3.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Decide whether the vendor's DAILY valuation table is as-filed
              (point-in-time usable) or restated, by dating the steps in the
              trailing earnings implied by marketcap / pe.
Entry Point:  Yes (python tools/dating_test_r3.py)
Dependencies: numpy, pandas
Last Sprint:  Sprint R3 — Look 3 of Q4 (2026-10-05)
================================================================================

The vendor quotes pe to one decimal place, so E = marketcap / pe carries a
rounding error of about 0.05/pe in relative terms — up to half a percent on a
low multiple. A naive "E changed" detector would therefore read rounding noise
as earnings steps. Instead each day's implied E is MATCHED against the ticker's
own trailing-earnings rows, with a per-day compatibility band derived from that
day's own pe precision. A day whose E is compatible with exactly one reported
value is matched; compatible with several, AMBIGUOUS; with none, UNMATCHED.

Matching is unrestricted: a day may match a row filed in its future. That is the
whole point — an unrestricted match is what lets the look-ahead show itself
rather than being assumed away.

Units, from the descriptions dictionary: marketcap is USD MILLIONS, netinccmnusd
is USD, so the implied earnings are scaled by 1e6 before comparison.

Dating, from the descriptions dictionary: for AR dimensions (ART) the
fundamentals date IS the SEC filing date; for MR dimensions (MRT) it is the
reportperiod. The ART date is therefore the point-in-time yardstick.
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "sharadar" / "snapshot_20261005c"
NAMES = ["KO", "NKE", "WMT"]
MARKETCAP_SCALE = 1e6          # USD millions -> USD
PE_HALF_DIGIT = 0.05           # pe is quoted to one decimal place
BAND_MULTIPLE = 1.5            # compatibility band = 1.5x the rounding bound
BAND_FLOOR = 1e-4


def load_daily(snap, ticker):
    df = pd.read_parquet(Path(snap) / ("dly_%s.parquet" % ticker))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "marketcap": pd.to_numeric(df["marketcap"], errors="coerce"),
        "pe": pd.to_numeric(df["pe"], errors="coerce"),
    }).sort_values("date").reset_index(drop=True)
    return out


def load_fundamentals(snap, ticker, dim):
    df = pd.read_parquet(Path(snap) / ("fun_%s_%s.parquet" % (ticker, dim)))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        "reportperiod": pd.to_datetime(df["reportperiod"]),
        "netinccmnusd": pd.to_numeric(df["netinccmnusd"], errors="coerce"),
        "sharesbas": pd.to_numeric(df["sharesbas"], errors="coerce"),
    }).sort_values("reportperiod").reset_index(drop=True)
    return out


def implied_earnings(daily):
    """E(t) in USD. NaN where pe is missing or non-positive."""
    mc = daily["marketcap"].to_numpy(dtype=float)
    pe = daily["pe"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(pe > 0, mc * MARKETCAP_SCALE / pe, np.nan)
    return e


def rounding_band(daily, multiple=BAND_MULTIPLE, floor=BAND_FLOOR):
    """Per-day relative compatibility band implied by pe's quoted precision."""
    pe = daily["pe"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(pe > 0, PE_HALF_DIGIT / pe, np.nan)
    return multiple * rel + floor


def match_vintages(E, band, values):
    """For each day, which reported value is E compatible with?

    Returns (idx, status): idx is the index into values, or -1. status is
    "matched", "ambiguous", "unmatched" or "no_data".
    """
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    idx = np.full(len(E), -1, dtype=int)
    status = np.array(["no_data"] * len(E), dtype=object)
    if not ok.any():
        return idx, status
    vi = np.flatnonzero(ok)
    V = values[vi]
    for t in range(len(E)):
        if not np.isfinite(E[t]) or not np.isfinite(band[t]):
            continue
        rel = np.abs(E[t] / V - 1.0)
        comp = np.flatnonzero(rel <= band[t])
        if len(comp) == 1:
            idx[t] = vi[comp[0]]
            status[t] = "matched"
        elif len(comp) > 1:
            # several reported values sit inside this day's rounding band
            idx[t] = vi[comp[np.argmin(rel[comp])]]
            status[t] = "ambiguous"
        else:
            status[t] = "unmatched"
    return idx, status


def find_steps(dates, idx, status, art):
    """Steps between consecutive MATCHED days whose matched vintage changes."""
    m = np.flatnonzero(status == "matched")
    steps = []
    for a, b in zip(m[:-1], m[1:]):
        if idx[a] == idx[b]:
            continue
        new = art.iloc[idx[b]]
        step_date = dates[b]
        filing = new["date"]
        lag = int((step_date - filing).days)
        # The value could have appeared any time after the previous MATCHED day,
        # so the earliest it could have moved is that day. If even THAT is on or
        # after the filing date, no look-ahead can hide in the ambiguity window.
        lag_lower = int((dates[a] - filing).days)
        steps.append({
            "step_date": str(step_date.date()),
            "prev_matched_reportperiod": str(art.iloc[idx[a]]["reportperiod"].date()),
            "new_matched_reportperiod": str(new["reportperiod"].date()),
            "art_filing_date": str(filing.date()),
            "lag_days_step_minus_filing": lag,
            "classification": "AS-FILED" if lag >= 0 else "LOOK-AHEAD",
            "gap_days_between_matched_days": int((step_date - dates[a]).days),
            "lag_days_lower_bound": lag_lower,
            "look_ahead_excluded_outright": bool(lag_lower >= 0),
        })
    return steps


def dimension_attribution(E, band, art, mrt):
    """Where ART and MRT differ for the same reportperiod, which does E match?"""
    j = art.merge(mrt, on="reportperiod", suffixes=("_art", "_mrt"))
    j = j[np.isfinite(j["netinccmnusd_art"]) & np.isfinite(j["netinccmnusd_mrt"])]
    rel = np.abs(j["netinccmnusd_art"] / j["netinccmnusd_mrt"] - 1.0)
    out = {"note": "a day is tested against the ART value SET and the MRT value "
                   "SET over the reportperiods where the two differ, with no "
                   "uniqueness requirement, so a day may resolve to a dimension "
                   "by coincidence with a different period: these counts are "
                   "LOOSE upper bounds, not step evidence",
           "reportperiods_in_both": int(len(j)),
           "reportperiods_where_they_differ": 0,
           "days_resolving_to_ART": 0, "days_resolving_to_MRT": 0,
           "days_resolving_to_neither": 0, "days_resolving_to_both": 0,
           "separable": False}
    med = np.nanmedian(band)
    sep = j[rel > 2 * med]
    out["reportperiods_where_they_differ"] = int(len(sep))
    if not len(sep):
        return out
    out["separable"] = True
    A = sep["netinccmnusd_art"].to_numpy(dtype=float)
    Mv = sep["netinccmnusd_mrt"].to_numpy(dtype=float)
    for t in range(len(E)):
        if not np.isfinite(E[t]) or not np.isfinite(band[t]):
            continue
        ha = (np.abs(E[t] / A - 1.0) <= band[t]).any()
        hm = (np.abs(E[t] / Mv - 1.0) <= band[t]).any()
        if ha and hm:
            out["days_resolving_to_both"] += 1
        elif ha:
            out["days_resolving_to_ART"] += 1
        elif hm:
            out["days_resolving_to_MRT"] += 1
        else:
            out["days_resolving_to_neither"] += 1
    return out


def verdict_for(steps):
    """AS-FILED, RESTATED or MIXED, from the step classifications."""
    if not steps:
        return "INDETERMINATE", 0, 0
    n_af = sum(s["classification"] == "AS-FILED" for s in steps)
    n_la = len(steps) - n_af
    if n_la == 0:
        return "AS-FILED", n_af, n_la
    if n_af == 0:
        return "RESTATED", n_af, n_la
    return "MIXED", n_af, n_la


def run_ticker(snap, ticker):
    daily = load_daily(snap, ticker)
    art = load_fundamentals(snap, ticker, "ART")
    mrt = load_fundamentals(snap, ticker, "MRT")
    E = implied_earnings(daily)
    band = rounding_band(daily)
    dates = daily["date"].to_numpy()
    dates = pd.DatetimeIndex(dates)
    idx, status = match_vintages(E, band, art["netinccmnusd"].to_numpy())
    steps = find_steps(dates, idx, status, art)
    v, n_af, n_la = verdict_for(steps)
    lags = [s["lag_days_step_minus_filing"] for s in steps]
    gaps = [s["gap_days_between_matched_days"] for s in steps]
    outright = sum(s["look_ahead_excluded_outright"] for s in steps)
    return {
        "ticker": ticker,
        "daily_rows": int(len(daily)),
        "daily_span": [str(dates.min().date()), str(dates.max().date())],
        "art_rows": int(len(art)),
        "art_span": [str(art["date"].min().date()), str(art["date"].max().date())],
        "mrt_rows": int(len(mrt)),
        "status_counts": {k: int((status == k).sum())
                          for k in ("matched", "ambiguous", "unmatched", "no_data")},
        "median_band_rel": float(np.nanmedian(band)),
        "steps_found": len(steps),
        "steps_as_filed": n_af,
        "steps_look_ahead": n_la,
        "lag_days": {"min": int(min(lags)) if lags else None,
                     "median": float(np.median(lags)) if lags else None,
                     "max": int(max(lags)) if lags else None},
        "gap_days_between_matched_days": {
            "min": int(min(gaps)) if gaps else None,
            "median": float(np.median(gaps)) if gaps else None,
            "max": int(max(gaps)) if gaps else None},
        "steps_with_look_ahead_excluded_outright": int(outright),
        "steps_where_ambiguity_could_mask_a_look_ahead": int(len(steps) - outright),
        "detection_bias_note": "days whose implied E sits inside the rounding band "
                               "of two reported values are AMBIGUOUS and excluded "
                               "from step detection, which can only date a step "
                               "LATER, i.e. biases the test TOWARD as-filed; "
                               "steps_with_look_ahead_excluded_outright is the "
                               "count immune to that bias",
        "verdict": v,
        "dimension_attribution": dimension_attribution(E, band, art, mrt),
        "steps": steps,
    }


def main():
    res = [run_ticker(SNAP, t) for t in NAMES]
    vs = {r["verdict"] for r in res}
    overall = vs.pop() if len(vs) == 1 else "MIXED"
    out = {
        "tool_version": VERSION,
        "snapshot": SNAP.name,
        "names": NAMES,
        "units_note": "marketcap USD millions, netinccmnusd USD; implied earnings "
                      "scaled by 1e6 (descriptions dictionary)",
        "dating_note": "for ART the fundamentals date IS the SEC filing date; for "
                       "MRT it is the reportperiod (descriptions dictionary)",
        "detector": {"pe_half_digit": PE_HALF_DIGIT, "band_multiple": BAND_MULTIPLE,
                     "band_floor": BAND_FLOOR,
                     "rule": "a day matches the unique reported value inside its own "
                             "pe-rounding band; several -> ambiguous, none -> unmatched"},
        "per_ticker": res,
        "verdict_daily_table": overall,
        "caveat": "assumes the free sample is built the same way as the paid table",
        "scope_note": "a provenance test on single-name valuation metadata; it "
                      "computes no return statistic and shares no universe with "
                      "any registered Stage R look",
    }
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
