"""
================================================================================
Script:       verify_universe.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Measure a Sharadar bulk snapshot into the universe numbers R1
              designs against: master composition, price coverage, survivorship,
              and the S&P 500 membership reconciliation.
Entry Point:  Yes — python tools/verify_universe.py --snapshot <label>
Dependencies: stdlib, pandas, pyarrow. Imports nothing from any other repository.
Last Sprint:  Sprint R0-C — Paid tier verified; bulk full-history pull (2026-09-27)
================================================================================

The price table runs to tens of millions of rows, so every statistic over it is
accumulated one parquet row group at a time: the table is never materialised
whole and peak memory is a function of the row-group size, not the table size.

The tracked report carries statistics only — counts, spans, per-year tables,
reconciliation tallies. Per-row vendor values stay in the untracked snapshot
tree, which is where the licence requires them to remain.

The survivorship claim is stated as a measured share, not asserted: the fraction
of price rows belonging to tickers the master now marks delisted is what makes a
point-in-time universe constructible, and it is printed rather than assumed.
================================================================================
"""

import argparse
import collections
import json
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The price table's own vendor code; the master covers five tables and only this
# slice describes the equity price universe.
PRICE_TABLE_CODE = "SEP"
# How many delisted names the presence check names. Enough to be more than
# anecdote, few enough to read.
DELISTED_SAMPLE_N = 10
# The presence check wants names that died well before the snapshot, so a
# surviving-by-accident row cannot explain a pass.
DELISTED_BEFORE = "2015-01-01"
# A last price date may fall a few sessions short of the master's own stamp.
PRESENCE_TOLERANCE_DAYS = 5


def snap_dir(label):
    return os.path.join(REPO_ROOT, "data", "sharadar", "snapshot_%s" % label)


def table_path(label, table):
    return os.path.join(snap_dir(label), "bulk_%s.parquet" % table)


def year_of(s):
    """Calendar year from a YYYY-MM-DD string column, as string. Pure."""
    return s.str.slice(0, 4)


def summarise_master(df):
    """Master composition: rows, per-table split, listed/delisted, key
    uniqueness, and price-date coverage. Pure: takes the frame, returns stats."""
    out = {"rows": int(len(df))}
    out["rows_by_table"] = {k: int(v) for k, v in
                            df["table"].value_counts().sort_index().items()}
    sep = df[df["table"] == PRICE_TABLE_CODE]
    out["price_table_code"] = PRICE_TABLE_CODE
    out["price_table_rows"] = int(len(sep))
    out["isdelisted_counts_all_tables"] = {
        (k if k else "(blank)"): int(v)
        for k, v in df["isdelisted"].value_counts(dropna=False).items()}
    out["price_table_active"] = int((sep["isdelisted"] == "N").sum())
    out["price_table_delisted"] = int((sep["isdelisted"] == "Y").sum())
    out["price_table_isdelisted_other"] = int(
        len(sep) - out["price_table_active"] - out["price_table_delisted"])

    # permaticker is the vendor's stable identity. It repeats across tables by
    # design, so uniqueness is a per-table property, not a whole-file one.
    out["permaticker_distinct_all_tables"] = int(df["permaticker"].nunique())
    out["permaticker_distinct_price_table"] = int(sep["permaticker"].nunique())
    out["permaticker_unique_within_price_table"] = bool(
        sep["permaticker"].is_unique)
    dup = (len(sep) - sep["permaticker"].nunique())
    out["permaticker_duplicate_rows_price_table"] = int(dup)
    out["ticker_distinct_price_table"] = int(sep["ticker"].nunique())
    out["ticker_unique_within_price_table"] = bool(sep["ticker"].is_unique)

    for col in ("firstpricedate", "lastpricedate"):
        present = sep[col].replace("", pd.NA).dropna()
        d = pd.to_datetime(present, errors="coerce").dropna()
        out["%s_present" % col] = int(len(present))
        out["%s_present_pct" % col] = (
            round(100.0 * len(present) / len(sep), 4) if len(sep) else None)
        out["%s_min" % col] = d.min().strftime("%Y-%m-%d") if len(d) else None
        out["%s_max" % col] = d.max().strftime("%Y-%m-%d") if len(d) else None
    return out


def scan_prices(path, delisted_tickers):
    """One bounded pass over the price table.

    Accumulates row count, distinct tickers, overall span, rows per calendar
    year, each ticker's last dated row, and how many rows belong to a ticker the
    master marks delisted. Peak memory is one row group, not one table.
    """
    pf = pq.ParquetFile(path)
    rows = 0
    per_year = collections.Counter()
    tickers = set()
    last_seen = {}
    first_seen = {}
    delisted_rows = 0
    dmin = dmax = None
    for i in range(pf.metadata.num_row_groups):
        tbl = pf.read_row_group(i, columns=["ticker", "date"])
        tk = tbl["ticker"].to_pandas()
        dt = tbl["date"].to_pandas()
        rows += len(tk)
        per_year.update(year_of(dt).value_counts().to_dict())
        tickers.update(tk.unique().tolist())
        lo, hi = dt.min(), dt.max()
        dmin = lo if dmin is None or lo < dmin else dmin
        dmax = hi if dmax is None or hi > dmax else dmax
        grp = dt.groupby(tk)
        for t, v in grp.max().items():
            if t not in last_seen or v > last_seen[t]:
                last_seen[t] = v
        for t, v in grp.min().items():
            if t not in first_seen or v < first_seen[t]:
                first_seen[t] = v
        if delisted_tickers:
            delisted_rows += int(tk.isin(delisted_tickers).sum())
        del tbl, tk, dt
    return {
        "rows": rows,
        "distinct_tickers": len(tickers),
        "date_min": dmin,
        "date_max": dmax,
        "rows_per_year": {k: int(per_year[k]) for k in sorted(per_year)},
        "row_groups_scanned": pf.metadata.num_row_groups,
        "delisted_rows": delisted_rows,
        "delisted_rows_pct": (round(100.0 * delisted_rows / rows, 4) if rows else None),
        "_last_seen": last_seen,
        "_first_seen": first_seen,
    }


def delisted_presence_check(master_sep, last_seen, before=DELISTED_BEFORE,
                           n=DELISTED_SAMPLE_N, tol_days=PRESENCE_TOLERANCE_DAYS):
    """Do long-dead names still carry their price history?

    For each sampled delisted ticker the master dates before `before`, the price
    table must hold rows and its last row must sit at that date (within
    tolerance). A survivorship-biased vendor would have dropped the name.
    """
    dead = master_sep[(master_sep["isdelisted"] == "Y")
                      & (master_sep["lastpricedate"] != "")
                      & (master_sep["lastpricedate"] < before)]
    dead = dead.sort_values("ticker").head(n)
    rows = []
    for _, r in dead.iterrows():
        tk = r["ticker"]
        seen = last_seen.get(tk)
        gap = None
        if seen:
            gap = int((pd.to_datetime(r["lastpricedate"]) - pd.to_datetime(seen)).days)
        rows.append({
            "ticker": tk,
            "permaticker": r["permaticker"],
            "master_lastpricedate": r["lastpricedate"],
            "prices_last_date": seen,
            "present": seen is not None,
            "days_short_of_master": gap,
            "verdict": ("PRESENT_AT_DEATH"
                        if seen is not None and gap is not None
                        and abs(gap) <= tol_days
                        else ("PRESENT_BUT_OFF_DATE" if seen is not None
                              else "ABSENT")),
        })
    return {
        "sampled": len(rows),
        "delisted_before_%s_available" % before: int(len(
            master_sep[(master_sep["isdelisted"] == "Y")
                       & (master_sep["lastpricedate"] != "")
                       & (master_sep["lastpricedate"] < before)])),
        "present": sum(1 for r in rows if r["present"]),
        "present_at_death": sum(1 for r in rows
                                if r["verdict"] == "PRESENT_AT_DEATH"),
        "absent": sum(1 for r in rows if r["verdict"] == "ABSENT"),
        "detail": rows,
    }


def reconcile_sp500(df):
    """Walk the S&P 500 membership forward and check it against the next snapshot.

    The vendor publishes dated membership snapshots ('historical' / 'current')
    plus discrete 'added' and 'removed' events. Applying the events to one
    snapshot must reproduce the next. Extras are names the walk holds that the
    snapshot does not; missing are names the snapshot holds that the walk lost.
    Pure: takes the frame, returns tallies.
    """
    df = df.copy()
    snap_actions = ("historical", "current")
    snaps = (df[df["action"].isin(snap_actions)]
             .groupby("date")["ticker"].apply(lambda s: set(s)).sort_index())
    events = df[df["action"].isin(("added", "removed"))]
    dates = list(snaps.index)
    steps = []
    for a, b in zip(dates, dates[1:]):
        held = set(snaps.loc[a])
        window = events[(events["date"] > a) & (events["date"] <= b)]
        added = set(window[window["action"] == "added"]["ticker"])
        removed = set(window[window["action"] == "removed"]["ticker"])
        walked = (held | added) - removed
        target = set(snaps.loc[b])
        steps.append({
            "from": a, "to": b,
            "held": len(held), "added": len(added), "removed": len(removed),
            "walked": len(walked), "target": len(target),
            "matches": len(walked & target),
            "extras": len(walked - target),
            "missing": len(target - walked),
        })
    exact = sum(1 for s in steps if s["extras"] == 0 and s["missing"] == 0)
    return {
        "snapshot_dates": len(dates),
        "snapshot_date_first": dates[0] if dates else None,
        "snapshot_date_last": dates[-1] if dates else None,
        "steps": len(steps),
        "steps_exact": exact,
        "steps_exact_pct": round(100.0 * exact / len(steps), 3) if steps else None,
        "total_matches": sum(s["matches"] for s in steps),
        "total_extras": sum(s["extras"] for s in steps),
        "total_missing": sum(s["missing"] for s in steps),
        "median_membership": int(pd.Series([s["target"] for s in steps]).median())
        if steps else None,
        "worst_steps": sorted(steps, key=lambda s: -(s["extras"] + s["missing"]))[:5],
    }


def events_per_year(df, action_col="action", date_col="date", actions=()):
    out = {}
    for act in actions:
        sub = df[df[action_col] == act]
        out[act] = {k: int(v) for k, v in
                    year_of(sub[date_col]).value_counts().sort_index().items()}
    return out


def disk_stats(label):
    d = snap_dir(label)
    files = {}
    total = 0
    for root, _dirs, names in os.walk(d):
        for n in names:
            p = os.path.join(root, n)
            sz = os.path.getsize(p)
            total += sz
            files[os.path.relpath(p, d)] = sz
    usage = shutil.disk_usage(d)
    return {
        "snapshot_bytes": total,
        "snapshot_gib": round(total / 1024 ** 3, 3),
        "files": dict(sorted(files.items(), key=lambda kv: -kv[1])),
        "volume_free_gib": round(usage.free / 1024 ** 3, 2),
        "volume_total_gib": round(usage.total / 1024 ** 3, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    args = ap.parse_args()
    label = args.snapshot

    master = pq.read_table(table_path(label, "tickers")).to_pandas()
    tick = summarise_master(master)
    sep = master[master["table"] == PRICE_TABLE_CODE]
    delisted = set(sep[sep["isdelisted"] == "Y"]["ticker"])

    prices = scan_prices(table_path(label, "stocks"), delisted)
    last_seen = prices.pop("_last_seen")
    prices.pop("_first_seen")
    presence = delisted_presence_check(sep, last_seen)

    sp = pq.read_table(table_path(label, "sp500")).to_pandas()
    sp_stats = {
        "rows": int(len(sp)),
        "action_counts": {k: int(v) for k, v in sp["action"].value_counts().items()},
        "date_min": sp["date"].min(),
        "date_max": sp["date"].max(),
        "events_per_year": events_per_year(sp, actions=("added", "removed")),
        "reconciliation": reconcile_sp500(sp),
    }

    act = pq.read_table(table_path(label, "actions"),
                        columns=["date", "action"]).to_pandas()
    act_stats = {
        "rows": int(len(act)),
        "action_counts": {k: int(v) for k, v in act["action"].value_counts().items()},
        "distinct_action_types": int(act["action"].nunique()),
        "date_min": act["date"].min(),
        "date_max": act["date"].max(),
    }

    report = {
        "snapshot": label,
        "tickers": tick,
        "stocks": prices,
        "delisted_presence_check": {k: v for k, v in presence.items()
                                    if k != "detail"},
        "sp500": sp_stats,
        "actions": act_stats,
        "disk": disk_stats(label),
    }

    out_json = os.path.join(REPO_ROOT, "reports",
                            "universe_verification_%s.json" % label)
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    # Per-row vendor values stay in the untracked snapshot tree.
    ex = os.path.join(snap_dir(label), "delisted_presence_%s.csv" % label)
    pd.DataFrame(presence["detail"]).to_csv(ex, index=False)

    print(json.dumps({k: v for k, v in report.items() if k != "disk"},
                     indent=2)[:200])
    print("report %s" % os.path.relpath(out_json, REPO_ROOT))
    print("examples (untracked) %s" % os.path.relpath(ex, REPO_ROOT))


if __name__ == "__main__":
    main()
