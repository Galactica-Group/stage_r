"""
================================================================================
Script:       sharadar_snapshot_diff.py
Version:      1.2.0
Phase:        Stage R — External Data Evaluation
Purpose:      Diff two Sharadar snapshots and classify per-ticker price changes
              as ADJUSTMENT, REVISION or TAIL; emit statistics-only reports.
Entry Point:  Yes — python tools/sharadar_snapshot_diff.py <label_a> <label_b>
Dependencies: stdlib, pandas, pyarrow. Makes no network request and imports
              nothing from any other repository.
Last Sprint:  Sprint R0-D — Registered-read emissions (2026-10-04)
================================================================================

Reads data/sharadar/snapshot_<label>/manifest.json in each snapshot, groups the
per-request parquets into logical vendor tables, and compares them on their key
fields. The substance is the price test: a bar whose unadjusted close or volume
moved after settling is a vendor REVISION; one whose adjusted series moved alone
is a corporate-action ADJUSTMENT. Tracked reports carry counts, ratios and
magnitude distributions only — never a price, volume or bar date. A capped
example file for human inspection is written outside the tracked tree.
================================================================================
"""

import argparse
import copy
import hashlib
import json
import os
import re
import sys

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRICE_TABLES = ("stocks", "funds")
# Tables whose rows are dated events rather than bars; tallied by event type.
EVENT_TABLES = ("actions", "sp500")
PRICE_FIELDS = ["open", "high", "low", "close", "volume", "closeadj", "closeunadj"]
# open/high/low/close/closeadj all carry split and stock-dividend adjustment.
ADJUSTABLE_FIELDS = ["open", "high", "low", "close", "closeadj"]
# closeunadj and volume are as-traded: a move here after settling is a revision.
UNADJUSTED_FIELDS = ["closeunadj", "volume"]
WATERMARK_FIELD = "lastupdated"
TAIL_DAYS = 7
AGE_BUCKETS = (("<=7", 0, 7), ("8-30", 8, 30), ("31-365", 31, 365), (">365", 366, None))
EXAMPLE_ROW_CAP = 10

# Keys fixed by the sprint brief. Anything not listed falls back to the vendor's
# own isprimarykey flags, and failing that to a whole-row set comparison.
BRIEF_KEYS = {
    "tickers": ["permaticker"],
    "sp500": ["ticker", "date", "action"],
    "stocks": ["ticker", "date"],
    "funds": ["ticker", "date"],
    "descriptions": ["table", "indicator"],
    "metrics": ["ticker", "date"],
}

# Manifest entry name -> logical vendor table. Ordered; first match wins.
# None means the entry is not table data (docs pages, schema index).
_NAME_RULES = (
    (r"^schema_index$", None),
    (r"^schema_(.+)$", None),
    (r"^docs_.*$", None),
    (r"^page_.*$", None),
    (r"^llms_.*$", None),
    (r"^desc_all$", "descriptions"),
    (r"^probe5_(.+)$", r"\1"),
    (r"^restrict_(.+)_[A-Z0-9]+$", r"\1"),
    (r"^tickers_stocks_\d+$", "tickers"),
    (r"^sp500_all_\d+$", "sp500"),
    (r"^stocks_.*$", "stocks"),
    (r"^funds_.*$", "funds"),
    # R0-C bulk pull: one entry per whole table.
    (r"^bulk_(.+)$", r"\1"),
    # R0-C tier probes: small samples of real tables, not separate tables.
    (r"^tier_tickers_free$", "tickers"),
    (r"^tier_stocks_[A-Z0-9.]+$", "stocks"),
    (r"^tier_sp500$", "sp500"),
)

MISSING = "\x00<missing>"


def entry_table(name):
    """Map a manifest entry name to its logical vendor table, or None."""
    for pattern, target in _NAME_RULES:
        m = re.match(pattern, name)
        if m:
            if target is None:
                return None
            return m.expand(target) if target.startswith("\\") else target
    return None


def snapshot_dir(root, label):
    return os.path.join(root, "data", "sharadar", "snapshot_%s" % label)


def read_manifest(root, label):
    path = os.path.join(snapshot_dir(root, label), "manifest.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def is_error_payload(entry):
    cols = set(entry.get("columns") or [])
    return bool(cols) and cols <= {"error", "description"}


def declared_schema(snap_dir):
    """Vendor-declared indicators and primary keys, read from desc_all.parquet."""
    path = os.path.join(snap_dir, "desc_all.parquet")
    if not os.path.exists(path):
        return {}, {}
    d = pd.read_parquet(path)
    indicators, keys = {}, {}
    for table, grp in d.groupby("table"):
        indicators[table] = sorted(set(grp["indicator"].astype(str)))
        pk = grp[grp["isprimarykey"].astype(str).str.upper() == "Y"]["indicator"].astype(str)
        if len(pk):
            keys[table] = sorted(set(pk))
    return indicators, keys


def resolve_key(table, declared_keys):
    """Return (key_fields, source_label). Empty key means whole-row set compare."""
    if table in BRIEF_KEYS:
        return list(BRIEF_KEYS[table]), "sprint brief (vendor declares: %s)" % (
            ",".join(declared_keys.get(table, [])) or "none")
    if table in declared_keys:
        return list(declared_keys[table]), "vendor isprimarykey (desc_all)"
    return [], "none (whole-row set comparison)"


def _norm(series):
    """Normalise a column to comparable strings; missing collapses to one token."""
    s = series.astype("object").where(series.notna(), None)
    out = s.map(lambda v: MISSING if v is None else str(v).strip())
    return out.mask(out.isin(["", "nan", "None", "NaN", "<NA>", "NaT"]), MISSING)


def _numeric(series):
    return pd.to_numeric(series.replace({MISSING: None}), errors="coerce")


def load_table(snap_dir, manifest, table):
    """Concatenate every entry belonging to `table`, tagged with its provenance."""
    frames, provenance, skipped = [], [], []
    for entry in manifest:
        if not entry.get("parquet_file"):
            continue
        if entry_table(entry["name"]) != table:
            continue
        if is_error_payload(entry):
            skipped.append({"entry": entry["name"], "reason": "vendor error payload"})
            continue
        path = os.path.join(snap_dir, entry["parquet_file"])
        if not os.path.exists(path):
            skipped.append({"entry": entry["name"], "reason": "parquet absent"})
            continue
        df = pd.read_parquet(path)
        if not len(df.columns):
            skipped.append({"entry": entry["name"], "reason": "no columns"})
            continue
        df = df.copy()
        df["__src"] = entry["name"]
        df["__fetched_at"] = entry.get("fetched_at_utc")
        frames.append(df)
        provenance.append({"entry": entry["name"], "rows": int(len(df))})
    if not frames:
        return None, provenance, skipped
    return pd.concat(frames, ignore_index=True), provenance, skipped


def dedupe_on_key(df, key, compare_cols):
    """Keep the first row per key (manifest order) and count value conflicts."""
    if not key or any(k not in df.columns for k in key):
        return df, None
    sig = None
    for col in compare_cols:
        part = _norm(df[col])
        sig = part if sig is None else sig.str.cat(part, sep="\x01")
    conflicts = 0
    if sig is not None:
        tmp = df[key].astype(str).copy()
        tmp["__sig"] = sig.values
        conflicts = int((tmp.groupby(key, dropna=False)["__sig"].nunique() > 1).sum())
    return df.drop_duplicates(subset=key, keep="first"), conflicts


def _keyed(df, key):
    idx = pd.MultiIndex.from_frame(df[key].astype(str)) if len(key) > 1 else pd.Index(
        df[key[0]].astype(str))
    out = df.copy()
    out.index = idx
    return out


def compare_keyed(a, b, key, vendor_cols, derived_cols):
    """Row membership and per-field change counts for one keyed table."""
    ak, bk = _keyed(a, key), _keyed(b, key)
    only_a = ak.index.difference(bk.index)
    only_b = bk.index.difference(ak.index)
    both = ak.index.intersection(bk.index)
    ab, bb = ak.loc[both], bk.loc[both]
    per_field, changed_masks = {}, {}
    for col in list(vendor_cols) + list(derived_cols):
        na, nb = _norm(ab[col]), _norm(bb[col])
        mask = (na.values != nb.values)
        per_field[col] = int(mask.sum())
        changed_masks[col] = mask
    def any_changed(cols):
        if not cols:
            return 0
        acc = None
        for col in cols:
            acc = changed_masks[col] if acc is None else (acc | changed_masks[col])
        return int(acc.sum())
    return {
        "rows_only_in_a": int(len(only_a)),
        "rows_only_in_b": int(len(only_b)),
        "rows_in_both": int(len(both)),
        "rows_in_both_differing_vendor_fields": any_changed(
            [c for c in vendor_cols if c != WATERMARK_FIELD]),
        "rows_in_both_differing_any_shared_field": any_changed(
            list(vendor_cols) + list(derived_cols)),
        "watermark_only_changes": per_field.get(WATERMARK_FIELD, 0),
        "per_field_changed": per_field,
    }, ab, bb, ak.loc[only_a], bk.loc[only_b]


def _sigs(df, cols):
    """Per-row signature over `cols`, as a Series aligned to df. Pure."""
    s = None
    for col in cols:
        part = _norm(df[col])
        s = part if s is None else s.str.cat(part, sep="\x01")
    return s if s is not None else pd.Series([""] * len(df), index=df.index)


def compare_rowsets(a, b, cols):
    """Whole-row set comparison for a table with no stated key."""
    def sig(df):
        s = None
        for col in cols:
            part = _norm(df[col])
            s = part if s is None else s.str.cat(part, sep="\x01")
        return set(s) if s is not None else set()
    sa, sb = sig(a), sig(b)
    counts = {
        "rows_only_in_a": len(sa - sb),
        "rows_only_in_b": len(sb - sa),
        "rows_in_both": len(sa & sb),
        "rows_in_both_differing_vendor_fields": None,
        "rows_in_both_differing_any_shared_field": None,
        "watermark_only_changes": None,
        "per_field_changed": {},
    }
    # The one-sided frames let a keyless table still be tallied by event type.
    only_a = a[~_sigs(a, cols).isin(sb).values] if len(a) else a
    only_b = b[~_sigs(b, cols).isin(sa).values] if len(b) else b
    return counts, only_a, only_b


def bucket_age(days):
    for label, lo, hi in AGE_BUCKETS:
        if days >= lo and (hi is None or days <= hi):
            return label
    return AGE_BUCKETS[-1][0]


def _dist(values):
    s = pd.Series([v for v in values if pd.notna(v)], dtype="float64")
    if not len(s):
        return None
    return {
        "n": int(len(s)),
        "min": float(s.min()),
        "median": float(s.median()),
        "p95": float(s.quantile(0.95)),
        "max": float(s.max()),
    }


# ---------------------------------------------------------------------------
# Registered reads (Sprint R0-D). Each function below answers one pre-registered
# read and is pure: it takes frames and dates, touches no file and no network.
# The diff reports against the reads; it does not choose them.
# ---------------------------------------------------------------------------

# R0-B-0's own split drives the headline REVISION count, so the number stays
# comparable with R0-B's "0 of 30 on five years". The R0-D brief's parenthetical
# widens as-traded to every price field; that alternative is emitted beside the
# headline rather than substituted for it, so the two rules can be compared
# instead of one being assumed.
REVISION_RULES = {
    "rule_tool": tuple(UNADJUSTED_FIELDS),
    "rule_brief_gloss": ("open", "high", "low", "close", "volume", "closeunadj"),
}
# A corporate action that can move an adjusted series.
ADJUSTING_ACTIONS = ("dividend", "split", "spinoff", "stockdividend")


def _by_year(values):
    """{'YYYY': count} over a date-like iterable; unparseable buckets 'unknown'."""
    d = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce")
    out = {}
    for v in d:
        k = "unknown" if pd.isna(v) else "%04d" % v.year
        out[k] = out.get(k, 0) + 1
    return out


def _by_date(values):
    """{'YYYY-MM-DD': count} over a date-like iterable."""
    d = pd.to_datetime(pd.Series(list(values), dtype="object"), errors="coerce")
    out = {}
    for v in d:
        k = "unknown" if pd.isna(v) else v.strftime("%Y-%m-%d")
        out[k] = out.get(k, 0) + 1
    return out


def _add_counts(target, addition):
    """Sum two tallies into a new dict, recursing into nested tallies. Pure."""
    out = dict(target or {})
    for k, v in (addition or {}).items():
        if isinstance(v, dict):
            out[k] = _add_counts(out.get(k) or {}, v)
        elif isinstance(v, int) and not isinstance(v, bool):
            out[k] = out.get(k, 0) + v
        else:
            out.setdefault(k, v)
    return out


def membership_reads(only_a_rows, only_b_rows, a_tickers, cutoff_date):
    """R1 and R2: rows that vanished, and rows that appeared, dated.

    R1 asks that no historical row disappear, so every only-A row is counted by
    the year it is dated in. R2 expects only-B rows to be dated after the A-side
    data edge; one dated at or before `cutoff_date` is a BACKFILL — history the
    vendor had already served once without it — split by whether its ticker was
    in A at all, because a name new to coverage arrives with its whole past.
    """
    out = {"cutoff_date": cutoff_date}
    a_dates = list(only_a_rows["date"]) if (only_a_rows is not None
                                            and "date" in only_a_rows) else []
    out["only_in_a_rows"] = len(a_dates)
    out["only_in_a_by_year"] = _by_year(a_dates)

    if only_b_rows is None or "date" not in only_b_rows:
        out["only_in_b_rows"] = 0
        out["after_cutoff_rows"] = 0
        out["after_cutoff_by_date"] = {}
        out["backfill_rows"] = 0
        out["backfill_by_year"] = {}
        out["backfill_new_listing_rows"] = 0
        out["backfill_existing_name_rows"] = 0
        out["backfill_new_listing_tickers"] = 0
        out["backfill_existing_name_tickers"] = 0
        return out

    d = pd.to_datetime(only_b_rows["date"], errors="coerce")
    cut = pd.Timestamp(cutoff_date) if cutoff_date else None
    after = d > cut if cut is not None else pd.Series(True, index=d.index)
    after = after.fillna(False)
    back = ~after

    out["only_in_b_rows"] = int(len(d))
    out["after_cutoff_rows"] = int(after.sum())
    out["after_cutoff_by_date"] = _by_date(only_b_rows.loc[after.values, "date"])
    out["backfill_rows"] = int(back.sum())
    out["backfill_by_year"] = _by_year(only_b_rows.loc[back.values, "date"])

    known = set(a_tickers or ())
    bt = only_b_rows["ticker"].astype(str) if "ticker" in only_b_rows else None
    if bt is None:
        out["backfill_new_listing_rows"] = None
        out["backfill_existing_name_rows"] = None
        out["backfill_new_listing_tickers"] = None
        out["backfill_existing_name_tickers"] = None
        return out
    bt_back = bt[back.values]
    is_new = ~bt_back.isin(known)
    out["backfill_new_listing_rows"] = int(is_new.sum())
    out["backfill_existing_name_rows"] = int((~is_new).sum())
    out["backfill_new_listing_tickers"] = int(bt_back[is_new.values].nunique())
    out["backfill_existing_name_tickers"] = int(bt_back[(~is_new).values].nunique())
    return out


def revision_reads(ab, bb, settle_cutoff):
    """R3's first half: value-changed rows classified REVISION or ADJUSTMENT.

    Row-level, not per-ticker: R3 asks how many *rows* carry an as-traded change
    and how they are dated, which the per-ticker class tally cannot answer. Both
    rules are measured — see REVISION_RULES.
    """
    present = [c for c in PRICE_FIELDS if c in ab.columns and c in bb.columns]
    masks = {}
    for col in present:
        na, nb = _norm(ab[col]), _norm(bb[col])
        masks[col] = pd.Series(na.values != nb.values, index=ab.index)
    any_change = None
    for m in masks.values():
        any_change = m if any_change is None else (any_change | m)
    if any_change is None:
        any_change = pd.Series(False, index=ab.index)

    dates = ab["date"] if "date" in ab.columns else pd.Series(index=ab.index, dtype="object")
    d = pd.to_datetime(dates, errors="coerce")
    cut = pd.Timestamp(settle_cutoff) if settle_cutoff else None

    out = {"settle_cutoff": settle_cutoff,
           "fields_compared": present,
           "value_changed_rows": int(any_change.sum()),
           "rules": {}}
    for rule, as_traded in REVISION_RULES.items():
        at = [c for c in as_traded if c in masks]
        rev = None
        for col in at:
            rev = masks[col] if rev is None else (rev | masks[col])
        if rev is None:
            rev = pd.Series(False, index=ab.index)
        adj = any_change & ~rev
        ent = {"as_traded_fields": at,
               "revision_rows": int(rev.sum()),
               "adjustment_only_rows": int(adj.sum())}
        if cut is not None:
            before = rev & (d < cut).fillna(False).values
            onafter = rev & (d >= cut).fillna(False).values
            ent["revision_rows_before_cutoff"] = int(before.sum())
            ent["revision_rows_on_or_after_cutoff"] = int(onafter.sum())
            ent["revision_before_cutoff_by_year"] = _by_year(dates[before.values])
            ent["revision_on_or_after_by_date"] = _by_date(dates[onafter.values])
        out["rules"][rule] = ent
    return out


def watermark_reads(ab, bb):
    """R4: does a value change always carry the watermark forward?

    A row whose value moved without `lastupdated` advancing would make the
    watermark unusable for incremental loading, so that count is reported on its
    own. Watermark movement with no value change is benign and counted too.
    """
    if WATERMARK_FIELD not in ab.columns or WATERMARK_FIELD not in bb.columns:
        return {"status": "no watermark column on both sides"}
    present = [c for c in PRICE_FIELDS if c in ab.columns and c in bb.columns]
    changed = None
    for col in present:
        m = pd.Series(_norm(ab[col]).values != _norm(bb[col]).values, index=ab.index)
        changed = m if changed is None else (changed | m)
    if changed is None:
        changed = pd.Series(False, index=ab.index)

    wa = pd.to_datetime(_norm(ab[WATERMARK_FIELD]).replace({MISSING: None}),
                        errors="coerce")
    wb = pd.to_datetime(_norm(bb[WATERMARK_FIELD]).replace({MISSING: None}),
                        errors="coerce")
    advanced = pd.Series((wb > wa).fillna(False).values, index=ab.index)
    moved = pd.Series(
        (_norm(ab[WATERMARK_FIELD]).values != _norm(bb[WATERMARK_FIELD]).values),
        index=ab.index)
    return {
        "fields_compared": present,
        "value_changed_rows": int(changed.sum()),
        "value_changed_with_watermark_advance": int((changed & advanced).sum()),
        "value_changed_without_watermark_advance": int((changed & ~advanced).sum()),
        "watermark_moved_without_value_change": int((moved & ~changed).sum()),
        "watermark_advanced_without_value_change": int((advanced & ~changed).sum()),
        "watermark_unparseable_rows": int((wa.isna() | wb.isna()).sum()),
    }


def delisted_split_reads(ab, bb, as_of_a="isdelisted"):
    """R5: per-field change counts split by whether A had the row delisted.

    A delisted name's record is closed history; any field moving on it is a
    different claim from a live name's `lastpricedate` ticking forward, so the
    two populations are counted separately rather than pooled.
    """
    if as_of_a not in ab.columns:
        return {"status": "no %s column" % as_of_a}
    flag = _norm(ab[as_of_a]).str.upper().isin(["Y", "TRUE", "1"])
    shared = [c for c in ab.columns
              if c in set(bb.columns) and not c.startswith("__") and c != as_of_a]
    out = {"delisted_rows": int(flag.sum()), "active_rows": int((~flag).sum()),
           "delisted_changed_by_field": {}, "active_changed_by_field": {},
           "delisted_rows_changed_any_field": 0,
           "active_rows_changed_any_field": 0}
    any_del, any_act = None, None
    for col in shared:
        m = pd.Series(_norm(ab[col]).values != _norm(bb[col]).values, index=ab.index)
        nd, na_ = int((m & flag).sum()), int((m & ~flag).sum())
        if nd:
            out["delisted_changed_by_field"][col] = nd
        if na_:
            out["active_changed_by_field"][col] = na_
        md, ma = (m & flag), (m & ~flag)
        any_del = md if any_del is None else (any_del | md)
        any_act = ma if any_act is None else (any_act | ma)
    out["delisted_rows_changed_any_field"] = int(any_del.sum()) if any_del is not None else 0
    out["active_rows_changed_any_field"] = int(any_act.sum()) if any_act is not None else 0
    return out


def by_type_reads(only_a_rows, only_b_rows, type_col="action", date_col="date"):
    """R6: membership change on an event table, tallied by the event's own type."""
    out = {"type_column": type_col}
    for side, rows in (("only_in_a", only_a_rows), ("only_in_b", only_b_rows)):
        if rows is None or type_col not in getattr(rows, "columns", []):
            out[side + "_rows"] = 0 if rows is None else int(len(rows))
            out[side + "_by_type"] = {}
            out[side + "_by_date"] = {}
            continue
        t = _norm(rows[type_col])
        tally = {}
        for v in t:
            tally[v] = tally.get(v, 0) + 1
        out[side + "_rows"] = int(len(rows))
        out[side + "_by_type"] = tally
        out[side + "_by_date"] = (_by_date(rows[date_col])
                                  if date_col in rows.columns else {})
    return out


def explain_adjustments(adjustment_tickers, actions_b, window_start, window_end,
                        kinds=ADJUSTING_ACTIONS):
    """R3's second half: is each ADJUSTMENT ticker backed by a corporate action?

    An adjusted series that moved with no dividend or split in the window is
    unexplained, which is the finding R3 asks for. Only the action's type and the
    ticker are reported — never the action's `value`, which is a vendor figure.
    """
    out = {"window": [window_start, window_end],
           "action_kinds": list(kinds),
           "adjustment_tickers": len(adjustment_tickers)}
    if actions_b is None or not len(adjustment_tickers):
        out["status"] = "no actions table" if actions_b is None else "no adjustment tickers"
        out["explained"] = 0
        out["unexplained"] = len(adjustment_tickers)
        out["explained_by_type"] = {}
        out["unexplained_tickers"] = sorted(adjustment_tickers)
        return out

    d = pd.to_datetime(actions_b["date"], errors="coerce")
    lo = pd.Timestamp(window_start) if window_start else None
    hi = pd.Timestamp(window_end) if window_end else None
    sel = pd.Series(True, index=actions_b.index)
    if lo is not None:
        sel &= (d > lo).fillna(False)
    if hi is not None:
        sel &= (d <= hi).fillna(False)
    kind = _norm(actions_b["action"]).str.lower()
    sel &= kind.isin([k.lower() for k in kinds]).values
    win = actions_b[sel.values]

    wanted = set(str(t) for t in adjustment_tickers)
    hit = win[win["ticker"].astype(str).isin(wanted)]
    explained = set(hit["ticker"].astype(str))
    by_type = {}
    for v in _norm(hit["action"]).str.lower():
        by_type[v] = by_type.get(v, 0) + 1
    out["actions_in_window"] = int(len(win))
    out["actions_in_window_on_adjustment_tickers"] = int(len(hit))
    out["explained"] = len(explained)
    out["unexplained"] = len(wanted - explained)
    out["explained_by_type"] = by_type
    out["unexplained_tickers"] = sorted(wanted - explained)
    return out


def price_analysis(ab, bb, key, examples, pooled=None):
    """Per-field magnitudes by age bucket and a per-ticker class for each ticker.

    When `pooled` is supplied, the raw abs/rel magnitudes are appended to it as
    well as summarised. A chunked run pools across partitions and recomputes the
    distribution once, so its numbers equal an unchunked run's rather than being
    an average of per-partition summaries."""
    ages = (pd.to_datetime(ab["__fetched_at"], errors="coerce", utc=True).dt.tz_localize(None)
            - pd.to_datetime(ab["date"], errors="coerce")).dt.days
    buckets = ages.map(lambda d: bucket_age(int(d)) if pd.notna(d) else "unknown")
    tickers = ab["ticker"].astype(str)
    dates = ab["date"].astype(str)

    fields, changed_any = {}, {}
    for col in PRICE_FIELDS:
        if col not in ab.columns or col not in bb.columns:
            continue
        na, nb = _norm(ab[col]), _norm(bb[col])
        mask = pd.Series(na.values != nb.values, index=ab.index)
        changed_any[col] = mask
        fa, fb = _numeric(na), _numeric(nb)
        absd = (fb.values - fa.values)
        absd = pd.Series(absd, index=ab.index).abs().where(mask)
        rel = (absd / _numeric(na).abs().replace(0, pd.NA)).where(mask)
        by_bucket = {}
        for label, _lo, _hi in AGE_BUCKETS:
            sel = mask & (buckets.values == label)
            by_bucket[label] = {
                "changed": int(sel.sum()),
                "abs": _dist(absd[sel].tolist()),
                "rel": _dist(rel[sel].tolist()),
            }
        fields[col] = {
            "changed": int(mask.sum()),
            "abs": _dist(absd[mask].tolist()),
            "rel": _dist(rel[mask].tolist()),
            "by_age_bucket": by_bucket,
        }
        if pooled is not None:
            slot = pooled.setdefault(col, {"abs": [], "rel": [], "by_bucket": {}})
            slot["abs"].extend(absd[mask].tolist())
            slot["rel"].extend(rel[mask].tolist())
            for label, _lo, _hi in AGE_BUCKETS:
                sel = mask & (buckets.values == label)
                bslot = slot["by_bucket"].setdefault(label, {"abs": [], "rel": []})
                bslot["abs"].extend(absd[sel].tolist())
                bslot["rel"].extend(rel[sel].tolist())
        if len(examples) < EXAMPLE_ROW_CAP:
            for i in ab.index[mask][: EXAMPLE_ROW_CAP - len(examples)]:
                examples.append({
                    "ticker": tickers.loc[i] if not isinstance(tickers.loc[i], pd.Series)
                    else tickers.loc[i].iloc[0],
                    "date": dates.loc[i] if not isinstance(dates.loc[i], pd.Series)
                    else dates.loc[i].iloc[0],
                    "field": col,
                    "old": str(na.loc[i]) if not isinstance(na.loc[i], pd.Series)
                    else str(na.loc[i].iloc[0]),
                    "new": str(nb.loc[i]) if not isinstance(nb.loc[i], pd.Series)
                    else str(nb.loc[i].iloc[0]),
                })

    classes, revision_detail = {}, {}
    old_bar = (ages > TAIL_DAYS).fillna(False)
    for ticker, idx in ab.groupby(tickers).groups.items():
        touched_old, touched_any, per_bucket = set(), set(), {}
        for col, mask in changed_any.items():
            m = mask.loc[idx]
            if not m.any():
                continue
            touched_any.add(col)
            mo = m & old_bar.loc[idx]
            if mo.any():
                touched_old.add(col)
                for label, _lo, _hi in AGE_BUCKETS:
                    sel = mo & (pd.Series(buckets.loc[idx].values, index=idx) == label)
                    if sel.any():
                        per_bucket.setdefault(label, {})[col] = int(sel.sum())
        if not touched_any:
            cls = "UNCHANGED"
        elif not touched_old:
            cls = "TAIL"
        elif touched_old & set(UNADJUSTED_FIELDS):
            cls = "REVISION"
        else:
            cls = "ADJUSTMENT"
        classes[str(ticker)] = cls
        if cls == "REVISION":
            revision_detail[str(ticker)] = {
                "fields_on_settled_bars": sorted(touched_old),
                "bars_by_age_bucket": per_bucket,
            }
    tally = {}
    for cls in classes.values():
        tally[cls] = tally.get(cls, 0) + 1
    return {
        "tickers_compared": len(classes),
        "class_tally": tally,
        "fields": fields,
        "revision_tickers": revision_detail,
        # In-memory only: stripped before the report is written, so the tracked
        # artifact carries class tallies and revision detail, nothing wider.
        "classes_by_ticker": classes,
    }


# ---------------------------------------------------------------------------
# Chunked comparison. The full-history price table does not fit in memory twice,
# so it is compared one ticker partition at a time. The key is (ticker, date),
# so partitioning the ticker space yields independent sub-problems whose results
# merge exactly — no approximation, and no cross-partition row can be missed.
# ---------------------------------------------------------------------------

def ticker_bucket(ticker, n_buckets):
    """Stable partition index for a ticker. Pure.

    md5, not hash(): the built-in is salted per process, which would put the same
    ticker in different partitions on the two sides of a comparison.
    """
    if n_buckets <= 1:
        return 0
    digest = hashlib.md5(str(ticker).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % n_buckets


def partition_column(table, key):
    """Which column may a table be partitioned on? Pure.

    Partitioning is only exact when the partition column is a component of the
    comparison key: two rows sharing a key must land in the same partition, or
    per-partition dedupe keeps both and inflates the row count. `tickers` keys on
    permaticker and 239 permatickers carry more than one ticker string, so
    partitioning that table by ticker would over-count by exactly those rows.

    Price tables must partition on ticker, because their per-ticker
    classification needs every bar of a ticker in one partition; ticker is a
    component of their (ticker, date) key, so that is also exact.
    """
    if table in PRICE_TABLES:
        return "ticker" if key and "ticker" in key else None
    if key:
        return "ticker" if "ticker" in key else key[0]
    # No declared key: the comparison is a whole-row set, whose signature
    # includes ticker, so identical rows cannot straddle a ticker partition.
    return "ticker"


def table_has_column(snap_dir, manifest, table, column):
    """Does this table's first parquet carry `column`? Reads metadata only."""
    for entry in manifest:
        if not entry.get("parquet_file") or is_error_payload(entry):
            continue
        if entry_table(entry["name"]) != table:
            continue
        path = os.path.join(snap_dir, entry["parquet_file"])
        if os.path.exists(path):
            return column in pq.ParquetFile(path).schema_arrow.names
    return False


def load_table_bucket(snap_dir, manifest, table, bucket, n_buckets,
                      column="ticker"):
    """Load only the rows of `table` whose `column` value falls in `bucket`.

    Reads one parquet row group at a time and discards the rest immediately, so
    peak memory is one row group plus the retained partition — not the table.
    """
    frames, provenance, skipped = [], [], []
    for entry in manifest:
        if not entry.get("parquet_file"):
            continue
        if entry_table(entry["name"]) != table:
            continue
        if is_error_payload(entry):
            skipped.append({"entry": entry["name"], "reason": "vendor error payload"})
            continue
        path = os.path.join(snap_dir, entry["parquet_file"])
        if not os.path.exists(path):
            skipped.append({"entry": entry["name"], "reason": "parquet absent"})
            continue
        pf = pq.ParquetFile(path)
        if column not in pf.schema_arrow.names:
            skipped.append({"entry": entry["name"],
                            "reason": "no %s column to partition on" % column})
            continue
        kept, total = [], 0
        for i in range(pf.metadata.num_row_groups):
            part = pf.read_row_group(i).to_pandas()
            total += len(part)
            if not len(part.columns):
                continue
            sel = part[column].map(lambda t: ticker_bucket(t, n_buckets) == bucket)
            part = part[sel.values]
            if len(part):
                kept.append(part)
            del sel
        if not kept:
            provenance.append({"entry": entry["name"], "rows": 0,
                               "rows_in_entry": total})
            continue
        df = pd.concat(kept, ignore_index=True)
        df["__src"] = entry["name"]
        df["__fetched_at"] = entry.get("fetched_at_utc")
        frames.append(df)
        provenance.append({"entry": entry["name"], "rows": int(len(df)),
                           "rows_in_entry": total})
    if not frames:
        return None, provenance, skipped
    return pd.concat(frames, ignore_index=True), provenance, skipped


def _sum_or_none(values):
    """Sum, treating None as unknown: any None makes the total unknown."""
    vals = list(values)
    if any(v is None for v in vals):
        return None
    return sum(vals)


def merge_table_entries(entries, pooled=None):
    """Combine per-partition comparison entries into one. Pure.

    Integer counts add, because the partitions are disjoint row sets. Per-ticker
    maps union, because a ticker lives in exactly one partition. Magnitude
    distributions are recomputed from `pooled` raw values rather than averaged
    out of per-partition summaries, so a chunked run reports the same numbers an
    unchunked run would.
    """
    if not entries:
        return {}
    if len(entries) == 1 and pooled is None:
        return dict(entries[0])
    out = dict(entries[0])
    out["partitions"] = len(entries)

    for field in ("rows_only_in_a", "rows_only_in_b", "rows_in_both",
                  "rows_in_both_differing_vendor_fields",
                  "rows_in_both_differing_any_shared_field",
                  "watermark_only_changes"):
        if any(field in e for e in entries):
            out[field] = _sum_or_none(e.get(field) for e in entries)

    per_field = {}
    for e in entries:
        for col, n in (e.get("per_field_changed") or {}).items():
            per_field[col] = per_field.get(col, 0) + n
    out["per_field_changed"] = per_field

    for field, side in (("rows_before_dedupe", None), ("rows_after_dedupe", None),
                        ("intra_snapshot_key_conflicts", None)):
        if field in entries[0] and isinstance(entries[0][field], dict):
            merged = {}
            for e in entries:
                for k, v in (e.get(field) or {}).items():
                    if v is None:
                        merged[k] = None
                    elif merged.get(k, 0) is not None:
                        merged[k] = merged.get(k, 0) + v
            out[field] = merged

    prov_a, prov_b, skipped = [], [], []
    for e in entries:
        prov_a.extend(e.get("source_entries_a") or [])
        prov_b.extend(e.get("source_entries_b") or [])
        skipped.extend(e.get("entries_skipped") or [])
    out["source_entries_a"] = _fold_provenance(prov_a)
    out["source_entries_b"] = _fold_provenance(prov_b)
    out["entries_skipped"] = _dedupe_dicts(skipped)

    pas = [e["price_analysis"] for e in entries if e.get("price_analysis")]
    if pas:
        out["price_analysis"] = _merge_price_analysis(pas, pooled)
    reads = [e["registered_reads"] for e in entries if e.get("registered_reads")]
    if reads:
        out["registered_reads"] = _merge_reads(reads)
    return out


def _fold_provenance(items):
    """Sum a partition's per-entry row counts back into one row per entry."""
    acc = {}
    for it in items:
        cur = acc.setdefault(it["entry"], {"entry": it["entry"], "rows": 0})
        cur["rows"] += it.get("rows", 0)
        if "rows_in_entry" in it:
            cur["rows_in_entry"] = it["rows_in_entry"]
    return list(acc.values())


def _dedupe_dicts(items):
    seen, out = set(), []
    for it in items:
        k = json.dumps(it, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _merge_reads(reads_list):
    """Merge per-partition registered-read blocks. Pure.

    Partitions are disjoint row sets and a ticker lives in exactly one of them,
    so integer counts add and tally dicts add key-wise; descriptive scalars are
    taken from the first partition that states them.
    """
    if not reads_list:
        return {}
    out = {}
    for name in sorted({k for r in reads_list for k in r}):
        blocks = [r[name] for r in reads_list if name in r]
        merged = {}
        for b in blocks:
            for k, v in b.items():
                if k == "rules":
                    continue  # nested one block per rule; merged below
                if isinstance(v, bool) or v is None:
                    merged.setdefault(k, v)
                elif isinstance(v, int):
                    merged[k] = merged.get(k, 0) + v
                elif isinstance(v, dict):
                    cur = merged.get(k)
                    merged[k] = _add_counts(cur, v) if isinstance(cur, dict) else dict(v)
                elif isinstance(v, list):
                    cur = merged.get(k) or []
                    merged[k] = cur + [x for x in v if x not in cur]
                else:
                    merged.setdefault(k, v)
        # "rules" nests one block per revision rule; merge each rule on its own.
        if name == "revision" and any("rules" in b for b in blocks):
            merged["rules"] = _merge_reads(
                [{r: b["rules"][r] for r in b["rules"]} for b in blocks if "rules" in b])
        out[name] = merged
    return out


def _merge_price_analysis(pas, pooled=None):
    """Merge per-partition price analyses; disjoint tickers make it exact."""
    out = {"tickers_compared": sum(p["tickers_compared"] for p in pas)}
    tally = {}
    for p in pas:
        for cls, n in (p.get("class_tally") or {}).items():
            tally[cls] = tally.get(cls, 0) + n
    out["class_tally"] = tally

    classes = {}
    revisions = {}
    for p in pas:
        classes.update(p.get("classes_by_ticker") or {})
        revisions.update(p.get("revision_tickers") or {})
    out["classes_by_ticker"] = classes
    out["revision_tickers"] = revisions

    fields = {}
    cols = []
    for p in pas:
        for col in (p.get("fields") or {}):
            if col not in cols:
                cols.append(col)
    for col in cols:
        changed = sum((p.get("fields") or {}).get(col, {}).get("changed", 0)
                      for p in pas)
        ent = {"changed": changed}
        pool = (pooled or {}).get(col, {})
        ent["abs"] = _dist(pool.get("abs", [])) if pool else None
        ent["rel"] = _dist(pool.get("rel", [])) if pool else None
        by_bucket = {}
        for label, _lo, _hi in AGE_BUCKETS:
            bchanged = sum(((p.get("fields") or {}).get(col, {})
                            .get("by_age_bucket", {}).get(label, {}) or {})
                           .get("changed", 0) for p in pas)
            bpool = (pool.get("by_bucket") or {}).get(label, {})
            by_bucket[label] = {
                "changed": bchanged,
                "abs": _dist(bpool.get("abs", [])) if bpool else None,
                "rel": _dist(bpool.get("rel", [])) if bpool else None,
            }
        ent["by_age_bucket"] = by_bucket
        fields[col] = ent
    out["fields"] = fields
    return out


def peak_rss_gib():
    """Peak resident set size of this process, in GiB."""
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Darwin reports bytes; Linux reports kilobytes.
    if sys.platform != "darwin":
        rss *= 1024
    return round(rss / 1024 ** 3, 3)


def diff_snapshots(root, label_a, label_b, examples_cap=EXAMPLE_ROW_CAP,
                   chunk_by_ticker=None, reads=None):
    dir_a, dir_b = snapshot_dir(root, label_a), snapshot_dir(root, label_b)
    man_a, man_b = read_manifest(root, label_a), read_manifest(root, label_b)
    _ind_a, keys_a = declared_schema(dir_a)
    indicators, _keys_b = declared_schema(dir_b)
    indicators = indicators or _ind_a

    def tables_of(man):
        out = {}
        for e in man:
            if not e.get("parquet_file") or is_error_payload(e):
                continue
            t = entry_table(e["name"])
            if t:
                out.setdefault(t, []).append(e["name"])
        return out

    reads = dict(reads or {})
    onlyb_cutoff = reads.get("onlyb_cutoff_date")
    revision_cutoff = reads.get("revision_cutoff_date")
    act_lo = reads.get("action_window_start")
    act_hi = reads.get("action_window_end")

    ta, tb = tables_of(man_a), tables_of(man_b)
    both_tables = sorted(set(ta) & set(tb))
    unmapped = sorted({e["name"] for m in (man_a, man_b) for e in m
                       if e.get("parquet_file") and not is_error_payload(e)
                       and entry_table(e["name"]) is None})

    report = {
        "tool": "sharadar_snapshot_diff.py",
        "version": "1.2.0",
        "snapshot_a": label_a,
        "snapshot_b": label_b,
        "manifest_entries": {label_a: len(man_a), label_b: len(man_b)},
        "fetched_at_utc_range": {
            label_a: [min(e["fetched_at_utc"] for e in man_a if e.get("fetched_at_utc")),
                      max(e["fetched_at_utc"] for e in man_a if e.get("fetched_at_utc"))],
            label_b: [min(e["fetched_at_utc"] for e in man_b if e.get("fetched_at_utc")),
                      max(e["fetched_at_utc"] for e in man_b if e.get("fetched_at_utc"))],
        },
        "tables_only_in_a": sorted(set(ta) - set(tb)),
        "tables_only_in_b": sorted(set(tb) - set(ta)),
        "entries_only_in_a": sorted({e["name"] for e in man_a} - {e["name"] for e in man_b}),
        "entries_only_in_b": sorted({e["name"] for e in man_b} - {e["name"] for e in man_a}),
        "entries_unmapped_to_a_table": unmapped,
        "registered_read_parameters": {
            "onlyb_cutoff_date": onlyb_cutoff,
            "revision_cutoff_date": revision_cutoff,
            "action_window_start": act_lo,
            "action_window_end": act_hi,
        },
        "tables": {},
    }
    examples = []

    def compare_one(table, a, prov_a, skip_a, b, prov_b, skip_b, pooled=None):
        """Compare one table, or one ticker partition of it, and return its entry."""
        key, key_source = resolve_key(table, keys_a)
        shared = [c for c in a.columns if c in set(b.columns) and not c.startswith("__")]
        declared_cols = set(indicators.get(table, []))
        vendor_cols = [c for c in shared if not declared_cols or c in declared_cols]
        derived_cols = [c for c in shared if c not in vendor_cols]
        usable_key = key and all(k in shared for k in key)
        entry = {
            "key_fields": key if usable_key else [],
            "key_source": key_source if usable_key else "none (whole-row set comparison)",
            "vendor_declared_key": keys_a.get(table, []),
            "shared_columns": shared,
            "derived_columns_excluded_from_headline": derived_cols,
            "columns_only_in_a": sorted(set(a.columns) - set(b.columns) - {"__src", "__fetched_at"}),
            "columns_only_in_b": sorted(set(b.columns) - set(a.columns) - {"__src", "__fetched_at"}),
            "source_entries_a": prov_a,
            "source_entries_b": prov_b,
            "entries_skipped": skip_a + skip_b,
        }
        if usable_key:
            a2, ca = dedupe_on_key(a, key, vendor_cols)
            b2, cb = dedupe_on_key(b, key, vendor_cols)
            entry["rows_before_dedupe"] = {label_a: int(len(a)), label_b: int(len(b))}
            entry["rows_after_dedupe"] = {label_a: int(len(a2)), label_b: int(len(b2))}
            entry["intra_snapshot_key_conflicts"] = {label_a: ca, label_b: cb}
            counts, ab, bb, oa, ob = compare_keyed(a2, b2, key, vendor_cols,
                                                   derived_cols)
            entry.update(counts)
            rd = {}
            if table in PRICE_TABLES and "date" in shared and "ticker" in shared:
                entry["price_analysis"] = price_analysis(
                    ab, bb, key, examples if len(examples) < examples_cap else [],
                    pooled=pooled)
                rd["membership"] = membership_reads(
                    oa, ob, set(a2["ticker"].astype(str)), onlyb_cutoff)
                rd["revision"] = revision_reads(ab, bb, revision_cutoff)
                rd["watermark"] = watermark_reads(ab, bb)
            if "isdelisted" in shared:
                rd["delisted_split"] = delisted_split_reads(ab, bb)
            if table in EVENT_TABLES and "action" in shared:
                rd["by_type"] = by_type_reads(oa, ob, "action")
            if rd:
                entry["registered_reads"] = rd
        else:
            counts, oa, ob = compare_rowsets(a, b, vendor_cols)
            entry.update(counts)
            if table in EVENT_TABLES and "action" in shared:
                entry["registered_reads"] = {"by_type": by_type_reads(oa, ob, "action")}
        return entry

    for table in both_tables:
        n_chunks = chunk_by_ticker or 0
        part_key, _ks = resolve_key(table, keys_a)
        part_col = partition_column(table, part_key)
        chunkable = (n_chunks > 1 and part_col is not None
                     and table_has_column(dir_a, man_a, table, part_col)
                     and table_has_column(dir_b, man_b, table, part_col))
        if not chunkable:
            a, prov_a, skip_a = load_table(dir_a, man_a, table)
            b, prov_b, skip_b = load_table(dir_b, man_b, table)
            if a is None or b is None:
                report["tables"][table] = {"status": "no data on one side"}
                continue
            entry = compare_one(table, a, prov_a, skip_a, b, prov_b, skip_b)
            if n_chunks > 1:
                entry["chunking"] = {
                    "mode": "whole table",
                    "reason": ("no column that is both present and a component "
                               "of the comparison key %s" % (part_key or "[]")),
                    "partition_column": part_col}
            report["tables"][table] = entry
            del a, b
            continue

        parts, pooled, empty, one_sided = [], {}, 0, 0
        for bucket in range(n_chunks):
            a, prov_a, skip_a = load_table_bucket(dir_a, man_a, table, bucket,
                                                 n_chunks, column=part_col)
            b, prov_b, skip_b = load_table_bucket(dir_b, man_b, table, bucket,
                                                 n_chunks, column=part_col)
            if a is None and b is None:
                empty += 1
                del a, b
                continue
            # A partition held by one snapshot only is still a membership
            # difference: standing it up as an empty frame on the absent side
            # counts its rows as only-A or only-B, where skipping the partition
            # would have dropped them silently.
            if a is None or b is None:
                one_sided += 1
                if a is None:
                    a = b.iloc[0:0].copy()
                else:
                    b = a.iloc[0:0].copy()
            parts.append(compare_one(table, a, prov_a, skip_a, b, prov_b, skip_b,
                                     pooled=pooled))
            del a, b
        if not parts:
            report["tables"][table] = {"status": "no data on one side"}
            continue
        entry = merge_table_entries(parts, pooled=pooled)
        entry["chunking"] = {
            "mode": "key-component partition",
            "partition_column": part_col,
            "buckets": n_chunks,
            "buckets_compared": len(parts),
            "buckets_empty_on_both_sides": empty,
            "buckets_present_on_one_side_only": one_sided,
            "peak_rss_gib": peak_rss_gib(),
        }
        report["tables"][table] = entry

    # R3's explanation join runs once, after the partitions have merged: it needs
    # every ADJUSTMENT ticker of the whole table against the whole actions table,
    # which no single ticker partition holds.
    for table in both_tables:
        entry = report["tables"].get(table)
        if not isinstance(entry, dict) or table not in PRICE_TABLES:
            continue
        pa = entry.get("price_analysis") or {}
        classes = pa.get("classes_by_ticker") or {}
        adj = sorted(t for t, c in classes.items() if c == "ADJUSTMENT")
        actions_b = None
        if "actions" in tb:
            actions_b, _pv, _sk = load_table(dir_b, man_b, "actions")
        entry.setdefault("registered_reads", {})["adjustment_explanation"] = \
            explain_adjustments(adj, actions_b, act_lo, act_hi)
        del actions_b

    report["chunk_by_ticker"] = chunk_by_ticker
    report["peak_rss_gib"] = peak_rss_gib()
    return report, examples[:examples_cap]


def _fmt_dist(d):
    if not d:
        return "n/a"
    return "n=%d min=%.6g med=%.6g p95=%.6g max=%.6g" % (
        d["n"], d["min"], d["median"], d["p95"], d["max"])


def render_markdown(report, examples_path, examples_rows):
    L = []
    L.append("# Sharadar snapshot stability: %s vs %s" % (report["snapshot_a"], report["snapshot_b"]))
    L.append("")
    L.append("Statistics only. No price, volume or bar-date value appears in this file.")
    L.append("")
    L.append("| field | value |")
    L.append("| --- | --- |")
    L.append("| tool version | %s |" % report["version"])
    for lbl in (report["snapshot_a"], report["snapshot_b"]):
        rng = report["fetched_at_utc_range"][lbl]
        L.append("| fetched_at_utc %s | %s .. %s |" % (lbl, rng[0], rng[1]))
        L.append("| manifest entries %s | %d |" % (lbl, report["manifest_entries"][lbl]))
        L.append("| manifest sha %s | %s |" % (lbl, report["manifest_sha256"].get(lbl, "n/a")))
    L.append("| tables only in A | %s |" % (", ".join(report["tables_only_in_a"]) or "none"))
    L.append("| tables only in B | %s |" % (", ".join(report["tables_only_in_b"]) or "none"))
    L.append("| entries only in A | %s |" % (", ".join(report["entries_only_in_a"]) or "none"))
    L.append("| entries only in B | %s |" % (", ".join(report["entries_only_in_b"]) or "none"))
    L.append("| entries unmapped to a table | %s |"
             % (", ".join(report["entries_unmapped_to_a_table"]) or "none"))
    L.append("")
    L.append("## Per-table row membership")
    L.append("")
    L.append("| table | key fields | key source | only A | only B | in both | differing (vendor fields) | watermark-only |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for t, e in sorted(report["tables"].items()):
        if e.get("status"):
            L.append("| %s | — | %s | — | — | — | — | — |" % (t, e["status"]))
            continue
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            t, ",".join(e["key_fields"]) or "(whole row)", e["key_source"],
            e["rows_only_in_a"], e["rows_only_in_b"], e["rows_in_both"],
            e["rows_in_both_differing_vendor_fields"], e["watermark_only_changes"]))
    L.append("")
    L.append("## Intra-snapshot key collapse")
    L.append("")
    L.append("Several requests overlap, so a table is the concatenation of its source entries")
    L.append("deduplicated on the key. A conflict is one key carrying two different value sets")
    L.append("inside a single snapshot — a narrowing artefact where the key used is narrower")
    L.append("than the vendor's declared key, not a between-snapshot change.")
    L.append("")
    L.append("| table | rows before dedupe (A / B) | rows after (A / B) | key conflicts (A / B) | source entries (A / B) |")
    L.append("| --- | --- | --- | --- | --- |")
    for t, e in sorted(report["tables"].items()):
        if "rows_before_dedupe" not in e:
            continue
        a, b = report["snapshot_a"], report["snapshot_b"]
        L.append("| %s | %s / %s | %s / %s | %s / %s | %d / %d |" % (
            t, e["rows_before_dedupe"][a], e["rows_before_dedupe"][b],
            e["rows_after_dedupe"][a], e["rows_after_dedupe"][b],
            e["intra_snapshot_key_conflicts"][a], e["intra_snapshot_key_conflicts"][b],
            len(e["source_entries_a"]), len(e["source_entries_b"])))
    L.append("")
    L.append("`lastupdated` is reported as a watermark count only; it never enters classification.")
    L.append("")
    L.append("## Per-field changed counts")
    L.append("")
    for t, e in sorted(report["tables"].items()):
        pf = e.get("per_field_changed") or {}
        if not pf:
            continue
        nz = {k: v for k, v in pf.items() if v}
        L.append("- **%s**: %s" % (t, ", ".join("%s=%d" % kv for kv in sorted(nz.items())) or "no field changed"))
    L.append("")
    L.append("## Registered reads (Sprint R0-D)")
    L.append("")
    rp = report.get("registered_read_parameters") or {}
    L.append("Parameters were fixed before the pull: " + ", ".join(
        "%s=%s" % (k, v) for k, v in sorted(rp.items())) or "none")
    L.append("")
    for t, e in sorted(report["tables"].items()):
        rd = (e or {}).get("registered_reads")
        if not rd:
            continue
        L.append("### %s" % t)
        L.append("")
        L.append("```")
        L.append(json.dumps(rd, indent=2, sort_keys=True))
        L.append("```")
        L.append("")
    L.append("## Price tables — magnitude of change and classification")
    L.append("")
    for t in PRICE_TABLES:
        e = report["tables"].get(t) or {}
        pa = e.get("price_analysis")
        if not pa:
            L.append("- **%s**: no price analysis (table absent or not keyed by ticker/date)." % t)
            continue
        L.append("### %s" % t)
        L.append("")
        L.append("Tickers compared: %d. Class tally: %s" % (
            pa["tickers_compared"],
            ", ".join("%s=%d" % kv for kv in sorted(pa["class_tally"].items())) or "none"))
        L.append("")
        L.append("| field | changed | abs magnitude | rel magnitude | <=7 | 8-30 | 31-365 | >365 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for f, fd in pa["fields"].items():
            bb = fd["by_age_bucket"]
            L.append("| %s | %d | %s | %s | %d | %d | %d | %d |" % (
                f, fd["changed"], _fmt_dist(fd["abs"]), _fmt_dist(fd["rel"]),
                bb["<=7"]["changed"], bb["8-30"]["changed"],
                bb["31-365"]["changed"], bb[">365"]["changed"]))
        L.append("")
        if pa["revision_tickers"]:
            L.append("REVISION tickers (unadjusted close or volume moved on a settled bar):")
            L.append("")
            L.append("| ticker | fields on settled bars | bars by age bucket |")
            L.append("| --- | --- | --- |")
            for tk, det in sorted(pa["revision_tickers"].items()):
                buckets = "; ".join("%s: %s" % (b, ", ".join("%s=%d" % kv for kv in sorted(v.items())))
                                    for b, v in sorted(det["bars_by_age_bucket"].items()))
                L.append("| %s | %s | %s |" % (tk, ", ".join(det["fields_on_settled_bars"]), buckets))
            L.append("")
        else:
            L.append("REVISION tickers: none.")
            L.append("")
    L.append("## Examples file")
    L.append("")
    L.append("Path (untracked, gitignored): `%s` — %d row(s), capped at %d."
             % (examples_path, examples_rows, EXAMPLE_ROW_CAP))
    L.append("")
    return "\n".join(L) + "\n"


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(root, label_a, label_b, chunk_by_ticker=None, reads=None):
    report, examples = diff_snapshots(root, label_a, label_b,
                                      chunk_by_ticker=chunk_by_ticker,
                                      reads=reads)
    report["manifest_sha256"] = {}
    for lbl in (label_a, label_b):
        mp = os.path.join(snapshot_dir(root, lbl), "manifest.json")
        report["manifest_sha256"][lbl] = sha256_file(mp) if os.path.exists(mp) else "n/a"

    ex_dir = os.path.join(root, "data", "sharadar")
    os.makedirs(ex_dir, exist_ok=True)
    ex_path = os.path.join(ex_dir, "diff_examples_%s_vs_%s.csv" % (label_a, label_b))
    pd.DataFrame(examples, columns=["ticker", "date", "field", "old", "new"]).to_csv(
        ex_path, index=False)
    rel_ex = os.path.relpath(ex_path, root)
    report["examples_file"] = {"path": rel_ex, "rows": len(examples), "cap": EXAMPLE_ROW_CAP}

    rep_dir = os.path.join(root, "reports")
    os.makedirs(rep_dir, exist_ok=True)
    base = "sharadar_stability_%s_vs_%s" % (label_a, label_b)
    if chunk_by_ticker:
        base += "_chunked%d" % chunk_by_ticker
    json_path = os.path.join(rep_dir, base + ".json")
    md_path = os.path.join(rep_dir, base + ".md")
    persisted = copy.deepcopy(report)
    for entry in persisted["tables"].values():
        if isinstance(entry, dict) and entry.get("price_analysis"):
            entry["price_analysis"].pop("classes_by_ticker", None)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(persisted, fh, indent=2, sort_keys=True)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(render_markdown(persisted, rel_ex, len(examples)))
    return report, json_path, md_path, ex_path


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diff two Sharadar snapshots (no network access).")
    ap.add_argument("label_a")
    ap.add_argument("label_b")
    ap.add_argument("--root", default=REPO_ROOT)
    ap.add_argument("--chunk-by-ticker", type=int, default=None,
                    help="compare ticker-keyed tables in N disjoint ticker "
                         "partitions; bounds peak memory on a table too large "
                         "to hold twice")
    ap.add_argument("--onlyb-cutoff-date", default=None,
                    help="A-side data edge: a B-only row dated at or before this "
                         "is a BACKFILL, not a new session")
    ap.add_argument("--revision-cutoff-date", default=None,
                    help="settle edge: as-traded changes dated before this are "
                         "counted separately as late corrections to old bars")
    ap.add_argument("--action-window-start", default=None,
                    help="exclusive lower bound for an explaining corporate action")
    ap.add_argument("--action-window-end", default=None,
                    help="inclusive upper bound for an explaining corporate action")
    args = ap.parse_args(argv)
    reads = {"onlyb_cutoff_date": args.onlyb_cutoff_date,
             "revision_cutoff_date": args.revision_cutoff_date,
             "action_window_start": args.action_window_start,
             "action_window_end": args.action_window_end}
    report, json_path, md_path, ex_path = run(args.root, args.label_a, args.label_b,
                                             chunk_by_ticker=args.chunk_by_ticker,
                                             reads=reads)
    print("wrote %s" % os.path.relpath(json_path, args.root))
    print("wrote %s" % os.path.relpath(md_path, args.root))
    print("examples %s (%d rows, untracked)" % (os.path.relpath(ex_path, args.root),
                                                report["examples_file"]["rows"]))
    for t, e in sorted(report["tables"].items()):
        pa = e.get("price_analysis")
        if pa:
            print("%s: %s" % (t, ", ".join("%s=%d" % kv for kv in sorted(pa["class_tally"].items()))))
    if args.chunk_by_ticker:
        print("chunked by ticker into %d partitions; peak RSS %.3f GiB"
              % (args.chunk_by_ticker, report["peak_rss_gib"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
