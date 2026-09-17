"""
================================================================================
Script:       sharadar_snapshot_diff.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Diff two Sharadar snapshots and classify per-ticker price changes
              as ADJUSTMENT, REVISION or TAIL; emit statistics-only reports.
Entry Point:  Yes — python tools/sharadar_snapshot_diff.py <label_a> <label_b>
Dependencies: stdlib, pandas, pyarrow. Makes no network request and imports
              nothing from any other repository.
Last Sprint:  Sprint R0-B-0 — Sharadar Snapshot Diff Tool (2026-09-17)
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
import json
import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRICE_TABLES = ("stocks", "funds")
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
    }, ab, bb


def compare_rowsets(a, b, cols):
    """Whole-row set comparison for a table with no stated key."""
    def sig(df):
        s = None
        for col in cols:
            part = _norm(df[col])
            s = part if s is None else s.str.cat(part, sep="\x01")
        return set(s) if s is not None else set()
    sa, sb = sig(a), sig(b)
    return {
        "rows_only_in_a": len(sa - sb),
        "rows_only_in_b": len(sb - sa),
        "rows_in_both": len(sa & sb),
        "rows_in_both_differing_vendor_fields": None,
        "rows_in_both_differing_any_shared_field": None,
        "watermark_only_changes": None,
        "per_field_changed": {},
    }


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


def price_analysis(ab, bb, key, examples):
    """Per-field magnitudes by age bucket and a per-ticker class for each ticker."""
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


def diff_snapshots(root, label_a, label_b, examples_cap=EXAMPLE_ROW_CAP):
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

    ta, tb = tables_of(man_a), tables_of(man_b)
    both_tables = sorted(set(ta) & set(tb))
    unmapped = sorted({e["name"] for m in (man_a, man_b) for e in m
                       if e.get("parquet_file") and not is_error_payload(e)
                       and entry_table(e["name"]) is None})

    report = {
        "tool": "sharadar_snapshot_diff.py",
        "version": "1.0.0",
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
        "tables": {},
    }
    examples = []
    for table in both_tables:
        a, prov_a, skip_a = load_table(dir_a, man_a, table)
        b, prov_b, skip_b = load_table(dir_b, man_b, table)
        if a is None or b is None:
            report["tables"][table] = {"status": "no data on one side"}
            continue
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
            counts, ab, bb = compare_keyed(a2, b2, key, vendor_cols, derived_cols)
            entry.update(counts)
            if table in PRICE_TABLES and "date" in shared and "ticker" in shared:
                entry["price_analysis"] = price_analysis(
                    ab, bb, key, examples if len(examples) < examples_cap else [])
        else:
            entry.update(compare_rowsets(a, b, vendor_cols))
        report["tables"][table] = entry
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


def run(root, label_a, label_b):
    report, examples = diff_snapshots(root, label_a, label_b)
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
    args = ap.parse_args(argv)
    report, json_path, md_path, ex_path = run(args.root, args.label_a, args.label_b)
    print("wrote %s" % os.path.relpath(json_path, args.root))
    print("wrote %s" % os.path.relpath(md_path, args.root))
    print("examples %s (%d rows, untracked)" % (os.path.relpath(ex_path, args.root),
                                                report["examples_file"]["rows"]))
    for t, e in sorted(report["tables"].items()):
        pa = e.get("price_analysis")
        if pa:
            print("%s: %s" % (t, ", ".join("%s=%d" % kv for kv in sorted(pa["class_tally"].items()))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
