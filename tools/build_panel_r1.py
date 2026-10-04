"""
================================================================================
Script:       build_panel_r1.py
Version:      1.0.0
Phase:        Stage R — Cross-Sectional Triage
Purpose:      Build the half-A monthly point-in-time panel for T-R1 from the
              20261004p snapshot, chunked by permaticker in bounded memory.
Entry Point:  Yes (python tools/build_panel_r1.py, run detached under nohup)
Dependencies: tools/xs_triage.py, numpy, pandas, pyarrow
Last Sprint:  Sprint R1 — Cross-Sectional Triage Harness (2026-10-04)
================================================================================

Three passes. Pass A reads the securities master, restricts it to table == 'SEP',
asserts the ticker -> permaticker map is 1:1 and keeps the common-stock rows.
Pass B streams the daily bulk table filtered to half A, maps every row onto a
permaticker and buckets it by permaticker so that one bucket holds whole names;
the global session calendar is accumulated here. Pass C computes each name's
monthly features against GLOBAL session windows (21 and 252) using a sparse
range-max table and prefix sums, then the cross-sectional top-1000 rule is
applied and the forward return attached.
The panel is UNTRACKED and lives on the external volume; no price leaves it.
================================================================================
"""

from __future__ import annotations

import gc
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xs_triage as xt

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
SNAP = ROOT / "data" / "sharadar" / "snapshot_20261004p"
STOCKS = SNAP / "bulk_stocks.parquet"
TICKERS = SNAP / "bulk_tickers.parquet"
OUT_PANEL = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA.parquet"
SCRATCH = ROOT / "data" / "derived" / "_r1_buckets"
LOGDIR = ROOT / "logs" / "r1"

N_BUCKETS = 32
BATCH_ROWS = 2_000_000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 3)


# --------------------------------------------------------------------- pass A
def pass_a(windows: dict) -> tuple[pd.DataFrame, dict]:
    log("PASS A: securities master")
    t = pq.read_table(TICKERS, columns=["table", "permaticker", "ticker", "category"]).to_pandas()
    sep = xt.sep_ticker_map(t)
    log(f"  SEP rows {len(sep)}, ticker 1:1 permaticker asserted")
    common, cats = xt.admitted_common_stock(sep)
    log(f"  common stock admitted: {len(common)} of {len(sep)}")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        log(f"    {k}: {v}")
    excluded = sorted(set(sep["category"].astype("string").fillna("<NA>")) - set(cats))
    log(f"  categories excluded: {excluded}")
    meta = {
        "sep_rows": int(len(sep)),
        "common_stock_rows": int(len(common)),
        "categories_admitted": cats,
        "categories_excluded": [str(x) for x in excluded],
        "attributes_note": "the master's attributes are current-vintage per D-R0-A",
    }
    return common[["permaticker", "ticker", "category"]].copy(), meta


# --------------------------------------------------------------------- pass B
def pass_b(common: pd.DataFrame, windows: dict) -> tuple[np.ndarray, dict]:
    log("PASS B: stream daily bulk, half A only, bucket by permaticker")
    cutoff = pd.Timestamp(windows["half_a"]["daily_row_exclusive_max"])
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)

    tmap = dict(zip(common["ticker"].astype(str), common["permaticker"].astype(np.int64)))
    dataset = ds.dataset(STOCKS, format="parquet")
    # The vendor's bulk parquet stores every column as a string; ISO dates
    # compare lexicographically exactly as they compare chronologically, so the
    # read-time filter is exact. The assert below is the second half of the guard.
    flt = ds.field("date") < pa.scalar(str(cutoff.date()), pa.large_string())

    writers: dict[int, pq.ParquetWriter] = {}
    cal_parts: list[np.ndarray] = []
    rows_in = rows_kept = 0
    max_date_seen = None
    schema = pa.schema([
        ("permaticker", pa.int64()), ("date", pa.timestamp("ns")),
        ("closeadj", pa.float64()), ("closeunadj", pa.float64()), ("volume", pa.float64()),
    ])

    scanner = dataset.scanner(
        columns=["ticker", "date", "closeadj", "closeunadj", "volume"],
        filter=flt, batch_size=BATCH_ROWS,
    )
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        rows_in += len(df)
        df["date"] = pd.to_datetime(df["date"])
        for c in ("closeadj", "closeunadj", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # GUARD (second half): filtered at read time, now asserted.
        mx = xt.assert_no_rows_at_or_after(df["date"], str(cutoff.date()), "pass_b batch")
        if mx is not None and (max_date_seen is None or mx > max_date_seen):
            max_date_seen = mx
        cal_parts.append(df["date"].unique())
        df["permaticker"] = df["ticker"].map(tmap)
        df = df.loc[df["permaticker"].notna()].copy()
        if df.empty:
            continue
        df["permaticker"] = df["permaticker"].astype(np.int64)
        rows_kept += len(df)
        df = df[["permaticker", "date", "closeadj", "closeunadj", "volume"]]
        df["bucket"] = (df["permaticker"] % N_BUCKETS).astype(np.int16)
        for b, g in df.groupby("bucket", sort=False):
            tbl = pa.Table.from_pandas(g.drop(columns=["bucket"]), schema=schema,
                                       preserve_index=False)
            if b not in writers:
                writers[int(b)] = pq.ParquetWriter(SCRATCH / f"b{int(b):02d}.parquet", schema)
            writers[int(b)].write_table(tbl)
        del df, g, batch
        gc.collect()
        if rows_in % (BATCH_ROWS * 5) == 0:
            log(f"  in {rows_in:,} kept {rows_kept:,} peakRSS {peak_rss_gib():.2f} GiB")

    for w in writers.values():
        w.close()
    calendar = np.unique(np.concatenate(cal_parts))
    calendar = np.sort(calendar.astype("datetime64[ns]"))
    log(f"  rows read {rows_in:,}; mapped to common-stock permatickers {rows_kept:,}")
    log(f"  global sessions {len(calendar)}: {calendar[0]} .. {calendar[-1]}")
    log(f"  max daily date loaded anywhere: {max_date_seen} (cutoff {cutoff.date()}, exclusive)")
    meta = {
        "daily_rows_read_half_a": int(rows_in),
        "daily_rows_mapped_common_stock": int(rows_kept),
        "global_sessions": int(len(calendar)),
        "calendar_first": str(pd.Timestamp(calendar[0]).date()),
        "calendar_last": str(pd.Timestamp(calendar[-1]).date()),
        "max_daily_date_loaded": str(pd.Timestamp(max_date_seen).date()),
    }
    return calendar, meta


# ------------------------------------------------------------ sparse range max
def sparse_max_table(a: np.ndarray) -> list[np.ndarray]:
    n = a.shape[0]
    levels = [a]
    k = 1
    while (1 << k) <= n:
        prev = levels[-1]
        step = 1 << (k - 1)
        levels.append(np.maximum(prev[: n - (1 << k) + 1], prev[step: step + n - (1 << k) + 1]))
        k += 1
    return levels


def range_max(levels: list[np.ndarray], lo: np.ndarray, hi: np.ndarray,
              n: int) -> np.ndarray:
    """
    Inclusive [lo, hi] range max, vectorized. An empty or out-of-range window
    (a name with no session inside it) yields NaN rather than an index error.
    """
    out = np.full(lo.shape, np.nan)
    valid = (lo <= hi) & (lo >= 0) & (hi <= n - 1)
    if not valid.any():
        return out
    lo_v, hi_v = lo[valid], hi[valid]
    k = np.floor(np.log2(hi_v - lo_v + 1)).astype(np.int64)
    res = np.empty(lo_v.shape, dtype=np.float64)
    for kk in np.unique(k):
        m = k == kk
        lvl = levels[int(kk)]
        step = 1 << int(kk)
        res[m] = np.maximum(lvl[lo_v[m]], lvl[hi_v[m] - step + 1])
    out[valid] = res
    return out


# --------------------------------------------------------------------- pass C
def name_features(gi: np.ndarray, closeadj: np.ndarray, closeunadj: np.ndarray,
                  volume: np.ndarray, dates: np.ndarray,
                  me_gi: np.ndarray, me_month: np.ndarray,
                  month_of_session: np.ndarray) -> pd.DataFrame:
    """
    Features for one permaticker at every month-end inside its span.
    All windows are GLOBAL session windows anchored on the month-end session.
    """
    n = gi.shape[0]
    if n == 0:
        return pd.DataFrame()

    # Daily returns on the name's own consecutive sessions.
    r = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = closeadj[1:] / closeadj[:-1] - 1.0
    r_ok = np.isfinite(r)
    r0 = np.where(r_ok, r, 0.0)
    c1 = np.concatenate([[0.0], np.cumsum(r0)])
    c2 = np.concatenate([[0.0], np.cumsum(r0 * r0)])
    cn = np.concatenate([[0], np.cumsum(r_ok.astype(np.int64))])

    dv = closeunadj * volume
    lv = sparse_max_table(np.where(np.isfinite(closeadj), closeadj, -np.inf))

    # Keep only month-ends at or after the name's first session.
    keep = (me_gi >= gi[0]) & (me_gi >= 0)
    if not keep.any():
        return pd.DataFrame()
    mg = me_gi[keep]
    mm = me_month[keep]

    hi21 = np.searchsorted(gi, mg, side="right")
    lo21 = np.searchsorted(gi, mg - (xt.LOOKBACK_SESSIONS - 1), side="left")
    n_sess21 = hi21 - lo21

    # The name's last session at or before the month end.
    last_pos = hi21 - 1
    have = last_pos >= 0
    mg, mm, hi21, lo21, n_sess21, last_pos = (
        mg[have], mm[have], hi21[have], lo21[have], n_sess21[have], last_pos[have]
    )
    if mg.size == 0:
        return pd.DataFrame()

    hi252 = hi21
    lo252 = np.searchsorted(gi, mg - (xt.VOL_WINDOW_SESSIONS - 1), side="left")

    # Median dollar volume over the 21-session window (<= 21 name rows).
    W = xt.LOOKBACK_SESSIONS
    block = np.full((mg.size, W), np.nan)
    for k in range(W):
        pos = lo21 + k
        ok = pos < hi21
        if ok.any():
            block[ok, k] = dv[pos[ok]]
    any_obs = np.isfinite(block).any(axis=1)
    med_dv = np.full(mg.size, np.nan)
    if any_obs.any():
        med_dv[any_obs] = np.nanmedian(block[any_obs], axis=1)

    # Stdev of daily returns over the 252-session window.
    cnt = cn[hi252] - cn[lo252]
    s1 = c1[hi252] - c1[lo252]
    s2 = c2[hi252] - c2[lo252]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = (s2 / cnt - mean * mean) * (cnt / np.maximum(cnt - 1, 1))
        sd = np.sqrt(np.maximum(var, 0.0))
    sd = np.where(cnt >= xt.VOL_MIN_PRESENT, sd, np.nan)

    # 252-session maximum closeadj -> HI52.
    rmax = range_max(lv, lo252, hi252 - 1, n)
    rmax = np.where(np.isfinite(rmax), rmax, np.nan)
    ca_t = closeadj[last_pos]
    with np.errstate(invalid="ignore", divide="ignore"):
        hi52 = np.where(np.isfinite(rmax) & (rmax > 0), ca_t / rmax, np.nan)

    # Per-calendar-month aggregates on the name's own sessions.
    msess = month_of_session[gi]
    dfm = pd.DataFrame({"m": msess, "ca": closeadj, "r": r})
    g = dfm.groupby("m", sort=True)
    last_ca = g["ca"].last()
    max_r = g["r"].max()
    n_in_month = g.size()

    mkey = pd.Index(mm)
    out = pd.DataFrame({
        "permaticker": np.int64(0),
        "month": mm,
        "month_end_gi": mg,
        "closeadj_t": ca_t,
        "closeunadj_t": closeunadj[last_pos],
        "n_sessions_21": n_sess21,
        "med_dollar_vol_21": med_dv,
        "LOWVOL": sd,
        "HI52": hi52,
        "n_ret_252": cnt,
        "MAX": max_r.reindex(mkey).to_numpy(),
        "n_sessions_month": n_in_month.reindex(mkey).fillna(0).to_numpy(),
        "monthly_close": last_ca.reindex(mkey).to_numpy(),
    })
    return out


def pass_c(calendar: np.ndarray, windows: dict) -> tuple[pd.DataFrame, dict]:
    log("PASS C: per-name monthly features, bucket by bucket")
    cal_s = pd.Series(pd.to_datetime(calendar))
    month_of_session = cal_s.dt.to_period("M").astype("int64").to_numpy()
    me_mask = month_of_session != np.concatenate([month_of_session[1:], [-1]])
    me_gi_all = np.flatnonzero(me_mask)
    me_month_all = month_of_session[me_gi_all]
    log(f"  month-ends in half A: {len(me_gi_all)} "
        f"({cal_s.iloc[me_gi_all[0]].date()} .. {cal_s.iloc[me_gi_all[-1]].date()})")

    cal_index = pd.DatetimeIndex(pd.to_datetime(calendar))
    parts: list[pd.DataFrame] = []
    n_names = 0
    for b in range(N_BUCKETS):
        f = SCRATCH / f"b{b:02d}.parquet"
        if not f.exists():
            continue
        df = pq.read_table(f).to_pandas()
        xt.assert_no_rows_at_or_after(df["date"], windows["half_a"]["daily_row_exclusive_max"],
                                      f"pass_c bucket {b}")
        df = df.sort_values(["permaticker", "date"], kind="stable")
        gi_all = cal_index.get_indexer(pd.DatetimeIndex(df["date"]))
        if (gi_all < 0).any():
            raise ValueError("a daily date is absent from the global calendar")
        df["gi"] = gi_all
        rows = []
        for perma, g in df.groupby("permaticker", sort=False):
            fe = name_features(
                g["gi"].to_numpy(), g["closeadj"].to_numpy(dtype=np.float64),
                g["closeunadj"].to_numpy(dtype=np.float64),
                g["volume"].to_numpy(dtype=np.float64),
                g["date"].to_numpy(), me_gi_all, me_month_all, month_of_session,
            )
            if fe.empty:
                continue
            fe["permaticker"] = np.int64(perma)
            rows.append(fe)
            n_names += 1
        if rows:
            parts.append(pd.concat(rows, ignore_index=True))
        del df, rows
        gc.collect()
        log(f"  bucket {b:02d} done; names so far {n_names}; peakRSS {peak_rss_gib():.2f} GiB")

    panel = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()
    log(f"  raw monthly rows {len(panel):,} over {n_names} names")
    return panel, {"names_with_features": int(n_names), "month_ends": int(len(me_gi_all))}


# --------------------------------------------------------------------- pass D
def pass_d(panel: pd.DataFrame, windows: dict) -> tuple[pd.DataFrame, dict]:
    log("PASS D: monthly returns, universe rules, forward return")
    panel = panel.sort_values(["permaticker", "month"], kind="stable").reset_index(drop=True)

    # Monthly month-end close series per name, reindexed onto a dense month grid,
    # so a name that stops trading gets a zero month after its last close.
    mmin, mmax = int(panel["month"].min()), int(panel["month"].max())
    log(f"  month index span {mmin}..{mmax}")

    # ret_m on the dense grid: me[m]/me[m-1]-1 if both present;
    # 0.0 if m absent but m-1 present (stopped trading -> then zero);
    # NaN if m-1 absent.
    pv = panel.pivot_table(index="month", columns="permaticker", values="monthly_close",
                           observed=True)
    pv = pv.reindex(range(mmin, mmax + 1))
    prev = pv.shift(1)
    ret = pv / prev - 1.0
    stopped = pv.isna() & prev.notna()
    ret = ret.where(~stopped, 0.0)

    # "Stopped trading in month m" is the vendor-delisting event: the name had a
    # month-end close in m-1 and none in m. Carried forward one month so a
    # formation month t can be asked whether the name it held stopped in t+1.
    stop_fwd = stopped.shift(-1)
    stop_long = stop_fwd.stack().rename("fwd_stopped").reset_index()
    stop_long.columns = ["month", "permaticker", "fwd_stopped"]

    ret_long = ret.stack().rename("ret_m").reset_index()
    ret_long.columns = ["month", "permaticker", "ret_m"]
    panel = panel.merge(ret_long, on=["month", "permaticker"], how="left")

    # REV1 is the return over month t.
    panel["REV1"] = panel["ret_m"]

    # MOM12_1 = me[t-1]/me[t-12] - 1, from the dense month-end close grid.
    mom = pv.shift(1) / pv.shift(12) - 1.0
    mom_long = mom.stack().rename("MOM12_1").reset_index()
    mom_long.columns = ["month", "permaticker", "MOM12_1"]
    panel = panel.merge(mom_long, on=["month", "permaticker"], how="left")

    # Forward return: the month AFTER formation.
    fwd = ret.shift(-1)
    fwd_long = fwd.stack().rename("fwd_ret").reset_index()
    fwd_long.columns = ["month", "permaticker", "fwd_ret"]
    panel = panel.merge(fwd_long, on=["month", "permaticker"], how="left")
    panel = panel.merge(stop_long, on=["month", "permaticker"], how="left")
    panel["fwd_stopped"] = panel["fwd_stopped"].fillna(False).astype(bool)

    # Universe rules 2, 3 and 4, in the registered order; rule 1 (common stock
    # only) was applied at the securities master in pass A.
    r2 = panel["closeunadj_t"].to_numpy() >= xt.MIN_PRICE
    r3 = panel["n_sessions_21"].to_numpy() >= xt.MIN_SESSIONS_TRADED
    log(f"  rule 2 (price>=5) keeps {int(r2.sum()):,} of {len(panel):,}")
    log(f"  rules 2+3 keep {int((r2 & r3).sum()):,}")
    cand, in_universe = xt.apply_universe_rules(panel)
    panel["cand"] = cand.to_numpy()
    panel["in_universe"] = in_universe.to_numpy()
    log(f"  candidates with a dollar-volume reading {int(panel['cand'].sum()):,}")

    # Half-A return-month ceiling: formation month t must have its forward
    # return inside half A, i.e. t+1 <= 2012-12.
    last_return_month = pd.Period(windows["half_a"]["return_months_through"], freq="M")
    last_formation = int(last_return_month.ordinal) - 1
    before = int(panel["in_universe"].sum())
    panel.loc[panel["month"] > last_formation, "in_universe"] = False
    log(f"  formation months capped at ordinal {last_formation} "
        f"({pd.Period(ordinal=last_formation, freq='M')}); "
        f"in-universe rows {before:,} -> {int(panel['in_universe'].sum()):,}")

    u = panel.loc[panel["in_universe"]]
    sizes = u.groupby("month", observed=True).size()
    # In-universe delistings: held at t and stopped trading during t+1, so the
    # held month earned its return to the last close and zero thereafter. This
    # is the scale of the missing-delisting-return bias; the vendor has none.
    deli = int(u["fwd_stopped"].sum())
    deli_names = int(u.loc[u["fwd_stopped"], "permaticker"].nunique())
    exact_zero_fwd = int(((u["fwd_ret"] == 0.0) & ~u["fwd_stopped"]).sum())

    panel["month_ts"] = panel["month"].map(
        lambda o: pd.Period(ordinal=int(o), freq="M").to_timestamp(how="end").normalize())

    meta = {
        "panel_rows": int(len(panel)),
        "in_universe_rows": int(panel["in_universe"].sum()),
        "universe_months": int(sizes.size),
        "universe_first_month": str(pd.Period(ordinal=int(sizes.index.min()), freq="M")),
        "universe_last_month": str(pd.Period(ordinal=int(sizes.index.max()), freq="M")),
        "universe_size_min": int(sizes.min()),
        "universe_size_median": float(sizes.median()),
        "universe_size_max": int(sizes.max()),
        "in_universe_delistings": deli,
        "in_universe_delisted_names": deli_names,
        "exact_zero_fwd_not_delisting": exact_zero_fwd,
    }
    log(f"  universe months {meta['universe_months']} "
        f"({meta['universe_first_month']}..{meta['universe_last_month']}), "
        f"size min/med/max {meta['universe_size_min']}/"
        f"{meta['universe_size_median']}/{meta['universe_size_max']}")
    log(f"  in-universe delistings (held at t, stopped trading in t+1) {deli} "
        f"across {deli_names} distinct names; "
        f"exact-zero forward returns that are NOT delistings {exact_zero_fwd}")
    return panel, meta


def main() -> int:
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    windows = json.load(open(ROOT / "config" / "windows.json"))
    log(f"build_panel_r1 v{VERSION}; harness v{xt.VERSION}; vintage {windows['vintage']}")
    log(f"half A: daily rows strictly before {windows['half_a']['daily_row_exclusive_max']}; "
        f"return months through {windows['half_a']['return_months_through']}")

    common, meta_a = pass_a(windows)
    calendar, meta_b = pass_b(common, windows)
    panel, meta_c = pass_c(calendar, windows)
    panel, meta_d = pass_d(panel, windows)

    keep = ["permaticker", "month", "month_ts", "in_universe", "cand",
            "MOM12_1", "REV1", "LOWVOL", "HI52", "MAX",
            "ret_m", "fwd_ret", "fwd_stopped", "closeunadj_t", "n_sessions_21",
            "n_sessions_month", "med_dollar_vol_21", "n_ret_252"]
    out = panel[keep].copy()
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PANEL, index=False)

    import hashlib
    h = hashlib.sha256()
    with open(OUT_PANEL, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    sha = h.hexdigest()

    wall = time.time() - t0
    meta = {"version": VERSION, "harness_version": xt.VERSION,
            "panel_path": str(OUT_PANEL.relative_to(ROOT)), "panel_sha256": sha,
            "wall_seconds": round(wall, 1), "peak_rss_gib": round(peak_rss_gib(), 2),
            **meta_a, **meta_b, **meta_c, **meta_d}
    (LOGDIR / "panel_build_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"PANEL {OUT_PANEL.name} rows {len(out):,} sha {sha}")
    log(f"wall {wall:.1f}s peakRSS {peak_rss_gib():.2f} GiB")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
