"""
================================================================================
Script:       build_panel_r1b.py
Version:      1.0.0
Phase:        Stage R — Cross-Sectional Triage
Purpose:      Build the half-A monthly panel for look T-R1b: the v1 panel under
              the tightened category rule, plus the v2 controls (sector, beta,
              log dollar volume).
Entry Point:  Yes (python tools/build_panel_r1b.py, run detached under nohup)
Dependencies: tools/xs_triage.py, tools/build_panel_r1.py (primitives), numpy,
              pandas, pyarrow
Last Sprint:  Sprint R1b — Harness v2 (2026-10-04)
================================================================================

Passes A to D reuse the v1 builder's primitives under the tightened rule-1
category set. Pass E then builds the daily market proxy as the equal-weight mean
return across the names selected into the universe at the most recent month-end
on or before each day, so it uses only PAST membership. Pass F computes each
name's trailing-252-session beta against that proxy by prefix sums. Pass G
attaches the sector (current-vintage, a stated limitation) and log dollar volume.
The panel is UNTRACKED and lives on the external volume.
================================================================================
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xs_triage as xt
import build_panel_r1 as v1

VERSION = "1.0.0"

ROOT = v1.ROOT
SCRATCH = v1.SCRATCH
LOGDIR = ROOT / "logs" / "r1b"
OUT_PANEL = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"

log = v1.log


def pass_a_tight(windows: dict) -> tuple[pd.DataFrame, dict]:
    log("PASS A: securities master, TIGHTENED rule 1")
    t = pq.read_table(v1.TICKERS,
                      columns=["table", "permaticker", "ticker", "category",
                               "sector"]).to_pandas()
    sep = xt.sep_ticker_map(t)
    log(f"  SEP rows {len(sep)}, ticker 1:1 permaticker asserted")
    tight, cats = xt.admitted_common_stock_tight(sep)
    log(f"  admitted (Domestic Common Stock + Primary Class only): "
        f"{len(tight)} of {len(sep)}")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        log(f"    {k}: {v}")
    excluded = sorted(set(sep["category"].astype("string").fillna("<NA>")) - set(cats))
    log(f"  categories excluded ({len(excluded)}): {excluded}")

    tight["sector"] = (tight["sector"].astype("string")
                       .fillna(xt.NO_SECTOR).replace("", xt.NO_SECTOR))
    sect = tight["sector"].value_counts().to_dict()
    log(f"  sectors: {len(sect)} groups; no-sector group "
        f"{int(sect.get(xt.NO_SECTOR, 0))} names")
    meta = {
        "sep_rows": int(len(sep)),
        "admitted_rows": int(len(tight)),
        "categories_admitted": cats,
        "categories_excluded": [str(x) for x in excluded],
        "sector_counts": {str(k): int(v) for k, v in sect.items()},
        "sector_note": "the tickers master's sector field is CURRENT-VINTAGE per "
                       "D-R0-A; it is not a point-in-time sector assignment",
        "attributes_note": "the master's attributes are current-vintage per D-R0-A",
    }
    # The vendor master stores every column as a string; the bucket files key on
    # an int64 permaticker, so the cast happens once, here.
    tight["permaticker"] = tight["permaticker"].astype(np.int64)
    return tight[["permaticker", "ticker", "category", "sector"]].copy(), meta


# --------------------------------------------------------------------- pass E
def pass_e_market(calendar: np.ndarray, panel: pd.DataFrame,
                  windows: dict) -> tuple[np.ndarray, dict]:
    """
    The daily market proxy: equal-weight mean daily return across the names
    selected into the universe at the most recent month-end on or before the
    day. A day in month m uses the universe chosen at month-end m-1, so the
    proxy never uses membership decided in the future.
    """
    log("PASS E: daily universe-equal-weight market proxy")
    cal_index = pd.DatetimeIndex(pd.to_datetime(calendar))
    n_days = len(cal_index)
    month_of_session = (pd.Series(cal_index).dt.to_period("M")
                        .astype("int64").to_numpy())

    univ = panel.loc[panel["in_universe"].astype(bool), ["permaticker", "month"]].copy()
    # A name selected at month-end m is held through month m+1.
    univ["hold_month"] = univ["month"] + 1
    univ = univ[["permaticker", "hold_month"]].drop_duplicates()
    log(f"  universe selections {len(univ):,} -> holding months "
        f"{univ['hold_month'].min()}..{univ['hold_month'].max()}")

    sum_r = np.zeros(n_days, dtype=np.float64)
    cnt_r = np.zeros(n_days, dtype=np.int64)

    for b in range(v1.N_BUCKETS):
        f = SCRATCH / f"b{b:02d}.parquet"
        if not f.exists():
            continue
        df = pq.read_table(f).to_pandas()
        xt.assert_no_rows_at_or_after(df["date"],
                                      windows["half_a"]["daily_row_exclusive_max"],
                                      f"pass_e bucket {b}")
        df = df.sort_values(["permaticker", "date"], kind="stable")
        gi = cal_index.get_indexer(pd.DatetimeIndex(df["date"]))
        ca = df["closeadj"].to_numpy(dtype=np.float64)
        perma = df["permaticker"].to_numpy()
        r = np.full(len(df), np.nan)
        same = perma[1:] == perma[:-1]
        with np.errstate(invalid="ignore", divide="ignore"):
            step = ca[1:] / ca[:-1] - 1.0
        r[1:] = np.where(same, step, np.nan)

        d = pd.DataFrame({"permaticker": perma, "gi": gi, "r": r})
        d["hold_month"] = month_of_session[gi]
        d = d.merge(univ.assign(_in=True), on=["permaticker", "hold_month"], how="left")
        sel = d["_in"].fillna(False).to_numpy(dtype=bool) & np.isfinite(d["r"].to_numpy())
        if sel.any():
            np.add.at(sum_r, d["gi"].to_numpy()[sel], d["r"].to_numpy()[sel])
            np.add.at(cnt_r, d["gi"].to_numpy()[sel], 1)
        del df, d
        gc.collect()

    with np.errstate(invalid="ignore", divide="ignore"):
        mkt = np.where(cnt_r > 0, sum_r / np.maximum(cnt_r, 1), np.nan)
    ok = np.isfinite(mkt)
    log(f"  market proxy defined on {int(ok.sum())} of {n_days} sessions; "
        f"first {cal_index[np.flatnonzero(ok)[0]].date()}")
    log(f"  names per day: min {int(cnt_r[ok].min())} median "
        f"{float(np.median(cnt_r[ok])):.0f} max {int(cnt_r[ok].max())}")
    log(f"  proxy daily mean {np.nanmean(mkt)*1e4:.3f} bps, "
        f"annualised vol {np.nanstd(mkt)*np.sqrt(252)*100:.2f}%")
    meta = {
        "market_sessions_defined": int(ok.sum()),
        "market_first_session": str(cal_index[np.flatnonzero(ok)[0]].date()),
        "market_names_per_day_min": int(cnt_r[ok].min()),
        "market_names_per_day_median": float(np.median(cnt_r[ok])),
        "market_names_per_day_max": int(cnt_r[ok].max()),
        "market_annualised_vol_pct": float(np.nanstd(mkt) * np.sqrt(252) * 100),
    }
    return mkt, meta


# --------------------------------------------------------------------- pass F
def name_betas(gi: np.ndarray, closeadj: np.ndarray, mkt: np.ndarray,
               me_gi: np.ndarray, me_month: np.ndarray) -> pd.DataFrame:
    """Trailing-252-session beta at each month-end, by prefix sums."""
    n = gi.shape[0]
    if n == 0:
        return pd.DataFrame()
    r = np.full(n, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        r[1:] = closeadj[1:] / closeadj[:-1] - 1.0
    m = mkt[gi]
    ok = np.isfinite(r) & np.isfinite(m)
    r0 = np.where(ok, r, 0.0)
    m0 = np.where(ok, m, 0.0)
    c_r = np.concatenate([[0.0], np.cumsum(r0)])
    c_m = np.concatenate([[0.0], np.cumsum(m0)])
    c_rm = np.concatenate([[0.0], np.cumsum(r0 * m0)])
    c_mm = np.concatenate([[0.0], np.cumsum(m0 * m0)])
    c_n = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])

    keep = me_gi >= gi[0]
    if not keep.any():
        return pd.DataFrame()
    mg, mm_ord = me_gi[keep], me_month[keep]
    hi = np.searchsorted(gi, mg, side="right")
    lo = np.searchsorted(gi, mg - (xt.VOL_WINDOW_SESSIONS - 1), side="left")
    have = hi > 0
    mg, mm_ord, hi, lo = mg[have], mm_ord[have], hi[have], lo[have]
    if mg.size == 0:
        return pd.DataFrame()

    cnt = c_n[hi] - c_n[lo]
    sr = c_r[hi] - c_r[lo]
    sm = c_m[hi] - c_m[lo]
    srm = c_rm[hi] - c_rm[lo]
    smm = c_mm[hi] - c_mm[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = srm / cnt - (sr / cnt) * (sm / cnt)
        var = smm / cnt - (sm / cnt) ** 2
        beta = np.where(var > 0, cov / var, np.nan)
    beta = np.where(cnt >= xt.VOL_MIN_PRESENT, beta, np.nan)
    return pd.DataFrame({"month": mm_ord, "beta": beta, "n_beta_obs": cnt})


def pass_f_beta(calendar: np.ndarray, mkt: np.ndarray, windows: dict) -> pd.DataFrame:
    log("PASS F: trailing-252-session beta against the market proxy")
    cal_index = pd.DatetimeIndex(pd.to_datetime(calendar))
    month_of_session = (pd.Series(cal_index).dt.to_period("M")
                        .astype("int64").to_numpy())
    me_mask = month_of_session != np.concatenate([month_of_session[1:], [-1]])
    me_gi_all = np.flatnonzero(me_mask)
    me_month_all = month_of_session[me_gi_all]

    parts = []
    for b in range(v1.N_BUCKETS):
        f = SCRATCH / f"b{b:02d}.parquet"
        if not f.exists():
            continue
        df = pq.read_table(f).to_pandas()
        xt.assert_no_rows_at_or_after(df["date"],
                                      windows["half_a"]["daily_row_exclusive_max"],
                                      f"pass_f bucket {b}")
        df = df.sort_values(["permaticker", "date"], kind="stable")
        df["gi"] = cal_index.get_indexer(pd.DatetimeIndex(df["date"]))
        rows = []
        for perma, g in df.groupby("permaticker", sort=False):
            fe = name_betas(g["gi"].to_numpy(),
                            g["closeadj"].to_numpy(dtype=np.float64), mkt,
                            me_gi_all, me_month_all)
            if fe.empty:
                continue
            fe["permaticker"] = np.int64(perma)
            rows.append(fe)
        if rows:
            parts.append(pd.concat(rows, ignore_index=True))
        del df, rows
        gc.collect()
        log(f"  bucket {b:02d} done; peakRSS {v1.peak_rss_gib():.2f} GiB")
    out = pd.concat(parts, ignore_index=True)
    log(f"  beta rows {len(out):,}; finite {int(np.isfinite(out['beta']).sum()):,}")
    return out


def main() -> int:
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    windows = json.load(open(ROOT / "config" / "windows.json"))
    log(f"build_panel_r1b v{VERSION}; harness v{xt.VERSION}; vintage {windows['vintage']}")
    log(f"half A: daily rows strictly before "
        f"{windows['half_a']['daily_row_exclusive_max']}; return months through "
        f"{windows['half_a']['return_months_through']}")

    tight, meta_a = pass_a_tight(windows)
    calendar, meta_b = v1.pass_b(tight, windows)
    panel, meta_c = v1.pass_c(calendar, windows)
    panel, meta_d = v1.pass_d(panel, windows)
    mkt, meta_e = pass_e_market(calendar, panel, windows)
    betas = pass_f_beta(calendar, mkt, windows)

    log("PASS G: attach controls")
    panel = panel.merge(betas, on=["permaticker", "month"], how="left")
    panel = panel.merge(tight[["permaticker", "sector"]], on="permaticker", how="left")
    panel["sector"] = panel["sector"].fillna(xt.NO_SECTOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = panel["med_dollar_vol_21"].to_numpy(dtype=np.float64)
        panel["log_dv"] = np.where(dv > 0, np.log(dv), np.nan)

    u = panel.loc[panel["in_universe"].astype(bool)]
    nb = int(np.isfinite(u["beta"].to_numpy()).sum())
    log(f"  in-universe rows {len(u):,}; finite beta {nb:,} "
        f"({nb/len(u)*100:.2f}%); finite log_dv "
        f"{int(np.isfinite(u['log_dv'].to_numpy()).sum()):,}")
    log(f"  in-universe sector groups {u['sector'].nunique()}; no-sector rows "
        f"{int((u['sector'] == xt.NO_SECTOR).sum()):,}")
    log(f"  beta in-universe: mean {np.nanmean(u['beta']):.3f} "
        f"p05 {np.nanpercentile(u['beta'],5):.3f} "
        f"p95 {np.nanpercentile(u['beta'],95):.3f}")

    keep = ["permaticker", "month", "month_ts", "in_universe", "cand",
            "MOM12_1", "REV1", "LOWVOL", "HI52", "MAX",
            "ret_m", "fwd_ret", "fwd_stopped", "closeunadj_t", "n_sessions_21",
            "n_sessions_month", "med_dollar_vol_21", "n_ret_252",
            "beta", "n_beta_obs", "sector", "log_dv"]
    out = panel[keep].copy()
    OUT_PANEL.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PANEL, index=False)

    import hashlib
    h = hashlib.sha256()
    with open(OUT_PANEL, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    sha = h.hexdigest()

    np.save(LOGDIR / "market_proxy_daily.npy", mkt)
    meta = {"version": VERSION, "harness_version": xt.VERSION,
            "panel_path": str(OUT_PANEL.relative_to(ROOT)), "panel_sha256": sha,
            "wall_seconds": round(time.time() - t0, 1),
            "peak_rss_gib": round(v1.peak_rss_gib(), 2),
            "in_universe_finite_beta": nb,
            "in_universe_finite_beta_pct": round(nb / len(u) * 100, 4),
            **meta_a, **meta_b, **meta_c, **meta_d, **meta_e}
    (LOGDIR / "panel_build_meta_v2.json").write_text(json.dumps(meta, indent=2))
    log(f"PANEL {OUT_PANEL.name} rows {len(out):,} sha {sha}")
    log(f"wall {time.time()-t0:.1f}s peakRSS {v1.peak_rss_gib():.2f} GiB")
    shutil.rmtree(SCRATCH, ignore_errors=True)
    log("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
