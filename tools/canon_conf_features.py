"""
================================================================================
Script:       canon_conf_features.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The 25 price-and-volume characteristics of Canon read 3, built for
              every in-universe row of the banked v2 half-A panel over the 168
              formation months 1998-12 to 2012-11, each ranked within its month
              and scaled to (-0.5, 0.5].
Entry Point:  Yes — python tools/canon_r3_features.py build | truncation
Dependencies: numpy, pandas, pyarrow; tools/xs_triage.py, tools/canon_r3_paths.py
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

Every read of bulk_stocks goes through xs_triage.filter_to_half_a, which filters
at read time and then asserts, so no daily row dated on or after the half-A
cutoff can reach a feature. open, high, low and close are never read.

All windows are trailing GLOBAL session windows ending at g, the global session
BEFORE the formation month's month-end session s_t, which is the builder's own
idiom one session earlier. A window with fewer present values than its stated
minimum gives NaN, and NaN survives the rank scaling as NaN.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

FIRST_FORMATION_MONTH = "1998-12"
LAST_FORMATION_MONTH = "2020-11"
N_FORMATION_MONTHS = 264

# name, window in global sessions (0 = not a daily window), minimum present
FEATURES: list[tuple[str, int, int]] = [
    ("REV1", 0, 0), ("MOM3_1", 0, 0), ("MOM6_1", 0, 0), ("MOM12_1", 0, 0),
    ("MOM12_7", 0, 0), ("SEAS12", 0, 0), ("REV5", 0, 0),
    ("VOL21", 21, 15), ("VOL63", 63, 45), ("VOL252", 252, 200),
    ("DOWNVOL63", 63, 45), ("MAX1", 0, 15), ("MIN1", 0, 15),
    ("SKEW63", 63, 45), ("UPFRAC63", 63, 45),
    ("HI52", 252, 200), ("LO52", 252, 200), ("AGEHI", 252, 200),
    ("HI63", 63, 45), ("DVTREND", 252, 200), ("AMIHUD21", 21, 15),
    ("VOLVOL63", 63, 45), ("PRC", 0, 0), ("BETA", 0, 0), ("LOGDV", 0, 0),
]
FEATURE_NAMES = [f[0] for f in FEATURES]
# BETA and LOGDV are taken from the banked panel, so they run through s_t, not g.
PANEL_FEATURES = {"BETA": "beta", "LOGDV": "log_dv"}
# The truncation test covers the 23 features this module computes from the daily
# data; BETA and LOGDV are the banked panel's own columns.
TRUNCATION_FEATURES = [n for n in FEATURE_NAMES if n not in PANEL_FEATURES]


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


# ----------------------------------------------------------------- the daily data
class Daily:
    """Half-A daily closeadj / closeunadj / volume, sorted by (name, session)."""

    def __init__(self, pm, gi, ca, cu, vol, start, end, slot, calendar,
                 month_of_session, rows_read, max_date):
        self.pm, self.gi, self.ca, self.cu, self.vol = pm, gi, ca, cu, vol
        self.start, self.end, self.slot = start, end, slot
        self.calendar = calendar
        self.month_of_session = month_of_session
        self.rows_read = rows_read
        self.max_date = max_date
        # the name's own consecutive-session closeadj return, the v2 builder's r
        r = np.full(ca.shape[0], np.nan)
        for s, e in zip(start, end):
            if e - s > 1:
                with np.errstate(invalid="ignore", divide="ignore"):
                    r[s + 1:e] = ca[s + 1:e] / ca[s:e - 1] - 1.0
        self.r = r

    def names(self):
        return self.slot


def load_daily_half_a(keep_permatickers=None, truncate_at_session=None) -> Daily:
    """
    Stream half-A bulk_stocks through xs_triage.filter_to_half_a.

    `truncate_at_session` is the truncation test's only lever: when given, every
    row on a global session index beyond it is dropped AFTER the calendar is
    built, so the calendar and the month-end grid are the ones the untruncated
    run uses and only the data visible to a feature changes.
    """
    windows = json.loads(P.windows_json().read_text())
    cut = windows["holdout"]["start"]

    tk = pq.read_table(P.bulk_tickers(),
                       columns=["table", "permaticker", "ticker"]).to_pandas()
    sep = xt.sep_ticker_map(tk)
    tmap = dict(zip(sep["ticker"].astype(str), sep["permaticker"].astype(np.int64)))

    dset = ds.dataset(P.bulk_stocks(), format="parquet")
    flt = ds.field("date") < pa.scalar(cut, pa.large_string())
    parts_pm, parts_dt, parts_ca, parts_cu, parts_vol = [], [], [], [], []
    cal_parts = []
    rows_read = 0
    max_date = None
    for batch in dset.scanner(
            columns=["ticker", "date", "closeadj", "closeunadj", "volume"],
            filter=flt, batch_size=2_000_000).to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        for c in ("closeadj", "closeunadj", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        # THE GATE: filter at read time, then assert. Both halves, always.
        df = xt.filter_to_half_a(df, "date", cut, "canon_r3 features")
        if df.empty:
            continue
        rows_read += len(df)
        m = df["date"].max()
        max_date = m if max_date is None or m > max_date else max_date
        cal_parts.append(df["date"].unique())
        pm = df["ticker"].map(tmap)
        ok = np.array(pm.notna().to_numpy(), dtype=bool, copy=True)
        if keep_permatickers is not None:
            ok &= pm.isin(keep_permatickers).to_numpy()
        if not ok.any():
            continue
        parts_pm.append(pm.to_numpy()[ok].astype(np.int64))
        parts_dt.append(df["date"].to_numpy()[ok])
        parts_ca.append(df["closeadj"].to_numpy(dtype=np.float64)[ok])
        parts_cu.append(df["closeunadj"].to_numpy(dtype=np.float64)[ok])
        parts_vol.append(df["volume"].to_numpy(dtype=np.float64)[ok])

    calendar = np.unique(np.concatenate(cal_parts))
    pm = np.concatenate(parts_pm)
    dt = np.concatenate(parts_dt)
    ca = np.concatenate(parts_ca)
    cu = np.concatenate(parts_cu)
    vol = np.concatenate(parts_vol)
    del parts_pm, parts_dt, parts_ca, parts_cu, parts_vol, cal_parts

    gi = np.searchsorted(calendar, dt).astype(np.int64)
    if truncate_at_session is not None:
        k = gi <= int(truncate_at_session)
        pm, gi, ca, cu, vol = pm[k], gi[k], ca[k], cu[k], vol[k]

    order = np.lexsort((gi, pm))
    pm, gi, ca, cu, vol = pm[order], gi[order], ca[order], cu[order], vol[order]
    uniq, start = np.unique(pm, return_index=True)
    end = np.append(start[1:], pm.shape[0])
    slot = {int(k): i for i, k in enumerate(uniq)}

    cd = calendar.astype("datetime64[D]")
    month_of_session = np.array(
        [pd.Period(pd.Timestamp(d), freq="M").ordinal for d in cd], dtype=np.int64)

    return Daily(pm, gi, ca, cu, vol, start, end, slot, calendar,
                 month_of_session, rows_read, max_date)


def month_end_sessions(d: Daily) -> dict[int, int]:
    """month ordinal -> global session index of that month's last session."""
    mo = d.month_of_session
    out: dict[int, int] = {}
    for i, m in enumerate(mo):
        out[int(m)] = i
    return out


# --------------------------------------------------------------- the 25 features
def _sd(x: np.ndarray) -> float:
    """Sample sd, ddof 1, over the finite values — the v2 builder's convention."""
    v = x[np.isfinite(x)]
    if v.size < 2:
        return np.nan
    return float(v.std(ddof=1))


def _skew(x: np.ndarray) -> float:
    """Fisher-Pearson moment coefficient m3 / m2**1.5 (biased), over finite values."""
    v = x[np.isfinite(x)]
    n = v.size
    if n < 3:
        return np.nan
    m = v.mean()
    d = v - m
    m2 = float((d * d).mean())
    if m2 <= 0:
        return np.nan
    m3 = float((d ** 3).mean())
    return m3 / (m2 ** 1.5)


def name_features(d: Daily, pmk: int, anchors: np.ndarray,
                  months: np.ndarray) -> dict[str, np.ndarray]:
    """
    The 23 daily-data features for one permaticker at the given (month, g) pairs.
    `anchors[i]` is g for formation month `months[i]`.
    """
    s = d.slot.get(int(pmk))
    nf = len(months)
    out = {n: np.full(nf, np.nan) for n in TRUNCATION_FEATURES}
    if s is None:
        return out
    a, b = d.start[s], d.end[s]
    gi = d.gi[a:b]
    ca = d.ca[a:b]
    cu = d.cu[a:b]
    vol = d.vol[a:b]
    r = d.r[a:b]
    if gi.size == 0:
        return out
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = cu * vol
    mo = d.month_of_session[gi]
    # last closeadj in each calendar month the name traded: me[m]
    chg = np.flatnonzero(np.diff(mo)) + 1
    last_idx = np.append(chg - 1, mo.size - 1)
    me_months = mo[last_idx]
    me_vals = ca[last_idx]

    def me(m: int) -> float:
        j = np.searchsorted(me_months, m)
        if j < me_months.size and me_months[j] == m:
            return float(me_vals[j])
        return np.nan

    def asof(g: int) -> float:
        p = np.searchsorted(gi, g, side="right") - 1
        return float(ca[p]) if p >= 0 else np.nan

    def asof_cu(g: int) -> float:
        p = np.searchsorted(gi, g, side="right") - 1
        return float(cu[p]) if p >= 0 else np.nan

    def win(g: int, w: int) -> tuple[int, int]:
        lo = np.searchsorted(gi, g - w + 1, side="left")
        hi = np.searchsorted(gi, g, side="right")
        return int(lo), int(hi)

    for i in range(nf):
        g = int(anchors[i])
        t = int(months[i])
        if g < 0:
            continue
        c_g = asof(g)

        # 1-7: month-end ratios and the five-session reversal
        m1, m3, m6, m11, m12 = me(t - 1), me(t - 3), me(t - 6), me(t - 11), me(t - 12)
        with np.errstate(invalid="ignore", divide="ignore"):
            if np.isfinite(c_g) and np.isfinite(m1) and m1 != 0:
                out["REV1"][i] = c_g / m1 - 1.0
            if np.isfinite(m1) and np.isfinite(m3) and m3 != 0:
                out["MOM3_1"][i] = m1 / m3 - 1.0
            if np.isfinite(m1) and np.isfinite(m6) and m6 != 0:
                out["MOM6_1"][i] = m1 / m6 - 1.0
            if np.isfinite(m1) and np.isfinite(m12) and m12 != 0:
                out["MOM12_1"][i] = m1 / m12 - 1.0
            if np.isfinite(m6) and np.isfinite(m12) and m12 != 0:
                out["MOM12_7"][i] = m6 / m12 - 1.0
            if np.isfinite(m11) and np.isfinite(m12) and m12 != 0:
                out["SEAS12"][i] = m11 / m12 - 1.0
            c_g5 = asof(g - 5)
            if np.isfinite(c_g) and np.isfinite(c_g5) and c_g5 != 0:
                out["REV5"][i] = c_g / c_g5 - 1.0

        # 8-11, 14, 15: return-window statistics
        for nm, w, mn in (("VOL21", 21, 15), ("VOL63", 63, 45), ("VOL252", 252, 200)):
            lo, hi = win(g, w)
            seg = r[lo:hi]
            if int(np.isfinite(seg).sum()) >= mn:
                out[nm][i] = _sd(seg)
        lo, hi = win(g, 63)
        seg = r[lo:hi]
        fin = seg[np.isfinite(seg)]
        if fin.size >= 45:
            dn = np.minimum(fin, 0.0)
            out["DOWNVOL63"][i] = float(np.sqrt((dn * dn).mean()))
            out["SKEW63"][i] = _skew(fin)
            out["UPFRAC63"][i] = float((fin > 0).mean())

        # 12, 13: the formation month's own extremes, through g
        sel = (mo == t) & (gi <= g)
        seg = r[sel]
        fin = seg[np.isfinite(seg)]
        if fin.size >= 15:
            out["MAX1"][i] = float(fin.max())
            out["MIN1"][i] = float(fin.min())

        # 16-19: distance from window extremes
        lo, hi = win(g, 252)
        seg = ca[lo:hi]
        okm = np.isfinite(seg)
        if int(okm.sum()) >= 200 and np.isfinite(c_g):
            vals = seg[okm]
            mx, mn_ = float(vals.max()), float(vals.min())
            with np.errstate(invalid="ignore", divide="ignore"):
                if mx > 0:
                    out["HI52"][i] = c_g / mx - 1.0
                if mn_ > 0:
                    out["LO52"][i] = c_g / mn_ - 1.0
            pos = np.flatnonzero(okm)[int(np.argmax(vals))] + lo
            out["AGEHI"][i] = (g - int(gi[pos])) / 252.0
        lo, hi = win(g, 63)
        seg = ca[lo:hi]
        okm = np.isfinite(seg)
        if int(okm.sum()) >= 45 and np.isfinite(c_g):
            mx = float(seg[okm].max())
            if mx > 0:
                out["HI63"][i] = c_g / mx - 1.0

        # 20-22: liquidity
        lo21, hi21 = win(g, 21)
        lo252, hi252 = win(g, 252)
        d21 = dv[lo21:hi21]
        d252 = dv[lo252:hi252]
        f21 = d21[np.isfinite(d21)]
        f252 = d252[np.isfinite(d252)]
        if f21.size >= 15 and f252.size >= 200:
            a1, a2 = float(np.median(f21)), float(np.median(f252))
            if a1 > 0 and a2 > 0:
                out["DVTREND"][i] = float(np.log(a1) - np.log(a2))
        rr = r[lo21:hi21]
        okm = np.isfinite(rr) & np.isfinite(d21) & (d21 > 0)
        if int(okm.sum()) >= 15:
            out["AMIHUD21"][i] = float(np.mean(np.abs(rr[okm]) / d21[okm]))
        d63lo, d63hi = win(g, 63)
        d63 = dv[d63lo:d63hi]
        okm = np.isfinite(d63) & (d63 > 0)
        if int(okm.sum()) >= 45:
            out["VOLVOL63"][i] = _sd(np.log(d63[okm]))

        # 23: the as-traded price level
        out["PRC"][i] = asof_cu(g)
    return out


def rank_scale_within_month(df: pd.DataFrame, cols: list[str],
                            month_col: str = "month") -> pd.DataFrame:
    """
    Average rank / n - 0.5, within each month, over the non-NaN values of each
    column; n is the number of ranked values, so the image is (-0.5, 0.5].
    NaN stays NaN. No raw value survives.
    """
    out = df.copy()
    for c in cols:
        def f(s):
            n = int(s.notna().sum())
            if n == 0:
                return s * np.nan
            return s.rank(method="average") / n - 0.5
        out[c] = out.groupby(month_col, observed=True)[c].transform(f)
    return out


def formation_months() -> tuple[int, int]:
    return (pd.Period(FIRST_FORMATION_MONTH, freq="M").ordinal,
            pd.Period(LAST_FORMATION_MONTH, freq="M").ordinal)


def build_features(panel: pd.DataFrame | None = None,
                   daily: Daily | None = None,
                   months_subset: list[int] | None = None,
                   rank_scale: bool = True) -> tuple[pd.DataFrame, dict]:
    """The feature matrix for every in-universe row of the formation window."""
    t0 = time.time()
    if panel is None:
        panel = pd.read_parquet(P.banked_panel())
    m_lo, m_hi = formation_months()
    u = panel.loc[panel["in_universe"].astype(bool)
                  & (panel["month"] >= m_lo) & (panel["month"] <= m_hi)].copy()
    if months_subset is not None:
        u = u.loc[u["month"].isin(months_subset)]
    keep = set(panel["permaticker"].astype(np.int64).tolist())
    if daily is None:
        daily = load_daily_half_a(keep_permatickers=keep)
    mes = month_end_sessions(daily)

    u = u.sort_values(["permaticker", "month"], kind="stable").reset_index(drop=True)
    u["s_t"] = u["month"].map(mes).astype("float64")
    u["g"] = u["s_t"] - 1.0
    cols = {n: np.full(len(u), np.nan) for n in TRUNCATION_FEATURES}
    pmv = u["permaticker"].to_numpy(dtype=np.int64)
    mv = u["month"].to_numpy(dtype=np.int64)
    gv = u["g"].to_numpy()
    bounds = np.flatnonzero(np.diff(pmv)) + 1
    starts = np.concatenate([[0], bounds])
    ends = np.concatenate([bounds, [len(u)]])
    for a, b in zip(starts, ends):
        res = name_features(daily, int(pmv[a]), gv[a:b], mv[a:b])
        for n in TRUNCATION_FEATURES:
            cols[n][a:b] = res[n]
    for n, v in cols.items():
        u[n] = v
    for feat, src in PANEL_FEATURES.items():
        u[feat] = u[src].to_numpy(dtype=np.float64)

    out = u[["permaticker", "month", "fwd_ret", "fwd_stopped", "beta", "log_dv",
             "sector", "in_universe"] + FEATURE_NAMES].copy()
    raw = out.copy()
    if rank_scale:
        out = rank_scale_within_month(out, FEATURE_NAMES)
    meta = {
        "version": VERSION,
        "rows": int(len(out)),
        "months": int(out["month"].nunique()),
        "first_month": str(pd.Period(ordinal=int(out["month"].min()), freq="M")),
        "last_month": str(pd.Period(ordinal=int(out["month"].max()), freq="M")),
        "daily_rows_read_half_a": int(daily.rows_read),
        "max_daily_date_loaded": str(pd.Timestamp(daily.max_date).date()),
        "global_sessions": int(daily.calendar.size),
        "wall_seconds": round(time.time() - t0, 1),
        "non_nan_share": {n: round(float(raw[n].notna().mean()), 6)
                          for n in FEATURE_NAMES},
    }
    return out, meta


def truncation_check(months: list[str], panel: pd.DataFrame | None = None,
                     daily_full: Daily | None = None) -> dict:
    """
    Section 2's no-lookahead test. For each named formation month, the feature
    value computed from daily data cut off after g must equal the value computed
    from the whole of half A. Raw (unscaled) features are compared, because the
    rank scaling is cross-sectional and would hide a per-name difference.
    """
    if panel is None:
        panel = pd.read_parquet(P.banked_panel())
    keep = set(panel["permaticker"].astype(np.int64).tolist())
    if daily_full is None:
        daily_full = load_daily_half_a(keep_permatickers=keep)
    mes = month_end_sessions(daily_full)
    res = {"months": {}, "features": TRUNCATION_FEATURES}
    worst = 0.0
    for lbl in months:
        t = pd.Period(lbl, freq="M").ordinal
        g = mes[t] - 1
        full, _ = build_features(panel=panel, daily=daily_full,
                                 months_subset=[t], rank_scale=False)
        dtr = load_daily_half_a(keep_permatickers=keep, truncate_at_session=g)
        trunc, _ = build_features(panel=panel, daily=dtr,
                                  months_subset=[t], rank_scale=False)
        del dtr
        full = full.sort_values("permaticker").reset_index(drop=True)
        trunc = trunc.sort_values("permaticker").reset_index(drop=True)
        assert (full["permaticker"].to_numpy() == trunc["permaticker"].to_numpy()).all()
        per = {}
        for n in TRUNCATION_FEATURES:
            a = full[n].to_numpy(dtype=np.float64)
            b = trunc[n].to_numpy(dtype=np.float64)
            same_nan = int((np.isnan(a) != np.isnan(b)).sum())
            both = np.isfinite(a) & np.isfinite(b)
            md = float(np.max(np.abs(a[both] - b[both]))) if both.any() else 0.0
            per[n] = {"nan_pattern_mismatches": same_nan, "max_abs_delta": md}
            worst = max(worst, md)
            if same_nan:
                worst = float("inf")
        res["months"][lbl] = {"g_global_session": int(g), "rows": int(len(full)),
                              "per_feature": per,
                              "identical": all(v["nan_pattern_mismatches"] == 0
                                               and v["max_abs_delta"] == 0.0
                                               for v in per.values())}
        log("truncation %s g=%d rows=%d identical=%s"
            % (lbl, g, len(full), res["months"][lbl]["identical"]))
    res["all_identical"] = all(m["identical"] for m in res["months"].values())
    res["worst_max_abs_delta"] = worst
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = (argv[0] if argv else "build").lower()
    if what == "build":
        feats, meta = build_features()
        P.derived().mkdir(parents=True, exist_ok=True)
        feats.to_parquet(P.feature_file(), index=False)
        meta["feature_file"] = str(P.feature_file())
        meta["feature_sha256"] = sha256_file(P.feature_file())
        (P.derived() / "features_meta.json").write_text(json.dumps(meta, indent=2))
        log("FEATURES %s rows %d sha %s"
            % (P.feature_file().name, meta["rows"], meta["feature_sha256"]))
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    if what == "truncation":
        ms = argv[1:] or ["1999-06", "2004-03", "2008-09", "2012-11"]
        res = truncation_check(ms)
        print(json.dumps(res, indent=2, sort_keys=True))
        return 0 if res["all_identical"] else 1
    raise SystemExit("usage: canon_r3_features.py build | truncation [YYYY-MM ...]")


if __name__ == "__main__":
    sys.exit(main())
