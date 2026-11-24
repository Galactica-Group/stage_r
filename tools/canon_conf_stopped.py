"""
================================================================================
Script:       canon_conf_stopped.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The stopped-stock exposure, reported and never voted: the book's
              own weight in names that stop trading (S1), an adverse bound on
              the net series if every such long were written down (S2), and the
              action-table counts around a name's last price date (S3).
Entry Point:  Yes — python tools/canon_conf_stopped.py s3
Dependencies: numpy, pandas, pyarrow; tools/xs_triage.py, tools/canon_conf_paths.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
================================================================================

Key 3 was struck by ruling (CONF-1b): the builder calls a stock stopped when it
has no observation in the next month, so there is no later trade to recover, and
a stopped row earns exactly zero under the registered scoring. What remains is
worth measuring, so it is measured and reported as a bound.

The harness does not return per-name weights, so they are recomputed here with
the harness's OWN fm_slope_weights on the same inputs, in the same order, and a
test holds that dotting them with the realised returns reproduces the harness's
gross series exactly. Nothing in this module feeds a label, a prediction, a key
or the verdict.
================================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

L_VALUES = (0.30, 1.00)
FOR_CAUSE_ACTIONS = ("bankruptcyliquidation", "regulatorydelisting",
                     "voluntarydelisting")
TAKEOVER_PREFIXES = ("acquisition", "merger")
ACTION_WINDOW_DAYS = 45


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------- the book's own weights
def month_weights(world: pd.DataFrame, pred: np.ndarray, scored,
                  fwd_col: str = "fwd_ret", month_col: str = "month",
                  key_col: str = "permaticker"):
    """
    Replay build_series_v2's month loop with the harness's own helpers and yield
    the weight vector it formed. Same inputs, same order, same skips.
    """
    sm = set(int(x) for x in scored)
    w = world.copy()
    w["_pred"] = pred
    need = [key_col, "_pred", fwd_col, "beta", "log_dv", "sector", "fwd_stopped"]
    sub = w.loc[w["in_universe"].astype(bool), [month_col] + need]
    sub = sub.loc[sub[month_col].isin(sm)]
    for month, g in sub.groupby(month_col, sort=True, observed=True):
        fwd = g[fwd_col].to_numpy(dtype=np.float64)
        beta = g["beta"].to_numpy(dtype=np.float64)
        ldv = g["log_dv"].to_numpy(dtype=np.float64)
        sig = g["_pred"].to_numpy(dtype=np.float64)
        keys = g[key_col].to_numpy()
        sect = g["sector"].to_numpy().astype(str)
        stopped = g["fwd_stopped"].to_numpy().astype(bool)
        ok = (np.isfinite(fwd) & np.isfinite(beta) & np.isfinite(ldv)
              & np.isfinite(sig))
        if ok.sum() < 50:
            continue
        fwd, beta, ldv, sig = fwd[ok], beta[ok], ldv[ok], sig[ok]
        keys, sect, stopped = keys[ok], sect[ok], stopped[ok]
        z = xt.rank_to_normal_z(sig, "HIGH")
        X, _, z_idx = xt.build_design(z, sect, beta, ldv)
        wv = xt.fm_slope_weights(X, z_idx)
        if wv is None:
            continue
        w_ret = xt.scale_to_gross_two(wv)
        yield int(month), keys, w_ret, fwd, stopped


def verify_gross(world: pd.DataFrame, pred: np.ndarray, scored,
                 harness_gross: np.ndarray, harness_months: np.ndarray,
                 fwd_col: str = "fwd_ret") -> dict:
    """The recomputed weights must reproduce the harness's gross series exactly."""
    months, gross = [], []
    for m, _k, w_ret, fwd, _s in month_weights(world, pred, scored, fwd_col=fwd_col):
        months.append(m)
        gross.append(float(np.dot(w_ret, fwd)))
    months = np.asarray(months, dtype=np.int64)
    gross = np.asarray(gross, dtype=np.float64)
    same_months = bool(months.size == harness_months.size
                       and (months == np.asarray(harness_months)).all())
    exact = bool(same_months
                 and gross.tobytes() == np.asarray(harness_gross,
                                                   dtype=np.float64).tobytes())
    return {"months_match": same_months, "bit_identical": exact,
            "n": int(months.size),
            "max_abs_delta": (float(np.max(np.abs(gross - np.asarray(harness_gross))))
                              if same_months and months.size else float("nan"))}


# ------------------------------------------------------------------- S1 and S2
def s1(world: pd.DataFrame, pred: np.ndarray, scored,
       fwd_col: str = "fwd_ret") -> pd.DataFrame:
    """Per scored month: the book's summed positive and negative weight in
    names whose fwd_stopped is true, and how many such positions it holds."""
    rows = []
    for m, _k, w_ret, _fwd, stopped in month_weights(world, pred, scored,
                                                     fwd_col=fwd_col):
        ws = w_ret[stopped]
        rows.append({"month": m,
                     "stopped_positions": int((stopped & (w_ret != 0.0)).sum()),
                     "stopped_weight_pos": float(ws[ws > 0].sum()) if ws.size else 0.0,
                     "stopped_weight_neg": float(ws[ws < 0].sum()) if ws.size else 0.0})
    return pd.DataFrame(rows)


def s2(net_base: np.ndarray, months: np.ndarray, s1_table: pd.DataFrame,
       L_values=L_VALUES) -> dict:
    """
    The adverse bound: the net BASE series less L times that month's summed
    POSITIVE weight in stopped names. Shorts gain nothing, so the negative
    weight is not credited.
    """
    pos = dict(zip(s1_table["month"], s1_table["stopped_weight_pos"]))
    base = np.asarray(net_base, dtype=np.float64)
    out = {}
    for L in L_values:
        adj = np.array([base[i] - float(L) * pos.get(int(m), 0.0)
                        for i, m in enumerate(months)], dtype=np.float64)
        mu = float(adj.mean())
        sd = float(adj.std(ddof=1)) if adj.size > 1 else float("nan")
        out["L_%.2f" % L] = {
            "L": float(L), "mean": mu, "mean_bps": mu * 1e4, "sd": sd,
            "sharpe": (mu / sd * np.sqrt(12.0)) if sd and sd > 0 else float("nan"),
            "total_charge_bps": float(
                sum(float(L) * pos.get(int(m), 0.0) for m in months)) * 1e4,
        }
    return out


def s1_summary(s1_table: pd.DataFrame) -> dict:
    if s1_table.empty:
        return {"months": 0}
    p = s1_table["stopped_weight_pos"].to_numpy(dtype=np.float64)
    n = s1_table["stopped_weight_neg"].to_numpy(dtype=np.float64)
    c = s1_table["stopped_positions"].to_numpy(dtype=np.int64)
    return {
        "months": int(len(s1_table)),
        "months_with_a_stopped_position": int((c > 0).sum()),
        "stopped_positions_total": int(c.sum()),
        "stopped_positions_max_in_a_month": int(c.max()),
        "weight_pos_mean": float(p.mean()), "weight_pos_max": float(p.max()),
        "weight_pos_sum": float(p.sum()),
        "weight_neg_mean": float(n.mean()), "weight_neg_min": float(n.min()),
        "weight_neg_sum": float(n.sum()),
    }


# -------------------------------------------------------------------------- S3
def s3_counts(panel: pd.DataFrame | None = None) -> dict:
    """
    Counts only, on the scored months. Two cohorts: rows the builder calls
    stopped, and rows whose name's LAST PRICE DATE falls inside the holding
    month t+1 — the stock that ends mid-holding, which fwd_stopped does not see.
    """
    import canon_conf_pipeline as PL
    if panel is None:
        panel = pd.read_parquet(P.panel_file())
    scored = set(PL.CONF_SCORED)
    u = panel.loc[panel["in_universe"].astype(bool)
                  & panel["month"].isin(scored)].copy()

    tk = pq.read_table(P.bulk_tickers(),
                       columns=["table", "permaticker", "ticker"]).to_pandas()
    sep = xt.sep_ticker_map(tk)
    pm2t = dict(zip(sep["permaticker"].astype(np.int64), sep["ticker"].astype(str)))

    cut = json.loads(P.windows_json().read_text())["holdout"]["start"]
    a = pq.read_table(P.bulk_actions(),
                      columns=["date", "action", "ticker"]).to_pandas()
    a["action"] = a["action"].astype(str)
    a["ticker"] = a["ticker"].astype(str)
    a["date"] = pd.to_datetime(a["date"], errors="coerce")
    a = a.loc[a["date"].notna()]
    a = xt.filter_to_half_a(a, "date", cut, "canon_conf stopped S3")
    acts: dict[str, list] = {}
    for t, ac, dt in a[["ticker", "action", "date"]].itertuples(index=False):
        acts.setdefault(t, []).append((ac, dt))

    lastp = _last_price_dates()

    def classify(k: int) -> tuple[bool, bool]:
        t = pm2t.get(int(k))
        lp = lastp.get(int(k))
        if t is None or lp is None:
            return False, False
        lo = pd.Timestamp(lp) - pd.Timedelta(days=ACTION_WINDOW_DAYS)
        hi = pd.Timestamp(lp) + pd.Timedelta(days=ACTION_WINDOW_DAYS)
        fc = tk_ = False
        for ac, dt in acts.get(t, ()):
            if lo <= dt <= hi:
                if ac in FOR_CAUSE_ACTIONS:
                    fc = True
                if ac.startswith(TAKEOVER_PREFIXES):
                    tk_ = True
        return fc, tk_

    res = {"scored_months": len(scored),
           "in_universe_scored_rows": int(len(u)),
           "action_window_days_either_side": ACTION_WINDOW_DAYS,
           "for_cause_types": list(FOR_CAUSE_ACTIONS),
           "takeover_prefixes": list(TAKEOVER_PREFIXES)}

    st = u.loc[u["fwd_stopped"].astype(bool)]
    fc = tko = 0
    for k in st["permaticker"]:
        f, t = classify(int(k))
        fc += int(f); tko += int(t)
    res["stopped"] = {"rows": int(len(st)), "for_cause": fc, "takeover": tko,
                      "share_of_scored_rows": round(len(st) / max(len(u), 1), 6)}

    # the stock that ends INSIDE the holding month t+1
    ends = []
    for k, t in zip(u["permaticker"], u["month"]):
        lp = lastp.get(int(k))
        if lp is None:
            continue
        if pd.Period(pd.Timestamp(lp), freq="M").ordinal == int(t) + 1:
            ends.append(int(k))
    fc2 = tko2 = 0
    for k in ends:
        f, t = classify(k)
        fc2 += int(f); tko2 += int(t)
    res["ends_inside_the_holding_month"] = {
        "rows": len(ends), "for_cause": fc2, "takeover": tko2,
        "share_of_scored_rows": round(len(ends) / max(len(u), 1), 6)}
    return res


def _last_price_dates() -> dict:
    """Per permaticker, its last observed price date before the cutoff."""
    import pyarrow as pa
    import pyarrow.dataset as ds
    cut = json.loads(P.windows_json().read_text())["holdout"]["start"]
    tk = pq.read_table(P.bulk_tickers(),
                       columns=["table", "permaticker", "ticker"]).to_pandas()
    sep = xt.sep_ticker_map(tk)
    smap = dict(zip(sep["ticker"].astype(str), sep["permaticker"].astype(np.int64)))
    dset = ds.dataset(P.bulk_stocks(), format="parquet")
    flt = ds.field("date") < pa.scalar(cut, pa.large_string())
    best: dict[int, pd.Timestamp] = {}
    for batch in dset.scanner(columns=["ticker", "date"], filter=flt,
                              batch_size=2_000_000).to_batches():
        if batch.num_rows == 0:
            continue
        df = batch.to_pandas()
        df["date"] = pd.to_datetime(df["date"])
        df = xt.filter_to_half_a(df, "date", cut, "canon_conf last price")
        if df.empty:
            continue
        pm = df["ticker"].map(smap)
        keep = np.array(pm.notna().to_numpy(), dtype=bool, copy=True)
        if not keep.any():
            continue
        g = pd.DataFrame({"pm": pm.to_numpy()[keep].astype(np.int64),
                          "d": df["date"].to_numpy()[keep]}
                         ).groupby("pm")["d"].max()
        for k, v in g.items():
            k = int(k)
            if k not in best or v > best[k]:
                best[k] = v
    return best


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = (argv[0] if argv else "s3").lower()
    if what == "s3":
        r = s3_counts()
        P.derived().mkdir(parents=True, exist_ok=True)
        (P.derived() / "s3_counts.json").write_text(json.dumps(r, indent=2))
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0
    raise SystemExit("usage: canon_conf_stopped.py s3")


if __name__ == "__main__":
    sys.exit(main())
