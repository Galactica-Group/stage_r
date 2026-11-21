"""
Synthetic fixtures for the Canon read 3 tests. Not a test module.

Every canon_r3 test runs on a synthetic snapshot and a synthetic panel written
into a temporary directory, with canon_r3_paths redirected there, so no test can
reach the banked panel, the real feature file or the vendor snapshot. The
synthetic bulk_stocks deliberately carries rows PAST the half-A cutoff so the
window guard has something to refuse.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import canon_r3_paths as P  # noqa: E402

N_NAMES = 40
SESSIONS = 1250                      # 2000-01-03 onward, business days
CUTOFF = "2004-01-01"                # rows on or after this must never load
SECTORS = ["Alpha", "Beta", "Gamma", "Delta"]


def _str_table(df: pd.DataFrame) -> pa.Table:
    """The vendor stores every column as a string; so does the synthetic one."""
    return pa.table({c: pa.array([None if pd.isna(v) else str(v) for v in df[c]],
                                 type=pa.large_string()) for c in df.columns})


def write_snapshot(tmp: Path, rng: np.random.Generator) -> dict:
    snap = tmp / "snapshot"
    snap.mkdir(parents=True, exist_ok=True)
    cal = pd.bdate_range("2000-01-03", periods=SESSIONS)
    names = ["SYN%03d" % i for i in range(N_NAMES)]
    perm = {n: 900000 + i for i, n in enumerate(names)}

    rows = []
    for n in names:
        steps = rng.normal(0.0004, 0.02, SESSIONS)
        ca = 20.0 * np.exp(np.cumsum(steps))
        vol = rng.lognormal(12.0, 0.6, SESSIONS)
        rows.append(pd.DataFrame({
            "ticker": n, "date": cal.strftime("%Y-%m-%d"),
            "open": ca, "high": ca, "low": ca, "close": ca,
            "volume": np.round(vol, 0), "closeadj": ca, "closeunadj": ca,
            "lastupdated": "2026-10-04"}))
    stocks = pd.concat(rows, ignore_index=True)
    pq.write_table(_str_table(stocks), snap / "bulk_stocks.parquet")

    tick = pd.DataFrame({
        "table": "SEP",
        "permaticker": [perm[n] for n in names],
        "ticker": names,
        "name": names,
        "exchange": "SYN",
        "isdelisted": "N",
        "category": "Domestic Common Stock",
        "sector": [SECTORS[i % len(SECTORS)] for i in range(len(names))],
    })
    pq.write_table(_str_table(tick), snap / "bulk_tickers.parquet")
    return {"snapshot": snap, "calendar": cal, "names": names, "perm": perm}


def write_panel(tmp: Path, meta: dict, rng: np.random.Generator,
                n_months: int = 8) -> dict:
    """
    A panel over the last `n_months` complete months strictly before the cutoff,
    every name in universe every month.
    """
    cal = meta["calendar"]
    cal = cal[cal < pd.Timestamp(CUTOFF)]
    months = sorted({pd.Period(d, freq="M") for d in cal})
    # drop the first (partial lookback) and keep the last n_months
    use = months[-n_months:]
    recs = []
    for m in use:
        for i, n in enumerate(meta["names"]):
            recs.append({
                "permaticker": meta["perm"][n], "month": m.ordinal,
                "month_ts": m.to_timestamp(), "in_universe": True, "cand": True,
                "MOM12_1": rng.normal(), "REV1": rng.normal(),
                "LOWVOL": abs(rng.normal()) + 0.01, "HI52": rng.normal(),
                "MAX": rng.normal(),
                "ret_m": rng.normal(0, 0.08), "fwd_ret": rng.normal(0, 0.08),
                "fwd_stopped": False, "closeunadj_t": 20.0 + i,
                "n_sessions_21": 21, "n_sessions_month": 21.0,
                "med_dollar_vol_21": 1e7 * (1 + i), "n_ret_252": 252,
                "beta": 1.0 + rng.normal(0, 0.3), "n_beta_obs": 252,
                "sector": SECTORS[i % len(SECTORS)], "log_dv": np.log(1e7 * (1 + i)),
            })
    panel = pd.DataFrame(recs)
    path = tmp / "panel_synth.parquet"
    panel.to_parquet(path, index=False)
    return {"panel_path": path, "panel": panel,
            "first_month": str(use[0]), "last_month": str(use[-1]),
            "months": [str(x) for x in use]}


def install(tmp: Path, seed: int = 11, n_months: int = 8) -> dict:
    """Write the synthetic world and point every canon_r3 path at it."""
    rng = np.random.default_rng(seed)
    meta = write_snapshot(tmp, rng)
    pan = write_panel(tmp, meta, rng, n_months=n_months)
    wj = tmp / "windows.json"
    wj.write_text(json.dumps({
        "vintage": "synthetic",
        "half_a": {"label": "half_A", "return_months_through": "2003-12",
                   "daily_row_exclusive_max": CUTOFF},
        "half_b": {"status": "UNOPENED"}, "holdout": {"status": "SEALED"}}))
    P.set_roots(repo=ROOT, snapshot=meta["snapshot"], banked_panel=pan["panel_path"],
                derived=tmp / "derived", scratch=tmp / "scratch", windows=wj)
    (tmp / "derived").mkdir(exist_ok=True)
    (tmp / "scratch").mkdir(exist_ok=True)
    out = {**meta, **pan, "windows": wj, "cutoff": CUTOFF, "tmp": tmp}
    return out


def snapshot_state() -> dict:
    import copy
    return copy.deepcopy(P._STATE)


def restore_state(st: dict) -> None:
    P._STATE.clear()
    P._STATE.update(st)
