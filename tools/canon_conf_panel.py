"""
================================================================================
Script:       canon_conf_panel.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      Build the v2 panel through formation month 2020-11 by driving the
              repo's own builder passes with the confirmation's window, and the
              tripwire that the banked half-A panel is reproduced exactly inside it.
Entry Point:  Yes — python tools/canon_conf_panel.py build | t1
Dependencies: numpy, pandas, pyarrow; tools/build_panel_r1.py,
              tools/build_panel_r1b.py, tools/xs_triage.py, tools/canon_conf_paths.py
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

Every builder pass takes its window as a `windows` ARGUMENT, so no copy of the
builder is needed: the passes are called with a window whose exclusive maximum
is the holdout's start, read from config/windows.json. The cutoff is applied as
an Arrow pushdown filter at read time and asserted after every batch by
xs_triage's own guard, which is the builder's own two-halved discipline.

The banked half-A panel is read for the tripwire only and is never rewritten.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

LAST_FORMATION_MONTH = "2020-11"


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def conf_windows() -> dict:
    """
    The confirmation's window, derived from config/windows.json and nothing else.

    The builder reads two fields: the exclusive maximum daily date, which becomes
    the holdout's start, and the last return month, which becomes the month
    before it so the formation cap lands on 2020-11.
    """
    w = json.loads(P.windows_json().read_text())
    holdout_start = w["holdout"]["start"]
    last_return = str(pd.Period(holdout_start, freq="M") - 1)
    if str(pd.Period(last_return, freq="M") - 1) != LAST_FORMATION_MONTH:
        raise AssertionError(
            "the holdout start %s does not cap formation at %s"
            % (holdout_start, LAST_FORMATION_MONTH))
    out = dict(w)
    out["half_a"] = dict(w["half_a"])
    out["half_a"]["daily_row_exclusive_max"] = holdout_start
    out["half_a"]["return_months_through"] = last_return
    out["_conf_note"] = (
        "the confirmation's window: daily rows strictly before the holdout's "
        "start, return months through %s, formation months through %s"
        % (last_return, LAST_FORMATION_MONTH))
    return out


def build_panel(out_path: Path | None = None) -> tuple[Path, dict]:
    import build_panel_r1 as v1
    import build_panel_r1b as v2

    t0 = time.time()
    out_path = Path(out_path or P.panel_file())
    scratch = P.scratch() / "_conf_buckets"
    P.scratch().mkdir(parents=True, exist_ok=True)
    # build_panel_r1b binds its own SCRATCH to v1's at import; both must move
    v1.SCRATCH = scratch
    v2.SCRATCH = scratch

    w = conf_windows()
    log("confirmation window: daily rows < %s; return months through %s"
        % (w["half_a"]["daily_row_exclusive_max"],
           w["half_a"]["return_months_through"]))

    tight, meta_a = v2.pass_a_tight(w)
    calendar, meta_b = v1.pass_b(tight, w)
    panel, meta_c = v1.pass_c(calendar, w)
    panel, meta_d = v1.pass_d(panel, w)
    mkt, meta_e = v2.pass_e_market(calendar, panel, w)
    betas = v2.pass_f_beta(calendar, mkt, w)

    log("PASS G: attach controls")
    panel = panel.merge(betas, on=["permaticker", "month"], how="left")
    panel = panel.merge(tight[["permaticker", "sector"]], on="permaticker", how="left")
    panel["sector"] = panel["sector"].fillna(xt.NO_SECTOR)
    with np.errstate(invalid="ignore", divide="ignore"):
        dv = panel["med_dollar_vol_21"].to_numpy(dtype=np.float64)
        panel["log_dv"] = np.where(dv > 0, np.log(dv), np.nan)

    keep = ["permaticker", "month", "month_ts", "in_universe", "cand",
            "MOM12_1", "REV1", "LOWVOL", "HI52", "MAX",
            "ret_m", "fwd_ret", "fwd_stopped", "closeunadj_t", "n_sessions_21",
            "n_sessions_month", "med_dollar_vol_21", "n_ret_252",
            "beta", "n_beta_obs", "sector", "log_dv"]
    out = panel[keep].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    shutil.rmtree(scratch, ignore_errors=True)

    u = out.loc[out["in_universe"].astype(bool)]
    meta = {
        "version": VERSION, "panel": str(out_path.name),
        "panel_sha256": sha256_file(out_path),
        "rows": int(len(out)), "months": int(out["month"].nunique()),
        "first_month": str(pd.Period(ordinal=int(out["month"].min()), freq="M")),
        "last_month": str(pd.Period(ordinal=int(out["month"].max()), freq="M")),
        "in_universe_rows": int(len(u)),
        "universe_months": int(u["month"].nunique()),
        "universe_first_month": str(pd.Period(ordinal=int(u["month"].min()), freq="M")),
        "universe_last_month": str(pd.Period(ordinal=int(u["month"].max()), freq="M")),
        "max_daily_date_loaded": str(meta_b.get("max_daily_date_loaded", "")),
        "daily_rows_read": int(meta_b.get("daily_rows_read_half_a", 0)),
        "global_sessions": int(meta_b.get("global_sessions", 0)),
        "wall_seconds": round(time.time() - t0, 1),
        "window": w["half_a"],
    }
    log("PANEL %s rows %d sha %s (%.0f s)"
        % (out_path.name, meta["rows"], meta["panel_sha256"][:16], meta["wall_seconds"]))
    return out_path, meta


# ------------------------------------------------------------------ tripwire T1
T1_LAST_MONTH = "2012-11"


def t1(panel_path: Path | None = None) -> dict:
    """
    Every row and column of the banked half-A panel, for months through 2012-11,
    reproduced exactly inside the new panel.
    """
    cut = pd.Period(T1_LAST_MONTH, freq="M").ordinal
    banked = pd.read_parquet(P.r3_banked_panel())
    new = pd.read_parquet(Path(panel_path or P.panel_file()))
    b = banked.loc[banked["month"] <= cut].sort_values(
        ["permaticker", "month"], kind="stable").reset_index(drop=True)
    n = new.loc[new["month"] <= cut].sort_values(
        ["permaticker", "month"], kind="stable").reset_index(drop=True)
    res = {"tripwire": "T1", "last_month": T1_LAST_MONTH,
           "banked_rows": int(len(b)), "new_rows": int(len(n)),
           "rows_match": bool(len(b) == len(n)),
           "columns_match": bool(list(b.columns) == list(n.columns)),
           "differences": []}
    if not res["rows_match"] or not res["columns_match"]:
        res["identical"] = False
        res["differences"].append(
            {"what": "shape or columns",
             "banked": [len(b), list(b.columns)], "new": [len(n), list(n.columns)]})
        return res
    for c in b.columns:
        x, y = b[c], n[c]
        if pd.api.types.is_float_dtype(x):
            a = x.to_numpy(dtype=np.float64)
            z = y.to_numpy(dtype=np.float64)
            bad = int(((np.isnan(a) != np.isnan(z))
                       | ((~np.isnan(a)) & (a != z))).sum())
        else:
            bad = int((x.to_numpy() != y.to_numpy()).sum())
        if bad:
            res["differences"].append({"column": c, "mismatched_rows": bad})
    res["identical"] = not res["differences"]
    return res


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = (argv[0] if argv else "build").lower()
    if what == "build":
        _, meta = build_panel()
        P.derived().mkdir(parents=True, exist_ok=True)
        (P.derived() / "panel_meta.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2, sort_keys=True))
        return 0
    if what == "t1":
        r = t1()
        print(json.dumps(r, indent=2, sort_keys=True))
        return 0 if r["identical"] else 1
    raise SystemExit("usage: canon_conf_panel.py build | t1")


if __name__ == "__main__":
    sys.exit(main())
