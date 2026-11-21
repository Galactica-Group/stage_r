"""
================================================================================
Script:       canon_r3_tripwire.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 (the machine on the stock panel)
Purpose:      The two environment tripwires for Sprint STOCK-1: T1 rebuilds the
              banked v2 half-A panel from the repo's own builder functions and
              compares it with the banked file; T2 recomputes the five banked
              T-R1b v2 monthly series from the banked panel with xs_triage's own
              functions and hashes them.
Entry Point:  Yes — python tools/canon_r3_tripwire.py [t1|t2|both]
Dependencies: numpy, pandas, pyarrow; tools/xs_triage.py, tools/build_panel_r1.py,
              tools/build_panel_r1b.py. Imports nothing from any other repository.
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================

The point of a tripwire is that it is computed before the package install and
again after it, by this one definition, so "nothing changed" is a measurement
and not an assurance. T1's comparison is a byte sha256 when the rebuild is
byte-identical to the banked file and otherwise a content hash of the frame,
defined once here (frame_content_hash) and used on both sides either way.

Neither tripwire writes to logs/, reports/ or the ledger. T1's rebuild and its
parquet buckets go to the scratch volume; the banked panel is read, never
rewritten.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
BANKED_PANEL = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"
BANKED_PANEL_SHA_PREFIX = "7f13b109"          # the T-R1b report's panel_sha256
SCRATCH = Path("/Volumes/MCS_DATA/canon_r3_scratch")

# The banked T-R1b net means, in bps, to the precision the report prints.
T_R1B_NET_MEAN_BPS = {
    "MOM12_1": 24.153716787256247,
    "REV1": 20.71903903220554,
    "LOWVOL": -32.0930970502304,
    "HI52": -49.62113998126755,
    "MAX": -18.236336088736618,
}
T_R1B_GROSS_MEAN_BPS = {
    "MOM12_1": 34.03649545889032,
    "REV1": 39.537526662209416,
    "LOWVOL": -24.10913635006003,
    "HI52": -37.52645442732366,
    "MAX": -0.9176671364192852,
}
T_R1B_MONTHS_VOTED = 168


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def frame_content_hash(df: pd.DataFrame) -> str:
    """
    One content hash of a frame, independent of parquet encoding.

    Rows are ordered by (permaticker, month) when both are present, else by the
    frame's own order; columns are taken in sorted name order so a column
    reordering cannot change the hash. Floats go in as float64 bytes, bools as
    uint8, integers as int64, everything else as NUL-joined UTF-8 of str().
    """
    d = df
    if "permaticker" in d.columns and "month" in d.columns:
        d = d.sort_values(["permaticker", "month"], kind="stable")
    h = hashlib.sha256()
    h.update(("|".join(sorted(d.columns)) + "||").encode("utf-8"))
    h.update(("rows=%d" % len(d)).encode("utf-8"))
    for name in sorted(d.columns):
        col = d[name]
        h.update(("\x00COL:" + name + "\x00").encode("utf-8"))
        if pd.api.types.is_bool_dtype(col):
            h.update(col.to_numpy(dtype=np.uint8, na_value=255).tobytes())
        elif pd.api.types.is_float_dtype(col):
            h.update(np.ascontiguousarray(
                col.to_numpy(dtype=np.float64, na_value=np.nan)).tobytes())
        elif pd.api.types.is_integer_dtype(col):
            h.update(np.ascontiguousarray(
                col.to_numpy(dtype=np.int64, na_value=-(2 ** 62))).tobytes())
        elif pd.api.types.is_datetime64_any_dtype(col):
            h.update(np.ascontiguousarray(
                col.to_numpy(dtype="datetime64[ns]").astype(np.int64)).tobytes())
        else:
            h.update("\x00".join("" if pd.isna(v) else str(v)
                                 for v in col).encode("utf-8"))
    return h.hexdigest()


def series_bundle_hash(series_map: dict[str, np.ndarray]) -> str:
    """One sha256 over the float64 bytes of a name-keyed bundle of series."""
    h = hashlib.sha256()
    for name in sorted(series_map):
        a = np.ascontiguousarray(np.asarray(series_map[name], dtype=np.float64))
        h.update(("\x00SER:%s:%d\x00" % (name, a.size)).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


# --------------------------------------------------------------------- T1
def rebuild_v2_panel(out_path: Path, scratch: Path = SCRATCH) -> tuple[Path, dict]:
    """
    Rebuild the v2 half-A panel with build_panel_r1b's own passes, writing to
    `out_path`. The builder's bucket scratch is redirected onto `scratch` and
    its main() is never called, so neither the banked panel nor anything under
    data/sharadar or logs/ is touched.
    """
    import build_panel_r1 as v1
    import build_panel_r1b as v2

    t0 = time.time()
    scratch.mkdir(parents=True, exist_ok=True)
    # build_panel_r1b binds its own SCRATCH to v1.SCRATCH at import, so both
    # module globals must be redirected or passes E and F glob the old path.
    v1.SCRATCH = scratch / "_t1_buckets"
    v2.SCRATCH = v1.SCRATCH
    windows = json.load(open(ROOT / "config" / "windows.json"))

    tight, _ = v2.pass_a_tight(windows)
    calendar, _ = v1.pass_b(tight, windows)
    panel, _ = v1.pass_c(calendar, windows)
    panel, _ = v1.pass_d(panel, windows)
    mkt, _ = v2.pass_e_market(calendar, panel, windows)
    betas = v2.pass_f_beta(calendar, mkt, windows)

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

    import shutil
    shutil.rmtree(v1.SCRATCH, ignore_errors=True)
    return out_path, {"rows": int(len(out)), "wall_seconds": round(time.time() - t0, 1)}


def t1(scratch: Path = SCRATCH) -> dict:
    banked_sha = sha256_file(BANKED_PANEL)
    rebuilt = scratch / "t1_panel_rebuild_v2.parquet"
    _, meta = rebuild_v2_panel(rebuilt, scratch)
    rebuilt_sha = sha256_file(rebuilt)
    byte_identical = bool(rebuilt_sha == banked_sha)
    res = {
        "tripwire": "T1",
        "banked_panel": str(BANKED_PANEL.relative_to(ROOT)),
        "banked_sha256": banked_sha,
        "banked_sha_prefix_matches_t_r1b_report":
            bool(banked_sha.startswith(BANKED_PANEL_SHA_PREFIX)),
        "rebuilt_sha256": rebuilt_sha,
        "byte_identical": byte_identical,
        "rows": meta["rows"],
        "rebuild_wall_seconds": meta["wall_seconds"],
    }
    if byte_identical:
        res["comparison"] = "byte sha256"
        res["value"] = rebuilt_sha
    else:
        a = frame_content_hash(pd.read_parquet(BANKED_PANEL))
        b = frame_content_hash(pd.read_parquet(rebuilt))
        res["comparison"] = "frame content hash"
        res["banked_content_hash"] = a
        res["rebuilt_content_hash"] = b
        res["content_identical"] = bool(a == b)
        res["value"] = b
    return res


# --------------------------------------------------------------------- T2
def recompute_t_r1b_series(panel_path: Path = BANKED_PANEL) -> dict:
    """
    Recompute T-R1b's five v2 monthly series from the banked panel using
    xs_triage's own build_series/build_series_v2, on run_t_r1b's own common
    month window (the intersection of all ten series' months).
    """
    panel = pd.read_parquet(panel_path)
    signals = list(xt.REGISTERED_SIGNALS)
    v1f, v2f = {}, {}
    for n in signals:
        d = xt.REGISTERED_SIGNALS[n]
        v1f[n] = xt.build_series(panel, n, d, "LS", xt.COST_BASE).frame()
        s, _ = xt.build_series_v2(panel, n, d, "LS", xt.COST_BASE)
        v2f[n] = s.frame()
    sets = ([set(v1f[n]["month"]) for n in signals]
            + [set(v2f[n]["month"]) for n in signals])
    common = sorted(set.intersection(*sets))

    def oc(f, col):
        g = f.loc[f["month"].isin(common)].sort_values("month")
        return g[col].to_numpy(dtype=np.float64)

    bundle = {}
    for n in signals:
        bundle["%s::gross" % n] = oc(v2f[n], "gross")
        bundle["%s::net" % n] = oc(v2f[n], "net")
    net_bps = {n: float(bundle["%s::net" % n].mean()) * 1e4 for n in signals}
    gross_bps = {n: float(bundle["%s::gross" % n].mean()) * 1e4 for n in signals}
    return {
        "tripwire": "T2",
        "months_common": len(common),
        "first_month": str(pd.Period(ordinal=int(common[0]), freq="M")),
        "last_month": str(pd.Period(ordinal=int(common[-1]), freq="M")),
        "series_sha256": series_bundle_hash(bundle),
        "net_mean_bps": net_bps,
        "gross_mean_bps": gross_bps,
    }


def t2_agrees(res: dict, tol_bps: float = 1e-9) -> dict:
    """Compare a T2 result with the banked T-R1b report to its printed precision."""
    bad = {}
    for n, v in T_R1B_NET_MEAN_BPS.items():
        d = abs(res["net_mean_bps"][n] - v)
        if d > tol_bps:
            bad["net::%s" % n] = d
    for n, v in T_R1B_GROSS_MEAN_BPS.items():
        d = abs(res["gross_mean_bps"][n] - v)
        if d > tol_bps:
            bad["gross::%s" % n] = d
    return {"months_match": res["months_common"] == T_R1B_MONTHS_VOTED,
            "max_abs_bps_delta": max(
                [abs(res["net_mean_bps"][n] - T_R1B_NET_MEAN_BPS[n])
                 for n in T_R1B_NET_MEAN_BPS]
                + [abs(res["gross_mean_bps"][n] - T_R1B_GROSS_MEAN_BPS[n])
                   for n in T_R1B_GROSS_MEAN_BPS]),
            "agrees": (not bad) and res["months_common"] == T_R1B_MONTHS_VOTED,
            "offenders": bad}


def env_versions() -> dict:
    import pyarrow
    out = {"python": sys.version.split()[0], "numpy": np.__version__,
           "pandas": pd.__version__, "pyarrow": pyarrow.__version__}
    try:
        import sklearn
        out["scikit-learn"] = sklearn.__version__
    except Exception:
        out["scikit-learn"] = None
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    which = (argv[0] if argv else "both").lower()
    out = {"tool_version": VERSION, "env": env_versions()}
    if which in ("t1", "both"):
        out["t1"] = t1()
    if which in ("t2", "both"):
        r = recompute_t_r1b_series()
        r["agreement"] = t2_agrees(r)
        out["t2"] = r
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
