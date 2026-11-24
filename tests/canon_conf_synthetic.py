"""
Synthetic fixtures for the confirmation's tests. Not a test module.

Every canon_conf test runs on synthetic frames written into a temporary
directory with canon_conf_paths redirected there, so no test can reach the
confirmation's panel, its feature file, read 3's banked artifacts or the vendor
snapshot.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import canon_conf_paths as P  # noqa: E402

SECTORS = ["Alpha", "Beta", "Gamma", "Delta"]
CUTOFF = "2021-01-01"


def snapshot_state() -> dict:
    return copy.deepcopy(P._STATE)


def restore_state(st: dict) -> None:
    P._STATE.clear()
    P._STATE.update(st)


def install(tmp: Path) -> dict:
    """Redirect every canon_conf path into tmp and write a windows.json."""
    wj = tmp / "windows.json"
    wj.write_text(json.dumps({
        "vintage": "synthetic",
        "half_a": {"label": "half_A", "return_months_through": "2012-12",
                   "daily_row_exclusive_max": "2013-01-01"},
        "half_b": {"start": "2013-01-01", "end": "2020-12-31"},
        "holdout": {"start": CUTOFF, "status": "SEALED"}}))
    P.set_roots(repo=tmp, snapshot=tmp / "snap", derived=tmp / "derived",
                scratch=tmp / "scratch", windows=wj)
    (tmp / "derived").mkdir(exist_ok=True)
    (tmp / "scratch").mkdir(exist_ok=True)
    (tmp / "snap").mkdir(exist_ok=True)
    return {"tmp": tmp, "windows": wj, "cutoff": CUTOFF}


def world_frame(n_names=200, n_months=36, first="1998-12", seed=5) -> pd.DataFrame:
    """A frame shaped like the confirmation's feature matrix."""
    import canon_conf_features as F
    rng = np.random.default_rng(seed)
    o = pd.Period(first, freq="M").ordinal
    recs = []
    for j in range(n_months):
        for i in range(n_names):
            r = {"permaticker": 900000 + i, "month": o + j, "in_universe": True,
                 "fwd_ret": float(rng.normal(0, 0.08)),
                 "fwd_stopped": bool(rng.random() < 0.004),
                 "beta": 1.0 + float(rng.normal(0, 0.2)),
                 "log_dv": 16.0 + float(rng.normal()),
                 "sector": SECTORS[i % len(SECTORS)]}
            for n in F.FEATURE_NAMES:
                r[n] = float(rng.uniform(-0.5, 0.5))
            recs.append(r)
    return pd.DataFrame(recs)


def write_panel_and_features(tmp: Path, frame: pd.DataFrame) -> dict:
    """The two parquets the runner's _load reads."""
    frame.to_parquet(P.feature_file(), index=False)
    panel = frame[["permaticker", "month", "in_universe", "fwd_ret",
                   "fwd_stopped", "beta", "log_dv", "sector"]].copy()
    panel.to_parquet(P.panel_file(), index=False)
    import canon_conf_features as F
    (P.derived() / "features_meta.json").write_text(json.dumps({
        "non_nan_share": {n: 1.0 for n in F.FEATURE_NAMES}}, indent=2))
    (P.derived() / "s3_counts.json").write_text(json.dumps({
        "in_universe_scored_rows": int(len(frame)),
        "stopped": {"rows": int(frame["fwd_stopped"].sum()), "for_cause": 0,
                    "takeover": 0, "share_of_scored_rows": 0.0},
        "ends_inside_the_holding_month": {"rows": 0, "for_cause": 0,
                                          "takeover": 0,
                                          "share_of_scored_rows": 0.0}}, indent=2))
    return {"panel": P.panel_file(), "features": P.feature_file()}


def write_provenance(tmp: Path) -> dict:
    """Stand-ins for read 3's seal and result, which the pins name."""
    (tmp / "registrations" / "canon_r3").mkdir(parents=True, exist_ok=True)
    (tmp / "reports" / "canon_r3").mkdir(parents=True, exist_ok=True)
    P.r3_seal().write_text(json.dumps({"look": "CANON-R3", "synthetic": True}))
    P.r3_result().write_text(json.dumps({
        "verdict": "SURVIVOR",
        "reported": {"series_statistics": {
            "gross": {"sharpe": 1.8304}, "net_base": {"sharpe": 1.4083},
            "net_stress": {"sharpe": 0.6800}}}}))
    (tmp / "requirements.lock").write_text("numpy==2.5.3\n")
    (tmp / "reports").mkdir(exist_ok=True)
    return {"r3_seal": P.r3_seal(), "r3_result": P.r3_result()}


def write_tools_copy(tmp: Path) -> Path:
    """The pins name tools by file; give the fake repo a tools directory."""
    import canon_conf_gate as G
    d = tmp / "tools"
    d.mkdir(exist_ok=True)
    for n in G.R3_TOOLS + G.CONF_TOOLS:
        (d / n).write_text("# %s\n" % n)
    return d


def write_ledger(tmp: Path) -> Path:
    """A 16-row ledger copied from the real one, for the rehearsal's append."""
    src = ROOT / "reports" / "triage_ledger.csv"
    (tmp / "reports").mkdir(exist_ok=True)
    P.ledger().write_text(src.read_text())
    return P.ledger()
