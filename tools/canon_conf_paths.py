"""
================================================================================
Script:       canon_conf_paths.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The one place every canon_conf tool asks for a path, so a test can
              redirect all of them at once and no test can reach the banked
              half-A artifacts, the confirmation artifacts or the vendor snapshot.
Entry Point:  No
Dependencies: stdlib
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

Paths are read through the accessor functions, never bound at import, so
set_roots() takes effect in modules that imported this one earlier.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

VERSION = "1.0.0"

_REPO = Path(__file__).resolve().parent.parent

_STATE = {
    "repo": _REPO,
    "snapshot": _REPO / "data" / "sharadar" / "snapshot_20261004p",
    "derived": _REPO / "data" / "derived" / "canon_r3c",
    "scratch": Path("/Volumes/MCS_DATA/canon_r3c_scratch"),
    "windows": _REPO / "config" / "windows.json",
}


def set_roots(**kw) -> None:
    for k, v in kw.items():
        if k not in _STATE:
            raise KeyError("unknown path root %r" % k)
        _STATE[k] = Path(v)


def repo() -> Path:
    return _STATE["repo"]


def snapshot() -> Path:
    return _STATE["snapshot"]


def bulk_stocks() -> Path:
    return snapshot() / "bulk_stocks.parquet"


def bulk_tickers() -> Path:
    return snapshot() / "bulk_tickers.parquet"


def bulk_actions() -> Path:
    return snapshot() / "bulk_actions.parquet"


def derived() -> Path:
    return _STATE["derived"]


def scratch() -> Path:
    return _STATE["scratch"]


def windows_json() -> Path:
    return _STATE["windows"]


# ---- the confirmation's own artifacts
def panel_file() -> Path:
    return derived() / "panel_monthly_20261004p_conf_v2.parquet"


def feature_file() -> Path:
    return derived() / "features_20261004p_conf_v2.parquet"


def banked_panel() -> Path:
    """
    The name the copied feature module calls (canon_conf_features is
    canon_r3_features plus declared substitutions, so it still asks for
    "banked_panel"). For the confirmation that is the confirmation's own panel.
    Read 3's banked half-A panel is r3_banked_panel().
    """
    return panel_file()


def features_meta() -> Path:
    return derived() / "features_meta.json"


def canon_dir() -> Path:
    return repo() / "registrations" / "canon_r3c"


def pool_dir() -> Path:
    return scratch() / "pool"


def pool_manifest() -> Path:
    return pool_dir() / "pool_manifest.json"


def pool_v1() -> Path:
    return canon_dir() / "pool_v1.json"


def controls_v1() -> Path:
    return canon_dir() / "controls_v1.json"


def draft_registration() -> Path:
    return canon_dir() / "registration_v1_draft.json"


def sealed_registration() -> Path:
    return canon_dir() / "registration.json"


def controls_receipt() -> Path:
    return canon_dir() / "controls_receipt.json"


def spent_marker() -> Path:
    return canon_dir() / "r3c_spent.json"


def lock_file() -> Path:
    return scratch() / "canon_r3c_observed.lock"


def requirements_lock() -> Path:
    return repo() / "requirements.lock"


def reports_dir() -> Path:
    return repo() / "reports" / "canon_r3c"


def read_result() -> Path:
    return reports_dir() / "r3c_result_v1.json"


def read_report() -> Path:
    return reports_dir() / "r3c_report_v1.md"


def ledger() -> Path:
    return repo() / "reports" / "triage_ledger.csv"


# ---- read 3's artifacts, read-only provenance
def r3_banked_panel() -> Path:
    return repo() / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"


def r3_feature_file() -> Path:
    return repo() / "data" / "derived" / "canon_r3" / "features_20261004p_halfA_v2.parquet"


def r3_pool_v1() -> Path:
    return repo() / "registrations" / "canon_r3" / "pool_v1.json"


def r3_seal() -> Path:
    return repo() / "registrations" / "canon_r3" / "registration_sealed.json"


def r3_result() -> Path:
    return repo() / "reports" / "canon_r3" / "read3_result_v1.json"
