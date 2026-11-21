"""
================================================================================
Script:       canon_r3_paths.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The one place every canon_r3 tool asks for a path, so a test can
              redirect all of them at once and no test can reach the banked
              panel, the feature file or the vendor snapshot.
Entry Point:  No
Dependencies: stdlib
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
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
    "banked_panel": _REPO / "data" / "derived"
                    / "panel_monthly_20261004p_halfA_v2.parquet",
    "derived": _REPO / "data" / "derived" / "canon_r3",
    "scratch": Path("/Volumes/MCS_DATA/canon_r3_scratch"),
    "windows": _REPO / "config" / "windows.json",
}


def set_roots(**kw) -> None:
    """Redirect any of: snapshot, banked_panel, derived, scratch, windows, repo."""
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


def banked_panel() -> Path:
    return _STATE["banked_panel"]


def derived() -> Path:
    return _STATE["derived"]


def scratch() -> Path:
    return _STATE["scratch"]


def windows_json() -> Path:
    return _STATE["windows"]


def feature_file() -> Path:
    return derived() / "features_20261004p_halfA_v2.parquet"


def pool_dir() -> Path:
    return scratch() / "pool"


def pool_manifest() -> Path:
    return pool_dir() / "pool_manifest.json"


def lock_file() -> Path:
    return scratch() / "canon_r3_observed.lock"


def controls_receipt() -> Path:
    return _STATE["repo"] / "registrations" / "canon_r3" / "controls_receipt_v1.json"
