"""
================================================================================
Script:       canon_conf_seal.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      Build a registration for the confirmation from its parts. The ONLY
              writer of one, draft or sealed.
Entry Point:  Yes — python tools/canon_conf_seal.py draft
Dependencies: stdlib, numpy, pandas; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-2 — the recount, the corrected floors, the read (2026-11-24)
================================================================================

A registration is JSON, because that is the form its gate reads — the lesson of
read 3's halted markdown seal (halt_1.json). The draft this sprint writes carries
every field a seal would and is named a draft; it pins nothing it has not hashed.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_gate as G  # noqa: E402
import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402
import canon_conf_features as F  # noqa: E402
import canon_r3_machine as M  # noqa: E402

VERSION = "1.0.0"

LOOK = "CANON-R3C"
SIGNAL = "MACHINE_HGB_25PV_CONF"
UNIVERSE = "top1000_tight_v2_halfB_96mo"
VINTAGE = "20261004p"


def _bars(pool_worlds) -> dict:
    """
    The bars as numbers. Key 1(b)'s bar is the THIRD-largest pool gross t, so a
    pool with fewer than three worlds has no bar to state and the field is
    omitted rather than invented.
    """
    import numpy as np
    if len(pool_worlds) < 3:
        return {"available": False,
                "reason": "the pool holds %d worlds; key 1(b)'s bar needs at "
                          "least three" % len(pool_worlds)}
    gm = np.array([w["gross"]["mean"] for w in pool_worlds], dtype=np.float64)
    gt = np.sort(np.array([w["gross"]["t"] for w in pool_worlds],
                          dtype=np.float64))[::-1]
    return {"key1a_centre_pool_median_gross_mean": float(np.median(gm)),
            "key1a_centre_bps": float(np.median(gm)) * 1e4,
            "key1b_third_largest_pool_gross_t": float(gt[2]),
            "key1b_max_exceedances": M.KEY1B_MAX_EXCEEDANCES,
            "key2_sharpe_bar": M.KEY2_SHARPE_BAR,
            "level": M.LEVEL,
            "n_worlds": int(gm.size),
            "available": True,
            "source": "pool_v1.json, as the read computes them"}


def build(out_path: Path, prose: dict, draft: bool = True,
          pool: dict | None = None) -> str:
    """Assemble a registration from its parts and return its sha256."""
    if pool is None and P.pool_v1().exists():
        pool = json.loads(P.pool_v1().read_text())
    body = {
        "look": LOOK,
        "read": "the confirmation of canon read 3",
        "status": "DRAFT, UNSEALED" if draft else "SEALED",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": {
            "window": {
                "formation_months": [F.FIRST_FORMATION_MONTH,
                                     F.LAST_FORMATION_MONTH,
                                     F.N_FORMATION_MONTHS],
                "cuts": [str(__import__("pandas").Period(ordinal=c, freq="M"))
                         for c in PL.CONF_CUTS],
                "scored_months": [PL.CONF_FIRST_SCORED, PL.CONF_LAST_SCORED,
                                  PL.CONF_N_SCORED_MONTHS],
                "returns_realised": ["2013-01", "2020-12"],
                "daily_cutoff_source": "config/windows.json holdout.start",
            },
            "features": F.FEATURE_NAMES,
            "label": "read 3's: the world's fwd_ret residualised within the "
                     "month on [1, sector dummies, beta, log dollar volume], "
                     "then a within-month normal score",
            "model": dict(M.MODEL_KW), "seeds": list(M.SEEDS),
            "book": "read 3's: the raw prediction into build_series_v2, "
                    "direction HIGH, harness v2's slope book",
            "costs": "read 3's COST_BASE and COST_STRESS, unchanged",
            "worlds": {"n": PL.N_WORLDS, "seed_base": PL.CONF_SEED_BASE,
                       "rule": "in every formation month of the window, "
                               "in-universe names in key order each take the "
                               "next name's fwd_ret and fwd_stopped, the last "
                               "takes the first's"},
            "voting": {"mean_block": M.MEAN_BLOCK, "R": M.R_VOTING,
                       "seed": PL.CONF_SEED_VOTING, "level": M.LEVEL},
            "verdict": {
                "key1a": "stationary bootstrap of months on the gross series "
                         "less the median gross mean of the 60 worlds; "
                         "one-sided p <= 0.05",
                "key1b": "the gross t equalled or exceeded by at most %d of the "
                         "60 worlds' gross t" % M.KEY1B_MAX_EXCEEDANCES,
                "key2": "annualised net BASE Sharpe >= %.1f" % M.KEY2_SHARPE_BAR,
                "ladder": "CONFIRMED if both keys hold; otherwise "
                          "NOT_CONFIRMED, naming each key that failed",
                "key3": "STRUCK by ruling (CONF-1b); the stopped-stock exposure "
                        "is reported as a bound and never voted",
            },
        },
        "ledger": {"batch_id": LOOK, "signal": SIGNAL,
                   "universe_string": UNIVERSE, "vintage_string": VINTAGE,
                   "harness": "tools/canon_conf_pipeline.py"},
        "data_pins": G.current_pins(),
        "provenance": {
            "read_3_seal_sha256": G.sha256_file(P.r3_seal())
            if P.r3_seal().exists() else None,
            "read_3_result_sha256": G.sha256_file(P.r3_result())
            if P.r3_result().exists() else None,
        },
    }
    if pool is not None:
        body["bars"] = _bars(pool.get("worlds", []))
    body.update(prose)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, indent=2, sort_keys=True))
    return G.sha256_file(out_path)


def seal(out_path: Path | None = None) -> str:
    """
    Write the real registration from the corrected parts. The prose comes from
    canon_conf_prose, which derives every figure from the measured artifacts, so
    no number in the seal is hand-entered.
    """
    import canon_conf_prose as PR
    out_path = Path(out_path or P.sealed_registration())
    return build(out_path, PR.build_prose(), draft=False)


def draft(out_path: Path | None = None) -> str:
    import canon_conf_prose as PR
    out_path = Path(out_path or P.draft_registration())
    return build(out_path, PR.build_prose(), draft=True)


def dummy(out_path: Path, note: str = "REHEARSAL dummy seal. Not a seal.") -> str:
    """The rehearsal's seal: the same builder, a dummy's prose."""
    return build(out_path, {"amendment_1": note}, draft=False)
