"""
================================================================================
Script:       run_canon_r3.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The runner: --stage controls runs the pool of 60 scrambled worlds,
              the known-null control, the plants and the MDE; --stage observed
              takes the read, and only behind the full gate.
Entry Point:  Yes — python tools/run_canon_r3.py --stage controls
                     python tools/run_canon_r3.py --stage observed
                         --registration <path> --sha <sha256>
Dependencies: numpy, pandas, sklearn; tools/canon_r3_*.py, tools/xs_triage.py
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================

The controls stage never constructs world 0: it calls build_world with k >= 1
only, and the plant refuses k == 0. The observed stage obtains world 0 from a
VerifiedRegistration the gate minted, never from a flag.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_r3_features as F  # noqa: E402
import canon_r3_gate as G  # noqa: E402
import canon_r3_machine as M  # noqa: E402
import canon_r3_paths as P  # noqa: E402
import canon_r3_tripwire as TW  # noqa: E402
import canon_r3_world as W  # noqa: E402
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

PLANT_WORLDS = tuple(range(1, 13))
PLANT_RHOS = (0.005, 0.01, 0.02, 0.04, 0.08)
PLANT_PASS_MIN = 10          # 7.4: "at least 10 of 12"


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _load() -> tuple[pd.DataFrame, np.ndarray]:
    feats = pd.read_parquet(P.feature_file())
    panel = pd.read_parquet(P.banked_panel())
    return feats.reset_index(drop=True), panel["permaticker"].unique()


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(repr(type(o)))


# ------------------------------------------------------------------ 7.1 the pool
def run_pool(feats, allpm, worlds=range(1, W.N_WORLDS + 1)) -> dict:
    P.pool_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for k in worlds:
        r = M.run_world(feats, int(k), all_permatickers=allpm)
        (P.pool_dir() / ("world_%02d.json" % k)).write_text(
            json.dumps(r, indent=1, default=_json))
        rows.append({"k": int(k),
                     "gross_t": r["stats"]["gross"]["t"],
                     "gross_mean": r["stats"]["gross"]["mean"],
                     "gross_sharpe": r["stats"]["gross"]["sharpe"],
                     "net_base_t": r["stats"]["net_base"]["t"],
                     "net_base_mean": r["stats"]["net_base"]["mean"],
                     "net_base_sharpe": r["stats"]["net_base"]["sharpe"],
                     "net_stress_sharpe": r["stats"]["net_stress"]["sharpe"],
                     "turnover_mean": r["stats"]["turnover_mean"],
                     "months": r["stats"]["months"],
                     "label_rows": r["label_rows"],
                     "wall_seconds": r["wall_seconds"]})
        log("world %02d  gross t %+7.3f  net_base Sharpe %+6.3f  %.1fs"
            % (k, rows[-1]["gross_t"], rows[-1]["net_base_sharpe"],
               rows[-1]["wall_seconds"]))
    man = {"version": VERSION, "env": TW.env_versions(),
           "n_worlds": len(rows), "worlds": rows,
           "feature_sha256": G.sha256_file(P.feature_file()),
           "banked_panel_sha256": G.sha256_file(P.banked_panel())}
    P.pool_manifest().write_text(json.dumps(man, indent=2, default=_json))
    man["pool_manifest_sha256"] = G.sha256_file(P.pool_manifest())
    return man


# ------------------------------------------------------- 7.2 the known null
def known_null(man: dict) -> dict:
    gm = np.array([w["gross_mean"] for w in man["worlds"]], dtype=np.float64)
    res = []
    for i, w in enumerate(man["worlds"]):
        others = np.delete(gm, i)
        g = np.array(json.loads(
            (P.pool_dir() / ("world_%02d.json" % w["k"])).read_text()
        )["series"]["gross"], dtype=np.float64)
        k1a = M.key1a(g, float(np.median(others)))
        res.append({"k": w["k"], "p": k1a["p"], "pass": k1a["pass"]})
    share = float(np.mean([r["pass"] for r in res]))
    gt = np.array([w["gross_t"] for w in man["worlds"]], dtype=np.float64)
    nt = np.array([w["net_base_t"] for w in man["worlds"]], dtype=np.float64)
    whole = []
    for i, w in enumerate(man["worlds"]):
        others_t = np.delete(gt, i)
        k1a = next(r for r in res if r["k"] == w["k"])
        b = M.key1b(w["gross_t"], others_t)
        k2 = M.key2(w["net_base_sharpe"])
        whole.append(M.verdict({"pass": k1a["pass"]}, b, k2))
    return {
        "n_worlds": len(res), "alpha": M.LEVEL,
        "share_significant": share,
        "accept_band": list(M.CONTROL_BAND),
        "pass": bool(xt.control_verdict(share, M.CONTROL_BAND)),
        "per_world": res,
        "gross_t": {"median": float(np.median(gt)), "sd": float(gt.std(ddof=1)),
                    "min": float(gt.min()), "max": float(gt.max())},
        "net_base_t": {"median": float(np.median(nt)), "sd": float(nt.std(ddof=1)),
                       "min": float(nt.min()), "max": float(nt.max())},
        "verdict_counts": {v: int(sum(1 for x in whole if x == v))
                           for v in sorted(set(whole))},
        "worlds_called_survivor": int(sum(1 for x in whole if x == "SURVIVOR")),
    }


# ------------------------------------------------------------- 7.3 the plants
def run_plants(feats, allpm, man: dict) -> dict:
    gm = {w["k"]: w["gross_mean"] for w in man["worlds"]}
    gt = {w["k"]: w["gross_t"] for w in man["worlds"]}
    out = {"rhos": list(PLANT_RHOS), "worlds": list(PLANT_WORLDS), "by_rho": {}}
    for j, rho in enumerate(PLANT_RHOS):
        recs = []
        for k in PLANT_WORLDS:
            r = M.run_world(feats, int(k), all_permatickers=allpm,
                            plant_rho=float(rho), plant_rho_index=j)
            g = np.array(r["series"]["gross"], dtype=np.float64)
            others_mean = np.array([gm[x] for x in gm if x != k], dtype=np.float64)
            others_t = np.array([gt[x] for x in gt if x != k], dtype=np.float64)
            a = M.key1a(g, float(np.median(others_mean)))
            b = M.key1b(r["stats"]["gross"]["t"], others_t)
            k2 = M.key2(r["stats"]["net_base"]["sharpe"])
            recs.append({"k": int(k), "p_1a": a["p"], "pass_1a": a["pass"],
                         "exceedances_1b": b["exceedances"], "pass_1b": b["pass"],
                         "pass_key1": bool(a["pass"] and b["pass"]),
                         "verdict": M.verdict(a, b, k2),
                         "gross_sharpe": r["stats"]["gross"]["sharpe"],
                         "net_base_sharpe": r["stats"]["net_base"]["sharpe"],
                         "turnover_mean": r["stats"]["turnover_mean"]})
            log("plant rho %.3f world %02d  1a %s 1b %s  gross Sharpe %+6.3f  %s"
                % (rho, k, recs[-1]["pass_1a"], recs[-1]["pass_1b"],
                   recs[-1]["gross_sharpe"], recs[-1]["verdict"]))
        n = len(recs)
        out["by_rho"]["%.3f" % rho] = {
            "rho": float(rho), "n_worlds": n,
            "share_pass_1a": float(np.mean([r["pass_1a"] for r in recs])),
            "share_pass_1b": float(np.mean([r["pass_1b"] for r in recs])),
            "share_pass_key1": float(np.mean([r["pass_key1"] for r in recs])),
            "n_pass_key1": int(sum(r["pass_key1"] for r in recs)),
            "share_survivor": float(np.mean([r["verdict"] == "SURVIVOR" for r in recs])),
            "verdict_counts": {v: int(sum(1 for r in recs if r["verdict"] == v))
                               for v in sorted({r["verdict"] for r in recs})},
            "median_gross_sharpe": float(np.median([r["gross_sharpe"] for r in recs])),
            "median_net_base_sharpe": float(
                np.median([r["net_base_sharpe"] for r in recs])),
            "median_turnover": float(np.median([r["turnover_mean"] for r in recs])),
            "per_world": recs,
        }
    return out


def mde(plants: dict) -> dict:
    hit = None
    for rho in PLANT_RHOS:
        b = plants["by_rho"]["%.3f" % rho]
        if b["n_pass_key1"] >= PLANT_PASS_MIN:
            hit = b
            break
    return {
        "rule": "the smallest rho at which key 1 passes in at least %d of 12"
                % PLANT_PASS_MIN,
        "rho": None if hit is None else hit["rho"],
        "n_pass_key1": None if hit is None else hit["n_pass_key1"],
        "median_gross_sharpe_at_mde": None if hit is None else hit["median_gross_sharpe"],
        "analytic_gross_sharpe_80pc_power_132_months": M.analytic_sharpe_for_power(),
        "detected": hit is not None,
    }


# ------------------------------------------------------------ 7.5 determinism
def determinism(feats, allpm, k: int = 1) -> dict:
    a = M.run_world(feats, k, all_permatickers=allpm)
    b = M.run_world(feats, k, all_permatickers=allpm)
    same = {}
    for s in ("gross", "net_base", "net_stress", "turnover"):
        x = np.asarray(a["series"][s], dtype=np.float64)
        y = np.asarray(b["series"][s], dtype=np.float64)
        same[s] = bool(x.tobytes() == y.tobytes())
    return {"k": int(k), "bit_identical": all(same.values()), "per_series": same}


# ------------------------------------------------------------------ the stages
def stage_controls() -> int:
    t0 = time.time()
    feats, allpm = _load()
    log("features %s; env %s" % (str(feats.shape), json.dumps(TW.env_versions())))
    man = run_pool(feats, allpm)
    log("pool of %d worlds written; manifest sha %s"
        % (man["n_worlds"], man["pool_manifest_sha256"]))
    kn = known_null(man)
    log("known null: share significant %.4f band %s pass %s"
        % (kn["share_significant"], kn["accept_band"], kn["pass"]))
    pl = run_plants(feats, allpm, man)
    md = mde(pl)
    det = determinism(feats, allpm, 1)
    fit_rows = json.loads(
        (P.pool_dir() / "world_01.json").read_text())["fit_diag"]
    _, fmeta = None, json.loads((P.derived() / "features_meta.json").read_text())
    out = {
        "stage": "controls", "version": VERSION, "env": TW.env_versions(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "design": {
            "features": F.FEATURE_NAMES,
            "formation_months": [F.FIRST_FORMATION_MONTH, F.LAST_FORMATION_MONTH],
            "scored_months": [M.FIRST_SCORED, M.LAST_SCORED, M.N_SCORED_MONTHS],
            "cuts": [str(pd.Period(ordinal=c, freq="M")) for c in M.cut_months()],
            "model": dict(M.MODEL_KW), "seeds": list(M.SEEDS),
            "voting": {"mean_block": M.MEAN_BLOCK, "R": M.R_VOTING,
                       "seed": M.SEED_VOTING, "level": M.LEVEL,
                       "key1b_max_exceedances": M.KEY1B_MAX_EXCEEDANCES,
                       "key2_sharpe_bar": M.KEY2_SHARPE_BAR},
        },
        "pool": {"n_worlds": man["n_worlds"], "worlds": man["worlds"],
                 "pool_manifest_sha256": man["pool_manifest_sha256"]},
        "known_null": kn, "plants": pl, "mde": md, "determinism": det,
        "fit_diagnostics_world_01": fit_rows,
        "feature_non_nan_share": fmeta["non_nan_share"],
        "feature_sha256": man["feature_sha256"],
        "banked_panel_sha256": man["banked_panel_sha256"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    dest = P.repo() / "registrations" / "canon_r3"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "controls_v1.json").write_text(json.dumps(out, indent=2, default=_json))
    log("controls written: %s" % (dest / "controls_v1.json"))
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def stage_observed(registration: str, sha: str) -> int:
    """
    The read. Refuses without a seal, without matching data pins, without a
    controls receipt written against that seal, and while the lock is held.
    """
    opened = G.open_the_gate(registration, sha)
    reg = opened["registration"]
    with G.single_run_lock():
        feats, allpm = _load()
        r = M.run_world(feats, 0, registration=reg, all_permatickers=allpm)
        print(json.dumps({"stage": "observed", "registration_sha": reg.sha256,
                          "stats": r["stats"]}, indent=2, default=_json))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["controls", "observed"])
    ap.add_argument("--registration")
    ap.add_argument("--sha")
    a = ap.parse_args(argv)
    if a.stage == "controls":
        return stage_controls()
    if not a.registration or not a.sha:
        raise SystemExit("--stage observed needs --registration and --sha")
    return stage_observed(a.registration, a.sha)


if __name__ == "__main__":
    sys.exit(main())
