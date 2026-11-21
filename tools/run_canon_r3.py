"""
================================================================================
Script:       run_canon_r3.py
Version:      1.1.0
Phase:        Stage R — Canon Read 3
Purpose:      The runner: --stage controls runs the pool of 60 scrambled worlds,
              the known-null control, the plants and the MDE; --stage observed
              takes the read, and only behind the full gate.
Entry Point:  Yes — python tools/run_canon_r3.py --stage controls
                     python tools/run_canon_r3.py --stage observed
                         --registration <path> --sha <sha256>
Dependencies: numpy, pandas, sklearn; tools/canon_r3_*.py, tools/xs_triage.py
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
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

VERSION = "1.1.0"

# FROZEN. The scratch pool manifest records this, not the tool's version, so the
# pool reproduces across tool versions. STOCK-2 compares against 559b9122.
POOL_MANIFEST_VERSION = "1.0.0"
POOL_V1_VERSION = "1.0.0"

PLANT_WORLDS = tuple(range(1, 13))
# (rho, seed index). The five STOCK-1 values KEEP indices 0 to 4 so their rows
# reproduce controls_v1.json exactly; the three added by STOCK-2 take 5, 6, 7.
PLANT_GRID = ((0.005, 0), (0.010, 1), (0.020, 2), (0.040, 3), (0.080, 4),
              (0.025, 5), (0.030, 6), (0.035, 7))
PLANT_RHOS = tuple(r for r, _ in PLANT_GRID)
PLANT_PASS_MIN = 10          # 7.4 / 3.3: "at least 10 of 12"

LEDGER_BATCH_ID = "CANON-R3"
LEDGER_SIGNAL = "MACHINE_HGB_25PV"
LEDGER_UNIVERSE = "top1000_tight_v2_halfA_132mo"
LEDGER_VINTAGE = "20261004p"


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
    man = {"version": POOL_MANIFEST_VERSION, "env": TW.env_versions(),
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
    out = {"rhos": list(PLANT_RHOS), "worlds": list(PLANT_WORLDS),
           "grid": [{"rho": r, "seed_index": j} for r, j in PLANT_GRID],
           "by_rho": {}}
    for rho, j in PLANT_GRID:
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


def floors(plants: dict) -> dict:
    """
    3.3: two measured floors over the eight-value grid, each with the median
    gross and net Sharpe of its books.
      key1  — the smallest rho at which key 1 passes in at least 10 of 12
      whole — the smallest rho at which the WHOLE verdict is SURVIVOR in >= 10 of 12
    """
    def find(pred):
        for rho in sorted(PLANT_RHOS):
            b = plants["by_rho"]["%.3f" % rho]
            if pred(b):
                return b
        return None

    k1 = find(lambda b: b["n_pass_key1"] >= PLANT_PASS_MIN)
    wh = find(lambda b: b["verdict_counts"].get("SURVIVOR", 0) >= PLANT_PASS_MIN)

    def pack(b, label, rule):
        if b is None:
            return {"label": label, "rule": rule, "detected": False, "rho": None}
        return {"label": label, "rule": rule, "detected": True, "rho": b["rho"],
                "n_pass_key1": b["n_pass_key1"],
                "n_survivor": b["verdict_counts"].get("SURVIVOR", 0),
                "median_gross_sharpe": b["median_gross_sharpe"],
                "median_net_base_sharpe": b["median_net_base_sharpe"]}

    return {
        "grid": sorted(PLANT_RHOS),
        "key1_floor": pack(k1, "key 1 floor",
                           "smallest rho with key 1 passing in >= %d of 12"
                           % PLANT_PASS_MIN),
        "whole_verdict_floor": pack(wh, "whole-verdict floor",
                                    "smallest rho with SURVIVOR in >= %d of 12"
                                    % PLANT_PASS_MIN),
        "analytic_gross_sharpe_80pc_power_132_months": M.analytic_sharpe_for_power(),
        "key2_sharpe_bar": M.KEY2_SHARPE_BAR,
    }


def bars(pool_worlds: list[dict]) -> dict:
    """
    3.4: the two bars as numbers, from the TRACKED pool.
      key 1(a)'s centre  — the pool's median gross mean
      key 1(b)'s bar     — the third-largest pool gross t, which the read's
                           gross t must EXCEED for at most 2 exceedances
    """
    gm = np.array([w["gross"]["mean"] for w in pool_worlds], dtype=np.float64)
    gt = np.sort(np.array([w["gross"]["t"] for w in pool_worlds],
                          dtype=np.float64))[::-1]
    return {
        "key1a_centre_pool_median_gross_mean": float(np.median(gm)),
        "key1a_centre_bps": float(np.median(gm)) * 1e4,
        "key1b_third_largest_pool_gross_t": float(gt[2]),
        "key1b_rule": "the read's gross t must exceed the third-largest pool "
                      "gross t, so at most 2 of 60 equal or exceed it",
        "n_worlds": int(gm.size),
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


# ------------------------------------------------------- 2.3 the tracked pool
def write_pool_v1(man: dict) -> dict:
    """
    The tracked pool: per world, mean / sd / t / Sharpe for gross, net BASE and
    net STRESS, turnover, and the scratch manifest's sha256. NO timing field, so
    the file is byte-reproducible; the scratch manifest is evidence, not an input.
    """
    worlds = []
    for w in man["worlds"]:
        r = json.loads((P.pool_dir() / ("world_%02d.json" % w["k"])).read_text())
        st = r["stats"]
        worlds.append({
            "k": int(w["k"]),
            "gross": {kk: st["gross"][kk] for kk in ("mean", "sd", "t", "sharpe")},
            "net_base": {kk: st["net_base"][kk] for kk in ("mean", "sd", "t", "sharpe")},
            "net_stress": {kk: st["net_stress"][kk] for kk in ("mean", "sd", "t", "sharpe")},
            "turnover_mean": st["turnover_mean"],
            "months": int(st["months"]),
            "label_rows": int(r["label_rows"]),
        })
    out = {
        "version": POOL_V1_VERSION,
        "n_worlds": len(worlds),
        "scratch_manifest_sha256": G.sha256_file(P.pool_manifest()),
        "feature_sha256": man["feature_sha256"],
        "banked_panel_sha256": man["banked_panel_sha256"],
        "env": TW.env_versions(),
        "worlds": worlds,
    }
    P.canon_dir().mkdir(parents=True, exist_ok=True)
    P.pool_v1().write_text(json.dumps(out, indent=2, default=_json))
    return out


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
    pv = write_pool_v1(man)
    log("tracked pool written: %s (%d worlds)" % (P.pool_v1().name, pv["n_worlds"]))
    pl = run_plants(feats, allpm, man)
    fl = floors(pl)
    br = bars(pv["worlds"])
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
        "known_null": kn, "plants": pl, "floors": fl, "bars": br,
        "determinism": det,
        "pool_v1_sha256": G.sha256_file(P.pool_v1()),
        "scratch_manifest_sha256": pv["scratch_manifest_sha256"],
        "fit_diagnostics_world_01": fit_rows,
        "feature_non_nan_share": fmeta["non_nan_share"],
        "feature_sha256": man["feature_sha256"],
        "banked_panel_sha256": man["banked_panel_sha256"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    P.canon_dir().mkdir(parents=True, exist_ok=True)
    P.controls_v2().write_text(json.dumps(out, indent=2, default=_json))
    log("controls written: %s sha %s"
        % (P.controls_v2().name, G.sha256_file(P.controls_v2())))
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def _md_report(res: dict) -> str:
    v, k1a, k1b, k2 = res["verdict"], res["key1a"], res["key1b"], res["key2"]
    rp, st = res["reported"], res["reported"]["series_statistics"]
    yr = "\n".join("| %d | %d | %+.3f | %+.3f |"
                    % (y["year"], y["months"], y["gross_mean_bps"],
                       y["net_base_mean_bps"]) for y in rp["by_year"])
    return f"""# Canon read 3 — the machine on the stock panel — {v}

The third and last registered read of the Canon campaign (D-CANON-CHARTER part
3), taken once, under seal `{res['registration_sha'][:16]}...`, at {res['utc']}.
Statistics only; no vendor price appears in this file.

- Verdict: **{v}**
- Window: {res['window']}, {rp['months_scored']} scored months, vintage {res['vintage']}
- Pins verified: {res['pins_checked']} artifacts.

## Key 1 — information, on GROSS, both parts

(a) stationary bootstrap of months, mean block {res['voting']['mean_block']},
R {res['voting']['R']}, seed {res['voting']['seed']}, on the gross series less the
median gross mean of the 60 worlds.

| | value |
|---|---|
| null centre (pool median gross mean) | {k1a['null_centre']:+.8f} ({k1a['null_centre']*1e4:+.4f} bps) |
| observed gross mean | {st['gross']['mean']:+.8f} ({st['gross']['mean_bps']:+.4f} bps) |
| p, one-sided | **{k1a['p']:.4f}** |
| level | {k1a['pass'] and 'p <= 0.05' or 'p > 0.05'} |
| **key 1(a)** | **{'PASS' if k1a['pass'] else 'FAIL'}** |

(b) the gross t against the 60 worlds' gross t.

| | value |
|---|---|
| observed gross t | {k1b['gross_t']:+.4f} |
| worlds equalling or exceeding it | **{k1b['exceedances']}** of {k1b['n_worlds']} |
| allowed | at most {k1b['max_allowed']} |
| p = (1 + count) / 61 | {k1b['p']:.4f} |
| **key 1(b)** | **{'PASS' if k1b['pass'] else 'FAIL'}** |

**Key 1 = {'PASS' if (k1a['pass'] and k1b['pass']) else 'FAIL'}**

## Key 2 — materiality, on NET BASE

| | value |
|---|---|
| net BASE annualised Sharpe | {k2['net_base_annualised_sharpe']:+.4f} |
| bar | {k2['bar']:.2f} |
| **key 2** | **{'PASS' if k2['pass'] else 'FAIL'}** |

## VERDICT: {v}

{res['verdict_meaning']}

## Reported, not voted

| series | n | mean (bps) | sd | t | annualised Sharpe |
|---|---|---|---|---|---|
| gross | {st['gross']['n']} | {st['gross']['mean_bps']:+.4f} | {st['gross']['sd']:.6f} | {st['gross']['t']:+.4f} | {st['gross']['sharpe']:+.4f} |
| net BASE | {st['net_base']['n']} | {st['net_base']['mean_bps']:+.4f} | {st['net_base']['sd']:.6f} | {st['net_base']['t']:+.4f} | {st['net_base']['sharpe']:+.4f} |
| net STRESS | {st['net_stress']['n']} | {st['net_stress']['mean_bps']:+.4f} | {st['net_stress']['sd']:.6f} | {st['net_stress']['t']:+.4f} | {st['net_stress']['sharpe']:+.4f} |

- Mean monthly traded notional (turnover): {rp['turnover_mean']:.4f}
- Break-even cost, borrow ignored: **{rp['break_even_bps_per_side']:+.4f} bps a side**
  (base is 5.0, stress 15.0)
- Bootstrap p of the net BASE series against zero: {rp['net_base_vs_zero']['p']:.4f}
- Months scored {rp['months_scored']}; skipped rank-deficient
  {rp['months_skipped_rank_deficient']}; skipped too small
  {rp['months_skipped_too_small']}
- Mean cross-section {rp['cross_section_mean']:.1f} of 1,000 (min {rp['cross_section_min']})

| year | months | gross mean (bps) | net BASE mean (bps) |
|---|---|---|---|
{yr}

## What this does not establish

Half B is unopened and the holdout sealed. Costs are provisional constants and no
certified equity cost row exists, so nothing here is a money test (D-P01-CLOSE).
The limits of the sealed registration apply unchanged.
"""


def stage_observed(registration: str, sha: str) -> int:
    """
    The read, taken once. Order: refuse if spent; verify the seal, the pins and
    the receipt; take the lock; write the spent marker; only then build world 0.
    """
    G.refuse_if_spent()
    opened = G.open_the_gate(registration, sha)
    reg = opened["registration"]
    head = G.git_head()
    log("gate open: seal %s, %d pins, receipt %s"
        % (reg.sha256[:16], opened["pins"]["pins_checked"],
           Path(opened["receipt"]["receipt"]).name))

    with G.single_run_lock():
        G.refuse_if_spent()                      # again, inside the lock
        spent = G.write_spent_marker(reg.sha256, head)
        log("SPENT marker written: %s" % spent.name)

        # The 60 world statistics come from the TRACKED pool and nowhere else.
        pool = json.loads(P.pool_v1().read_text())
        if pool["n_worlds"] != W.N_WORLDS:
            raise AssertionError("the tracked pool holds %d worlds, not %d"
                                 % (pool["n_worlds"], W.N_WORLDS))
        gross_means = np.array([w["gross"]["mean"] for w in pool["worlds"]],
                               dtype=np.float64)
        gross_ts = np.array([w["gross"]["t"] for w in pool["worlds"]],
                            dtype=np.float64)

        feats, allpm = _load()
        r = M.run_world(feats, 0, registration=reg, all_permatickers=allpm)
        series = {k: np.asarray(v, dtype=(np.int64 if k == "months" else np.float64))
                  for k, v in r["series"].items()}
        st = r["stats"]

        k1a = M.key1a(series["gross"], float(np.median(gross_means)))
        k1b = M.key1b(st["gross"]["t"], gross_ts)
        k2 = M.key2(st["net_base"]["sharpe"])
        verdict = M.verdict(k1a, k1b, k2)
        rep = M.full_report(series, st, r["book_diag"])

        meanings = {
            "NOTHING":
                "No information before costs at the 5 per cent level on 132 "
                "months. It rules out an edge at or above the key 1 floor and "
                "not a smaller one, which at this turnover would not clear the "
                "net bar in any case.",
            "SIGNIFICANT_IMMATERIAL":
                "Information before costs, at a level the null reaches one time "
                "in twenty, and a net Sharpe under 0.5 at base costs: an edge "
                "this book cannot keep.",
            "SURVIVOR":
                "A candidate, not a finding. It earns one confirmation on half B "
                "under a new registration, after a half-B guard is built; the "
                "holdout stays sealed; costs are provisional, so no money test "
                "follows without a certified equity cost row (D-P01-CLOSE).",
        }
        res = {
            "look": LEDGER_BATCH_ID, "stage": "observed", "version": VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "head": head,
            "registration": str(Path(registration).name),
            "registration_sha": reg.sha256,
            "pins_checked": opened["pins"]["pins_checked"],
            "pins": opened["pins"]["pins"],
            "receipt": opened["receipt"],
            "spent_marker": str(spent.name),
            "vintage": LEDGER_VINTAGE, "signal": LEDGER_SIGNAL,
            "window": LEDGER_UNIVERSE,
            "env": TW.env_versions(),
            "voting": {"mean_block": M.MEAN_BLOCK, "R": M.R_VOTING,
                       "seed": M.SEED_VOTING, "level": M.LEVEL},
            "pool_source": {"file": P.pool_v1().name,
                            "sha256": G.sha256_file(P.pool_v1()),
                            "n_worlds": pool["n_worlds"],
                            "median_gross_mean": float(np.median(gross_means)),
                            "third_largest_gross_t":
                                float(np.sort(gross_ts)[::-1][2])},
            "key1a": k1a, "key1b": k1b, "key2": k2,
            "key1": bool(k1a["pass"] and k1b["pass"]),
            "verdict": verdict,
            "verdict_meaning": meanings[verdict],
            "reported": rep,
            "series": {k: series[k].tolist() for k in series},
            "fit_diag": r["fit_diag"],
            "label_rows": r["label_rows"],
        }
        P.reports_dir().mkdir(parents=True, exist_ok=True)
        P.read_result().write_text(json.dumps(res, indent=2, default=_json))
        P.read_report().write_text(_md_report(res))

        # statuses read back from disk, not assumed
        disk = {
            "result": {"path": str(P.read_result().relative_to(P.repo())),
                       "exists": P.read_result().exists(),
                       "bytes": P.read_result().stat().st_size,
                       "sha256": G.sha256_file(P.read_result())},
            "report": {"path": str(P.read_report().relative_to(P.repo())),
                       "exists": P.read_report().exists(),
                       "bytes": P.read_report().stat().st_size,
                       "sha256": G.sha256_file(P.read_report())},
            "spent": {"path": str(spent.relative_to(P.repo())),
                      "exists": spent.exists(),
                      "sha256": G.sha256_file(spent)},
        }
        ledger = append_ledger_row(reg.sha256, st, k1a, verdict)
        print(json.dumps({"verdict": verdict, "key1a": k1a, "key1b": k1b,
                          "key2": k2, "reported": rep, "on_disk": disk,
                          "ledger": ledger}, indent=2, default=_json))
    return 0


def append_ledger_row(reg_sha: str, st: dict, k1a: dict, verdict: str) -> dict:
    """One row by the repo's own chain definition. Verified before and after."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ledger_chain as lc
    led = P.ledger()
    ok, n, head, bad = lc.verify(led)
    if not ok:
        raise AssertionError("the ledger chain is broken before the append: %r" % bad)
    harness_sha = G.sha256_file(P.repo() / "tools" / "canon_r3_machine.py")
    vals = [LEDGER_BATCH_ID, time.strftime("%Y%m%d"), LEDGER_VINTAGE,
            reg_sha[:16], harness_sha[:16], LEDGER_SIGNAL, LEDGER_UNIVERSE,
            M.N_SCORED_MONTHS, "%.4f" % st["net_base"]["mean_bps"],
            "%.4f" % k1a["p"], verdict]
    rs = lc.row_sha(head, vals)
    with open(led, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    ok2, n2, head2, bad2 = lc.verify(led)
    if not ok2:
        raise AssertionError("the ledger chain broke on the append: %r" % bad2)
    log("ledger %d -> %d rows; head %s" % (n, n2, head2[:16]))
    return {"rows_before": n, "rows_after": n2, "head": head2,
            "chain_ok": bool(ok2), "harness_sha": harness_sha[:16]}


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
