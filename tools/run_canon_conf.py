"""
================================================================================
Script:       run_canon_conf.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      --stage controls runs T3, then the pool of 60, the known null, the
              plants, the floors and the bars; --stage observed takes the read,
              and only behind the gate.
Entry Point:  Yes — python tools/run_canon_conf.py --stage controls
                     python tools/run_canon_conf.py --stage observed
                         --registration <path> --sha <sha256>
Dependencies: numpy, pandas, sklearn; tools/canon_conf_*.py, tools/xs_triage.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
================================================================================

The controls stage never constructs world 0: it calls run_world with k >= 1 only,
and the pipeline refuses k == 0 without a verified registration and always on
read 3's cuts. T3 runs first and the stage refuses on a mismatch.
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
import canon_conf_gate as G  # noqa: E402
import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402
import canon_conf_seal as SEAL  # noqa: E402
import canon_conf_stopped as S  # noqa: E402
import canon_conf_tripwires as TW  # noqa: E402
import canon_r3_machine as M  # noqa: E402
import xs_triage as xt  # noqa: E402

VERSION = "1.0.0"

POOL_V1_VERSION = "1.0.0"
PLANT_WORLDS = tuple(range(1, 13))
PLANT_GRID = ((0.005, 0), (0.010, 1), (0.020, 2), (0.025, 3), (0.030, 4),
              (0.035, 5), (0.040, 6), (0.080, 7))
PLANT_RHOS = tuple(r for r, _ in PLANT_GRID)
PLANT_PASS_MIN = 10

CONTROL_BAND = (0.01, 0.10)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


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


def _load():
    feats = pd.read_parquet(P.feature_file()).reset_index(drop=True)
    panel = pd.read_parquet(P.panel_file())
    return feats, panel["permaticker"].unique()


def key1a(gross, centre):
    return M.key1a(gross, centre, R=M.R_VOTING, seed=PL.CONF_SEED_VOTING,
                   mean_block=M.MEAN_BLOCK)


def verdict(k1a, k1b, k2) -> str:
    if k1a["pass"] and k1b["pass"] and k2["pass"]:
        return "CONFIRMED"
    return "NOT_CONFIRMED"


def failed_keys(k1a, k1b, k2) -> list[str]:
    out = []
    if not k1a["pass"]:
        out.append("key1a")
    if not k1b["pass"]:
        out.append("key1b")
    if not k2["pass"]:
        out.append("key2")
    return out


# ------------------------------------------------------------------- 4.1 pool
def run_pool(feats, allpm, worlds=range(1, PL.N_WORLDS + 1)) -> dict:
    P.pool_dir().mkdir(parents=True, exist_ok=True)
    rows = []
    for k in worlds:
        r = PL.run_world(feats, int(k), PL.CONF_CUTS, PL.CONF_SCORED,
                         seed_base=PL.CONF_SEED_BASE, all_permatickers=allpm,
                         stopped=True)
        (P.pool_dir() / ("world_%02d.json" % k)).write_text(
            json.dumps(r, indent=1, default=_json))
        st, sb = r["stats"], r["stopped_bound"]
        rows.append({
            "k": int(k),
            "gross": {f: st["gross"][f] for f in ("mean", "sd", "t", "sharpe")},
            "net_base": {f: st["net_base"][f] for f in ("mean", "sd", "t", "sharpe")},
            "net_stress": {f: st["net_stress"][f] for f in ("mean", "sd", "t", "sharpe")},
            "turnover_mean": st["turnover_mean"], "months": int(st["months"]),
            "label_rows": int(r["label_rows"]),
            "s1": sb["s1_summary"],
            "s2": {kk: {f: vv[f] for f in ("L", "mean", "sharpe")}
                   for kk, vv in sb["s2"].items()},
        })
        log("world %02d  gross t %+7.3f  net_base Sharpe %+6.3f  "
            "stopped pos-wt mean %.5f  %.1fs"
            % (k, st["gross"]["t"], st["net_base"]["sharpe"],
               sb["s1_summary"]["weight_pos_mean"], r["wall_seconds"]))
    man = {"version": POOL_V1_VERSION, "n_worlds": len(rows), "worlds": rows,
           "feature_sha256": G.sha256_file(P.feature_file()),
           "panel_sha256": G.sha256_file(P.panel_file()),
           "env": _env()}
    P.pool_manifest().write_text(json.dumps(man, indent=2, default=_json))
    P.canon_dir().mkdir(parents=True, exist_ok=True)
    P.pool_v1().write_text(json.dumps(man, indent=2, default=_json))
    return man


def _env() -> dict:
    import pyarrow, sklearn
    return {"python": sys.version.split()[0], "numpy": np.__version__,
            "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
            "scikit-learn": sklearn.__version__}


# ------------------------------------------------------------ 4.2 known null
def known_null(man: dict) -> dict:
    gm = np.array([w["gross"]["mean"] for w in man["worlds"]], dtype=np.float64)
    gt = np.array([w["gross"]["t"] for w in man["worlds"]], dtype=np.float64)
    res, whole = [], []
    for i, w in enumerate(man["worlds"]):
        others = np.delete(gm, i)
        g = np.array(json.loads(
            (P.pool_dir() / ("world_%02d.json" % w["k"])).read_text()
        )["series"]["gross"], dtype=np.float64)
        a = key1a(g, float(np.median(others)))
        b = M.key1b(w["gross"]["t"], np.delete(gt, i))
        k2 = M.key2(w["net_base"]["sharpe"])
        res.append({"k": w["k"], "p": a["p"], "pass": a["pass"]})
        whole.append(verdict(a, b, k2))
    share = float(np.mean([r["pass"] for r in res]))
    ns = np.array([w["net_base"]["sharpe"] for w in man["worlds"]], dtype=np.float64)
    falls = {}
    for key in sorted(man["worlds"][0]["s2"]):
        d = np.array([w["net_base"]["sharpe"] - w["s2"][key]["sharpe"]
                      for w in man["worlds"]], dtype=np.float64)
        falls[key] = {"L": man["worlds"][0]["s2"][key]["L"],
                      "median_fall_in_net_base_sharpe": float(np.median(d)),
                      "max_fall": float(d.max())}
    return {
        "n_worlds": len(res), "alpha": M.LEVEL,
        "share_significant": share, "accept_band": list(CONTROL_BAND),
        "pass": bool(xt.control_verdict(share, CONTROL_BAND)),
        "per_world": res,
        "gross_t": {"median": float(np.median(gt)), "sd": float(gt.std(ddof=1)),
                    "min": float(gt.min()), "max": float(gt.max())},
        "net_base_sharpe": {"median": float(np.median(ns)),
                            "sd": float(ns.std(ddof=1)),
                            "min": float(ns.min()), "max": float(ns.max())},
        "verdict_counts": {v: int(sum(1 for x in whole if x == v))
                           for v in sorted(set(whole))},
        "worlds_called_confirmed": int(sum(1 for x in whole if x == "CONFIRMED")),
        "s2_median_fall": falls,
    }


# ---------------------------------------------------------------- 4.3 plants
def run_plants(feats, allpm, man: dict) -> dict:
    gm = {w["k"]: w["gross"]["mean"] for w in man["worlds"]}
    gt = {w["k"]: w["gross"]["t"] for w in man["worlds"]}
    out = {"rhos": list(PLANT_RHOS), "worlds": list(PLANT_WORLDS),
           "grid": [{"rho": r, "seed_index": j} for r, j in PLANT_GRID],
           "by_rho": {}}
    for rho, j in PLANT_GRID:
        recs = []
        for k in PLANT_WORLDS:
            r = PL.run_world(feats, int(k), PL.CONF_CUTS, PL.CONF_SCORED,
                             seed_base=PL.CONF_SEED_BASE, all_permatickers=allpm,
                             plant_rho=float(rho), plant_rho_index=j)
            g = np.array(r["series"]["gross"], dtype=np.float64)
            om = np.array([gm[x] for x in gm if x != k], dtype=np.float64)
            ot = np.array([gt[x] for x in gt if x != k], dtype=np.float64)
            a = key1a(g, float(np.median(om)))
            b = M.key1b(r["stats"]["gross"]["t"], ot)
            k2 = M.key2(r["stats"]["net_base"]["sharpe"])
            recs.append({"k": int(k), "p_1a": a["p"], "pass_1a": a["pass"],
                         "exceedances_1b": b["exceedances"], "pass_1b": b["pass"],
                         "pass_key1": bool(a["pass"] and b["pass"]),
                         "pass_key2": bool(k2["pass"]),
                         "verdict": verdict(a, b, k2),
                         "gross_sharpe": r["stats"]["gross"]["sharpe"],
                         "net_base_sharpe": r["stats"]["net_base"]["sharpe"],
                         "turnover_mean": r["stats"]["turnover_mean"]})
            log("plant rho %.3f world %02d  1a %s 1b %s k2 %s  gross Shp %+6.3f  %s"
                % (rho, k, recs[-1]["pass_1a"], recs[-1]["pass_1b"],
                   recs[-1]["pass_key2"], recs[-1]["gross_sharpe"],
                   recs[-1]["verdict"]))
        n = len(recs)
        out["by_rho"]["%.3f" % rho] = {
            "rho": float(rho), "seed_index": j, "n_worlds": n,
            "share_pass_1a": float(np.mean([r["pass_1a"] for r in recs])),
            "share_pass_1b": float(np.mean([r["pass_1b"] for r in recs])),
            "share_pass_key1": float(np.mean([r["pass_key1"] for r in recs])),
            "n_pass_key1": int(sum(r["pass_key1"] for r in recs)),
            "share_pass_key2": float(np.mean([r["pass_key2"] for r in recs])),
            "share_confirmed": float(np.mean([r["verdict"] == "CONFIRMED"
                                              for r in recs])),
            "n_confirmed": int(sum(r["verdict"] == "CONFIRMED" for r in recs)),
            "median_gross_sharpe": float(np.median([r["gross_sharpe"] for r in recs])),
            "median_net_base_sharpe": float(
                np.median([r["net_base_sharpe"] for r in recs])),
            "median_turnover": float(np.median([r["turnover_mean"] for r in recs])),
            "per_world": recs,
        }
    return out


# ---------------------------------------------------------------- 4.4 floors
def floors(plants: dict) -> dict:
    def find(pred):
        for rho in sorted(PLANT_RHOS):
            b = plants["by_rho"]["%.3f" % rho]
            if pred(b):
                return b
        return None

    k1 = find(lambda b: b["n_pass_key1"] >= PLANT_PASS_MIN)
    wh = find(lambda b: b["n_confirmed"] >= PLANT_PASS_MIN)

    def pack(b, label, rule):
        if b is None:
            return {"label": label, "rule": rule, "detected": False, "rho": None}
        return {"label": label, "rule": rule, "detected": True, "rho": b["rho"],
                "n_pass_key1": b["n_pass_key1"], "n_confirmed": b["n_confirmed"],
                "median_gross_sharpe": b["median_gross_sharpe"],
                "median_net_base_sharpe": b["median_net_base_sharpe"]}

    return {"grid": sorted(PLANT_RHOS),
            "key1_floor": pack(k1, "key 1 floor",
                               "smallest rho with key 1 passing in >= %d of 12"
                               % PLANT_PASS_MIN),
            "verdict_floor": pack(wh, "verdict floor",
                                  "smallest rho with CONFIRMED in >= %d of 12"
                                  % PLANT_PASS_MIN),
            "analytic_gross_sharpe_80pc_power_96_months":
                M.analytic_sharpe_for_power(PL.CONF_N_SCORED_MONTHS),
            "key2_sharpe_bar": M.KEY2_SHARPE_BAR}


# ------------------------------------------------------------------ 4.5 bars
def bars(pool_worlds) -> dict:
    gm = np.array([w["gross"]["mean"] for w in pool_worlds], dtype=np.float64)
    gt = np.sort(np.array([w["gross"]["t"] for w in pool_worlds],
                          dtype=np.float64))[::-1]
    return {"key1a_centre_pool_median_gross_mean": float(np.median(gm)),
            "key1a_centre_bps": float(np.median(gm)) * 1e4,
            "key1b_third_largest_pool_gross_t": float(gt[2]),
            "key1b_rule": "the read's gross t must exceed the third-largest pool "
                          "gross t, so at most 2 of 60 equal or exceed it",
            "n_worlds": int(gm.size)}


# ---------------------------------------------------------- 4.6 determinism
def determinism(feats, allpm, k: int = 1) -> dict:
    a = PL.run_world(feats, k, PL.CONF_CUTS, PL.CONF_SCORED,
                     seed_base=PL.CONF_SEED_BASE, all_permatickers=allpm)
    b = PL.run_world(feats, k, PL.CONF_CUTS, PL.CONF_SCORED,
                     seed_base=PL.CONF_SEED_BASE, all_permatickers=allpm)
    same = {}
    for s in ("gross", "net_base", "net_stress", "turnover"):
        x = np.asarray(a["series"][s], dtype=np.float64)
        y = np.asarray(b["series"][s], dtype=np.float64)
        same[s] = bool(x.tobytes() == y.tobytes())
    return {"k": int(k), "bit_identical": all(same.values()), "per_series": same,
            "cross_section_mean": a["book_diag"]["cross_section_mean"],
            "cross_section_min": a["book_diag"]["cross_section_min"]}


# ----------------------------------------------------------------- the stages
def stage_controls() -> int:
    t0 = time.time()
    log("T3 first: the pipeline must reproduce banked half-A worlds exactly")
    t3 = TW.t3()
    if not t3["identical"]:
        raise AssertionError("T3 mismatch, the controls stage refuses: %s"
                             % json.dumps(t3, default=_json)[:400])
    log("T3 OK: %d worlds, 0 mismatches" % len(t3["per_world"]))

    feats, allpm = _load()
    log("features %s; env %s" % (str(feats.shape), json.dumps(_env())))
    man = run_pool(feats, allpm)
    log("pool of %d worlds written; pool_v1 sha %s"
        % (man["n_worlds"], G.sha256_file(P.pool_v1())[:16]))
    kn = known_null(man)
    log("known null: share %.4f band %s pass %s"
        % (kn["share_significant"], kn["accept_band"], kn["pass"]))
    pl = run_plants(feats, allpm, man)
    fl = floors(pl)
    br = bars(man["worlds"])
    det = determinism(feats, allpm, 1)
    s3 = json.loads((P.derived() / "s3_counts.json").read_text())
    fmeta = json.loads(P.features_meta().read_text())
    fit_rows = json.loads((P.pool_dir() / "world_01.json").read_text())["fit_diag"]

    out = {
        "stage": "controls", "look": SEAL.LOOK, "version": VERSION,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": _env(), "t3": t3,
        "design": {
            "cuts": [str(pd.Period(ordinal=c, freq="M")) for c in PL.CONF_CUTS],
            "scored_months": [PL.CONF_FIRST_SCORED, PL.CONF_LAST_SCORED,
                              PL.CONF_N_SCORED_MONTHS],
            "world_seed_base": PL.CONF_SEED_BASE,
            "plant_seed_base": PL.CONF_PLANT_SEED_BASE,
            "voting_seed": PL.CONF_SEED_VOTING,
            "model": dict(M.MODEL_KW), "seeds": list(M.SEEDS),
            "key3": "STRUCK by ruling (CONF-1b)",
        },
        "pool": {"n_worlds": man["n_worlds"], "worlds": man["worlds"]},
        "known_null": kn, "plants": pl, "floors": fl, "bars": br,
        "determinism": det,
        "stopped_bound_s3": s3,
        "feature_non_nan_share": fmeta["non_nan_share"],
        "fit_diagnostics_world_01": fit_rows,
        "pool_v1_sha256": G.sha256_file(P.pool_v1()),
        "feature_sha256": man["feature_sha256"],
        "panel_sha256": man["panel_sha256"],
        "wall_seconds": round(time.time() - t0, 1),
    }
    P.canon_dir().mkdir(parents=True, exist_ok=True)
    P.controls_v1().write_text(json.dumps(out, indent=2, default=_json))
    log("controls written: %s sha %s"
        % (P.controls_v1().name, G.sha256_file(P.controls_v1())[:16]))
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def _md_report(res: dict) -> str:
    v, k1a, k1b, k2 = res["verdict"], res["key1a"], res["key1b"], res["key2"]
    rp = res["reported"]
    st = rp["series_statistics"]
    yr = "\n".join("| %d | %d | %+.3f | %+.3f |"
                   % (y["year"], y["months"], y["gross_mean_bps"],
                      y["net_base_mean_bps"]) for y in rp["by_year"])
    return f"""# Canon read 3, the confirmation — {v}

Taken once under seal `{res['registration_sha'][:16]}...` at {res['utc']}.
Statistics only; no vendor value appears in this file.

- Verdict: **{v}**{"" if v == "CONFIRMED" else "  (failed: %s)" % ", ".join(res['failed_keys'])}
- Window: {res['window']}, {rp['months_scored']} scored months, vintage {res['vintage']}
- Pins verified: {res['pins_checked']}

## Key 1(a)

| | value |
|---|---|
| null centre (pool median gross mean) | {k1a['null_centre']:+.10f} ({k1a['null_centre']*1e4:+.4f} bps) |
| observed gross mean | {st['gross']['mean']:+.10f} ({st['gross']['mean_bps']:+.4f} bps) |
| p, one-sided | **{k1a['p']:.4f}** |
| key 1(a) | **{'PASS' if k1a['pass'] else 'FAIL'}** |

## Key 1(b)

| | value |
|---|---|
| observed gross t | {k1b['gross_t']:+.6f} |
| worlds equalling or exceeding it | **{k1b['exceedances']}** of {k1b['n_worlds']} |
| allowed | at most {k1b['max_allowed']} |
| key 1(b) | **{'PASS' if k1b['pass'] else 'FAIL'}** |

## Key 2

| | value |
|---|---|
| net BASE annualised Sharpe | {k2['net_base_annualised_sharpe']:+.4f} |
| bar | {k2['bar']:.2f} |
| key 2 | **{'PASS' if k2['pass'] else 'FAIL'}** |

## VERDICT: {v}

{res['verdict_meaning']}

## Reported, not voted

| series | n | mean (bps) | sd | t | annualised Sharpe | ratio to read 3 |
|---|---|---|---|---|---|---|
| gross | {st['gross']['n']} | {st['gross']['mean_bps']:+.4f} | {st['gross']['sd']:.6f} | {st['gross']['t']:+.4f} | {st['gross']['sharpe']:+.4f} | {rp['ratio_to_read3']['gross_sharpe']:+.3f} |
| net BASE | {st['net_base']['n']} | {st['net_base']['mean_bps']:+.4f} | {st['net_base']['sd']:.6f} | {st['net_base']['t']:+.4f} | {st['net_base']['sharpe']:+.4f} | {rp['ratio_to_read3']['net_base_sharpe']:+.3f} |
| net STRESS | {st['net_stress']['n']} | {st['net_stress']['mean_bps']:+.4f} | {st['net_stress']['sd']:.6f} | {st['net_stress']['t']:+.4f} | {st['net_stress']['sharpe']:+.4f} | {rp['ratio_to_read3']['net_stress_sharpe']:+.3f} |

- Turnover: {rp['turnover_mean']:.4f}
- Break-even cost, borrow ignored: **{rp['break_even_bps_per_side']:+.4f} bps a side**
- Bootstrap p of net BASE against zero: {rp['net_base_vs_zero']['p']:.4f}
- Months scored {rp['months_scored']}; skipped rank {rp['months_skipped_rank_deficient']}; skipped small {rp['months_skipped_too_small']}
- Mean cross-section {rp['cross_section_mean']:.1f} of 1,000 (min {rp['cross_section_min']})

| year | months | gross (bps) | net BASE (bps) |
|---|---|---|---|
{yr}

### The stopped-stock bound, reported and never voted

{json.dumps(rp['stopped_bound']['s1_summary'], indent=2)}

{json.dumps(rp['stopped_bound']['s2'], indent=2)}

## What this does not establish

The 2021 holdout stays sealed. Costs are provisional and no certified equity cost
row exists, so nothing here is a money test (D-P01-CLOSE). The limits of the
registration apply unchanged.
"""


def stage_observed(registration: str, sha: str) -> int:
    G.refuse_if_spent()
    opened = G.open_the_gate(registration, sha)
    reg = opened["registration"]
    head = G.git_head()
    log("gate open: seal %s, %d pins, receipt %s"
        % (reg.sha256[:16], opened["pins"]["pins_checked"],
           Path(opened["receipt"]["receipt"]).name))
    with G.single_run_lock():
        G.refuse_if_spent()
        spent = G.write_spent_marker(reg.sha256, head)
        log("SPENT marker written: %s" % spent.name)

        pool = json.loads(P.pool_v1().read_text())
        if pool["n_worlds"] != PL.N_WORLDS:
            raise AssertionError("the tracked pool holds %d worlds, not %d"
                                 % (pool["n_worlds"], PL.N_WORLDS))
        gm = np.array([w["gross"]["mean"] for w in pool["worlds"]], dtype=np.float64)
        gt = np.array([w["gross"]["t"] for w in pool["worlds"]], dtype=np.float64)

        feats, allpm = _load()
        r = PL.run_world(feats, 0, PL.CONF_CUTS, PL.CONF_SCORED,
                         seed_base=PL.CONF_SEED_BASE, all_permatickers=allpm,
                         stopped=True, registration=reg)
        series = {k: np.asarray(v, dtype=(np.int64 if k == "months" else np.float64))
                  for k, v in r["series"].items()}
        st = r["stats"]
        a = key1a(series["gross"], float(np.median(gm)))
        b = M.key1b(st["gross"]["t"], gt)
        k2 = M.key2(st["net_base"]["sharpe"])
        v = verdict(a, b, k2)
        rep = M.full_report(series, st, r["book_diag"])
        rep["stopped_bound"] = r["stopped_bound"]
        r3 = json.loads(P.r3_result().read_text())["reported"]["series_statistics"]
        rep["ratio_to_read3"] = {
            "gross_sharpe": st["gross"]["sharpe"] / r3["gross"]["sharpe"],
            "net_base_sharpe": st["net_base"]["sharpe"] / r3["net_base"]["sharpe"],
            "net_stress_sharpe": st["net_stress"]["sharpe"] / r3["net_stress"]["sharpe"],
        }
        meanings = {
            "CONFIRMED":
                "The candidate held on eight unseen years at provisional costs. "
                "A research finding, not a money finding; the 2021 holdout stays "
                "sealed for a decision of its own, and no money test follows "
                "without a certified equity cost row (D-P01-CLOSE).",
            "NOT_CONFIRMED":
                "The keys that failed are named. The candidate is closed; half A "
                "stands in the record as a discovery that did not replicate; "
                "nothing is re-specified and re-read on half B.",
        }
        res = {
            "look": SEAL.LOOK, "stage": "observed", "version": VERSION,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "head": head, "registration": str(Path(registration).name),
            "registration_sha": reg.sha256,
            "pins_checked": opened["pins"]["pins_checked"],
            "pins": opened["pins"]["pins"], "receipt": opened["receipt"],
            "spent_marker": str(spent.name),
            "vintage": SEAL.VINTAGE, "signal": SEAL.SIGNAL,
            "window": SEAL.UNIVERSE, "env": _env(),
            "voting": {"mean_block": M.MEAN_BLOCK, "R": M.R_VOTING,
                       "seed": PL.CONF_SEED_VOTING, "level": M.LEVEL},
            "pool_source": {"file": P.pool_v1().name,
                            "sha256": G.sha256_file(P.pool_v1()),
                            "n_worlds": pool["n_worlds"],
                            "median_gross_mean": float(np.median(gm)),
                            "third_largest_gross_t": float(np.sort(gt)[::-1][2])},
            "key1a": a, "key1b": b, "key2": k2,
            "key1": bool(a["pass"] and b["pass"]),
            "verdict": v, "failed_keys": failed_keys(a, b, k2),
            "verdict_meaning": meanings[v], "reported": rep,
            "series": {k: series[k].tolist() for k in series},
            "fit_diag": r["fit_diag"], "label_rows": r["label_rows"],
        }
        P.reports_dir().mkdir(parents=True, exist_ok=True)
        P.read_result().write_text(json.dumps(res, indent=2, default=_json))
        P.read_report().write_text(_md_report(res))
        disk = {kind: {"path": str(p.relative_to(P.repo())), "exists": p.exists(),
                       "bytes": p.stat().st_size, "sha256": G.sha256_file(p)}
                for kind, p in (("result", P.read_result()),
                                ("report", P.read_report()),
                                ("spent", spent))}
        led = append_ledger_row(reg.sha256, st, a, v)
        print(json.dumps({"verdict": v, "failed_keys": res["failed_keys"],
                          "key1a": a, "key1b": b, "key2": k2, "reported": rep,
                          "on_disk": disk, "ledger": led},
                         indent=2, default=_json))
    return 0


def append_ledger_row(reg_sha: str, st: dict, k1a: dict, v: str) -> dict:
    import ledger_chain as lc
    led = P.ledger()
    ok, n, head, bad = lc.verify(led)
    if not ok:
        raise AssertionError("the ledger chain is broken before the append: %r" % bad)
    harness = G.sha256_file(P.repo() / "tools" / "canon_conf_pipeline.py")
    vals = [SEAL.LOOK, time.strftime("%Y%m%d"), SEAL.VINTAGE, reg_sha[:16],
            harness[:16], SEAL.SIGNAL, SEAL.UNIVERSE, PL.CONF_N_SCORED_MONTHS,
            "%.4f" % st["net_base"]["mean_bps"], "%.4f" % k1a["p"], v]
    rs = lc.row_sha(head, vals)
    with open(led, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    ok2, n2, head2, bad2 = lc.verify(led)
    if not ok2:
        raise AssertionError("the ledger chain broke on the append: %r" % bad2)
    log("ledger %d -> %d rows; head %s" % (n, n2, head2[:16]))
    return {"rows_before": n, "rows_after": n2, "head": head2,
            "chain_ok": bool(ok2), "harness_sha": harness[:16]}


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
