"""
================================================================================
Script:       canon_conf_tripwires.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      Tripwires T2 and T3: the banked half-A feature values reproduced
              exactly inside the confirmation's feature file, and the
              window-parameterised pipeline reproducing banked half-A worlds.
Entry Point:  Yes — python tools/canon_conf_tripwires.py t2 | t3
Dependencies: numpy, pandas, pyarrow; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

T1 lives in canon_conf_panel, beside the panel it guards. T2 and T3 are here
because they compare two artifacts rather than build one.
================================================================================
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402

VERSION = "1.0.0"

T2_LAST_MONTH = "2012-11"
T3_WORLDS = (1, 2, 3, 4, 5)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _col_mismatches(a: pd.Series, b: pd.Series) -> int:
    if pd.api.types.is_float_dtype(a):
        x = a.to_numpy(dtype=np.float64)
        y = b.to_numpy(dtype=np.float64)
        return int(((np.isnan(x) != np.isnan(y))
                    | ((~np.isnan(x)) & (x != y))).sum())
    return int((a.to_numpy() != b.to_numpy()).sum())


def t2() -> dict:
    """
    For every month through 2012-11, the confirmation's feature values equal the
    banked half-A feature file's exactly — every row, every one of the 25
    features and every column beside them.
    """
    import canon_conf_features as F
    cut = pd.Period(T2_LAST_MONTH, freq="M").ordinal
    banked = pd.read_parquet(P.r3_feature_file())
    new = pd.read_parquet(P.feature_file())
    b = banked.loc[banked["month"] <= cut].sort_values(
        ["permaticker", "month"], kind="stable").reset_index(drop=True)
    n = new.loc[new["month"] <= cut].sort_values(
        ["permaticker", "month"], kind="stable").reset_index(drop=True)
    res = {"tripwire": "T2", "last_month": T2_LAST_MONTH,
           "banked_rows": int(len(b)), "new_rows": int(len(n)),
           "rows_match": bool(len(b) == len(n)),
           "columns_match": bool(list(b.columns) == list(n.columns)),
           "features_checked": len(F.FEATURE_NAMES), "differences": []}
    if not (res["rows_match"] and res["columns_match"]):
        res["identical"] = False
        res["differences"].append({"what": "shape or columns",
                                   "banked": [len(b), list(b.columns)],
                                   "new": [len(n), list(n.columns)]})
        return res
    for c in b.columns:
        bad = _col_mismatches(b[c], n[c])
        if bad:
            res["differences"].append({"column": c, "mismatched_rows": bad})
    res["identical"] = not res["differences"]
    return res


def t3(worlds=T3_WORLDS) -> dict:
    """
    Given read 3's cuts, its seed base 7000, the banked half-A panel and the
    banked half-A feature file, the window-parameterised pipeline reproduces
    worlds 1 to 5 of registrations/canon_r3/pool_v1.json exactly, on every
    statistic. This is what proves the lift of the window to a parameter changed
    nothing.
    """
    import canon_conf_pipeline as PL
    banked_pool = json.loads(P.r3_pool_v1().read_text())
    by_k = {w["k"]: w for w in banked_pool["worlds"]}
    feats = pd.read_parquet(P.r3_feature_file()).reset_index(drop=True)
    panel = pd.read_parquet(P.r3_banked_panel())
    allpm = panel["permaticker"].unique()
    cuts = PL.R3_CUTS
    scored = PL.R3_SCORED
    out = {"tripwire": "T3", "worlds": list(worlds), "seed_base": PL.R3_SEED_BASE,
           "cuts": [str(pd.Period(ordinal=c, freq="M")) for c in cuts],
           "per_world": [], "mismatches": 0}
    for k in worlds:
        t0 = time.time()
        r = PL.run_world(feats, int(k), cuts=cuts, scored=scored,
                         seed_base=PL.R3_SEED_BASE, all_permatickers=allpm)
        st = r["stats"]
        want = by_k[int(k)]
        diffs = []
        for leg in ("gross", "net_base", "net_stress"):
            for f in ("mean", "sd", "t", "sharpe"):
                g, w = st[leg][f], want[leg][f]
                if g != w:
                    diffs.append({"field": "%s.%s" % (leg, f),
                                  "banked": w, "pipeline": g})
        for f in ("turnover_mean", "months"):
            g = st[f] if f == "turnover_mean" else st["months"]
            if g != want[f]:
                diffs.append({"field": f, "banked": want[f], "pipeline": g})
        out["per_world"].append({"k": int(k), "mismatches": len(diffs),
                                 "diffs": diffs,
                                 "gross_t": st["gross"]["t"],
                                 "banked_gross_t": want["gross"]["t"],
                                 "seconds": round(time.time() - t0, 1)})
        out["mismatches"] += len(diffs)
        log("T3 world %d: %d mismatches (gross t %+.6f vs banked %+.6f)"
            % (k, len(diffs), st["gross"]["t"], want["gross"]["t"]))
    out["identical"] = out["mismatches"] == 0
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    what = (argv[0] if argv else "t2").lower()
    if what == "t2":
        r = t2()
    elif what == "t3":
        r = t3()
    else:
        raise SystemExit("usage: canon_conf_tripwires.py t2 | t3")
    print(json.dumps(r, indent=2, sort_keys=True))
    return 0 if r["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
