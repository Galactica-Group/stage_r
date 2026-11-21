"""
================================================================================
Script:       canon_r3_machine.py
Version:      1.1.0
Phase:        Stage R — Canon Read 3
Purpose:      The machine of Canon read 3: the within-month neutralised label,
              the eleven-fit purged walk, the harness-v2 slope book, the three
              monthly series and the two keys of the verdict.
Entry Point:  No
Dependencies: numpy, pandas, sklearn; tools/xs_triage.py, tools/canon_r3_*.py
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================

The label is a training target only. It never enters the feature matrix: the
matrix is built from FEATURE_NAMES alone and a test asserts that the label and
the forward return are absent from it.

The walk trains through C-2 and predicts from C, so the last training month's
forward return is realised in C-1 and is known when the fit is made; formation
month C-1 is the embargo.
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_r3_features as F  # noqa: E402
import xs_triage as xt  # noqa: E402

from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

VERSION = "1.1.0"

MODEL_KW = dict(
    loss="squared_error", max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
    min_samples_leaf=200, l2_regularization=10.0, max_features=0.5,
    max_bins=255, early_stopping=False,
)
SEEDS = (20261121, 20261122, 20261123)
N_THREADS = 8

FIRST_CUT = "2001-12"
LAST_CUT = "2011-12"
N_CUTS = 11
FIRST_SCORED = "2001-12"
LAST_SCORED = "2012-11"
N_SCORED_MONTHS = 132

MEAN_BLOCK = 3.0
R_VOTING = 10_000
SEED_VOTING = 20261121
LEVEL = 0.05
KEY1B_MAX_EXCEEDANCES = 2
KEY2_SHARPE_BAR = 0.5

CONTROL_BAND = (0.01, 0.10)


def log(msg: str) -> None:
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def cut_months() -> list[int]:
    o = pd.Period(FIRST_CUT, freq="M").ordinal
    return [o + 12 * i for i in range(N_CUTS)]


def scored_months() -> list[int]:
    a = pd.Period(FIRST_SCORED, freq="M").ordinal
    b = pd.Period(LAST_SCORED, freq="M").ordinal
    return list(range(a, b + 1))


# ----------------------------------------------------------------------- label
def make_label(world: pd.DataFrame, month_col: str = "month") -> np.ndarray:
    """
    The world's fwd_ret residualised within the month on [1, sector dummies,
    beta, log dv] by xs_triage.residual_next_month_return, then turned into a
    within-month normal score. Rows with no defined residual get NaN and are
    not trained on.
    """
    resid = xt.residual_next_month_return(world, month_col=month_col,
                                          fwd_col="fwd_ret",
                                          universe_col="in_universe")
    y = np.full(len(world), np.nan)
    pos = {ix: i for i, ix in enumerate(world.index)}
    r = pd.Series(np.nan, index=world.index, dtype=np.float64)
    r.loc[resid.index] = resid.to_numpy(dtype=np.float64)
    tmp = pd.DataFrame({"m": world[month_col].to_numpy(),
                        "r": r.to_numpy(dtype=np.float64)}, index=world.index)
    for _, g in tmp.groupby("m", sort=True, observed=True):
        z = xt.rank_to_normal_z(g["r"].to_numpy(dtype=np.float64), "HIGH")
        for ix, v in zip(g.index, z):
            y[pos[ix]] = v
    return y


# ------------------------------------------------------------------- the walk
def walk_predict(world: pd.DataFrame, y: np.ndarray,
                 feature_names: list[str] | None = None,
                 month_col: str = "month") -> tuple[np.ndarray, list[dict]]:
    """
    Eleven fits, three seeds each, the prediction their mean. Returns a
    prediction array aligned to `world`'s rows (NaN outside the scored months)
    and one diagnostic record per cut.
    """
    feature_names = list(F.FEATURE_NAMES if feature_names is None else feature_names)
    X = world[feature_names].to_numpy(dtype=np.float64)
    m = world[month_col].to_numpy(dtype=np.int64)
    pred = np.full(len(world), np.nan)
    diags = []
    for C in cut_months():
        tr = (m >= pd.Period(F.FIRST_FORMATION_MONTH, freq="M").ordinal) & \
             (m <= C - 2) & np.isfinite(y)
        te = (m >= C) & (m <= C + 11)
        if tr.sum() < 1000 or te.sum() == 0:
            diags.append({"cut": str(pd.Period(ordinal=int(C), freq="M")),
                          "train_rows": int(tr.sum()), "score_rows": int(te.sum()),
                          "fit_seconds": 0.0, "skipped": True})
            continue
        t0 = time.time()
        acc = np.zeros(int(te.sum()))
        for s in SEEDS:
            mdl = HistGradientBoostingRegressor(random_state=s, **MODEL_KW)
            mdl.fit(X[tr], y[tr])
            acc += mdl.predict(X[te])
        pred[te] = acc / len(SEEDS)
        diags.append({"cut": str(pd.Period(ordinal=int(C), freq="M")),
                      "train_first": str(pd.Period(ordinal=int(m[tr].min()), freq="M")),
                      "train_last": str(pd.Period(ordinal=int(m[tr].max()), freq="M")),
                      "score_first": str(pd.Period(ordinal=int(m[te].min()), freq="M")),
                      "score_last": str(pd.Period(ordinal=int(m[te].max()), freq="M")),
                      "train_rows": int(tr.sum()), "score_rows": int(te.sum()),
                      "fit_seconds": round(time.time() - t0, 2), "skipped": False})
    return pred, diags


# -------------------------------------------------------------------- the book
def build_book(world: pd.DataFrame, pred: np.ndarray,
               month_col: str = "month") -> tuple[dict, dict]:
    """
    The prediction into harness v2's slope book, gross and net at both cost
    specs, over the scored months only. build_series_v2 applies
    rank_to_normal_z(signal, 'HIGH') itself, which is clause 5.1's z.
    """
    sm = set(scored_months())
    w = world.copy()
    w["_pred"] = pred
    w = w.loc[w[month_col].isin(sm)].copy()
    s_base, diag = xt.build_series_v2(w, "_pred", "HIGH", "LS", xt.COST_BASE)
    s_stress, _ = xt.build_series_v2(w, "_pred", "HIGH", "LS", xt.COST_STRESS)
    fb, fs = s_base.frame(), s_stress.frame()
    if not fb["month"].equals(fs["month"]):
        raise AssertionError("the two cost specs produced different month sets")
    series = {
        "months": fb["month"].to_numpy(dtype=np.int64),
        "gross": fb["gross"].to_numpy(dtype=np.float64),
        "net_base": fb["net"].to_numpy(dtype=np.float64),
        "net_stress": fs["net"].to_numpy(dtype=np.float64),
        "turnover": fb["turnover"].to_numpy(dtype=np.float64),
    }
    return series, diag


def stats(x: np.ndarray) -> dict:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    n = a.size
    if n < 2:
        return {"n": int(n), "mean": float("nan"), "sd": float("nan"),
                "t": float("nan"), "sharpe": float("nan"), "mean_bps": float("nan")}
    mu, sd = float(a.mean()), float(a.std(ddof=1))
    return {"n": int(n), "mean": mu, "sd": sd, "mean_bps": mu * 1e4,
            "t": (mu / sd * np.sqrt(n)) if sd > 0 else float("nan"),
            "sharpe": (mu / sd * np.sqrt(12.0)) if sd > 0 else float("nan")}


def series_stats(series: dict) -> dict:
    out = {k: stats(series[k]) for k in ("gross", "net_base", "net_stress")}
    out["turnover_mean"] = float(np.mean(series["turnover"]))
    out["months"] = int(series["months"].size)
    return out


# ----------------------------------------------------------------- the verdict
def key1a(gross: np.ndarray, null_centre: float, R: int = R_VOTING,
          seed: int = SEED_VOTING, mean_block: float = MEAN_BLOCK) -> dict:
    """xs_triage's stationary bootstrap on the gross series less the null centre."""
    shifted = np.asarray(gross, dtype=np.float64) - float(null_centre)
    res = xt.block_bootstrap_p(shifted, mean_block=mean_block, R=R, seed=seed)
    res.pop("boots", None)
    res["null_centre"] = float(null_centre)
    res["pass"] = bool(np.isfinite(res["p"]) and res["p"] <= LEVEL)
    return res


def key1b(gross_t: float, world_ts: np.ndarray,
          max_exc: int = KEY1B_MAX_EXCEEDANCES) -> dict:
    w = np.asarray(world_ts, dtype=np.float64)
    w = w[np.isfinite(w)]
    count = int((w >= float(gross_t)).sum())
    return {"gross_t": float(gross_t), "n_worlds": int(w.size),
            "exceedances": count, "p": (1.0 + count) / (w.size + 1.0),
            "max_allowed": int(max_exc), "pass": bool(count <= max_exc)}


def key2(net_base_sharpe: float, bar: float = KEY2_SHARPE_BAR) -> dict:
    return {"net_base_annualised_sharpe": float(net_base_sharpe), "bar": float(bar),
            "pass": bool(np.isfinite(net_base_sharpe) and net_base_sharpe >= bar)}


def verdict(k1a: dict, k1b: dict, k2: dict) -> str:
    k1 = bool(k1a["pass"] and k1b["pass"])
    if not k1:
        return "NOTHING"
    return "SURVIVOR" if k2["pass"] else "SIGNIFICANT_IMMATERIAL"


def break_even_bps_per_side(gross_mean: float, mean_turnover: float) -> float:
    """
    The per-side cost in bps at which this book's gross mean is exactly eaten.
    cost = (bps/1e4) * turnover, so bps = gross_mean * 1e4 / turnover.
    Borrow is ignored, as registered.
    """
    if not np.isfinite(gross_mean) or not np.isfinite(mean_turnover) \
            or mean_turnover <= 0:
        return float("nan")
    return float(gross_mean * 1e4 / mean_turnover)


def net_vs_zero_p(net: np.ndarray, R: int = R_VOTING, seed: int = SEED_VOTING,
                  mean_block: float = MEAN_BLOCK) -> dict:
    """Reported, never voted: the same bootstrap on the net series against zero."""
    res = xt.block_bootstrap_p(np.asarray(net, dtype=np.float64),
                               mean_block=mean_block, R=R, seed=seed)
    res.pop("boots", None)
    return res


def yearly_means(months: np.ndarray, gross: np.ndarray,
                 net_base: np.ndarray) -> list[dict]:
    """Reported, never voted: the mean gross and net BASE of each scored year."""
    yr = np.array([pd.Period(ordinal=int(m), freq="M").year for m in months])
    out = []
    for y in sorted(set(int(x) for x in yr)):
        k = yr == y
        out.append({"year": int(y), "months": int(k.sum()),
                    "gross_mean_bps": float(np.mean(gross[k])) * 1e4,
                    "net_base_mean_bps": float(np.mean(net_base[k])) * 1e4})
    return out


def full_report(series: dict, st: dict, book_diag: dict) -> dict:
    """Everything clause 2.4 reports beside the verdict. None of it votes."""
    return {
        "series_statistics": {k: st[k] for k in ("gross", "net_base", "net_stress")},
        "turnover_mean": st["turnover_mean"],
        "break_even_bps_per_side": break_even_bps_per_side(
            st["gross"]["mean"], st["turnover_mean"]),
        "net_base_vs_zero": net_vs_zero_p(series["net_base"]),
        "by_year": yearly_means(series["months"], series["gross"],
                                series["net_base"]),
        "months_scored": int(series["months"].size),
        "months_skipped_rank_deficient": book_diag["months_skipped_rank_deficient"],
        "months_skipped_too_small": book_diag["months_skipped_too_small"],
        "cross_section_mean": book_diag["cross_section_mean"],
        "cross_section_min": book_diag["cross_section_min"],
    }


def analytic_sharpe_for_power(n_months: int = N_SCORED_MONTHS) -> float:
    """Annualised gross Sharpe detectable at 80% power, one-sided alpha 0.05."""
    return float((xt.Z_95 + xt.Z_80) / np.sqrt(n_months) * np.sqrt(12.0))


# ------------------------------------------------------------- one world, end to end
def run_world(features: pd.DataFrame, k: int, registration=None,
              all_permatickers=None, plant_rho: float | None = None,
              plant_rho_index: int | None = None) -> dict:
    """Targets -> label -> walk -> book -> statistics, for one world."""
    import canon_r3_world as W
    t0 = time.time()
    world = W.build_world(features, k, registration=registration,
                          all_permatickers=all_permatickers)
    if plant_rho is not None:
        world = W.plant_feature(world, k, plant_rho, plant_rho_index)
    y = make_label(world)
    pred, diags = walk_predict(world, y)
    series, bdiag = build_book(world, pred)
    st = series_stats(series)
    return {
        "k": int(k), "plant_rho": plant_rho,
        "series": {kk: series[kk].tolist() for kk in series},
        "stats": st, "book_diag": bdiag, "fit_diag": diags,
        "label_rows": int(np.isfinite(y).sum()),
        "wall_seconds": round(time.time() - t0, 1),
    }
