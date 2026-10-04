"""
================================================================================
Script:       run_t_r1b.py
Version:      1.0.0
Phase:        Stage R — Cross-Sectional Triage
Purpose:      Run look T-R1b (look 2 of Q4 2026): the five textbook signals
              re-read under harness v2, after v2 is proven on the real panel.
Entry Point:  Yes (python tools/run_t_r1b.py, run detached under nohup)
Dependencies: tools/xs_triage.py, numpy, pandas
Last Sprint:  Sprint R1b — Harness v2 (2026-10-04)
================================================================================

Staged so nothing observed is emitted until v2 has earned the right to certify:
  STAGE 1  build the v1 and v2 series (no statistic emitted)
  STAGE 2  MDE, v1 beside v2, from each construction's own null spread
  STAGE 3  known-null control on v2, gross arm; HALT outside [0.01, 0.10]
  STAGE 4  semi-synthetic positive control on real data; HALT below 0.80
  STAGE 5  the observed five, and only then
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
import xs_triage as xt

VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"
LOGDIR = ROOT / "logs" / "r1b"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"

VINTAGE = "20261004p"
BATCH_ID = "T-R1b"
BAR = 0.005
R_VOTING = 2000
R_CONTROL = 500
N_CONTROL = 200
R_PLANT = 500
N_PLANT = 50
TARGET_PLANT_SHARPE = 0.8
PLANT_DETECTION_FLOOR = 0.80
MEAN_BLOCK = 3.0
SEED_VOTING = 20261004
SEED_CONTROL = 777
SEED_PLANT = 4242

SUBPERIODS = [("1998-2002", "1998-01-01", "2002-12-31"),
              ("2003-2007", "2003-01-01", "2007-12-31"),
              ("2008-2012", "2008-01-01", "2012-12-31")]

LEDGER_FIELDS = ["batch_id", "date", "vintage", "registration_sha", "harness_sha",
                 "signal", "universe", "months", "ls_net_mean_bps", "p_premium",
                 "verdict"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def row_sha(prev: str, values: list) -> str:
    return hashlib.sha256("\x00".join([prev] + [str(v) for v in values]).encode()).hexdigest()


def halt(reason: str, payload: dict) -> int:
    log(f"HALT: {reason}")
    log("Verdicts are WITHHELD and no observed statistic is emitted.")
    (LOGDIR / "t_r1b_HALT.json").write_text(
        json.dumps({"halt": reason, "payload": payload}, indent=2, default=str))
    return 15


def main() -> int:
    t0 = time.time()
    log(f"run_t_r1b v{VERSION}; harness v{xt.VERSION}")
    windows = json.loads((ROOT / "config" / "windows.json").read_text())
    reg_sha = sha_file(LOGDIR / "registration_v2.json")
    add_sha = sha_file(LOGDIR / "registration_v2a_addendum.json")
    harness_sha = sha_file(ROOT / "tools" / "xs_triage.py")
    log(f"registration v2 sha {reg_sha}")
    log(f"addendum v2a sha    {add_sha}")
    log(f"harness sha         {harness_sha}")

    panel = pd.read_parquet(PANEL)
    panel_sha = sha_file(PANEL)
    xt.assert_no_rows_at_or_after(panel["month_ts"],
                                  windows["half_a"]["daily_row_exclusive_max"],
                                  "panel load")
    held = panel.loc[panel["in_universe"].astype(bool)]
    log(f"panel sha {panel_sha}")
    log(f"panel rows {len(panel):,}; in-universe {len(held):,}; max month "
        f"{held['month_ts'].max().date()} "
        f"(cutoff {windows['half_a']['daily_row_exclusive_max']}, exclusive)")

    signals = list(xt.REGISTERED_SIGNALS)

    # ------------------------------------------------- STAGE 1: series, silent
    log("STAGE 1: build the v1 and v2 series (no statistic emitted yet)")
    v1s: dict[str, dict] = {}
    v2s: dict[str, dict] = {}
    diags: dict[str, dict] = {}
    for n in signals:
        d = xt.REGISTERED_SIGNALS[n]
        v1s[n] = {
            "LS_BASE": xt.build_series(panel, n, d, "LS", xt.COST_BASE).frame(),
        }
        s_ls_b, dg = xt.build_series_v2(panel, n, d, "LS", xt.COST_BASE)
        s_ls_s, _ = xt.build_series_v2(panel, n, d, "LS", xt.COST_STRESS)
        s_lo_b, _ = xt.build_series_v2(panel, n, d, "LONG_ONLY", xt.COST_BASE)
        v2s[n] = {"LS_BASE": s_ls_b.frame(), "LS_STRESS": s_ls_s.frame(),
                  "LONG_ONLY_BASE": s_lo_b.frame()}
        diags[n] = dg
        log(f"  {n:<8} dir {d:<4} v1 months {len(v1s[n]['LS_BASE']):>3}  "
            f"v2 months {len(v2s[n]['LS_BASE']):>3}  "
            f"v2 cross-section mean {dg['cross_section_mean']:.1f} "
            f"min {dg['cross_section_min']}  "
            f"skipped rank {dg['months_skipped_rank_deficient']} "
            f"small {dg['months_skipped_too_small']}")

    # One window for everything, so the v1-vs-v2 MDE comparison is like for like
    # and one shared month index can serve all five voted series.
    sets = ([set(v1s[n]["LS_BASE"]["month"]) for n in signals]
            + [set(v2s[n]["LS_BASE"]["month"]) for n in signals])
    common = sorted(set.intersection(*sets))
    log(f"  common month window across all ten series: {len(common)} months "
        f"({pd.Period(ordinal=int(common[0]), freq='M')}.."
        f"{pd.Period(ordinal=int(common[-1]), freq='M')})")

    def oc(f: pd.DataFrame, col: str) -> np.ndarray:
        g = f.loc[f["month"].isin(common)].sort_values("month")
        return g[col].to_numpy(dtype=np.float64)

    v1_net = {n: oc(v1s[n]["LS_BASE"], "net") for n in signals}
    v2_net = {n: oc(v2s[n]["LS_BASE"], "net") for n in signals}

    # ------------------------------------------- STAGE 2: MDE, v1 beside v2
    log("STAGE 2: MDE from each construction's own null spread "
        "(80% power, alpha 0.05 one-sided)")
    nulls_v1, _ = xt.shared_batch_nulls(v1_net, mean_block=MEAN_BLOCK, R=R_VOTING,
                                        seed=SEED_VOTING)
    nulls_v2, shared_idx = xt.shared_batch_nulls(v2_net, mean_block=MEAN_BLOCK,
                                                 R=R_VOTING, seed=SEED_VOTING)
    mde = {}
    for n in signals:
        m1 = xt.mde_from_null_sd(nulls_v1[n]["null_sd"])
        m2 = xt.mde_from_null_sd(nulls_v2[n]["null_sd"])
        mde[n] = {"v1_bps": m1 * 1e4, "v2_bps": m2 * 1e4, "ratio": m2 / m1,
                  "v1_vol_pct": float(v1_net[n].std(ddof=1) * np.sqrt(12) * 100),
                  "v2_vol_pct": float(v2_net[n].std(ddof=1) * np.sqrt(12) * 100)}
        log(f"  {n:<8} MDE v1 {m1*1e4:8.2f} bps -> v2 {m2*1e4:8.2f} bps  "
            f"ratio {m2/m1:5.3f}   vol v1 {mde[n]['v1_vol_pct']:5.2f}% -> "
            f"v2 {mde[n]['v2_vol_pct']:5.2f}%")
    n_halved = sum(1 for n in signals if mde[n]["ratio"] <= 0.5)
    pred = n_halved >= 4
    log(f"  PREDICTION was: v2's MDE at most half v1's on at least four of five. "
        f"Achieved on {n_halved} of 5 -> prediction {'HELD' if pred else 'FAILED'}")

    # ----------------------------------------- STAGE 3: known-null control v2
    log("STAGE 3: known-null control on v2 (gross arm, ratified addendum v1a/A1)")
    phi = xt.median_rank_autocorr(panel, signals, universe_col="in_universe")
    log(f"  batch median month-to-month rank autocorrelation phi = {phi:.4f}")
    log(f"  {N_CONTROL} AR(1)-noise signals through the full v2 pipeline at "
        f"R={R_CONTROL}")
    control = xt.known_null_control(panel, n_synthetic=N_CONTROL, phi=phi,
                                    spec=xt.COST_BASE, R=R_CONTROL,
                                    seed=SEED_CONTROL, mean_block=MEAN_BLOCK,
                                    calibration_arm="gross", builder="v2")
    log(f"  share at p<=0.05  GROSS (calibration arm) {control['share_gross']:.4f}")
    log(f"  share at p<=0.05  NET   (conservative)    {control['share_net']:.4f}")
    log(f"  median p gross {control['p_gross_median']:.3f} "
        f"net {control['p_net_median']:.3f}; mean turnover "
        f"{control['mean_turnover']:.3f}")
    log(f"  accept band {control['accept_band']} -> PASS={control['pass']}")
    if not control["pass"]:
        return halt("the v2 known-null control fell outside its band",
                    {"control": control})

    # ------------------------------------- STAGE 4: positive control, real data
    log("STAGE 4: semi-synthetic positive control on the real panel")
    log("  the planted term is the REALISED next-month residual return; the leak "
        "is deliberate and is what makes this a positive control")
    work = panel.loc[panel["in_universe"].astype(bool)].copy()
    work["_resid"] = xt.residual_next_month_return(panel)
    resid = work["_resid"].to_numpy()

    def plant_gross_sharpe(sigma: float) -> float:
        rng = np.random.default_rng(SEED_PLANT)
        work["_plant"] = xt.plant_signal(resid, sigma, rng)
        s, _ = xt.build_series_v2(work, "_plant", "HIGH", "LS", xt.COST_BASE)
        return xt.annualised_sharpe(s.frame()["gross"].to_numpy())

    lo, hi = 1.0, 128.0
    for _ in range(18):
        mid = (lo * hi) ** 0.5
        if plant_gross_sharpe(mid) > TARGET_PLANT_SHARPE:
            lo = mid
        else:
            hi = mid
    sigma_star = (lo * hi) ** 0.5
    sh_star = plant_gross_sharpe(sigma_star)
    log(f"  calibrated on ONE planting (seed {SEED_PLANT}): sigma* = "
        f"{sigma_star:.6f} -> v2 gross annualised Sharpe {sh_star:.4f} "
        f"(target {TARGET_PLANT_SHARPE})")
    log(f"  residual sd {np.nanstd(resid):.6f}; planted signal is "
        f"{np.nanstd(resid)/sigma_star*100:.3f}% signal to noise")

    def run_plantings(sigma: float, label: str) -> dict:
        det = {"v2_net": 0, "v2_gross": 0, "v1_net": 0, "v1_gross": 0}
        stats = []
        rng_master = np.random.default_rng(SEED_PLANT + 1)
        for _ in range(N_PLANT):
            rng = np.random.default_rng(int(rng_master.integers(0, 2**31 - 1)))
            work["_plant"] = xt.plant_signal(resid, sigma, rng)
            s2, _ = xt.build_series_v2(work, "_plant", "HIGH", "LS", xt.COST_BASE)
            f2 = s2.frame()
            f1 = xt.build_series(work, "_plant", "HIGH", "LS", xt.COST_BASE).frame()
            seed_i = int(rng_master.integers(0, 2**31 - 1))
            det["v2_net"] += xt.block_bootstrap_p(oc(f2, "net"), MEAN_BLOCK,
                                                  R_PLANT, seed_i)["p"] <= 0.05
            det["v2_gross"] += xt.block_bootstrap_p(oc(f2, "gross"), MEAN_BLOCK,
                                                    R_PLANT, seed_i)["p"] <= 0.05
            det["v1_net"] += xt.block_bootstrap_p(oc(f1, "net"), MEAN_BLOCK,
                                                  R_PLANT, seed_i)["p"] <= 0.05
            det["v1_gross"] += xt.block_bootstrap_p(oc(f1, "gross"), MEAN_BLOCK,
                                                    R_PLANT, seed_i)["p"] <= 0.05
            stats.append({"v2_net_bps": float(oc(f2, "net").mean() * 1e4),
                          "v2_gross_bps": float(oc(f2, "gross").mean() * 1e4),
                          "v1_net_bps": float(oc(f1, "net").mean() * 1e4),
                          "v2_turnover": float(f2["turnover"].mean())})
        rates = {k: v / N_PLANT for k, v in det.items()}
        log(f"  [{label}] {N_PLANT} fresh plantings, R={R_PLANT}. "
            f"Detection at p<=0.05:")
        log(f"    v2 NET {rates['v2_net']:.3f}  v2 GROSS {rates['v2_gross']:.3f}  "
            f"v1 NET {rates['v1_net']:.3f}  v1 GROSS {rates['v1_gross']:.3f}")
        log(f"    planted v2 gross "
            f"{np.mean([p['v2_gross_bps'] for p in stats]):8.2f} bps -> net "
            f"{np.mean([p['v2_net_bps'] for p in stats]):8.2f} bps  "
            f"(cost {np.mean([p['v2_gross_bps']-p['v2_net_bps'] for p in stats]):6.2f}), "
            f"turnover {np.mean([p['v2_turnover'] for p in stats]):.3f}")
        return {"sigma": float(sigma), "detection_rates": rates,
                "planted_v2_gross_bps": float(np.mean([p["v2_gross_bps"] for p in stats])),
                "planted_v2_net_bps": float(np.mean([p["v2_net_bps"] for p in stats])),
                "planted_v1_net_bps": float(np.mean([p["v1_net_bps"] for p in stats])),
                "planted_turnover_v2": float(np.mean([p["v2_turnover"] for p in stats]))}

    # CONTROL A — as registered: calibrated on GROSS Sharpe 0.8.
    ctrl_a = run_plantings(sigma_star, "A: gross-calibrated, as registered")

    # CONTROL B — added by addendum v2a/B2: calibrated on NET Sharpe 0.8, to
    # measure what the voting statistic can actually see.
    def plant_net_sharpe(sigma: float) -> float:
        rng = np.random.default_rng(SEED_PLANT)
        work["_plant"] = xt.plant_signal(resid, sigma, rng)
        s, _ = xt.build_series_v2(work, "_plant", "HIGH", "LS", xt.COST_BASE)
        return xt.annualised_sharpe(s.frame()["net"].to_numpy())

    lo_b, hi_b = 0.05, 64.0
    for _ in range(20):
        mid = (lo_b * hi_b) ** 0.5
        if plant_net_sharpe(mid) > TARGET_PLANT_SHARPE:
            lo_b = mid
        else:
            hi_b = mid
    sigma_net = (lo_b * hi_b) ** 0.5
    log(f"  addendum v2a/B2: sigma_net* = {sigma_net:.6f} -> v2 NET annualised "
        f"Sharpe {plant_net_sharpe(sigma_net):.4f}")
    ctrl_b = run_plantings(sigma_net, "B: net-calibrated, addendum v2a/B2")

    # ADJUDICATION — addendum v2a/B1: the registered floor is read on the arm
    # the calibration names, which is GROSS. The NET rate is carried as the
    # sensitivity caveat, not discarded.
    positive_control = {
        "control_A_as_registered": {
            "calibrated_on": "v2 GROSS annualised Sharpe",
            "sigma_star": float(sigma_star),
            "calibrated_gross_sharpe": float(sh_star),
            "target": TARGET_PLANT_SHARPE, **ctrl_a},
        "control_B_addendum_v2a": {
            "calibrated_on": "v2 NET annualised Sharpe",
            "sigma_net_star": float(sigma_net),
            "target": TARGET_PLANT_SHARPE, **ctrl_b},
        "n_plantings": N_PLANT, "R": R_PLANT, "floor": PLANT_DETECTION_FLOOR,
        "adjudicated_on": "control A, GROSS arm (addendum v2a/B1: the arm the "
                          "registered calibration names)",
        "pass": bool(ctrl_a["detection_rates"]["v2_gross"] >= PLANT_DETECTION_FLOOR),
        "net_sensitivity_caveat": "Control A's NET detection rate is "
                                  f"{ctrl_a['detection_rates']['v2_net']:.3f}. A "
                                  "planting calibrated to GROSS Sharpe 0.8 rotates "
                                  "almost fully, so a fixed cost of about 18.7 bps "
                                  "a month consumes most of it. Both controls "
                                  "therefore measure a FAST signal and understate "
                                  "net sensitivity for a slow one; the operative "
                                  "per-signal power statement is each signal's own "
                                  "MDE, reported as observed-over-MDE.",
        "residual_sd": float(np.nanstd(resid)),
        "deliberate_leak": "the planted term is the realised next-month residual "
                           "return; a positive control must know the future",
    }
    log(f"  floor {PLANT_DETECTION_FLOOR} on control A GROSS -> "
        f"PASS={positive_control['pass']}")
    if not positive_control["pass"]:
        return halt("the v2 positive control did not reach its detection floor "
                    "on the arm its calibration names; an instrument that cannot "
                    "see a planted real-data edge cannot certify a nil",
                    {"positive_control": positive_control})

    # ------------------------------------------------- STAGE 5: the observed
    log("STAGE 5: OBSERVED statistics and verdicts under v2")
    out: dict[str, dict] = {}
    for n in signals:
        lsb, lss, lo_b = (v2s[n]["LS_BASE"], v2s[n]["LS_STRESS"],
                          v2s[n]["LONG_ONLY_BASE"])
        net_c = v2_net[n]
        p_vote = nulls_v2[n]["p"]
        mean_c = float(net_c.mean())
        v = xt.verdict(p_vote, mean_c, BAR)
        lo_net = oc(lo_b, "net")
        lo_null = xt.p_from_index_matrix(lo_net, shared_idx)
        rec = {
            "direction": xt.REGISTERED_SIGNALS[n],
            "months_full_sample_v2": int(len(lsb)),
            "months_voted": int(len(net_c)),
            "net_mean_bps": mean_c * 1e4,
            "net_sharpe": xt.annualised_sharpe(net_c),
            "gross_mean_bps": float(oc(lsb, "gross").mean()) * 1e4,
            "gross_sharpe": xt.annualised_sharpe(oc(lsb, "gross")),
            "stress_mean_bps": float(oc(lss, "net").mean()) * 1e4,
            "stress_sharpe": xt.annualised_sharpe(oc(lss, "net")),
            "p_premium_voting": float(p_vote),
            "mde_v2_bps": mde[n]["v2_bps"], "mde_v1_bps": mde[n]["v1_bps"],
            "mde_ratio_v2_over_v1": mde[n]["ratio"],
            "vol_annualised_pct_v2": mde[n]["v2_vol_pct"],
            "vol_annualised_pct_v1": mde[n]["v1_vol_pct"],
            "power_ratio_abs_obs_over_mde": float(abs(mean_c) / (mde[n]["v2_bps"] / 1e4)),
            "long_only_net_mean_bps": float(lo_net.mean()) * 1e4,
            "long_only_net_sharpe": xt.annualised_sharpe(lo_net),
            "long_only_p": float(lo_null["p"]),
            "turnover_mean": float(lsb["turnover"].mean()),
            "turnover_long_only_mean": float(lo_b["turnover"].mean()),
            "subperiods_net": xt.subperiod_means(lsb, SUBPERIODS, "net"),
            "v2_diagnostics": diags[n],
            "verdict": v,
        }
        out[n] = rec
        log(f"  {n:<8} NET {rec['net_mean_bps']:8.3f} bps  Sharpe "
            f"{rec['net_sharpe']:7.3f}  p {p_vote:.4f}  -> {v}")
        log(f"           GROSS {rec['gross_mean_bps']:8.3f}  STRESS "
            f"{rec['stress_mean_bps']:8.3f}  LONG-ONLY "
            f"{rec['long_only_net_mean_bps']:8.3f} (p {rec['long_only_p']:.4f})")
        log(f"           vol {rec['vol_annualised_pct_v2']:5.2f}% (v1 "
            f"{rec['vol_annualised_pct_v1']:5.2f}%)  turnover "
            f"{rec['turnover_mean']:.3f}  MDE {rec['mde_v2_bps']:7.2f} bps  "
            f"obs/MDE {rec['power_ratio_abs_obs_over_mde']:.3f}")
        for lab, dd in rec["subperiods_net"].items():
            log(f"           {lab}: {dd['mean_bps']:8.3f} bps over "
                f"{dd['months']} months")

    mat = pd.DataFrame({n: v2_net[n] for n in signals})
    corr = mat.corr().to_numpy()
    neff = xt.effective_n(corr)
    log("CORRELATION of the five v2 NET series (common window):")
    log("          " + "  ".join(f"{n:>8}" for n in signals))
    for i, n in enumerate(signals):
        log(f"  {n:<8}" + "  ".join(f"{corr[i, j]:8.3f}" for j in range(len(signals))))
    log(f"  batch N_eff = {neff:.3f} (N = {len(signals)})")

    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID, "look": "look 2 of Q4 2026", "run_date": run_date,
        "vintage": VINTAGE, "runner_version": VERSION,
        "harness_version": xt.VERSION, "registration_sha256": reg_sha,
        "addendum_v2a_sha256": add_sha,
        "harness_sha256": harness_sha, "panel_sha256": panel_sha,
        "windows": windows,
        "panel_summary": json.loads((LOGDIR / "panel_build_meta_v2.json").read_text()),
        "verdict_bar": {"n_signals": 5, "n_looks_on_half_a": 2, "bar": BAR,
                        "derivation": "0.05 / (5 x 2)"},
        "voting_null": {"name": "stationary block bootstrap of months on the "
                                "demeaned v2 NET series",
                        "mean_block": MEAN_BLOCK, "R": R_VOTING,
                        "seed": SEED_VOTING,
                        "shared_month_index": list(shared_idx.shape),
                        "months": len(common),
                        "first_month": str(pd.Period(ordinal=int(common[0]), freq="M")),
                        "last_month": str(pd.Period(ordinal=int(common[-1]), freq="M"))},
        "permutation_null": "RETIRED from the harness (ratified ruling); it "
                            "treated each month's realisation as noise-free and "
                            "its turnover confound produced two false positives "
                            "in the T-R1 batch",
        "mde_v1_vs_v2": mde,
        "mde_prediction": {"stated": "v2's MDE at most half v1's on at least "
                                     "four of the five",
                           "n_at_or_below_half": n_halved, "held": bool(pred)},
        "known_null_control": control,
        "positive_control": positive_control,
        "signals": out,
        "correlation_net": {signals[i]: {signals[j]: float(corr[i, j])
                                         for j in range(len(signals))}
                            for i in range(len(signals))},
        "batch_n_eff": float(neff),
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    REPORTS.mkdir(exist_ok=True)
    rp = REPORTS / f"t_r1b_{run_date}.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    log(f"report {rp.relative_to(ROOT)} sha {sha_file(rp)}")

    lines = [l for l in LEDGER.read_text().strip().split("\n") if l]
    prev = lines[-1].split(",")[-1]
    rows = []
    for n in signals:
        r = out[n]
        vals = [BATCH_ID, run_date, VINTAGE, reg_sha[:16], harness_sha[:16], n,
                f"top{xt.TOP_N_BY_DOLLAR_VOLUME}_tight_v2_halfA", r["months_voted"],
                f"{r['net_mean_bps']:.4f}", f"{r['p_premium_voting']:.4f}",
                r["verdict"]]
        rs = row_sha(prev, vals)
        rows.append(",".join([str(x) for x in vals] + [prev, rs]))
        prev = rs
    with open(LEDGER, "a") as f:
        f.write("\n".join(rows) + "\n")
    log(f"ledger appended {len(rows)} rows; head {prev[:16]}")
    log(f"DONE wall {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
