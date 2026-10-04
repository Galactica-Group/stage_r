"""
================================================================================
Script:       run_t_r1.py
Version:      1.0.0
Phase:        Stage R — Cross-Sectional Triage
Purpose:      Run look T-R1: five textbook price-only signals on the half-A
              panel, in the registered order MDE -> control -> observed.
Entry Point:  Yes (python tools/run_t_r1.py, run detached under nohup)
Dependencies: tools/xs_triage.py, numpy, pandas
Last Sprint:  Sprint R1 — Cross-Sectional Triage Harness (2026-10-04)
================================================================================

The run is staged so that nothing about the observed means is printed, stored or
returned until stages 1 to 4 have been emitted. Stage 1 builds the five realised
long-short net series. Stage 2 emits each signal's MDE from the voting null's
own spread (the null is demeaned, so its spread does not encode the observed
mean). Stage 3 runs the known-null control and HALTS the run with verdicts
withheld if its calibration arm falls outside the registered band. Only stage 5
reveals the observed statistics and the verdicts.
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

# ---------------------------------------------------------------------------
# RETIRED INSTRUMENT, kept here so look T-R1 stays reproducible.
#
# The within-date rank permutation was the charter's null. Sprint R1b retired it
# from the harness entirely on two measured grounds, both visible in T-R1's own
# output: it treats each month's factor realisation as noise-free, and because
# permuting across names destroys month-to-month persistence the permuted book
# rotated roughly twice as hard as the real one (3.25 against 1.17-3.20), so on a
# net statistic it flattered any persistent signal. In the T-R1 batch it would
# have reported two false positives — MOM12_1 at p 0.0050 against a voting
# 0.1639, and HI52 at p 0.0299 on a negative mean.
#
# It lives in this historical runner and nowhere else. Nothing new may use it.
# ---------------------------------------------------------------------------
def within_date_permutation_p(panel: pd.DataFrame, signal_col: str, direction: str,
                              mode: str, spec: "xt.CostSpec", R: int = 200, seed: int = 0,
                              fwd_col: str = "fwd_ret", universe_col: str = "in_universe",
                              month_col: str = "month", key_col: str = "permaticker") -> dict:
    """
    REPORTED, NOT VOTED. Permutes the signal across names WITHIN each date.

    Two independent reasons it does not vote, the second measured here:
      (1) it treats each month's factor realisation as noise-free and so
          overstates significance;
      (2) permuting across names destroys the signal's month-to-month
          PERSISTENCE, so the permuted book rotates far harder than the real
          one and pays far more cost. On a NET statistic that alone makes any
          persistent real signal look significant whatever its edge. The
          permuted and real turnovers are returned so the size of that artifact
          is visible, and the gross p is returned beside the net p because the
          gross comparison is at least not confounded by it.
    """
    obs = xt.build_series(panel, signal_col, direction, mode, spec,
                       fwd_col=fwd_col, universe_col=universe_col,
                       month_col=month_col, key_col=key_col)
    obs_net = float(np.mean(obs.net)) if obs.net else float("nan")
    obs_gross = float(np.mean(obs.gross)) if obs.gross else float("nan")
    obs_turn = float(np.mean(obs.turnover)) if obs.turnover else float("nan")

    rng = np.random.default_rng(seed)
    work = panel.copy()
    ge_net = ge_gross = 0
    turns: list[float] = []
    for _ in range(R):
        work["_perm"] = work.groupby(month_col, sort=False, observed=True)[
            signal_col].transform(lambda s, _r=rng: s.to_numpy()[_r.permutation(len(s))])
        s = xt.build_series(work, "_perm", direction, mode, spec, fwd_col=fwd_col,
                         universe_col=universe_col, month_col=month_col, key_col=key_col)
        if not s.net:
            continue
        if float(np.mean(s.net)) >= obs_net:
            ge_net += 1
        if float(np.mean(s.gross)) >= obs_gross:
            ge_gross += 1
        turns.append(float(np.mean(s.turnover)))
    return {"p": (1.0 + ge_net) / (R + 1.0),
            "p_gross": (1.0 + ge_gross) / (R + 1.0),
            "R": R, "obs_mean": obs_net, "obs_mean_gross": obs_gross,
            "obs_turnover": obs_turn,
            "permuted_turnover": float(np.mean(turns)) if turns else float("nan")}


ROOT = Path(__file__).resolve().parent.parent
PANEL = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA.parquet"
LOGDIR = ROOT / "logs" / "r1"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"

VINTAGE = "20261004p"
BATCH_ID = "T-R1"
BAR = 0.01
R_VOTING = 2000
R_PERM = 200
R_CONTROL = 500
N_CONTROL = 200
MEAN_BLOCK = 3.0
SEED_VOTING = 20261004
SEED_PERM = 20261004
SEED_CONTROL = 777

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


def row_sha(prev: str, values: list[str]) -> str:
    payload = "\x00".join([prev] + [str(v) for v in values])
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> int:
    t0 = time.time()
    log(f"run_t_r1 v{VERSION}; harness v{xt.VERSION}")

    windows = json.loads((ROOT / "config" / "windows.json").read_text())
    reg_sha = sha_file(LOGDIR / "registration_v1.json")
    add_sha = sha_file(LOGDIR / "registration_v1a_addendum.json")
    harness_sha = sha_file(ROOT / "tools" / "xs_triage.py")
    log(f"registration sha {reg_sha}")
    log(f"addendum v1a sha {add_sha}")
    log(f"harness sha      {harness_sha}")

    # ---------------------------------------------------------------- panel
    panel = pd.read_parquet(PANEL)
    panel_sha = sha_file(PANEL)
    xt.assert_no_rows_at_or_after(panel["month_ts"],
                                  windows["half_a"]["daily_row_exclusive_max"],
                                  "panel load")
    held = panel.loc[panel["in_universe"].astype(bool)]
    log(f"panel sha {panel_sha}")
    log(f"panel rows {len(panel):,}; in-universe {len(held):,}; "
        f"max month {held['month_ts'].max().date()} "
        f"(cutoff {windows['half_a']['daily_row_exclusive_max']}, exclusive)")

    signals = list(xt.REGISTERED_SIGNALS)

    # ------------------------------------------------- STAGE 1: realised series
    log("STAGE 1: build the realised monthly series (no statistic emitted yet)")
    series: dict[str, dict] = {}
    for name in signals:
        d = xt.REGISTERED_SIGNALS[name]
        rec = {}
        for mode in ("LS", "LONG_ONLY"):
            for spec in (xt.COST_BASE, xt.COST_STRESS):
                if mode == "LONG_ONLY" and spec.label == "STRESS":
                    continue
                f = xt.build_series(panel, name, d, mode, spec).frame()
                rec[f"{mode}_{spec.label}"] = f
        series[name] = rec
        n = len(rec["LS_BASE"])
        log(f"  {name:<8} direction {d:<4} months {n:>3} "
            f"({rec['LS_BASE']['month'].min()}..{rec['LS_BASE']['month'].max()})")

    # Common month window: addendum A2 — one shared month index needs one window.
    month_sets = [set(series[n]["LS_BASE"]["month"]) for n in signals]
    common = sorted(set.intersection(*month_sets))
    log(f"  common month window: {len(common)} months "
        f"({pd.Period(ordinal=int(common[0]), freq='M')}.."
        f"{pd.Period(ordinal=int(common[-1]), freq='M')})")

    def on_common(f: pd.DataFrame, col: str) -> np.ndarray:
        g = f.loc[f["month"].isin(common)].sort_values("month")
        return g[col].to_numpy(dtype=np.float64)

    ls_net_common = {n: on_common(series[n]["LS_BASE"], "net") for n in signals}

    # --------------------------------------------- STAGE 2: MDE, before anything
    log("STAGE 2: MDE per signal, from the voting null's own spread "
        "(demeaned, so it cannot encode the observed mean)")
    nulls, shared_idx = xt.shared_batch_nulls(ls_net_common, mean_block=MEAN_BLOCK,
                                              R=R_VOTING, seed=SEED_VOTING)
    mde = {}
    for n in signals:
        sd = nulls[n]["null_sd"]
        mde[n] = xt.mde_from_null_sd(sd)
        log(f"  {n:<8} null_sd {sd*1e4:8.3f} bps  MDE {mde[n]*1e4:8.3f} bps/month "
            f"(80% power, alpha 0.05 one-sided)")
    log(f"  one shared month-index draw for all five series: {shared_idx.shape}")

    # ------------------------------------------- STAGE 3: known-null control
    log("STAGE 3: known-null control")
    phi = xt.median_rank_autocorr(panel, signals, universe_col="in_universe")
    log(f"  batch median month-to-month rank autocorrelation phi = {phi:.4f}")
    log(f"  {N_CONTROL} AR(1)-noise signals through the full pipeline at R={R_CONTROL}")
    control = xt.known_null_control(panel, n_synthetic=N_CONTROL, phi=phi,
                                    spec=xt.COST_BASE, R=R_CONTROL, seed=SEED_CONTROL,
                                    mean_block=MEAN_BLOCK, calibration_arm="gross")
    log(f"  share at p<=0.05  GROSS (calibration arm) {control['share_gross']:.4f}")
    log(f"  share at p<=0.05  NET   (conservative)    {control['share_net']:.4f}")
    log(f"  accept band {control['accept_band']} -> PASS={control['pass']}")
    log(f"  control mean turnover {control['mean_turnover']:.3f}; "
        f"median p gross {control['p_gross_median']:.3f} net {control['p_net_median']:.3f}")
    if not control["pass"]:
        log("HALT: the known-null control fell outside its band. "
            "Verdicts are WITHHELD and no observed statistic is emitted.")
        (LOGDIR / "t_r1_HALT.json").write_text(json.dumps(
            {"halt": "known-null control outside band", "control": {
                k: v for k, v in control.items()}}, indent=2))
        return 15

    # -------------------------------- STAGE 4: the non-voting permutation null
    log("STAGE 4: within-date rank permutation, REPORTED NOT VOTED "
        "(it treats each month's factor realisation as noise-free and so "
        "overstates significance)")
    perm = {}
    for n in signals:
        sub = panel.loc[panel["in_universe"].astype(bool),
                        ["month", "permaticker", n, "fwd_ret", "in_universe"]].copy()
        sub = sub.loc[sub["month"].isin(common)]
        perm[n] = within_date_permutation_p(
            sub, n, xt.REGISTERED_SIGNALS[n], "LS", xt.COST_BASE,
            R=R_PERM, seed=SEED_PERM)
        log(f"  {n:<8} permutation p(net) {perm[n]['p']:.4f}  p(gross) "
            f"{perm[n]['p_gross']:.4f}  turnover real {perm[n]['obs_turnover']:.3f} "
            f"vs permuted {perm[n]['permuted_turnover']:.3f} (R={R_PERM})")

    # ------------------------------------------------ STAGE 5: the observed
    log("STAGE 5: OBSERVED statistics and verdicts")
    out: dict[str, dict] = {}
    for n in signals:
        lsb = series[n]["LS_BASE"]
        lss = series[n]["LS_STRESS"]
        lo = series[n]["LONG_ONLY_BASE"]

        ls_net_c = ls_net_common[n]
        p_vote = nulls[n]["p"]
        mean_c = float(ls_net_c.mean())
        v = xt.verdict(p_vote, mean_c, BAR)

        lo_net_c = on_common(lo, "net")
        lo_null = xt.p_from_index_matrix(lo_net_c, shared_idx)

        rec = {
            "direction": xt.REGISTERED_SIGNALS[n],
            "months_full_sample": int(len(lsb)),
            "months_voted": int(len(ls_net_c)),
            "ls_net_mean_bps": mean_c * 1e4,
            "ls_net_sharpe": xt.annualised_sharpe(ls_net_c),
            "ls_gross_mean_bps": float(on_common(lsb, "gross").mean()) * 1e4,
            "ls_gross_sharpe": xt.annualised_sharpe(on_common(lsb, "gross")),
            "ls_stress_mean_bps": float(on_common(lss, "net").mean()) * 1e4,
            "ls_stress_sharpe": xt.annualised_sharpe(on_common(lss, "net")),
            "p_premium_voting": float(p_vote),
            "mde_bps": mde[n] * 1e4,
            "null_sd_bps": nulls[n]["null_sd"] * 1e4,
            "p_permutation_not_voted": float(perm[n]["p"]),
            "p_permutation_gross_not_voted": float(perm[n]["p_gross"]),
            "permutation_turnover_real": float(perm[n]["obs_turnover"]),
            "permutation_turnover_permuted": float(perm[n]["permuted_turnover"]),
            "long_only_net_mean_bps": float(lo_net_c.mean()) * 1e4,
            "long_only_net_sharpe": xt.annualised_sharpe(lo_net_c),
            "long_only_p": float(lo_null["p"]),
            "turnover_ls_mean": float(lsb["turnover"].mean()),
            "turnover_long_only_mean": float(lo["turnover"].mean()),
            "subperiods_ls_net": xt.subperiod_means(lsb, SUBPERIODS, "net"),
            "ls_net_vol_annualised_pct": float(ls_net_c.std(ddof=1) * np.sqrt(12) * 100),
            "power_ratio_abs_obs_over_mde": float(abs(mean_c) / mde[n])
            if mde[n] > 0 else float("nan"),
            "verdict": v,
        }
        out[n] = rec
        log(f"  {n:<8} LS NET {rec['ls_net_mean_bps']:8.3f} bps  "
            f"Sharpe {rec['ls_net_sharpe']:7.3f}  p {p_vote:.4f}  -> {v}")
        log(f"           GROSS {rec['ls_gross_mean_bps']:8.3f}  "
            f"STRESS {rec['ls_stress_mean_bps']:8.3f}  "
            f"LONG-ONLY {rec['long_only_net_mean_bps']:8.3f} (p {rec['long_only_p']:.4f})  "
            f"turnover {rec['turnover_ls_mean']:.3f}")
        for lab, d in rec["subperiods_ls_net"].items():
            log(f"           {lab}: {d['mean_bps']:8.3f} bps over {d['months']} months")

    # ----------------------------------------- correlation matrix and N_eff
    mat = pd.DataFrame({n: ls_net_common[n] for n in signals})
    corr = mat.corr().to_numpy()
    neff = xt.effective_n(corr)
    log("CORRELATION of the five LS NET series (common window):")
    log("          " + "  ".join(f"{n:>8}" for n in signals))
    for i, n in enumerate(signals):
        log(f"  {n:<8}" + "  ".join(f"{corr[i, j]:8.3f}" for j in range(len(signals))))
    log(f"  batch N_eff = N^2 / sum(rho) = {neff:.3f} (N = {len(signals)})")

    # ------------------------------------------------------------- artifacts
    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID, "run_date": run_date, "vintage": VINTAGE,
        "runner_version": VERSION, "harness_version": xt.VERSION,
        "registration_sha256": reg_sha, "addendum_v1a_sha256": add_sha,
        "harness_sha256": harness_sha, "panel_sha256": panel_sha,
        "windows": windows,
        "panel_summary": json.loads((LOGDIR / "panel_build_meta.json").read_text()),
        "verdict_bar": {"n_signals": len(signals), "bar": BAR, "derivation": "0.05/5"},
        "voting_null": {"name": "stationary block bootstrap of months on the "
                                "demeaned LS NET series",
                        "mean_block": MEAN_BLOCK, "R": R_VOTING, "seed": SEED_VOTING,
                        "shared_month_index": list(shared_idx.shape),
                        "window": "the batch common month window (addendum A2)",
                        "months": len(common),
                        "first_month": str(pd.Period(ordinal=int(common[0]), freq="M")),
                        "last_month": str(pd.Period(ordinal=int(common[-1]), freq="M"))},
        "reported_not_voted": {
            "name": "within-date rank permutation", "R": R_PERM,
            "disclosure_1": "treats each month's factor realisation as noise-free "
                            "and so overstates significance",
            "disclosure_2": "permuting across names destroys month-to-month "
                            "persistence, so the permuted book rotates harder and "
                            "pays more cost than the real one; on a NET statistic "
                            "that alone flatters any persistent signal. Real and "
                            "permuted turnovers are reported per signal, and the "
                            "gross p is given beside the net p as the unconfounded "
                            "comparison.",
            "status": "continuity with the charter only; never votes"},
        "costs_provisional": {
            "BASE": {"bps_per_side": xt.COST_BASE.bps_per_side,
                     "borrow_bps_per_year": xt.COST_BASE.borrow_bps_per_year},
            "STRESS": {"bps_per_side": xt.COST_STRESS.bps_per_side,
                       "borrow_bps_per_year": xt.COST_STRESS.borrow_bps_per_year},
            "note": "PROVISIONAL. LONG-ONLY charges the Q5 leg only; the "
                    "universe-equal-weight benchmark leg is not traded."},
        "known_null_control": {k: v for k, v in control.items()},
        "signals": out,
        "correlation_ls_net": {signals[i]: {signals[j]: float(corr[i, j])
                                            for j in range(len(signals))}
                               for i in range(len(signals))},
        "batch_n_eff": float(neff),
        "power_finding": {
            "statement": "This look is structurally UNDERPOWERED on the registered "
                         "statistic. The minimum detectable monthly long-short mean "
                         "is 115 to 210 bps, because the quintile spread carries "
                         "roughly 8 to 11 per cent monthly volatility and the voted "
                         "window is 168 months. Real equity factor premia are an "
                         "order of magnitude smaller than that. Every observed "
                         "effect is well inside its own MDE (the largest, MOM12_1, "
                         "reaches 0.40 of it; three of the five are under 0.05).",
            "consequence": "A NOTHING here means 'no effect of implausibly large "
                           "size was found'. It is NOT evidence that these premia "
                           "are absent, and it must not be read as one.",
            "max_power_ratio": float(max(
                out[n]["power_ratio_abs_obs_over_mde"] for n in signals)),
            "signals_under_one_twentieth_of_mde": [
                n for n in signals
                if out[n]["power_ratio_abs_obs_over_mde"] < 0.05],
        },
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    REPORTS.mkdir(exist_ok=True)
    rp = REPORTS / f"t_r1_{run_date}.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    log(f"report {rp.relative_to(ROOT)} sha {sha_file(rp)}")

    # ----------------------------------------------------- sha-chained ledger
    new = not LEDGER.exists()
    if new:
        LEDGER.write_text(",".join(LEDGER_FIELDS + ["prev_row_sha", "row_sha"]) + "\n")
        prev = "GENESIS"
    else:
        lines = [l for l in LEDGER.read_text().strip().split("\n") if l]
        prev = lines[-1].split(",")[-1] if len(lines) > 1 else "GENESIS"
    rows = []
    for n in signals:
        r = out[n]
        vals = [BATCH_ID, run_date, VINTAGE, reg_sha[:16], harness_sha[:16], n,
                f"top{xt.TOP_N_BY_DOLLAR_VOLUME}_common_halfA", r["months_voted"],
                f"{r['ls_net_mean_bps']:.4f}", f"{r['p_premium_voting']:.4f}",
                r["verdict"]]
        rs = row_sha(prev, vals)
        rows.append(",".join([str(v) for v in vals] + [prev, rs]))
        prev = rs
    with open(LEDGER, "a") as f:
        f.write("\n".join(rows) + "\n")
    log(f"ledger {LEDGER.relative_to(ROOT)}: appended {len(rows)} rows; head {prev[:16]}")
    log(f"DONE wall {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
