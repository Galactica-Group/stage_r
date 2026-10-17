"""
================================================================================
Script:       run_t_r4.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R4 runner in two stages: the registered controls, and then —
              only on their written pass receipt — the observed half-A read.
Entry Point:  Yes — python tools/run_t_r4.py --stage controls|observed
                     --registration logs/r4/registration_v1.json --sha <sha256>
Dependencies: stdlib, numpy, pandas, tools/crypto_trend_r4.py, tools/ledger_chain.py.
              Imports nothing from any other repository.
Last Sprint:  Sprint R4-1 — T-R4 crypto trend: seal, controls, half-A read (2026-10-17)
================================================================================

The two stages exist to make the registered order of operations mechanical
rather than a promise. --stage controls walks steps 1 to 4 and writes
logs/r4/controls_pass.json on a pass or logs/r4/controls_halt.json on a halt,
exiting non-zero in the second case. --stage observed refuses to start unless
that receipt exists AND its registration_sha equals the live seal's, so a read
cannot be taken against a registration the controls never saw.

Steps 1 to 3 form no figure belonging to either observed book: the rates are
frequency, and the MDE and the control p values are computed from trend-side
Sharpes, which differ from dSR by the one constant this stage never forms. Step
4 must form the buy-and-hold Sharpe to calibrate q against an absolute dSR
target; it is used and discarded, printed nowhere and written nowhere.
================================================================================
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import crypto_trend_r4 as e  # noqa: E402
import ledger_chain as lc  # noqa: E402

VERSION = "1.0.0"
BATCH_ID = "T-R4"
PREDICATE = "TREND10_LONG_OR_CASH_TOP20"
LOGDIR = ROOT / "logs" / "r4"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"
RECEIPT = LOGDIR / "controls_pass.json"
HALTFILE = LOGDIR / "controls_halt.json"

LEDGER_FIELDS = ["batch_id", "date", "vintage", "registration_sha", "harness_sha",
                 "signal", "universe", "months", "ls_net_mean_bps", "p_premium",
                 "verdict"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def rel(p):
    """Repo-relative when it can be, absolute otherwise: logs must never raise."""
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Shared set-up: seal, universe, panel
# ---------------------------------------------------------------------------
def prepare(reg_path, sha):
    reg = e.load_registration(reg_path, sha)
    log("registration %s sha %s" % (Path(reg_path).name, reg["_sha256"]))
    man_path = ROOT / reg["data"]["manifest_tracked_copy"]
    man_sha = e.sha256_file(man_path)
    if man_sha != reg["data"]["manifest_sha256"]:
        raise AssertionError("manifest sha mismatch: %s vs registered %s"
                             % (man_sha, reg["data"]["manifest_sha256"]))
    manifest = json.loads(man_path.read_text())
    uni = e.resolve_universe(manifest, reg)
    log("universe: vintage %d, excluded union %d, pool %d"
        % (uni["vintage_symbols"], uni["excluded_union"], uni["pool_size"]))

    first_earn, last_earn, cut = e.window_bounds(reg)
    log("window: earning months %s..%s; bars loaded strictly before %s "
        "(half B and the holdout are never read)" % (first_earn, last_earn, cut.date()))
    vdir = ROOT / "data" / "binance_spot_1d" / reg["data"]["vintage"].split("_")[-1]
    daily, spans = e.load_vintage(vdir, uni["pool"], cut)
    log("symbols with half-A bars: %d of %d in the pool" % (len(daily), uni["pool_size"]))
    recs = e.sleeve_records(daily)
    log("sleeves inside the window: %d" % len(recs))
    panel = e.build_panel(recs, first_earn, last_earn)
    log("first month-end with >=2 eligible sleeves: %s"
        % panel["first_two_eligible_month_end"])
    log("SCORED WINDOW %s .. %s (%d months); sleeves in panel %d; assets %d"
        % (panel["first_scored_month"], panel["last_scored_month"],
           len(panel["months"]), panel["sleeves_in_panel"], len(panel["assets"])))
    return reg, uni, spans, recs, panel, man_sha


# ---------------------------------------------------------------------------
# Stage: controls
# ---------------------------------------------------------------------------
def stage_controls(args):
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reg, uni, spans, recs, panel, man_sha = prepare(args.registration, args.sha)

    bps = float(reg["costs"]["base_bps_per_side"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])
    kn = reg["controls"]["known_null"]
    pc = reg["controls"]["positive_control"]
    band = tuple(kn["band"])
    bar = float(reg["verdict"]["bar"])

    st_rule = panel["state_rule"]
    member = panel["member"]

    # ---- step 1: state path and rates (frequency only) ----
    log("STEP 1 - state path and switching RATES (frequency, not performance)")
    rates = e.measure_markov_rates(st_rule, member)
    sw = e.switch_count(panel, st_rule)
    act = member.any(axis=0)
    rate_summary = {
        "switches_total": sw,
        "sleeves_with_member_months": int(act.sum()),
        "p01_median": float(np.median(rates["p01"][act])),
        "p10_median": float(np.median(rates["p10"][act])),
        "p01_mean": float(rates["p01"][act].mean()),
        "p10_mean": float(rates["p10"][act].mean()),
    }
    log("  switches %d over %d sleeves; median p01 %.4f, median p10 %.4f"
        % (sw, rate_summary["sleeves_with_member_months"],
           rate_summary["p01_median"], rate_summary["p10_median"]))

    # ---- step 2: MDE ----
    log("STEP 2 - MDE from the random books' spread")
    rng_v = np.random.default_rng(seed_vote)
    S = e.simulate_markov(rates, member, R_vote, rng_v)
    net_null, _, _ = e.evaluate_books(panel, S, bps)
    s_null = e.annualised_sharpe(net_null)
    s_null = s_null[np.isfinite(s_null)]
    mde = e.mde_from_null(s_null)
    log("  null spread (trend-side Sharpe, = dSR up to the one constant this "
        "stage never forms): median %+.4f  sd %.4f  p95 %+.4f  R=%d"
        % (np.median(s_null), s_null.std(ddof=1), np.percentile(s_null, 95), len(s_null)))
    log("  MDE at 80%% power = %+.4f dSR" % mde)
    del S, net_null

    # ---- step 3: known-null control ----
    log("STEP 3 - known-null control: %d random books, fresh %d-book null each"
        % (kn["N"], kn["R_each"]))
    rng_c = np.random.default_rng(int(kn["seed"]))
    books = e.simulate_markov(rates, member, int(kn["N"]), rng_c)
    ps = np.empty(int(kn["N"]))
    for i in range(int(kn["N"])):
        ps[i] = e.timing_p(panel, books[i], bps, int(kn["R_each"]), rng_c)["p"]
        if (i + 1) % 50 == 0:
            log("  %d/%d  running share p<=%.2f = %.4f"
                % (i + 1, kn["N"], bar, (ps[:i + 1] <= bar).mean()))
    share = float((ps <= bar).mean())
    known_null = {"N": int(kn["N"]), "R_each": int(kn["R_each"]),
                  "seed": int(kn["seed"]), "share_p_le_bar": share,
                  "band": list(band), "mean_p": float(ps.mean()),
                  "median_p": float(np.median(ps)),
                  "in_band": e.band_ok(share, band)}
    log("  share at p<=%.2f = %.4f  (band %.2f-%.2f)  mean p %.4f"
        % (bar, share, band[0], band[1], ps.mean()))
    del books
    if not known_null["in_band"]:
        return halt_out(reg, "known-null control share %.4f outside [%.2f, %.2f]"
                        % (share, band[0], band[1]),
                        mde, known_null, None, rate_summary, t0)
    log("  KNOWN-NULL CONTROL PASSES")

    # ---- step 4: positive control ----
    log("STEP 4 - positive control: calibrate q on ONE draw to dSR ~ %.2f"
        % pc["q_target_dSR"])
    # the one buy-and-hold figure this stage must form: the absolute dSR target
    # cannot be hit without it. Used, never printed, never written.
    net_bh, _, _ = e.evaluate_books(panel, member[None, :, :], bps)
    s_bh = float(e.annualised_sharpe(net_bh[0]))
    cal = []
    for q in pc["q_grid"]:
        st = e.plant_states(panel, float(q), 1,
                            np.random.default_rng(int(pc["seed"])))[0]
        nt, _, _ = e.evaluate_books(panel, st[None, :, :], bps)
        cal.append({"q": float(q),
                    "planted_dsr": float(e.annualised_sharpe(nt[0]) - s_bh)})
        log("  q=%.2f -> planted dSR %+.4f" % (q, cal[-1]["planted_dsr"]))
    target = float(pc["q_target_dSR"])
    best = min(cal, key=lambda r: abs(r["planted_dsr"] - target))
    q_star = best["q"]
    log("  calibrated q = %.2f (one-draw planted dSR %+.4f)"
        % (q_star, best["planted_dsr"]))

    rng_p = np.random.default_rng(int(pc["seed"]))
    hits, pp = 0, []
    for i in range(int(pc["N_draws"])):
        st = e.plant_states(panel, q_star, 1, rng_p)[0]
        o = e.timing_p(panel, st, bps, int(pc["R_each"]), rng_p)
        pp.append(o["p"])
        hits += o["p"] <= bar
        if (i + 1) % 25 == 0:
            log("  %d/%d  detection %.2f" % (i + 1, pc["N_draws"], hits / (i + 1)))
    det = hits / int(pc["N_draws"])
    positive = {"q_grid": cal, "q_star": q_star,
                "one_draw_dsr_at_q_star": best["planted_dsr"],
                "N_draws": int(pc["N_draws"]), "R_each": int(pc["R_each"]),
                "seed": int(pc["seed"]), "detection_rate": det,
                "floor": float(pc["detection_floor"]),
                "median_p": float(np.median(pp)),
                "passes": e.detection_ok(det, pc["detection_floor"]),
                "leak": pc["leak"]}
    log("  detection %.2f at q=%.2f (floor %.2f)"
        % (det, q_star, pc["detection_floor"]))
    if not positive["passes"]:
        return halt_out(reg, "positive control detection %.2f below floor %.2f"
                        % (det, pc["detection_floor"]),
                        mde, known_null, positive, rate_summary, t0)
    log("  POSITIVE CONTROL PASSES")

    receipt = {
        "registration_sha": reg["_sha256"],
        "mde": mde,
        "known_null_share": known_null["share_p_le_bar"],
        "q_chosen": q_star,
        "detection": det,
        "utc": utc_now(),
        "runner_version": VERSION, "engine_version": e.VERSION,
        "engine_sha256": e.sha256_file(ROOT / "tools" / "crypto_trend_r4.py"),
        "manifest_sha256": man_sha,
        "rates_frequency_only": rate_summary,
        "known_null_control": known_null,
        "positive_control": positive,
        "panel": panel_summary(panel),
        "note": "steps 1-3 formed no figure of either observed book; step 4 formed "
                "the buy-and-hold Sharpe to calibrate q and neither printed nor "
                "recorded it",
        "wall_seconds": round(time.time() - t0, 1),
    }
    if HALTFILE.exists():
        HALTFILE.unlink()
    RECEIPT.write_text(json.dumps(receipt, indent=2, default=str))
    log("receipt %s sha %s" % (rel(RECEIPT), e.sha256_file(RECEIPT)))
    log("CONTROLS PASS - the observed stage is unlocked")
    return 0


def halt_out(reg, reason, mde, known_null, positive, rate_summary, t0):
    log("HALT: %s" % reason)
    log("No observed figure is computed. No look is spent. No ledger row.")
    payload = {
        "halt": reason, "registration_sha": reg["_sha256"], "mde": mde,
        "known_null_share": (known_null or {}).get("share_p_le_bar"),
        "q_chosen": (positive or {}).get("q_star"),
        "detection": (positive or {}).get("detection_rate"),
        "utc": utc_now(), "rates_frequency_only": rate_summary,
        "known_null_control": known_null, "positive_control": positive,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if RECEIPT.exists():
        RECEIPT.unlink()
    HALTFILE.write_text(json.dumps(payload, indent=2, default=str))
    log("halt file %s" % rel(HALTFILE))
    return 15


def panel_summary(panel):
    n = panel["n_members"]
    return {
        "first_two_eligible_month_end": panel["first_two_eligible_month_end"],
        "first_scored_month": panel["first_scored_month"],
        "last_scored_month": panel["last_scored_month"],
        "months": int(len(panel["months"])),
        "sleeves_in_panel": panel["sleeves_in_panel"],
        "sleeves_total": panel["sleeves_total"],
        "assets": len(panel["assets"]),
        "n_members_min": int(n.min()), "n_members_max": int(n.max()),
        "n_members_months_at_top_n": int((n == e.TOP_N).sum()),
        "n_members_path": [int(x) for x in n],
        "forced_exit_member_months": panel["forced_exit_member_months"],
    }


# ---------------------------------------------------------------------------
# Stage: observed
# ---------------------------------------------------------------------------
def stage_observed(args):
    t0 = time.time()
    if not RECEIPT.exists():
        log("REFUSED: no controls pass receipt at %s" % rel(RECEIPT))
        log("The registered order of operations puts the observed read after the "
            "controls. Run --stage controls first.")
        return 2
    receipt = json.loads(RECEIPT.read_text())
    live = e.sha256_file(args.registration)
    if receipt.get("registration_sha") != live:
        log("REFUSED: the receipt was written against registration sha %s but the "
            "live file is %s" % (receipt.get("registration_sha"), live))
        return 2
    log("controls receipt accepted (sha %s)" % receipt["registration_sha"])

    reg, uni, spans, recs, panel, man_sha = prepare(args.registration, args.sha)
    bps = float(reg["costs"]["base_bps_per_side"])
    bps_stress = float(reg["costs"]["stress_bps_per_side"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])
    bar = float(reg["verdict"]["bar"])
    sug = 0.10

    st_rule, member = panel["state_rule"], panel["member"]
    months = panel["months"]
    M = len(months)

    out = {}
    for label, b in (("base", bps), ("stress", bps_stress)):
        net_tr, _, turn_tr = e.evaluate_books(panel, st_rule[None, :, :], b)
        net_bh, _, turn_bh = e.evaluate_books(panel, member[None, :, :], b)
        net_tr, net_bh = net_tr[0], net_bh[0]
        turn_tr, turn_bh = turn_tr[0], turn_bh[0]
        d_obs = float(e.dsr(net_tr, net_bh))
        dm = float(e.dmu(net_tr, net_bh))
        o = e.timing_p(panel, st_rule, b, R_vote, np.random.default_rng(seed_vote))
        out[label] = {
            "bps_per_side": b, "dsr": d_obs, "timing_p": o["p"], "dmu": dm,
            "mean_monthly_net_diff": float(net_tr.mean() - net_bh.mean()),
            "TREND": e.book_metrics(net_tr), "BUY_AND_HOLD": e.book_metrics(net_bh),
            "turnover_mean_annual_trend": float(turn_tr.mean() * 12),
            "turnover_mean_annual_bh": float(turn_bh.mean() * 12),
            "R_effective": o["R_effective"],
        }
        log("  [%s %.1f bps] dSR %+.4f  timing p %.4f  dMu %+.4f"
            % (label, b, d_obs, o["p"], dm))
        for nm in ("TREND", "BUY_AND_HOLD"):
            mm = out[label][nm]
            log("    %-13s CAGR %+8.2f%%  vol %7.2f%%  Sharpe %+.4f  maxDD %8.2f%%"
                % (nm, 100 * mm["cagr"], 100 * mm["ann_vol"], mm["sharpe"],
                   100 * mm["max_drawdown"]))

    inv = e.invested_share(panel, st_rule)
    sw = e.switch_count(panel, st_rule)
    years = M / 12.0
    log("  invested share: sleeve-months %.4f, weighted %.4f; switches %d (%.1f/yr)"
        % (inv["sleeve_month_share"], inv["weighted_share"], sw, sw / years))

    b0 = out["base"]
    d_obs, p_obs, dm = b0["dsr"], b0["timing_p"], b0["dmu"]
    if p_obs <= bar and d_obs > 0 and dm >= 0:
        verdict = "SOMETHING"
    elif p_obs <= bar and d_obs > 0 and dm < 0:
        verdict = "RISK_ONLY"
    elif p_obs <= sug and d_obs > 0:
        verdict = "SUGGESTIVE"
    else:
        verdict = "NOTHING"
    log("VERDICT: %s  (p=%.4f, dSR=%+.4f, dMu=%+.4f, bar %.2f)"
        % (verdict, p_obs, d_obs, dm, bar))

    reg_sha = reg["_sha256"]
    harness_sha = e.sha256_file(ROOT / "tools" / "crypto_trend_r4.py")
    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID,
        "look": reg["look_budget"],
        "run_date": run_date,
        "vintage": reg["ledger"]["vintage_string"],
        "runner_version": VERSION, "engine_version": e.VERSION,
        "registration_sha256": reg_sha, "engine_sha256": harness_sha,
        "manifest_sha256": man_sha,
        "controls_receipt_sha256": e.sha256_file(RECEIPT),
        "controls_receipt": {k: receipt[k] for k in
                             ("mde", "known_null_share", "q_chosen", "detection", "utc")},
        "universe": {k: v for k, v in uni.items() if k != "pool"},
        "universe_pool_size": uni["pool_size"],
        "panel": panel_summary(panel),
        "months_scored": M,
        "observed": out,
        "descriptives": {"invested_share": inv, "switches_total": sw,
                         "switches_per_year": sw / years},
        "verdict": {"verdict": verdict, "bar": bar, "suggestive_bar": sug,
                    "single_test": True,
                    "promotion": ("one registered half-B look (2022-2024)"
                                  if verdict == "SOMETHING" else "none")},
        "survivorship_note": reg["universe"]["survivorship"],
        "known_omission": reg["data"]["known_omission"],
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    LOGDIR.mkdir(parents=True, exist_ok=True)
    rp = LOGDIR / "t_r4_halfA_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    log("report %s sha %s" % (rel(rp), e.sha256_file(rp)))

    ok, n, head, bad = lc.verify(LEDGER)
    if not ok:
        raise AssertionError("ledger chain is broken before the append: %r" % bad)
    vals = [BATCH_ID, run_date, reg["ledger"]["vintage_string"],
            reg_sha[:16], harness_sha[:16], PREDICATE,
            reg["ledger"]["universe_string"], M,
            "%.4f" % (1e4 * b0["mean_monthly_net_diff"]), "%.4f" % p_obs, verdict]
    rs = lc.row_sha(head, vals)
    with open(LEDGER, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    log("ledger appended 1 row; head %s" % rs[:16])
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-R4 crypto trend runner")
    ap.add_argument("--stage", required=True, choices=["controls", "observed"])
    ap.add_argument("--registration", required=True)
    ap.add_argument("--sha", required=True)
    args = ap.parse_args(argv)
    log("run_t_r4 v%s; engine v%s; stage %s" % (VERSION, e.VERSION, args.stage))
    if args.stage == "controls":
        return stage_controls(args)
    return stage_observed(args)


if __name__ == "__main__":
    sys.exit(main())
