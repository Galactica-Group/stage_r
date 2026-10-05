"""
================================================================================
Script:       run_t_r3.py
Version:      1.0.0
Phase:        Stage R — Timing Triage
Purpose:      Run look T-R3 (look 3 of Q4 2026): an ETF ten-month trend book,
              long-or-cash, against buy-and-hold of the same ETFs, on half A.
Entry Point:  Yes (python tools/run_t_r3.py)
Dependencies: tools/etf_trend_r3.py, numpy, pandas
Last Sprint:  Sprint R3 — Look 3 of Q4 (2026-10-05)
================================================================================

Staged so that nothing observed is emitted until the instrument has earned it:
  STAGE 1  MDE, from the random books' own spread
  STAGE 2  known-null control; HALT outside [0.01, 0.10]
  STAGE 3  semi-synthetic positive control; HALT below 0.80 detection
  STAGE 4  the observed book, and only then

The observed dSR is not computed before stage 4. Stage 1 needs only the null
books, which are generated from the rule's switching FREQUENCY, never from its
performance.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dating_test_r3 as dt  # noqa: E402
import etf_trend_r3 as e  # noqa: E402

VERSION = "1.0.0"
BATCH_ID = "T-R3"
VINTAGE = "funds_20261005f"
VDIR = ROOT / "data" / "sharadar" / VINTAGE
LOGDIR = ROOT / "logs" / "r3"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"

TICKERS = [t for c in e.CLASSES for t in e.CLASSES[c]] + [e.CASH]
BAR = 0.05
SUGGESTIVE_BAR = 0.10
R_VOTING = 2000
N_CONTROL, R_CONTROL = 200, 500
N_PLANT, R_PLANT = 50, 500
Q_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]
Q_TARGET = 0.4
PLANT_FLOOR = 0.80
CONTROL_BAND = (0.01, 0.10)
R_BOOT, MEAN_BLOCK = 5000, 6.0
SEED_VOTING, SEED_CONTROL, SEED_PLANT, SEED_BOOT = 20261005, 777, 4242, 20261005
SUBPERIODS = [("2001-2002", "2001-01", "2002-12"),
              ("2003-2007", "2003-01", "2007-12"),
              ("2008-2012", "2008-01", "2012-12")]

LEDGER_FIELDS = ["batch_id", "date", "vintage", "registration_sha", "harness_sha",
                 "signal", "universe", "months", "ls_net_mean_bps", "p_premium",
                 "verdict"]


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def row_sha(prev, values):
    return hashlib.sha256("\x00".join([prev] + [str(v) for v in values]).encode()).hexdigest()


def halt(reason, payload):
    log("HALT: %s" % reason)
    log("Verdicts are WITHHELD and no observed statistic is emitted.")
    (LOGDIR / "t_r3_HALT.json").write_text(
        json.dumps({"halt": reason, "payload": payload}, indent=2, default=str))
    return 15


def sub_mask(months, lo, hi):
    s = months.astype(str)
    return (s >= lo) & (s <= hi)


def dating_block():
    """Section C: is the vendor's daily valuation table as-filed or restated?"""
    res = [dt.run_ticker(dt.SNAP, t) for t in dt.NAMES]
    vs = {r["verdict"] for r in res}
    lag = np.array([x["lag_days_step_minus_filing"] for r in res for x in r["steps"]])
    gap = np.array([x["gap_days_between_matched_days"] for r in res for x in r["steps"]])
    pooled = {
        "steps": int(len(lag)),
        "lag_exactly_zero": int((lag == 0).sum()),
        "lag_positive_late": int((lag > 0).sum()),
        "lag_negative_look_ahead": int((lag < 0).sum()),
        "share_dated_exactly_on_the_filing_date": float((lag == 0).mean()),
        "masking_window_days_median": float(np.median(gap)),
        "masking_window_days_p90": float(np.percentile(gap, 90)),
        "tightly_dated_gap_le_5d": int((gap <= 5).sum()),
    }
    return {
        "tool_version": dt.VERSION, "snapshot": dt.SNAP.name, "names": dt.NAMES,
        "units_note": "marketcap USD millions, netinccmnusd USD; implied earnings "
                      "scaled by 1e6 (descriptions dictionary)",
        "dating_note": "for ART the fundamentals date IS the SEC filing date; for "
                       "MRT it is the reportperiod (descriptions dictionary)",
        "detector": {"pe_half_digit": dt.PE_HALF_DIGIT,
                     "band_multiple": dt.BAND_MULTIPLE, "band_floor": dt.BAND_FLOOR,
                     "rule": "a day matches the unique reported value inside its own "
                             "pe-rounding band; several -> ambiguous, none -> unmatched"},
        "pooled": pooled, "per_ticker": res,
        "verdict_daily_table": (vs.copy().pop() if len(vs) == 1 else "MIXED"),
        "caveat": "assumes the free sample is built the same way as the paid table",
        "scope_note": "a provenance test on single-name valuation metadata; it "
                      "computes no return statistic and shares no universe with "
                      "any registered Stage R look",
    }


def main(append_ledger=True):
    t0 = time.time()
    log("run_t_r3 v%s; engine v%s" % (VERSION, e.VERSION))
    windows = json.loads((ROOT / "config" / "windows.json").read_text())
    cut = windows["half_a"]["daily_row_exclusive_max"]
    reg_sha = sha_file(LOGDIR / "registration_v4.json")
    harness_sha = sha_file(ROOT / "tools" / "etf_trend_r3.py")
    log("registration v4 sha %s" % reg_sha)
    log("engine sha           %s" % harness_sha)
    log("window exclusive max %s (half B and the holdout are never loaded)" % cut)

    # ---------------- data ----------------
    prices, spans = e.load_fund_prices(VDIR, TICKERS, cut)
    log("loaded %d tickers; max date across all = %s"
        % (len(prices), max(s[1] for s in spans.values())))
    assert max(s[1] for s in spans.values()) < cut

    panel = e.build_panel(prices)
    months = panel["months"]
    M = len(months)
    log("class entry month-ends: %s" % json.dumps(panel["entry_month"]))
    log("first month-end with >=2 classes eligible: %s" % panel["first_two_class_month"])
    log("first month with a cash (SHY) return:      %s" % panel["first_cash_return_month"])
    log("SCORED WINDOW %s .. %s  (%d months)" % (months[0], months[-1], M))

    valid, shy = panel["valid"], panel["shy_r"]
    net_bh, _, turn_bh = e.evaluate_books(panel, valid[None, :, :], e.BPS_BASE)
    net_bh, turn_bh = net_bh[0], turn_bh[0]

    # ---------------- STAGE 1: MDE ----------------
    log("STAGE 1 — MDE from the random books' spread (no observed statistic yet)")
    rates_rule = e.measure_markov_rates(panel["state_rule"], valid)
    rng_v = np.random.default_rng(SEED_VOTING)
    S_null = e.simulate_markov(rates_rule, valid, R_VOTING, rng_v)
    net_null, _, _ = e.evaluate_books(panel, S_null, e.BPS_BASE)
    d_null = e.dsr(net_null, net_bh[None, :], shy[None, :])
    d_null = d_null[np.isfinite(d_null)]
    mde = e.mde_from_null(d_null)
    log("  null dSR: median %+.4f  sd %.4f  p95 %+.4f  (R=%d)"
        % (np.median(d_null), d_null.std(ddof=1), np.percentile(d_null, 95), len(d_null)))
    log("  MDE at 80%% power = %+.4f dSR" % mde)
    del net_null

    # ---------------- STAGE 2: known-null control ----------------
    log("STAGE 2 — known-null control: %d random books, fresh %d-book null each"
        % (N_CONTROL, R_CONTROL))
    rng_c = np.random.default_rng(SEED_CONTROL)
    books = e.simulate_markov(rates_rule, valid, N_CONTROL, rng_c)
    ps = np.empty(N_CONTROL)
    for i in range(N_CONTROL):
        ps[i] = e.timing_p(panel, books[i], e.BPS_BASE, R_CONTROL, rng_c,
                           net_bh=net_bh)["p"]
        if (i + 1) % 50 == 0:
            log("  %d/%d  running share p<=0.05 = %.4f"
                % (i + 1, N_CONTROL, (ps[:i + 1] <= BAR).mean()))
    share = float((ps <= BAR).mean())
    control = {"N": N_CONTROL, "R_each": R_CONTROL, "seed": SEED_CONTROL,
               "share_p_le_0.05": share, "band": list(CONTROL_BAND),
               "mean_p": float(ps.mean()), "median_p": float(np.median(ps)),
               "in_band": bool(CONTROL_BAND[0] <= share <= CONTROL_BAND[1])}
    log("  share at p<=0.05 = %.4f  (band %.2f-%.2f)  mean p %.4f"
        % (share, CONTROL_BAND[0], CONTROL_BAND[1], ps.mean()))
    if not control["in_band"]:
        return halt("known-null control outside [0.01, 0.10]: share %.4f" % share,
                    {"control": control, "mde": mde})
    log("  CONTROL PASSES")
    del books

    # ---------------- STAGE 3: positive control ----------------
    log("STAGE 3 — positive control: calibrate q on ONE draw to dSR ~ %.2f" % Q_TARGET)
    rng_p = np.random.default_rng(SEED_PLANT)
    cal = []
    for q in Q_GRID:
        st = e.plant_states(panel, q, 1, np.random.default_rng(SEED_PLANT))[0]
        nt, _, _ = e.evaluate_books(panel, st[None, :, :], e.BPS_BASE)
        d = float(e.dsr(nt[0], net_bh, shy))
        cal.append({"q": q, "dsr": d})
        log("  q=%.2f -> planted dSR %+.4f" % (q, d))
    q_star = min(cal, key=lambda r: abs(r["dsr"] - Q_TARGET))["q"]
    log("  calibrated q = %.2f" % q_star)

    hits, pp = 0, []
    for i in range(N_PLANT):
        st = e.plant_states(panel, q_star, 1, rng_p)[0]
        o = e.timing_p(panel, st, e.BPS_BASE, R_PLANT, rng_p, net_bh=net_bh)
        pp.append(o["p"]); hits += o["p"] <= BAR
        if (i + 1) % 25 == 0:
            log("  %d/%d  detection %.2f" % (i + 1, N_PLANT, hits / (i + 1)))
    det = hits / N_PLANT
    positive = {"q_grid": cal, "q_star": q_star, "N_draws": N_PLANT,
                "R_each": R_PLANT, "seed": SEED_PLANT, "detection_rate": det,
                "floor": PLANT_FLOOR, "median_p": float(np.median(pp)),
                "passes": bool(det >= PLANT_FLOOR),
                "leak": "DELIBERATE — the planted state reads the sign of the "
                        "return it is about to earn; it proves the instrument "
                        "can see a real timing edge and is never a result"}
    log("  detection %.2f at q=%.2f (floor %.2f)" % (det, q_star, PLANT_FLOOR))
    if not positive["passes"]:
        return halt("positive control detection %.2f below %.2f" % (det, PLANT_FLOOR),
                    {"control": control, "positive": positive, "mde": mde})
    log("  POSITIVE CONTROL PASSES")

    # ---------------- STAGE 4: the observed ----------------
    log("STAGE 4 — the observed book")
    st_rule = panel["state_rule"]
    net_tr, _, turn_tr = e.evaluate_books(panel, st_rule[None, :, :], e.BPS_BASE)
    net_tr, turn_tr = net_tr[0], turn_tr[0]
    d_obs = float(e.dsr(net_tr, net_bh, shy))
    p_obs = float((d_null >= d_obs).mean())

    m_tr = e.book_metrics(net_tr, shy)
    m_bh = e.book_metrics(net_bh, shy)
    for nm, mm in (("TREND", m_tr), ("BUY-AND-HOLD", m_bh)):
        log("  %-13s CAGR %+7.2f%%  vol %6.2f%%  Sharpe %+.4f  maxDD %7.2f%%  worst12m %7.2f%%"
            % (nm, 100 * mm["cagr"], 100 * mm["ann_vol"], mm["sharpe_excess_cash"],
               100 * mm["max_drawdown"], 100 * mm["worst_rolling_12m"]))
    log("  dSR = %+.4f   timing p = %.4f" % (d_obs, p_obs))

    boot = e.paired_bootstrap(net_tr, net_bh, shy, MEAN_BLOCK, R_BOOT,
                              np.random.default_rng(SEED_BOOT))
    log("  bootstrap 90%% dSR [%+.4f, %+.4f]  mean monthly diff %+.2f bps [%+.2f, %+.2f]"
        % (boot["dsr_ci90"][0], boot["dsr_ci90"][1],
           1e4 * boot["mean_monthly_diff_point"],
           1e4 * boot["mean_monthly_diff_ci90"][0], 1e4 * boot["mean_monthly_diff_ci90"][1]))

    cash_by_cls = e.cash_time_by_class(panel, st_rule)
    sw = e.switch_count(panel, st_rule)
    years = M / 12.0
    log("  time in cash: %s" % json.dumps({k: round(v, 4) for k, v in cash_by_cls.items()}))
    log("  switches %d over %.1f years = %.2f/yr (TREND); buy-and-hold 0"
        % (sw, years, sw / years))

    subs = {}
    for nm, lo, hi in SUBPERIODS:
        msk = sub_mask(months, lo, hi)
        n = int(msk.sum())
        subs[nm] = {"months": n, "dsr": (float(e.dsr(net_tr[msk], net_bh[msk], shy[msk]))
                                         if n >= 3 else None),
                    "thin": bool(n < 12)}
        log("  sub-period %-9s n=%3d  dSR %s%s" % (
            nm, n, ("%+.4f" % subs[nm]["dsr"]) if subs[nm]["dsr"] is not None else "n/a",
            "  [THIN]" if subs[nm]["thin"] else ""))

    ex08 = ~sub_mask(months, "2008-01", "2008-12")
    d_ex08 = float(e.dsr(net_tr[ex08], net_bh[ex08], shy[ex08]))
    log("  dSR excluding calendar 2008 (n=%d) = %+.4f  [DESCRIPTIVE]"
        % (int(ex08.sum()), d_ex08))

    net_tr_s, _, _ = e.evaluate_books(panel, st_rule[None, :, :], e.BPS_STRESS)
    net_bh_s, _, _ = e.evaluate_books(panel, valid[None, :, :], e.BPS_STRESS)
    d_stress = float(e.dsr(net_tr_s[0], net_bh_s[0], shy))
    rng_vs = np.random.default_rng(SEED_VOTING)
    S_ns = e.simulate_markov(rates_rule, valid, R_VOTING, rng_vs)
    net_ns, _, _ = e.evaluate_books(panel, S_ns, e.BPS_STRESS)
    d_ns = e.dsr(net_ns, net_bh_s[0][None, :], shy[None, :])
    d_ns = d_ns[np.isfinite(d_ns)]
    p_stress = float((d_ns >= d_stress).mean())
    log("  STRESS (15 bps) dSR %+.4f  timing p %.4f" % (d_stress, p_stress))
    del S_ns, net_ns

    # six-class robustness read (RULE-CASH-NOT-A-WEIGHTED-CLASS, falsifiable)
    six = e.CLASSES.copy(); six["cash"] = [e.CASH]
    p6 = e.build_panel(prices, classes=six)
    n6b, _, _ = e.evaluate_books(p6, p6["valid"][None, :, :], e.BPS_BASE)
    n6t, _, _ = e.evaluate_books(p6, p6["state_rule"][None, :, :], e.BPS_BASE)
    d6 = float(e.dsr(n6t[0], n6b[0], p6["shy_r"]))
    log("  six-class variant (SHY as a weighted class): dSR %+.4f over %d months "
        "%s..%s  [ROBUSTNESS, not the registered design]"
        % (d6, len(p6["months"]), p6["months"][0], p6["months"][-1]))

    if p_obs <= BAR and d_obs > 0:
        verdict = "SOMETHING"
    elif p_obs <= SUGGESTIVE_BAR and d_obs > 0:
        verdict = "SUGGESTIVE"
    else:
        verdict = "NOTHING"
    log("VERDICT: %s  (p=%.4f, dSR=%+.4f, bar %.2f)" % (verdict, p_obs, d_obs, BAR))

    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID, "look": "look 3 of Q4 2026", "run_date": run_date,
        "vintage": VINTAGE, "runner_version": VERSION, "engine_version": e.VERSION,
        "registration_sha256": reg_sha, "engine_sha256": harness_sha,
        "windows": windows,
        "data": {"tickers": TICKERS, "spans": spans,
                 "manifest_sha256": sha_file(VDIR / "manifest.json")},
        "panel": {
            "classes": {k: v for k, v in e.CLASSES.items()}, "cash_proxy": e.CASH,
            "lookback_months": e.LOOKBACK,
            "class_entry_month_end": panel["entry_month"],
            "first_two_class_month_end": panel["first_two_class_month"],
            "first_cash_return_month": panel["first_cash_return_month"],
            "binding_constraint": ("the cash proxy" if panel["first_cash_return_month"]
                                   > panel["first_two_class_month"] else "two classes"),
            "first_scored_month": str(months[0]), "last_scored_month": str(months[-1]),
            "months": M,
        },
        "order_of_operations": ["MDE", "known-null control", "positive control", "observed"],
        "mde": {"dsr_at_80pct_power": mde,
                "null_median": float(np.median(d_null)),
                "null_sd": float(d_null.std(ddof=1)),
                "null_p95": float(np.percentile(d_null, 95)),
                "R": int(len(d_null)), "seed": SEED_VOTING},
        "known_null_control": control,
        "positive_control": positive,
        "observed": {
            "TREND": m_tr, "BUY_AND_HOLD": m_bh,
            "dsr": d_obs, "timing_p": p_obs,
            "bootstrap_reported_not_voting": boot,
            "time_in_cash_by_class": cash_by_cls,
            "switches_total": sw, "switches_per_year": sw / years,
            "turnover_mean_annual_trend": float(turn_tr.mean() * 12),
            "turnover_mean_annual_bh": float(turn_bh.mean() * 12),
            "dsr_by_subperiod": subs,
            "dsr_excluding_2008": {"dsr": d_ex08, "months": int(ex08.sum()),
                                   "status": "DESCRIPTIVE"},
            "stress": {"bps_per_side": e.BPS_STRESS, "dsr": d_stress,
                       "timing_p": p_stress},
        },
        "robustness_six_class_variant": {
            "dsr": d6, "months": len(p6["months"]),
            "first_scored_month": str(p6["months"][0]),
            "last_scored_month": str(p6["months"][-1]),
            "status": "ROBUSTNESS read on the rejected reading of 'cash: SHY' as a "
                      "sixth weighted class; not the registered design",
        },
        "verdict": {"verdict": verdict, "bar": BAR, "suggestive_bar": SUGGESTIVE_BAR,
                    "single_test": True,
                    "promotion": ("one registered half-B look (2013-2020)"
                                  if verdict == "SOMETHING" else "none")},
        "dating_test": dating_block(),
        "prior_stated_in_registration": "SOMETHING",
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    rp = REPORTS / ("t_r3_%s.json" % run_date)
    rp.write_text(json.dumps(report, indent=2, default=str))
    log("report %s sha %s" % (rp.relative_to(ROOT), sha_file(rp)))

    if not append_ledger:
        log("ledger append SUPPRESSED (--no-ledger): this look's row is already banked")
        log("DONE wall %.1fs" % (time.time() - t0))
        return 0

    lines = [l for l in LEDGER.read_text().strip().split("\n") if l]
    prev = lines[-1].split(",")[-1]
    vals = [BATCH_ID, run_date, VINTAGE, reg_sha[:16], harness_sha[:16],
            "TREND10_LONG_OR_CASH", "etf16_5class_halfA", M,
            "%.4f" % (1e4 * boot["mean_monthly_diff_point"]), "%.4f" % p_obs, verdict]
    rs = row_sha(prev, vals)
    with open(LEDGER, "a") as f:
        f.write(",".join([str(x) for x in vals] + [prev, rs]) + "\n")
    log("ledger appended 1 row; head %s" % rs[:16])
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(append_ledger="--no-ledger" not in sys.argv))
