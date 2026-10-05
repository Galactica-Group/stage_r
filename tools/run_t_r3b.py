"""
================================================================================
Script:       run_t_r3b.py
Version:      1.0.0
Phase:        Stage R — Timing Triage
Purpose:      Run T-R3b: the half-B confirmation of T-R3. The same ETF ten-month
              trend rule, unchanged, scored on 2013-01..2020-12.
Entry Point:  Yes (python tools/run_t_r3b.py)
Dependencies: tools/etf_trend_r3.py (UNCHANGED, sha-asserted), numpy, pandas
Last Sprint:  Sprint R3b — Half-B confirmation (2026-10-05)
================================================================================

This runner redesigns nothing. Everything that defines the measurement — the
universe, eligibility, the weights, the rule, the cash leg, the costs, the
statistic, the voting null, the R values, the seeds and both controls — is
inherited from registration v4 (c3349c76) through the engine, which is NOT
edited here and whose sha256 is asserted before any data is read.

The only new thing is the window. The panel is built exactly as registered from
the full pre-2021 history and the scored rows are then SLICED to the half-B
months. Slicing after the build is what makes the state for 2013-01 identical to
the registered construction: it is formed from month-end closes through 2012-12,
which are signal inputs and are never scored.

Nothing dated 2021-01-01 or later is loaded. The engine filters at read; this
runner asserts it again on the built panel, and a test proves the assertion
bites on a constructed 2021 row.
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
import etf_trend_r3 as e  # noqa: E402

VERSION = "1.0.0"
BATCH_ID = "T-R3b"
VINTAGE = "funds_20261005f"
VDIR = ROOT / "data" / "sharadar" / VINTAGE
LOGDIR = ROOT / "logs" / "r3b"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"

# the engine is frozen for this sprint; a change is a HALT, not a surprise
ENGINE_SHA = "d024f6ff265a17ee4533c60cf1dc19b7e8f666d0619ede680efe4a5fdf2e91e8"
V4_SHA = "c3349c76745f93ca9bded73b2ee38ed52cb3ad2b1a273c42ddcea72a93ee614b"

HOLDOUT_START = "2021-01-01"
HOLDOUT_MONTH = "2021-01"
HALF_B = ("2013-01", "2020-12")
HALF_B_MONTHS = 96
JOINED = ("2002-08", "2020-12")
JOINED_MONTHS = 221

TICKERS = [t for c in e.CLASSES for t in e.CLASSES[c]] + [e.CASH]
BAR = 0.05
R_VOTING = 2000
N_CONTROL, R_CONTROL = 200, 500
N_PLANT, R_PLANT = 50, 500
Q_GRID = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65, 0.70, 0.75, 0.80]
Q_TARGET = 0.4
PLANT_FLOOR = 0.80
CONTROL_BAND = (0.01, 0.10)
R_BOOT, MEAN_BLOCK = 5000, 6.0
SEED_VOTING, SEED_CONTROL, SEED_PLANT, SEED_BOOT = 20261005, 777, 4242, 20261005
YEARS = list(range(2013, 2021))

LEDGER_FIELDS = ["batch_id", "date", "vintage", "registration_sha", "harness_sha",
                 "signal", "universe", "months", "ls_net_mean_bps", "p_premium",
                 "verdict"]

SLICED_KEYS = ("months", "w", "valid", "state_rule", "A", "shy_r", "n_present")


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
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (LOGDIR / "t_r3b_HALT.json").write_text(
        json.dumps({"halt": reason, "payload": payload}, indent=2, default=str))
    return 15


def assert_engine_unchanged(path=None, expected=ENGINE_SHA):
    """The measurement is frozen for this sprint. Pure; raises on any edit."""
    p = Path(path) if path else (ROOT / "tools" / "etf_trend_r3.py")
    got = sha_file(p)
    if got != expected:
        raise AssertionError(
            "engine %s changed: expected sha256 %s, found %s — the R3b brief "
            "makes any engine edit a HALT" % (p.name, expected, got))
    return got


def assert_no_holdout(months, holdout_month=HOLDOUT_MONTH):
    """No scored or warm-up month may reach the sealed holdout."""
    s = pd.Index(months).astype(str)
    bad = s[s >= holdout_month]
    if len(bad):
        raise AssertionError(
            "%d month(s) at or past the sealed holdout %s reached the panel: %s"
            % (len(bad), holdout_month, list(bad[:5])))
    return True


def slice_panel(panel, lo, hi):
    """Restrict a built panel's SCORED rows to [lo, hi] inclusive, by month.

    Eligibility, the weights and the rule's states were computed on the full
    history before this slice, so the state for the first retained month still
    reads the month-end closes that precede it.
    """
    s = panel["months"].astype(str)
    m = np.asarray((s >= lo) & (s <= hi))
    if not m.any():
        raise AssertionError("no months in [%s, %s]" % (lo, hi))
    first, last = int(np.flatnonzero(m)[0]), int(np.flatnonzero(m)[-1])
    if not m[first:last + 1].all():
        raise AssertionError("requested window is not contiguous")
    out = dict(panel)
    for k in SLICED_KEYS:
        out[k] = panel[k][m]
    out["window"] = [lo, hi]
    return out


def books_for(panel, bps=e.BPS_BASE):
    """Net monthly returns of the TREND and BUY-AND-HOLD books."""
    nt, _, tt = e.evaluate_books(panel, panel["state_rule"][None, :, :], bps)
    nb, _, tb = e.evaluate_books(panel, panel["valid"][None, :, :], bps)
    return nt[0], nb[0], tt[0], tb[0]


def six_class_dsr(prices, lo, hi, bps=e.BPS_BASE):
    six = e.CLASSES.copy()
    six["cash"] = [e.CASH]
    p6 = slice_panel(e.build_panel(prices, classes=six), lo, hi)
    nt, nb, _, _ = books_for(p6, bps)
    return float(e.dsr(nt, nb, p6["shy_r"])), len(p6["months"])


def main(append_ledger=True):
    t0 = time.time()
    log("run_t_r3b v%s; engine v%s" % (VERSION, e.VERSION))
    engine_sha = assert_engine_unchanged()
    log("engine sha ASSERTED UNCHANGED %s" % engine_sha)
    reg_sha = sha_file(LOGDIR / "registration_v5.json")
    v4 = sha_file(ROOT / "logs" / "r3" / "registration_v4.json")
    if v4 != V4_SHA:
        return halt("registration v4 changed: %s" % v4, {"found": v4})
    windows = json.loads((ROOT / "config" / "windows.json").read_text())
    log("registration v5 sha %s" % reg_sha)
    log("registration v4 sha %s (inherited, unchanged)" % v4)
    log("holdout %s onward is NEVER loaded" % HOLDOUT_START)

    # ---------------- data ----------------
    prices, spans = e.load_fund_prices(VDIR, TICKERS, HOLDOUT_START)
    loaded_max = max(s[1] for s in spans.values())
    log("loaded %d tickers; max date across all = %s" % (len(prices), loaded_max))
    assert loaded_max < HOLDOUT_START

    full = e.build_panel(prices)
    assert_no_holdout(full["months"])
    log("panel built as registered: %s .. %s (%d months)"
        % (full["months"][0], full["months"][-1], len(full["months"])))
    if (str(full["months"][0]), str(full["months"][-1])) != JOINED:
        return halt("joined window is %s..%s, expected %s..%s"
                    % (full["months"][0], full["months"][-1], *JOINED), {})
    if len(full["months"]) != JOINED_MONTHS:
        return halt("joined months %d, expected %d" % (len(full["months"]), JOINED_MONTHS), {})

    panel = slice_panel(full, *HALF_B)
    assert_no_holdout(panel["months"])
    M = len(panel["months"])
    log("SCORED WINDOW %s .. %s  (%d months)" % (panel["months"][0], panel["months"][-1], M))
    if M != HALF_B_MONTHS:
        return halt("scored months %d, expected %d" % (M, HALF_B_MONTHS), {})
    log("classes eligible every scored month: %s  (class weight %.4f throughout)"
        % (bool((panel["n_present"] == 5).all()), 1 / 5))

    valid, shy = panel["valid"], panel["shy_r"]
    net_bh, _, turn_bh = e.evaluate_books(panel, valid[None, :, :], e.BPS_BASE)
    net_bh, turn_bh = net_bh[0], turn_bh[0]

    # ---------------- STAGE 1: MDE ----------------
    log("STAGE 1 — MDE from the half-B null's spread (no observed statistic yet)")
    rates_rule = e.measure_markov_rates(panel["state_rule"], valid)
    S_null = e.simulate_markov(rates_rule, valid, R_VOTING, np.random.default_rng(SEED_VOTING))
    net_null, _, _ = e.evaluate_books(panel, S_null, e.BPS_BASE)
    d_null = e.dsr(net_null, net_bh[None, :], shy[None, :])
    d_null = d_null[np.isfinite(d_null)]
    mde = e.mde_from_null(d_null)
    log("  null dSR: median %+.4f  sd %.4f  p95 %+.4f  (R=%d)"
        % (np.median(d_null), d_null.std(ddof=1), np.percentile(d_null, 95), len(d_null)))
    log("  MDE at 80%% power = %+.4f dSR" % mde)
    del net_null, S_null

    # ---------------- STAGE 2: known-null control ----------------
    log("STAGE 2 — known-null control: %d random books, fresh %d-book null each"
        % (N_CONTROL, R_CONTROL))
    rng_c = np.random.default_rng(SEED_CONTROL)
    books = e.simulate_markov(rates_rule, valid, N_CONTROL, rng_c)
    ps = np.empty(N_CONTROL)
    for i in range(N_CONTROL):
        ps[i] = e.timing_p(panel, books[i], e.BPS_BASE, R_CONTROL, rng_c, net_bh=net_bh)["p"]
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
    log("STAGE 3 — positive control: calibrate q on ONE half-B draw to dSR ~ %.2f" % Q_TARGET)
    rng_p = np.random.default_rng(SEED_PLANT)
    cal = []
    for q in Q_GRID:
        st = e.plant_states(panel, q, 1, np.random.default_rng(SEED_PLANT))[0]
        nt, _, _ = e.evaluate_books(panel, st[None, :, :], e.BPS_BASE)
        cal.append({"q": q, "dsr": float(e.dsr(nt[0], net_bh, shy))})
        log("  q=%.2f -> planted dSR %+.4f" % (q, cal[-1]["dsr"]))
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
    positive = {"q_grid": cal, "q_star": q_star, "N_draws": N_PLANT, "R_each": R_PLANT,
                "seed": SEED_PLANT, "detection_rate": det, "floor": PLANT_FLOOR,
                "median_p": float(np.median(pp)), "passes": bool(det >= PLANT_FLOOR),
                "leak": "DELIBERATE — the planted state reads the sign of the return "
                        "it is about to earn; it proves the instrument can see a real "
                        "timing edge and is never a result"}
    log("  detection %.2f at q=%.2f (floor %.2f)" % (det, q_star, PLANT_FLOOR))
    if not positive["passes"]:
        return halt("positive control detection %.2f below %.2f" % (det, PLANT_FLOOR),
                    {"control": control, "positive": positive, "mde": mde})
    log("  POSITIVE CONTROL PASSES")

    # ---------------- STAGE 4: the observed ----------------
    log("STAGE 4 — the observed book")
    net_tr, _, turn_tr = e.evaluate_books(panel, panel["state_rule"][None, :, :], e.BPS_BASE)
    net_tr, turn_tr = net_tr[0], turn_tr[0]
    d_obs = float(e.dsr(net_tr, net_bh, shy))
    p_obs = float((d_null >= d_obs).mean())

    m_tr, m_bh = e.book_metrics(net_tr, shy), e.book_metrics(net_bh, shy)
    for nm, mm in (("TREND", m_tr), ("BUY-AND-HOLD", m_bh)):
        log("  %-13s CAGR %+7.2f%%  vol %6.2f%%  Sharpe %+.4f  maxDD %7.2f%%  worst12m %7.2f%%"
            % (nm, 100 * mm["cagr"], 100 * mm["ann_vol"], mm["sharpe_excess_cash"],
               100 * mm["max_drawdown"], 100 * mm["worst_rolling_12m"]))
    log("  dSR = %+.4f   timing p = %.4f   (MDE %+.4f)" % (d_obs, p_obs, mde))

    boot = e.paired_bootstrap(net_tr, net_bh, shy, MEAN_BLOCK, R_BOOT,
                              np.random.default_rng(SEED_BOOT))
    log("  bootstrap 90%% dSR [%+.4f, %+.4f]  mean monthly diff %+.2f bps [%+.2f, %+.2f]"
        % (boot["dsr_ci90"][0], boot["dsr_ci90"][1],
           1e4 * boot["mean_monthly_diff_point"],
           1e4 * boot["mean_monthly_diff_ci90"][0], 1e4 * boot["mean_monthly_diff_ci90"][1]))

    cash_by_cls = e.cash_time_by_class(panel, panel["state_rule"])
    sw = e.switch_count(panel, panel["state_rule"])
    years = M / 12.0
    log("  time in cash: %s" % json.dumps({k: round(v, 4) for k, v in cash_by_cls.items()}))
    log("  switches %d over %.1f years = %.2f/yr (TREND); buy-and-hold 0"
        % (sw, years, sw / years))

    ms = panel["months"].astype(str)
    by_year = {}
    for y in YEARS:
        k = np.asarray(ms.str.startswith(str(y)))
        n = int(k.sum())
        by_year[str(y)] = {"months": n,
                           "dsr": float(e.dsr(net_tr[k], net_bh[k], shy[k])) if n >= 3 else None,
                           "trend_net_return": float(np.prod(1 + net_tr[k]) - 1),
                           "bh_net_return": float(np.prod(1 + net_bh[k]) - 1)}
        r = by_year[str(y)]
        log("  %d  n=%2d  dSR %+8.4f   TREND %+7.2f%%   BUY-AND-HOLD %+7.2f%%"
            % (y, n, r["dsr"], 100 * r["trend_net_return"], 100 * r["bh_net_return"]))

    ex20 = ~np.asarray(ms.str.startswith("2020"))
    d_ex20 = float(e.dsr(net_tr[ex20], net_bh[ex20], shy[ex20]))
    log("  dSR excluding calendar 2020 (n=%d) = %+.4f  [DESCRIPTIVE]"
        % (int(ex20.sum()), d_ex20))

    nts, nbs, _, _ = books_for(panel, e.BPS_STRESS)
    d_stress = float(e.dsr(nts, nbs, shy))
    S_ns = e.simulate_markov(rates_rule, valid, R_VOTING, np.random.default_rng(SEED_VOTING))
    net_ns, _, _ = e.evaluate_books(panel, S_ns, e.BPS_STRESS)
    d_ns = e.dsr(net_ns, nbs[None, :], shy[None, :])
    d_ns = d_ns[np.isfinite(d_ns)]
    p_stress = float((d_ns >= d_stress).mean())
    log("  STRESS (15 bps) dSR %+.4f  timing p %.4f" % (d_stress, p_stress))
    del S_ns, net_ns

    d6, n6 = six_class_dsr(prices, *HALF_B)
    log("  six-class variant: dSR %+.4f over %d months  [REPORTED, NOT VOTED]" % (d6, n6))

    if d_obs <= 0:
        verdict = "REJECT"
    elif p_obs <= BAR:
        verdict = "CONFIRM"
    else:
        verdict = "CONSISTENT"
    log("VERDICT: %s  (p=%.4f, dSR=%+.4f, bar %.2f)" % (verdict, p_obs, d_obs, BAR))

    # ---------------- STAGE 5: joined, DESCRIPTIVE ONLY ----------------
    log("STAGE 5 — the two halves joined, %s..%s  [DESCRIPTIVE ONLY: half A has been seen]"
        % JOINED)
    jt, jb, _, _ = books_for(full, e.BPS_BASE)
    jshy = full["shy_r"]
    j_tr, j_bh = e.book_metrics(jt, jshy), e.book_metrics(jb, jshy)
    d_join = float(e.dsr(jt, jb, jshy))
    j_boot = e.paired_bootstrap(jt, jb, jshy, MEAN_BLOCK, R_BOOT,
                                np.random.default_rng(SEED_BOOT))
    for nm, mm in (("TREND", j_tr), ("BUY-AND-HOLD", j_bh)):
        log("  %-13s CAGR %+7.2f%%  vol %6.2f%%  Sharpe %+.4f  maxDD %7.2f%%"
            % (nm, 100 * mm["cagr"], 100 * mm["ann_vol"], mm["sharpe_excess_cash"],
               100 * mm["max_drawdown"]))
    log("  joined dSR %+.4f   bootstrap 90%% [%+.4f, %+.4f]   [NO VERDICT ATTACHES]"
        % (d_join, j_boot["dsr_ci90"][0], j_boot["dsr_ci90"][1]))

    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID,
        "look": "half-B confirmation of T-R3 (not a discovery look; spends no Q4 look)",
        "run_date": run_date, "vintage": VINTAGE,
        "runner_version": VERSION, "engine_version": e.VERSION,
        "registration_v5_sha256": reg_sha, "registration_v4_sha256": v4,
        "engine_sha256": engine_sha, "engine_sha_asserted_unchanged": True,
        "vendor_requests_this_sprint": 0,
        "windows_json": windows,
        "windows_json_sha256": sha_file(ROOT / "config" / "windows.json"),
        "data": {"tickers": TICKERS, "spans": spans,
                 "manifest_sha256": sha_file(VDIR / "manifest.json"),
                 "holdout_exclusive_min": HOLDOUT_START,
                 "max_date_loaded": loaded_max},
        "panel": {
            "classes": {k: v for k, v in e.CLASSES.items()}, "cash_proxy": e.CASH,
            "lookback_months": e.LOOKBACK,
            "built_window": [str(full["months"][0]), str(full["months"][-1])],
            "built_months": len(full["months"]),
            "scored_window": [str(panel["months"][0]), str(panel["months"][-1])],
            "scored_months": M,
            "all_five_classes_eligible_every_scored_month": bool((panel["n_present"] == 5).all()),
            "class_weight_throughout": 1 / 5,
            "warm_up_note": "the state for 2013-01 is formed from month-end closes "
                            "through 2012-12; pre-2013 rows are signal inputs only "
                            "and are never scored",
            "first_month_buy_in_note": "the first scored month pays a full 1.0 "
                                       "turnover buy-in in BOTH books, the same "
                                       "convention half A used at 2002-08",
        },
        "order_of_operations": ["MDE", "known-null control", "positive control",
                               "observed", "joined (descriptive)"],
        "mde": {"dsr_at_80pct_power": mde, "null_median": float(np.median(d_null)),
                "null_sd": float(d_null.std(ddof=1)),
                "null_p95": float(np.percentile(d_null, 95)),
                "R": int(len(d_null)), "seed": SEED_VOTING},
        "known_null_control": control,
        "positive_control": positive,
        "observed": {
            "TREND": m_tr, "BUY_AND_HOLD": m_bh, "dsr": d_obs, "timing_p": p_obs,
            "bootstrap_reported_not_voting": boot,
            "time_in_cash_by_class": cash_by_cls,
            "switches_total": sw, "switches_per_year": sw / years,
            "turnover_mean_annual_trend": float(turn_tr.mean() * 12),
            "turnover_mean_annual_bh": float(turn_bh.mean() * 12),
            "dsr_by_calendar_year": by_year,
            "dsr_excluding_2020": {"dsr": d_ex20, "months": int(ex20.sum()),
                                   "status": "DESCRIPTIVE"},
            "stress": {"bps_per_side": e.BPS_STRESS, "dsr": d_stress, "timing_p": p_stress},
            "six_class_variant": {"dsr": d6, "months": n6,
                                  "status": "REPORTED, NOT VOTED"},
        },
        "joined_descriptive_only": {
            "window": [str(full["months"][0]), str(full["months"][-1])],
            "months": len(full["months"]),
            "TREND": j_tr, "BUY_AND_HOLD": j_bh, "dsr": d_join,
            "bootstrap": j_boot,
            "status": "DESCRIPTIVE ONLY — half A has been seen, so no verdict "
                      "attaches to the joined window and it votes on nothing",
        },
        "verdict": {"verdict": verdict, "bar": BAR, "single_test": True,
                    "rule": {"CONFIRM": "p <= 0.05 and dSR > 0",
                             "CONSISTENT": "dSR > 0 and p > 0.05",
                             "REJECT": "dSR <= 0"}},
        "prior_stated_in_registration": "CONSISTENT or REJECT",
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    rp = REPORTS / ("t_r3b_%s.json" % run_date)
    rp.write_text(json.dumps(report, indent=2, default=str))
    log("report %s sha %s" % (rp.relative_to(ROOT), sha_file(rp)))

    if not append_ledger:
        log("ledger append SUPPRESSED (--no-ledger)")
        log("DONE wall %.1fs" % (time.time() - t0))
        return 0

    lines = [l for l in LEDGER.read_text().strip().split("\n") if l]
    prev = lines[-1].split(",")[-1]
    vals = [BATCH_ID, run_date, VINTAGE, reg_sha[:16], engine_sha[:16],
            "TREND10_LONG_OR_CASH", "etf16_5class_halfB", M,
            "%.4f" % (1e4 * boot["mean_monthly_diff_point"]), "%.4f" % p_obs, verdict]
    rs = row_sha(prev, vals)
    with open(LEDGER, "a") as f:
        f.write(",".join([str(x) for x in vals] + [prev, rs]) + "\n")
    log("ledger appended 1 row; head %s" % rs[:16])
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(append_ledger="--no-ledger" not in sys.argv))
