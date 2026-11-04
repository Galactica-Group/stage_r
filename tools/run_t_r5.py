"""
================================================================================
Script:       run_t_r5.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R5 runner in two stages: the registered controls, and then —
              only on their written pass receipt — the observed half-A read.
Entry Point:  Yes — python tools/run_t_r5.py --stage controls|observed
                     --registration logs/r5/registration_v1.json --sha <sha256>
Dependencies: stdlib, numpy, pandas, tools/item408_event_study_r5.py,
              tools/ledger_chain.py. Imports nothing from any other repository.
Last Sprint:  Sprint R5-2 — T-R5 10b5-1 plan terminations: seal, controls,
              half-A read (2026-11-04)
================================================================================

The two stages make the registered order of operations mechanical rather than a
promise. --stage controls walks steps 1 to 5 — resolution, the market index, the
MDE, the known-null control, the positive control — and writes
logs/r5/controls_pass.json on a pass or logs/r5/controls_halt.json on a halt,
exiting non-zero in the second case. --stage observed refuses to start unless
that receipt exists AND its registration_sha equals the live seal's, so a read
cannot be taken against a registration the controls never saw.

Steps 1 to 4 form no figure of the observed event set: the counts are counts,
the market index knows nothing of any disclosure, and the MDE and the control p
values live entirely on shifted dates. Step 5 must form the observed mean inside
the planted mean it tests, because the registered plant is added to the real
events' own voted windows; that figure is used and discarded, printed in no log
line and written into no receipt.
================================================================================
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import item408_event_study_r5 as e  # noqa: E402
import ledger_chain as lc  # noqa: E402

VERSION = "1.0.0"
BATCH_ID = "T-R5"
LOGDIR = ROOT / "logs" / "r5"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"
RECEIPT = LOGDIR / "controls_pass.json"
HALTFILE = LOGDIR / "controls_halt.json"

HEX64 = re.compile(r"\b([0-9a-f]{64})\b")


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT))
    except ValueError:
        return str(p)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# The seal's data pins
# ---------------------------------------------------------------------------
def data_paths_and_pins(reg):
    """Paths and sha256 pins, read out of the registration's own data strings."""
    ev_s = reg["data"]["events"]
    px_s = reg["data"]["prices"]
    tk_s = reg["data"]["tickers"]
    ev_path = re.search(r"(data/[\w/\-.]+events\.parquet)", ev_s).group(1)
    man_path = re.search(r"(manifests/[\w/\-.]+\.json)", ev_s).group(1)
    snap = re.search(r"(snapshot_\w+)", px_s).group(1)
    ev_hex = HEX64.findall(ev_s)
    if len(ev_hex) != 2:
        raise AssertionError("the events string must pin two sha256s, found %d"
                             % len(ev_hex))
    return {
        "events": (ROOT / ev_path, ev_hex[0]),
        "manifest": (ROOT / man_path, ev_hex[1]),
        "stocks": (ROOT / "data" / "sharadar" / snap / "bulk_stocks.parquet",
                   HEX64.findall(px_s)[0]),
        "tickers": (ROOT / "data" / "sharadar" / snap / "bulk_tickers.parquet",
                    HEX64.findall(tk_s)[0]),
        "snapshot": snap,
    }


def assert_data_pins(pins):
    out = {}
    for name in ("events", "manifest", "stocks", "tickers"):
        path, want = pins[name]
        got = e.sha256_file(path)
        if got != want:
            raise AssertionError("%s sha256 mismatch: %s has %s, the seal pins %s"
                                 % (name, rel(path), got, want))
        out[name] = got
        log("  pin OK %-9s %s %s" % (name, got[:16], rel(path)))
    return out


# ---------------------------------------------------------------------------
# Steps 1 and 2: resolution, then the market index
# ---------------------------------------------------------------------------
def prepare(reg_path, sha):
    """Seal, data pins, resolution counts, the market index, the window book.

    Nothing here forms a return of an observed event: the counts are counts and
    the market index is built from the whole snapshot.
    """
    reg = e.load_registration(reg_path, sha)
    log("registration %s sha %s" % (Path(reg_path).name, reg["_sha256"]))
    pins = data_paths_and_pins(reg)
    shas = assert_data_pins(pins)

    log("loading the frozen snapshot (five columns, cast at read)")
    t0 = time.time()
    book = e.load_prices(pins["stocks"][0])
    log("  %d ticker-days, %d tickers, %d trading days %s .. %s  (%.1fs)"
        % (len(book.code), len(book.vocab), book.L, book.date_str(0),
           book.date_str(book.L - 1), time.time() - t0))

    ev = pd.read_parquet(pins["events"][0])
    sep = pd.read_parquet(pins["tickers"][0],
                          columns=["table", "permaticker", "ticker", "secfilings"])
    log("STEP 1 - RESOLUTION: %d rows in the vintage" % len(ev))
    res, counts = e.resolve_events(ev, sep, book)
    for k in ("rows_in_vintage", "A_termination_flag", "B_no_adoption_flag",
              "C_parseable_T", "D_lag_ge_zero", "E_form_10Q_or_10K",
              "F_direction_split", "F_after_purchase_exclusion",
              "G_one_event_per_filing", "G_filing_date_on_the_calendar",
              "H_mapping", "I_unique_on_permaticker_D", "I_collapsed",
              "J_history_250_rows_before_T", "K_price_floor_on_D_plus_1",
              "K_dollar_volume_floor", "L_complete_D1_to_D61"):
        log("  %-32s %s" % (k, counts[k]))

    halves = e.split_halves(res, reg)
    for k in ("half_A", "half_B", "holdout"):
        counts["M_%s" % k] = int(len(halves[k]))
        log("  %-32s %d" % ("M_%s" % k, len(halves[k])))

    log("STEP 2 - MARKET INDEX (equal weight, whole snapshot)")
    cache = pins["events"][0].parent / "market_ew.parquet"
    if cache.exists():
        mf = pd.read_parquet(cache)
        if len(mf) != book.L:
            raise AssertionError("cached market index has %d rows, calendar has %d"
                                 % (len(mf), book.L))
        market = {"mean_log_return": mf["mean_log_return"].to_numpy(dtype=float),
                  "n_tickers": mf["n_tickers"].to_numpy(dtype=np.int64)}
        log("  cache hit %s" % rel(cache))
    else:
        t0 = time.time()
        market = e.build_market_index(book)
        mf = e.market_frame(book, market)
        mf.to_parquet(cache, index=False)
        log("  built in %.1fs and cached to %s" % (time.time() - t0, rel(cache)))
    msha = e.sha256_file(cache)
    nt = market["n_tickers"]
    log("  market_ew.parquet sha %s; dates %s .. %s; days with any ticker %d"
        % (msha, mf["date"].iloc[0], mf["date"].iloc[-1], int((nt > 0).sum())))
    for d in ("2005-06-15", "2015-06-15", "2025-06-16"):
        hit = mf.index[mf["date"] == d]
        if len(hit):
            log("    tickers averaged on %s: %d" % (d, int(nt[hit[0]])))

    return reg, pins, shas, counts, halves, book, market, msha, mf


def build_window_book(book, market, res_all, half):
    """Cumulative abnormal returns for half A's issuers, with the real-event mask."""
    wb = e.WindowBook(book, list(half["ticker"]), market)
    sub = res_all.loc[res_all["ticker"].isin(set(wb.tickers))]
    groups_all = np.array([wb.gidx[t] for t in sub["ticker"]], dtype=np.int64)
    t_idx = np.array([book.roll_forward(t) if book.roll_forward(t) is not None
                      else 0 for t in sub["T"]], dtype=np.int64)
    d_idx_all = sub["D_idx"].to_numpy(dtype=np.int64)
    wb.set_blocked(groups_all, t_idx, d_idx_all)
    groups = np.array([wb.gidx[t] for t in half["ticker"]], dtype=np.int64)
    d_idx = half["D_idx"].to_numpy(dtype=np.int64)
    return wb, groups, d_idx, int(len(sub))


# ---------------------------------------------------------------------------
# Stage: controls
# ---------------------------------------------------------------------------
def stage_controls(args):
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    reg, pins, shas, counts, halves, book, market, msha, _ = prepare(
        args.registration, args.sha)

    half = halves["half_A"]
    n_events = int(len(half))
    bar = float(reg["verdict"]["bar"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])
    kn = reg["controls"]["known_null"]
    pc = reg["controls"]["positive_control"]
    band = tuple(kn["band"])

    if n_events < e.MIN_EVENTS:
        return halt_out(reg, "half A resolved %d events, the registration's "
                        "minimum is %d" % (n_events, e.MIN_EVENTS),
                        None, None, None, counts, n_events, msha, t0)

    wb, groups, d_idx, n_real = build_window_book(book, market, _all_resolved(halves),
                                                  half)
    log("window book: %d issuers, %d real events masked as T-60..D+61"
        % (len(wb.tickers), n_real))

    # ---- step 3: MDE from the null's own spread ----
    log("STEP 3 - MDE from the common-shift null (%d replicates, seed %d)"
        % (R_vote, seed_vote))
    nullo = e.common_shift_null(wb, groups, d_idx, R_vote,
                               np.random.default_rng(seed_vote))
    nm = nullo["mcar"]
    mde = e.mde_from_null(nm)
    null_spread = {"median": float(np.median(nm)), "sd": float(nm.std(ddof=1)),
                   "p95": float(np.percentile(nm, 95)), "R": int(len(nm)),
                   "redraws": nullo["redraws"]}
    log("  null mCAR: median %+.6f  sd %.6f  p95 %+.6f  (R=%d, redraws %d)"
        % (null_spread["median"], null_spread["sd"], null_spread["p95"],
           null_spread["R"], null_spread["redraws"]))
    log("  MDE at 80%% power = %+.6f mCAR" % mde)

    # ---- step 4: known-null control ----
    log("STEP 4 - known-null control: %d pseudo sets, fresh %d-replicate null each"
        % (kn["N"], kn["R_each"]))
    rng_c = np.random.default_rng(int(kn["seed"]))
    ps = np.empty(int(kn["N"]))
    for i in range(int(kn["N"])):
        while True:
            k_i = e.draw_shift(rng_c)
            m_i, ret, _ = e.shift_mcar(wb, groups, d_idx, k_i, n_events)
            if np.isfinite(m_i) and ret >= e.RETENTION_MIN:
                break
        nn = e.common_shift_null(wb, groups, d_idx, int(kn["R_each"]), rng_c,
                                base_shift=k_i)
        ps[i] = e.null_p(m_i, nn["mcar"])
        if (i + 1) % 50 == 0:
            log("  %d/%d  running share p<=%.2f = %.4f"
                % (i + 1, kn["N"], bar, (ps[:i + 1] <= bar).mean()))
    share = float((ps <= bar).mean())
    known_null = {"N": int(kn["N"]), "R_each": int(kn["R_each"]),
                  "seed": int(kn["seed"]), "share_p_le_bar": share,
                  "band": list(band), "mean_p": float(ps.mean()),
                  "median_p": float(np.median(ps)),
                  "in_band": e.band_ok(share, band)}
    log("  share at p<=%.2f = %.4f  (band %.2f-%.2f)  mean p %.4f  median p %.4f"
        % (bar, share, band[0], band[1], ps.mean(), np.median(ps)))
    if not known_null["in_band"]:
        return halt_out(reg, "known-null control share %.4f outside [%.2f, %.2f]"
                        % (share, band[0], band[1]),
                        mde, known_null, None, counts, n_events, msha, t0,
                        null_spread)
    log("  KNOWN-NULL CONTROL PASSES")

    # ---- step 5: positive control ----
    target = float(pc["target_mCAR"])
    log("STEP 5 - positive control: calibrate X on ONE draw to a planted mean "
        "of %.4f" % target)
    cal = e.calibrate_X(n_events, target, int(pc["seed"]))
    log("  X calibrated = %.6f  (reference draw mean of U(0,2) = %.6f; "
        "one-draw planted mean = %.6f)"
        % (cal["X"], cal["reference_mean_u"], cal["one_draw_planted_mean"]))
    # the one observed figure this stage must form: the plant rides the real
    # events' own voted windows. Used here, printed nowhere, written nowhere.
    obs_vec = e.voted_returns(wb, groups, d_idx)
    if not np.isfinite(obs_vec).all():
        raise AssertionError("%d half-A events have an incomplete voted window "
                             "after the completeness filter"
                             % int((~np.isfinite(obs_vec)).sum()))
    _internal_obs = float(obs_vec.mean())
    rng_p = np.random.default_rng(int(pc["seed"]))
    hits, pp = 0, []
    for i in range(int(pc["N_draws"])):
        drift = e.plant_drift(n_events, cal["X"], rng_p)
        planted = _internal_obs + float(drift.mean())
        nn = e.common_shift_null(wb, groups, d_idx, int(pc["R_each"]), rng_p)
        p = e.null_p(planted, nn["mcar"])
        pp.append(p)
        hits += p <= bar
        if (i + 1) % 25 == 0:
            log("  %d/%d  detection %.2f" % (i + 1, pc["N_draws"], hits / (i + 1)))
    det = hits / int(pc["N_draws"])
    positive = {"X_calibrated": cal["X"],
                "reference_mean_u": cal["reference_mean_u"],
                "one_draw_planted_mean": cal["one_draw_planted_mean"],
                "planted_mean_if_X_were_the_target": cal[
                    "planted_mean_at_X_equals_target"],
                "target_mCAR": target, "N_draws": int(pc["N_draws"]),
                "R_each": int(pc["R_each"]), "seed": int(pc["seed"]),
                "detection_rate": det, "floor": float(pc["detection_floor"]),
                "median_p": float(np.median(pp)),
                "passes": e.detection_ok(det, pc["detection_floor"]),
                "leak": pc["leak"]}
    log("  detection %.2f over %d draws (floor %.2f); median p %.4f"
        % (det, pc["N_draws"], pc["detection_floor"], np.median(pp)))
    if not positive["passes"]:
        return halt_out(reg, "positive control detection %.2f below floor %.2f"
                        % (det, pc["detection_floor"]),
                        mde, known_null, positive, counts, n_events, msha, t0,
                        null_spread)
    log("  POSITIVE CONTROL PASSES")

    receipt = {
        "registration_sha": reg["_sha256"],
        "n_events": n_events,
        "mde": mde,
        "known_null_share": known_null["share_p_le_bar"],
        "X_calibrated": positive["X_calibrated"],
        "detection": det,
        "utc": utc_now(),
        "runner_version": VERSION, "engine_version": e.VERSION,
        "engine_sha256": e.sha256_file(ROOT / "tools" / "item408_event_study_r5.py"),
        "data_sha256": shas, "market_ew_sha256": msha,
        "resolution_counts": counts,
        "null_spread": null_spread,
        "known_null_control": known_null,
        "positive_control": positive,
        "note": "steps 1-4 formed no figure of the observed event set; step 5 "
                "formed the observed mean inside the planted mean it tests, "
                "because the registered plant rides the real events' own voted "
                "windows, and neither printed nor recorded it",
        "wall_seconds": round(time.time() - t0, 1),
    }
    if HALTFILE.exists():
        HALTFILE.unlink()
    RECEIPT.write_text(json.dumps(receipt, indent=2, default=str))
    log("receipt %s sha %s" % (rel(RECEIPT), e.sha256_file(RECEIPT)))
    log("CONTROLS PASS - the observed stage is unlocked")
    return 0


def _all_resolved(halves):
    return pd.concat([halves["half_A"], halves["half_B"], halves["holdout"]],
                     ignore_index=True)


def halt_out(reg, reason, mde, known_null, positive, counts, n_events, msha, t0,
             null_spread=None):
    log("HALT: %s" % reason)
    log("No observed figure is computed. No look is spent. No ledger row.")
    payload = {
        "halt": reason, "registration_sha": reg["_sha256"], "n_events": n_events,
        "mde": mde, "known_null_share": (known_null or {}).get("share_p_le_bar"),
        "X_calibrated": (positive or {}).get("X_calibrated"),
        "detection": (positive or {}).get("detection_rate"),
        "utc": utc_now(), "resolution_counts": counts,
        "market_ew_sha256": msha, "null_spread": null_spread,
        "known_null_control": known_null, "positive_control": positive,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if RECEIPT.exists():
        RECEIPT.unlink()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    HALTFILE.write_text(json.dumps(payload, indent=2, default=str))
    log("halt file %s" % rel(HALTFILE))
    return 15


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

    reg, pins, shas, counts, halves, book, market, msha, _mf = prepare(
        args.registration, args.sha)
    half = halves["half_A"]
    n_events = int(len(half))
    if n_events < e.MIN_EVENTS:
        raise AssertionError("half A resolved %d events, below the minimum %d"
                             % (n_events, e.MIN_EVENTS))
    bar = float(reg["verdict"]["bar"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])

    wb, groups, d_idx, n_real = build_window_book(book, market,
                                                  _all_resolved(halves), half)
    log("STEP 6 - OBSERVED")

    gross = e.voted_returns(wb, groups, d_idx) + e.COST_FRACTION
    net = gross - e.COST_FRACTION
    mcar_net = float(net.mean())
    mcar_gross = float(gross.mean())
    nullo = e.common_shift_null(wb, groups, d_idx, R_vote,
                               np.random.default_rng(seed_vote))
    p = e.null_p(mcar_net, nullo["mcar"])
    log("  n %d   mCAR net %+.6f   gross %+.6f   p %.4f"
        % (n_events, mcar_net, mcar_gross, p))

    median = float(np.median(net))
    share_pos = float((net > 0).mean())

    s20, n20 = wb.window(groups, d_idx + e.VOTE_FROM, d_idx + e.SHORT_TO)
    ok20 = n20 == e.SHORT_N
    w20 = {"mean": float(s20[ok20].mean()), "n": int(ok20.sum())}

    t_idx = np.array([(book.roll_forward(t) if book.roll_forward(t) is not None
                       else 0) for t in half["T"]], dtype=np.int64)
    pre_a, pre_b = t_idx + 1, d_idx + 1
    pre = {"mean": None, "n": 0}
    okp = pre_b >= pre_a
    if okp.any():
        sp, npre = wb.window(groups[okp], pre_a[okp], pre_b[okp])
        want = (pre_b[okp] - pre_a[okp] + 1)
        full = npre == want
        if full.any():
            pre = {"mean": float(sp[full].mean()), "n": int(full.sum()),
                   "mean_days": float(want[full].mean())}

    ta, tb = t_idx - (e.PRETERM_LAG - 1), t_idx
    preterm = {"mean": None, "n": 0}
    okt = (ta >= 0) & (tb < wb.L)
    if okt.any():
        st, nt_ = wb.window(groups[okt], ta[okt], tb[okt])
        full = nt_ == e.PRETERM_LAG
        if full.any():
            preterm = {"mean": float(st[full].mean()), "n": int(full.sum())}

    path = {}
    for kk in e.PATH_POINTS:
        s, n = wb.window(groups, d_idx + e.VOTE_FROM, d_idx + 1 + kk)
        okk = n == kk
        path["d%d" % kk] = {"mean": float(s[okk].mean()), "n": int(okk.sum())}

    by_month = e.events_by_month(half["file_date"])
    dsplit = {k: int(v) for k, v in
              sorted(half["direction_mix"].value_counts().items())}
    tmix = e.title_mix(half["titles"])
    lag_days = half["file_date"].sub(half["T"]).dt.days

    verdict = e.verdict_of(p, mcar_net, bar)
    log("  median %+.6f  share positive %.4f" % (median, share_pos))
    log("  20-day window D+2..D+21 mean %+.6f (n %d)" % (w20["mean"], w20["n"]))
    log("  pre-disclosure T..D+1 mean %s (n %d)"
        % ("%+.6f" % pre["mean"] if pre["mean"] is not None else "n/a", pre["n"]))
    log("  pre-termination T-21..T mean %s (n %d)"
        % ("%+.6f" % preterm["mean"] if preterm["mean"] is not None else "n/a",
           preterm["n"]))
    for kk in e.PATH_POINTS:
        log("  path +%-2d  %+.6f (n %d)"
            % (kk, path["d%d" % kk]["mean"], path["d%d" % kk]["n"]))
    log("VERDICT: %s  (p=%.4f, mCAR=%+.6f, bar %.2f, materiality %.3f)"
        % (verdict, p, mcar_net, bar, e.MATERIALITY))

    reg_sha = reg["_sha256"]
    harness_sha = e.sha256_file(ROOT / "tools" / "item408_event_study_r5.py")
    run_date = time.strftime("%Y%m%d")
    report = {
        "batch_id": BATCH_ID,
        "look": reg["look_budget"],
        "run_date": run_date,
        "vintage": reg["ledger"]["vintage_string"],
        "runner_version": VERSION, "engine_version": e.VERSION,
        "registration_sha256": reg_sha, "engine_sha256": harness_sha,
        "data_sha256": shas, "market_ew_sha256": msha,
        "controls_receipt_sha256": e.sha256_file(RECEIPT),
        "controls_receipt": {k: receipt[k] for k in
                             ("n_events", "mde", "known_null_share",
                              "X_calibrated", "detection", "utc")},
        "resolution_counts": counts,
        "issuers": len(wb.tickers),
        "real_events_masked": n_real,
        "observed": {
            "n_events": n_events,
            "mCAR_net": mcar_net, "mCAR_gross": mcar_gross,
            "cost_fraction": e.COST_FRACTION,
            "p_common_shift": p,
            "null_median": float(np.median(nullo["mcar"])),
            "null_sd": float(nullo["mcar"].std(ddof=1)),
            "null_p95": float(np.percentile(nullo["mcar"], 95)),
            "null_R": int(len(nullo["mcar"])), "null_redraws": nullo["redraws"],
            "median_net": median, "share_positive": share_pos,
            "window_20d_D2_D21": w20,
            "pre_disclosure_T_to_D1": pre,
            "pre_termination_Tm21_to_T": preterm,
            "path_after_D_plus_1": path,
        },
        "descriptives": {
            "events_by_filing_month": by_month,
            "direction_mix": dsplit,
            "direction_split_at_row_level": counts["F_direction_split"],
            "title_mix": tmix,
            "individuals_total": int(half["n_individuals"].sum()),
            "individuals_mean": float(half["n_individuals"].mean()),
            "disclosure_lag_days": {
                "median": float(lag_days.median()), "mean": float(lag_days.mean()),
                "min": int(lag_days.min()), "max": int(lag_days.max())},
            "forms": {k: int(v) for k, v in half["form"].value_counts().items()},
        },
        "verdict": {
            "verdict": verdict, "bar": bar, "materiality": e.MATERIALITY,
            "suggestive_hi": e.SUGGESTIVE_HI, "single_test": True,
            "promotion": ("one registered half-B look"
                          if verdict == "SOMETHING" else "none"),
        },
        "ledger_months_column": "carries n_events; this look has no monthly book",
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    LOGDIR.mkdir(parents=True, exist_ok=True)
    rp = LOGDIR / "t_r5_halfA_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    log("report %s sha %s" % (rel(rp), e.sha256_file(rp)))

    ok, n, head, bad = lc.verify(LEDGER)
    if not ok:
        raise AssertionError("ledger chain is broken before the append: %r" % bad)
    vals = [BATCH_ID, run_date, reg["ledger"]["vintage_string"],
            reg_sha[:16], harness_sha[:16], reg["ledger"]["signal"],
            reg["ledger"]["universe_string"], n_events,
            "%.4f" % (1e4 * mcar_net), "%.4f" % p, verdict]
    rs = lc.row_sha(head, vals)
    with open(LEDGER, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    log("ledger appended 1 row; head %s" % rs[:16])
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-R5 Item 408 termination runner")
    ap.add_argument("--stage", required=True, choices=["controls", "observed"])
    ap.add_argument("--registration", required=True)
    ap.add_argument("--sha", required=True)
    args = ap.parse_args(argv)
    log("run_t_r5 v%s; engine v%s; stage %s" % (VERSION, e.VERSION, args.stage))
    if args.stage == "controls":
        return stage_controls(args)
    return stage_observed(args)


if __name__ == "__main__":
    sys.exit(main())
