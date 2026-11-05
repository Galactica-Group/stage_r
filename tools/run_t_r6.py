"""
================================================================================
Script:       run_t_r6.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The T-R6 runner: resolution and the benchmark, then the registered
              controls, then — only on their written pass receipt — the observed
              half-A read.
Entry Point:  Yes — python tools/run_t_r6.py --stage resolve|controls|observed
                     --registration logs/r6/registration_v1.json --sha <sha256>
Dependencies: stdlib, numpy, pandas, tools/form4_event_study_r6.py,
              tools/ledger_chain.py. Imports nothing from any other repository.
Last Sprint:  Sprint R6-1 — T-R6 insider open-market purchases: seal, controls,
              half-A read (2026-11-05)
================================================================================

The stages make the registered order of operations mechanical rather than a
promise. --stage resolve walks steps 1 and 2 only — the event set resolved and
counted at every clause, and the investable benchmark built and cached — and
forms no figure of any event's return. --stage controls repeats those two and
then walks steps 3 to 5 — the MDE, the known-null control, the positive control
— writing logs/r6/controls_pass.json on a pass or logs/r6/controls_halt.json on
a halt, exiting non-zero in the second case. Both name the null_family they were
written under, so a receipt from one null design cannot be read as another's.
--stage observed refuses to start unless that receipt exists AND its
registration_sha equals the live seal's, so a read cannot be taken against a
registration the controls never saw.

Steps 1 to 4 form no figure of the observed event set: the counts are counts,
the benchmark knows nothing of any filing, and the MDE and the control p values
live entirely on permuted issuer-date pairings. Step 5 must form the observed
mean inside the planted mean it tests, because the registered plant is added to
the real events' own voted windows; that figure is used and discarded, printed
in no log line and written into no receipt.

The subgroup and by-year descriptives are formed in stage_observed and nowhere
else, which a test asserts by reading this file's own syntax tree.
================================================================================
"""

import argparse
import gc
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
import form4_event_study_r6 as e  # noqa: E402
import ledger_chain as lc  # noqa: E402

VERSION = "1.0.0"
BATCH_ID = "T-R6"
LOGDIR = ROOT / "logs" / "r6"
REPORTS = ROOT / "reports"
LEDGER = REPORTS / "triage_ledger.csv"
RECEIPT = LOGDIR / "controls_pass.json"
HALTFILE = LOGDIR / "controls_halt.json"

HEX64 = re.compile(r"\b([0-9a-f]{64})\b")
BENCH_PROBE_DATES = ("2008-06-16", "2012-06-15", "2015-06-15")
R5_INDEX = ROOT / "data" / "edgar_item408" / "20261104" / "market_ew_cal1000.parquet"


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
    """Paths and sha256 pins, read out of the registration's own data strings.

    The events string pins the MANIFEST's sha256 and says the parquet's own is
    "as recorded" there, so the parquet's pin is read out of the manifest the
    seal pins — one indirection, deliberately, because the manifest is the
    tracked record of the vintage.
    """
    ev_s = reg["data"]["events"]
    px_s = reg["data"]["prices"]
    tk_s = reg["data"]["tickers"]
    ev_path = re.search(r"(data/[\w/\-.]+purchases\.parquet)", ev_s).group(1)
    man_path = re.search(r"(manifests/[\w/\-.]+\.json)", ev_s).group(1)
    snap = re.search(r"(snapshot_\w+)", px_s).group(1)
    ev_hex = HEX64.findall(ev_s)
    if len(ev_hex) != 1:
        raise AssertionError("the events string must pin exactly one sha256 "
                             "(the manifest's), found %d" % len(ev_hex))
    man_sha = ev_hex[0]
    pur_sha = e.manifest_recorded_sha(ROOT / man_path, "purchases.parquet")
    return {
        "events": (ROOT / ev_path, pur_sha),
        "manifest": (ROOT / man_path, man_sha),
        "stocks": (ROOT / "data" / "sharadar" / snap / "bulk_stocks.parquet",
                   HEX64.findall(px_s)[0]),
        "tickers": (ROOT / "data" / "sharadar" / snap / "bulk_tickers.parquet",
                    HEX64.findall(tk_s)[0]),
        "snapshot": snap,
    }


def assert_data_pins(pins):
    out = {}
    for name in ("manifest", "events", "stocks", "tickers"):
        path, want = pins[name]
        got = e.sha256_file(path)
        if got != want:
            raise AssertionError("%s sha256 mismatch: %s has %s, the seal pins %s"
                                 % (name, rel(path), got, want))
        out[name] = got
        log("  pin OK %-9s %s %s" % (name, got[:16], rel(path)))
    return out


# ---------------------------------------------------------------------------
# Steps 1 and 2: resolution, then the investable benchmark
# ---------------------------------------------------------------------------
def prepare(reg_path, sha):
    """Seal, data pins, resolution counts, the benchmark, the halves.

    Nothing here forms a return of an observed event: the counts are counts and
    the benchmark is built from the whole snapshot.
    """
    stage_times = {}
    reg = e.load_registration(reg_path, sha)
    log("registration %s sha %s" % (Path(reg_path).name, reg["_sha256"]))
    pins = data_paths_and_pins(reg)
    shas = assert_data_pins(pins)

    min_tk = e.calendar_min_tickers(reg)
    log("loading the frozen snapshot (five columns, cast at read); the calendar "
        "keeps dates carrying >= %d tickers" % min_tk)
    t0 = time.time()
    book = e.load_prices(pins["stocks"][0], min_tk)
    stage_times["load_prices_s"] = round(time.time() - t0, 1)
    log("  %d ticker-days, %d tickers, %d trading days %s .. %s  (%.1fs)"
        % (len(book.code), len(book.vocab), book.L, book.date_str(0),
           book.date_str(book.L - 1), stage_times["load_prices_s"]))
    cs = book.cal_stats
    log("  calendar: %d dates in the snapshot, %d are trading days, %d excluded "
        "(%d rows dropped); thinnest trading day %d tickers, fattest excluded "
        "date %d" % (cs["dates_in_snapshot"], cs["trading_days"],
                     cs["dates_excluded"], cs["rows_on_excluded_dates"],
                     cs["tickers_on_the_thinnest_trading_day"],
                     cs["tickers_on_the_fattest_excluded_date"]))

    # ---- step 2: the investable benchmark, built once and cached ----
    log("STEP 2 - INVESTABLE BENCHMARK (equal weight, three floors on the "
        "prior day, whole snapshot)")
    cache = pins["events"][0].parent / ("index_investable_cal%d.parquet" % min_tk)
    members = {}
    if cache.exists():
        mf = pd.read_parquet(cache)
        if len(mf) != book.L:
            raise AssertionError("cached benchmark has %d rows, calendar has %d"
                                 % (len(mf), book.L))
        index = {"mean_log_return": mf["mean_log_return"].to_numpy(dtype=float),
                 "n_tickers": mf["n_tickers"].to_numpy(dtype=np.int64)}
        log("  cache hit %s" % rel(cache))
        stage_times["benchmark_s"] = 0.0
    else:
        t0 = time.time()
        dvmed = e.rolling_dollar_volume_median(book)
        log("  rolling %d-day dollar-volume median over %d rows in %.1fs"
            % (e.BENCH_DV_WINDOW, len(dvmed), time.time() - t0))
        t1 = time.time()
        index = e.build_investable_index(book, dvmed)
        mf = e.index_frame(book, index)
        mf.to_parquet(cache, index=False)
        log("  built in %.1fs and cached to %s" % (time.time() - t1, rel(cache)))
        for d in BENCH_PROBE_DATES:
            hit = mf.index[mf["date"] == d]
            if len(hit):
                members[d] = e.index_members_on(book, dvmed, int(hit[0]))
        del dvmed
        gc.collect()
        stage_times["benchmark_s"] = round(time.time() - t0, 1)
    msha = e.sha256_file(cache)
    nt = index["n_tickers"]
    log("  %s sha %s; dates %s .. %s; trading days %d; days with any ticker %d"
        % (cache.name, msha, mf["date"].iloc[0], mf["date"].iloc[-1], book.L,
           int((nt > 0).sum())))
    r5_counts = {}
    if R5_INDEX.exists():
        r5f = pd.read_parquet(R5_INDEX)
        for d in BENCH_PROBE_DATES:
            h = r5f.index[r5f["date"] == d]
            if len(h):
                r5_counts[d] = int(r5f["n_tickers"].iloc[h[0]])
    for d in BENCH_PROBE_DATES:
        hit = mf.index[mf["date"] == d]
        if len(hit):
            log("    tickers averaged on %s: investable %d   (R5 all-ticker %s)"
                % (d, int(nt[hit[0]]), r5_counts.get(d, "n/a")))

    # ---- step 1: resolution ----
    t0 = time.time()
    pur = pd.read_parquet(pins["events"][0])
    sep = pd.read_parquet(pins["tickers"][0],
                          columns=["table", "permaticker", "ticker", "secfilings"])
    log("STEP 1 - RESOLUTION: %d purchase rows in the vintage" % len(pur))
    res, counts = e.resolve_events(pur, sep, book)
    del pur, sep
    gc.collect()
    res = e.with_D_dates(res, book)
    stage_times["resolution_s"] = round(time.time() - t0, 1)
    for k in ("rows_in_vintage", "A_lag_0_to_90", "A_removed_lag_negative",
              "A_removed_lag_over_90", "A_removed_lag_missing", "B_not_routine",
              "B_removed_routine", "C_filing_day_on_the_calendar",
              "C_dropped_off_calendar", "D_events", "E_mapping",
              "E_duplicate_permaticker_D_pairs", "F_history_250_rows_before_D",
              "G_price_floor_on_D_plus_1", "G_dollar_volume_floor",
              "H_window_status", "H_complete_D1_to_D61",
              "H_delisted_in_window", "H_truncated_at_snapshot_edge",
              "H_hole", "H_complete_or_delisted"):
        log("  %-34s %s" % (k, counts[k]))
    counts["calendar"] = dict(book.cal_stats)
    counts["benchmark_members_on_probe_dates"] = {
        d: len(v) for d, v in members.items()}
    counts["r5_all_ticker_index_on_probe_dates"] = r5_counts

    halves = e.split_halves(res, reg)
    for k in ("half_A", "half_B", "holdout"):
        counts["I_%s" % k] = int(len(halves[k]))
        log("  %-34s %d" % ("I_%s" % k, len(halves[k])))
    counts["I_outside_every_regime"] = int(
        len(res) - sum(len(halves[k]) for k in ("half_A", "half_B", "holdout")))
    log("  %-34s %d" % ("I_outside_every_regime",
                        counts["I_outside_every_regime"]))

    return (reg, pins, shas, counts, halves, res, book, index, msha, mf,
            members, stage_times)


def build_window_book(book, market, res_all, half):
    """Cumulative abnormal returns for half A's issuers, plus the real-vote mask.

    The mask covers every resolved real event of those issuers across all three
    date regimes, not just half A's, so the contamination COUNT is an honest
    superset. Nothing is dropped on its account (RULE-NO-DROP).
    """
    wb = e.WindowBook(book, list(half["ticker"]), market)
    sub = res_all.loc[res_all["ticker"].isin(set(wb.tickers))]
    groups_all = np.array([wb.gidx[t] for t in sub["ticker"]], dtype=np.int64)
    d_idx_all = sub["D_idx"].to_numpy(dtype=np.int64)
    exit_all = sub["exit_idx"].to_numpy(dtype=np.int64)
    wb.set_real_voted(groups_all, d_idx_all, exit_all)
    groups = np.array([wb.gidx[t] for t in half["ticker"]], dtype=np.int64)
    d_idx = half["D_idx"].to_numpy(dtype=np.int64)
    return wb, groups, d_idx, int(len(sub))


def assert_truncation_unreachable(d_idx, L):
    """No half-A permuted pair can end at the snapshot's own edge.

    r5.score_pairs is the arbiter for permuted pairs and knows nothing of the
    truncation clause, so the clause must be structurally out of reach at these
    dates: the latest half-A window closes far before the vintage ends.
    """
    latest = int(np.max(d_idx)) + e.VOTE_TO
    edge = int(L) - 1 - e.TRUNCATION_EDGE
    if latest > edge:
        raise AssertionError(
            "a half-A window closes at position %d, inside the snapshot's "
            "%d-day truncation edge (%d)" % (latest, e.TRUNCATION_EDGE, edge))
    return {"latest_window_close_idx": latest, "truncation_edge_idx": edge,
            "margin_trading_days": int(edge - latest)}


# ---------------------------------------------------------------------------
# Stage: resolve (steps 1 and 2 only)
# ---------------------------------------------------------------------------
def stage_resolve(args):
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (reg, pins, shas, counts, halves, res, book, index, msha, mf, members,
     st) = prepare(args.registration, args.sha)
    half = halves["half_A"]
    n_events = int(len(half))
    log("half A resolved %d events; the registration's minimum is %d"
        % (n_events, e.MIN_EVENTS))
    if n_events < e.MIN_EVENTS:
        log("HALT at resolution: %d < %d. No look is spent." % (n_events,
                                                                e.MIN_EVENTS))
    ins = half["n_insiders"].to_numpy(dtype=np.int64) if n_events else np.array([0])
    val = half["total_value"].to_numpy(dtype=float) if n_events else np.array([0.0])
    dist = {
        "insiders": {str(k): int(v) for k, v in
                     sorted(pd.Series(ins).value_counts().items())[:12]},
        "insiders_ge_2": int((ins >= e.INSIDER_SUBGROUP).sum()),
        "value_quartiles_usd": [float(x) for x in
                                np.percentile(val, [25, 50, 75])],
        "value_gt_100k": int((val > e.VALUE_SUBGROUP).sum()),
        "n_purchases_median": float(np.median(
            half["n_purchases"].to_numpy(dtype=np.int64))) if n_events else None,
    }
    log("  half-A insider counts %s" % dist["insiders"])
    log("  half-A events with >= %d insiders: %d" % (e.INSIDER_SUBGROUP,
                                                     dist["insiders_ge_2"]))
    log("  half-A total value quartiles USD %s; > %d USD: %d"
        % (["%.0f" % x for x in dist["value_quartiles_usd"]],
           int(e.VALUE_SUBGROUP), dist["value_gt_100k"]))
    out = {
        "registration_sha": reg["_sha256"], "utc": utc_now(),
        "runner_version": VERSION, "engine_version": e.VERSION,
        "data_sha256": shas, "benchmark_sha256": msha,
        "benchmark_members_sample": {d: sorted(v)[:40] for d, v in
                                     members.items()},
        "benchmark_member_counts": {d: len(v) for d, v in members.items()},
        "r5_all_ticker_index_on_probe_dates":
            counts["r5_all_ticker_index_on_probe_dates"],
        "resolution_counts": counts, "n_half_A": n_events,
        "half_A_distributions": dist,
        "stage_times": st,
        "wall_seconds": round(time.time() - t0, 1),
    }
    p = LOGDIR / "resolution.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    log("resolution %s sha %s" % (rel(p), e.sha256_file(p)))
    return 0 if n_events >= e.MIN_EVENTS else 15


# ---------------------------------------------------------------------------
# Stage: controls
# ---------------------------------------------------------------------------
def stage_controls(args):
    t0 = time.time()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    (reg, pins, shas, counts, halves, res, book, index, msha, _mf, _mem,
     st) = prepare(args.registration, args.sha)

    half = halves["half_A"]
    n_events = int(len(half))
    bar = float(reg["verdict"]["bar"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])
    kn = reg["controls"]["known_null"]
    pc = reg["controls"]["positive_control"]
    band = tuple(kn["band"])
    null_family = str(reg["voting_null"]["name"])

    if n_events < e.MIN_EVENTS:
        return halt_out(reg, "half A resolved %d events, the registration's "
                        "minimum is %d" % (n_events, e.MIN_EVENTS),
                        None, None, None, counts, n_events, msha, t0, st)

    tw = time.time()
    wb, groups, d_idx, n_real = build_window_book(book, index, res, half)
    st["window_book_s"] = round(time.time() - tw, 1)
    log("window book: %d issuers, %d real voted windows marked for the "
        "contamination count (%.1fs)" % (len(wb.tickers), n_real,
                                         st["window_book_s"]))
    trunc = assert_truncation_unreachable(d_idx, wb.L)
    log("  truncation clause out of reach for every permuted pair: latest "
        "half-A close %d, edge %d, margin %d trading days"
        % (trunc["latest_window_close_idx"], trunc["truncation_edge_idx"],
           trunc["margin_trading_days"]))

    # ---- step 3: MDE from the null's own spread ----
    log("STEP 3 - MDE from the %s null (%d replicates, seed %d)"
        % (null_family, R_vote, seed_vote))
    tm = time.time()
    try:
        nullo = e.permutation_null(wb, groups, d_idx, R_vote,
                                   np.random.default_rng(seed_vote))
    except AssertionError as exc:
        # The registered null could not produce a single replicate. Nothing is
        # relaxed here: the retention rule is the seal's, the redraw cap is the
        # reused R5 implementation's, and the diagnostic below says only how far
        # the mechanism falls short. Changing either is a re-registration.
        st["mde_null_s"] = round(time.time() - tm, 1)
        diag = retention_halt_diagnostic(book, wb, groups, d_idx)
        return halt_out(reg, "STEP 3: the registered %s null could not be "
                        "formed - %s" % (null_family, exc),
                        None, None, None, counts, n_events, msha, t0, st,
                        None, diag)
    st["mde_null_s"] = round(time.time() - tm, 1)
    nm = nullo["mcar"]
    mde = e.mde_from_null(nm)
    null_spread = {"null_family": null_family,
                   "median": float(np.median(nm)), "sd": float(nm.std(ddof=1)),
                   "p95": float(np.percentile(nm, 95)), "R": int(len(nm)),
                   "redraws": nullo["redraws"],
                   "mean_retention": float(nullo["retention"].mean()),
                   "min_retention": float(nullo["retention"].min()),
                   "mean_contaminated_share": float(nullo["contaminated"].mean()),
                   "max_contaminated_share": float(nullo["contaminated"].max())}
    log("  null mCAR: median %+.6f  sd %.6f  p95 %+.6f  (R=%d, redraws %d, "
        "%.1fs)" % (null_spread["median"], null_spread["sd"],
                    null_spread["p95"], null_spread["R"], null_spread["redraws"],
                    st["mde_null_s"]))
    log("  mean retention %.4f (min %.4f); mean contaminated share %.4f "
        "(max %.4f)" % (null_spread["mean_retention"],
                        null_spread["min_retention"],
                        null_spread["mean_contaminated_share"],
                        null_spread["max_contaminated_share"]))
    log("  MDE at 80%% power = %+.6f distance" % mde)

    # ---- step 4: known-null control ----
    log("STEP 4 - known-null control: %d random permutations fed in as the "
        "observed set, fresh %d-replicate null each" % (kn["N"], kn["R_each"]))
    tk = time.time()
    rng_c = np.random.default_rng(int(kn["seed"]))
    ps = np.empty(int(kn["N"]))
    for i in range(int(kn["N"])):
        while True:
            sig_i = e.draw_permutation(rng_c, n_events)
            m_i, ret, _, _ = e.permuted_mcar(wb, groups, d_idx, sig_i)
            if np.isfinite(m_i) and ret >= e.RETENTION_MIN:
                break
        nn = e.permutation_null(wb, groups, d_idx, int(kn["R_each"]), rng_c)
        ps[i] = e.null_p(m_i, nn["mcar"])
        if (i + 1) % 25 == 0:
            log("  %d/%d  running share p<=%.2f = %.4f  (%.0fs)"
                % (i + 1, kn["N"], bar, (ps[:i + 1] <= bar).mean(),
                   time.time() - tk))
    st["known_null_s"] = round(time.time() - tk, 1)
    share = float((ps <= bar).mean())
    known_null = {"null_family": null_family,
                  "N": int(kn["N"]), "R_each": int(kn["R_each"]),
                  "seed": int(kn["seed"]), "share_p_le_bar": share,
                  "band": list(band), "mean_p": float(ps.mean()),
                  "median_p": float(np.median(ps)),
                  "in_band": e.band_ok(share, band)}
    log("  share at p<=%.2f = %.4f  (band %.2f-%.2f)  mean p %.4f  median p "
        "%.4f  (%.1fs)" % (bar, share, band[0], band[1], ps.mean(),
                           np.median(ps), st["known_null_s"]))
    if not known_null["in_band"]:
        return halt_out(reg, "known-null control share %.4f outside [%.2f, %.2f]"
                        % (share, band[0], band[1]),
                        mde, known_null, None, counts, n_events, msha, t0, st,
                        null_spread)
    log("  KNOWN-NULL CONTROL PASSES")

    # ---- step 5: positive control ----
    target = float(pc["target_mCAR"])
    log("STEP 5 - positive control: calibrate X on ONE draw to a planted mean "
        "of %.4f" % target)
    tp = time.time()
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
        nn = e.permutation_null(wb, groups, d_idx, int(pc["R_each"]), rng_p)
        p = e.null_p(planted, nn["mcar"])
        pp.append(p)
        hits += p <= bar
        if (i + 1) % 10 == 0:
            log("  %d/%d  detection %.2f  (%.0fs)"
                % (i + 1, pc["N_draws"], hits / (i + 1), time.time() - tp))
    st["positive_control_s"] = round(time.time() - tp, 1)
    det = hits / int(pc["N_draws"])
    positive = {"null_family": null_family, "X_calibrated": cal["X"],
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
    log("  detection %.2f over %d draws (floor %.2f); median p %.4f  (%.1fs)"
        % (det, pc["N_draws"], pc["detection_floor"], np.median(pp),
           st["positive_control_s"]))
    if not positive["passes"]:
        return halt_out(reg, "positive control detection %.2f below floor %.2f"
                        % (det, pc["detection_floor"]),
                        mde, known_null, positive, counts, n_events, msha, t0,
                        st, null_spread)
    log("  POSITIVE CONTROL PASSES")

    st["controls_total_s"] = round(time.time() - t0, 1)
    receipt = {
        "registration_sha": reg["_sha256"],
        "null_family": null_family,
        "n_events": n_events,
        "mde": mde,
        "known_null_share": known_null["share_p_le_bar"],
        "X_calibrated": positive["X_calibrated"],
        "detection": det,
        "utc": utc_now(),
        "runner_version": VERSION, "engine_version": e.VERSION,
        "engine_sha256": e.sha256_file(ROOT / "tools" / "form4_event_study_r6.py"),
        "r5_engine_sha256": e.sha256_file(
            ROOT / "tools" / "item408_event_study_r5.py"),
        "data_sha256": shas, "benchmark_sha256": msha,
        "resolution_counts": counts,
        "null_spread": null_spread,
        "known_null_control": known_null,
        "positive_control": positive,
        "truncation_reachability": trunc,
        "note": "steps 1-4 formed no figure of the observed event set; step 5 "
                "formed the observed mean inside the planted mean it tests, "
                "because the registered plant rides the real events' own voted "
                "windows, and neither printed nor recorded it",
        "mean_retention": null_spread["mean_retention"],
        "mean_contaminated_share": null_spread["mean_contaminated_share"],
        "stage_times": st,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if HALTFILE.exists():
        HALTFILE.unlink()
    RECEIPT.write_text(json.dumps(receipt, indent=2, default=str))
    log("receipt %s sha %s" % (rel(RECEIPT), e.sha256_file(RECEIPT)))
    log("CONTROLS PASS - the observed stage is unlocked")
    return 0


def retention_halt_diagnostic(book, wb, groups, d_idx, n_draws=200,
                              seed=20261106):
    """What a step-3 retention halt must say: how short, and why.

    Every figure here lives on permuted pairings and on the issuers' trading
    histories. No observed event return is formed, so a halt can be explained
    without spending the look it refuses to take.
    """
    log("  measuring the shortfall (%d uniform bijections, the exact "
        "issuer-date coverage matrix, and the issuers' spans)" % n_draws)
    t = time.time()
    prof = e.retention_profile(wb, groups, d_idx, n_draws,
                               np.random.default_rng(int(seed)))
    cov = e.coverage_density(wb, groups, d_idx)
    spans = e.issuer_span_profile(book, wb, groups, d_idx)
    log("  identity pairing retention %.6f (%d of %d) - the scorer covers "
        "every real event by its own issuer"
        % (prof["identity_retention"], prof["identity_used"], prof["n_events"]))
    log("  uniform bijections: retention mean %.4f sd %.4f min %.4f max %.4f; "
        "share >= %.2f = %.3f"
        % (prof["mean"], prof["sd"], prof["min"], prof["max"],
           prof["required"], prof["share_at_or_above"]["0.90"]))
    log("  the requirement sits %.1f sd above the mean of the achievable "
        "distribution" % (prof["sd_to_the_requirement"] or float("nan")))
    log("  exact coverage: %d issuers x %d dates, event-weighted ceiling "
        "%.4f (unweighted density %.4f)"
        % (cov["distinct_issuers"], cov["distinct_dates"],
           cov["event_weighted_expected_retention"],
           cov["pair_density_unweighted"]))
    log("  why: of %d issuers, %d list after %s, %d delist before %s plus the "
        "window, and only %d span it whole"
        % (spans["issuers"], spans["listing_after_the_earliest_date"],
           spans["date_range"][0],
           spans["delisting_before_the_latest_date_plus_window"],
           spans["date_range"][1], spans["spanning_the_whole_window"]))
    return {"retention_profile": prof, "coverage": cov,
            "issuer_spans": spans, "seconds": round(time.time() - t, 1)}


def halt_out(reg, reason, mde, known_null, positive, counts, n_events, msha, t0,
             st=None, null_spread=None, diagnostic=None):
    log("HALT: %s" % reason)
    log("No observed figure is computed. No look is spent. No ledger row.")
    payload = {
        "halt": reason, "registration_sha": reg["_sha256"],
        "null_family": str(reg["voting_null"]["name"]), "n_events": n_events,
        "mde": mde, "known_null_share": (known_null or {}).get("share_p_le_bar"),
        "X_calibrated": (positive or {}).get("X_calibrated"),
        "detection": (positive or {}).get("detection_rate"),
        "utc": utc_now(), "resolution_counts": counts,
        "runner_version": VERSION, "engine_version": e.VERSION,
        "engine_sha256": e.sha256_file(ROOT / "tools" / "form4_event_study_r6.py"),
        "r5_engine_sha256": e.sha256_file(
            ROOT / "tools" / "item408_event_study_r5.py"),
        "benchmark_sha256": msha, "null_spread": null_spread,
        "known_null_control": known_null, "positive_control": positive,
        "diagnostic": diagnostic,
        "stage_times": st or {},
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

    (reg, pins, shas, counts, halves, res, book, index, msha, _mf, _mem,
     st) = prepare(args.registration, args.sha)
    half = halves["half_A"]
    n_events = int(len(half))
    if n_events < e.MIN_EVENTS:
        raise AssertionError("half A resolved %d events, below the minimum %d"
                             % (n_events, e.MIN_EVENTS))
    bar = float(reg["verdict"]["bar"])
    R_vote = int(reg["voting_null"]["R"])
    seed_vote = int(reg["voting_null"]["seed"])
    null_family = str(reg["voting_null"]["name"])
    if receipt.get("null_family") != null_family:
        log("REFUSED: the receipt was written under null family %r but the live "
            "seal names %r" % (receipt.get("null_family"), null_family))
        return 2

    tw = time.time()
    wb, groups, d_idx, n_real = build_window_book(book, index, res, half)
    st["window_book_s"] = round(time.time() - tw, 1)
    trunc = assert_truncation_unreachable(d_idx, wb.L)
    log("STEP 6 - OBSERVED")

    net, n_ret, ok_obs, exit_obs = e.score_pairs(wb, groups, d_idx)
    if not bool(ok_obs.all()):
        raise AssertionError("%d half-A events are not covered by their own "
                             "issuer after resolution" % int((~ok_obs).sum()))
    gross = net + e.COST_FRACTION
    mcar_net = float(net.mean())
    mcar_gross = float(gross.mean())
    tn = time.time()
    nullo = e.permutation_null(wb, groups, d_idx, R_vote,
                               np.random.default_rng(seed_vote))
    st["observed_null_s"] = round(time.time() - tn, 1)
    nullmed = e.null_median(nullo["mcar"])
    p = e.null_p(mcar_net, nullo["mcar"])
    dist = e.distance_of(mcar_net, nullo["mcar"])
    con_share = float(nullo["contaminated"].mean())
    dist_corr = e.corrected_distance(dist, con_share)
    log("  n %d   mCAR net %+.6f   gross %+.6f" % (n_events, mcar_net,
                                                   mcar_gross))
    log("  null median %+.6f   DISTANCE %+.6f   p %.4f  (%s)"
        % (nullmed, dist, p, null_family))
    log("  mean contaminated share %.4f   corrected distance %+.6f"
        % (con_share, dist_corr))

    median = float(np.median(net))
    share_pos = float((net > 0).mean())

    s20, n20 = wb.window(groups, d_idx + e.VOTE_FROM, d_idx + e.SHORT_TO)
    ok20 = n20 == e.SHORT_N
    w20 = {"mean": float(s20[ok20].mean()), "n": int(ok20.sum())}

    t_idx = np.array([(book.roll_forward(t) if book.roll_forward(t) is not None
                       else -1) for t in half["T"]], dtype=np.int64)
    pre_a, pre_b = t_idx + 1, d_idx + 1
    pre = {"mean": None, "n": 0}
    okp = (t_idx >= 0) & (pre_b >= pre_a)
    if okp.any():
        sp, npre = wb.window(groups[okp], pre_a[okp], pre_b[okp])
        want = (pre_b[okp] - pre_a[okp] + 1)
        full = npre == want
        if full.any():
            pre = {"mean": float(sp[full].mean()), "n": int(full.sum()),
                   "mean_days": float(want[full].mean())}

    ta, tb = t_idx - (e.PREPURCHASE_LAG - 1), t_idx
    prepur = {"mean": None, "n": 0}
    okt = (t_idx >= 0) & (ta >= 0) & (tb < wb.L)
    if okt.any():
        sq, nq = wb.window(groups[okt], ta[okt], tb[okt])
        full = nq == e.PREPURCHASE_LAG
        if full.any():
            prepur = {"mean": float(sq[full].mean()), "n": int(full.sum())}

    path = {}
    for kk in e.PATH_POINTS:
        s, n = wb.window(groups, d_idx + e.VOTE_FROM, d_idx + 1 + kk)
        okk = n == kk
        path["d%d" % kk] = {"mean": float(s[okk].mean()), "n": int(okk.sum())}

    subgroups = e.subgroup_distances(half, net, nullmed)
    by_year = e.events_by_year(half["D_date"], net, nullmed)
    wstat = {k: int(v) for k, v in
             sorted(half["window_status"].value_counts().items())}
    nret = half["n_returns"].to_numpy(dtype=np.int64)
    lag_days = (pd.to_datetime(half["D_date"]) - pd.to_datetime(half["T"])).dt.days

    verdict = e.verdict_of(p, dist, bar)
    log("  median %+.6f  share positive %.4f" % (median, share_pos))
    log("  window status %s; returns per event min %d median %d"
        % (wstat, int(nret.min()), int(np.median(nret))))
    log("  20-day window D+2..D+21 mean %+.6f (n %d)" % (w20["mean"], w20["n"]))
    log("  pre-filing T..D+1 mean %s (n %d)"
        % ("%+.6f" % pre["mean"] if pre["mean"] is not None else "n/a", pre["n"]))
    log("  pre-purchase T-%d..T mean %s (n %d)"
        % (e.PREPURCHASE_LAG,
           "%+.6f" % prepur["mean"] if prepur["mean"] is not None else "n/a",
           prepur["n"]))
    for kk in e.PATH_POINTS:
        log("  path +%-2d  %+.6f (n %d)"
            % (kk, path["d%d" % kk]["mean"], path["d%d" % kk]["n"]))
    for k, v in subgroups.items():
        log("  subgroup %-26s n %-6s distance %s"
            % (k, v["n"], "%+.6f" % v["distance"]
               if v["distance"] is not None else "n/a"))
    log("VERDICT: %s  (p=%.4f, distance=%+.6f, bar %.2f, materiality %.3f)"
        % (verdict, p, dist, bar, e.MATERIALITY))

    reg_sha = reg["_sha256"]
    harness_sha = e.sha256_file(ROOT / "tools" / "form4_event_study_r6.py")
    run_date = time.strftime("%Y%m%d")
    st["observed_total_s"] = round(time.time() - t0, 1)
    report = {
        "batch_id": BATCH_ID,
        "look": reg["look_budget"],
        "run_date": run_date,
        "vintage": reg["ledger"]["vintage_string"],
        "runner_version": VERSION, "engine_version": e.VERSION,
        "registration_sha256": reg_sha, "engine_sha256": harness_sha,
        "r5_engine_sha256": e.sha256_file(
            ROOT / "tools" / "item408_event_study_r5.py"),
        "data_sha256": shas, "benchmark_sha256": msha,
        "controls_receipt_sha256": e.sha256_file(RECEIPT),
        "controls_receipt": {k: receipt[k] for k in
                             ("n_events", "mde", "known_null_share",
                              "X_calibrated", "detection", "utc")},
        "resolution_counts": counts,
        "issuers": len(wb.tickers),
        "real_events_masked": n_real,
        "truncation_reachability": trunc,
        "observed": {
            "n_events": n_events,
            "mCAR_net": mcar_net, "mCAR_gross": mcar_gross,
            "cost_fraction": e.COST_FRACTION,
            "null_family": null_family,
            "null_median": nullmed,
            "distance": dist,
            "distance_contamination_corrected": dist_corr,
            "p_permutation": p,
            "null_sd": float(nullo["mcar"].std(ddof=1)),
            "null_p95": float(np.percentile(nullo["mcar"], 95)),
            "null_R": int(len(nullo["mcar"])), "null_redraws": nullo["redraws"],
            "null_mean_retention": float(nullo["retention"].mean()),
            "null_min_retention": float(nullo["retention"].min()),
            "null_mean_contaminated_share": con_share,
            "null_max_contaminated_share": float(nullo["contaminated"].max()),
            "median_net": median, "share_positive": share_pos,
            "window_status": wstat,
            "returns_per_event": {"min": int(nret.min()),
                                  "median": float(np.median(nret)),
                                  "max": int(nret.max())},
            "window_20d_D2_D21": w20,
            "pre_filing_T_to_D1": pre,
            "pre_purchase_Tm21_to_T": prepur,
            "path_after_D_plus_1": path,
            "subgroup_distances": subgroups,
            "events_by_year": by_year,
        },
        "descriptives": {
            "calendar": counts.get("calendar", {}),
            "insiders_total": int(half["n_insiders"].sum()),
            "insiders_mean": float(half["n_insiders"].mean()),
            "purchases_total": int(half["n_purchases"].sum()),
            "purchases_mean": float(half["n_purchases"].mean()),
            "total_value_usd": {
                "median": float(np.median(half["total_value"])),
                "mean": float(half["total_value"].mean())},
            "dv_median_at_D_usd": {
                "median": float(np.median(half["dv_median_at_D"])),
                "p25": float(np.percentile(half["dv_median_at_D"], 25)),
                "p75": float(np.percentile(half["dv_median_at_D"], 75))},
            "disclosure_lag_days": {
                "median": float(lag_days.median()), "mean": float(lag_days.mean()),
                "min": int(lag_days.min()), "max": int(lag_days.max())},
            "distinct_issuers": int(half["ticker"].nunique()),
        },
        "verdict": {
            "verdict": verdict, "bar": bar, "materiality": e.MATERIALITY,
            "suggestive_hi": e.SUGGESTIVE_HI, "single_test": True,
            "measured_from": "the null's median, never zero",
            "promotion": ("one registered half-B look"
                          if verdict == "SOMETHING" else "none"),
        },
        "ledger_months_column": "carries n_events; this look has no monthly book",
        "ledger_bps_column": "carries the DISTANCE in bps, not mCAR",
        "stage_times": st,
        "wall_seconds": round(time.time() - t0, 1),
        "licence_note": "statistics only; no vendor price appears in this file",
    }
    LOGDIR.mkdir(parents=True, exist_ok=True)
    rp = LOGDIR / "t_r6_halfA_report.json"
    rp.write_text(json.dumps(report, indent=2, default=str))
    log("report %s sha %s" % (rel(rp), e.sha256_file(rp)))

    ok, n, head, bad = lc.verify(LEDGER)
    if not ok:
        raise AssertionError("ledger chain is broken before the append: %r" % bad)
    vals = [BATCH_ID, run_date, reg["ledger"]["vintage_string"],
            reg_sha[:16], harness_sha[:16], reg["ledger"]["signal"],
            reg["ledger"]["universe_string"], n_events,
            "%.4f" % (1e4 * dist), "%.4f" % p, verdict]
    rs = lc.row_sha(head, vals)
    with open(LEDGER, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    log("ledger appended 1 row; head %s" % rs[:16])
    log("DONE wall %.1fs" % (time.time() - t0))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="T-R6 Form 4 purchase runner")
    ap.add_argument("--stage", required=True,
                    choices=["resolve", "controls", "observed"])
    ap.add_argument("--registration", required=True)
    ap.add_argument("--sha", required=True)
    args = ap.parse_args(argv)
    log("run_t_r6 v%s; engine v%s; stage %s" % (VERSION, e.VERSION, args.stage))
    if args.stage == "resolve":
        return stage_resolve(args)
    if args.stage == "controls":
        return stage_controls(args)
    return stage_observed(args)


if __name__ == "__main__":
    sys.exit(main())
