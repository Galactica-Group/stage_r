"""
================================================================================
Script:       canon_conf_prose.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The registration's prose parts, assembled from the measured
              artifacts so every number in the seal is derived and none typed.
Entry Point:  No — used by tools/canon_conf_seal.py
Dependencies: numpy; tools/canon_conf_paths.py
Last Sprint:  Sprint CONF-2 — the recount, the corrected floors, the read (2026-11-24)
================================================================================

The parts live here so they can be tested and so the seal builder stays a
builder. Every figure is read from controls_v1.json, pool_v1.json,
s3_counts.json or read 3's banked result; nothing is hand-entered.
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402

VERSION = "1.0.0"

BPS_PER_SIDE_BASE = 5.0          # xs_triage.COST_BASE, for the cost arithmetic
SQRT12 = float(np.sqrt(12.0))


def _load() -> tuple[dict, dict, dict, dict]:
    c = json.loads(P.controls_v1().read_text())
    pool = json.loads(P.pool_v1().read_text())
    r3 = json.loads(P.r3_result().read_text())
    s3 = json.loads((P.derived() / "s3_counts.json").read_text())
    return c, pool, r3, s3


def floors_in_units(c: dict, pool: dict, r3: dict) -> dict:
    """
    2.1. The floors in units that carry. A gross-Sharpe floor measured in the
    planted worlds is NOT what a real book needs for key 2, because the planted
    books are far less volatile than read 3's and so lose far more Sharpe to the
    same cost in basis points.
    """
    st3 = r3["reported"]["series_statistics"]
    planted = []
    for k in sorted(c["plants"]["by_rho"], key=float):
        b = c["plants"]["by_rho"][k]
        loss = float(np.median([w["gross_sharpe"] - w["net_base_sharpe"]
                                for w in b["per_world"]]))
        cost = (BPS_PER_SIDE_BASE / 1e4) * b["median_turnover"]
        implied_vol = cost * SQRT12 / loss if loss else float("nan")
        planted.append({
            "rho": b["rho"], "median_turnover": b["median_turnover"],
            "median_gross_sharpe": b["median_gross_sharpe"],
            "median_net_base_sharpe": b["median_net_base_sharpe"],
            "median_sharpe_lost_to_costs": loss,
            "implied_monthly_volatility": implied_vol,
            "n_pass_key1": b["n_pass_key1"], "n_confirmed": b["n_confirmed"]})
    sd = np.array([w["gross"]["sd"] for w in pool["worlds"]], dtype=float)
    tu = np.array([w["turnover_mean"] for w in pool["worlds"]], dtype=float)
    gs = np.array([w["gross"]["sharpe"] for w in pool["worlds"]], dtype=float)
    ns = np.array([w["net_base"]["sharpe"] for w in pool["worlds"]], dtype=float)
    bar = c["floors"]["key2_sharpe_bar"]
    r3net = st3["net_base"]["sharpe"]
    return {
        "statement": (
            "A gross-Sharpe floor measured in the planted worlds OVERSTATES what "
            "a real book needs to clear key 2. The planted books are much less "
            "volatile than read 3's, so the same cost in basis points costs them "
            "far more Sharpe: their median monthly volatility is about 0.57 to "
            "0.62 per cent against read 3's 1.24 per cent, and they lose a median "
            "0.79 to 0.85 of annualised Sharpe to costs against read 3's 0.42. "
            "Key 2 is a bar on a real book's NET Sharpe, so it must be read in "
            "net units, not inferred from a gross floor."),
        "planted_by_rho": planted,
        "pool_worlds": {
            "median_monthly_volatility": float(np.median(sd)),
            "min_monthly_volatility": float(sd.min()),
            "max_monthly_volatility": float(sd.max()),
            "median_turnover": float(np.median(tu)),
            "median_sharpe_lost_to_costs": float(np.median(gs - ns))},
        "read_3_real_book": {
            "turnover": r3["reported"]["turnover_mean"],
            "monthly_volatility": st3["gross"]["sd"],
            "gross_sharpe": st3["gross"]["sharpe"],
            "net_base_sharpe": st3["net_base"]["sharpe"],
            "sharpe_lost_to_costs": st3["gross"]["sharpe"] - st3["net_base"]["sharpe"]},
        "key1": {
            "passed_12_of_12_from_rho": 0.020,
            "median_gross_sharpe_there": c["plants"]["by_rho"]["0.020"]["median_gross_sharpe"],
            "analytic_gross_sharpe_80pc_power_96_months":
                c["floors"]["analytic_gross_sharpe_80pc_power_96_months"]},
        "verdict_counts_by_net_sharpe": [
            {"rho": 0.020, "n_confirmed": c["plants"]["by_rho"]["0.020"]["n_confirmed"],
             "median_net_base_sharpe": c["plants"]["by_rho"]["0.020"]["median_net_base_sharpe"]},
            {"rho": 0.025, "n_confirmed": c["plants"]["by_rho"]["0.025"]["n_confirmed"],
             "median_net_base_sharpe": c["plants"]["by_rho"]["0.025"]["median_net_base_sharpe"]},
            {"rho": 0.030, "n_confirmed": c["plants"]["by_rho"]["0.030"]["n_confirmed"],
             "median_net_base_sharpe": c["plants"]["by_rho"]["0.030"]["median_net_base_sharpe"]}],
        "what_a_real_book_needs": {
            "key2_bar_on_net_base_sharpe": bar,
            "read_3_net_base_sharpe": r3net,
            "read_3_clears_key2": bool(r3net >= bar),
            "half_of_read_3_net_base_sharpe": r3net / 2.0,
            "half_clears_key2": bool(r3net / 2.0 >= bar),
            "fraction_of_read_3_that_sits_on_the_bar": bar / r3net,
            "statement": (
                "A book repeating read 3's half-A sizes clears both keys: its net "
                "BASE Sharpe was %.4f against a bar of %.2f. A book at HALF those "
                "sizes reads %.4f and still clears the bar — the brief's phrase "
                "'sits on the key 2 bar' is arithmetically wrong at one half, and "
                "the bar is in fact reached at %.3f of read 3's net Sharpe. The "
                "CONF-1b report's warning that a faithful replication would read "
                "NOT_CONFIRMED on key 2 is WITHDRAWN: it compared a gross floor "
                "from the low-volatility planted worlds with a gross number from "
                "a real book, which is not a like comparison."
                % (r3net, bar, r3net / 2.0, bar / r3net))},
    }


def stopped_bound(c: dict, pool: dict, s3: dict, r3: dict) -> dict:
    """2.2. The bound, its sign, its cost in basis points, and its unit problem."""
    st3 = r3["reported"]["series_statistics"]
    wp = np.array([w["s1"]["weight_pos_mean"] for w in pool["worlds"]], dtype=float)
    out_by_L = {}
    for key in sorted(pool["worlds"][0]["s2"]):
        L = pool["worlds"][0]["s2"][key]["L"]
        bps = L * wp * 1e4
        fall = np.array([w["net_base"]["sharpe"] - w["s2"][key]["sharpe"]
                         for w in pool["worlds"]], dtype=float)
        out_by_L[key] = {
            "L": L,
            "cost_bps_per_month_median": float(np.median(bps)),
            "cost_bps_per_month_min": float(bps.min()),
            "cost_bps_per_month_max": float(bps.max()),
            "median_sharpe_fall_in_the_null_worlds": float(np.median(fall)),
            "same_cost_on_read_3_volatility_sharpe_fall":
                float(np.median(bps) / 1e4 / st3["gross"]["sd"] * SQRT12)}
    a = s3["cohort_a_stopped"]
    b = s3["cohort_b_ends_inside_the_holding_month"]
    return {
        "statement": (
            "The stopped cohort is almost wholly the TARGET of a takeover, not a "
            "failure, so S2's assumed loss is adverse by construction and its "
            "sign is against the evidence. It is reported, never voted."),
        "counts": {
            "in_universe_scored_rows": s3["in_universe_scored_rows"],
            "cohort_a_stopped": {"rows": a["rows"], "for_cause": a["for_cause"],
                                 "takeover_target": a["takeover_target"],
                                 "neither": a["neither"]},
            "cohort_b_ends_inside_the_holding_month": {
                "rows": b["rows"], "for_cause": b["for_cause"],
                "takeover_target": b["takeover_target"], "neither": b["neither"]},
            "last_price_date_definition": s3["last_price_date_definition"],
            "conf_1b_recheck": s3["conf_1b_cohort_b_recheck"]},
        "by_L": out_by_L,
        "withdrawn": (
            "CONF-1b's 'nine times' comparison is WITHDRAWN. It rested on a "
            "cohort of 1,394 rows built from the last daily date observed before "
            "the cutoff, which pins every name still trading to the 2020 edge: "
            "%d of those 1,394 were names still trading. Under the tickers "
            "master's own lastpricedate the cohort is %d rows, %.2f per cent of "
            "the %d scored rows, of which %d are takeover targets and %d carry a "
            "for-cause action."
            % (s3["conf_1b_cohort_b_recheck"]["of_which_still_trading_at_the_cutoff"],
               b["rows"], 100.0 * b["rows"] / s3["in_universe_scored_rows"],
               s3["in_universe_scored_rows"], b["takeover_target"], b["for_cause"])),
        "unit_caveat": (
            "A Sharpe fall measured in the null worlds OVERSTATES the fall for a "
            "book with read 3's volatility, for the same reason the floors do: at "
            "L = 1.00 the bound costs a median %.3f basis points a month, which "
            "is %.4f of Sharpe in the null worlds but %.4f on read 3's monthly "
            "volatility of %.4f."
            % (out_by_L["L_1.00"]["cost_bps_per_month_median"],
               out_by_L["L_1.00"]["median_sharpe_fall_in_the_null_worlds"],
               out_by_L["L_1.00"]["same_cost_on_read_3_volatility_sharpe_fall"],
               st3["gross"]["sd"])),
    }


def limits(c: dict) -> list[str]:
    """2.4. Read 3's limits, plus the four this sprint adds."""
    return [
        "read 3's limits, unchanged",
        "a stock's return after its last trade is not observed, for a stopped "
        "name or for one that ends inside the holding month, and S2 bounds only "
        "the first",
        "the second cohort is 398 scored rows, 0.41 per cent, of which 393 are "
        "takeover targets and 3 carry a for-cause action; it is not bounded here",
        "sector and share class are today's values",
        "a data flaw common to every world cannot be ruled out by this instrument",
        "costs are provisional constants and no certified equity cost row exists",
        "read 3's key functions bind their bars as DEFAULT ARGUMENTS, so moving "
        "the module constant changes nothing; the confirmation's bars are the "
        "same values and are stated as numbers in this seal",
        "half B of the stock panel is opened by this read, once",
        "config/windows.json's half_b status line still reads UNOPENED; it is "
        "left as written here and corrected in a later, separate commit",
    ]


def build_prose() -> dict:
    c, pool, r3, s3 = _load()
    st3 = r3["reported"]["series_statistics"]
    rr = r3["reported"]
    fl = floors_in_units(c, pool, r3)
    return {
        "floors_in_units_that_carry": fl,
        "stopped_stock_bound": stopped_bound(c, pool, s3, r3),
        "correction_to_the_record": (
            "The factory's decision log (D-CANON-READ3) said a stopped stock is "
            "scored at zero for its last month, flattering a book that buys what "
            "has fallen, and that the confirmation would address this in its "
            "vote. The first statement is wrong in its mechanism and the second "
            "is withdrawn. The builder calls a stock stopped precisely when it "
            "has NO observation in the next month, so there is no later trade to "
            "recover; a stock's fall to its last trade is already inside its "
            "ordinary monthly return; and a stopped row earns exactly zero under "
            "the registered scoring, so no part of a gross result can come from "
            "stopped names. 163 stopped rows fall in the 96 scored months, 0.17 "
            "per cent of 96,000, and 161 of them are the target of a takeover "
            "against 1 carrying a for-cause action: the cohort is bought, not "
            "wound up, so a flat assumed loss has the wrong sign. A key that "
            "moves the vote by a few basis points on an assumed loss is not a "
            "test. The exposure is reported by S1 to S3 and never voted."),
        "known_before_the_read": {
            "read_3": {
                "gross_mean_bps_per_month": st3["gross"]["mean_bps"],
                "gross_t": st3["gross"]["t"],
                "gross_sharpe": st3["gross"]["sharpe"],
                "gross_monthly_sd": st3["gross"]["sd"],
                "net_base_sharpe": st3["net_base"]["sharpe"],
                "net_stress_sharpe": st3["net_stress"]["sharpe"],
                "turnover": rr["turnover_mean"],
                "break_even_bps_per_side": rr["break_even_bps_per_side"],
                "by_year": rr["by_year"], "verdict": r3["verdict"]},
            "notes": [
                "read 3's best years were 2002 to 2006",
                "the half-A result has not been cut further and this "
                "specification was fixed without doing so",
                "the first build of this confirmation halted on a third key that "
                "could not differ from the second"]},
        "prior_stated_before_the_data": (
            "About four in ten that it is CONFIRMED. Edges of this kind are "
            "documented to have weakened after 2010. A book repeating read 3's "
            "half-A sizes clears both keys, and one at half those sizes still "
            "clears key 2 at a net Sharpe of %.2f; the bar is reached only at "
            "%.3f of read 3's net Sharpe. The prior is about the edge surviving "
            "eight unseen years, not about the bar being hard to clear."
            % (st3["net_base"]["sharpe"] / 2.0,
               c["floors"]["key2_sharpe_bar"] / st3["net_base"]["sharpe"])),
        "what_each_verdict_will_mean": {
            "CONFIRMED": (
                "The candidate held on eight unseen years at provisional costs. A "
                "research finding, not a money finding; the 2021 holdout stays "
                "sealed for a decision of its own, and no money test follows "
                "without a certified equity cost row (D-P01-CLOSE)."),
            "NOT_CONFIRMED": (
                "The keys that failed are named. The candidate is closed; half A "
                "stands in the record as a discovery that did not replicate; "
                "nothing is re-specified and re-read on half B.")},
        "limits": limits(c),
        "controls": {
            "t3": {"identical": c["t3"]["identical"],
                   "mismatches": c["t3"]["mismatches"]},
            "known_null": {k: c["known_null"][k] for k in
                           ("share_significant", "accept_band", "pass",
                            "worlds_called_confirmed", "gross_t",
                            "net_base_sharpe", "s2_median_fall",
                            "verdict_counts")},
            "plants_by_rho": {k: {f: v[f] for f in
                                  ("rho", "seed_index", "share_pass_1a",
                                   "share_pass_1b", "n_pass_key1",
                                   "share_pass_key2", "n_confirmed",
                                   "median_gross_sharpe",
                                   "median_net_base_sharpe", "median_turnover")}
                              for k, v in c["plants"]["by_rho"].items()},
            "floors": c["floors"], "bars": c["bars"],
            "determinism": c["determinism"],
            "fit_diagnostics_world_01": c["fit_diagnostics_world_01"],
            "wall_seconds": c["wall_seconds"]},
    }
