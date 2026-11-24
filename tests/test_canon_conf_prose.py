"""
================================================================================
Script:       test_canon_conf_prose.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The corrections of sprint CONF-2: the recount's definitions, the
              floors stated in net units, the withdrawn claims, and that every
              figure in the registration's prose is derived rather than typed.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-2 — the recount, the corrected floors, the read (2026-11-24)
================================================================================
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_conf_paths as P  # noqa: E402
import canon_conf_prose as PR  # noqa: E402
import canon_conf_stopped as S  # noqa: E402


# ------------------------------------------------- the recount's definitions
def test_the_takeover_test_is_the_target_side_only():
    """
    CONF-1b matched the prefixes "acquisition" and "merger", which also match the
    ACQUIRER's side (acquisitionof, mergerfrom). The recount names the target's.
    """
    assert S.TAKEOVER_TARGET_ACTIONS == ("acquisitionby", "mergerto")
    assert not hasattr(S, "TAKEOVER_PREFIXES")
    for bad in ("acquisitionof", "mergerfrom"):
        assert bad not in S.TAKEOVER_TARGET_ACTIONS


def test_the_for_cause_types_are_unchanged():
    assert S.FOR_CAUSE_ACTIONS == ("bankruptcyliquidation", "regulatorydelisting",
                                   "voluntarydelisting")


def test_the_recount_uses_the_masters_dated_field_not_the_observed_maximum():
    import ast
    src = (ROOT / "tools" / "canon_conf_stopped.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_master_last_price_dates")
    assert "lastpricedate" in ast.dump(fn)
    s3fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                and n.name == "s3_counts")
    d = ast.dump(s3fn)
    assert "_master_last_price_dates" in d
    # the observed maximum survives only to measure CONF-1b's error
    assert "_observed_last_price_dates" in d


def test_the_two_definitions_of_a_last_price_date_are_not_the_same():
    """Non-vacuity: a name alive past the cutoff differs under the two rules."""
    import pandas as pd
    cut = pd.Timestamp("2021-01-01")
    observed_edge = pd.Timestamp("2020-12-31")      # what CONF-1b would record
    master_alive = pd.Timestamp("2026-10-02")       # what the master records
    assert observed_edge < cut <= master_alive
    assert pd.Period(observed_edge, freq="M").ordinal != \
        pd.Period(master_alive, freq="M").ordinal


# ------------------------------------------------------ the recounted census
def test_the_s3_artifact_carries_both_cohorts_and_the_recheck():
    f = P.derived() / "s3_counts.json"
    if not f.exists():
        pytest.skip("no s3 artifact yet")
    s3 = json.loads(f.read_text())
    for k in ("cohort_a_stopped", "cohort_b_ends_inside_the_holding_month",
              "conf_1b_cohort_b_recheck", "last_price_date_definition",
              "takeover_target_types"):
        assert k in s3, k
    for c in ("cohort_a_stopped", "cohort_b_ends_inside_the_holding_month"):
        rec = s3[c]
        for k in ("rows", "for_cause", "takeover_target", "neither",
                  "action_type_census_by_rows"):
            assert k in rec, (c, k)
        assert rec["for_cause"] + rec["takeover_target"] >= rec["rows"] - rec["neither"]


def test_the_second_cohort_is_inside_the_sprints_stop_gate():
    """G3: at most 600 rows and at most 10 of them for-cause."""
    f = P.derived() / "s3_counts.json"
    if not f.exists():
        pytest.skip("no s3 artifact yet")
    b = json.loads(f.read_text())["cohort_b_ends_inside_the_holding_month"]
    assert b["rows"] <= 600, b["rows"]
    assert b["for_cause"] <= 10, b["for_cause"]


def test_the_recheck_explains_conf_1bs_number():
    f = P.derived() / "s3_counts.json"
    if not f.exists():
        pytest.skip("no s3 artifact yet")
    r = json.loads(f.read_text())["conf_1b_cohort_b_recheck"]
    assert r["rows_under_conf_1b_definition"] == 1394
    assert r["of_which_still_trading_at_the_cutoff"] > 0


# -------------------------------------------------- the floors in net units
def _prose():
    for f in (P.controls_v1(), P.pool_v1(), P.r3_result(),
              P.derived() / "s3_counts.json"):
        if not f.exists():
            pytest.skip("the measured artifacts are not all present")
    return PR.build_prose()


def test_the_floors_statement_is_in_net_units_and_names_the_volatility_gap():
    p = _prose()["floors_in_units_that_carry"]
    assert "OVERSTATES" in p["statement"]
    assert p["read_3_real_book"]["monthly_volatility"] > \
        p["pool_worlds"]["max_monthly_volatility"], \
        "the premise is that read 3's book is MORE volatile than the worlds'"
    assert p["read_3_real_book"]["sharpe_lost_to_costs"] < \
        p["pool_worlds"]["median_sharpe_lost_to_costs"], \
        "and so loses LESS Sharpe to the same cost"


def test_no_statement_says_only_a_gross_floor_can_confirm():
    p = _prose()
    blob = json.dumps(p)
    assert "only an edge at or above" not in blob
    assert "can read CONFIRMED" not in blob


def test_the_half_size_arithmetic_is_stated_correctly():
    p = _prose()["floors_in_units_that_carry"]["what_a_real_book_needs"]
    r3net, bar = p["read_3_net_base_sharpe"], p["key2_bar_on_net_base_sharpe"]
    assert p["half_of_read_3_net_base_sharpe"] == pytest.approx(r3net / 2)
    assert p["half_clears_key2"] is True, \
        "half of read 3's net Sharpe clears the 0.5 bar; the brief said it sits on it"
    assert p["fraction_of_read_3_that_sits_on_the_bar"] == pytest.approx(bar / r3net)
    assert "arithmetically wrong at one half" in p["statement"]
    assert "WITHDRAWN" in p["statement"]


def test_the_key1_and_verdict_counts_are_the_measured_ones():
    p = _prose()["floors_in_units_that_carry"]
    c = json.loads(P.controls_v1().read_text())
    assert p["key1"]["median_gross_sharpe_there"] == \
        c["plants"]["by_rho"]["0.020"]["median_gross_sharpe"]
    for rec in p["verdict_counts_by_net_sharpe"]:
        b = c["plants"]["by_rho"]["%.3f" % rec["rho"]]
        assert rec["n_confirmed"] == b["n_confirmed"]
        assert rec["median_net_base_sharpe"] == b["median_net_base_sharpe"]


# ------------------------------------------------------- the bound's prose
def test_the_bound_states_its_sign_is_against_the_evidence():
    p = _prose()["stopped_stock_bound"]
    a = p["counts"]["cohort_a_stopped"]
    assert a["takeover_target"] > 50 * max(a["for_cause"], 1)
    assert "sign is against the evidence" in p["statement"]
    assert "nine times" in p["withdrawn"] and "WITHDRAWN" in p["withdrawn"]


def test_the_bound_reports_basis_points_at_each_L_and_the_unit_caveat():
    p = _prose()["stopped_stock_bound"]
    assert set(p["by_L"]) == {"L_0.30", "L_1.00"}
    for key, rec in p["by_L"].items():
        assert rec["cost_bps_per_month_median"] > 0
        assert rec["same_cost_on_read_3_volatility_sharpe_fall"] < \
            rec["median_sharpe_fall_in_the_null_worlds"], key
    assert "OVERSTATES" in p["unit_caveat"]


# --------------------------------------------------------------- the limits
def test_the_limits_carry_the_four_additions():
    lim = " ".join(_prose()["limits"])
    assert "DEFAULT ARGUMENTS" in lim
    assert "half B of the stock panel is opened by this read, once" in lim
    assert "UNOPENED" in lim and "separate commit" in lim
    assert "398 scored rows" in lim


def test_the_prior_is_unchanged_at_about_four_in_ten():
    p = _prose()["prior_stated_before_the_data"]
    assert "four in ten" in p
    assert "0.355" in p or "0.36" in p


# -------------------------------------------------- every figure is derived
def test_the_seal_path_is_registration_v1_and_nothing_is_named_sealed():
    assert P.sealed_registration().name == "registration_v1.json"
    hits = [q for q in ROOT.rglob("*")
            if "sealed" in q.name.lower() and ".git" not in q.parts
            and "venv" not in q.parts and "__pycache__" not in q.parts]
    for h in hits:
        assert h.parent == ROOT / "registrations" / "canon_r3", str(h)


def test_the_prose_module_hand_enters_no_statistic():
    """
    Guard against a typed number drifting from the artifact. The only float
    literals the module may carry are the cost constant, sqrt(12)'s argument,
    the rho labels and plain arithmetic.
    """
    import ast
    src = (ROOT / "tools" / "canon_conf_prose.py").read_text()
    allowed = {5.0, 12.0, 1e4, 2.0, 100.0, 0.020, 0.025, 0.030, 0.41, 50.0}
    bad = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Constant) and isinstance(n.value, float):
            if n.value not in allowed:
                bad.append(n.value)
    assert not bad, "hand-entered floats in the prose module: %s" % bad
