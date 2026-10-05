"""
================================================================================
Script:       test_coverage_probes.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the tier check's table-list extension: how --tables parses,
              what a coverage probe asks for, and that a non-200 body is kept
              whole rather than truncated.
Entry Point:  No — pytest
Dependencies: stdlib (ast, importlib, pathlib); pandas; the probe module
Last Sprint:  Sprint R3-0 — Tier check takes a table list (2026-10-05)
================================================================================

Falsifiable throughout: each test fails on a constructed violation of the
property it names. The separation that matters most is the one between the two
questions — "is the key paid?" and "does the plan reach this table?" — because
conflating them would read a table that is simply off the plan as a downgraded
key and halt the sprint on a false verdict.
================================================================================
"""

import ast
import importlib.util
import pathlib

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"


def _probe_module():
    spec = importlib.util.spec_from_file_location("sharadar_probe_cov", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _probe_module()


# ----------------------------------------------------------------------------
# Argument parsing.
# ----------------------------------------------------------------------------

def test_a_comma_list_becomes_a_table_list_in_order(mod):
    assert mod.parse_table_list("funds,metrics,daily") == ["funds", "metrics", "daily"]


def test_whitespace_around_names_is_stripped(mod):
    assert mod.parse_table_list(" funds , metrics ") == ["funds", "metrics"]


def test_empty_and_none_both_mean_no_coverage_probes(mod):
    assert mod.parse_table_list(None) == []
    assert mod.parse_table_list("") == []
    assert mod.parse_table_list("   ") == []
    assert mod.parse_table_list(",,") == []


def test_a_repeated_name_is_not_requested_twice(mod):
    """A duplicate would spend a request from a fixed budget for no new fact."""
    assert mod.parse_table_list("funds,metrics,funds") == ["funds", "metrics"]


def test_the_cli_exposes_tables_and_main_passes_it_to_tier_check(mod):
    """AST tripwire: the flag must exist and actually reach tier_check."""
    tree = ast.parse(PROBE.read_text())
    flags = [n.args[0].value for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "add_argument"
             and n.args and isinstance(n.args[0], ast.Constant)]
    assert "--tables" in flags
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "tier_check"]
    assert calls, "main never calls tier_check"
    assert any(k.arg == "coverage_tables" for c in calls for k in c.keywords), \
        "tier_check is called without coverage_tables; --tables would be inert"


# ----------------------------------------------------------------------------
# What a coverage probe asks for.
# ----------------------------------------------------------------------------

def test_each_table_gets_one_small_probe_on_its_own_endpoint(mod):
    specs = mod.coverage_probes(["funds", "metrics"], limit=5)
    assert [s["path"] for s in specs] == ["data/funds", "data/metrics"]
    assert [s["table"] for s in specs] == ["funds", "metrics"]
    assert all(s["params"]["limit"] == 5 for s in specs)
    assert all(s["coverage"] is True for s in specs)


def test_a_coverage_probe_carries_no_paid_gated_flag(mod):
    """classify_tier keys on paid_gated; a coverage spec must not supply one."""
    for s in mod.coverage_probes(["fundamentals"]):
        assert "paid_gated" not in s


def test_a_coverage_403_cannot_make_the_tier_read_free(mod):
    """The separation under test: an off-plan table is not a downgraded key."""
    verdict = mod.classify_tier([
        {"name": "tier_tickers_free", "paid_gated": False, "http_status": 200, "rows": 5},
        {"name": "tier_stocks_TSLA", "paid_gated": True, "http_status": 200, "rows": 5},
        {"name": "tier_sp500", "paid_gated": True, "http_status": 200, "rows": 5},
    ])
    assert verdict["verdict"] == "PAID"
    # The coverage refusal exists alongside it and changes nothing.
    assert mod.coverage_probes(["fundamentals"])[0]["table"] == "fundamentals"


# ----------------------------------------------------------------------------
# Non-200 bodies are the finding, so they are kept whole.
# ----------------------------------------------------------------------------

def test_a_non_200_body_is_returned_verbatim_not_truncated(mod):
    body = "Exceeds plan. " + ("x" * 5000)
    assert mod.body_for_report(403, body) == body
    assert len(mod.body_for_report(403, body)) == len(body)


def test_a_200_body_is_still_truncated_to_a_sample(mod):
    body = "y" * 5000
    assert mod.body_for_report(200, body, limit=400) == "y" * 400


def test_every_non_200_status_keeps_its_whole_body(mod):
    body = "z" * 1200
    for status in (401, 402, 403, 404, 429, 500, None):
        assert mod.body_for_report(status, body) == body


def test_an_absent_body_does_not_raise(mod):
    assert mod.body_for_report(403, None) == ""


def test_the_probe_reports_the_full_body_for_a_refusal(mod):
    """AST tripwire: the coverage branch must route through body_for_report."""
    src = PROBE.read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "tier_check")
    names = {n.func.id for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "body_for_report" in names, \
        "tier_check does not call body_for_report; a refusal would be truncated"


# ----------------------------------------------------------------------------
# The span reported is the span of the rows in hand.
# ----------------------------------------------------------------------------

def test_the_date_span_of_the_returned_rows_is_reported(mod):
    df = pd.DataFrame({"ticker": ["A", "A"], "date": ["2001-01-02", "2020-06-30"]})
    assert mod.frame_date_span(df) == ("date", "2001-01-02", "2020-06-30")


def test_a_table_dating_rows_under_another_name_is_still_spanned(mod):
    df = pd.DataFrame({"ticker": ["A"], "calendardate": ["1999-12-31"]})
    assert mod.frame_date_span(df) == ("calendardate", "1999-12-31", "1999-12-31")


def test_an_undated_frame_spans_nothing_rather_than_guessing(mod):
    assert mod.frame_date_span(pd.DataFrame({"ticker": ["A"]})) is None
    assert mod.frame_date_span(None) is None


def test_unparseable_dates_do_not_fabricate_a_span(mod):
    df = pd.DataFrame({"date": ["not-a-date", ""]})
    assert mod.frame_date_span(df) is None
