"""
================================================================================
Script:       test_tier_check.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Pin the paid-tier classifier: a 403 on a paid-gated endpoint must
              read FREE and halt, and only a clean 200-with-rows on every gated
              endpoint may read PAID.
Entry Point:  No — pytest
Dependencies: stdlib (ast, importlib, pathlib); the probe module
Last Sprint:  Sprint R0-C — Paid-tier verification (2026-09-26)
================================================================================

Falsifiable throughout: each test fails on a constructed violation of the
property it names. The R0-C observation (tier_stocks_TSLA 403 "Exceeds free
tier") is carried as a fixture so the verdict that halted the sprint stays
reproducible without a network.
================================================================================
"""

import ast
import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"


def _probe_module():
    spec = importlib.util.spec_from_file_location("sharadar_probe_tier", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _probe_module()


def _r(name, gated, status, rows):
    return {"name": name, "paid_gated": gated, "http_status": status, "rows": rows}


BASELINE_OK = _r("tier_tickers_free", False, 200, 5)


# ----------------------------------------------------------------------------
# The verdict that stopped R0-C.
# ----------------------------------------------------------------------------

def test_the_r0c_observation_reads_free_not_paid(mod):
    """Exactly what the vendor returned on 2026-09-26: TSLA 403, sp500 200."""
    out = mod.classify_tier([
        BASELINE_OK,
        _r("tier_stocks_TSLA", True, 403, 1),
        _r("tier_sp500", True, 200, 5),
    ])
    assert out["verdict"] == "FREE"
    assert "tier_stocks_TSLA" in out["reason"]


def test_one_403_among_passes_still_reads_free(mod):
    """A single gated 403 is decisive; it is not outvoted by its neighbours."""
    out = mod.classify_tier([
        BASELINE_OK,
        _r("a", True, 200, 5), _r("b", True, 200, 5), _r("c", True, 403, 1),
    ])
    assert out["verdict"] == "FREE"


def test_all_gated_200_with_rows_reads_paid(mod):
    out = mod.classify_tier([
        BASELINE_OK,
        _r("tier_stocks_TSLA", True, 200, 5),
        _r("tier_sp500", True, 200, 5),
    ])
    assert out["verdict"] == "PAID"


# ----------------------------------------------------------------------------
# PAID is not reachable by accident.
# ----------------------------------------------------------------------------

def test_gated_200_with_zero_rows_is_not_paid(mod):
    """An empty 200 is not entitlement; it must not be crowned PAID."""
    out = mod.classify_tier([
        BASELINE_OK,
        _r("tier_stocks_TSLA", True, 200, 0),
        _r("tier_sp500", True, 200, 5),
    ])
    assert out["verdict"] == "INDETERMINATE"


def test_gated_none_rows_is_not_paid(mod):
    """An unparsed body carries no row count and cannot certify a tier."""
    out = mod.classify_tier([
        BASELINE_OK,
        _r("tier_stocks_TSLA", True, 200, None),
        _r("tier_sp500", True, 200, 5),
    ])
    assert out["verdict"] == "INDETERMINATE"


def test_gated_500_is_indeterminate_not_free(mod):
    """A server fault is not evidence about the plan; FREE would be a lie."""
    out = mod.classify_tier([
        BASELINE_OK,
        _r("tier_stocks_TSLA", True, 500, None),
        _r("tier_sp500", True, 200, 5),
    ])
    assert out["verdict"] == "INDETERMINATE"


def test_broken_baseline_is_reported_as_such(mod):
    """A dead key or wrong base URL must not be reported as a tier verdict."""
    out = mod.classify_tier([
        _r("tier_tickers_free", False, 401, None),
        _r("tier_stocks_TSLA", True, 200, 5),
    ])
    assert out["verdict"] == "BASELINE_FAILED"


def test_no_gated_probe_cannot_certify_anything(mod):
    out = mod.classify_tier([BASELINE_OK])
    assert out["verdict"] == "INDETERMINATE"


# ----------------------------------------------------------------------------
# The probe set itself, and the halt wiring.
# ----------------------------------------------------------------------------

def test_tier_probes_include_the_endpoint_that_discriminates(mod):
    """stocks?ticker=<x> is the measured discriminator; losing it guts the check."""
    gated = [p for p in mod.TIER_PROBES if p["paid_gated"]]
    assert len(gated) >= 1
    paths = {p["path"] for p in gated}
    assert "data/stocks" in paths
    stocks = next(p for p in gated if p["path"] == "data/stocks")
    assert stocks["params"].get("ticker"), "the discriminator is ticker-filtered"
    assert any(not p["paid_gated"] for p in mod.TIER_PROBES), "need a baseline probe"


def test_a_non_paid_verdict_exits_nonzero(mod):
    """The halt is wired: main() must not fall through to a pull on FREE."""
    src = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "main")
    found = False
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        cond = ast.unparse(node.test)
        # ast.unparse normalises string quoting; match on the token, not the quotes.
        if "PAID" not in cond or "!=" not in cond:
            continue
        if any(isinstance(c, ast.Raise) and "SystemExit" in ast.unparse(c)
               for c in ast.walk(node)):
            found = True
    assert found, "no `verdict != PAID -> raise SystemExit` guard found in main()"


def test_plan_headers_surfaces_quota_headers_and_drops_noise(mod):
    got = mod.plan_headers({"X-RateLimit-Limit": "5000", "Content-Type": "application/json",
                            "X-Plan": "prices-full", "Date": "today"})
    assert got == {"X-RateLimit-Limit": "5000", "X-Plan": "prices-full"}
