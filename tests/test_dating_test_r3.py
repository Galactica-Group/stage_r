"""Tests for the daily-valuation-table dating test (tools/dating_test_r3.py).

The load-bearing test is test_pe_rounding_is_not_read_as_an_earnings_step: the
vendor quotes pe to one decimal, so a naive detector would manufacture hundreds
of spurious steps out of ordinary price movement.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import dating_test_r3 as d  # noqa: E402


def art_frame(rows):
    """rows: list of (filing_date, reportperiod, netinccmnusd)."""
    return pd.DataFrame({
        "date": pd.to_datetime([r[0] for r in rows]),
        "reportperiod": pd.to_datetime([r[1] for r in rows]),
        "netinccmnusd": [float(r[2]) for r in rows],
        "sharesbas": [1e9] * len(rows),
    })


# ---------------------------------------------------------------------------
# units and the band
# ---------------------------------------------------------------------------
def test_implied_earnings_rescales_millions_to_dollars():
    daily = pd.DataFrame({"marketcap": [1000.0], "pe": [10.0]})
    assert d.implied_earnings(daily)[0] == pytest.approx(100e6)


def test_implied_earnings_is_nan_on_a_non_positive_multiple():
    daily = pd.DataFrame({"marketcap": [1000.0, 1000.0], "pe": [0.0, -5.0]})
    assert np.isnan(d.implied_earnings(daily)).all()


def test_rounding_band_follows_the_quoted_precision_of_pe():
    daily = pd.DataFrame({"marketcap": [1.0, 1.0], "pe": [25.0, 5.0]})
    b = d.rounding_band(daily)
    assert b[0] == pytest.approx(1.5 * 0.05 / 25 + 1e-4)
    assert b[1] > b[0], "a lower multiple is less precise, so a wider band"


# ---------------------------------------------------------------------------
# matching
# ---------------------------------------------------------------------------
def test_match_is_unique_when_candidates_are_far_apart():
    E = np.array([5.0e9])
    band = np.array([0.003])
    idx, st = d.match_vintages(E, band, [4.0e9, 5.0e9, 6.0e9])
    assert st[0] == "matched" and idx[0] == 1


def test_match_is_ambiguous_when_two_candidates_sit_inside_the_band():
    E = np.array([5.0e9])
    band = np.array([0.01])
    idx, st = d.match_vintages(E, band, [5.0e9, 5.02e9])
    assert st[0] == "ambiguous"


def test_match_is_unmatched_when_nothing_is_compatible():
    E = np.array([9.9e9])
    band = np.array([0.003])
    idx, st = d.match_vintages(E, band, [5.0e9, 6.0e9])
    assert st[0] == "unmatched" and idx[0] == -1


def test_match_is_unrestricted_so_a_future_filing_can_win():
    """The look-ahead has to be able to show itself, not be assumed away."""
    E = np.array([7.0e9])
    band = np.array([0.003])
    idx, st = d.match_vintages(E, band, [5.0e9, 7.0e9])
    assert st[0] == "matched" and idx[0] == 1


def test_non_finite_candidates_are_skipped_not_matched():
    E = np.array([5.0e9])
    band = np.array([0.003])
    idx, st = d.match_vintages(E, band, [np.nan, 5.0e9])
    assert st[0] == "matched" and idx[0] == 1


# ---------------------------------------------------------------------------
# the load-bearing non-vacuity test
# ---------------------------------------------------------------------------
def test_pe_rounding_is_not_read_as_an_earnings_step():
    """Constant true earnings, moving price, pe rounded to 1dp -> zero steps."""
    n = 600
    cap_usd = 100e9 * np.cumprod(1 + np.random.default_rng(1).normal(0, 0.01, n))
    earn = 5.0e9
    pe_round = np.round(cap_usd / earn, 1)
    daily = pd.DataFrame({"marketcap": cap_usd / 1e6, "pe": pe_round})
    E = d.implied_earnings(daily)
    band = d.rounding_band(daily)
    art = art_frame([("2020-01-30", "2019-12-31", earn)])
    idx, st = d.match_vintages(E, band, art["netinccmnusd"].to_numpy())
    assert (st == "matched").all(), "rounding alone must stay inside the band"
    dates = pd.DatetimeIndex(pd.date_range("2020-02-03", periods=n, freq="B"))
    assert d.find_steps(dates, idx, st, art) == []


def test_a_real_earnings_change_is_read_as_exactly_one_step():
    n = 40
    cap_usd = np.full(n, 100e9)
    earn = np.where(np.arange(n) < 20, 5.0e9, 5.5e9)
    pe_round = np.round(cap_usd / earn, 1)
    daily = pd.DataFrame({"marketcap": cap_usd / 1e6, "pe": pe_round})
    E = d.implied_earnings(daily)
    band = d.rounding_band(daily)
    dates = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n, freq="B"))
    art = art_frame([("2019-10-30", "2019-09-30", 5.0e9),
                     ("2020-01-29", "2019-12-31", 5.5e9)])
    idx, st = d.match_vintages(E, band, art["netinccmnusd"].to_numpy())
    steps = d.find_steps(dates, idx, st, art)
    assert len(steps) == 1
    assert steps[0]["new_matched_reportperiod"] == "2019-12-31"
    assert steps[0]["step_date"] == str(dates[20].date())


# ---------------------------------------------------------------------------
# dating and classification
# ---------------------------------------------------------------------------
def _one_step(step_offset_days, filing):
    n = 40
    cap_usd = np.full(n, 100e9)
    earn = np.where(np.arange(n) < 20, 5.0e9, 5.5e9)
    daily = pd.DataFrame({"marketcap": cap_usd / 1e6,
                          "pe": np.round(cap_usd / earn, 1)})
    E = d.implied_earnings(daily)
    band = d.rounding_band(daily)
    dates = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=n, freq="D"))
    art = art_frame([("2019-10-30", "2019-09-30", 5.0e9),
                     (filing, "2019-12-31", 5.5e9)])
    idx, st = d.match_vintages(E, band, art["netinccmnusd"].to_numpy())
    return d.find_steps(dates, idx, st, art)


def test_a_step_on_or_after_the_filing_date_is_as_filed():
    steps = _one_step(0, "2020-01-18")     # step lands 2020-01-21
    assert len(steps) == 1
    assert steps[0]["classification"] == "AS-FILED"
    assert steps[0]["lag_days_step_minus_filing"] == 3


def test_a_step_before_the_filing_date_is_look_ahead():
    steps = _one_step(0, "2020-02-15")     # step lands 2020-01-21, filed later
    assert len(steps) == 1
    assert steps[0]["classification"] == "LOOK-AHEAD"
    assert steps[0]["lag_days_step_minus_filing"] == -25


def test_verdict_space_is_closed():
    af = [{"classification": "AS-FILED"}]
    la = [{"classification": "LOOK-AHEAD"}]
    assert d.verdict_for(af)[0] == "AS-FILED"
    assert d.verdict_for(la)[0] == "RESTATED"
    assert d.verdict_for(af + la)[0] == "MIXED"
    assert d.verdict_for([])[0] == "INDETERMINATE"


# ---------------------------------------------------------------------------
# dimension attribution
# ---------------------------------------------------------------------------
def test_dimension_attribution_reports_inseparable_when_art_equals_mrt():
    art = art_frame([("2020-01-29", "2019-12-31", 5.0e9)])
    mrt = art_frame([("2019-12-31", "2019-12-31", 5.0e9)])
    E = np.array([5.0e9]); band = np.array([0.003])
    out = d.dimension_attribution(E, band, art, mrt)
    assert out["separable"] is False
    assert out["reportperiods_where_they_differ"] == 0


def test_dimension_attribution_resolves_to_the_dimension_the_value_matches():
    art = art_frame([("2020-01-29", "2019-12-31", 5.0e9)])
    mrt = art_frame([("2019-12-31", "2019-12-31", 6.0e9)])
    band = np.array([0.003, 0.003])
    out = d.dimension_attribution(np.array([5.0e9, 6.0e9]), band, art, mrt)
    assert out["separable"] is True
    assert out["days_resolving_to_ART"] == 1
    assert out["days_resolving_to_MRT"] == 1


# ---------------------------------------------------------------------------
# the ambiguity-window bound and the neither counter
# ---------------------------------------------------------------------------
def test_look_ahead_is_excluded_outright_only_when_the_window_start_clears_filing():
    """The bound must key on the PREVIOUS matched day, not the step day."""
    steps = _one_step(0, "2020-01-18")      # step 2020-01-21, prev matched 2020-01-20
    assert steps[0]["lag_days_lower_bound"] == 2
    assert steps[0]["look_ahead_excluded_outright"] is True
    late = _one_step(0, "2020-01-21")       # filed exactly on the step day
    assert late[0]["lag_days_lower_bound"] == -1
    assert late[0]["look_ahead_excluded_outright"] is False
    assert late[0]["classification"] == "AS-FILED", \
        "the point estimate still clears; only the bound does not"


def test_dimension_attribution_counts_days_that_resolve_to_neither():
    """The counter must actually increment, or it reads a reassuring zero."""
    art = art_frame([("2020-01-29", "2019-12-31", 5.0e9)])
    mrt = art_frame([("2019-12-31", "2019-12-31", 6.0e9)])
    out = d.dimension_attribution(np.array([9.9e9]), np.array([0.003]), art, mrt)
    assert out["separable"] is True
    assert out["days_resolving_to_neither"] == 1
    assert out["days_resolving_to_ART"] == 0 and out["days_resolving_to_MRT"] == 0
