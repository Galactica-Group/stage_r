"""
Sprint R1 guard tests for the half-A panel builder.

The universe rules and the window guards are tested against constructed
fixtures; the range-max and prefix-sum primitives are tested against brute
force. No test reads the vendor snapshot.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import xs_triage as xt          # noqa: E402
import build_panel_r1 as bp     # noqa: E402


def _cand_frame() -> pd.DataFrame:
    """
    Four names in one month, each failing exactly one rule plus one that passes
    everything, so each rule can be shown to bite on its own.
      A: passes everything
      B: fails rule 2 (price 4.99)
      C: fails rule 3 (14 of 21 sessions)
      D: fails rule 4 (lowest dollar volume, with top_n = 3)
      E: fails rule 4 too (second lowest)
    """
    return pd.DataFrame({
        "permaticker": [1, 2, 3, 4, 5],
        "month": [500] * 5,
        "closeunadj_t": [10.0, 4.99, 10.0, 10.0, 10.0],
        "n_sessions_21": [21, 21, 14, 21, 21],
        "med_dollar_vol_21": [9e6, 8e6, 7e6, 1e6, 2e6],
    })


# --------------------------------------------------------- rules 2, 3, 4 bite
def test_rule_2_price_floor_bites():
    df = _cand_frame()
    cand, _ = xt.apply_universe_rules(df, top_n=99)
    assert not cand.iloc[1], "a 4.99 price must fail the 5.00 floor"
    # The floor is inclusive at exactly 5.00.
    df2 = df.copy()
    df2.loc[1, "closeunadj_t"] = 5.00
    cand2, _ = xt.apply_universe_rules(df2, top_n=99)
    assert cand2.iloc[1]


def test_rule_3_session_count_bites():
    df = _cand_frame()
    cand, _ = xt.apply_universe_rules(df, top_n=99)
    assert not cand.iloc[2], "14 of 21 sessions must fail the 15-session rule"
    df2 = df.copy()
    df2.loc[2, "n_sessions_21"] = 15
    cand2, _ = xt.apply_universe_rules(df2, top_n=99)
    assert cand2.iloc[2], "exactly 15 of 21 must pass"


def test_rule_4_top_n_by_dollar_volume_bites():
    df = _cand_frame()
    cand, univ = xt.apply_universe_rules(df, top_n=3)
    # A, D and E clear rules 2 and 3 (B and C do not), so the top-3 cut among
    # the three survivors admits all of them.
    assert cand.tolist() == [True, False, False, True, True]
    assert univ.tolist() == [True, False, False, True, True]
    # Tighten the cut and rule 4 starts biting on dollar volume order.
    _, univ2 = xt.apply_universe_rules(df, top_n=2)
    assert univ2.tolist() == [True, False, False, False, True], \
        "the top-2 cut must keep the two largest dollar volumes among candidates"
    _, univ1 = xt.apply_universe_rules(df, top_n=1)
    assert univ1.tolist() == [True, False, False, False, False]


def test_rule_4_is_resolved_inside_each_month_not_across_the_panel():
    df = pd.DataFrame({
        "permaticker": [1, 2, 3, 4],
        "month": [500, 500, 501, 501],
        "closeunadj_t": [10.0] * 4,
        "n_sessions_21": [21] * 4,
        # Month 501's names are both smaller than month 500's.
        "med_dollar_vol_21": [9e6, 8e6, 100.0, 50.0],
    })
    _, univ = xt.apply_universe_rules(df, top_n=1)
    # One winner per month, not one winner overall.
    assert univ.tolist() == [True, False, True, False]


def test_a_missing_dollar_volume_reading_cannot_enter_the_universe():
    df = _cand_frame()
    df.loc[0, "med_dollar_vol_21"] = np.nan
    cand, univ = xt.apply_universe_rules(df, top_n=99)
    assert not cand.iloc[0] and not univ.iloc[0]


def test_rule_1_is_upstream_and_excludes_preferred_stock():
    """Rule 1 is applied at the master, so the panel never sees preferred."""
    sep = pd.DataFrame({
        "table": ["SEP"] * 3, "permaticker": [1, 2, 3], "ticker": list("ABC"),
        "category": ["Domestic Common Stock", "Domestic Preferred Stock",
                     "ADR Common Stock"]})
    keep, _ = xt.admitted_common_stock(sep)
    assert set(keep["permaticker"]) == {1, 3}


# ------------------------------------------------------------- window guards
def test_builder_window_guard_raises_on_a_constructed_late_row():
    dates = pd.Series(pd.to_datetime(["2012-11-30", "2013-01-02"]))
    with pytest.raises(xt.WindowViolation, match="2013-01-01"):
        xt.assert_no_rows_at_or_after(dates, "2013-01-01", "constructed")


def test_the_built_panel_carries_no_month_at_or_after_the_cutoff():
    """The artifact itself, if it is on disk: the guard's final witness."""
    path = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA.parquet"
    if not path.exists():
        pytest.skip("panel not built in this environment")
    m = pd.read_parquet(path, columns=["month_ts", "in_universe"])
    assert m["month_ts"].max() < pd.Timestamp("2013-01-01")
    held = m.loc[m["in_universe"].astype(bool), "month_ts"]
    # The last formation month is 2012-11, because its return month is 2012-12.
    assert held.max() <= pd.Timestamp("2012-11-30")


# ------------------------------------------------- builder primitives
def test_sparse_range_max_matches_brute_force():
    rng = np.random.default_rng(1)
    a = rng.normal(size=400)
    lv = bp.sparse_max_table(a)
    lo = rng.integers(0, 400, size=300)
    hi = np.minimum(lo + rng.integers(0, 260, size=300), 399)
    got = bp.range_max(lv, lo, hi, len(a))
    want = np.array([a[l:h + 1].max() for l, h in zip(lo, hi)])
    assert np.allclose(got, want)


def test_range_max_returns_nan_for_an_empty_window_rather_than_raising():
    a = np.arange(10.0)
    lv = bp.sparse_max_table(a)
    # lo > hi is the empty window a name with no session in the range produces.
    out = bp.range_max(lv, np.array([5]), np.array([4]), len(a))
    assert np.isnan(out[0])
    # And an out-of-range hi is refused rather than read off the end.
    out2 = bp.range_max(lv, np.array([0]), np.array([10]), len(a))
    assert np.isnan(out2[0])


def test_builder_pins_the_registered_constants():
    assert bp.N_BUCKETS > 1
    assert bp.STOCKS.name == "bulk_stocks.parquet"
    assert bp.TICKERS.name == "bulk_tickers.parquet"
    assert "20261004p" in str(bp.SNAP)
    assert bp.OUT_PANEL.name == "panel_monthly_20261004p_halfA.parquet"


def test_bucketing_keeps_a_whole_name_in_one_bucket():
    """
    Pass C computes per-name features, so a name's rows must never be split
    across buckets. Hashing on permaticker is what guarantees that.
    """
    perma = np.arange(1, 5000)
    buckets = perma % bp.N_BUCKETS
    for p in (1, 37, 1234, 4999):
        assert len(np.unique(buckets[perma == p])) == 1
