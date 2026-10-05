"""Tests for the R3b half-B window machinery (tools/run_t_r3b.py).

Three things are pinned: the engine is unchanged (the brief makes any edit a
HALT), the sealed holdout cannot reach the panel, and the slice keeps exactly
the half-B months while leaving the warm-up states that precede them intact.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import etf_trend_r3 as e  # noqa: E402
import run_t_r3b as r  # noqa: E402

VDIR = ROOT / "data" / "sharadar" / r.VINTAGE
HAVE_DATA = (VDIR / "etf_SPY.parquet").exists()


def daily_from_monthly(closes, start="2000-01"):
    per = pd.period_range(start=start, periods=len(closes), freq="M")
    idx = pd.DatetimeIndex([p.end_time.normalize() for p in per])
    return pd.Series(np.asarray(closes, dtype=float), index=idx)


def built_panel(n=40, start="2010-01"):
    rng = np.random.default_rng(3)
    spec = {t: list(100 * np.cumprod(1 + rng.normal(0.005, 0.04, n)))
            for t in ("A1", "B1")}
    prices = {t: daily_from_monthly(v, start=start) for t, v in spec.items()}
    prices[e.CASH] = daily_from_monthly(list(100 * np.cumprod(np.full(n, 1.001))),
                                        start=start)
    return e.build_panel(prices, classes={"ca": ["A1"], "cb": ["B1"]})


# ---------------------------------------------------------------------------
# the engine is frozen
# ---------------------------------------------------------------------------
def test_engine_sha_is_unchanged():
    """The R3b brief makes any engine edit a HALT; this is the assertion."""
    assert r.assert_engine_unchanged() == r.ENGINE_SHA
    assert r.ENGINE_SHA == ("d024f6ff265a17ee4533c60cf1dc19b7e8f"
                            "666d0619ede680efe4a5fdf2e91e8")


def test_engine_sha_guard_bites_on_a_tampered_copy(tmp_path):
    src = ROOT / "tools" / "etf_trend_r3.py"
    dst = tmp_path / "etf_trend_r3.py"
    shutil.copy(src, dst)
    dst.write_text(dst.read_text() + "\n# tampered\n")
    with pytest.raises(AssertionError, match="changed"):
        r.assert_engine_unchanged(path=dst)


def test_inherited_registration_sha_is_pinned_in_code():
    assert r.V4_SHA == ("c3349c76745f93ca9bded73b2ee38ed5"
                        "2cb3ad2b1a273c42ddcea72a93ee614b")


# ---------------------------------------------------------------------------
# the holdout guard
# ---------------------------------------------------------------------------
def test_holdout_guard_bites_on_a_constructed_2021_month():
    months = pd.period_range("2020-10", periods=4, freq="M")   # reaches 2021-01
    with pytest.raises(AssertionError, match="holdout"):
        r.assert_no_holdout(months)


def test_holdout_guard_passes_the_half_b_window():
    months = pd.period_range("2013-01", "2020-12", freq="M")
    assert r.assert_no_holdout(months) is True
    assert len(months) == r.HALF_B_MONTHS


def test_holdout_guard_bites_on_a_single_offending_month_among_many():
    months = pd.PeriodIndex(list(pd.period_range("2013-01", "2020-12", freq="M"))
                            + [pd.Period("2021-06", freq="M")])
    with pytest.raises(AssertionError, match="2021"):
        r.assert_no_holdout(months)


def test_a_constructed_2021_row_is_filtered_at_read_and_never_reaches_the_panel(tmp_path):
    """The engine filters; the runner's assertion is the second line of defence."""
    d = pd.date_range("2020-11-01", "2021-03-01", freq="D")
    pd.DataFrame({"ticker": "ZZZ", "date": d.astype(str),
                  "closeadj": np.arange(len(d), dtype=float)}
                 ).to_parquet(tmp_path / "etf_ZZZ.parquet")
    out, spans = e.load_fund_prices(tmp_path, ["ZZZ"], r.HOLDOUT_START)
    assert out["ZZZ"].index.max() < pd.Timestamp(r.HOLDOUT_START)
    assert spans["ZZZ"][1] == "2020-12-31"
    kept = pd.PeriodIndex(out["ZZZ"].index, freq="M").unique()
    assert r.assert_no_holdout(kept) is True


# ---------------------------------------------------------------------------
# the slice
# ---------------------------------------------------------------------------
def test_slice_keeps_exactly_the_requested_months():
    p = built_panel(n=40, start="2010-01")
    lo, hi = "2012-01", "2012-12"
    s = r.slice_panel(p, lo, hi)
    assert [str(m) for m in s["months"]] == \
        [str(m) for m in pd.period_range(lo, hi, freq="M")]
    assert len(s["months"]) == 12
    for k in r.SLICED_KEYS:
        assert len(s[k]) == 12, k


def test_slice_leaves_the_warm_up_states_that_precede_it_intact():
    """The first retained month must carry the state the full build gave it."""
    p = built_panel(n=40, start="2010-01")
    s = r.slice_panel(p, "2012-01", "2012-12")
    j = int(np.flatnonzero(p["months"].astype(str) == "2012-01")[0])
    assert np.array_equal(s["state_rule"][0], p["state_rule"][j])
    assert np.array_equal(s["valid"][0], p["valid"][j])
    assert np.allclose(s["w"][0], p["w"][j])
    assert s["shy_r"][0] == p["shy_r"][j]


def test_slice_rejects_an_empty_or_non_contiguous_request():
    p = built_panel(n=40, start="2010-01")
    with pytest.raises(AssertionError, match="no months"):
        r.slice_panel(p, "1990-01", "1990-12")


def test_slice_does_not_mutate_the_panel_it_was_given():
    p = built_panel(n=40, start="2010-01")
    before = len(p["months"])
    r.slice_panel(p, "2012-01", "2012-12")
    assert len(p["months"]) == before


def test_a_sliced_panel_still_evaluates_to_the_same_rows_as_the_full_one():
    """Slicing must not change the returns of the months it keeps, only the
    first month's buy-in cost, which is the declared convention."""
    p = built_panel(n=40, start="2010-01")
    lo, hi = "2012-01", "2012-12"
    s = r.slice_panel(p, lo, hi)
    nf, _, _ = e.evaluate_books(p, p["state_rule"][None, :, :], e.BPS_BASE)
    ns, _, _ = e.evaluate_books(s, s["state_rule"][None, :, :], e.BPS_BASE)
    j = int(np.flatnonzero(p["months"].astype(str) == lo)[0])
    # months after the first retained one are identical
    assert np.allclose(nf[0][j + 1:j + 12], ns[0][1:12])
    # the first retained month differs only by the buy-in
    assert not np.isclose(nf[0][j], ns[0][1 - 1])


# ---------------------------------------------------------------------------
# the real window, when the data volume is mounted
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not HAVE_DATA, reason="vintage funds_20261005f not mounted")
def test_real_built_and_scored_windows_are_the_registered_ones():
    prices, spans = e.load_fund_prices(VDIR, r.TICKERS, r.HOLDOUT_START)
    assert max(s[1] for s in spans.values()) < r.HOLDOUT_START
    full = e.build_panel(prices)
    assert r.assert_no_holdout(full["months"]) is True
    assert (str(full["months"][0]), str(full["months"][-1])) == r.JOINED
    assert len(full["months"]) == r.JOINED_MONTHS == 221
    half_b = r.slice_panel(full, *r.HALF_B)
    assert (str(half_b["months"][0]), str(half_b["months"][-1])) == r.HALF_B
    assert len(half_b["months"]) == r.HALF_B_MONTHS == 96
    assert (half_b["n_present"] == 5).all(), "all five classes eligible throughout"
