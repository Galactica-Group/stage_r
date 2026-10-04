"""
Sprint R1 guard tests for the cross-sectional triage harness.

Every test is falsifiable against a constructed fixture and hand-checked where
the brief asks for hand-checking. No test reads the vendor snapshot, so none of
them carries a price.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import xs_triage as xt  # noqa: E402


# ----------------------------------------------------------------- windows
def test_windows_config_pins_half_a_and_seals_the_rest():
    w = json.loads((ROOT / "config" / "windows.json").read_text())
    assert w["half_a"]["return_months_through"] == "2012-12"
    assert w["half_a"]["daily_row_exclusive_max"] == "2013-01-01"
    assert w["half_b"]["status"] == "UNOPENED"
    assert w["holdout"]["status"] == "SEALED"
    assert w["vintage"] == "20261004p"


def test_window_guard_accepts_half_a_and_raises_on_a_constructed_late_row():
    ok = pd.Series(pd.to_datetime(["1998-01-02", "2012-12-31"]))
    assert xt.assert_no_rows_at_or_after(ok, "2013-01-01", "ok") == pd.Timestamp("2012-12-31")

    late = pd.Series(pd.to_datetime(["2012-12-31", "2013-01-02"]))
    with pytest.raises(xt.WindowViolation):
        xt.assert_no_rows_at_or_after(late, "2013-01-01", "late")

    # The boundary itself is excluded: 2013-01-01 is half B, not half A.
    boundary = pd.Series(pd.to_datetime(["2013-01-01"]))
    with pytest.raises(xt.WindowViolation):
        xt.assert_no_rows_at_or_after(boundary, "2013-01-01", "boundary")


def test_filter_to_half_a_removes_late_rows_and_then_asserts():
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2012-12-28", "2012-12-31", "2013-01-02", "2020-06-01"]), "v": [1, 2, 3, 4]})
    out = xt.filter_to_half_a(df, "date", "2013-01-01", "t")
    assert len(out) == 2
    assert out["date"].max() == pd.Timestamp("2012-12-31")


# ------------------------------------------------------- permaticker identity
def test_sep_map_is_one_to_one_within_the_vintage():
    t = pd.DataFrame({
        "table": ["SEP", "SEP", "SF1", "SF1"],
        "permaticker": [1, 2, 1, 3],
        "ticker": ["AAA", "BBB", "AAA", "CCC"],
        "category": ["Domestic Common Stock"] * 4,
    })
    sep = xt.sep_ticker_map(t)
    assert len(sep) == 2
    assert set(sep["ticker"]) == {"AAA", "BBB"}


def test_sep_map_refuses_a_recycled_ticker_or_a_duplicated_permaticker():
    dup_ticker = pd.DataFrame({
        "table": ["SEP", "SEP"], "permaticker": [1, 2],
        "ticker": ["AAA", "AAA"], "category": ["Domestic Common Stock"] * 2})
    with pytest.raises(ValueError, match="ticker not unique"):
        xt.sep_ticker_map(dup_ticker)

    dup_perma = pd.DataFrame({
        "table": ["SEP", "SEP"], "permaticker": [1, 1],
        "ticker": ["AAA", "BBB"], "category": ["Domestic Common Stock"] * 2})
    with pytest.raises(ValueError, match="permaticker not unique"):
        xt.sep_ticker_map(dup_perma)


# ------------------------------------------------- universe rule 1 bites
def test_rule_1_common_stock_admits_common_and_excludes_preferred():
    sep = pd.DataFrame({
        "table": ["SEP"] * 5, "permaticker": [1, 2, 3, 4, 5],
        "ticker": list("ABCDE"),
        "category": ["Domestic Common Stock", "ADR Common Stock",
                     "Domestic Preferred Stock", "Canadian Common Stock",
                     "ADR Preferred Stock"],
    })
    keep, counts = xt.admitted_common_stock(sep)
    assert set(keep["ticker"]) == {"A", "B", "D"}
    assert sum(counts.values()) == 3
    assert "Domestic Preferred Stock" not in counts


# --------------------------------------------------- quintile arithmetic
def test_quintiles_are_equal_count_and_hand_checked():
    v = np.arange(10.0)          # 10 names, 2 per quintile
    q = xt.quintiles_by_month(v)
    assert q.tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    # Q1 holds the lowest values, Q5 the highest.
    assert v[q == 1].max() < v[q == 5].min()


def test_quintiles_exclude_nan_and_refuse_a_too_small_cross_section():
    v = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0])
    q = xt.quintiles_by_month(v)
    assert q[1] == 0                       # NaN is excluded
    assert sorted(q[q > 0].tolist()) == [1, 2, 3, 4, 5]
    assert xt.quintiles_by_month(np.array([1.0, 2.0])).tolist() == [0, 0]


def test_registered_long_side_follows_the_direction():
    assert xt.long_side_quintile("HIGH") == 5 and xt.short_side_quintile("HIGH") == 1
    assert xt.long_side_quintile("LOW") == 1 and xt.short_side_quintile("LOW") == 5
    with pytest.raises(ValueError):
        xt.long_side_quintile("SIDEWAYS")


def test_ls_and_long_only_weight_arithmetic_is_hand_checked():
    q = np.array([1, 1, 3, 3, 5, 5], dtype=np.int8)
    univ = np.ones(6, dtype=bool)

    w_ls = xt.target_weights(q, 5, 1, "LS", univ)
    # Long Q5 at +1/2 each, short Q1 at -1/2 each, Q3 flat.
    assert np.allclose(w_ls, [-0.5, -0.5, 0.0, 0.0, 0.5, 0.5])
    assert np.isclose(w_ls.sum(), 0.0)
    assert np.isclose(np.abs(w_ls).sum(), 2.0)          # gross 2

    w_lo = xt.target_weights(q, 5, 1, "LONG_ONLY", univ)
    # Q5 at +1/2, every name short 1/6 of the benchmark.
    assert np.allclose(w_lo, [-1/6, -1/6, -1/6, -1/6, 0.5 - 1/6, 0.5 - 1/6])
    assert np.isclose(w_lo.sum(), 0.0)


def test_ls_return_is_q5_minus_q1_by_hand():
    q = np.array([1, 1, 5, 5], dtype=np.int8)
    w = xt.target_weights(q, 5, 1, "LS", np.ones(4, dtype=bool))
    fwd = np.array([0.01, 0.03, 0.10, 0.06])
    # Q5 mean 0.08, Q1 mean 0.02 -> spread 0.06
    assert np.isclose(float(np.dot(w, fwd)), 0.06)


def test_long_only_return_is_q5_minus_universe_equal_weight_by_hand():
    q = np.array([1, 1, 5, 5], dtype=np.int8)
    w = xt.target_weights(q, 5, 1, "LONG_ONLY", np.ones(4, dtype=bool))
    fwd = np.array([0.01, 0.03, 0.10, 0.06])
    # Q5 mean 0.08, universe mean 0.05 -> active 0.03
    assert np.isclose(float(np.dot(w, fwd)), 0.03)


# ------------------------------------------------------------ cost arithmetic
def test_cost_specs_carry_the_registered_provisional_numbers():
    assert xt.COST_BASE.bps_per_side == 5.0
    assert xt.COST_BASE.borrow_bps_per_year == 50.0
    assert xt.COST_BASE.commission_bps == 0.0
    assert xt.COST_STRESS.bps_per_side == 15.0
    assert xt.COST_STRESS.borrow_bps_per_year == 100.0


def test_traded_notional_charges_each_side_once_by_hand():
    w_prev = np.array([0.5, 0.5, 0.0, 0.0])
    w_now = np.array([0.0, 0.0, 0.5, 0.5])
    # Two sells of 0.5 and two buys of 0.5 -> 2.0 gross notional.
    assert np.isclose(xt.traded_notional(w_now, w_prev), 2.0)


def test_ls_month_cost_is_hand_checked_at_base_and_stress():
    # A full rotation of a long-short book: longs out 1, longs in 1,
    # shorts out 1, shorts in 1 -> 4.0 gross notional.
    w_prev = np.array([0.5, 0.5, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0])
    w_now = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, -0.5, -0.5])

    cost, turn = xt.month_costs(w_now, w_prev, xt.COST_BASE)
    assert np.isclose(turn, 4.0)
    # 5 bps x 4.0 = 20 bps; borrow 50 bps/yr on short notional 1.0 = 50/12 bps.
    expected = 0.0005 * 4.0 + (0.0050 / 12.0) * 1.0
    assert np.isclose(cost, expected)
    assert np.isclose(cost * 1e4, 20.0 + 50.0 / 12.0)

    cost_s, _ = xt.month_costs(w_now, w_prev, xt.COST_STRESS)
    expected_s = 0.0015 * 4.0 + (0.0100 / 12.0) * 1.0
    assert np.isclose(cost_s, expected_s)
    assert np.isclose(cost_s * 1e4, 60.0 + 100.0 / 12.0)
    assert cost_s > cost


def test_borrow_is_charged_on_short_notional_only_and_never_long_only():
    w_long_only_book = np.array([0.5, 0.5])
    cost, _ = xt.month_costs(w_long_only_book, np.zeros(2), xt.COST_BASE)
    assert np.isclose(cost, 0.0005 * 1.0)      # no short notional -> no borrow

    q = np.array([5, 5, 1, 1], dtype=np.int8)
    univ = np.ones(4, dtype=bool)
    # The TRADED vector for LONG_ONLY holds the Q5 leg only; the benchmark leg
    # appears in the RETURN vector and is never charged.
    w_trd = xt.traded_weights(q, 5, 1, "LONG_ONLY", univ)
    w_ret = xt.target_weights(q, 5, 1, "LONG_ONLY", univ)
    assert np.allclose(w_trd, [0.5, 0.5, 0.0, 0.0])
    assert w_trd.min() >= 0.0 and w_ret.min() < 0.0

    cost_lo, turn_lo = xt.month_costs(w_trd, np.zeros(4), xt.COST_BASE)
    # Only the Q5 leg is traded: notional 1.0 at 5 bps, and no borrow.
    assert np.isclose(turn_lo, 1.0)
    assert np.isclose(cost_lo, 0.0005)
    # Stress triples the per-side charge and still adds no borrow.
    cost_s, _ = xt.month_costs(w_trd, np.zeros(4), xt.COST_STRESS)
    assert np.isclose(cost_s, 0.0015)


def test_ls_traded_and_return_weights_are_the_same_vector():
    """For LS both legs are real positions, so there is nothing untraded."""
    q = np.array([1, 1, 3, 5, 5], dtype=np.int8)
    univ = np.ones(5, dtype=bool)
    assert np.allclose(xt.traded_weights(q, 5, 1, "LS", univ),
                       xt.target_weights(q, 5, 1, "LS", univ))


# ------------------------------------------------------- shared month index
def test_block_bootstrap_shares_one_month_index_across_the_ls_series():
    rng = np.random.default_rng(11)
    a = rng.normal(size=60)
    series = {"S1": a, "S2": rng.normal(size=60), "S3": a.copy()}
    res, idx = xt.shared_batch_nulls(series, mean_block=3.0, R=64, seed=5)

    assert idx.shape == (64, 60)
    # One shared draw: identical series must get byte-identical null draws.
    assert np.array_equal(res["S1"]["boots"], res["S3"]["boots"])
    assert res["S1"]["p"] == res["S3"]["p"]
    # And a different series on the SAME draw differs.
    assert not np.array_equal(res["S1"]["boots"], res["S2"]["boots"])
    # The shared draw is a valid month index.
    assert idx.min() >= 0 and idx.max() <= 59


def test_shared_batch_nulls_refuses_series_of_different_lengths():
    with pytest.raises(ValueError, match="one month index cannot be shared"):
        xt.shared_batch_nulls({"a": np.zeros(10), "b": np.zeros(11)}, R=4, seed=1)


def test_stationary_bootstrap_blocks_are_contiguous_more_often_than_not():
    rng = np.random.default_rng(3)
    idx = xt.stationary_bootstrap_indices(500, 3.0, rng)
    steps = (idx[1:] - idx[:-1]) % 500
    # mean block 3 => roughly two of three steps continue the block.
    assert 0.5 < float((steps == 1).mean()) < 0.85


def test_the_voting_null_is_centred_at_zero_so_the_mde_does_not_leak_the_mean():
    x = np.full(80, 0.01) + np.random.default_rng(4).normal(scale=0.001, size=80)
    res = xt.block_bootstrap_p(x, mean_block=3.0, R=400, seed=2)
    # Demeaning is what makes it a null: the draw is centred on zero, not 0.01.
    assert abs(res["null_mean"]) < 0.3 * res["null_sd"] + 1e-9
    assert res["p"] < 0.05           # a strong constant mean is still detected
    mde = xt.mde_from_null_sd(res["null_sd"])
    assert np.isclose(mde, (xt.Z_95 + xt.Z_80) * res["null_sd"])
    assert mde > 0


def test_a_zero_mean_series_is_not_significant():
    x = np.random.default_rng(9).normal(scale=0.02, size=120)
    res = xt.block_bootstrap_p(x - x.mean(), mean_block=3.0, R=400, seed=3)
    assert res["p"] > 0.2


# ----------------------------------------------------------------- verdicts
def test_verdict_bar_is_the_registered_five_signal_bar():
    assert xt.verdict(0.004, 0.001, 0.01) == "SOMETHING"
    assert xt.verdict(0.010, 0.001, 0.01) == "SOMETHING"      # inclusive
    assert xt.verdict(0.030, 0.001, 0.01) == "SUGGESTIVE"
    assert xt.verdict(0.050, 0.001, 0.01) == "SUGGESTIVE"     # inclusive
    assert xt.verdict(0.051, 0.001, 0.01) == "NOTHING"
    # A negative mean can never be SOMETHING however small the p.
    assert xt.verdict(0.0001, -0.001, 0.01) == "NOTHING"
    assert xt.verdict(float("nan"), 0.001, 0.01) == "INSTRUMENT-INVALID"


def test_effective_n_is_five_when_independent_and_one_when_identical():
    assert np.isclose(xt.effective_n(np.eye(5)), 5.0)
    assert np.isclose(xt.effective_n(np.ones((5, 5))), 1.0)


def test_annualised_sharpe_is_hand_checked():
    x = np.array([0.01, 0.03, 0.02, 0.02, 0.02, 0.02])
    expected = x.mean() / x.std(ddof=1) * np.sqrt(12)
    assert np.isclose(xt.annualised_sharpe(x), expected)


# ------------------------------------------------------- synthetic panels
# Calibrated so the planted gross long-short Sharpe is about 1.0 (measured
# 1.062 at 1000 names over 180 months, seed 21).
POSITIVE_CONTROL_BETA = 0.0006


def _synthetic_panel(n_names: int, n_months: int, beta: float, seed: int,
                     phi: float = 0.9, noise: float = 0.08) -> pd.DataFrame:
    """
    A panel whose signal carries a planted cross-sectional edge of strength
    beta: fwd_ret = beta * (signal standardized within the month) + noise.
    beta = 0 removes the edge entirely.

    The signal is PERSISTENT (AR(1), phi 0.9) like the real signals are, so the
    control pays a realistic turnover rather than rotating fully every month.
    A white-noise fixture would be rotated 100% monthly and its costs would say
    more about the fixture than about the harness.
    """
    rng = np.random.default_rng(seed)
    months = np.tile(np.arange(n_months), n_names)
    keys = np.repeat(np.arange(n_names), n_months)
    sig = xt.ar1_noise_signal(months, keys, phi, rng)
    df = pd.DataFrame({"month": months, "permaticker": keys, "sig": sig,
                       "in_universe": True})
    z = df.groupby("month")["sig"].transform(lambda s: (s - s.mean()) / s.std())
    df["fwd_ret"] = beta * z + rng.normal(scale=noise, size=len(df))
    return df


def test_positive_control_a_planted_edge_reads_something():
    panel = _synthetic_panel(1000, 180, beta=POSITIVE_CONTROL_BETA, seed=21)
    f = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_BASE).frame()
    sharpe = xt.annualised_sharpe(f["gross"].to_numpy())
    assert 0.8 < sharpe < 1.3, f"positive control miscalibrated: gross Sharpe {sharpe}"
    # Realistic turnover, not a full monthly rotation.
    assert f["turnover"].mean() < 2.0
    res = xt.block_bootstrap_p(f["net"].to_numpy(), mean_block=3.0, R=500, seed=1)
    assert xt.verdict(res["p"], float(f["net"].mean()), 0.01) == "SOMETHING"
    # It survives costs, which is the point of reading it NET.
    assert float(f["net"].mean()) > 0


def test_stress_costs_degrade_the_planted_edge_and_the_harness_reads_it():
    """
    The same planted gross-Sharpe-1.0 edge is read under both cost regimes.
    At base it survives; at stress (15 bps a side, 100 bps a year borrow) the
    cost hurdle is close to the whole gross premium and the edge is all but
    erased. The harness must show that degradation rather than hide it.
    """
    panel = _synthetic_panel(1000, 180, beta=POSITIVE_CONTROL_BETA, seed=21)
    base = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_BASE).frame()
    stress = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_STRESS).frame()

    assert np.isclose(base["gross"].mean(), stress["gross"].mean())   # gross is cost-free
    assert stress["net"].mean() < base["net"].mean()
    # Stress eats most of a gross-Sharpe-1.0 cross-sectional premium at this turnover.
    assert stress["net"].mean() < 0.25 * base["net"].mean()
    assert xt.annualised_sharpe(stress["net"].to_numpy()) < \
        xt.annualised_sharpe(base["net"].to_numpy())


def test_positive_control_with_the_edge_removed_reads_nothing():
    panel = _synthetic_panel(1000, 180, beta=0.0, seed=21)
    # Same fixture, same seed, same costs; only the planted edge is gone.
    s = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_BASE)
    f = s.frame()
    res = xt.block_bootstrap_p(f["net"].to_numpy(), mean_block=3.0, R=500, seed=1)
    assert xt.verdict(res["p"], float(f["net"].mean()), 0.01) == "NOTHING"


def test_control_verdict_band_passes_inside_and_fails_outside():
    """The acceptance test itself, as a pure function: both directions."""
    band = (0.01, 0.10)
    assert xt.control_verdict(0.05, band)
    assert xt.control_verdict(0.01, band)        # inclusive
    assert xt.control_verdict(0.10, band)        # inclusive
    assert not xt.control_verdict(0.00, band)    # mechanically nil -> HALT
    assert not xt.control_verdict(0.30, band)    # inflated -> HALT
    assert not xt.control_verdict(float("nan"), band)


def test_known_null_control_calibrates_on_gross_and_reports_net_beside_it():
    """
    Addendum A1. A no-edge panel put through the full pipeline: the GROSS arm
    sits at the null boundary and so estimates the test's size; the NET arm is
    mechanically near nil because a no-edge strategy paying costs has a true
    mean below zero. Both are reported; the band is adjudicated on gross.
    """
    panel = _synthetic_panel(300, 160, beta=0.0, seed=33)
    out = xt.known_null_control(panel, n_synthetic=60, phi=0.9, spec=xt.COST_BASE,
                                R=200, seed=44)
    assert out["calibration_arm"] == "gross"
    assert out["share_at_or_below_alpha"] == out["share_gross"]
    # Size on the boundary arm is not inflated.
    assert out["share_gross"] <= 0.15, out
    # The cost arm is strictly the more conservative of the two.
    assert out["share_net"] <= out["share_gross"], out
    assert out["p_net_median"] > out["p_gross_median"], out


def test_the_mean_block_of_three_is_load_bearing_against_serial_dependence():
    """
    RED WITNESS for the voting null's block structure. On a monthly series with
    genuine serial dependence and a TRUE mean of zero, an i.i.d. bootstrap
    understates the variance of the mean and its size inflates badly; the
    registered mean block of 3 is far closer to nominal. Measured here at
    phi 0.8 over 180 months: i.i.d. ~0.34 against block-3 ~0.17.

    Stated honestly in the other direction too: at phi 0 the block of 3 is
    mildly conservative-wrong (size ~0.07 against a nominal 0.05), so the
    in-run control on the real pipeline is what settles the actual size.
    """
    def ar1(n, phi, rng):
        e = rng.standard_normal(n)
        x = np.empty(n)
        x[0] = e[0]
        for t in range(1, n):
            x[t] = phi * x[t - 1] + np.sqrt(1 - phi * phi) * e[t]
        return x

    rng = np.random.default_rng(5)
    blocked, iid = [], []
    for i in range(120):
        x = ar1(180, 0.8, rng)
        blocked.append(xt.block_bootstrap_p(x, mean_block=3.0, R=150, seed=i)["p"])
        iid.append(xt.block_bootstrap_p(x, mean_block=1.0, R=150, seed=i)["p"])
    size_blocked = float((np.asarray(blocked) <= 0.05).mean())
    size_iid = float((np.asarray(iid) <= 0.05).mean())
    assert size_iid > size_blocked, (size_iid, size_blocked)
    assert size_iid > 0.20, size_iid          # the i.i.d. draw is plainly inflated
    assert size_blocked < size_iid - 0.05     # and blocking materially repairs it


def test_a_longer_block_widens_the_null_on_a_dependent_series():
    """The same property read through the null's spread rather than its size."""
    rng = np.random.default_rng(7)
    x = np.empty(240)
    e = rng.standard_normal(240)
    x[0] = e[0]
    for t in range(1, 240):
        x[t] = 0.8 * x[t - 1] + np.sqrt(1 - 0.64) * e[t]
    sd_iid = xt.block_bootstrap_p(x, mean_block=1.0, R=400, seed=1)["null_sd"]
    sd_blk = xt.block_bootstrap_p(x, mean_block=3.0, R=400, seed=1)["null_sd"]
    assert sd_blk > sd_iid
    # A wider null means a larger MDE: dependence costs detectable effect.
    assert xt.mde_from_null_sd(sd_blk) > xt.mde_from_null_sd(sd_iid)


def test_ar1_noise_carries_the_requested_persistence():
    months = np.tile(np.arange(300), 40)
    keys = np.repeat(np.arange(40), 300)
    rng = np.random.default_rng(5)
    x = xt.ar1_noise_signal(months, keys, 0.9, rng)
    df = pd.DataFrame({"m": months, "k": keys, "x": x}).sort_values(["k", "m"])
    rhos = [np.corrcoef(g["x"].to_numpy()[:-1], g["x"].to_numpy()[1:])[0, 1]
            for _, g in df.groupby("k")]
    assert 0.80 < float(np.mean(rhos)) < 0.96
    white = xt.ar1_noise_signal(months, keys, 0.0, rng)
    dfw = pd.DataFrame({"m": months, "k": keys, "x": white}).sort_values(["k", "m"])
    rw = [np.corrcoef(g["x"].to_numpy()[:-1], g["x"].to_numpy()[1:])[0, 1]
          for _, g in dfw.groupby("k")]
    assert abs(float(np.mean(rw))) < 0.12


def test_median_rank_autocorr_reads_high_for_sticky_and_low_for_white():
    months = np.tile(np.arange(60), 200)
    keys = np.repeat(np.arange(200), 60)
    rng = np.random.default_rng(6)
    sticky = pd.DataFrame({"month": months, "permaticker": keys, "in_universe": True,
                           "s": xt.ar1_noise_signal(months, keys, 0.95, rng)})
    white = pd.DataFrame({"month": months, "permaticker": keys, "in_universe": True,
                          "s": xt.ar1_noise_signal(months, keys, 0.0, rng)})
    assert xt.median_rank_autocorr(sticky, ["s"]) > 0.80
    assert abs(xt.median_rank_autocorr(white, ["s"])) < 0.15


# ------------------------------------------------------------------ mutation
def test_mutation_flipping_the_registered_direction_flips_the_ls_mean_sign():
    """
    MUTATION. If the registered direction is flipped, the long-short mean must
    change sign. A harness that ignored `direction` would return the same number
    twice and this test would fail.
    """
    panel = _synthetic_panel(500, 120, beta=0.01, seed=77)
    as_registered = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_BASE).frame()
    flipped = xt.build_series(panel, "sig", "LOW", "LS", xt.COST_BASE).frame()
    m_reg = float(as_registered["gross"].mean())
    m_flip = float(flipped["gross"].mean())
    assert m_reg > 0 and m_flip < 0, (m_reg, m_flip)
    assert np.isclose(m_reg, -m_flip, rtol=1e-9), (m_reg, m_flip)


def test_mutation_a_direction_blind_weighting_would_be_caught():
    """The same mutation stated at the weight layer: HIGH and LOW must differ."""
    q = np.array([1, 1, 5, 5], dtype=np.int8)
    univ = np.ones(4, dtype=bool)
    w_high = xt.target_weights(q, xt.long_side_quintile("HIGH"),
                               xt.short_side_quintile("HIGH"), "LS", univ)
    w_low = xt.target_weights(q, xt.long_side_quintile("LOW"),
                              xt.short_side_quintile("LOW"), "LS", univ)
    assert np.allclose(w_high, -w_low)
    assert not np.allclose(w_high, w_low)


def test_registered_signal_directions_are_the_five_in_the_registration():
    reg = json.loads((ROOT / "logs" / "r1" / "registration_v1.json").read_text())
    assert set(reg["signals"]) == set(xt.REGISTERED_SIGNALS)
    for name, spec in reg["signals"].items():
        assert spec["long"] == xt.REGISTERED_SIGNALS[name], name


def test_registered_universe_constants_match_the_registration():
    assert xt.MIN_PRICE == 5.00
    assert xt.MIN_SESSIONS_TRADED == 15
    assert xt.LOOKBACK_SESSIONS == 21
    assert xt.TOP_N_BY_DOLLAR_VOLUME == 1000
    assert xt.VOL_WINDOW_SESSIONS == 252
    assert xt.VOL_MIN_PRESENT == 200
    assert xt.N_QUINTILES == 5


# ------------------------------------------------- period-ordinal reporting
def test_month_ordinals_convert_to_timestamps_not_nanoseconds():
    """
    The panel's month column holds Period ordinals. Reading them with
    pd.to_datetime would treat them as nanoseconds since the epoch and select
    nothing, which is how the first T-R1 run reported every sub-period as
    "nan over 0 months".
    """
    months = pd.Series([336, 514])            # 1998-01 and 2012-11
    ts = xt.months_to_timestamps(months)
    assert ts.iloc[0] == pd.Timestamp("1998-01-31")
    assert ts.iloc[1] == pd.Timestamp("2012-11-30")
    # The failure mode this replaces.
    assert pd.to_datetime(months).iloc[0].year == 1970


def test_subperiod_means_select_the_right_months_from_ordinals():
    months = [pd.Period(f"{y}-{m:02d}", freq="M").ordinal
              for y in range(1998, 2013) for m in range(1, 13)]
    frame = pd.DataFrame({"month": months, "net": np.linspace(0.001, 0.01, len(months))})
    out = xt.subperiod_means(frame, [("1998-2002", "1998-01-01", "2002-12-31"),
                                     ("2003-2007", "2003-01-01", "2007-12-31"),
                                     ("2008-2012", "2008-01-01", "2012-12-31")])
    assert out["1998-2002"]["months"] == 60
    assert out["2003-2007"]["months"] == 60
    assert out["2008-2012"]["months"] == 60
    assert all(np.isfinite(d["mean_bps"]) for d in out.values())
    # The spans are disjoint and exhaust the frame.
    assert sum(d["months"] for d in out.values()) == len(months)


def test_subperiod_means_are_hand_checked():
    months = [pd.Period("2004-01", freq="M").ordinal,
              pd.Period("2004-02", freq="M").ordinal,
              pd.Period("2010-01", freq="M").ordinal]
    frame = pd.DataFrame({"month": months, "net": [0.01, 0.03, 0.05]})
    out = xt.subperiod_means(frame, [("2003-2007", "2003-01-01", "2007-12-31"),
                                     ("2008-2012", "2008-01-01", "2012-12-31")])
    assert out["2003-2007"]["months"] == 2
    assert np.isclose(out["2003-2007"]["mean_bps"], 200.0)     # mean 0.02 -> 200 bps
    assert out["2008-2012"]["months"] == 1
    assert np.isclose(out["2008-2012"]["mean_bps"], 500.0)


# ------------------------------- the permutation null, RETIRED from the harness
def test_the_permutation_null_is_no_longer_part_of_the_harness():
    """
    Ratified ruling, Sprint R1b: the within-date rank permutation is retired
    from the harness entirely. It survives only inside the historical T-R1
    runner, so that look stays reproducible, and nothing new may reach for it.
    """
    assert not hasattr(xt, "within_date_permutation_p")
    src = (ROOT / "tools" / "xs_triage.py").read_text()
    assert "within_date_permutation_p" not in src
    assert "RETIRED" in src, "the harness must say why it is gone"
    import run_t_r1
    assert hasattr(run_t_r1, "within_date_permutation_p")


def test_the_retired_permutation_null_rotated_harder_than_the_real_signal():
    """
    The measured reason it was retired, kept as a standing record. Permuting
    across names destroys the signal's month-to-month persistence, so the
    permuted book trades more and pays more cost; a persistent real signal then
    beats it on net whatever its edge.
    """
    import run_t_r1
    panel = _synthetic_panel(400, 90, beta=0.0, seed=55, phi=0.95)
    res = run_t_r1.within_date_permutation_p(panel, "sig", "HIGH", "LS",
                                             xt.COST_BASE, R=12, seed=3)
    assert res["permuted_turnover"] > res["obs_turnover"], res
    # No edge at all, yet the net permutation p is flattered toward significance
    # while the gross p (unconfounded by turnover) is not.
    assert res["p"] < res["p_gross"], res
    assert res["p_gross"] > 0.1, res
