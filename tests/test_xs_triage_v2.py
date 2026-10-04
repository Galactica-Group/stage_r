"""
Sprint R1b guard tests for harness v2 — the neutralised cross-sectional
statistic. v1 is untouched and keeps its own suite.

The neutrality property is the whole point of v2, so it is asserted directly on
the weights to 1e-9, and a pure-beta fixture proves v2 reads it away while v1
does not.
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


# ------------------------------------------------------------- normal scores
def test_norm_ppf_matches_known_quantiles():
    for p, want in [(0.5, 0.0), (0.975, 1.9599639845400545),
                    (0.025, -1.9599639845400545), (0.001, -3.090232306167813),
                    (0.999, 3.090232306167813)]:
        assert abs(xt.norm_ppf(p) - want) < 1e-12, p
    assert np.isnan(xt.norm_ppf(0.0)) and np.isnan(xt.norm_ppf(1.0))


def test_normal_scores_are_symmetric_and_cached():
    s = xt.normal_scores(1000)
    assert abs(float(s.mean())) < 1e-12
    assert np.allclose(s, -s[::-1])
    assert np.all(np.diff(s) > 0)
    assert xt.normal_scores(1000) is xt.normal_scores(1000)   # cached by n


def test_rank_to_normal_z_applies_the_registered_direction():
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    hi = xt.rank_to_normal_z(v, "HIGH")
    lo = xt.rank_to_normal_z(v, "LOW")
    # HIGH: the largest value gets the largest z. LOW: the smallest does.
    assert np.argmax(hi) == 4 and np.argmax(lo) == 0
    assert np.allclose(hi, -lo)
    assert np.isclose(hi.mean(), 0.0, atol=1e-12)
    with pytest.raises(ValueError):
        xt.rank_to_normal_z(v, "SIDEWAYS")


def test_rank_to_normal_z_is_rank_only_so_outliers_cannot_dominate():
    a = xt.rank_to_normal_z(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), "HIGH")
    b = xt.rank_to_normal_z(np.array([1.0, 2.0, 3.0, 4.0, 1e9]), "HIGH")
    assert np.allclose(a, b)


def test_rank_to_normal_z_passes_nan_through():
    z = xt.rank_to_normal_z(np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0]), "HIGH")
    assert np.isnan(z[1])
    assert np.isfinite(z[[0, 2, 3, 4, 5]]).all()


# ------------------------------------------------------- the v2 weights
def _fixture(n=300, seed=1):
    rng = np.random.default_rng(seed)
    sector = np.array([f"S{i % 4}" for i in range(n)])
    beta = rng.normal(1.0, 0.3, n)
    ldv = rng.normal(15.0, 1.5, n)
    sig = rng.normal(size=n)
    return sector, beta, ldv, sig


def test_v2_weights_have_unit_z_exposure_and_zero_control_exposure_to_1e9():
    """THE defining property of v2, asserted on the weights themselves."""
    sector, beta, ldv, sig = _fixture()
    z = xt.rank_to_normal_z(sig, "HIGH")
    X, names, z_idx = xt.build_design(z, sector, beta, ldv)
    w = xt.fm_slope_weights(X, z_idx)
    assert w is not None
    exposures = w @ X
    assert abs(exposures[z_idx] - 1.0) < 1e-9, "unit z exposure"
    for j, nm in enumerate(names):
        if j == z_idx:
            continue
        assert abs(exposures[j]) < 1e-9, f"exposure to {nm} is {exposures[j]}"
    # Dollar-neutral, so neutrality extends to the DROPPED sector level too.
    assert abs(float(w.sum())) < 1e-9
    for lv in np.unique(sector):
        d = (sector == lv).astype(float)
        assert abs(float(w @ d)) < 1e-9, f"exposure to dropped/retained {lv}"


def test_v2_weights_are_minimum_norm_among_portfolios_meeting_the_constraints():
    sector, beta, ldv, sig = _fixture(n=200, seed=3)
    z = xt.rank_to_normal_z(sig, "HIGH")
    X, _, z_idx = xt.build_design(z, sector, beta, ldv)
    w = xt.fm_slope_weights(X, z_idx)
    rng = np.random.default_rng(9)
    for _ in range(20):
        # Any perturbation inside the null space of X' keeps the constraints and
        # must not shrink the norm.
        v = rng.normal(size=w.shape[0])
        v -= X @ np.linalg.pinv(X.T @ X) @ (X.T @ v)      # project onto null(X')
        assert np.linalg.norm(w + 0.1 * v) >= np.linalg.norm(w) - 1e-9


def test_scaling_gives_gross_two_and_one_long_one_short():
    sector, beta, ldv, sig = _fixture()
    z = xt.rank_to_normal_z(sig, "HIGH")
    X, _, z_idx = xt.build_design(z, sector, beta, ldv)
    ws = xt.scale_to_gross_two(xt.fm_slope_weights(X, z_idx))
    assert np.isclose(float(np.abs(ws).sum()), 2.0, atol=1e-12)
    assert np.isclose(float(ws[ws > 0].sum()), 1.0, atol=1e-9)
    assert np.isclose(float(ws[ws < 0].sum()), -1.0, atol=1e-9)
    assert np.isclose(float(ws.sum()), 0.0, atol=1e-9)


def test_rank_deficient_design_returns_none_rather_than_a_non_neutral_book():
    n = 20
    sector = np.array(["S0"] * n)
    beta = np.ones(n)                      # constant -> collinear with intercept
    ldv = np.ones(n)
    z = xt.rank_to_normal_z(np.arange(float(n)), "HIGH")
    X, _, z_idx = xt.build_design(z, sector, beta, ldv)
    assert xt.fm_slope_weights(X, z_idx) is None
    # And a design with fewer rows than columns is refused too.
    sector2, beta2, ldv2, sig2 = _fixture(n=4, seed=2)
    z2 = xt.rank_to_normal_z(sig2, "HIGH")
    X2, _, zi2 = xt.build_design(z2, sector2, beta2, ldv2)
    assert xt.fm_slope_weights(X2, zi2) is None


def test_the_sector_missing_group_is_its_own_sector_and_is_neutralised():
    n = 240
    rng = np.random.default_rng(5)
    sector = np.array([xt.NO_SECTOR if i % 6 == 0 else f"S{i % 3}" for i in range(n)])
    beta = rng.normal(1.0, 0.3, n)
    ldv = rng.normal(15.0, 1.0, n)
    z = xt.rank_to_normal_z(rng.normal(size=n), "HIGH")
    X, names, z_idx = xt.build_design(z, sector, beta, ldv)
    assert any(nm == f"sector::{xt.NO_SECTOR}" for nm in names) or \
        sorted(np.unique(sector))[0] == xt.NO_SECTOR
    w = xt.fm_slope_weights(X, z_idx)
    assert w is not None
    d = (sector == xt.NO_SECTOR).astype(float)
    assert abs(float(w @ d)) < 1e-9, "the no-sector group must be neutralised too"
    assert int(d.sum()) == n // 6


def test_build_design_drops_one_sector_level_and_any_empty_level():
    sector, beta, ldv, sig = _fixture()
    z = xt.rank_to_normal_z(sig, "HIGH")
    X, names, _ = xt.build_design(z, sector, beta, ldv)
    n_levels = len(np.unique(sector))
    # intercept + z + (levels - 1) dummies + beta + log_dv
    assert len(names) == 2 + (n_levels - 1) + 2
    assert names[0] == "intercept" and names[1] == "z"
    assert names[-2] == "beta" and names[-1] == "log_dv"
    assert np.linalg.matrix_rank(X) == X.shape[1]


def test_centring_beta_and_log_dv_cannot_change_the_z_slope():
    sector, beta, ldv, sig = _fixture(seed=7)
    z = xt.rank_to_normal_z(sig, "HIGH")
    rng = np.random.default_rng(11)
    r = 0.3 * z + 0.5 * beta + rng.normal(scale=0.05, size=len(z))
    X, _, z_idx = xt.build_design(z, sector, beta, ldv)
    slope_centred = xt.fm_slope_weights(X, z_idx) @ r
    # The same design built on uncentred regressors.
    Xu = X.copy()
    Xu[:, -2] = beta
    Xu[:, -1] = ldv
    slope_raw = xt.fm_slope_weights(Xu, z_idx) @ r
    assert np.isclose(slope_centred, slope_raw, rtol=1e-9)


# ------------------------------------- the pure-beta fixture (the mutation target)
def _pure_beta_panel(n_names=400, n_months=90, seed=17) -> pd.DataFrame:
    """
    A panel whose SIGNAL IS BETA and whose returns are driven only by beta
    times a market factor. There is no idiosyncratic cross-sectional edge, so a
    beta-neutral statistic must read about nothing; a raw quintile sort must
    read the factor and so read something.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(1.0, 0.35, n_names)
    sector = np.array([f"S{i % 4}" for i in range(n_names)])
    ldv = rng.normal(15.0, 1.0, n_names)
    rows = []
    for m in range(n_months):
        mkt = rng.normal(0.012, 0.045)                 # a positive market drift
        fwd = beta * mkt + rng.normal(scale=0.02, size=n_names)
        rows.append(pd.DataFrame({
            "month": m, "permaticker": np.arange(n_names),
            "sig": beta, "beta": beta, "log_dv": ldv, "sector": sector,
            "fwd_ret": fwd, "in_universe": True}))
    return pd.concat(rows, ignore_index=True)


def test_a_pure_beta_signal_reads_about_zero_under_v2_and_nonzero_under_v1():
    """
    v2's reason for existing. LOWVOL's v1 long-short was largely a short market
    position; this fixture is that effect in pure form.
    """
    panel = _pure_beta_panel()
    s2, _ = xt.build_series_v2(panel, "sig", "HIGH", "LS", xt.COST_BASE)
    f2 = s2.frame()
    f1 = xt.build_series(panel, "sig", "HIGH", "LS", xt.COST_BASE).frame()

    m1 = abs(float(f1["gross"].mean()))
    m2 = abs(float(f2["gross"].mean()))
    assert m1 > 20e-4, f"v1 must pick up the factor, got {m1*1e4:.2f} bps"
    assert m2 < 2e-4, f"v2 must neutralise it, got {m2*1e4:.2f} bps"
    assert m2 < 0.1 * m1
    # And the volatility collapses, which is the MDE argument.
    assert f2["gross"].std(ddof=1) < 0.25 * f1["gross"].std(ddof=1)


def test_v2_still_reads_a_genuine_idiosyncratic_edge_in_the_same_fixture():
    """The neutralisation must not be a blanket deadener."""
    panel = _pure_beta_panel()
    rng = np.random.default_rng(23)
    extra = rng.normal(size=len(panel))
    panel["sig2"] = panel["beta"].to_numpy() * 0.0 + extra
    panel["fwd_ret"] = panel["fwd_ret"].to_numpy() + 0.01 * extra
    s2, _ = xt.build_series_v2(panel, "sig2", "HIGH", "LS", xt.COST_BASE)
    assert float(s2.frame()["gross"].mean()) > 5e-4


# ------------------------------------------------------------- long-only v2
def test_v2_long_only_is_the_top_quintile_of_residualised_z_minus_the_universe():
    panel = _pure_beta_panel(n_names=250, n_months=40)
    s, _ = xt.build_series_v2(panel, "sig", "HIGH", "LONG_ONLY", xt.COST_BASE)
    f = s.frame()
    assert len(f) > 0
    # The traded leg is long-only, so it can never be charged borrow: the
    # long-only net must sit above what the same gross would net under stress.
    s_stress, _ = xt.build_series_v2(panel, "sig", "HIGH", "LONG_ONLY",
                                     xt.COST_STRESS)
    assert f["net"].mean() > s_stress.frame()["net"].mean()
    assert f["turnover"].mean() > 0


# --------------------------------------------- the deliberate positive-control leak
def test_the_positive_control_leak_is_deliberate_and_named():
    """
    plant_signal takes the REALISED next-month residual return. That is future
    information and it is the point: a positive control proves the instrument
    can see an edge that is really there. This test exists so the leak can never
    be mistaken for an accident, and so that nothing else in the harness grows a
    dependency on it.
    """
    assert "LEAK IS DELIBERATE" in xt.plant_signal.__doc__
    assert "never used for a verdict" in xt.plant_signal.__doc__
    assert "REALISED future information" in xt.residual_next_month_return.__doc__

    resid = np.array([0.1, -0.2, 0.3])
    zero_noise = xt.plant_signal(resid, 0.0, np.random.default_rng(1))
    assert np.allclose(zero_noise, resid), "sigma 0 must be the pure planted term"
    # More noise means less of the planted term, monotonically in correlation.
    r = np.random.default_rng(2).normal(size=4000)
    c_small = np.corrcoef(r, xt.plant_signal(r, 0.5, np.random.default_rng(3)))[0, 1]
    c_big = np.corrcoef(r, xt.plant_signal(r, 8.0, np.random.default_rng(3)))[0, 1]
    assert c_small > c_big


def test_residualised_next_month_return_removes_the_controls():
    panel = _pure_beta_panel(n_names=200, n_months=30)
    resid = xt.residual_next_month_return(panel)
    sub = panel.loc[panel["in_universe"].astype(bool)].copy()
    sub["resid"] = resid
    # Inside each month the residual is orthogonal to beta by construction.
    for _, g in sub.groupby("month"):
        rr = g["resid"].to_numpy()
        bb = g["beta"].to_numpy()
        if np.isfinite(rr).sum() < 50:
            continue
        assert abs(float(np.corrcoef(rr, bb)[0, 1])) < 1e-8
        assert abs(float(rr.mean())) < 1e-10


# --------------------------------------------------------- tightened category rule
def test_the_tightened_category_rule_admits_only_two_categories():
    sep = pd.DataFrame({
        "table": ["SEP"] * 7, "permaticker": range(7), "ticker": list("ABCDEFG"),
        "category": ["Domestic Common Stock",
                     "Domestic Common Stock Primary Class",
                     "Domestic Common Stock Secondary Class",
                     "ADR Common Stock",
                     "Canadian Common Stock",
                     "Domestic Preferred Stock",
                     "ADR Common Stock Primary Class"]})
    keep, counts = xt.admitted_common_stock_tight(sep)
    assert set(keep["ticker"]) == {"A", "B"}
    assert counts == {"Domestic Common Stock": 1,
                      "Domestic Common Stock Primary Class": 1}
    # v1's looser rule would have admitted five of these; both still exist.
    keep_v1, _ = xt.admitted_common_stock(sep)
    assert len(keep_v1) == 6 and len(keep) == 2


def test_tight_categories_constant_matches_the_registration():
    reg = json.loads((ROOT / "logs" / "r1b" / "registration_v2.json").read_text())
    rule = reg["universe_rules_in_order"][0]
    for c in xt.TIGHT_CATEGORIES:
        assert c in rule
    assert len(xt.TIGHT_CATEGORIES) == 2
    assert reg["verdict_bar"]["bar"] == 0.005


# ------------------------------------------------------------- window guards
def test_the_rebuilt_v2_panel_carries_no_month_at_or_after_the_cutoff():
    path = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"
    if not path.exists():
        pytest.skip("v2 panel not built in this environment")
    m = pd.read_parquet(path, columns=["month_ts", "in_universe"])
    assert m["month_ts"].max() < pd.Timestamp("2013-01-01")
    held = m.loc[m["in_universe"].astype(bool), "month_ts"]
    assert held.max() <= pd.Timestamp("2012-11-30")


def test_the_rebuilt_v2_panel_carries_the_v2_controls():
    path = ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet"
    if not path.exists():
        pytest.skip("v2 panel not built in this environment")
    df = pd.read_parquet(path, columns=["in_universe", "beta", "log_dv", "sector"])
    u = df.loc[df["in_universe"].astype(bool)]
    assert np.isfinite(u["log_dv"].to_numpy()).all()
    assert np.isfinite(u["beta"].to_numpy()).mean() > 0.9
    assert u["sector"].notna().all()
    assert (u["sector"] == xt.NO_SECTOR).sum() > 0, "the no-sector group must exist"
