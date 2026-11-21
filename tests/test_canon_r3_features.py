"""
================================================================================
Script:       test_canon_r3_features.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The feature builder's guards: the half-A window filter, windows
              that end at g and not at s_t, the within-month rank scaling, and
              the truncation property on a synthetic panel.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas, pyarrow; tools/canon_r3_*.py
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_r3_features as F  # noqa: E402
import canon_r3_paths as P  # noqa: E402
import canon_r3_synthetic as S  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path, monkeypatch):
    """No test may reach the banked panel, the feature file or the snapshot."""
    st = S.snapshot_state()
    info = S.install(tmp_path)
    monkeypatch.setattr(F, "FIRST_FORMATION_MONTH", info["months"][2])
    monkeypatch.setattr(F, "LAST_FORMATION_MONTH", info["months"][-1])
    yield info
    S.restore_state(st)


def test_the_fixture_really_redirected_every_path(synthetic):
    assert str(synthetic["tmp"]) in str(P.banked_panel())
    assert str(synthetic["tmp"]) in str(P.bulk_stocks())
    assert str(synthetic["tmp"]) in str(P.feature_file())
    assert not P.banked_panel().samefile(
        ROOT / "data" / "derived" / "panel_monthly_20261004p_halfA_v2.parquet") \
        if (ROOT / "data" / "derived").exists() else True


# ------------------------------------------------------------ the window guard
def test_the_loader_never_returns_a_row_at_or_after_the_cutoff(synthetic):
    keep = set(synthetic["perm"].values())
    d = F.load_daily_half_a(keep_permatickers=keep)
    cut = pd.Timestamp(synthetic["cutoff"])
    assert pd.Timestamp(d.max_date) < cut, d.max_date
    assert pd.Timestamp(d.calendar.max()) < cut
    # the synthetic snapshot really does hold post-cutoff rows, so this bites
    import pyarrow.parquet as pq
    allrows = pq.read_table(P.bulk_stocks(), columns=["date"]).column("date").to_pylist()
    assert max(allrows) >= synthetic["cutoff"], "the fixture must carry post-cut rows"


def test_the_loader_calls_the_shared_half_a_filter(synthetic):
    """Structural: the gate is xs_triage.filter_to_half_a, by name, in the loader."""
    src = (ROOT / "tools" / "canon_r3_features.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_daily_half_a")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "filter_to_half_a"]
    assert calls, "load_daily_half_a does not call xs_triage.filter_to_half_a"


def test_a_constructed_late_row_raises_through_the_shared_guard():
    import xs_triage as xt
    df = pd.DataFrame({"date": pd.to_datetime(["2003-12-30", "2004-01-02"])})
    with pytest.raises(xt.WindowViolation):
        xt.assert_no_rows_at_or_after(df["date"], S.CUTOFF, "test")


# ----------------------------------------------------- the anchor is g, not s_t
def test_the_anchor_is_the_session_before_the_month_end(synthetic):
    keep = set(synthetic["perm"].values())
    d = F.load_daily_half_a(keep_permatickers=keep)
    mes = F.month_end_sessions(d)
    panel = pd.read_parquet(P.banked_panel())
    m_lo, m_hi = F.formation_months()
    u = panel.loc[panel["in_universe"] & (panel["month"] >= m_lo)
                  & (panel["month"] <= m_hi)]
    m = int(u["month"].max())
    assert mes[m] - 1 >= 0
    feats, _ = F.build_features(panel=panel, daily=d, months_subset=[m],
                                rank_scale=False)
    # REV5 reads closeadj at g and at g-5; recompute by hand for one name
    pmk = int(feats["permaticker"].iloc[0])
    g = mes[m] - 1
    s = d.slot[pmk]
    a, b = d.start[s], d.end[s]
    gi, ca = d.gi[a:b], d.ca[a:b]
    at = lambda x: float(ca[np.searchsorted(gi, x, side="right") - 1])
    want = at(g) / at(g - 5) - 1.0
    got = float(feats.loc[feats["permaticker"] == pmk, "REV5"].iloc[0])
    assert abs(got - want) < 1e-12, (got, want)
    # and it is NOT the s_t-anchored value, which the fixture makes different
    wrong = at(mes[m]) / at(mes[m] - 5) - 1.0
    assert abs(got - wrong) > 1e-12, "the fixture failed to separate g from s_t"


def test_truncating_the_daily_data_at_g_changes_no_feature(synthetic):
    """Section 2's truncation test, on the synthetic panel."""
    panel = pd.read_parquet(P.banked_panel())
    res = F.truncation_check([synthetic["months"][-1], synthetic["months"][-3]],
                             panel=panel)
    assert res["all_identical"], res
    assert res["worst_max_abs_delta"] == 0.0
    assert len(res["features"]) == 23


# --------------------------------------------------------- the rank scaling
def test_the_rank_scaling_is_within_the_month_and_lands_in_the_half_open_band():
    rng = np.random.default_rng(3)
    frames = []
    for i, m in enumerate([100, 101, 102]):
        frames.append(pd.DataFrame({
            "month": m, "permaticker": np.arange(50),
            # a deliberate per-month level shift: an across-month ranking cannot
            # give every month the same top value
            "X": rng.normal(size=50) + 100.0 * i}))
    df = pd.concat(frames, ignore_index=True)
    out = F.rank_scale_within_month(df, ["X"])
    for m, g in out.groupby("month"):
        n = len(g)
        assert abs(float(g["X"].max()) - 0.5) < 1e-12, m
        assert abs(float(g["X"].min()) - (1.0 / n - 0.5)) < 1e-12, m
        # ranks 1..n scaled by rank/n - 0.5 have mean exactly 1/(2n)
        assert abs(float(g["X"].mean()) - 1.0 / (2 * n)) < 1e-12, m
    assert float(out["X"].max()) <= 0.5 + 1e-12
    assert float(out["X"].min()) > -0.5


def test_the_rank_scaling_keeps_nan_as_nan():
    df = pd.DataFrame({"month": [1] * 5, "X": [1.0, np.nan, 3.0, np.nan, 5.0]})
    out = F.rank_scale_within_month(df, ["X"])
    assert out["X"].isna().tolist() == [False, True, False, True, False]
    v = out.loc[out["X"].notna(), "X"].to_numpy()
    assert abs(v.max() - 0.5) < 1e-12 and len(v) == 3


def test_no_raw_value_survives_the_build(synthetic):
    panel = pd.read_parquet(P.banked_panel())
    feats, meta = F.build_features(panel=panel)
    for n in F.FEATURE_NAMES:
        v = feats[n].dropna().to_numpy()
        if v.size:
            assert v.min() > -0.5 - 1e-12 and v.max() <= 0.5 + 1e-12, n
    assert meta["rows"] == len(feats)
    assert set(F.FEATURE_NAMES) <= set(feats.columns)


def test_the_feature_list_is_the_registered_twenty_five():
    assert len(F.FEATURE_NAMES) == 25
    assert len(set(F.FEATURE_NAMES)) == 25
    assert F.FEATURE_NAMES[:7] == ["REV1", "MOM3_1", "MOM6_1", "MOM12_1",
                                   "MOM12_7", "SEAS12", "REV5"]
    assert F.FEATURE_NAMES[-3:] == ["PRC", "BETA", "LOGDV"]
    assert set(F.PANEL_FEATURES) == {"BETA", "LOGDV"}
    assert len(F.TRUNCATION_FEATURES) == 23


def test_open_high_low_and_close_are_never_read():
    src = (ROOT / "tools" / "canon_r3_features.py").read_text()
    tree = ast.parse(src)
    read = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                and n.func.attr == "scanner":
            for kw in n.keywords:
                if kw.arg == "columns":
                    read |= {e.value for e in kw.value.elts}
    assert read == {"ticker", "date", "closeadj", "closeunadj", "volume"}, read
