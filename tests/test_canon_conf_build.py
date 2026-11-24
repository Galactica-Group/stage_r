"""
================================================================================
Script:       test_canon_conf_build.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The confirmation's build guards: the 2021 cutoff, the generated
              features copy, the window as a parameter, the scramble, the
              stopped-stock bound's isolation from the vote, and the gate.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas, pyarrow; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
================================================================================
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_conf_copies as C  # noqa: E402
import canon_conf_features as F  # noqa: E402
import canon_conf_gate as G  # noqa: E402
import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402
import canon_conf_stopped as S  # noqa: E402
import canon_conf_synthetic as SY  # noqa: E402
import canon_r3_machine as M  # noqa: E402
import xs_triage as xt  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    """No test may reach the real panel, features, snapshot or read 3's files."""
    st = SY.snapshot_state()
    info = SY.install(tmp_path)
    yield info
    SY.restore_state(st)


def test_the_fixture_redirected_every_path(synthetic):
    tmp = str(synthetic["tmp"])
    for p in (P.repo(), P.snapshot(), P.derived(), P.scratch(), P.windows_json(),
              P.panel_file(), P.feature_file(), P.pool_v1(), P.controls_v1(),
              P.canon_dir(), P.spent_marker(), P.lock_file(), P.reports_dir(),
              P.read_result(), P.ledger(), P.requirements_lock(),
              P.r3_seal(), P.r3_result()):
        assert str(p).startswith(tmp), str(p)


# ----------------------------------------------------------- 5.1 the copy rule
def test_the_features_copy_is_the_original_plus_the_declared_substitutions():
    """Re-derive it from the original and find it byte-identical."""
    derived = C.derive("features")
    on_disk = C.copy_path("features").read_text()
    assert derived == on_disk, "canon_conf_features.py is not the original plus " \
                               "the declared substitutions"


def test_every_declared_substitution_anchor_is_unique_in_the_original():
    orig = C.original_path("features").read_text()
    for old, _new in C.COPIES["features"][2]:
        assert orig.count(old) == 1, (old, orig.count(old))


def test_the_copy_changes_only_the_window_and_its_own_identity():
    orig = C.original_path("features").read_text().splitlines()
    copy = C.copy_path("features").read_text().splitlines()
    assert len(orig) == len(copy), "the copy changed the line count"
    changed = [i for i, (a, b) in enumerate(zip(orig, copy)) if a != b]
    assert len(changed) == len(C.COPIES["features"][2]), changed
    window_or_identity = ("FORMATION_MONTH", "N_FORMATION_MONTHS", "windows[",
                          "Script:", "Phase:", "canon_r3_paths", "Last Sprint:")
    for i in changed:
        assert any(t in orig[i] for t in window_or_identity), (i, orig[i])


def test_the_copy_carries_the_confirmation_window():
    assert F.FIRST_FORMATION_MONTH == "1998-12"
    assert F.LAST_FORMATION_MONTH == "2020-11"
    assert F.N_FORMATION_MONTHS == 264
    lo, hi = F.formation_months()
    assert hi - lo + 1 == 264


# ----------------------------------------------------------- the 2021 cutoff
def test_the_cutoff_is_the_holdout_start_from_the_config(synthetic):
    import canon_conf_panel as CP
    w = CP.conf_windows()
    assert w["half_a"]["daily_row_exclusive_max"] == "2021-01-01"
    assert w["half_a"]["return_months_through"] == "2020-12"


def test_the_feature_loader_reads_the_holdout_start_not_half_a():
    src = C.copy_path("features").read_text()
    assert 'cut = windows["holdout"]["start"]' in src
    assert 'windows["half_a"]["daily_row_exclusive_max"]' not in src


def test_the_loader_calls_the_shared_half_a_filter():
    tree = ast.parse(C.copy_path("features").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "load_daily_half_a")
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute)
             and n.func.attr == "filter_to_half_a"]
    assert calls, "the feature loader does not call xs_triage.filter_to_half_a"


def test_a_constructed_2021_row_raises_through_the_shared_guard():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-12-31", "2021-01-04"])})
    with pytest.raises(xt.WindowViolation):
        xt.assert_no_rows_at_or_after(df["date"], "2021-01-01", "test")


def test_the_panel_builder_and_the_honest_module_name_no_2013_cutoff():
    src = (ROOT / "tools" / "canon_conf_panel.py").read_text()
    assert "2013-01-01" not in src
    assert not (ROOT / "tools" / "canon_conf_honest.py").exists(), \
        "the struck key's module must be gone"


# ------------------------------------------------------ the window as parameter
def test_the_eight_cuts_and_ninety_six_scored_months():
    assert len(PL.CONF_CUTS) == 8
    assert str(pd.Period(ordinal=PL.CONF_CUTS[0], freq="M")) == "2012-12"
    assert str(pd.Period(ordinal=PL.CONF_CUTS[-1], freq="M")) == "2019-12"
    assert len(PL.CONF_SCORED) == PL.CONF_N_SCORED_MONTHS == 96
    assert str(pd.Period(ordinal=PL.CONF_SCORED[0], freq="M")) == "2012-12"
    assert str(pd.Period(ordinal=PL.CONF_SCORED[-1], freq="M")) == "2020-11"


def test_every_scored_month_is_covered_by_exactly_one_cut():
    cov = {}
    for C_ in PL.CONF_CUTS:
        for m in range(C_, C_ + 12):
            cov[m] = cov.get(m, 0) + 1
    for m in PL.CONF_SCORED:
        assert cov.get(m) == 1, m
    assert set(cov) == set(PL.CONF_SCORED)


class SpyModel:
    seen: list = []

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        SpyModel.seen.append(X.shape)
        return self

    def predict(self, X):
        return np.random.default_rng(self.rs).normal(size=X.shape[0])


def test_the_walk_trains_through_c_minus_two_and_scores_from_c(monkeypatch):
    w = SY.world_frame(n_names=120, n_months=40, first="1998-12")
    monkeypatch.setattr(PL, "HistGradientBoostingRegressor", SpyModel)
    monkeypatch.setattr(M, "SEEDS", (1,))
    cuts = [pd.Period("2001-12", freq="M").ordinal]
    y = np.asarray(np.random.default_rng(0).normal(size=len(w)))
    _, diags = PL.walk_predict(w, y, cuts, first_formation="1998-12")
    live = [d for d in diags if not d["skipped"]]
    assert live, diags
    for d in live:
        Cp = pd.Period(d["cut"], freq="M")
        assert d["train_last"] == str(Cp - 2), (d["cut"], d["train_last"])
        assert d["score_first"] == str(Cp)
        assert d["train_first"] == "1998-12"


def test_the_feature_matrix_is_the_twenty_five_features(monkeypatch):
    w = SY.world_frame(n_names=120, n_months=40)
    SpyModel.seen = []
    monkeypatch.setattr(PL, "HistGradientBoostingRegressor", SpyModel)
    monkeypatch.setattr(M, "SEEDS", (1,))
    cuts = [pd.Period("2001-12", freq="M").ordinal]
    PL.walk_predict(w, np.asarray(np.random.default_rng(0).normal(size=len(w))), cuts)
    assert SpyModel.seen
    for shape in SpyModel.seen:
        assert shape[1] == 25, shape


def test_neither_the_label_nor_a_realised_return_is_a_feature():
    for bad in ("fwd_ret", "fwd_stopped", "label", "y", "ret_m", "sector",
                "honest_fwd_ret"):
        assert bad not in F.FEATURE_NAMES


# ------------------------------------------------------------- the scramble
def _frame(n=20, months=(100, 101, 102), seed=5):
    rng = np.random.default_rng(seed)
    recs = []
    for m in months:
        for i in range(n):
            recs.append({"permaticker": 1000 + i, "month": m,
                         "fwd_ret": float(rng.normal()), "fwd_stopped": False,
                         "in_universe": True, "beta": 1.0, "log_dv": 16.0,
                         "sector": "A"})
    return pd.DataFrame(recs)


def test_the_scramble_is_not_the_identity():
    f = _frame()
    w = PL.scramble_targets(f, 1, PL.CONF_SEED_BASE)
    assert (w["fwd_ret"].to_numpy() != f["fwd_ret"].to_numpy()).all()


def test_the_scramble_preserves_each_months_multiset_and_moves_no_month():
    f = _frame()
    w = PL.scramble_targets(f, 3, PL.CONF_SEED_BASE)
    for m in f["month"].unique():
        a = np.sort(f.loc[f["month"] == m, "fwd_ret"].to_numpy())
        b = np.sort(w.loc[w["month"] == m, "fwd_ret"].to_numpy())
        assert a.tobytes() == b.tobytes(), m
        src = set(f.loc[f["month"] == m, "fwd_ret"].round(12))
        assert set(w.loc[w["month"] == m, "fwd_ret"].round(12)) <= src, m


def test_the_seed_base_is_seventeen_thousand_and_is_load_bearing():
    f = _frame()
    a = PL.scramble_targets(f, 1, 17000)["fwd_ret"].to_numpy()
    b = PL.scramble_targets(f, 1, 7000)["fwd_ret"].to_numpy()
    assert PL.CONF_SEED_BASE == 17000
    assert a.tobytes() != b.tobytes(), "the seed base does not change the world"
    want = np.random.default_rng(17000 + 4).random(20)
    assert np.array_equal(PL.world_keys(f["permaticker"], 4, 17000).to_numpy(), want)


# --------------------------------------------------------------- world zero
def test_world_zero_is_refused_on_a_flag():
    f = _frame()
    with pytest.raises(PL.WorldZeroRefused):
        PL.build_world(f, 0, PL.CONF_CUTS)


def test_world_zero_is_refused_always_on_read_threes_cuts(synthetic, tmp_path):
    f = _frame()
    p = tmp_path / "reg.json"
    p.write_text('{"data_pins": {}}')
    reg = G.load_registration(p, G.sha256_file(p))
    with pytest.raises(PL.WorldZeroRefused):
        PL.build_world(f, 0, PL.R3_CUTS, registration=reg)


def test_world_zero_passes_on_the_confirmation_cuts_with_a_minted_registration(
        synthetic, tmp_path):
    f = _frame()
    p = tmp_path / "reg.json"
    p.write_text('{"data_pins": {}}')
    reg = G.load_registration(p, G.sha256_file(p))
    out = PL.build_world(f, 0, PL.CONF_CUTS, registration=reg)
    assert out["fwd_ret"].to_numpy().tobytes() == f["fwd_ret"].to_numpy().tobytes()


def test_a_verified_registration_cannot_be_constructed_directly():
    with pytest.raises(PermissionError):
        PL.VerifiedConfRegistration(path="/x", sha256="0" * 64, data={})


def test_the_plant_refuses_world_zero():
    with pytest.raises(PL.PlantRefused):
        PL.plant_feature(_frame(), 0, 0.02, 2)
