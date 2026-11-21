"""
================================================================================
Script:       test_canon_r3_machine.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The machine's guards: the walk trains through C-2 and predicts
                from C, the feature matrix is the 25 features and nothing else,
                the label never enters it, and the two keys of the verdict do
                what they say.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas, sklearn; tools/canon_r3_*.py
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_r3_features as F  # noqa: E402
import canon_r3_machine as M  # noqa: E402
import canon_r3_paths as P  # noqa: E402
import canon_r3_synthetic as S  # noqa: E402
import canon_r3_world as W  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    st = S.snapshot_state()
    P.set_roots(repo=ROOT, snapshot=tmp_path, banked_panel=tmp_path / "none.parquet",
                derived=tmp_path / "d", scratch=tmp_path / "s",
                windows=tmp_path / "w.json")
    (tmp_path / "d").mkdir(exist_ok=True)
    (tmp_path / "s").mkdir(exist_ok=True)
    yield tmp_path
    S.restore_state(st)


def synth_world(n_names=300, n_months=60, first="1998-12", seed=4) -> pd.DataFrame:
    """A frame shaped like the feature matrix, with every column the walk needs."""
    rng = np.random.default_rng(seed)
    o = pd.Period(first, freq="M").ordinal
    recs = []
    for j in range(n_months):
        for i in range(n_names):
            r = {"permaticker": 1000 + i, "month": o + j, "in_universe": True,
                 "fwd_ret": float(rng.normal(0, 0.08)), "fwd_stopped": False,
                 "beta": 1.0 + float(rng.normal(0, 0.2)),
                 "log_dv": 16.0 + float(rng.normal()),
                 "sector": "ABCD"[i % 4]}
            for n in F.FEATURE_NAMES:
                r[n] = float(rng.uniform(-0.5, 0.5))
            recs.append(r)
    return pd.DataFrame(recs)


class SpyModel:
    """Records the shape of every X it is fitted on; predicts noise."""
    seen = []

    def __init__(self, random_state=None, **kw):
        self.rs = random_state

    def fit(self, X, y):
        SpyModel.seen.append((X.shape, int(np.isfinite(y).sum())))
        self._n = X.shape[1]
        return self

    def predict(self, X):
        return np.random.default_rng(self.rs).normal(size=X.shape[0])


# ------------------------------------------------------------------- the walk
def test_the_walk_trains_through_c_minus_two_and_scores_from_c(monkeypatch):
    w = synth_world()
    monkeypatch.setattr(M, "HistGradientBoostingRegressor", SpyModel)
    monkeypatch.setattr(M, "SEEDS", (1,))
    monkeypatch.setattr(M, "FIRST_CUT", "2001-12")
    monkeypatch.setattr(M, "LAST_CUT", "2003-12")
    monkeypatch.setattr(M, "N_CUTS", 3)
    y = np.asarray(np.random.default_rng(0).normal(size=len(w)))
    _, diags = M.walk_predict(w, y)
    live = [d for d in diags if not d["skipped"]]
    assert live, diags
    for d in live:
        C = pd.Period(d["cut"], freq="M")
        assert d["train_last"] == str(C - 2), (d["cut"], d["train_last"])
        assert d["score_first"] == str(C), (d["cut"], d["score_first"])
        assert d["score_last"] == str(C + 11), (d["cut"], d["score_last"])
        assert d["train_first"] == F.FIRST_FORMATION_MONTH


def test_the_eleven_cuts_are_the_registered_ones():
    cuts = [str(pd.Period(ordinal=c, freq="M")) for c in M.cut_months()]
    assert len(cuts) == 11
    assert cuts[0] == "2001-12" and cuts[-1] == "2011-12"
    assert cuts == ["%d-12" % y for y in range(2001, 2012)]


def test_the_scored_window_is_one_hundred_and_thirty_two_months():
    sm = M.scored_months()
    assert len(sm) == M.N_SCORED_MONTHS == 132
    assert str(pd.Period(ordinal=sm[0], freq="M")) == "2001-12"
    assert str(pd.Period(ordinal=sm[-1], freq="M")) == "2012-11"


def test_every_scored_month_is_covered_by_exactly_one_cut():
    covered = {}
    for C in M.cut_months():
        for m in range(C, C + 12):
            covered[m] = covered.get(m, 0) + 1
    for m in M.scored_months():
        assert covered.get(m) == 1, m
    assert set(covered) == set(M.scored_months())


# ------------------------------------------------------- the feature matrix
def test_the_feature_matrix_is_the_twenty_five_features_and_nothing_else(monkeypatch):
    w = synth_world(n_names=100, n_months=40)
    SpyModel.seen = []
    monkeypatch.setattr(M, "HistGradientBoostingRegressor", SpyModel)
    monkeypatch.setattr(M, "SEEDS", (1,))
    monkeypatch.setattr(M, "FIRST_CUT", "2001-12")
    monkeypatch.setattr(M, "N_CUTS", 1)
    y = np.asarray(np.random.default_rng(0).normal(size=len(w)))
    M.walk_predict(w, y)
    assert SpyModel.seen, "no fit was observed"
    for shape, _ in SpyModel.seen:
        assert shape[1] == 25, shape


def test_neither_the_label_nor_the_forward_return_is_a_feature():
    for bad in ("fwd_ret", "fwd_stopped", "label", "y", "ret_m", "sector"):
        assert bad not in F.FEATURE_NAMES


def test_the_label_is_a_target_only_and_is_not_among_the_columns_fitted(monkeypatch):
    w = synth_world(n_names=100, n_months=40)
    w["label"] = np.random.default_rng(1).normal(size=len(w))
    SpyModel.seen = []
    monkeypatch.setattr(M, "HistGradientBoostingRegressor", SpyModel)
    monkeypatch.setattr(M, "SEEDS", (1,))
    monkeypatch.setattr(M, "FIRST_CUT", "2001-12")
    monkeypatch.setattr(M, "N_CUTS", 1)
    M.walk_predict(w, w["label"].to_numpy())
    for shape, _ in SpyModel.seen:
        assert shape[1] == len(F.FEATURE_NAMES) == 25, shape


# ------------------------------------------------------------------- the label
def test_the_label_is_a_within_month_normal_score_of_the_residual():
    w = synth_world(n_names=120, n_months=6)
    y = M.make_label(w)
    tmp = pd.DataFrame({"m": w["month"].to_numpy(), "y": y})
    for m, g in tmp.groupby("m"):
        v = g["y"].dropna().to_numpy()
        if v.size >= 50:
            assert abs(float(v.mean())) < 1e-9, m
            assert abs(float(v.max()) + float(v.min())) < 1e-9, m


def test_a_month_too_small_for_a_residual_gets_no_label():
    w = synth_world(n_names=10, n_months=3)
    y = M.make_label(w)
    assert not np.isfinite(y).any(), "a 10-name month should not be labelled"


# ---------------------------------------------------------------- the verdict
def test_key1a_tests_against_the_null_centre_not_zero():
    rng = np.random.default_rng(2)
    g = rng.normal(0.002, 0.008, 132)
    a = M.key1a(g, 0.0, R=400, seed=1)
    b = M.key1a(g, 0.004, R=400, seed=1)
    assert a["p"] < b["p"], (a["p"], b["p"])
    assert a["null_centre"] == 0.0 and b["null_centre"] == 0.004


def test_key1b_counts_exceedances_and_allows_at_most_two():
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert M.key1b(5.0, ts)["pass"] is True
    assert M.key1b(5.0, ts)["exceedances"] == 0
    assert M.key1b(2.0, ts)["exceedances"] == 3
    assert M.key1b(2.0, ts)["pass"] is False
    r = M.key1b(3.0, ts)
    assert r["exceedances"] == 2 and r["pass"] is True
    assert abs(r["p"] - 3.0 / 6.0) < 1e-12


def test_key2_is_the_net_base_sharpe_bar():
    assert M.key2(0.5)["pass"] is True
    assert M.key2(0.4999)["pass"] is False
    assert M.key2(float("nan"))["pass"] is False
    assert M.KEY2_SHARPE_BAR == 0.5


def test_the_verdict_ladder():
    yes, no = {"pass": True}, {"pass": False}
    assert M.verdict(no, yes, yes) == "NOTHING"
    assert M.verdict(yes, no, yes) == "NOTHING"
    assert M.verdict(yes, yes, no) == "SIGNIFICANT_IMMATERIAL"
    assert M.verdict(yes, yes, yes) == "SURVIVOR"


def test_the_model_specification_is_read_twos():
    assert M.MODEL_KW["max_iter"] == 300
    assert M.MODEL_KW["learning_rate"] == 0.05
    assert M.MODEL_KW["max_leaf_nodes"] == 15
    assert M.MODEL_KW["min_samples_leaf"] == 200
    assert M.MODEL_KW["l2_regularization"] == 10.0
    assert M.MODEL_KW["max_features"] == 0.5
    assert M.MODEL_KW["max_bins"] == 255
    assert M.MODEL_KW["early_stopping"] is False
    assert M.MODEL_KW["loss"] == "squared_error"
    assert M.SEEDS == (20261121, 20261122, 20261123)
    assert M.R_VOTING == 10_000 and M.MEAN_BLOCK == 3.0
    assert M.SEED_VOTING == 20261121 and M.LEVEL == 0.05


def test_the_analytic_power_sharpe_is_the_textbook_value():
    got = M.analytic_sharpe_for_power(132)
    want = (1.6448536269514722 + 0.8416212335729143) / np.sqrt(132) * np.sqrt(12)
    assert abs(got - want) < 1e-12
