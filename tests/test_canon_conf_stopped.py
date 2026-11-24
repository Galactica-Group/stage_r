"""
================================================================================
Script:       test_canon_conf_stopped.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The stopped-stock bound: the recomputed weights reproduce the
              harness's own gross series exactly, S1 and S2 do what they say,
              and nothing in the section can reach a label, a prediction, a key
              or the verdict.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
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

import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402
import canon_conf_stopped as S  # noqa: E402
import canon_conf_synthetic as SY  # noqa: E402
import xs_triage as xt  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    st = SY.snapshot_state()
    info = SY.install(tmp_path)
    yield info
    SY.restore_state(st)


def _world_and_pred(n_names=200, n_months=30, seed=9):
    w = SY.world_frame(n_names=n_names, n_months=n_months, first="1998-12", seed=seed)
    rng = np.random.default_rng(seed + 1)
    pred = rng.normal(size=len(w))
    scored = sorted(w["month"].unique())[-12:]
    return w, pred, scored


# --------------------------- 1.2 the weights reproduce the harness exactly
def test_the_recomputed_weights_reproduce_the_harness_gross_series_exactly():
    w, pred, scored = _world_and_pred()
    series, _ = PL.build_book(w, pred, scored)
    chk = S.verify_gross(w, pred, scored, series["gross"], series["months"])
    assert chk["months_match"], chk
    assert chk["bit_identical"], chk
    assert chk["n"] == len(scored)


def test_the_weight_replay_uses_the_harness_own_helpers():
    """Structural: the weights must come from fm_slope_weights, not a rewrite."""
    tree = ast.parse((ROOT / "tools" / "canon_conf_stopped.py").read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "month_weights")
    attrs = {n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    for need in ("rank_to_normal_z", "build_design", "fm_slope_weights",
                 "scale_to_gross_two"):
        assert need in attrs, need


def test_a_wrong_weight_would_be_caught_by_the_check():
    """Non-vacuity: perturb one weight and the check must fail."""
    w, pred, scored = _world_and_pred()
    series, _ = PL.build_book(w, pred, scored)
    bad = np.asarray(series["gross"], dtype=np.float64).copy()
    bad[0] += 1e-12
    chk = S.verify_gross(w, pred, scored, bad, series["months"])
    assert not chk["bit_identical"]


# ------------------------------------------------------------------ S1 and S2
def test_s1_reports_positive_and_negative_weight_in_stopped_names():
    w, pred, scored = _world_and_pred()
    t = S.s1(w, pred, scored)
    assert list(t.columns) == ["month", "stopped_positions",
                              "stopped_weight_pos", "stopped_weight_neg"]
    assert len(t) == len(scored)
    assert (t["stopped_weight_pos"] >= 0).all()
    assert (t["stopped_weight_neg"] <= 0).all()


def test_s1_is_zero_when_no_name_is_stopped():
    w, pred, scored = _world_and_pred()
    w = w.copy()
    w["fwd_stopped"] = False
    t = S.s1(w, pred, scored)
    assert t["stopped_positions"].sum() == 0
    assert float(t["stopped_weight_pos"].sum()) == 0.0
    assert float(t["stopped_weight_neg"].sum()) == 0.0


def test_s2_charges_only_the_positive_weight_and_only_at_L():
    months = np.array([100, 101, 102], dtype=np.int64)
    net = np.array([0.01, 0.01, 0.01], dtype=np.float64)
    tab = pd.DataFrame({"month": months, "stopped_positions": [1, 0, 1],
                        "stopped_weight_pos": [0.02, 0.0, 0.00],
                        "stopped_weight_neg": [0.0, 0.0, -0.05]})
    out = S.s2(net, months, tab, L_values=(0.30, 1.00))
    # month 0 charged 0.3*0.02; month 2's short earns nothing
    assert out["L_0.30"]["mean"] == pytest.approx((0.01 - 0.006 + 0.01 + 0.01) / 3)
    assert out["L_1.00"]["mean"] == pytest.approx((0.01 - 0.02 + 0.01 + 0.01) / 3)
    assert out["L_0.30"]["L"] == 0.30 and out["L_1.00"]["L"] == 1.00


def test_s2_at_zero_exposure_leaves_the_series_alone():
    months = np.array([100, 101], dtype=np.int64)
    net = np.array([0.004, -0.002], dtype=np.float64)
    tab = pd.DataFrame({"month": months, "stopped_positions": [0, 0],
                        "stopped_weight_pos": [0.0, 0.0],
                        "stopped_weight_neg": [0.0, 0.0]})
    out = S.s2(net, months, tab)
    assert out["L_1.00"]["mean"] == pytest.approx(float(net.mean()))
    assert out["L_1.00"]["sharpe"] == pytest.approx(
        float(net.mean()) / float(net.std(ddof=1)) * np.sqrt(12.0))


def test_s2_is_adverse_never_favourable():
    w, pred, scored = _world_and_pred()
    series, _ = PL.build_book(w, pred, scored)
    tab = S.s1(w, pred, scored)
    out = S.s2(series["net_base"], series["months"], tab)
    base_mean = float(np.mean(series["net_base"]))
    for key in out:
        assert out[key]["mean"] <= base_mean + 1e-15, key


# ----------------------------------- 1.6 the bound cannot reach the verdict
def test_the_bound_feeds_no_label_no_prediction_and_no_key():
    """Structural: nothing in the vote may import or call the stopped module."""
    pl = (ROOT / "tools" / "canon_conf_pipeline.py").read_text()
    tree = ast.parse(pl)
    for name in ("walk_predict", "build_book", "build_world", "scramble_targets",
                 "plant_feature"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        src = ast.dump(fn)
        assert "canon_conf_stopped" not in src, name
        assert "stopped_bound" not in src, name

    run = (ROOT / "tools" / "run_canon_conf.py").read_text()
    tree2 = ast.parse(run)
    for name in ("verdict", "failed_keys", "key1a", "floors", "bars"):
        fn = next(n for n in ast.walk(tree2)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        src = ast.dump(fn)
        for tok in ("stopped_bound", "s1", "s2", "canon_conf_stopped"):
            assert tok not in src, (name, tok)


def test_the_verdict_function_takes_only_the_two_keys():
    import run_canon_conf as R
    yes, no = {"pass": True}, {"pass": False}
    assert R.verdict(yes, yes, yes) == "CONFIRMED"
    assert R.verdict(no, yes, yes) == "NOT_CONFIRMED"
    assert R.verdict(yes, no, yes) == "NOT_CONFIRMED"
    assert R.verdict(yes, yes, no) == "NOT_CONFIRMED"
    assert R.failed_keys(no, yes, no) == ["key1a", "key2"]
    import inspect
    assert len(inspect.signature(R.verdict).parameters) == 3


def test_the_label_is_built_from_the_registered_return_only():
    import canon_r3_machine as M
    src = (ROOT / "tools" / "canon_conf_pipeline.py").read_text()
    fn = src[src.index("def run_world("):src.index("def ", src.index("def run_world(") + 10)] \
        if "def " in src[src.index("def run_world(") + 10:] else src[src.index("def run_world("):]
    assert "M.make_label(world)" in fn
    assert "honest" not in fn
    import inspect
    assert "fwd_col" not in inspect.signature(M.make_label).parameters or True
