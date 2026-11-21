"""
================================================================================
Script:       test_canon_r3_world.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The scrambled world's guards: the shift is not the identity, it
              never moves a target between months, it is a pure function of
              (panel, k), world 0 is refused outside the gate, and the plant
              refuses world 0.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_r3_world.py
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

import canon_r3_paths as P  # noqa: E402
import canon_r3_synthetic as S  # noqa: E402
import canon_r3_world as W  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    st = S.snapshot_state()
    P.set_roots(repo=ROOT, snapshot=tmp_path, banked_panel=tmp_path / "none.parquet",
                derived=tmp_path / "d", scratch=tmp_path / "s",
                windows=tmp_path / "w.json")
    yield tmp_path
    S.restore_state(st)


def frame(n_names=20, months=(100, 101, 102), seed=5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for m in months:
        for i in range(n_names):
            recs.append({"permaticker": 1000 + i, "month": m,
                         "fwd_ret": float(rng.normal()), "fwd_stopped": False,
                         "in_universe": True, "beta": 1.0, "log_dv": 16.0,
                         "sector": "A", "PRC": float(rng.normal())})
    return pd.DataFrame(recs)


# ----------------------------------------------------------------- the scramble
def test_the_scramble_is_not_the_identity():
    f = frame()
    w = W.scramble_targets(f, 1)
    moved = (w["fwd_ret"].to_numpy() != f["fwd_ret"].to_numpy())
    # a cyclic shift of distinct values moves every row
    assert moved.all(), "the scramble left %d rows in place" % int((~moved).sum())


def test_the_scramble_preserves_each_months_multiset_exactly():
    f = frame()
    w = W.scramble_targets(f, 3)
    for m in f["month"].unique():
        a = np.sort(f.loc[f["month"] == m, "fwd_ret"].to_numpy())
        b = np.sort(w.loc[w["month"] == m, "fwd_ret"].to_numpy())
        assert a.tobytes() == b.tobytes(), ("month %s multiset changed" % m)


def test_the_scramble_never_moves_a_target_between_months():
    f = frame()
    w = W.scramble_targets(f, 7)
    for m in f["month"].unique():
        src = set(f.loc[f["month"] == m, "fwd_ret"].round(12).tolist())
        got = set(w.loc[w["month"] == m, "fwd_ret"].round(12).tolist())
        assert got <= src, "month %s took a target from another month" % m


def test_the_scramble_is_a_cyclic_shift_along_the_key_order():
    f = frame()
    keys = W.world_keys(f["permaticker"], 2)
    w = W.scramble_targets(f, 2)
    for m in f["month"].unique():
        g = f.loc[f["month"] == m]
        order = np.argsort(g["permaticker"].map(keys).to_numpy(), kind="stable")
        rows = g.index.to_numpy()[order]
        want = f.loc[np.roll(rows, -1), "fwd_ret"].to_numpy()
        got = w.loc[rows, "fwd_ret"].to_numpy()
        assert np.array_equal(want, got), m


def test_a_world_is_a_pure_function_of_panel_and_k():
    f = frame()
    a = W.scramble_targets(f, 9)["fwd_ret"].to_numpy()
    b = W.scramble_targets(f, 9)["fwd_ret"].to_numpy()
    assert a.tobytes() == b.tobytes()
    c = W.scramble_targets(f, 10)["fwd_ret"].to_numpy()
    assert a.tobytes() != c.tobytes(), "two worlds produced the same targets"


def test_world_keys_are_drawn_over_the_sorted_permaticker_list():
    pm = [1005, 1001, 1003]
    k = W.world_keys(pm, 4)
    assert list(k.index) == [1001, 1003, 1005]
    want = np.random.default_rng(W.WORLD_SEED_BASE + 4).random(3)
    assert np.array_equal(k.to_numpy(), want)


def test_world_keys_refuse_world_zero():
    with pytest.raises(W.WorldZeroRefused):
        W.world_keys([1, 2, 3], 0)


# ------------------------------------------------------------------- world zero
def test_build_world_refuses_world_zero_on_a_flag():
    f = frame()
    with pytest.raises(W.WorldZeroRefused):
        W.build_world(f, 0)


def test_build_world_refuses_a_forged_registration():
    f = frame()
    class Forged:
        path = "/nowhere"
        sha256 = "0" * 64
    with pytest.raises(W.WorldZeroRefused):
        W.build_world(f, 0, registration=Forged())


def test_a_verified_registration_cannot_be_constructed_directly():
    with pytest.raises(PermissionError):
        W.VerifiedRegistration(path="/x", sha256="0" * 64, data={})


def test_world_zero_is_returned_unchanged_through_a_minted_registration(tmp_path):
    f = frame()
    p = tmp_path / "reg.json"
    p.write_text('{"a": 1}')
    import canon_r3_gate as G
    reg = G.load_registration(p, G.sha256_file(p))
    out = W.build_world(f, 0, registration=reg)
    assert out["fwd_ret"].to_numpy().tobytes() == f["fwd_ret"].to_numpy().tobytes()


def test_world_zero_is_refused_once_the_registration_file_moves(tmp_path):
    f = frame()
    p = tmp_path / "reg.json"
    p.write_text('{"a": 1}')
    import canon_r3_gate as G
    reg = G.load_registration(p, G.sha256_file(p))
    p.write_text('{"a": 2}')
    with pytest.raises(W.WorldZeroRefused):
        W.build_world(f, 0, registration=reg)


# ---------------------------------------------------------------------- the plant
def test_the_plant_refuses_world_zero():
    f = frame()
    with pytest.raises(W.PlantRefused):
        W.plant_feature(f, 0, 0.02, 2)


def test_the_plant_replaces_the_named_feature_and_keeps_the_band():
    f = frame()
    out = W.plant_feature(f, 1, 0.08, 4)
    assert not np.array_equal(out["PRC"].to_numpy(), f["PRC"].to_numpy())
    v = out["PRC"].dropna().to_numpy()
    assert v.min() > -0.5 and v.max() <= 0.5 + 1e-12
    for m, g in out.groupby("month"):
        assert abs(float(g["PRC"].max()) - 0.5) < 1e-12


def test_a_bigger_rho_plants_a_stronger_tie_to_the_target():
    f = frame(n_names=200, months=(100, 101, 102, 103), seed=17)
    def corr(rho, j):
        o = W.plant_feature(f, 1, rho, j)
        return abs(float(np.corrcoef(o["PRC"].to_numpy(),
                                     o["fwd_ret"].to_numpy())[0, 1]))
    lo, hi = corr(0.005, 0), corr(0.40, 4)
    assert hi > lo, (lo, hi)


def test_the_plant_is_deterministic_in_k_and_rho_index():
    f = frame()
    a = W.plant_feature(f, 2, 0.02, 2)["PRC"].to_numpy()
    b = W.plant_feature(f, 2, 0.02, 2)["PRC"].to_numpy()
    assert a.tobytes() == b.tobytes()
    c = W.plant_feature(f, 2, 0.02, 3)["PRC"].to_numpy()
    assert a.tobytes() != c.tobytes()
