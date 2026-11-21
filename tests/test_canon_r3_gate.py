"""
================================================================================
Script:       test_canon_r3_gate.py
Version:      1.1.0
Phase:        Stage R — Canon Read 3
Purpose:      The gate's guards: the seal sha check, the data pins, the controls
              receipt, the single-run lock, and that world 0 is reachable only
              through the whole chain.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_r3_gate.py
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import canon_r3_gate as G  # noqa: E402
import canon_r3_paths as P  # noqa: E402
import canon_r3_synthetic as S  # noqa: E402
import canon_r3_world as W  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    """Pins are computed over a temporary tree; the real artifacts are unreachable."""
    st = S.snapshot_state()
    fake_repo = tmp_path / "repo"
    (fake_repo / "tools").mkdir(parents=True)
    for n in G.PINNED_CODE_NAMES:
        (fake_repo / "tools" / n).write_text("# %s\n" % n)
    (tmp_path / "derived").mkdir()
    (tmp_path / "scratch").mkdir()
    (tmp_path / "panel.parquet").write_text("panel")
    (tmp_path / "derived" / "features_20261004p_halfA_v2.parquet").write_text("feat")
    (tmp_path / "scratch" / "pool").mkdir()
    (tmp_path / "scratch" / "pool" / "pool_manifest.json").write_text("{}")
    (fake_repo / "requirements.lock").write_text("numpy==2.5.3\n")
    (fake_repo / "registrations" / "canon_r3").mkdir(parents=True)
    (fake_repo / "registrations" / "canon_r3" / "pool_v1.json").write_text('{"n": 60}')
    (fake_repo / "registrations" / "canon_r3" / "controls_v2.json").write_text("{}")
    P.set_roots(repo=fake_repo, snapshot=tmp_path,
                banked_panel=tmp_path / "panel.parquet",
                derived=tmp_path / "derived", scratch=tmp_path / "scratch",
                windows=tmp_path / "w.json")
    yield tmp_path
    S.restore_state(st)


def write_seal(tmp: Path, pins: dict | None = None, name="reg.json") -> tuple[Path, str]:
    p = tmp / name
    body = {"look": "canon read 3", "data_pins": pins if pins is not None
            else G.current_pins()}
    p.write_text(json.dumps(body, indent=1, sort_keys=True))
    return p, G.sha256_file(p)


def write_receipt(sha: str) -> Path:
    r = P.controls_receipt()
    r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text(json.dumps({"registration_sha": sha}))
    return r


# -------------------------------------------------------------- the seal check
def test_load_registration_refuses_a_sha_mismatch(synthetic):
    p, sha = write_seal(synthetic)
    with pytest.raises(G.SealMismatch):
        G.load_registration(p, "0" * 64)
    reg = G.load_registration(p, sha)
    assert isinstance(reg, W.VerifiedRegistration) and reg.sha256 == sha


def test_load_registration_refuses_an_absent_file(synthetic):
    with pytest.raises(G.SealMismatch):
        G.load_registration(synthetic / "nope.json", "0" * 64)


def test_a_byte_flip_in_the_seal_breaks_the_binding(synthetic):
    p, sha = write_seal(synthetic)
    p.write_text(p.read_text() + " ")
    with pytest.raises(G.SealMismatch):
        G.load_registration(p, sha)


# -------------------------------------------------------------- the data pins
def test_the_pinned_set_covers_every_canon_tool_and_the_harness(synthetic):
    names = set(G.pinned_paths())
    assert "xs_triage.py" in names
    assert {"banked_panel", "feature_file", "pool_v1", "controls_v2",
            "requirements_lock"} <= names
    assert "pool_manifest" not in names, \
        "the scratch manifest must not be a pin: the scratch volume is not tracked"
    for n in ("canon_r3_features.py", "canon_r3_world.py", "canon_r3_machine.py",
              "canon_r3_gate.py", "canon_r3_paths.py", "canon_r3_tripwire.py",
              "canon_r3_mutants.py", "canon_r3_rehearsal.py",
              "run_canon_r3.py"):
        assert n in names, n


def test_matching_data_pins_pass(synthetic):
    p, sha = write_seal(synthetic)
    reg = G.load_registration(p, sha)
    res = G.verify_data_pins(reg)
    assert res["pins_checked"] == len(G.current_pins())


def test_a_data_pin_mismatch_is_refused(synthetic):
    pins = G.current_pins()
    pins["feature_file"] = "0" * 64
    p, sha = write_seal(synthetic, pins=pins)
    reg = G.load_registration(p, sha)
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(reg)


def test_a_moved_artifact_is_refused_against_a_good_seal(synthetic):
    p, sha = write_seal(synthetic)
    reg = G.load_registration(p, sha)
    (synthetic / "derived" / "features_20261004p_halfA_v2.parquet").write_text("x")
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(reg)


def test_a_seal_that_pins_nothing_is_refused(synthetic):
    p, sha = write_seal(synthetic, pins={})
    reg = G.load_registration(p, sha)
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(reg)


def test_a_pin_naming_an_unpinned_artifact_is_refused(synthetic):
    pins = G.current_pins()
    pins["something_else"] = "0" * 64
    p, sha = write_seal(synthetic, pins=pins)
    reg = G.load_registration(p, sha)
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(reg)


# ---------------------------------------------------------------- the receipt
def test_the_observed_stage_refuses_without_a_receipt(synthetic):
    p, sha = write_seal(synthetic)
    reg = G.load_registration(p, sha)
    with pytest.raises(G.ReceiptMissing):
        G.verify_controls_receipt(reg)


def test_the_observed_stage_refuses_a_receipt_for_another_seal(synthetic):
    p, sha = write_seal(synthetic)
    reg = G.load_registration(p, sha)
    write_receipt("f" * 64)
    with pytest.raises(G.ReceiptMissing):
        G.verify_controls_receipt(reg)


def test_a_receipt_written_against_this_seal_passes(synthetic):
    p, sha = write_seal(synthetic)
    reg = G.load_registration(p, sha)
    write_receipt(sha)
    assert G.verify_controls_receipt(reg)["registration_sha"] == sha


# ------------------------------------------------------------------- the lock
def test_the_single_run_lock_refuses_a_second_entry(synthetic):
    with G.single_run_lock():
        with pytest.raises(G.LockHeld):
            with G.single_run_lock():
                pass


def test_the_lock_is_released_on_exit(synthetic):
    with G.single_run_lock():
        assert P.lock_file().exists()
    assert not P.lock_file().exists()
    with G.single_run_lock():
        pass


def test_the_lock_is_released_even_when_the_body_raises(synthetic):
    with pytest.raises(ValueError):
        with G.single_run_lock():
            raise ValueError("boom")
    assert not P.lock_file().exists()


# --------------------------------------------------------------- the whole chain
def test_the_whole_chain_opens_only_with_seal_pins_and_receipt(synthetic):
    p, sha = write_seal(synthetic)
    with pytest.raises(G.ReceiptMissing):
        G.open_the_gate(p, sha)
    write_receipt(sha)
    opened = G.open_the_gate(p, sha)
    assert isinstance(opened["registration"], W.VerifiedRegistration)
    assert opened["registration_sha"] == sha


def test_world_zero_needs_the_chain_not_a_flag(synthetic):
    f = pd.DataFrame({"permaticker": [1, 2, 3], "month": [1, 1, 1],
                      "fwd_ret": [0.1, 0.2, 0.3], "fwd_stopped": False})
    with pytest.raises(W.WorldZeroRefused):
        W.build_world(f, 0)
    p, sha = write_seal(synthetic)
    write_receipt(sha)
    reg = G.open_the_gate(p, sha)["registration"]
    out = W.build_world(f, 0, registration=reg)
    assert out["fwd_ret"].tolist() == [0.1, 0.2, 0.3]


def test_the_runner_takes_the_lock_and_opens_the_gate_in_the_observed_stage():
    """Structural: --stage observed must call open_the_gate and single_run_lock."""
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "stage_observed")
    names = {n.func.attr for n in ast.walk(fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "open_the_gate" in names
    assert "single_run_lock" in names


def test_the_controls_stage_never_asks_for_world_zero():
    """Structural: no call in the controls path passes a literal 0 as k."""
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    tree = ast.parse(src)
    for fname in ("run_pool", "run_plants", "determinism"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == fname)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("run_world", "build_world"):
                for a in n.args[1:2]:
                    assert not (isinstance(a, ast.Constant) and a.value == 0), fname


# ----------------------------------------------------------- the spent marker
def test_the_spent_marker_refuses_a_second_read(synthetic):
    G.refuse_if_spent()                        # absent: no refusal
    G.write_spent_marker("a" * 64, "deadbeef")
    with pytest.raises(G.AlreadySpent):
        G.refuse_if_spent()


def test_the_spent_marker_records_the_seal_the_time_and_the_head(synthetic):
    p = G.write_spent_marker("b" * 64, "cafebabe")
    body = json.loads(p.read_text())
    assert body["registration_sha256"] == "b" * 64
    assert body["head"] == "cafebabe"
    assert body["look"] == "CANON-R3"
    assert body["utc"].endswith("Z") and len(body["utc"]) == 20


def test_the_spent_marker_leaves_no_temporary_behind(synthetic):
    p = G.write_spent_marker("c" * 64, "x")
    assert p.exists()
    assert not p.with_suffix(".json.tmp").exists()


def test_the_runner_writes_the_spent_marker_after_the_gate_and_before_world_zero():
    """Structural: order matters, so assert it in the source."""
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "stage_observed")
    seq = [n.func.attr for n in ast.walk(fn)
           if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    order = [s for s in seq if s in ("refuse_if_spent", "open_the_gate",
                                     "single_run_lock", "write_spent_marker",
                                     "run_world")]
    assert order.index("open_the_gate") < order.index("write_spent_marker")
    assert order.index("single_run_lock") < order.index("write_spent_marker")
    assert order.index("write_spent_marker") < order.index("run_world")
    assert order[0] == "refuse_if_spent"


def test_the_runner_refuses_if_spent_before_it_opens_the_gate():
    src = (ROOT / "tools" / "run_canon_r3.py").read_text()
    fn = src[src.index("def stage_observed"):src.index("def append_ledger_row")]
    assert fn.index("refuse_if_spent") < fn.index("open_the_gate")
    assert fn.count("refuse_if_spent") == 2, \
        "refuse_if_spent must be checked once before the lock and once inside it"
