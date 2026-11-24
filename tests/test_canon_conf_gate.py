"""
================================================================================
Script:       test_canon_conf_gate.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The confirmation gate's five parts, and that the read takes the
              pool from the tracked file and nowhere else.
Entry Point:  No — pytest
Dependencies: stdlib, numpy, pandas; tools/canon_conf_*.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
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

import canon_conf_gate as G  # noqa: E402
import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402
import canon_conf_seal as SEAL  # noqa: E402
import canon_conf_synthetic as SY  # noqa: E402


@pytest.fixture(autouse=True)
def synthetic(tmp_path):
    st = SY.snapshot_state()
    info = SY.install(tmp_path)
    SY.write_tools_copy(tmp_path)
    SY.write_provenance(tmp_path)
    frame = SY.world_frame(n_names=60, n_months=6)
    SY.write_panel_and_features(tmp_path, frame)
    P.canon_dir().mkdir(parents=True, exist_ok=True)
    P.pool_v1().write_text(json.dumps({"n_worlds": 60, "worlds": []}))
    P.controls_v1().write_text("{}")
    yield info
    SY.restore_state(st)


def write_seal(tmp: Path, pins=None, name="registration.json"):
    p = P.canon_dir() / name
    body = {"look": "CANON-R3C",
            "data_pins": G.current_pins() if pins is None else pins}
    p.write_text(json.dumps(body, indent=1, sort_keys=True))
    return p, G.sha256_file(p)


def write_receipt(sha):
    r = P.controls_receipt()
    r.parent.mkdir(parents=True, exist_ok=True)
    r.write_text(json.dumps({"registration_sha": sha}))
    return r


# ------------------------------------------------------------- part 1, the seal
def test_load_registration_refuses_a_sha_mismatch(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    with pytest.raises(G.SealMismatch):
        G.load_registration(p, "0" * 64)
    assert isinstance(G.load_registration(p, sha), PL.VerifiedConfRegistration)


def test_load_registration_refuses_an_absent_file(synthetic):
    with pytest.raises(G.SealMismatch):
        G.load_registration(P.canon_dir() / "nope.json", "0" * 64)


def test_a_byte_flip_breaks_the_binding(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    p.write_text(p.read_text() + " ")
    with pytest.raises(G.SealMismatch):
        G.load_registration(p, sha)


# ------------------------------------------------------------- part 2, the pins
def test_the_pinned_set_is_the_registered_one(synthetic):
    names = set(G.pinned_paths())
    for n in ("conf_panel", "conf_feature_file", "conf_pool_v1",
              "conf_controls_v1", "requirements_lock", "r3_seal", "r3_result"):
        assert n in names, n
    for n in G.R3_TOOLS:
        assert n in names, n
    for n in G.CONF_TOOLS:
        assert n in names, n
    assert "xs_triage.py" in names
    assert "canon_conf_pipeline.py" in names
    assert "canon_conf_stopped.py" in names
    assert not any("spent" in n or "r3c_result" in n for n in names)


def test_matching_pins_pass_and_a_mismatch_is_refused(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    reg = G.load_registration(p, sha)
    assert G.verify_data_pins(reg)["pins_checked"] == len(G.current_pins())
    pins = G.current_pins()
    pins["conf_feature_file"] = "0" * 64
    p2, sha2 = write_seal(synthetic["tmp"], pins=pins, name="r2.json")
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(G.load_registration(p2, sha2))


def test_a_moved_artifact_is_refused_against_a_good_seal(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    reg = G.load_registration(p, sha)
    P.controls_v1().write_text('{"moved": true}')
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(reg)


def test_a_seal_that_pins_nothing_is_refused(synthetic):
    p, sha = write_seal(synthetic["tmp"], pins={})
    with pytest.raises(G.DataPinMismatch):
        G.verify_data_pins(G.load_registration(p, sha))


# ---------------------------------------------------------- part 3, the receipt
def test_the_observed_stage_refuses_without_a_receipt(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    with pytest.raises(G.ReceiptMissing):
        G.verify_controls_receipt(G.load_registration(p, sha))


def test_a_receipt_for_another_seal_is_refused(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    write_receipt("f" * 64)
    with pytest.raises(G.ReceiptMissing):
        G.verify_controls_receipt(G.load_registration(p, sha))


def test_a_receipt_written_against_this_seal_passes(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    write_receipt(sha)
    assert G.verify_controls_receipt(G.load_registration(p, sha))[
        "registration_sha"] == sha


# ------------------------------------------------------------- part 4, the lock
def test_the_single_run_lock_refuses_a_second_entry(synthetic):
    with G.single_run_lock():
        with pytest.raises(G.LockHeld):
            with G.single_run_lock():
                pass


def test_the_lock_is_released_on_exit_and_on_a_raise(synthetic):
    with G.single_run_lock():
        assert P.lock_file().exists()
    assert not P.lock_file().exists()
    with pytest.raises(ValueError):
        with G.single_run_lock():
            raise ValueError("boom")
    assert not P.lock_file().exists()


# ------------------------------------------------------ part 5, the spent marker
def test_the_spent_marker_refuses_a_second_read(synthetic):
    G.refuse_if_spent()
    G.write_spent_marker("a" * 64, "deadbeef")
    with pytest.raises(G.AlreadySpent):
        G.refuse_if_spent()


def test_the_spent_marker_records_the_seal_the_time_and_the_head(synthetic):
    p = G.write_spent_marker("b" * 64, "cafebabe")
    body = json.loads(p.read_text())
    assert body["look"] == "CANON-R3C"
    assert body["registration_sha256"] == "b" * 64
    assert body["head"] == "cafebabe"
    assert body["utc"].endswith("Z") and len(body["utc"]) == 20
    assert not p.with_suffix(".json.tmp").exists()


# --------------------------------------------------------------- the whole chain
def test_the_chain_opens_only_with_seal_pins_and_receipt(synthetic):
    p, sha = write_seal(synthetic["tmp"])
    with pytest.raises(G.ReceiptMissing):
        G.open_the_gate(p, sha)
    write_receipt(sha)
    opened = G.open_the_gate(p, sha)
    assert isinstance(opened["registration"], PL.VerifiedConfRegistration)
    assert opened["registration_sha"] == sha


def test_the_runner_orders_the_five_parts_correctly():
    src = (ROOT / "tools" / "run_canon_conf.py").read_text()
    fn = src[src.index("def stage_observed"):src.index("def append_ledger_row")]
    for a, b in (("refuse_if_spent", "open_the_gate"),
                 ("open_the_gate", "single_run_lock"),
                 ("single_run_lock", "write_spent_marker"),
                 ("write_spent_marker", "run_world")):
        assert fn.index(a) < fn.index(b), (a, b)
    assert fn.count("refuse_if_spent") == 2


def test_the_read_takes_the_pool_from_the_tracked_file_and_nowhere_else():
    src = (ROOT / "tools" / "run_canon_conf.py").read_text()
    fn = src[src.index("def stage_observed"):src.index("def append_ledger_row")]
    assert "P.pool_v1()" in fn
    assert "pool_manifest" not in fn
    assert "pool_dir" not in fn


def test_the_controls_stage_runs_t3_first_and_refuses_on_a_mismatch():
    src = (ROOT / "tools" / "run_canon_conf.py").read_text()
    fn = src[src.index("def stage_controls"):src.index("def _md_report")]
    assert fn.index("TW.t3()") < fn.index("_load()")
    assert "refuses" in fn and "AssertionError" in fn


def test_the_controls_stage_never_asks_for_world_zero():
    src = (ROOT / "tools" / "run_canon_conf.py").read_text()
    tree = ast.parse(src)
    for name in ("run_pool", "run_plants", "determinism"):
        fn = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == name)
        for n in ast.walk(fn):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr == "run_world":
                for a in n.args[1:2]:
                    assert not (isinstance(a, ast.Constant) and a.value == 0), name


# ------------------------------------------------------------- the seal builder
def test_the_seal_builder_is_the_only_writer_of_a_registration(synthetic):
    out = P.canon_dir() / "registration_v1_draft.json"
    sha = SEAL.build(out, {"note": "test"}, draft=True, pool=None)
    assert out.exists() and sha == G.sha256_file(out)
    body = json.loads(out.read_text())
    assert body["look"] == "CANON-R3C"
    assert body["status"] == "DRAFT, UNSEALED"
    assert body["design"]["verdict"]["key3"].startswith("STRUCK")
    assert isinstance(body["data_pins"], dict) and body["data_pins"]
    # a pool too small for a third-largest t states no bar rather than inventing one
    assert body["bars"]["available"] is False


def test_the_seal_builder_states_the_bars_once_the_pool_is_large_enough(synthetic):
    worlds = [{"k": k, "gross": {"mean": 1e-5 * k, "t": 0.1 * k}}
              for k in range(1, 61)]
    out = P.canon_dir() / "registration_v1_draft.json"
    SEAL.build(out, {"note": "test"}, draft=True, pool={"worlds": worlds})
    bars = json.loads(out.read_text())["bars"]
    assert bars["n_worlds"] == 60
    assert bars["key1b_third_largest_pool_gross_t"] == pytest.approx(0.1 * 58)
    assert bars["key2_sharpe_bar"] == 0.5 and bars["level"] == 0.05
    assert bars["available"] is True


def test_the_seals_bars_and_the_runners_bars_agree_on_the_same_pool(synthetic):
    """The seal STATES the bars; the read RECOMPUTES them. They must agree."""
    import numpy as np
    import run_canon_conf as R
    rng = np.random.default_rng(3)
    worlds = [{"k": k,
               "gross": {"mean": float(rng.normal(0, 6e-4)),
                         "t": float(rng.normal(0, 1))}}
              for k in range(1, 61)]
    a = SEAL._bars(worlds)
    b = R.bars(worlds)
    assert a["key1a_centre_pool_median_gross_mean"] == \
        b["key1a_centre_pool_median_gross_mean"]
    assert a["key1b_third_largest_pool_gross_t"] == \
        b["key1b_third_largest_pool_gross_t"]
    assert a["n_worlds"] == b["n_worlds"] == 60


def test_no_other_tool_writes_a_registration():
    import re
    for p in sorted((ROOT / "tools").glob("canon_conf_*.py")):
        if p.name == "canon_conf_seal.py":
            continue
        src = p.read_text()
        assert "registration.json" not in src or "sealed_registration" in src, p.name
        assert not re.search(r"registration_v1_draft\.json[\"']\s*\)?\s*\.write_text",
                             src), p.name
