"""
================================================================================
Script:       test_canon_r3_env.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      Environment guards for the scikit-learn install: the live venv
              equals the tracked lock; the certified harness and both panel
              builders import neither scipy nor sklearn; every canon_r3 tool
              imports only the standard library, numpy, pandas, pyarrow, sklearn
              and Stage_R's own tools, and nothing from the factory tree.
Entry Point:  No — pytest
Dependencies: stdlib (ast, pathlib, subprocess, sys)
Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)
================================================================================

Falsifiable structural assertions: each fails on a constructed violation of the
property it names, and each is shown to fail against a mutant of its guard.
================================================================================
"""

from __future__ import annotations

import ast
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
LOCK = REPO / "requirements.lock"
VENV_PY = REPO / "venv" / "bin" / "python"

# The harness and the builders are certified code; they must stay on the
# dependency set they were certified with.
CERTIFIED_UNCHANGED = ["xs_triage.py", "build_panel_r1.py", "build_panel_r1b.py"]
FORBIDDEN_IN_CERTIFIED = {"scipy", "sklearn"}

# What a canon_r3 tool may import beyond the standard library.
CANON_ALLOWED_THIRD_PARTY = {"numpy", "pandas", "pyarrow", "sklearn"}
FACTORY_DIR_NAME = "Monte_Carlo_Strategies"


def _canon_tools() -> list[pathlib.Path]:
    return sorted((REPO / "tools").glob("canon_r3_*.py")) + \
           sorted((REPO / "tools").glob("run_canon_r3.py"))


def _own_tool_names() -> set[str]:
    return {p.stem for p in (REPO / "tools").glob("*.py")}


def imported_modules(src: str) -> tuple[set[str], bool]:
    """Top-level module names imported by `src`, and whether any is relative."""
    mods: set[str] = set()
    relative = False
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                relative = True
            elif node.module:
                mods.add(node.module.split(".")[0])
    return mods, relative


# ------------------------------------------------------------------ the lock
def test_the_lock_is_tracked_and_non_empty():
    assert LOCK.exists(), "requirements.lock must be tracked"
    lines = [l for l in LOCK.read_text().splitlines() if l.strip()]
    assert len(lines) >= 20
    assert all("==" in l for l in lines), "every line must pin a version"


def test_the_live_venv_equals_the_lock():
    out = subprocess.run([str(VENV_PY), "-m", "pip", "freeze"],
                         capture_output=True, text=True, check=True).stdout
    live = sorted(l.strip() for l in out.splitlines() if l.strip())
    locked = sorted(l.strip() for l in LOCK.read_text().splitlines() if l.strip())
    assert live == locked, (
        "the live venv and requirements.lock disagree; only in live: %s; only in lock: %s"
        % (sorted(set(live) - set(locked)), sorted(set(locked) - set(live))))


def test_the_lock_pins_the_six_new_packages_at_the_installed_versions():
    locked = {l.split("==")[0].lower(): l.split("==")[1]
              for l in LOCK.read_text().splitlines() if l.strip()}
    assert locked["scikit-learn"] == "1.9.1"
    assert locked["scipy"] == "1.18.1"
    assert locked["joblib"] == "1.6.0"
    assert locked["narwhals"] == "2.29.0"
    assert locked["cloudpickle"] == "3.1.3"
    assert locked["threadpoolctl"] == "3.8.0"


def test_the_lock_did_not_move_the_three_packages_the_harness_runs_on():
    locked = {l.split("==")[0].lower(): l.split("==")[1]
              for l in LOCK.read_text().splitlines() if l.strip()}
    assert locked["numpy"] == "2.5.3"
    assert locked["pandas"] == "3.0.5"
    assert locked["pyarrow"] == "25.0.1"


# ------------------------------------------- the certified code is untouched
@pytest.mark.parametrize("name", CERTIFIED_UNCHANGED)
def test_certified_code_imports_neither_scipy_nor_sklearn(name):
    mods, _ = imported_modules((REPO / "tools" / name).read_text(encoding="utf-8"))
    offenders = mods & FORBIDDEN_IN_CERTIFIED
    assert not offenders, "%s now imports %s" % (name, sorted(offenders))


@pytest.mark.parametrize("name", CERTIFIED_UNCHANGED)
def test_certified_code_names_no_scipy_or_sklearn_at_all(name):
    src = (REPO / "tools" / name).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            base = node
            while isinstance(base, ast.Attribute):
                base = base.value
            if isinstance(base, ast.Name) and base.id in FORBIDDEN_IN_CERTIFIED:
                raise AssertionError("%s reaches %s" % (name, base.id))


# ----------------------------------------------------- the canon_r3 tools
def test_there_is_at_least_one_canon_r3_tool():
    assert _canon_tools(), "no canon_r3 tool found; the guard would be vacuous"


def test_canon_tools_import_only_the_allowed_set():
    own = _own_tool_names()
    stdlib = set(sys.stdlib_module_names)
    for p in _canon_tools():
        mods, relative = imported_modules(p.read_text(encoding="utf-8"))
        assert not relative, "%s uses a relative import" % p.name
        offenders = {m for m in mods
                     if m not in stdlib
                     and m not in CANON_ALLOWED_THIRD_PARTY
                     and m not in own}
        assert not offenders, "%s imports %s" % (p.name, sorted(offenders))


def test_canon_tools_import_nothing_from_the_factory_tree():
    for p in _canon_tools():
        src = p.read_text(encoding="utf-8")
        assert FACTORY_DIR_NAME not in src, "%s names the factory tree" % p.name
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert FACTORY_DIR_NAME not in a.name
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("src.", "1_data", "2_research", "3_validation", "4_analysis"))


def test_canon_tools_record_the_environment_versions():
    """1.6: every canon_r3 artifact records the five library versions."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "canon_r3_tripwire", REPO / "tools" / "canon_r3_tripwire.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    env = mod.env_versions()
    assert set(env) == {"python", "numpy", "pandas", "pyarrow", "scikit-learn"}
    assert all(env[k] for k in env), env
