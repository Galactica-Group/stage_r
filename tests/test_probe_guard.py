"""
================================================================================
Script:       test_probe_guard.py
Version:      1.1.0
Phase:        Stage R — External Data Evaluation
Purpose:      Static guard on tools/sharadar_probe.py and
              tools/sharadar_snapshot_diff.py: dependency containment,
              factory-path isolation, API-key hygiene, and — for the probe — the
              returned-date-range assertion that stops silent partial snapshots.
Entry Point:  No — pytest
Dependencies: stdlib (ast, importlib, pathlib, sys); pandas via the probe module
Last Sprint:  Sprint R0-B-0 — Snapshot Diff Tool; probe v1.1 (2026-09-17)
================================================================================

These are falsifiable structural assertions, not smoke tests: each fails on a
constructed violation of the property it names.
================================================================================
"""

import ast
import importlib.util
import pathlib
import sys

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"
DIFF = REPO / "tools" / "sharadar_snapshot_diff.py"
ALLOWED_THIRD_PARTY = {"requests", "pandas", "pyarrow"}
# The diff tool reads snapshots off disk; it has no business reaching a network.
DIFF_ALLOWED_THIRD_PARTY = {"pandas", "pyarrow"}
FACTORY_DIR_NAME = "Monte_Carlo_Strategies"


def _tree():
    return ast.parse(PROBE.read_text(encoding="utf-8"))


def _top_level_imports(tree):
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                mods.add(node.module.split(".")[0])
            elif node.level:
                mods.add("__RELATIVE__")
    return mods


def test_imports_are_stdlib_or_allowed_third_party():
    mods = _top_level_imports(_tree())
    assert "__RELATIVE__" not in mods, "relative import found; probe must be self-contained"
    stdlib = set(sys.stdlib_module_names)
    offenders = {m for m in mods if m not in stdlib and m not in ALLOWED_THIRD_PARTY}
    assert not offenders, "disallowed imports: %s" % sorted(offenders)


def test_no_string_literal_contains_the_factory_path():
    """The probe must never name the factory tree; the env path is an argument."""
    offenders = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if FACTORY_DIR_NAME in node.value:
                offenders.append(node.value)
    assert not offenders, "factory path literal(s) in probe: %s" % offenders


def test_env_path_is_a_required_argument_not_a_default():
    """--env-path must be required, so no factory location can be baked in."""
    src = PROBE.read_text(encoding="utf-8")
    assert '"--env-path", required=True' in src, "--env-path must be required"


def test_key_is_read_from_a_file_path_and_never_logged():
    tree = _tree()
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "read_api_key" in fns, "key must be read by a dedicated function"
    reader = fns["read_api_key"]
    # the reader opens a path and returns a parsed value; it must not print or log
    called = {
        n.func.id for n in ast.walk(reader)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "open" in called, "read_api_key must read from a file path"
    assert "print" not in called, "read_api_key must not print the key"
    # the logger must redact before writing
    log = fns["log"] if "log" in fns else None
    assert log is not None, "probe must have a log() function"
    redacts = any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "redact"
        for n in ast.walk(log)
    )
    assert redacts, "log() must pass its message through redact()"


def test_redact_replaces_the_live_key_value():
    """redact() must scrub the literal key, not only the query-string pattern."""
    src = PROBE.read_text(encoding="utf-8")
    assert "self.api_key, \"<REDACTED>\"" in src.replace("'", '"'), \
        "redact() must replace the api_key value itself"


def test_all_writes_stay_inside_this_repo():
    """Every output path is composed from REPO_ROOT, which is derived from __file__."""
    src = PROBE.read_text(encoding="utf-8")
    assert "REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))" in src
    for marker in ["os.path.join(REPO_ROOT, \"data\"", "os.path.join(REPO_ROOT, \"logs\""]:
        assert marker in src, "expected repo-rooted output path: %s" % marker


# ----------------------------------------------------------------------------
# The same containment rules, applied to the snapshot diff tool.
# ----------------------------------------------------------------------------

def _diff_tree():
    return ast.parse(DIFF.read_text(encoding="utf-8"))


def test_diff_imports_are_stdlib_or_allowed_and_exclude_requests():
    mods = _top_level_imports(_diff_tree())
    assert "__RELATIVE__" not in mods, "relative import found; the tool must be self-contained"
    stdlib = set(sys.stdlib_module_names)
    offenders = {m for m in mods if m not in stdlib and m not in DIFF_ALLOWED_THIRD_PARTY}
    assert not offenders, "disallowed imports: %s" % sorted(offenders)
    assert "requests" not in mods, "the diff tool must make no network request"


def test_diff_names_no_factory_path():
    offenders = [n.value for n in ast.walk(_diff_tree())
                 if isinstance(n, ast.Constant) and isinstance(n.value, str)
                 and FACTORY_DIR_NAME in n.value]
    assert not offenders, "factory path literal(s) in the diff tool: %s" % offenders


def test_diff_handles_no_api_key_at_all():
    """The diff tool never touches credentials: nothing to leak, nothing to redact."""
    src = DIFF.read_text(encoding="utf-8").lower()
    for token in ("api_key", "sharadar_api", "--env-path", "authorization", "bearer"):
        assert token not in src, "diff tool references credentials: %s" % token


def test_diff_writes_are_rooted_in_this_repo_by_default():
    src = DIFF.read_text(encoding="utf-8")
    assert 'REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))' in src
    assert 'ap.add_argument("--root", default=REPO_ROOT)' in src
    for marker in ['os.path.join(root, "data", "sharadar")', 'os.path.join(root, "reports")']:
        assert marker in src, "expected repo-rooted output path: %s" % marker


# ----------------------------------------------------------------------------
# Probe v1.1: the returned span must cover the requested window.
# ----------------------------------------------------------------------------

def _probe_module():
    spec = importlib.util.spec_from_file_location("sharadar_probe", PROBE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def probe_mod():
    return _probe_module()


def test_check_date_range_catches_the_vendor_year_clip(probe_mod):
    """The R0-A finding: a five-year request answered with about one year, HTTP 200."""
    out = probe_mod.check_date_range(
        {"from": "2021-09-15", "to": "2026-09-12"},
        pd.DataFrame({"date": ["2025-09-15", "2026-09-11"]}))
    assert out["status"] == "SHORTFALL"
    assert out["requested"] == ["2021-09-15", "2026-09-12"]
    assert out["returned"] == ["2025-09-15", "2026-09-11"]
    assert out["gaps_days"]["start_days_short"] == 1461


def test_check_date_range_passes_an_aligned_window_with_a_weekend_edge(probe_mod):
    out = probe_mod.check_date_range(
        {"from": "2021-09-15", "to": "2022-09-14"},
        pd.DataFrame({"date": ["2021-09-15", "2022-09-13"]}))
    assert out["status"] == "ok"


def test_check_date_range_is_silent_when_no_window_was_requested(probe_mod):
    out = probe_mod.check_date_range({"limit": 5}, pd.DataFrame({"date": ["2026-01-01"]}))
    assert out["status"] == "not_applicable"


def test_check_date_range_flags_a_window_answered_with_no_rows(probe_mod):
    out = probe_mod.check_date_range(
        {"from": "2024-01-01", "to": "2024-12-31"}, pd.DataFrame({"date": []}))
    assert out["status"] == "SHORTFALL"
    assert out["returned"] == [None, None]


def test_check_date_range_tolerance_boundary_is_load_bearing(probe_mod):
    """Exactly at tolerance passes; one day beyond it does not."""
    tol = probe_mod.MAX_EDGE_GAP_DAYS
    end_ok = (pd.Timestamp("2024-12-31") - pd.Timedelta(days=tol)).strftime("%Y-%m-%d")
    end_bad = (pd.Timestamp("2024-12-31") - pd.Timedelta(days=tol + 1)).strftime("%Y-%m-%d")
    base = {"from": "2024-01-01", "to": "2024-12-31"}
    assert probe_mod.check_date_range(
        base, pd.DataFrame({"date": ["2024-01-01", end_ok]}))["status"] == "ok"
    assert probe_mod.check_date_range(
        base, pd.DataFrame({"date": ["2024-01-01", end_bad]}))["status"] == "SHORTFALL"


def test_capture_calls_the_check_and_halts_on_shortfall():
    """Call-site tripwire: the assertion must be wired into capture(), not merely defined."""
    tree = _tree()
    capture = [n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "capture"]
    assert capture, "probe must have a capture() function"
    body = capture[0]
    called = {n.func.id for n in ast.walk(body)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "check_date_range" in called, "capture() must check the returned date range"
    raises = [n for n in ast.walk(body) if isinstance(n, ast.Raise)]
    assert any(isinstance(r.exc, ast.Call) and getattr(r.exc.func, "id", "") == "SystemExit"
               for r in raises), "capture() must halt on a shortfall"
