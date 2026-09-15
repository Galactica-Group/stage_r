"""
================================================================================
Script:       test_probe_guard.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      Static guard on tools/sharadar_probe.py: dependency containment,
              factory-path isolation, and API-key hygiene.
Entry Point:  No — pytest
Dependencies: stdlib only (ast, pathlib, sys)
Last Sprint:  Sprint R0-A — Sharadar Free-Tier Probe (2026-09-15)
================================================================================

These are falsifiable structural assertions, not smoke tests: each fails on a
constructed violation of the property it names.
================================================================================
"""

import ast
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
PROBE = REPO / "tools" / "sharadar_probe.py"
ALLOWED_THIRD_PARTY = {"requests", "pandas", "pyarrow"}
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
