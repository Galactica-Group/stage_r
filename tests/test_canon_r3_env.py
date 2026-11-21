"""
================================================================================
Script:       test_canon_r3_env.py
Version:      1.1.0
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
import hashlib
import json
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


# --------------------------------------------------- R5: statistics only
# The repo's own test_tracked_reports_carry_no_vendor_value is scoped to the
# snapshot-diff report pair, so the canon_r3 artifacts get their own guard.
#
# The rule forbids a vendor VALUE, not the NAME of a vendor column: the draft
# registration has to be able to say in prose which column the returns come
# from. So the JSON artifacts are scanned for a vendor column used AS A KEY
# (which is how a value would arrive), and every artifact is scanned for the
# per-name identifier, which is how a per-security row would arrive.
VENDOR_VALUE_KEYS = {"open", "high", "low", "close", "closeadj", "closeunadj",
                     "volume", "price", "prices", "dividend", "permaticker"}
IDENTITY_TOKEN = "permaticker"

WORLD_STAT_KEYS = {
    "k", "gross_t", "gross_mean", "gross_sharpe", "net_base_t", "net_base_mean",
    "net_base_sharpe", "net_stress_sharpe", "turnover_mean", "months",
    "label_rows", "wall_seconds",
}


def _vendor_keys(obj, found=None):
    """Every vendor column name used as a key anywhere in a parsed JSON tree."""
    found = set() if found is None else found
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in VENDOR_VALUE_KEYS:
                found.add(str(k).lower())
            _vendor_keys(v, found)
    elif isinstance(obj, list):
        for v in obj:
            _vendor_keys(v, found)
    return found


def _per_name_table_rows(md: str) -> list[str]:
    """
    Markdown rows that would carry a per-security value: a table row naming the
    identity column or a vendor price column. A registration has to be able to
    SAY in prose which column returns come from and what the scramble is keyed
    on; what it may not do is tabulate per-name data.
    """
    bad = []
    for line in md.splitlines():
        t = line.strip().lower()
        if not t.startswith("|"):
            continue
        cells = [c.strip() for c in t.strip("|").split("|")]
        if any(c in VENDOR_VALUE_KEYS or c.strip("`") in VENDOR_VALUE_KEYS
               for c in cells):
            bad.append(line.strip())
    return bad


def _string_values(obj):
    """Every string value anywhere in a parsed JSON tree."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _string_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _string_values(v)


def test_the_string_value_check_bites_on_a_table_hidden_in_json_prose():
    """Positive control: a per-name table smuggled into a JSON string value."""
    blob = {"text": "intro\n| permaticker | closeunadj |\n|---|---|\n| 635685 | 41.4 |\n"}
    rows = [r for v in _string_values(blob) for r in _per_name_table_rows(v)]
    assert rows, "a per-name table in a string value must be caught"
    ok = {"text": "the scramble is keyed on the sorted permaticker list;\n"
                  "returns come from closeadj.\n| series | mean |\n|---|---|\n"
                  "| gross | +1.0 |\n"}
    assert [r for v in _string_values(ok) for r in _per_name_table_rows(v)] == []


def test_the_vendor_value_check_rejects_a_constructed_artifact():
    """Positive control: the helpers must bite on a planted per-name price row."""
    planted = {"rows": [{"permaticker": 635685, "closeunadj": 41.37}]}
    assert _vendor_keys(planted) == {"permaticker", "closeunadj"}
    assert _vendor_keys({"pool": {"worlds": [{"gross_sharpe": 0.07}]}}) == set()
    assert IDENTITY_TOKEN in json.dumps(planted)


def test_the_per_name_table_check_bites_on_a_table_and_not_on_prose():
    """Positive control for the markdown rule."""
    table = ("| permaticker | closeunadj |\n|---|---|\n| 635685 | 41.37 |\n")
    assert _per_name_table_rows(table)
    prose = ("World k shifts each month's targets along the sorted permaticker\n"
             "list; returns come from closeadj and the filter from closeunadj.\n"
             "| series | mean |\n|---|---|\n| gross | +1.0 |\n")
    assert _per_name_table_rows(prose) == []


def test_tracked_canon_artifacts_carry_no_vendor_value():
    dirs = [REPO / "registrations" / "canon_r3", REPO / "reports" / "canon_r3"]
    dirs = [d for d in dirs if d.exists()]
    if not dirs:
        pytest.skip("no canon_r3 artifacts yet")
    files = sorted(f for d in dirs
                   for f in list(d.glob("*.json")) + list(d.glob("*.md")))
    assert files, "the guard would be vacuous"
    for f in files:
        text = f.read_text()
        if f.suffix == ".json":
            blob = json.loads(text)
            bad = _vendor_keys(blob)
            assert not bad, "%s uses vendor columns as keys: %s" % (f.name, sorted(bad))
            # A JSON artifact may carry PROSE in a string value — the sealed
            # registration embeds its own markdown text byte for byte, and that
            # text has to be able to say the scramble is keyed on the sorted
            # permaticker list. What prose may not do is tabulate per-name data,
            # so every string value is checked by the markdown rule.
            for v in _string_values(blob):
                rows = _per_name_table_rows(v)
                assert not rows, "%s tabulates per-name data in a string value: %s" \
                    % (f.name, rows[:2])
        else:
            rows = _per_name_table_rows(text)
            assert not rows, "%s tabulates per-name data: %s" % (f.name, rows[:2])


def test_the_controls_artifacts_report_only_whitelisted_world_statistics():
    cs = sorted((REPO / "registrations" / "canon_r3").glob("controls_v*.json")) \
        if (REPO / "registrations" / "canon_r3").exists() else []
    if not cs:
        pytest.skip("no controls artifact yet")
    for c in cs:
        blob = json.loads(c.read_text())
        worlds = blob["pool"]["worlds"]
        assert len(worlds) == 60, c.name
        for w in worlds:
            assert set(w) <= WORLD_STAT_KEYS, (c.name, set(w) - WORLD_STAT_KEYS)


def test_the_tracked_pool_reports_only_whitelisted_statistics():
    f = REPO / "registrations" / "canon_r3" / "pool_v1.json"
    if not f.exists():
        pytest.skip("no tracked pool yet")
    blob = json.loads(f.read_text())
    assert blob["n_worlds"] == 60
    leg_keys = {"mean", "sd", "t", "sharpe"}
    for w in blob["worlds"]:
        assert set(w) == {"k", "gross", "net_base", "net_stress",
                          "turnover_mean", "months", "label_rows"}, set(w)
        for leg in ("gross", "net_base", "net_stress"):
            assert set(w[leg]) == leg_keys, (leg, set(w[leg]))
        assert "wall_seconds" not in json.dumps(w)


SEAL_JSON = "registration_sealed.json"
SEAL_MD = "registration_sealed.md"


def _sealed_named_files(root):
    return sorted(p for p in root.rglob("*")
                  if "sealed" in p.name.lower()
                  and ".git" not in p.parts and "venv" not in p.parts
                  and "__pycache__" not in p.parts)


def test_the_only_files_named_sealed_are_the_seal_and_the_halted_seal():
    """
    2.4. Two files may carry "sealed" in their names and no others:
    registration_sealed.json, the seal the read is taken against, and
    registration_sealed.md, the halted seal, which must have halt_1.json beside
    it. The JSON's text field must hash to the markdown's sha256, and the
    receipt must name the JSON seal.
    """
    canon = REPO / "registrations" / "canon_r3"
    hits = _sealed_named_files(REPO)
    names = [h.name for h in hits]
    assert set(names) <= {SEAL_JSON, SEAL_MD}, names
    for h in hits:
        assert h.parent == canon, str(h.relative_to(REPO))

    md = canon / SEAL_MD
    js = canon / SEAL_JSON
    if md.exists():
        assert (canon / "halt_1.json").exists(), \
            "the markdown seal is halted and must have halt_1.json beside it"
    if js.exists():
        blob = json.loads(js.read_text())
        assert md.exists(), "the JSON seal names a text file that must be tracked"
        assert blob["text"] == md.read_text(), \
            "the JSON seal's text is not the markdown seal byte for byte"
        live_md = hashlib.sha256(md.read_bytes()).hexdigest()
        assert blob["text_sha256"] == live_md, \
            "text_sha256 %s is not the markdown seal's sha256 %s" % (
                blob["text_sha256"], live_md)
        receipt = canon / "controls_receipt.json"
        if receipt.exists():
            rec = json.loads(receipt.read_text())
            live_js = hashlib.sha256(js.read_bytes()).hexdigest()
            assert rec.get("registration_sha") == live_js, \
                "the receipt names %s but the JSON seal hashes to %s" % (
                    rec.get("registration_sha"), live_js)


def test_the_sealed_name_rule_rejects_a_constructed_third_file(tmp_path):
    """Positive control: a third "sealed"-named file must break the predicate."""
    (tmp_path / "registration_sealed.json").write_text("{}")
    (tmp_path / "registration_sealed.md").write_text("x")
    (tmp_path / "registration_sealed_v2.json").write_text("{}")
    names = {p.name for p in _sealed_named_files(tmp_path)}
    assert not names <= {SEAL_JSON, SEAL_MD}, names
    assert "registration_sealed_v2.json" in names


def test_the_text_equality_check_bites_on_a_constructed_divergence(tmp_path):
    """Positive control for the byte-for-byte text rule."""
    md = tmp_path / "registration_sealed.md"
    md.write_text("the sealed text\n")
    good = {"text": md.read_text(),
            "text_sha256": hashlib.sha256(md.read_bytes()).hexdigest()}
    assert good["text"] == md.read_text()
    assert good["text_sha256"] == hashlib.sha256(md.read_bytes()).hexdigest()
    bad = {"text": "the sealed text with a word added\n",
           "text_sha256": good["text_sha256"]}
    assert bad["text"] != md.read_text()
    assert hashlib.sha256(bad["text"].encode()).hexdigest() != bad["text_sha256"]


def test_the_spent_marker_is_not_pinned_and_the_pool_is(tmp_path):
    """The read writes the spent marker, so it cannot be a pinned artifact."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "canon_r3_gate_probe", REPO / "tools" / "canon_r3_gate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    names = set(mod.pinned_paths())
    assert "pool_v1" in names and "requirements_lock" in names
    assert not any("spent" in n for n in names)
    assert not any("read3" in n for n in names)
