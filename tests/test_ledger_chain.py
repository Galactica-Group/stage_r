"""Tests for the triage ledger's sha chain (tools/ledger_chain.py, verify_ledger.py).

The real ledger is the fixture for the happy path, because the property that
matters is that THIS file replays. Tampering is done on a temp copy; the tracked
ledger is never written by this battery.

The AST tripwire exists so ledger_chain.row_sha cannot drift away from the
definition the four look runners carry. It records honestly that three runners
match character-for-character and the fourth is a two-statement variant, and it
closes that gap with a behavioural battery that must agree byte-for-byte.
"""

from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import ledger_chain as lc  # noqa: E402
import verify_ledger as vl  # noqa: E402

LEDGER = ROOT / "reports" / "triage_ledger.csv"
RUNNERS = ["run_t_r1.py", "run_t_r1b.py", "run_t_r3.py", "run_t_r3b.py"]

# The two forms the four runners actually carry, pinned verbatim. Three of them
# are the canonical one-liner; run_t_r1.py binds the payload first.
CANONICAL_BODY = ('return hashlib.sha256("\\x00".join([prev] + [str(v) for v in '
                  'values]).encode()).hexdigest()')
R1_BODY = ('payload = "\\x00".join([prev] + [str(v) for v in values])\n'
           'return hashlib.sha256(payload.encode()).hexdigest()')


def row_sha_source(path):
    src = Path(path).read_text()
    tree = ast.parse(src)
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and n.name == "row_sha":
            return (ast.get_source_segment(src, n),
                    "\n".join(ast.get_source_segment(src, st) for st in n.body))
    raise AssertionError("no row_sha in %s" % path)


# ------------------------------------------------------------------ chain
def test_chain_ok_on_the_real_ledger():
    ok, n, head, bad = lc.verify(LEDGER)
    assert ok, bad
    assert bad is None
    assert n >= 12
    assert len(head) == 64


def test_verify_ledger_cli_prints_chain_ok(capsys):
    rc = vl.main([str(LEDGER)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CHAIN OK" in out
    assert "rows" in out


def test_tampered_value_reports_broken_at_the_right_row(tmp_path):
    hdr, rows = lc.read_rows(LEDGER)
    target = 7                      # 1-indexed data row to corrupt
    rows[target - 1][-3] = "TAMPERED"
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    ok, n, head, bad = lc.verify(p)
    assert ok is False
    assert bad["row"] == target
    assert bad["kind"] == "row_sha"


def test_tampered_prev_link_reports_broken_at_the_right_row(tmp_path):
    hdr, rows = lc.read_rows(LEDGER)
    target = 3
    rows[target - 1][-2] = "0" * 64
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    ok, n, head, bad = lc.verify(p)
    assert ok is False
    assert bad["row"] == target
    assert bad["kind"] == "prev_row_sha"


def test_deleted_row_breaks_the_chain(tmp_path):
    hdr, rows = lc.read_rows(LEDGER)
    del rows[4]
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    ok, n, head, bad = lc.verify(p)
    assert ok is False
    assert bad["row"] == 5
    assert bad["kind"] == "prev_row_sha"


def test_cli_exit_1_and_names_the_row_on_a_broken_chain(tmp_path, capsys):
    hdr, rows = lc.read_rows(LEDGER)
    rows[1][-1] = "f" * 64
    p = tmp_path / "t.csv"
    with open(p, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(hdr)
        w.writerows(rows)
    rc = vl.main([str(p)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "CHAIN BROKEN" in out
    assert "first bad row 2" in out


def test_first_row_chains_from_the_literal_genesis():
    _, rows = lc.read_rows(LEDGER)
    assert rows[0][-2] == "GENESIS"
    assert lc.GENESIS == "GENESIS"


def test_head_is_the_last_stored_row_sha():
    _, rows = lc.read_rows(LEDGER)
    ok, n, head, _ = lc.verify(LEDGER)
    assert ok
    assert head == rows[-1][-1]


# --------------------------------------------------------------- tripwire
def test_ast_tripwire_row_sha_body_matches_every_runner():
    mine_full, mine_body = row_sha_source(ROOT / "tools" / "ledger_chain.py")
    assert mine_body == CANONICAL_BODY
    seen = {}
    for r in RUNNERS:
        full, body = row_sha_source(ROOT / "tools" / r)
        seen[r] = body
    # three runners are character-identical to ours, body AND full def for two
    for r in ("run_t_r1b.py", "run_t_r3.py", "run_t_r3b.py"):
        assert seen[r] == mine_body, r
    for r in ("run_t_r3.py", "run_t_r3b.py"):
        full, _ = row_sha_source(ROOT / "tools" / r)
        assert full == mine_full, r
    # run_t_r1.py is the pinned two-statement variant, recorded not hidden
    assert seen["run_t_r1.py"] == R1_BODY
    assert seen["run_t_r1.py"] != mine_body


@pytest.mark.parametrize("runner", RUNNERS)
def test_every_runner_row_sha_agrees_byte_for_byte(runner):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rr_" + runner.replace(".", "_"), ROOT / "tools" / runner)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    battery = [
        ("GENESIS", []),
        ("GENESIS", ["T-R4", "20261017", "v", "a", "b", "c", "d", 43, "1.0", "0.5", "NOTHING"]),
        ("0" * 64, [""]),
        ("x", [None, 0, -1, 1.5, "comma,inside", "\x00embedded"]),
        ("unicode", ["币安人生", "ok"]),
    ]
    for prev, vals in battery:
        assert mod.row_sha(prev, vals) == lc.row_sha(prev, vals)


def test_row_sha_is_order_sensitive_and_field_sensitive():
    a = lc.row_sha("p", ["a", "b"])
    assert a != lc.row_sha("p", ["b", "a"])
    assert a != lc.row_sha("q", ["a", "b"])
    assert a != lc.row_sha("p", ["ab"])
    assert a == lc.row_sha("p", ["a", "b"])


def test_no_credential_is_read_by_the_chain_tools():
    for f in ("ledger_chain.py", "verify_ledger.py"):
        path = ROOT / "tools" / f
        src = path.read_text()
        doc = ast.get_docstring(ast.parse(src)) or ""
        code = src.replace(doc, "")
        for forbidden in ("dotenv", "load_dotenv", ".env", "API_KEY", "api_key",
                          "secret", "X-MBX-APIKEY"):
            assert forbidden not in code, (f, forbidden)
