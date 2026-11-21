"""
================================================================================
Script:       canon_r3_rehearsal.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3
Purpose:      The rehearsal of clause 2.6: in a throwaway copy of the repo with
              no data symlink, stand up the full post-read tree — seal, receipt,
              spent marker, result, report and one CANON-R3 ledger row appended
              by the repo's own chain definition — once for each of the three
              verdicts, and prove the chain verifies and the suite accepts it.
Entry Point:  Yes — python tools/canon_r3_rehearsal.py [scratch_root]
Dependencies: stdlib only
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================

Whatever the suite would refuse after the read is found here, before the seal
exists, where it costs nothing. The dummy artifacts carry invented statistics
and no vendor value; the copy has no data/ at all, so nothing can reach the
banked panel or the snapshot.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/tmp/stock2_rehearsal")
PY = REPO / "venv" / "bin" / "python"

VERDICTS = ("NOTHING", "SIGNIFICANT_IMMATERIAL", "SURVIVOR")

CANON_TESTS = ["tests/test_canon_r3_env.py", "tests/test_canon_r3_features.py",
               "tests/test_canon_r3_world.py", "tests/test_canon_r3_machine.py",
               "tests/test_canon_r3_gate.py", "tests/test_canon_r3_read.py"]
# The one test that cannot run in a copy: it shells out to the copy's venv.
DESELECT = ["tests/test_canon_r3_env.py::test_the_live_venv_equals_the_lock"]


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_copy(root: Path, tag: str) -> Path:
    work = root / tag
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    for d in ("tools", "tests", "config"):
        shutil.copytree(REPO / d, work / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("requirements.lock", "README.md"):
        shutil.copy2(REPO / f, work / f)
    (work / "registrations").mkdir()
    shutil.copytree(REPO / "registrations" / "canon_r3",
                    work / "registrations" / "canon_r3") \
        if (REPO / "registrations" / "canon_r3").exists() else \
        (work / "registrations" / "canon_r3").mkdir(parents=True)
    (work / "reports").mkdir(exist_ok=True)
    shutil.copy2(REPO / "reports" / "triage_ledger.csv",
                 work / "reports" / "triage_ledger.csv")
    assert not (work / "data").exists(), "the rehearsal copy must carry no data"
    return work


def dummy_pool(n=60) -> dict:
    """Invented statistics in the tracked pool's exact shape."""
    ws = []
    for k in range(1, n + 1):
        ws.append({"k": k,
                   "gross": {"mean": 1e-5 * k, "sd": 0.0078,
                             "t": 0.02 * k - 0.6, "sharpe": 0.01 * k - 0.3},
                   "net_base": {"mean": -1.4e-3, "sd": 0.0078, "t": -2.1,
                                "sharpe": -0.64},
                   "net_stress": {"mean": -4e-3, "sd": 0.0078, "t": -5.0,
                                  "sharpe": -1.7},
                   "turnover_mean": 2.54, "months": 132, "label_rows": 165120})
    return {"version": "1.0.0", "n_worlds": n,
            "scratch_manifest_sha256": "0" * 64,
            "feature_sha256": "1" * 64, "banked_panel_sha256": "2" * 64,
            "env": {"python": "3.13.3", "numpy": "2.5.3", "pandas": "3.0.5",
                    "pyarrow": "25.0.1", "scikit-learn": "1.9.1"},
            "worlds": ws}


def dummy_controls() -> dict:
    return {"stage": "controls", "version": "1.1.0",
            "pool": {"n_worlds": 60,
                     "worlds": [{"k": k, "gross_t": 0.0, "gross_mean": 0.0,
                                 "gross_sharpe": 0.0, "net_base_t": -2.0,
                                 "net_base_mean": -1e-3, "net_base_sharpe": -0.6,
                                 "net_stress_sharpe": -1.7, "turnover_mean": 2.5,
                                 "months": 132, "label_rows": 165120,
                                 "wall_seconds": 24.0}
                                for k in range(1, 61)]}}


def stand_up(work: Path, verdict: str) -> dict:
    canon = work / "registrations" / "canon_r3"
    canon.mkdir(parents=True, exist_ok=True)
    (canon / "pool_v1.json").write_text(json.dumps(dummy_pool(), indent=2))
    (canon / "controls_v2.json").write_text(json.dumps(dummy_controls(), indent=2))

    seal = canon / "registration_sealed.md"
    seal.write_text("# REHEARSAL SEAL (%s)\n\nNot a seal. Statistics only.\n"
                    % verdict)
    seal_sha = sha256_file(seal)

    (canon / "controls_receipt.json").write_text(json.dumps(
        {"look": "CANON-R3", "registration_sha": seal_sha,
         "controls_v2_sha256": sha256_file(canon / "controls_v2.json")}, indent=2))
    (canon / "read3_spent.json").write_text(json.dumps(
        {"look": "CANON-R3", "registration_sha256": seal_sha,
         "utc": "2026-11-21T00:00:00Z", "head": "0" * 40}, indent=2))

    rep = work / "reports" / "canon_r3"
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "read3_result_v1.json").write_text(json.dumps(
        {"look": "CANON-R3", "verdict": verdict, "registration_sha": seal_sha,
         "key1a": {"p": 0.5, "pass": verdict != "NOTHING"},
         "key1b": {"exceedances": 1, "pass": verdict != "NOTHING"},
         "key2": {"net_base_annualised_sharpe": -0.6,
                  "pass": verdict == "SURVIVOR"},
         "reported": {"turnover_mean": 2.54}}, indent=2))
    (rep / "read3_report_v1.md").write_text(
        "# Canon read 3 — REHEARSAL — %s\n\nStatistics only.\n" % verdict)

    # the ledger row, by the repo's own chain definition
    sys.path.insert(0, str(work / "tools"))
    for m in ("ledger_chain",):
        sys.modules.pop(m, None)
    import ledger_chain as lc
    led = work / "reports" / "triage_ledger.csv"
    ok, n, head, bad = lc.verify(led)
    if not ok:
        raise AssertionError("the copied ledger is already broken: %r" % bad)
    vals = ["CANON-R3", "20261121", "20261004p", seal_sha[:16], "a" * 16,
            "MACHINE_HGB_25PV", "top1000_tight_v2_halfA_132mo", 132,
            "%.4f" % -14.0, "%.4f" % 0.5, verdict]
    rs = lc.row_sha(head, vals)
    with open(led, "a") as f:
        f.write(",".join([str(x) for x in vals] + [head, rs]) + "\n")
    ok2, n2, head2, bad2 = lc.verify(led)
    return {"seal_sha256": seal_sha, "rows_before": n, "rows_after": n2,
            "chain_ok": bool(ok2), "chain_head": head2[:16],
            "chain_bad": bad2 if not ok2 else None}


def run_pytest(cwd: Path) -> tuple[int, str]:
    args = ([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider"]
            + CANON_TESTS + sum([["--deselect", d] for d in DESELECT], []))
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    out = {"version": VERSION, "root": str(root), "runs": {}}
    bad = 0
    for v in VERDICTS:
        work = build_copy(root, "r_" + v.lower())
        st = stand_up(work, v)
        rc, log = run_pytest(work)
        last = log.strip().splitlines()[-1] if log.strip() else ""
        green = rc == 0
        out["runs"][v] = {**st, "suite_green": green, "pytest": last}
        print("%-24s chain %s %d->%d  suite %s   %s"
              % (v, "OK" if st["chain_ok"] else "BROKEN",
                 st["rows_before"], st["rows_after"],
                 "GREEN" if green else "RED", last))
        if not green:
            print(log[-4000:])
        if not (green and st["chain_ok"]):
            bad += 1
        shutil.rmtree(work, ignore_errors=True)
    out["all_green"] = bad == 0
    (root / "rehearsal_results.json").write_text(json.dumps(out, indent=2))
    print("\nrehearsal: %s" % ("ALL GREEN" if bad == 0 else "%d FAILED" % bad))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
