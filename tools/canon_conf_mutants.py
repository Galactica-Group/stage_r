"""
================================================================================
Script:       canon_conf_mutants.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The mutant harness: for each guard of the confirmation, break it in
              a throwaway copy of the repo with no data symlink and show the test
              that names it go red.
Entry Point:  Yes — python tools/canon_conf_mutants.py [scratch_root]
Dependencies: stdlib only
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
================================================================================
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/tmp/conf1b_mut")
PY = REPO / "venv" / "bin" / "python"

TESTS = ["tests/test_canon_conf_build.py", "tests/test_canon_conf_stopped.py",
         "tests/test_canon_conf_gate.py"]
T_B = "tests/test_canon_conf_build.py"
T_S = "tests/test_canon_conf_stopped.py"
T_G = "tests/test_canon_conf_gate.py"

F_FEAT = "tools/canon_conf_features.py"
F_PANEL = "tools/canon_conf_panel.py"
F_PIPE = "tools/canon_conf_pipeline.py"
F_GATE = "tools/canon_conf_gate.py"
F_RUN = "tools/run_canon_conf.py"

COPY_RULE = T_B + "::test_the_features_copy_is_the_original_plus_the_declared_substitutions"

MUTANTS: dict[str, tuple[str, list[tuple[str, str]], list[str]]] = {
    "cutoff_both_halves_removed": (
        F_FEAT,
        [("filter=flt, batch_size=2_000_000).to_batches():",
          "batch_size=2_000_000).to_batches():"),
         ('        df = xt.filter_to_half_a(df, "date", cut, "canon_r3 features")\n',
          "")],
        [T_B + "::test_the_loader_calls_the_shared_half_a_filter"],
    ),
    "cutoff_assert_alone_removed": (
        F_FEAT,
        [('        df = xt.filter_to_half_a(df, "date", cut, "canon_r3 features")\n',
          "")],
        [T_B + "::test_the_loader_calls_the_shared_half_a_filter"],
    ),
    "cutoff_points_at_half_a_not_the_holdout": (
        F_FEAT,
        [('cut = windows["holdout"]["start"]',
          'cut = windows["half_a"]["daily_row_exclusive_max"]')],
        [T_B + "::test_the_feature_loader_reads_the_holdout_start_not_half_a"],
    ),
    "panel_cutoff_hardcoded_to_2013": (
        F_PANEL,
        [('    holdout_start = w["holdout"]["start"]',
          '    holdout_start = "2013-01-01"')],
        [T_B + "::test_the_cutoff_is_the_holdout_start_from_the_config",
         T_B + "::test_the_panel_builder_and_the_honest_module_name_no_2013_cutoff"],
    ),
    "window_ends_at_s_t_not_g": (
        F_FEAT,
        [('    u["g"] = u["s_t"] - 1.0', '    u["g"] = u["s_t"]')],
        [COPY_RULE],
    ),
    "undeclared_change_in_the_generated_copy": (
        F_FEAT,
        [("VERSION = \"1.0.0\"", "VERSION = \"1.0.0\"\n_SNEAKED_IN = True")],
        [COPY_RULE,
         T_B + "::test_the_copy_changes_only_the_window_and_its_own_identity"],
    ),
    "train_through_c_minus_one": (
        F_PIPE,
        [("             (m <= C - 2) & np.isfinite(y)",
          "             (m <= C - 1) & np.isfinite(y)")],
        [T_B + "::test_the_walk_trains_through_c_minus_two_and_scores_from_c"],
    ),
    "label_reaches_the_feature_matrix": (
        F_PIPE,
        [("    X = world[feature_names].to_numpy(dtype=np.float64)",
          '    X = world[feature_names + ["fwd_ret"]].to_numpy(dtype=np.float64)')],
        [T_B + "::test_the_feature_matrix_is_the_twenty_five_features"],
    ),
    "scramble_is_the_identity": (
        F_PIPE,
        [("        pieces.append((rows, np.roll(rows, -1)))",
          "        pieces.append((rows, rows))")],
        [T_B + "::test_the_scramble_is_not_the_identity"],
    ),
    "scramble_moves_a_target_between_months": (
        F_PIPE,
        [("""    src = np.concatenate([p[1] for p in pieces])
    dst = np.concatenate([p[0] for p in pieces])""",
          """    dst = np.concatenate([p[0] for p in pieces])
    src = np.roll(dst, -1)""")],
        [T_B + "::test_the_scramble_preserves_each_months_multiset_and_moves_no_month"],
    ),
    "the_stopped_bound_reaches_the_verdict": (
        F_RUN,
        [("""def verdict(k1a, k1b, k2) -> str:
    if k1a["pass"] and k1b["pass"] and k2["pass"]:""",
          """def verdict(k1a, k1b, k2) -> str:
    import canon_conf_stopped as s2_leak          # noqa: F401
    if k1a["pass"] and k1b["pass"] and k2["pass"]:""")],
        [T_S + "::test_the_bound_feeds_no_label_no_prediction_and_no_key"],
    ),
    "world_zero_built_outside_the_gate": (
        F_PIPE,
        [("        if not isinstance(registration, VerifiedConfRegistration):",
          "        if False:")],
        [T_B + "::test_world_zero_is_refused_on_a_flag"],
    ),
    "world_zero_accepted_on_the_half_a_cuts": (
        F_PIPE,
        [("        if list(cuts) == list(R3_CUTS):", "        if False:")],
        [T_B + "::test_world_zero_is_refused_always_on_read_threes_cuts"],
    ),
    "gate_part_1_seal_sha_check_removed": (
        F_GATE,
        [("    if got != expected_sha:", "    if False:")],
        [T_G + "::test_load_registration_refuses_a_sha_mismatch",
         T_G + "::test_a_byte_flip_breaks_the_binding"],
    ),
    "gate_part_2_data_pin_check_removed": (
        F_GATE,
        [("    if bad:\n        raise DataPinMismatch",
          "    if False:\n        raise DataPinMismatch")],
        [T_G + "::test_matching_pins_pass_and_a_mismatch_is_refused",
         T_G + "::test_a_moved_artifact_is_refused_against_a_good_seal"],
    ),
    "gate_part_3_receipt_check_removed": (
        F_GATE,
        [("    if got != reg.sha256:", "    if False:")],
        [T_G + "::test_a_receipt_for_another_seal_is_refused"],
    ),
    "gate_part_4_lock_not_exclusive": (
        F_GATE,
        [("os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY")],
        [T_G + "::test_the_single_run_lock_refuses_a_second_entry"],
    ),
    "gate_part_5_spent_check_removed": (
        F_GATE,
        [("    if p.exists():\n        raise AlreadySpent",
          "    if False:\n        raise AlreadySpent")],
        [T_G + "::test_the_spent_marker_refuses_a_second_read"],
    ),
    "read_takes_the_pool_from_scratch": (
        F_RUN,
        [("        pool = json.loads(P.pool_v1().read_text())",
          "        pool = json.loads(P.pool_manifest().read_text())")],
        [T_G + "::test_the_read_takes_the_pool_from_the_tracked_file_and_nowhere_else"],
    ),
}


def build_template(root: Path) -> Path:
    tpl = root / "_template"
    if tpl.exists():
        shutil.rmtree(tpl)
    tpl.mkdir(parents=True)
    for d in ("tools", "tests", "config"):
        shutil.copytree(REPO / d, tpl / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in ("requirements.lock", "README.md"):
        shutil.copy2(REPO / f, tpl / f)
    (tpl / "reports").mkdir(exist_ok=True)
    shutil.copy2(REPO / "reports" / "triage_ledger.csv",
                 tpl / "reports" / "triage_ledger.csv")
    assert not (tpl / "data").exists(), "the template must carry no data"
    return tpl


def run_pytest(cwd: Path, args: list[str]) -> tuple[int, str]:
    p = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider"]
                       + args, cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    tpl = build_template(root)

    rc, out = run_pytest(tpl, TESTS)
    base = out.strip().splitlines()[-1] if out.strip() else ""
    if rc != 0:
        print("BASELINE NOT GREEN in the template:\n%s" % out[-4000:])
        return 1
    print("baseline in the copy: %s" % base)

    results = []
    for name, (rel, subs, nodes) in MUTANTS.items():
        work = root / ("m_" + name)
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(tpl, work, ignore=shutil.ignore_patterns("__pycache__"))
        src = (work / rel).read_text()
        ok = True
        for old, new in subs:
            if src.count(old) != 1:
                print("MUTANT %s: anchor appears %d times, expected 1: %r"
                      % (name, src.count(old), old[:60]))
                results.append({"mutant": name, "killed": False,
                                "reason": "anchor not unique"})
                ok = False
                break
            src = src.replace(old, new)
        if ok:
            (work / rel).write_text(src)
            rc, out = run_pytest(work, nodes)
            killed = rc != 0
            last = out.strip().splitlines()[-1] if out.strip() else ""
            results.append({"mutant": name, "file": rel, "nodes": nodes,
                            "killed": killed, "pytest": last})
            print("%-44s %s   %s" % (name, "KILLED  " if killed else "SURVIVED",
                                     last))
        shutil.rmtree(work, ignore_errors=True)

    n, k = len(results), sum(1 for r in results if r["killed"])
    print("\nmutants killed %d of %d" % (k, n))
    (root / "mutant_results.json").write_text(json.dumps(
        {"baseline": base, "killed": k, "total": n, "results": results}, indent=2))
    return 0 if k == n else 1


if __name__ == "__main__":
    sys.exit(main())
