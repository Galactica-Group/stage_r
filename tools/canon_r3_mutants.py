"""
================================================================================
Script:       canon_r3_mutants.py
Version:      1.1.0
Phase:        Stage R — Canon Read 3
Purpose:      The mutant harness: for each guard, break it in a throwaway copy
              of the repo and show the test that names it goes red.
Entry Point:  Yes — python tools/canon_r3_mutants.py [scratch_root]
Dependencies: stdlib only
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================

Mutants never run in the working tree. The harness builds a template copy under
the scratch root with NO data symlink, proves the canon tests green there, then
for each mutant copies the template, applies one exact string substitution, and
runs only the node ids that mutant is supposed to kill. A mutant that leaves its
test green is a finding, not a pass.
================================================================================
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "1.1.0"

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/tmp/stock1_mut")
PY = REPO / "venv" / "bin" / "python"

CANON_TESTS = ["tests/test_canon_r3_features.py", "tests/test_canon_r3_world.py",
               "tests/test_canon_r3_machine.py", "tests/test_canon_r3_gate.py",
               "tests/test_canon_r3_read.py"]

T_FEAT = "tests/test_canon_r3_features.py"
T_WORLD = "tests/test_canon_r3_world.py"
T_MACH = "tests/test_canon_r3_machine.py"
T_GATE = "tests/test_canon_r3_gate.py"
T_READ = "tests/test_canon_r3_read.py"

# name -> (file, [(old, new), ...], [node ids that must now FAIL])
MUTANTS: dict[str, tuple[str, list[tuple[str, str]], list[str]]] = {
    "half_a_filter_removed_both_halves": (
        "tools/canon_r3_features.py",
        [("filter=flt, batch_size=2_000_000).to_batches():",
          "batch_size=2_000_000).to_batches():"),
         ('        df = xt.filter_to_half_a(df, "date", cut, "canon_r3 features")\n',
          "")],
        [T_FEAT + "::test_the_loader_never_returns_a_row_at_or_after_the_cutoff"],
    ),
    "half_a_assert_call_removed": (
        "tools/canon_r3_features.py",
        [('        df = xt.filter_to_half_a(df, "date", cut, "canon_r3 features")\n',
          "")],
        [T_FEAT + "::test_the_loader_calls_the_shared_half_a_filter"],
    ),
    "window_ends_at_s_t_not_g": (
        "tools/canon_r3_features.py",
        [('    u["g"] = u["s_t"] - 1.0', '    u["g"] = u["s_t"]')],
        [T_FEAT + "::test_the_anchor_is_the_session_before_the_month_end",
         T_FEAT + "::test_truncating_the_daily_data_at_g_changes_no_feature"],
    ),
    "rank_scaled_across_months": (
        "tools/canon_r3_features.py",
        [('        out[c] = out.groupby(month_col, observed=True)[c].transform(f)',
          '        out[c] = f(out[c])')],
        [T_FEAT + "::test_the_rank_scaling_is_within_the_month_and_lands_in_the_half_open_band"],
    ),
    "train_through_c_minus_one": (
        "tools/canon_r3_machine.py",
        [("             (m <= C - 2) & np.isfinite(y)",
          "             (m <= C - 1) & np.isfinite(y)")],
        [T_MACH + "::test_the_walk_trains_through_c_minus_two_and_scores_from_c"],
    ),
    "train_through_c": (
        "tools/canon_r3_machine.py",
        [("             (m <= C - 2) & np.isfinite(y)",
          "             (m <= C) & np.isfinite(y)")],
        [T_MACH + "::test_the_walk_trains_through_c_minus_two_and_scores_from_c"],
    ),
    "label_column_reaches_the_feature_matrix": (
        "tools/canon_r3_machine.py",
        [("    X = world[feature_names].to_numpy(dtype=np.float64)",
          '    X = world[feature_names + ["fwd_ret"]].to_numpy(dtype=np.float64)')],
        [T_MACH + "::test_the_feature_matrix_is_the_twenty_five_features_and_nothing_else",
         T_MACH + "::test_the_label_is_a_target_only_and_is_not_among_the_columns_fitted"],
    ),
    "scramble_is_the_identity": (
        "tools/canon_r3_world.py",
        [("        rolled = np.roll(rows, -1)          # each takes the NEXT name's target",
          "        rolled = rows")],
        [T_WORLD + "::test_the_scramble_is_not_the_identity",
         T_WORLD + "::test_the_scramble_is_a_cyclic_shift_along_the_key_order"],
    ),
    "scramble_moves_a_target_between_months": (
        "tools/canon_r3_world.py",
        [("""    src = np.concatenate([p[1] for p in pieces])
    dst = np.concatenate([p[0] for p in pieces])""",
          """    dst = np.concatenate([p[0] for p in pieces])
    src = np.roll(dst, -1)""")],
        [T_WORLD + "::test_the_scramble_preserves_each_months_multiset_exactly"],
    ),
    "plant_accepts_world_zero": (
        "tools/canon_r3_world.py",
        [('''    if int(k) < 1:
        raise PlantRefused("the plant is for scrambled worlds only; got k=%r" % k)
''', "")],
        [T_WORLD + "::test_the_plant_refuses_world_zero"],
    ),
    "world_zero_built_outside_the_gate": (
        "tools/canon_r3_world.py",
        [('''        if not isinstance(registration, VerifiedRegistration):''',
          '''        if False:''')],
        [T_WORLD + "::test_build_world_refuses_world_zero_on_a_flag"],
    ),
    "seal_sha_check_removed": (
        "tools/canon_r3_gate.py",
        [("    if got != expected_sha:", "    if False:")],
        [T_GATE + "::test_load_registration_refuses_a_sha_mismatch",
         T_GATE + "::test_a_byte_flip_in_the_seal_breaks_the_binding"],
    ),
    "data_pin_check_removed": (
        "tools/canon_r3_gate.py",
        [("    if bad:\n        raise DataPinMismatch", "    if False:\n        raise DataPinMismatch")],
        [T_GATE + "::test_a_data_pin_mismatch_is_refused",
         T_GATE + "::test_a_moved_artifact_is_refused_against_a_good_seal"],
    ),
    "receipt_check_removed": (
        "tools/canon_r3_gate.py",
        [("    if got != reg.sha256:", "    if False:")],
        [T_GATE + "::test_the_observed_stage_refuses_a_receipt_for_another_seal"],
    ),
    # --- STOCK-2 additions ---
    "spent_check_removed": (
        "tools/canon_r3_gate.py",
        [("    if p.exists():\n        raise AlreadySpent(",
          "    if False:\n        raise AlreadySpent(")],
        [T_GATE + "::test_the_spent_marker_refuses_a_second_read"],
    ),
    "read_takes_the_pool_from_scratch": (
        "tools/run_canon_r3.py",
        [("        pool = json.loads(P.pool_v1().read_text())",
          "        pool = json.loads(P.pool_manifest().read_text())")],
        [T_READ + "::test_the_read_takes_the_pool_from_the_tracked_file_and_nowhere_else"],
    ),
    "tracked_pool_carries_timing": (
        "tools/run_canon_r3.py",
        [('            "label_rows": int(r["label_rows"]),',
          '            "label_rows": int(r["label_rows"]),\n'
          '            "wall_seconds": r["wall_seconds"],')],
        [T_READ + "::test_the_tracked_pool_writer_records_every_registered_statistic"],
    ),
    "spent_marker_written_before_the_gate": (
        "tools/run_canon_r3.py",
        [("""    G.refuse_if_spent()
    opened = G.open_the_gate(registration, sha)""",
          """    G.refuse_if_spent()
    G.write_spent_marker("x" * 64, "x")
    opened = G.open_the_gate(registration, sha)""")],
        [T_GATE + "::test_the_runner_writes_the_spent_marker_after_the_gate_and_before_world_zero"],
    ),
    "key1b_bar_is_not_the_third_largest": (
        "tools/run_canon_r3.py",
        [("        \"key1b_third_largest_pool_gross_t\": float(gt[2]),",
          "        \"key1b_third_largest_pool_gross_t\": float(gt[1]),")],
        [T_READ + "::test_the_bars_are_the_pool_median_and_the_third_largest_t",
         T_READ + "::test_the_third_largest_t_is_exactly_the_key1b_boundary"],
    ),
    "a_banked_rho_loses_its_seed_index": (
        "tools/run_canon_r3.py",
        [("PLANT_GRID = ((0.005, 0), (0.010, 1), (0.020, 2), (0.040, 3), (0.080, 4),",
          "PLANT_GRID = ((0.005, 0), (0.010, 1), (0.020, 2), (0.040, 7), (0.080, 4),")],
        [T_READ + "::test_the_five_banked_rhos_keep_their_seed_indices",
         T_READ + "::test_every_seed_index_is_used_once"],
    ),
    "single_run_lock_not_exclusive": (
        "tools/canon_r3_gate.py",
        [("os.O_CREAT | os.O_EXCL | os.O_WRONLY", "os.O_CREAT | os.O_WRONLY")],
        [T_GATE + "::test_the_single_run_lock_refuses_a_second_entry"],
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
    # the ledger is a tracked, statistics-only artifact the read tests parse
    (tpl / "reports").mkdir(exist_ok=True)
    shutil.copy2(REPO / "reports" / "triage_ledger.csv",
                 tpl / "reports" / "triage_ledger.csv")
    assert not (tpl / "data").exists(), "the template must carry no data"
    return tpl


def run_pytest(cwd: Path, args: list[str]) -> tuple[int, str]:
    p = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider"] + args,
                       cwd=cwd, capture_output=True, text=True)
    return p.returncode, (p.stdout + p.stderr)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    tpl = build_template(root)

    rc, out = run_pytest(tpl, CANON_TESTS)
    base = out.strip().splitlines()[-1] if out.strip() else ""
    if rc != 0:
        print("BASELINE NOT GREEN in the template:\n%s" % out[-3000:])
        return 1
    print("baseline in the copy: %s" % base)

    results = []
    for name, (rel, subs, nodes) in MUTANTS.items():
        work = root / ("m_" + name)
        if work.exists():
            shutil.rmtree(work)
        shutil.copytree(tpl, work, ignore=shutil.ignore_patterns("__pycache__"))
        src = (work / rel).read_text()
        for old, new in subs:
            if src.count(old) != 1:
                print("MUTANT %s: anchor appears %d times, expected 1"
                      % (name, src.count(old)))
                results.append({"mutant": name, "killed": False,
                                "reason": "anchor not unique"})
                break
            src = src.replace(old, new)
        else:
            (work / rel).write_text(src)
            rc, out = run_pytest(work, nodes)
            killed = rc != 0
            last = out.strip().splitlines()[-1] if out.strip() else ""
            results.append({"mutant": name, "file": rel, "nodes": nodes,
                            "killed": killed, "pytest": last})
            print("%-44s %s   %s" % (name, "KILLED  " if killed else "SURVIVED", last))
        shutil.rmtree(work, ignore_errors=True)

    n = len(results)
    k = sum(1 for r in results if r["killed"])
    print("\nmutants killed %d of %d" % (k, n))
    (root / "mutant_results.json").write_text(json.dumps(
        {"baseline": base, "killed": k, "total": n, "results": results}, indent=2))
    return 0 if k == n else 1


if __name__ == "__main__":
    sys.exit(main())
