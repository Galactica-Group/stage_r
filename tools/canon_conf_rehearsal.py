"""
================================================================================
Script:       canon_conf_rehearsal.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The dress rehearsal: run the confirmation's own command lines end
              to end, in a copy of the repo with no data symlink, on synthetic
              data, once for a tree that ends CONFIRMED and once NOT_CONFIRMED.
Entry Point:  Yes — python tools/canon_conf_rehearsal.py [scratch_root]
Dependencies: stdlib only (it drives the real code in a subprocess-free copy)
Last Sprint:  Sprint CONF-2 — the recount, the corrected floors, the read (2026-11-24)
================================================================================

Read 3's rehearsal checked the files and never the gate, and a seal the gate
could not parse sailed through it. This one runs the gate and the observed stage
by their real entry points, and lists every input each stage reads.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

VERSION = "1.0.0"

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = Path("/tmp/conf1b_dress")
PY = REPO / "venv" / "bin" / "python"

TESTS = ["tests/test_canon_conf_build.py", "tests/test_canon_conf_stopped.py",
         "tests/test_canon_conf_gate.py"]

STEPS: list[tuple[str, bool, str]] = []


def step(name, ok, detail=""):
    STEPS.append((name, bool(ok), str(detail)))
    print("%-62s %s  %s" % (name, "DONE  " if ok else "FAILED", detail), flush=True)
    if not ok:
        raise SystemExit("rehearsal failed at: %s" % name)


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
    (work / "reports").mkdir(exist_ok=True)
    shutil.copy2(REPO / "reports" / "triage_ledger.csv",
                 work / "reports" / "triage_ledger.csv")
    assert not (work / "data").exists()
    return work


STAGE_INPUTS = {
    "controls": [
        "config/windows.json (via canon_conf_paths, for the cutoff)",
        "data/derived/canon_r3c/<panel>.parquet            (P.panel_file)",
        "data/derived/canon_r3c/<features>.parquet         (P.feature_file)",
        "data/derived/canon_r3c/features_meta.json         (non-NaN shares)",
        "data/derived/canon_r3c/s3_counts.json             (the S3 counts)",
        "registrations/canon_r3/pool_v1.json               (T3's banked worlds)",
        "data/derived/panel_monthly_...halfA_v2.parquet    (T3, banked)",
        "data/derived/canon_r3/features_...halfA_v2.parquet (T3, banked)",
    ],
    "observed": [
        "the registration named by --registration, hashed against --sha",
        "registrations/canon_r3c/controls_receipt.json      (the receipt)",
        "registrations/canon_r3c/pool_v1.json               (the 60 worlds)",
        "data/derived/canon_r3c/<panel>.parquet             (permatickers)",
        "data/derived/canon_r3c/<features>.parquet          (the matrix)",
        "reports/canon_r3/read3_result_v1.json              (read 3's Sharpes)",
        "every pinned artifact, for its sha256",
        "reports/triage_ledger.csv                          (the chain)",
    ],
}


def run_in_copy(work: Path, code: str) -> tuple[int, str]:
    p = subprocess.run([str(PY), "-c", code], cwd=work, capture_output=True,
                       text=True)
    return p.returncode, p.stdout + p.stderr


DRIVER = r'''
import json, sys, hashlib
from pathlib import Path
sys.path.insert(0, "tools"); sys.path.insert(0, "tests")
import canon_conf_paths as P
TMP = Path("synthetic").resolve()
TMP.mkdir(exist_ok=True)
import canon_conf_synthetic as SY
SY.install(TMP)
# the copy's repo root must stay the copy, so pins and the ledger resolve there
P.set_roots(repo=Path(".").resolve())
P.set_roots(derived=TMP / "derived", scratch=TMP / "scratch", snapshot=TMP / "snap",
            windows=TMP / "windows.json")
for d in (P.derived(), P.scratch(), P.canon_dir(), P.reports_dir()):
    d.mkdir(parents=True, exist_ok=True)
outside = [str(p) for p in (P.derived(), P.scratch(), P.windows_json())
           if not str(p).startswith(str(TMP))]
assert not outside, outside
import canon_conf_features as F
frame = SY.world_frame(n_names=%(NAMES)d, n_months=%(MONTHS)d, first="1998-12", seed=%(SEED)d)
SY.write_panel_and_features(TMP, frame)
SY.write_provenance(Path(".").resolve())
import canon_conf_pipeline as PL, canon_conf_gate as G, canon_conf_seal as SEAL
import run_canon_conf as R, canon_r3_machine as M
# a small window: %(NCUTS)d cut(s), the last 12 months scored
months = sorted(frame["month"].unique())
PL.CONF_CUTS = [int(months[-13])]
PL.CONF_SCORED = [int(m) for m in months[-12:]]
PL.CONF_N_SCORED_MONTHS = len(PL.CONF_SCORED)
monkey = {"TARGET": "%(TARGET)s"}
# --- controls stage, with T3 stubbed: T3 needs the banked half-A artifacts,
#     which a copy with no data symlink cannot hold. It is exercised for real
#     against the real artifacts, outside the rehearsal.
import canon_conf_tripwires as TW
TW.t3 = lambda worlds=None: {"tripwire":"T3","identical":True,"mismatches":0,
                             "per_world":[{"k":k,"mismatches":0} for k in (1,2,3,4,5)],
                             "stubbed_in_the_rehearsal": True}
PL.N_WORLDS = %(NWORLDS)d
R.PLANT_WORLDS = (1, 2)
R.PLANT_GRID = ((0.02, 2), (0.08, 7))
R.PLANT_RHOS = (0.02, 0.08)
R.PLANT_PASS_MIN = 1
M.R_VOTING = 200
rc = R.stage_controls()
assert rc == 0
pool = json.loads(P.pool_v1().read_text())
print("CONTROLS_OK", pool["n_worlds"], sha256_file := hashlib.sha256(
    P.pool_v1().read_bytes()).hexdigest()[:12])
# --- the seal, by canon_conf_seal, and a receipt
seal = P.sealed_registration()          # registration_v1.json, as the real one is
sha = SEAL.dummy(seal, "REHEARSAL dummy seal. Not a seal.")
P.controls_receipt().write_text(json.dumps({"registration_sha": sha}))
print("SEAL_OK", sha[:12])
# --- the gate, opened on it
opened = G.open_the_gate(seal, sha)
print("GATE_OK", opened["pins"]["pins_checked"])
# --- force the verdict word this tree is for.
# NOTE: read 3's key2/key1b bind their bars as DEFAULT ARGUMENTS, so moving the
# module constant has no effect; the functions themselves are patched. Only the
# rehearsal does this, and only to exercise both post-read trees.
def _k2(sharpe, bar=0.5, _pass=None):
    return {"net_base_annualised_sharpe": float(sharpe), "bar": float(bar),
            "pass": bool(_pass)}
if monkey["TARGET"] == "CONFIRMED":
    R.key1a = lambda g, c: {"p": 0.0001, "pass": True, "null_centre": float(c),
                            "n": len(g), "R": 200, "null_sd": 1e-3,
                            "null_mean": 0.0, "mean_block": 3.0}
    R.M.key1b = lambda t, ts, max_exc=2: {"gross_t": float(t), "n_worlds": len(ts),
                                          "exceedances": 0, "p": 1/61,
                                          "max_allowed": max_exc, "pass": True}
    R.M.key2 = lambda s, bar=0.5: _k2(s, bar, True)
else:
    R.M.key2 = lambda s, bar=0.5: _k2(s, bar, False)
# --- the observed stage, by its real command line
rc = R.main(["--stage", "observed", "--registration", str(seal), "--sha", sha])
assert rc == 0
res = json.loads(P.read_result().read_text())
print("OBSERVED_OK", res["verdict"], "|failed:", ",".join(res["failed_keys"]) or "none")
rp = res["reported"]
need = ("series_statistics","turnover_mean","break_even_bps_per_side",
        "net_base_vs_zero","by_year","months_scored",
        "months_skipped_rank_deficient","months_skipped_too_small",
        "cross_section_mean","cross_section_min","stopped_bound","ratio_to_read3")
print("REPORTED_OK", sum(k in rp for k in need), len(need))
print("ONDISK_OK", P.read_result().exists(), P.read_report().exists(),
      P.spent_marker().exists())
import ledger_chain as lc
ok, n, head, bad = lc.verify(P.ledger())
print("LEDGER_OK", ok, n, head[:12])
# --- a second observed run must be refused
try:
    R.main(["--stage","observed","--registration",str(seal),"--sha",sha])
    print("SECOND_RUN_NOT_REFUSED")
except G.AlreadySpent as e:
    print("SECOND_RUN_REFUSED", type(e).__name__)
# --- a markdown seal must be refused
md = P.canon_dir() / "registration_v1.md"
md.write_text("# not json\n")
try:
    G.load_registration(md, sha256_file2 := hashlib.sha256(md.read_bytes()).hexdigest())
    print("MARKDOWN_NOT_REFUSED")
except Exception as e:
    print("MARKDOWN_REFUSED", type(e).__name__)
'''


def rehearse(root: Path, target: str) -> dict:
    tag = "r_" + target.lower()
    work = build_copy(root, tag)
    step("%s 1 copy built, no data symlink" % tag,
         not (work / "data").exists(),
         "%d files" % sum(1 for _ in work.rglob("*") if _.is_file()))
    code = DRIVER % {"NAMES": 220, "MONTHS": 30, "SEED": 11, "NCUTS": 1,
                     "NWORLDS": 6, "TARGET": target}
    rc, out = run_in_copy(work, code)
    if rc != 0:
        print(out[-5000:])
    step("%s 2 driver completed" % tag, rc == 0, "rc=%d" % rc)
    got = {}
    for line in out.splitlines():
        if line.split(" ")[0].endswith("_OK") or "_REFUSED" in line \
                or "_NOT_REFUSED" in line:
            got[line.split(" ")[0]] = line
    step("%s 3 controls stage produced a pool" % tag, "CONTROLS_OK" in got,
         got.get("CONTROLS_OK", ""))
    step("%s 4 dummy JSON seal from canon_conf_seal" % tag, "SEAL_OK" in got,
         got.get("SEAL_OK", ""))
    step("%s 5 gate opened on it" % tag, "GATE_OK" in got, got.get("GATE_OK", ""))
    step("%s 6 observed stage reached a verdict word" % tag,
         "OBSERVED_OK" in got and target in got.get("OBSERVED_OK", ""),
         got.get("OBSERVED_OK", ""))
    step("%s 7 every reported quantity present" % tag,
         got.get("REPORTED_OK", "").endswith("12 12"), got.get("REPORTED_OK", ""))
    step("%s 8 result, report and spent marker on disk" % tag,
         got.get("ONDISK_OK") == "ONDISK_OK True True True",
         got.get("ONDISK_OK", ""))
    step("%s 9 ledger row appended, chain verified" % tag,
         got.get("LEDGER_OK", "").startswith("LEDGER_OK True 17"),
         got.get("LEDGER_OK", ""))
    step("%s 10 a second observed run is refused" % tag,
         "SECOND_RUN_REFUSED" in got, got.get("SECOND_RUN_REFUSED", ""))
    step("%s 11 a markdown seal is refused" % tag,
         "MARKDOWN_REFUSED" in got, got.get("MARKDOWN_REFUSED", ""))
    p = subprocess.run([str(PY), "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "tests/", "--deselect",
                        "tests/test_canon_r3_env.py::test_the_live_venv_equals_the_lock"],
                       cwd=work, capture_output=True, text=True)
    last = (p.stdout + p.stderr).strip().splitlines()[-1]
    if p.returncode != 0:
        print((p.stdout + p.stderr)[-4000:])
    step("%s 12 the suite in the copy is green" % tag, p.returncode == 0, last)
    shutil.rmtree(work, ignore_errors=True)
    return {"target": target, "lines": got}


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0]) if argv else DEFAULT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    print("INPUTS EACH STAGE READS")
    for stage, items in STAGE_INPUTS.items():
        print("  --stage %s:" % stage)
        for i in items:
            print("      %s" % i)
    print()
    out = {"version": VERSION, "stage_inputs": STAGE_INPUTS, "runs": {}}
    for target in ("NOT_CONFIRMED", "CONFIRMED"):
        out["runs"][target] = rehearse(root, target)
    out["all_done"] = all(ok for _, ok, _ in STEPS)
    (root / "rehearsal_steps.json").write_text(json.dumps(
        {"steps": [{"step": n, "ok": ok, "detail": d} for n, ok, d in STEPS],
         **out}, indent=2))
    print("\nREHEARSAL: %d of %d steps DONE"
          % (sum(1 for _, ok, _ in STEPS if ok), len(STEPS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
