"""
================================================================================
Script:       canon_conf_gate.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The confirmation's gate, in five parts: the seal's sha256, the
              data pins, a controls receipt, a single-run lock, and a spent
              marker written after the gate opens and before world 0 is built.
Entry Point:  No — used by tools/run_canon_conf.py
Dependencies: stdlib; tools/canon_conf_paths.py, tools/canon_conf_pipeline.py
Last Sprint:  Sprint CONF-1b — key 3 struck, the bound measured (2026-11-24)
================================================================================

The only place a VerifiedConfRegistration is minted, and so the only route to
the confirmation's world 0. Read 3's world 0 is not reachable from here at all:
the pipeline refuses it on read 3's cuts regardless of what this gate says.
================================================================================
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_conf_paths as P  # noqa: E402
import canon_conf_pipeline as PL  # noqa: E402

VERSION = "1.0.0"

# every canon_r3 tool the pipeline imports
R3_TOOLS = ("xs_triage.py", "canon_r3_features.py", "canon_r3_machine.py")
# every canon_conf tool
CONF_TOOLS = ("canon_conf_copies.py", "canon_conf_features.py",
              "canon_conf_gate.py", "canon_conf_mutants.py",
              "canon_conf_panel.py", "canon_conf_paths.py",
              "canon_conf_pipeline.py", "canon_conf_rehearsal.py",
              "canon_conf_seal.py", "canon_conf_stopped.py",
              "canon_conf_tripwires.py", "run_canon_conf.py")


class SealMismatch(AssertionError):
    """The registration file's sha256 is not the one --sha passed."""


class DataPinMismatch(AssertionError):
    """A pinned artifact's sha256 is not the one the seal pins."""


class ReceiptMissing(AssertionError):
    """The observed stage was asked for without a usable controls receipt."""


class LockHeld(RuntimeError):
    """Another observed run holds the single-run lock."""


class AlreadySpent(RuntimeError):
    """The one read has already been taken."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def pinned_paths() -> dict[str, Path]:
    tools = P.repo() / "tools"
    out: dict[str, Path] = {}
    for n in R3_TOOLS + CONF_TOOLS:
        out[n] = tools / n
    out["conf_panel"] = P.panel_file()
    out["conf_feature_file"] = P.feature_file()
    out["conf_pool_v1"] = P.pool_v1()
    out["conf_controls_v2"] = P.controls_v2()
    out["requirements_lock"] = P.requirements_lock()
    # provenance: read 3's seal and its result
    out["r3_seal"] = P.r3_seal()
    out["r3_result"] = P.r3_result()
    return out


def current_pins() -> dict[str, str]:
    return {k: (sha256_file(v) if v.exists() else None)
            for k, v in pinned_paths().items()}


def load_registration(path, expected_sha) -> PL.VerifiedConfRegistration:
    path = Path(path)
    if not path.exists():
        raise SealMismatch("no registration at %s" % path)
    got = sha256_file(path)
    if got != expected_sha:
        raise SealMismatch(
            "registration sha256 mismatch: file %s has %s, --sha said %s"
            % (path, got, expected_sha))
    data = json.loads(path.read_text())
    return PL._mint_registration(path, got, data)


def verify_data_pins(reg: PL.VerifiedConfRegistration) -> dict:
    pins = reg.data.get("data_pins")
    if not isinstance(pins, dict) or not pins:
        raise DataPinMismatch("the seal pins no data")
    live = current_pins()
    bad = {}
    for name, want in pins.items():
        if name not in live:
            bad[name] = {"pinned": want, "live": "<not a pinned artifact>"}
        elif live[name] != want:
            bad[name] = {"pinned": want, "live": live[name]}
    if bad:
        raise DataPinMismatch("data pin mismatch: %s" % json.dumps(bad, sort_keys=True))
    return {"pins_checked": len(pins), "pins": dict(pins)}


def verify_controls_receipt(reg: PL.VerifiedConfRegistration, receipt_path=None) -> dict:
    receipt_path = Path(receipt_path or P.controls_receipt())
    if not receipt_path.exists():
        raise ReceiptMissing("no controls receipt at %s" % receipt_path)
    rec = json.loads(receipt_path.read_text())
    got = rec.get("registration_sha")
    if got != reg.sha256:
        raise ReceiptMissing(
            "the receipt was written against registration sha %s but the live "
            "seal is %s" % (got, reg.sha256))
    return {"receipt": str(receipt_path), "registration_sha": got}


class single_run_lock:
    def __init__(self, path=None):
        self.path = Path(path or P.lock_file())
        self.fd = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as e:
            raise LockHeld("the observed-run lock is held: %s" % self.path) from e
        os.write(self.fd, ("pid %d\n" % os.getpid()).encode())
        return self

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        return False


def refuse_if_spent(path=None) -> None:
    p = Path(path or P.spent_marker())
    if p.exists():
        raise AlreadySpent("the confirmation is already spent: %s" % p)


def write_spent_marker(registration_sha: str, head: str, path=None) -> Path:
    p = Path(path or P.spent_marker())
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({
        "look": "CANON-R3C",
        "registration_sha256": registration_sha,
        "utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": head,
    }, indent=2)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(body)
    os.replace(tmp, p)
    return p


def git_head(repo=None) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              cwd=str(repo or P.repo()), capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def open_the_gate(registration, sha, receipt_path=None) -> dict:
    reg = load_registration(registration, sha)
    pins = verify_data_pins(reg)
    rec = verify_controls_receipt(reg, receipt_path)
    return {"registration": reg, "registration_sha": reg.sha256,
            "pins": pins, "receipt": rec}
