"""
================================================================================
Script:       canon_r3_gate.py
Version:      1.1.0
Phase:        Stage R — Canon Read 3
Purpose:      The gate: seal verification, data pins, the controls receipt and
              the single-run lock, on the R4/R5/R6 pattern. The only place a
              VerifiedRegistration is minted, and so the only route to world 0.
Entry Point:  No — used by tools/run_canon_r3.py
Dependencies: stdlib; tools/canon_r3_paths.py, tools/canon_r3_world.py
Last Sprint:  Sprint STOCK-2 — Canon read 3: seal and read (2026-11-21)
================================================================================

A read that can be taken without its seal is not sealed. Every refusal below is
a raise, each with its own test, and each test is shown to fail against a mutant
of the check it names.
================================================================================
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import canon_r3_paths as P  # noqa: E402
import canon_r3_world as W  # noqa: E402

VERSION = "1.1.0"

# Every canon_r3 tool, plus the certified harness, is pinned by the seal.
PINNED_CODE_NAMES = (
    "xs_triage.py",
    "canon_r3_paths.py",
    "canon_r3_features.py",
    "canon_r3_world.py",
    "canon_r3_machine.py",
    "canon_r3_gate.py",
    "canon_r3_mutants.py",
    "canon_r3_rehearsal.py",
    "canon_r3_tripwire.py",
    "run_canon_r3.py",
)


class SealMismatch(AssertionError):
    """The registration file's sha256 is not the one --sha passed."""


class DataPinMismatch(AssertionError):
    """A pinned artifact's sha256 is not the one the seal pins."""


class ReceiptMissing(AssertionError):
    """The observed stage was asked for without a usable controls receipt."""


class LockHeld(RuntimeError):
    """Another observed run holds the single-run lock."""


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def pinned_paths() -> dict[str, Path]:
    """Logical name -> path, for everything a seal must pin."""
    tools = P.repo() / "tools"
    out = {n: tools / n for n in PINNED_CODE_NAMES}
    out["banked_panel"] = P.banked_panel()
    out["feature_file"] = P.feature_file()
    # The TRACKED pool, not the scratch manifest: the scratch volume is not
    # under version control and the read's 60 world statistics come from here.
    out["pool_v1"] = P.pool_v1()
    out["controls_v2"] = P.controls_v2()
    out["requirements_lock"] = P.requirements_lock()
    return out


def current_pins() -> dict[str, str]:
    """The sha256 of every pinned artifact that exists right now."""
    out = {}
    for name, p in pinned_paths().items():
        out[name] = sha256_file(p) if p.exists() else None
    return out


def load_registration(path, expected_sha) -> W.VerifiedRegistration:
    """Load the seal and refuse any file whose sha256 is not the one passed in."""
    path = Path(path)
    if not path.exists():
        raise SealMismatch("no registration at %s" % path)
    got = sha256_file(path)
    if got != expected_sha:
        raise SealMismatch(
            "registration sha256 mismatch: file %s has %s, --sha said %s"
            % (path, got, expected_sha))
    data = json.loads(path.read_text())
    return W._mint_registration(path, got, data)


def verify_data_pins(reg: W.VerifiedRegistration) -> dict:
    """
    Compare every sha256 the seal pins with the artifact on disk. Refuse on any
    mismatch, and refuse a seal that pins nothing.
    """
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


def verify_controls_receipt(reg: W.VerifiedRegistration, receipt_path=None) -> dict:
    """The observed stage needs a receipt written against THIS seal."""
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
    """An exclusive lock file, so one observed run cannot overlap another."""

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


class AlreadySpent(RuntimeError):
    """The one read has already been taken."""


def refuse_if_spent(path=None) -> None:
    p = Path(path or P.spent_marker())
    if p.exists():
        raise AlreadySpent(
            "read 3 is already spent: %s" % p)


def write_spent_marker(registration_sha: str, head: str, path=None) -> Path:
    """
    Written atomically AFTER the seal, the pins and the receipt verify and
    BEFORE world 0 is built, so a crash between the two leaves the look spent
    rather than repeatable.
    """
    import datetime
    p = Path(path or P.spent_marker())
    p.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps({
        "look": "CANON-R3",
        "registration_sha256": registration_sha,
        "utc": datetime.datetime.now(datetime.timezone.utc)
                        .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "head": head,
    }, indent=2)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(body)
    os.replace(tmp, p)                      # atomic within the filesystem
    return p


def git_head(repo=None) -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo or P.repo()),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


def open_the_gate(registration, sha, receipt_path=None) -> dict:
    """
    The whole chain, in order: seal, data pins, receipt. Returns the verified
    registration and what it verified. World 0 is reachable only from here.
    """
    reg = load_registration(registration, sha)
    pins = verify_data_pins(reg)
    rec = verify_controls_receipt(reg, receipt_path)
    return {"registration": reg, "registration_sha": reg.sha256,
            "pins": pins, "receipt": rec}
