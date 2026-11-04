"""Tests that the R3, R3b and R4 registration seals are under version control.

The seals were written to logs/, which is gitignored, so the hashes the triage
ledger commits to were not themselves recoverable from the repository. Sprint
R5-0 copies them to registrations/ as the tracked record. These tests bind the
tracked copies to the ledger: for every ledger row whose batch_id names one of
those three looks, the row's registration_sha must equal the first sixteen hex
characters of the sha256 of the tracked file for that batch.

The mutation this battery is built to catch is a tracked copy that drifts from
the sealed bytes the ledger row was written against -- a flipped byte, a
re-serialised JSON, or a copy taken from the wrong look.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "reports" / "triage_ledger.csv"

# batch_id in the ledger -> the tracked registration that batch was sealed with.
BATCH_TO_REGISTRATION = {
    "T-R3": Path("registrations/r3/registration_v4.json"),
    "T-R3b": Path("registrations/r3b/registration_v5.json"),
    "T-R4": Path("registrations/r4/registration_v1.json"),
}

# Every file that must be tracked and must parse as JSON.
TRACKED_REGISTRATIONS = [
    Path("registrations/r3/registration_v4.json"),
    Path("registrations/r3b/registration_v5.json"),
    Path("registrations/r4/registration_v1.json"),
    Path("registrations/r4/controls_pass.json"),
    Path("registrations/r4/t_r4_halfA_report.json"),
]


def sha256_hex(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def ledger_rows() -> list[dict[str, str]]:
    with LEDGER.open(newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_tracked_registration_exists_and_parses_as_json():
    for rel in TRACKED_REGISTRATIONS:
        path = REPO / rel
        assert path.is_file(), f"missing tracked registration: {rel}"
        json.loads(path.read_text())


def test_mapped_batches_are_all_present_in_the_ledger():
    seen = {row["batch_id"] for row in ledger_rows()}
    for batch in BATCH_TO_REGISTRATION:
        assert batch in seen, f"ledger has no row for {batch}"


def test_ledger_registration_sha_matches_the_tracked_copy():
    checked = 0
    for row in ledger_rows():
        rel = BATCH_TO_REGISTRATION.get(row["batch_id"])
        if rel is None:
            continue
        expected = sha256_hex(REPO / rel)[:16]
        assert row["registration_sha"] == expected, (
            f"{row['batch_id']}: ledger {row['registration_sha']} != "
            f"tracked {rel} {expected}"
        )
        checked += 1
    assert checked == len(BATCH_TO_REGISTRATION)


def test_a_byte_flip_breaks_the_binding(tmp_path):
    """Non-vacuity: the assertion above fails on a one-byte change."""
    rel = BATCH_TO_REGISTRATION["T-R4"]
    original = (REPO / rel).read_bytes()
    row = next(r for r in ledger_rows() if r["batch_id"] == "T-R4")
    assert row["registration_sha"] == hashlib.sha256(original).hexdigest()[:16]

    flipped = tmp_path / "flipped.json"
    mutated = bytearray(original)
    mutated[0] = mutated[0] ^ 0x01
    flipped.write_bytes(bytes(mutated))
    assert sha256_hex(flipped)[:16] != row["registration_sha"]
