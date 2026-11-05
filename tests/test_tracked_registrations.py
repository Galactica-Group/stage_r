"""Tests that the R3, R3b, R4 and R5 registration seals are under version control.

The seals were written to logs/, which is gitignored, so the hashes the triage
ledger commits to were not themselves recoverable from the repository. Sprint
R5-0 copies them to registrations/ as the tracked record, and R5-2 onward seal
to registrations/ directly. These tests bind the tracked copies to the ledger:
for every ledger row whose batch_id names one of those looks, the row's
registration_sha must equal the first sixteen hex characters of the sha256 of
the tracked file for that batch.

T-R5 is sealed with registration_v2.json. The v1 seal and the controls halt it
produced are tracked beside it as the record of a null family that could not
vote, and they are required to parse, but no ledger row was ever written
against v1: no look was spent on it.

Sprint R6-0 adds the T-R5 side of the R4 pattern: the v2 controls receipt and
the half-A report are tracked, and every receipt is bound to the seal tracked
beside it by the seal's own sha256, so a receipt copied from the wrong look
cannot sit silently next to the right seal.

Sprint R6-1 seals T-R6 and HALTS it at its own step 3: the registered issuer
permutation could not retain the 90 per cent the seal demands, so no observed
figure was formed and no look was spent. The T-R6 seal and the halt it produced
are tracked here beside R5's, and T-R6 deliberately does NOT appear in
BATCH_TO_REGISTRATION, because that mapping is for seals that govern a ledger
row and T-R6 governs none. The assertion that protects the no-look-spent claim
is test_a_halted_seal_governs_no_ledger_row, which now covers both halted
seals.

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
    "T-R5": Path("registrations/r5/registration_v2.json"),
}

# Every file that must be tracked and must parse as JSON.
TRACKED_REGISTRATIONS = [
    Path("registrations/r3/registration_v4.json"),
    Path("registrations/r3b/registration_v5.json"),
    Path("registrations/r4/registration_v1.json"),
    Path("registrations/r4/controls_pass.json"),
    Path("registrations/r4/t_r4_halfA_report.json"),
    Path("registrations/r5/registration_v1.json"),
    Path("registrations/r5/controls_halt_v1.json"),
    Path("registrations/r5/registration_v2.json"),
    Path("registrations/r5/controls_pass_v2.json"),
    Path("registrations/r5/t_r5_halfA_report.json"),
    Path("registrations/r6/registration_v1.json"),
    Path("registrations/r6/controls_halt_v1.json"),
]

# A look's receipts -> the seal they must name. The receipt carries the seal's
# full sha256 under the key given; a receipt that names a different seal than
# the one tracked beside it is the drift this mapping catches. The R4 look
# established the shape; T-R5 follows it.
RECEIPT_TO_SEAL = {
    Path("registrations/r4/controls_pass.json"): (
        Path("registrations/r4/registration_v1.json"), "registration_sha"),
    Path("registrations/r4/t_r4_halfA_report.json"): (
        Path("registrations/r4/registration_v1.json"), "registration_sha256"),
    Path("registrations/r5/controls_pass_v2.json"): (
        Path("registrations/r5/registration_v2.json"), "registration_sha"),
    Path("registrations/r5/t_r5_halfA_report.json"): (
        Path("registrations/r5/registration_v2.json"), "registration_sha256"),
}

# A seal that governed no ledger row: v1 halted at its controls, before any
# observed figure was formed.
SEALED_BUT_NEVER_READ = [Path("registrations/r5/registration_v1.json"),
                         Path("registrations/r6/registration_v1.json")]

# A halted seal -> the halt record tracked beside it, and the step it died at.
HALTED = {
    Path("registrations/r5/registration_v1.json"): (
        Path("registrations/r5/controls_halt_v1.json"), "STEP 3"),
    Path("registrations/r6/registration_v1.json"): (
        Path("registrations/r6/controls_halt_v1.json"), "STEP 3"),
}


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


def test_a_halted_seal_governs_no_ledger_row():
    """v1 halted at step 3; nothing may ever have been banked against it."""
    for rel in SEALED_BUT_NEVER_READ:
        sha16 = sha256_hex(REPO / rel)[:16]
        for row in ledger_rows():
            assert row["registration_sha"] != sha16, (
                f"{rel} halted at its controls but ledger row "
                f"{row['batch_id']} was written against it")


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


def test_r5_tracked_receipt_and_half_a_report_are_present():
    """R6-0: the T-R5 controls receipt and half-A report are under git."""
    for rel in (Path("registrations/r5/controls_pass_v2.json"),
                Path("registrations/r5/t_r5_halfA_report.json")):
        path = REPO / rel
        assert path.is_file(), f"missing tracked T-R5 artefact: {rel}"
        json.loads(path.read_text())


def test_every_receipt_names_the_seal_tracked_beside_it():
    checked = 0
    for rel, (seal_rel, key) in RECEIPT_TO_SEAL.items():
        receipt = json.loads((REPO / rel).read_text())
        assert key in receipt, f"{rel} carries no {key}"
        assert receipt[key] == sha256_hex(REPO / seal_rel), (
            f"{rel}: {key} {receipt[key]} != sha256 of {seal_rel}")
        checked += 1
    assert checked == len(RECEIPT_TO_SEAL)


def test_a_receipt_pointing_at_the_wrong_seal_is_caught(tmp_path):
    """Non-vacuity: the binding above fails when the seal named is another look's."""
    r5_receipt = json.loads(
        (REPO / "registrations/r5/controls_pass_v2.json").read_text())
    r4_seal = sha256_hex(REPO / "registrations/r4/registration_v1.json")
    assert r5_receipt["registration_sha"] != r4_seal

    swapped = tmp_path / "swapped.json"
    r5_receipt["registration_sha"] = r4_seal
    swapped.write_text(json.dumps(r5_receipt))
    reloaded = json.loads(swapped.read_text())
    assert reloaded["registration_sha"] != sha256_hex(
        REPO / "registrations/r5/registration_v2.json")


def test_the_half_a_report_names_the_controls_receipt_tracked_beside_it():
    report = json.loads(
        (REPO / "registrations/r5/t_r5_halfA_report.json").read_text())
    assert report["controls_receipt_sha256"] == sha256_hex(
        REPO / "registrations/r5/controls_pass_v2.json")


def test_every_halted_seal_has_its_halt_tracked_beside_it():
    """A halt is the record of a look that was refused, not of one not taken."""
    for seal_rel, (halt_rel, step) in HALTED.items():
        halt = REPO / halt_rel
        assert halt.is_file(), f"missing tracked halt: {halt_rel}"
        h = json.loads(halt.read_text())
        assert h["registration_sha"] == sha256_hex(REPO / seal_rel), (
            f"{halt_rel} names a seal other than {seal_rel}")
        assert step in h["halt"], (halt_rel, h["halt"])


def test_a_halt_records_no_figure_downstream_of_where_it_died():
    """Every figure the halted step never reached must be absent, not zero."""
    h = json.loads(
        (REPO / "registrations/r6/controls_halt_v1.json").read_text())
    for key in ("mde", "known_null_share", "X_calibrated", "detection",
                "null_spread", "known_null_control", "positive_control"):
        assert h[key] is None, (key, h[key])
    assert h["n_events"] >= 5000, "the resolution minimum was cleared"


def test_the_t_r6_halt_says_why_the_null_could_not_be_formed():
    """The shortfall is measured, not asserted: the halt must carry the numbers."""
    h = json.loads(
        (REPO / "registrations/r6/controls_halt_v1.json").read_text())
    d = h["diagnostic"]
    prof, cov = d["retention_profile"], d["coverage"]
    # the scorer is sound: each real event is covered by its own issuer
    assert prof["identity_retention"] == 1.0
    # and no uniform bijection reaches the sealed requirement
    assert prof["required"] == 0.90
    assert prof["reachable"] is False
    assert prof["max"] < prof["required"]
    assert prof["share_at_or_above"]["0.90"] == 0.0
    # the ceiling is structural: the exact coverage matrix agrees with the draws
    assert abs(cov["event_weighted_expected_retention"] - prof["mean"]) < 1e-3
    assert d["issuer_spans"]["spanning_the_whole_window"] < \
        d["issuer_spans"]["issuers"]


def test_the_t_r6_halt_was_produced_by_an_unmodified_r5_engine():
    """T-R6 reuses R5's scorer; the ledger's own T-R5 row pins its bytes."""
    h = json.loads(
        (REPO / "registrations/r6/controls_halt_v1.json").read_text())
    row = next(r for r in ledger_rows() if r["batch_id"] == "T-R5")
    assert h["r5_engine_sha256"][:16] == row["harness_sha"]


def test_the_ledger_carries_no_t_r6_row():
    assert all(r["batch_id"] != "T-R6" for r in ledger_rows())
