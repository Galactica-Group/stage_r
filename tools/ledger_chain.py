"""
================================================================================
Script:       ledger_chain.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      The triage ledger's sha-chain: the one row_sha definition, and a
              whole-file verifier that replays the chain from GENESIS.
Entry Point:  No — imported by tools/verify_ledger.py and tools/run_t_r4.py
Dependencies: stdlib only. Imports nothing from any other repository.
Last Sprint:  Sprint R4-1 — T-R4 crypto trend: seal, controls, half-A read (2026-10-17)
================================================================================

row_sha below is character-identical to the definition carried by the look
runners, so a row banked by any of them replays here. The chain is the artifact;
this module only reads it. Nothing in here appends.

verify replays from the literal string GENESIS and checks two things per row: the
stored prev_row_sha equals the previous row's stored row_sha, and the stored
row_sha equals row_sha(prev, the row's data fields). The first row that fails
either check is named, because a broken chain is only useful if it says where.
================================================================================
"""

import csv
import hashlib
from pathlib import Path

VERSION = "1.0.0"

GENESIS = "GENESIS"


def row_sha(prev, values):
    return hashlib.sha256("\x00".join([prev] + [str(v) for v in values]).encode()).hexdigest()


def read_rows(path):
    """(header, rows) from the ledger csv. Blank trailing lines are dropped."""
    with open(path, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r]
    if not rows:
        raise ValueError("empty ledger: %s" % path)
    return rows[0], rows[1:]


def verify(path):
    """Replay the chain. Returns (ok, n_rows, head, first_bad).

    first_bad is None on a good chain, else
    {row, kind, stored, expected} with row 1-indexed over data rows.
    """
    path = Path(path)
    _, data = read_rows(path)
    prev = GENESIS
    for i, r in enumerate(data, start=1):
        if len(r) < 3:
            return False, len(data), prev, {
                "row": i, "kind": "short_row", "stored": ",".join(r),
                "expected": "at least one data field plus prev_row_sha and row_sha"}
        vals, stored_prev, stored_rs = r[:-2], r[-2], r[-1]
        if stored_prev != prev:
            return False, len(data), prev, {
                "row": i, "kind": "prev_row_sha", "stored": stored_prev,
                "expected": prev}
        calc = row_sha(prev, vals)
        if calc != stored_rs:
            return False, len(data), prev, {
                "row": i, "kind": "row_sha", "stored": stored_rs, "expected": calc}
        prev = stored_rs
    return True, len(data), prev, None
