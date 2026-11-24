"""
================================================================================
Script:       canon_conf_copies.py
Version:      1.0.0
Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)
Purpose:      The declared record of every file this sprint carries as a copy of
              a canon_r3 file, with the exact substitutions that make it one.
Entry Point:  No
Dependencies: stdlib
Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)
================================================================================

R3 of the brief: where a canon_r3 function fixes its window by a constant, carry
a copy with only those constants changed. This module is how "only those" is
proven: a test re-derives each copy by applying SUBSTITUTIONS to the original and
asserts the result equals the committed copy byte for byte. Nothing else can
have changed, because nothing else was applied.
================================================================================
"""

from __future__ import annotations

from pathlib import Path

VERSION = "1.0.0"

_TOOLS = Path(__file__).resolve().parent

# copy name -> (original, copy, [(old, new), ...])
COPIES: dict[str, tuple[str, str, list[tuple[str, str]]]] = {
    "features": (
        "canon_r3_features.py",
        "canon_conf_features.py",
        [
            # the window: formation months run to 2020-11, not 2012-11
            ('LAST_FORMATION_MONTH = "2012-11"', 'LAST_FORMATION_MONTH = "2020-11"'),
            ("N_FORMATION_MONTHS = 168", "N_FORMATION_MONTHS = 264"),
            # the cutoff: the holdout's start, not half A's exclusive max
            ('cut = windows["half_a"]["daily_row_exclusive_max"]',
             'cut = windows["holdout"]["start"]'),
            # the module's own identity
            ('Script:       canon_r3_features.py',
             'Script:       canon_conf_features.py'),
            ('Phase:        Stage R — Canon Read 3',
             'Phase:        Stage R — Canon Read 3 Confirmation (CANON-R3C)'),
            ('import canon_r3_paths as P  # noqa: E402',
             'import canon_conf_paths as P  # noqa: E402'),
            ('Last Sprint:  Sprint STOCK-1 — Canon read 3 built and controlled (2026-11-21)',
             'Last Sprint:  Sprint CONF-1 — the confirmation built and controlled (2026-11-23)'),
        ],
    ),
}


def original_path(name: str) -> Path:
    return _TOOLS / COPIES[name][0]


def copy_path(name: str) -> Path:
    return _TOOLS / COPIES[name][1]


def derive(name: str) -> str:
    """Apply the declared substitutions to the original and return the text."""
    orig, _, subs = COPIES[name]
    text = (_TOOLS / orig).read_text(encoding="utf-8")
    for old, new in subs:
        if text.count(old) != 1:
            raise AssertionError(
                "%s: substitution anchor appears %d times, expected exactly 1: %r"
                % (name, text.count(old), old))
        text = text.replace(old, new)
    return text


def write(name: str) -> Path:
    p = copy_path(name)
    p.write_text(derive(name))
    return p
