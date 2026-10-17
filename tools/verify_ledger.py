"""
================================================================================
Script:       verify_ledger.py
Version:      1.0.0
Phase:        Stage R — External Data Evaluation
Purpose:      CLI over ledger_chain.verify: print CHAIN OK or CHAIN BROKEN with
              the row count and the chain head.
Entry Point:  Yes — python tools/verify_ledger.py [reports/triage_ledger.csv]
Dependencies: stdlib, tools/ledger_chain.py. Imports nothing from any other
              repository.
Last Sprint:  Sprint R4-1 — T-R4 crypto trend: seal, controls, half-A read (2026-10-17)
================================================================================

Exit 0 on a good chain, 1 on a broken one, so the verifier can gate a commit.
Read-only: it never writes and never appends.
================================================================================
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ledger_chain as lc  # noqa: E402

VERSION = "1.0.0"

DEFAULT = Path(__file__).resolve().parents[1] / "reports" / "triage_ledger.csv"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else DEFAULT
    ok, n, head, bad = lc.verify(path)
    if ok:
        print("CHAIN OK   rows %d   head %s" % (n, head))
        return 0
    print("CHAIN BROKEN   rows %d   first bad row %d (%s)"
          % (n, bad["row"], bad["kind"]))
    print("  stored   %s" % bad["stored"])
    print("  expected %s" % bad["expected"])
    return 1


if __name__ == "__main__":
    sys.exit(main())
