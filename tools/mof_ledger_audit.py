"""Read-only counts from one arm's cost ledger.

Four facts the gate between a phase's paired blocks needs, and nothing else:
how many calls were ``refused`` by the token ceiling, how many ran and came
back ``ok: false`` (``failed``), how many settled without a usage report
(``unreported``), and how many reserved call ids never got a matching
settlement (``outstanding``).

``outstanding`` is not a verdict on its own. While a runner is alive it is an
in-flight call and is normal; it is evidence of loss only once the arm has
completed, so the caller decides -- see ``tools/mof_gate.py``.

Nothing here writes to a ledger. A line that is not valid JSON is counted in
``unreadable`` rather than skipped.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

RESERVED_EVENT = "reserved"
REQUEST_EVENT = "request"
REFUSED_EVENT = "refused"


@dataclass(frozen=True)
class LedgerAudit:
    """What one ledger file contains. Frozen: a reading, not a handle."""

    path: str
    exists: bool
    rows: int = 0
    requests: int = 0
    failed: int = 0
    refused: int = 0
    unreported: int = 0
    unreadable: int = 0
    reserved: int = 0
    settled: int = 0
    outstanding: int = 0
    outstanding_call_ids: tuple[str, ...] = ()
    #: Calls charged nothing while holding a reservation. Observed, not
    #: explained: the billed spend is unknown, and the gate does not fail on it.
    zero_charge: int = 0


def _upper_bound(row: dict) -> int:
    """The reservation a row was held against, or 0 if the row does not say."""
    value = row.get("reservation_upper_bound")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def audit_ledger(path: str | Path) -> LedgerAudit:
    """Count what ``path`` contains.

    A path that is not a file comes back as ``exists=False`` rather than
    raising; a line that is not valid JSON is counted in ``unreadable``.
    """
    ledger = Path(path)
    if not ledger.is_file():
        return LedgerAudit(path=str(ledger), exists=False)

    rows = requests = failed = refused = unreported = unreadable = zero_charge = 0
    reserved_ids: set[str] = set()
    settled_ids: set[str] = set()

    with ledger.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows += 1
            try:
                row = json.loads(line)
            except ValueError:
                unreadable += 1
                continue
            if not isinstance(row, dict):
                unreadable += 1
                continue

            event = row.get("event")
            call_id = row.get("call_id")
            if event == REQUEST_EVENT:
                requests += 1
                if call_id is not None:
                    settled_ids.add(str(call_id))
                if not row.get("ok"):
                    failed += 1
                # A missing field counts as unreported: a row written without it
                # is not evidence that usage was reported.
                if not row.get("usage_reported", False):
                    unreported += 1
                if int(row.get("charged") or 0) == 0 and _upper_bound(row) > 0:
                    zero_charge += 1
            elif event == REFUSED_EVENT:
                # A refused call never took a reservation, so it settles nothing.
                refused += 1
            elif event == RESERVED_EVENT and call_id is not None:
                reserved_ids.add(str(call_id))

    outstanding = sorted(reserved_ids - settled_ids)
    return LedgerAudit(
        path=str(ledger),
        exists=True,
        rows=rows,
        requests=requests,
        failed=failed,
        refused=refused,
        unreported=unreported,
        unreadable=unreadable,
        reserved=len(reserved_ids),
        settled=len(reserved_ids & settled_ids),
        outstanding=len(outstanding),
        outstanding_call_ids=tuple(outstanding),
        zero_charge=zero_charge,
    )


if __name__ == "__main__":  # one-off inspection: python tools/mof_ledger_audit.py <ledger>...
    for target in sys.argv[1:]:
        result = audit_ledger(target)
        print(
            f"{result.path}: exists={result.exists} rows={result.rows} "
            f"requests={result.requests} failed={result.failed} refused={result.refused} "
            f"unreported={result.unreported} zero_charge={result.zero_charge} "
            f"reserved={result.reserved} settled={result.settled} "
            f"outstanding={result.outstanding} unreadable={result.unreadable}"
        )
