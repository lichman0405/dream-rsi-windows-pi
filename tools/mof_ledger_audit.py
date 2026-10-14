"""Read-only counts from one arm's cost ledger.

Five facts the gate between a phase's paired blocks needs: how many calls were
``refused`` by the token ceiling, how many ran and came back ``ok: false``
(``failed``), how many settled without a usage report (``unreported``), how many
reserved call ids never got a matching settlement (``outstanding``), and -- only
when a caller hands in a recorded approval -- how many of those failures a human
has approved by exact identity.

An approved failure stays a failed call. ``failed`` counts it and
``approved_failed`` counts the subset a waiver covers; an approval never reduces
``failed``. The upper bounds of approved calls are summed into
``conservative_unknown_reserve``, which is kept apart from recorded spend
because the provider's actual billing for those calls is unknown.

``outstanding`` is not a verdict on its own. While a runner is alive it is an
in-flight call and is normal; it is evidence of loss only once the arm has
completed, so the caller decides -- see ``tools/mof_gate.py``.

Nothing here writes to a ledger. A line that is not valid JSON is counted in
``unreadable`` rather than skipped.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

RESERVED_EVENT = "reserved"
REQUEST_EVENT = "request"
REFUSED_EVENT = "refused"

#: The fields an approval identity is made of -- exactly the fields the recorded
#: approval lists. Every one of them is read from the ledger row itself; the
#: fingerprint is over these keys and no others, so a row differing in any of
#: them cannot match.
IDENTITY_FIELDS = (
    "event", "run_id", "phase", "method", "split_seed", "repeat", "call_id",
    "seq", "role", "ok", "error_type", "charged", "usage_reported",
    "reservation_upper_bound", "tokens",
)


@dataclass(frozen=True)
class ApprovedFailure:
    """One failed request a human has approved, named by exact identity.

    ``sha256`` is the fingerprint of the approval's identity object and
    ``run_id`` is the run it was taken over: the ledger handed to the audit must
    be that run's own file. Matching is deliberately narrow -- the row must
    reproduce the fingerprint from its own fields, and at most one row per
    ledger is covered.
    """

    sha256: str
    run_id: str


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
    #: Failed calls the supplied approval covers. Always a subset of ``failed``.
    approved_failed: int = 0
    #: Upper bounds of those calls. A conservative stand-in, never recorded or
    #: billed spend, and never added to what the ledger says was charged.
    conservative_unknown_reserve: int = 0


def _upper_bound(row: dict) -> int:
    """The reservation a row was held against, or 0 if the row does not say."""
    value = row.get("reservation_upper_bound")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def identity_sha256(identity: dict) -> str:
    """SHA-256 over the canonical JSON of an approval identity."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_identity(row: dict) -> dict:
    """The identity a ledger row carries, taken from the row itself.

    Nothing is filled in from the approval, not even the run-scoped fields the
    ledger also writes: a row whose run, phase, method, seed or repeat differs
    is a different identity, and projecting the approval's scope over it would
    hide exactly that.
    """
    return {key: row.get(key) for key in IDENTITY_FIELDS}


def _matches_approval(row: dict, approved: ApprovedFailure) -> bool:
    return identity_sha256(_row_identity(row)) == approved.sha256


def audit_ledger(path: str | Path, *, approved: ApprovedFailure | None = None) -> LedgerAudit:
    """Count what ``path`` contains.

    A path that is not a file comes back as ``exists=False`` rather than
    raising; a line that is not valid JSON is counted in ``unreadable``. With no
    ``approved`` argument every failure counts against the caller, which is how
    every existing caller and every other phase keeps behaving.
    """
    ledger = Path(path)
    if not ledger.is_file():
        return LedgerAudit(path=str(ledger), exists=False)

    if approved is not None and ledger.parent.name != approved.run_id:
        # The approval names the run it was taken over, and a run's ledger is
        # the file inside that run's own directory. Any other file cannot use
        # it, whatever the rows inside happen to say.
        approved = None

    rows = requests = failed = refused = unreported = unreadable = zero_charge = 0
    approved_failed = approved_reserve = 0
    waiver_used = False
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
                    # At most one waiver, and only for a row that reproduces the
                    # approved identity exactly. A second copy of the same failure
                    # is not covered by the same approval.
                    if (approved is not None and not waiver_used
                            and _matches_approval(row, approved)):
                        waiver_used = True
                        approved_failed += 1
                        approved_reserve += _upper_bound(row)
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
        approved_failed=approved_failed,
        conservative_unknown_reserve=approved_reserve,
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
