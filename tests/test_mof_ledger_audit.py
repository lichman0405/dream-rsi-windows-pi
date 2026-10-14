"""Synthetic-ledger tests for tools/mof_ledger_audit.py.

Run with ``python -m pytest`` from the repository root, which is what puts
``tools`` on the import path. The gate is not imported: it reads argv, opens the
registry and exits at import time. No real ledger, run id or call id is touched;
every call id below is generated fresh.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from tools.mof_ledger_audit import ApprovedFailure, audit_ledger

LIMIT = 30_000_000


def _call_id() -> str:
    return uuid.uuid4().hex


def _reserved(call_id: str, *, role: str = "discovery", upper: int = 23_312) -> dict:
    return {
        "event": "reserved", "call_id": call_id, "role": role,
        "reservation_upper_bound": upper, "committed": upper,
        "ceiling": LIMIT, "limit": LIMIT,
    }


def _refused(call_id: str, *, role: str = "discovery", upper: int = 23_312) -> dict:
    return {
        "event": "refused", "call_id": call_id, "role": role,
        "reservation_upper_bound": upper, "committed": LIMIT,
        "ceiling": LIMIT, "limit": LIMIT,
    }


def _request(
    call_id: str,
    *,
    seq: int = 1,
    role: str = "discovery",
    ok: bool = True,
    error_type: str | None = None,
    charged: int = 1_000,
    reported: bool = True,
    total: int | None = 1_000,
    upper: int = 23_312,
    duration_s: float = 12.5,
    **context,
) -> dict:
    row = {
        "event": "request", "seq": seq, "call_id": call_id, "role": role,
        "ok": ok, "error_type": error_type, "duration_s": duration_s,
        "reservation_upper_bound": upper, "usage_reported": reported,
        "tokens": {
            "input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
            "reasoning": 0, "unknown": not reported,
            "total": total if reported else None,
        },
        "charged": charged, "spent": 0, "limit": LIMIT,
    }
    # The ledger also writes the run it belongs to on every row; callers pass
    # those through ``context`` so a test can vary them.
    row.update(context)
    return row


def _write(tmp_path: Path, rows: list[dict], name: str = "cost-ledger.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


# ------------------------------------------------------------------ clean baseline


def test_successful_ledger_is_clean(tmp_path):
    first, second = _call_id(), _call_id()
    path = _write(tmp_path, [
        _reserved(first), _request(first, seq=1),
        _reserved(second), _request(second, seq=2, charged=4_321, total=4_321),
    ])
    audit = audit_ledger(path)

    assert audit.exists is True
    assert audit.requests == 2
    assert audit.failed == 0
    assert audit.refused == 0
    assert audit.unreported == 0
    assert audit.zero_charge == 0
    assert audit.unreadable == 0
    assert audit.outstanding == 0
    assert (audit.reserved, audit.settled) == (2, 2)


def test_missing_ledger_is_reported_absent_rather_than_clean(tmp_path):
    # The audit reports absence; turning it into a failure is the gate's job.
    audit = audit_ledger(tmp_path / "never-written.jsonl")

    assert audit.exists is False
    assert audit.requests == 0
    assert audit.outstanding == 0
    assert audit.outstanding_call_ids == ()


# ------------------------------------------------------------------ reservations


def test_matched_reservation_is_not_outstanding(tmp_path):
    call = _call_id()
    path = _write(tmp_path, [_reserved(call), _request(call)])
    audit = audit_ledger(path)

    assert audit.outstanding == 0
    assert audit.outstanding_call_ids == ()
    assert audit.settled == 1


def test_unmatched_reservation_is_outstanding(tmp_path):
    settled, abandoned = _call_id(), _call_id()
    path = _write(tmp_path, [
        _reserved(settled), _request(settled),
        _reserved(abandoned),  # reserved, never settled
    ])
    audit = audit_ledger(path)

    assert audit.outstanding == 1
    assert audit.outstanding_call_ids == (abandoned,)


# ------------------------------------------------------------------ failures


def test_b_like_failed_request_counts_as_failed(tmp_path):
    # A discovery call that produced output, was cut off, and settled against a
    # zero usage report. Counted; not explained.
    cut_off = _call_id()
    path = _write(tmp_path, [
        _reserved(cut_off),
        _request(cut_off, seq=60, ok=False, error_type="PI_OUTPUT_LIMIT",
                 charged=0, reported=True, total=0, duration_s=55.537),
    ])
    audit = audit_ledger(path)

    assert audit.failed == 1
    assert audit.unreported == 0   # usage was "reported" -- as a total of zero
    assert audit.zero_charge == 1  # charged nothing against a live reservation
    assert audit.outstanding == 0  # the reservation was released


def test_a_like_failed_request_counts_as_failed(tmp_path):
    # A policy-developer call that failed with no usage report: charged at its
    # reservation rather than at zero.
    call = _call_id()
    path = _write(tmp_path, [
        _reserved(call, role="policy_development", upper=889_000),
        _request(call, seq=7, role="policy_development", ok=False,
                 error_type="PI_AGENT_ERROR", charged=889_000, reported=False,
                 total=None, upper=889_000),
    ])
    audit = audit_ledger(path)

    assert audit.failed == 1
    assert audit.unreported == 1
    assert audit.zero_charge == 0  # it was charged its reservation
    assert audit.outstanding == 0


def test_refusal_is_counted(tmp_path):
    path = _write(tmp_path, [_refused(_call_id())])
    audit = audit_ledger(path)

    assert audit.refused == 1
    assert audit.failed == 0
    assert audit.unreported == 0
    # A refusal never took a reservation, so it cannot leave one outstanding.
    assert audit.reserved == 0
    assert audit.outstanding == 0


def test_request_without_reported_usage_is_counted(tmp_path):
    call = _call_id()
    path = _write(tmp_path, [
        _reserved(call),
        _request(call, reported=False, total=None, charged=23_312, upper=23_312),
    ])
    audit = audit_ledger(path)

    assert audit.unreported == 1
    assert audit.failed == 0
    assert audit.zero_charge == 0  # charged the reservation, so not free


def test_unreadable_rows_are_counted_not_skipped(tmp_path):
    # The well-formed row keeps its usage report, so the only defect here is the
    # corrupt line: it must be counted, not skipped, and must not smear onto the
    # request next to it.
    call = _call_id()
    path = tmp_path / "cost-ledger.jsonl"
    path.write_text(
        json.dumps(_request(call)) + "\n" + "not json\n",
        encoding="utf-8",
    )
    audit = audit_ledger(path)

    assert audit.rows == 2
    assert audit.unreadable == 1
    assert audit.requests == 1
    assert audit.unreported == 0


def test_missing_usage_reported_field_counts_as_unreported(tmp_path):
    # A row that does not carry the field at all is not evidence that usage was
    # reported, so the count is fail-closed.
    call = _call_id()
    row = _request(call)
    row.pop("usage_reported")
    path = _write(tmp_path, [_reserved(call), row])
    audit = audit_ledger(path)

    assert audit.requests == 1
    assert audit.unreported == 1
    assert audit.failed == 0
    assert audit.outstanding == 0


def test_refusal_does_not_close_a_reserved_call(tmp_path):
    # Synthetic and deliberately malformed: the writer never issues a refusal for
    # a call id it has already reserved. Only a request settlement closes a
    # reservation, so a reused id must not be read as settled.
    call = _call_id()
    path = _write(tmp_path, [_reserved(call), _refused(call)])
    audit = audit_ledger(path)

    assert audit.refused == 1
    assert audit.reserved == 1
    assert audit.settled == 0
    assert audit.outstanding == 1
    assert audit.outstanding_call_ids == (call,)


# ------------------------------------------------- the single approved failure
# The tiny identity metadata of the one failure the human user approved. No
# program, label, credential or session metadata, and nothing here is read from
# a ledger or an approval file at test time.
APPROVED_RUN_ID = "mof-ab-s101-r1-B"
APPROVED_SHA256 = "330566cf47932d61160173329075d5ebe50ea250b0710432466545df56f17dc9"
APPROVED_CALL_ID = "229d0d5d413b4ec988dfcaf79b1d029c"
APPROVED_SEQ = 60
APPROVED_RESERVE = 23_312
#: The run-scoped fields the ledger writes on every row. A row that disagrees
#: with any of them carries a different identity.
APPROVED_CONTEXT = {
    "run_id": APPROVED_RUN_ID, "phase": "ab", "method": "B",
    "split_seed": 101, "repeat": 1,
}


def _approval(
    *,
    run_id: str = APPROVED_RUN_ID,
    sha256: str = APPROVED_SHA256,
) -> ApprovedFailure:
    return ApprovedFailure(sha256=sha256, run_id=run_id)


def _approved_pair(
    call_id: str = APPROVED_CALL_ID,
    *,
    seq: int = APPROVED_SEQ,
    error_type: str = "PI_OUTPUT_LIMIT",
    charged: int = 0,
    total: int = 0,
    upper: int = APPROVED_RESERVE,
    **context,
) -> list[dict]:
    """The reserve and settlement of one failed discovery call, as written."""
    return [
        _reserved(call_id, upper=upper),
        _request(call_id, seq=seq, ok=False, error_type=error_type, charged=charged,
                 reported=True, total=total, upper=upper, duration_s=55.537,
                 **{**APPROVED_CONTEXT, **context}),
    ]


def _write_run(tmp_path: Path, run_id: str, rows: list[dict]) -> Path:
    """A ledger filed under its run id, the way the runner writes it."""
    directory = tmp_path / run_id
    directory.mkdir()
    path = directory / "cost-ledger.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_approved_failure_stays_strict_without_an_approval(tmp_path):
    # The default for every existing caller and every other phase.
    path = _write_run(tmp_path, APPROVED_RUN_ID, _approved_pair())
    audit = audit_ledger(path)

    assert audit.failed == 1
    assert audit.approved_failed == 0
    assert audit.conservative_unknown_reserve == 0


def test_exact_approved_failure_is_counted_separately(tmp_path):
    path = _write_run(tmp_path, APPROVED_RUN_ID, _approved_pair())
    audit = audit_ledger(path, approved=_approval())

    assert audit.failed == 1  # still counted: an approval hides nothing
    assert audit.approved_failed == 1
    assert audit.conservative_unknown_reserve == APPROVED_RESERVE
    assert audit.zero_charge == 1
    assert audit.outstanding == 0


def test_another_call_id_is_not_waived(tmp_path):
    rows = _approved_pair("0123456789abcdef0123456789abcdef")
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)
    assert audit.conservative_unknown_reserve == 0


def test_another_sequence_is_not_waived(tmp_path):
    rows = _approved_pair(seq=APPROVED_SEQ + 1)
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_row_from_another_run_is_not_waived(tmp_path):
    # The file is the approved run's, so only the row's own run_id differs. The
    # approval's scope is never projected over the row to paper that over.
    rows = _approved_pair(run_id="mof-ab-s211-r1-B")
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)
    assert audit.conservative_unknown_reserve == 0


def test_row_from_another_phase_is_not_waived(tmp_path):
    rows = _approved_pair(phase="formal2")
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_row_for_another_method_is_not_waived(tmp_path):
    rows = _approved_pair(method="A")
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_row_from_another_split_seed_is_not_waived(tmp_path):
    rows = _approved_pair(split_seed=211)
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_row_from_another_repeat_is_not_waived(tmp_path):
    rows = _approved_pair(repeat=2)
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_ledger_of_another_run_cannot_use_the_approval(tmp_path):
    # The rows are the approved identity's; the file is not in the approved
    # run's own directory, so the approval does not reach it.
    other = _write_run(tmp_path, "mof-ab-s211-r1-B", _approved_pair())
    audit = audit_ledger(other, approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)
    assert audit.conservative_unknown_reserve == 0


def test_another_error_type_is_not_waived(tmp_path):
    rows = _approved_pair(error_type="PI_AGENT_ERROR")
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_changed_token_report_is_not_waived(tmp_path):
    # Same call, same sequence, different reported total: the identity the
    # approval was taken over no longer holds.
    rows = _approved_pair(total=7)
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)
    assert audit.conservative_unknown_reserve == 0


def test_changed_charge_is_not_waived(tmp_path):
    rows = _approved_pair(charged=1)
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert (audit.failed, audit.approved_failed) == (1, 0)


def test_second_identical_failed_row_is_not_covered(tmp_path):
    rows = _approved_pair() + _approved_pair()
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert audit.failed == 2          # both copies are failures
    assert audit.approved_failed == 1  # at most one waiver
    assert audit.conservative_unknown_reserve == APPROVED_RESERVE
    assert audit.outstanding == 0


def test_removed_approved_failure_leaves_nothing_to_waive(tmp_path):
    # The gate requires the approved failure to still be there, so a ledger with
    # the row taken out comes back with approved_failed == 0 and fails there.
    call = _call_id()
    path = _write_run(tmp_path, APPROVED_RUN_ID, [_reserved(call), _request(call)])
    audit = audit_ledger(path, approved=_approval())

    assert audit.failed == 0
    assert audit.approved_failed == 0
    assert audit.conservative_unknown_reserve == 0


def test_conservative_reserve_is_kept_apart_from_recorded_spend(tmp_path):
    other = _call_id()
    rows = [*_approved_pair(),
        _reserved(other, upper=4_321),
        _request(other, seq=APPROVED_SEQ + 1, charged=4_321, total=4_321, upper=4_321),
    ]
    audit = audit_ledger(_write_run(tmp_path, APPROVED_RUN_ID, rows), approved=_approval())

    assert audit.approved_failed == 1
    assert audit.conservative_unknown_reserve == APPROVED_RESERVE
    # The reserve is its own field: the audit exposes no total that a caller
    # could mistake for recorded spend with the reserve folded in.
    assert not hasattr(audit, "charged_total")
