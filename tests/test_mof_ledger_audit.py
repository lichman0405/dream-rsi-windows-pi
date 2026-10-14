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

from tools.mof_ledger_audit import audit_ledger

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
) -> dict:
    return {
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
