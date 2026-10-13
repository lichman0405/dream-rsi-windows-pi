"""The token ceiling is enforced before the call, not audited after it.

`CallBudget` counts discovery and developer sessions; it cannot bound tokens,
because one session may make many requests. `TokenBudget` is the thing that
actually stops a method, so the properties pinned here are the ones a report
that quotes a ceiling depends on. None of these tests make a model call.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dream_rsi.runtime.token_budget import (
    ROLE_DISCOVERY,
    ROLE_POLICY_DEVELOPMENT,
    TokenBudget,
    estimate_tokens,
    spend_from_ledger,
)


def _reserve(budget: TokenBudget, *, role: str = ROLE_DISCOVERY, prompt: str = "x" * 300):
    return budget.reserve(role=role, prompt=prompt, system_prompt="",
                          max_output_bytes=1024)


def test_a_call_that_would_exceed_the_ceiling_is_refused_not_truncated() -> None:
    budget = TokenBudget(limit=20_000)
    assert _reserve(budget) is not None
    budget.settle(
        _reserve(budget), {"total": 19_000, "reported_messages": 1},
        role=ROLE_DISCOVERY, duration_s=1.0, ok=True,
    )
    assert _reserve(budget) is None, "the second call must be refused before it is sent"
    assert budget.refusals == 1


def test_settlement_replaces_the_reservation_with_the_measured_cost() -> None:
    budget = TokenBudget(limit=1_000_000)
    grant = _reserve(budget)
    assert grant is not None
    settlement = budget.settle(
        grant, {"input": 100, "output": 20, "total": 120, "reported_messages": 1},
        role=ROLE_DISCOVERY, duration_s=2.0, ok=True,
    )
    assert settlement.charged == 120
    assert settlement.usage_reported is True
    assert budget.spent == 120, "a reservation is an estimate until usage arrives"


def test_missing_usage_is_charged_conservatively_and_marked_unknown() -> None:
    """The failure mode this guards: an unreported call recorded as zero."""
    budget = TokenBudget(limit=1_000_000)
    grant = _reserve(budget)
    assert grant is not None
    settlement = budget.settle(
        grant, {"total": 0, "reported_messages": 0},
        role=ROLE_DISCOVERY, duration_s=1.0, ok=False, error_type="TIMEOUT",
    )
    assert settlement.usage_reported is False
    assert settlement.actual_tokens is None
    assert settlement.charged == grant.reservation > 0


def test_the_development_reserve_is_held_back_from_discovery_only() -> None:
    budget = TokenBudget(limit=100_000, development_reserve=10_000)
    _reserve(budget)
    budget.settle(
        _reserve(budget), {"total": 89_000, "reported_messages": 1},
        role=ROLE_DISCOVERY, duration_s=1.0, ok=True,
    )
    assert _reserve(budget, role=ROLE_DISCOVERY) is None, (
        "discovery must stop at limit - reserve so the developer still fits"
    )
    assert _reserve(budget, role=ROLE_POLICY_DEVELOPMENT) is not None, (
        "policy development may spend the whole limit"
    )


def test_every_decision_lands_in_the_ledger(tmp_path: Path) -> None:
    ledger = tmp_path / "cost-ledger.jsonl"
    budget = TokenBudget(limit=8_000, ledger_path=ledger,
                         context={"run_id": "r", "method": "A"})
    grant = _reserve(budget)
    budget.settle(grant, {"input": 100, "output": 20, "total": 7_000,
                          "reported_messages": 1},
                  role=ROLE_DISCOVERY, duration_s=0.5, ok=True, provider="p", model="m")
    # The next reservation is ~4.5 k against 1 k of remaining headroom.
    assert _reserve(budget) is None

    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    kinds = [row["event"] for row in rows]
    assert kinds.count("reserved") >= 1
    assert "request" in kinds
    assert "refused" in kinds
    assert budget.refusals == 1
    request = next(row for row in rows if row["event"] == "request")
    for key in ("role", "duration_s", "tokens", "usage_reported", "cost", "charged"):
        assert key in request, f"the ledger row must carry {key}"
    assert request["run_id"] == "r", "the row must say which run and arm it belongs to"
    assert request["cost"]["known"] is False, (
        "money is reported only when the service reported it"
    )


def test_a_resumed_run_inherits_what_it_already_spent(tmp_path: Path) -> None:
    ledger = tmp_path / "cost-ledger.jsonl"
    first = TokenBudget(limit=1_000_000, ledger_path=ledger)
    first.settle(_reserve(first), {"total": 400_000, "reported_messages": 1},
                 role=ROLE_DISCOVERY, duration_s=1.0, ok=True)

    assert spend_from_ledger(ledger) == 400_000
    resumed = TokenBudget(limit=1_000_000, ledger_path=ledger,
                          prior_spent=spend_from_ledger(ledger))
    assert resumed.spent == 400_000, (
        "a restart must not hand the arm a fresh budget"
    )


def test_the_reservation_bound_grows_with_the_request() -> None:
    small = estimate_tokens("hi", "", 1024)
    large = estimate_tokens("x" * 100_000, "", 1024)
    assert large > small
    # The unit is tokens, not bytes: the bound divides the payload by the
    # bytes-per-token floor instead of charging one unit per byte, and adds a
    # fixed allowance for the chat scaffolding the caller cannot measure.
    from dream_rsi.runtime.token_budget import BYTES_PER_TOKEN_FLOOR, PROTOCOL_ALLOWANCE

    assert estimate_tokens("x" * 3_000, "", 0) == (
        3_000 // BYTES_PER_TOKEN_FLOOR + PROTOCOL_ALLOWANCE
    )
    # And it stays an upper bound on the payload rather than an underestimate.
    assert estimate_tokens("x" * 3_000, "", 0) >= 3_000 // BYTES_PER_TOKEN_FLOOR


@pytest.mark.parametrize("bad", [0, -1])
def test_a_non_positive_ceiling_is_rejected(bad: int) -> None:
    with pytest.raises(ValueError):
        TokenBudget(limit=bad)


def test_an_interrupted_call_is_still_charged_on_resume(tmp_path: Path) -> None:
    """A reservation with no settlement is a call the service may have billed.

    This is the double-spend a resume has to avoid: the process died while the
    call was in flight, so no `request` row was ever written, and a resume that
    summed only settled rows would hand the arm back headroom it had already
    used.
    """
    ledger = tmp_path / "cost-ledger.jsonl"
    first = TokenBudget(limit=1_000_000, ledger_path=ledger)
    grant = _reserve(first)
    assert grant is not None
    first.settle(grant, {"total": 30_000, "reported_messages": 1},
                 role=ROLE_DISCOVERY, duration_s=1.0, ok=True)
    orphan = _reserve(first)          # reserved, then the process dies
    assert orphan is not None

    # 30_000 settled plus the orphan's reservation, not 30_000 alone.
    assert spend_from_ledger(ledger) == 30_000 + orphan.reservation
    resumed = TokenBudget(limit=1_000_000, ledger_path=ledger,
                          prior_spent=spend_from_ledger(ledger))
    assert resumed.spent == 30_000 + orphan.reservation


def test_a_phase_ceiling_is_the_sum_of_the_arm_ceilings() -> None:
    from dream_rsi.mof.protocol import PHASES

    formal = PHASES["formal"]
    assert formal.total_token_cap == 3 * 10 * 30_000_000
    assert PHASES["pilot"].total_token_cap == 3 * 4 * 5_000_000
    # The two phases must keep the per-arm ceiling the design fixes, because the
    # pilot's measurements are only a basis for the formal phase if they are
    # measuring the same arms under the same cap.
    assert (formal.methods["C"].W, formal.methods["C"].K1) == (1, 96)
    assert formal.split_seeds == (101, 211, 307, 401, 503)
    assert formal.repeats == (1, 2)
