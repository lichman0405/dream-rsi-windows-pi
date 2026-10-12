"""Per-method token budget with reservations, and a per-request cost ledger.

The experiment's budget unit is a **total token count for one method in one
paired block**, and the rule is a hard ceiling: once a method has spent it, the
next call is refused rather than started. Making that true under `W = 8`
concurrent discovery workers needs two things that counting alone does not give:

1. **Reservation before the call.** The workers all ask before any of them
   settles, so a budget that only summed completed calls would let eight
   in-flight calls jointly overshoot a limit none of them individually crossed.
   :meth:`TokenBudget.reserve` grants or refuses under a lock, and the grant is
   what the worker must hold before it starts.
2. **Settlement to the provider's own numbers.** A reservation is an estimate;
   when the service reports usage, the ledger records *that*, and the estimate
   is discarded. The estimate is a conservative upper bound built from the
   bytes actually sent plus a ceiling on the reply, because no tokenizer is
   available in the worker.

Accounting rules that the reports depend on:

* **Missing usage is unknown, never zero.** A call whose service response
  carried no usage record is settled at its reservation and flagged
  ``usage_reported: false``. Reporting it as 0 would silently underspend the
  budget; reporting the reservation as if measured would overstate precision. It
  is recorded as unknown and charged conservatively.
* **Cost is only reported when the provider reports it.** No price table is
  invented here, so ``cost.known`` is false unless the service supplied an
  amount. The summary then reports tokens and says the money figure is unknown.
* **Every request gets a row**, refused ones included, with the role, the run
  and block it belonged to, its wall-clock duration, and the model identity it
  was sent to. That file is the cost ledger the design asks for.
"""
from __future__ import annotations

import json
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Bytes per token used to bound a request before it is sent. English and source
#: code run around 3.5-4 bytes per token for the usual tokenizers; assuming 3
#: keeps the bound above the true count instead of below it.
BYTES_PER_TOKEN_FLOOR = 3

#: Fixed allowance for chat-template scaffolding, role markers and the reply
#: envelope, none of which the caller can measure from here.
PROTOCOL_ALLOWANCE = 4096

#: Roles a call may have. The policy developer's calls are separated because A
#: has to keep budget back for them while B and C never make one.
ROLE_DISCOVERY = "discovery"
ROLE_POLICY_DEVELOPMENT = "policy_development"


def estimate_tokens(prompt: str, system_prompt: str, max_output_bytes: int) -> int:
    """A conservative upper bound on the tokens one call can consume."""
    sent = len(prompt.encode("utf-8")) + len(system_prompt.encode("utf-8"))
    ceiling = sent + max(0, int(max_output_bytes))
    return math.ceil(ceiling / BYTES_PER_TOKEN_FLOOR) + PROTOCOL_ALLOWANCE


@dataclass(frozen=True)
class Grant:
    call_id: str
    reservation: int


@dataclass(frozen=True)
class Settlement:
    """What one settled call cost, and where it sits in the ledger's order.

    ``seq`` is the join key between a cost row and the work it paid for: the
    discovery backend writes it into the node's generation record, so the
    quality staircase can be plotted against the tokens that actually produced
    each candidate rather than against a clock.
    """

    seq: int
    charged: int
    usage_reported: bool
    actual_tokens: int | None


@dataclass
class TokenBudget:
    """A hard ceiling on total tokens, with reservations and a JSONL ledger."""

    limit: int
    ledger_path: Path | None = None
    context: dict[str, Any] = field(default_factory=dict)
    #: Held back from discovery so that A can still afford its policy
    #: development after a long search. Discovery stops at ``limit - reserve``;
    #: policy development may spend the whole limit.
    development_reserve: int = 0

    def __post_init__(self) -> None:
        if self.limit <= 0:
            raise ValueError("token budget must be positive")
        if self.development_reserve < 0:
            raise ValueError("development reserve must be >= 0")
        if self.development_reserve >= self.limit:
            raise ValueError("development reserve must leave discovery some budget")
        self.limit = int(self.limit)
        self.ledger_path = Path(self.ledger_path) if self.ledger_path else None
        self._lock = threading.Lock()
        self._reserved: dict[str, int] = {}
        self._spent = 0
        self._refusals = 0
        self._calls = 0

    # ------------------------------------------------------------------ reserve

    def reserve(
        self, *, role: str, prompt: str, system_prompt: str, max_output_bytes: int,
    ) -> Grant | None:
        """Claim budget for one call, or return ``None`` to refuse it."""
        upper = estimate_tokens(prompt, system_prompt, max_output_bytes)
        call_id = uuid.uuid4().hex
        with self._lock:
            in_flight = sum(self._reserved.values())
            ceiling = self.limit if role == ROLE_POLICY_DEVELOPMENT else (
                self.limit - self.development_reserve
            )
            if self._spent + in_flight + upper > ceiling:
                self._refusals += 1
                self._write({
                    "event": "refused", "call_id": call_id, "role": role,
                    "reservation_upper_bound": upper,
                    "committed": self._spent + in_flight,
                    "ceiling": ceiling, "limit": self.limit,
                })
                return None
            self._reserved[call_id] = upper
            self._write({
                "event": "reserved", "call_id": call_id, "role": role,
                "reservation_upper_bound": upper,
                "committed": self._spent + in_flight + upper,
                "ceiling": ceiling, "limit": self.limit,
            })
        return Grant(call_id=call_id, reservation=upper)

    # ------------------------------------------------------------------ settle

    def settle(
        self, grant: Grant, usage: dict[str, int], *,
        role: str, duration_s: float, ok: bool, error_type: str | None = None,
        provider: str | None = None, model: str | None = None,
        session_id: str | None = None, provider_cost: dict[str, Any] | None = None,
    ) -> Settlement:
        """Replace the reservation with what the call actually cost."""
        reported = int(usage.get("reported_messages", 0)) > 0
        actual = max(0, int(usage.get("total", 0))) if reported else 0
        with self._lock:
            reserved = self._reserved.pop(grant.call_id, grant.reservation)
            charged = actual if reported else reserved
            self._spent += charged
            self._calls += 1
            seq = self._calls
            self._write({
                "event": "request",
                "seq": seq,
                "call_id": grant.call_id,
                "role": role,
                "ok": ok,
                "error_type": error_type,
                "session_id": session_id,
                "provider": provider,
                "model": model,
                "duration_s": round(float(duration_s), 3),
                "reservation_upper_bound": reserved,
                "usage_reported": reported,
                "tokens": {
                    "input": int(usage.get("input", 0)),
                    "output": int(usage.get("output", 0)),
                    "cache_read": int(usage.get("cache_read", 0)),
                    "cache_write": int(usage.get("cache_write", 0)),
                    "reasoning": int(usage.get("reasoning", 0)),
                    "total": actual if reported else None,
                    "unknown": not reported,
                },
                "cost": _cost_record(provider_cost),
                "charged": charged,
                "spent": self._spent,
                "limit": self.limit,
            })
        return Settlement(seq=seq, charged=charged, usage_reported=reported,
                          actual_tokens=actual if reported else None)

    # ------------------------------------------------------------------ report

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            in_flight = sum(self._reserved.values())
            return {
                "limit": self.limit,
                "development_reserve": self.development_reserve,
                "spent": self._spent,
                "reserved": in_flight,
                "refusals": self._refusals,
                "requests": self._calls,
                "remaining": max(0, self.limit - self._spent - in_flight),
            }

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def refusals(self) -> int:
        with self._lock:
            return self._refusals

    @property
    def requests(self) -> int:
        with self._lock:
            return self._calls

    def exhausted(self, *, role: str = ROLE_DISCOVERY) -> bool:
        with self._lock:
            ceiling = self.limit if role == ROLE_POLICY_DEVELOPMENT else (
                self.limit - self.development_reserve
            )
            return self._spent + sum(self._reserved.values()) >= ceiling

    def to_dict(self) -> dict[str, Any]:
        return {**self.context, **self.snapshot()}

    # ------------------------------------------------------------------ io

    def _write(self, record: dict[str, Any]) -> None:
        if self.ledger_path is None:
            return
        row = {**self.context, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record}
        try:
            self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
            with self.ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except OSError:
            # A ledger that cannot be written must not abort a paid call that is
            # already in flight; the run records the failure through its notes.
            pass


def _cost_record(provider_cost: dict[str, Any] | None) -> dict[str, Any]:
    """Money is only ever reported when the service reported it."""
    if not provider_cost:
        return {"known": False, "currency": None, "amount": None,
                "reason": "provider reported no cost"}
    amount = provider_cost.get("amount")
    currency = provider_cost.get("currency")
    if amount is None or currency is None:
        return {"known": False, "currency": currency, "amount": None,
                "reason": "provider cost payload was incomplete"}
    return {"known": True, "currency": str(currency), "amount": float(amount)}


__all__ = [
    "BYTES_PER_TOKEN_FLOOR", "Grant", "PROTOCOL_ALLOWANCE", "ROLE_DISCOVERY",
    "ROLE_POLICY_DEVELOPMENT", "Settlement", "TokenBudget", "estimate_tokens",
]
