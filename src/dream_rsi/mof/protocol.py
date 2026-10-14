"""The frozen MOF experiment protocol: what runs, in what order, under what cap.

This module is the machine-readable form of ``docs/experiments/MOF_SCREENING_DESIGN.md``.
Everything a run depends on -- the loop parameters, the method definitions, the
token ceilings, the checkpoint grid, the split seeds, the paired-block order and
the model identity -- is fixed here, hashed, and written into every block's
artifacts. Changing a number changes the digest, which is how a resumed or
re-analysed run can tell that it is not the run it claims to be.

The two phases are *both* pre-registered, and the pilot's job is explicitly not
to produce a headline result:

``pilot``   two paired blocks on the first two development splits. It measures
            feasibility -- session token distribution, history growth, whether a
            complete outer iteration finishes, where truncation lands, batch
            waiting time -- and decides whether the formal phase is affordable
            under the protocol or needs re-freezing first.
``formal``  ten paired blocks (five split seeds x two repeats) at the full
            ceiling. Its result is the one the report leads with.

Method definitions are the design's three Pi arms:

======  =========================  ===================================
Method  Policy                     Role
======  =========================  ===================================
A       learned ``pi_t``           the experimental arm
B       frozen ``incumbent.py``    primary control: is learning worth it
C       frozen ``serial.py``       auxiliary: search order at W = 1
======  =========================  ===================================
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

#: The dataset this protocol is pinned to. Any change is a new protocol.
DATASET = {
    "name": "CRAFTED",
    "version": "2.0.1",
    "doi": "10.5281/zenodo.10120180",
    "archive_sha256": "f29f7b3d6b7d6c51fd113138bc4a811c448d4dcce66ed37fddbfa4141a238dde",
    "condition": {"gas": "CO2", "temperature_K": 298, "pressure_Pa": 100000,
                  "forcefield": "UFF", "charge_scheme": "DDEC"},
    "prepared_dir": "CRAFTED-2.0.1-CO2-298-UFF-DDEC-v1",
}

#: The model, pinned by name. The service does not expose a build id, so the
#: provider+model string plus the recorded Pi version is the whole identity a
#: run can honestly claim.
MODEL = {"provider": "cc-switch-deep-seek", "model": "deepseek-v4-pro"}

#: Extended reasoning, fixed to ``off`` for every call in every arm.
#:
#: This is a *service parameter*, and the design requires it to be recorded and
#: identical across arms, which it is. It was fixed after probe calls on this
#: machine, before any experimental result existed, because the measured spread
#: between settings is the difference between a runnable protocol and an
#: unaffordable one:
#:
#: ===========  ==========  ==============  =============
#: setting      wall clock  total tokens    reasoning tok
#: ===========  ==========  ==============  =============
#: service      626 s       (not recorded)  ~33 000
#: ``low``      527 s       34 107          29 207
#: ``minimal``  642 s       42 844          36 761
#: ``off``       35 s        5 310               0
#: ===========  ==========  ==============  =============
#:
#: The lower settings did not lower anything: only ``off`` stops the model
#: reasoning, and it was also the only one that returned a complete file. All
#: three arms use this same setting, so the comparison the experiment is about
#: -- learned policy against frozen policy -- is unaffected.
THINKING = "off"

#: The score's K, from §3.4 of the design.
K = 10

#: Which implementation of the algorithm produced a phase's numbers.
#:
#: Distinct from the protocol digest on purpose. The digest identifies the
#: *parameters* -- arms, seeds, blocks, budgets -- and those did not change. This
#: identifies the *implementation*, and that did: the first formal attempt wrote
#: one full observation per decision step and its A arm was rejected by the
#: provider for exceeding the context window. Revision 2 stores each node record
#: once and refers to it by id.
#:
#: It is recorded in every phase record rather than folded into the digest,
#: because folding it in would move the digest of phases that have already run;
#: the pilot's 9f5940ed and formal v1's 10d9a668 both still reproduce from this
#: code, and a reviewer can tell the two implementations apart by this field.
#:
#: 1 -- one full observation per decision step (trajectory format 2)
#: 2 -- node records stored once, steps refer to them by id (format 3)
IMPLEMENTATION_REVISION = 2

#: The provider's context window, in tokens, for the model this protocol pins.
#: Recorded because a request that exceeds it is rejected outright -- which is
#: how the first formal A arm died -- so the runner has to know the number to
#: refuse the call itself and record why.
CONTEXT_LIMIT_TOKENS = 1_048_576

#: Seed handed to candidates so their own splitting is deterministic. Fixed, not
#: per-block: a candidate is a program, and giving it a different seed per block
#: would make the paired comparison a comparison of two programs.
CANDIDATE_SEED = 20261012


@dataclass(frozen=True)
class MethodSpec:
    """One arm of the experiment."""

    key: str
    name: str
    incumbent_policy: str
    uses_developer: bool
    W: int
    K1: int
    M: int


@dataclass(frozen=True)
class PhaseSpec:
    """One phase's budget and schedule."""

    key: str
    worlds: int
    token_budget: int
    session_cap: int
    split_seeds: tuple[int, ...]
    repeats: tuple[int, ...]
    checkpoints: tuple[int, ...]
    methods: Mapping[str, MethodSpec]

    @property
    def blocks(self) -> int:
        return len(self.split_seeds) * len(self.repeats)

    @property
    def total_token_cap(self) -> int:
        """The phase ceiling the design states as ``methods x blocks x budget``.

        It is an arithmetic consequence of the per-arm ceilings rather than a
        separate decision, and it is checked anyway: a block that is resumed
        repeatedly after repeated interruptions is the one case where "each arm
        stayed inside its own ceiling" and "the phase stayed inside its total"
        can stop being the same statement.
        """
        return len(self.methods) * self.blocks * self.token_budget


def _methods(*, w: int, k1: int, a_M: int, c_k1: int) -> dict[str, MethodSpec]:
    """The three arms. ``c_k1`` is separate because the design fixes it per phase.

    The design's parameter table gives method C ``K1 = 64`` for the pilot and
    ``96`` for the formal phase -- C is serial, so its depth *is* its budget in a
    way A's and B's is not. Hard-coding the pilot's value here would have run the
    formal phase with the pilot's search depth, which is the kind of silent
    deviation from the frozen protocol that the whole pre-registration exists to
    prevent.
    """
    return {
        "A": MethodSpec(key="A", name="dream_rsi", incumbent_policy="policies/incumbent.py",
                        uses_developer=True, W=w, K1=k1, M=a_M),
        "B": MethodSpec(key="B", name="frozen_initial", incumbent_policy="policies/incumbent.py",
                        uses_developer=False, W=w, K1=k1, M=1),
        "C": MethodSpec(key="C", name="frozen_serial", incumbent_policy="policies/serial.py",
                        uses_developer=False, W=1, K1=c_k1, M=1),
    }


#: Development splits, in the order the design fixes them (`101, 211, 307, 401,
#: 503`). The pilot uses the first two.
SPLIT_SEEDS: tuple[int, ...] = (101, 211, 307, 401, 503)

PHASES: Mapping[str, PhaseSpec] = {
    "pilot": PhaseSpec(
        key="pilot", worlds=2,
        token_budget=5_000_000, session_cap=128,
        split_seeds=SPLIT_SEEDS[:2], repeats=(1, 2),
        checkpoints=(500_000, 1_000_000, 2_000_000, 5_000_000),
        methods=_methods(w=8, k1=8, a_M=2, c_k1=64),
    ),
    "formal": PhaseSpec(
        key="formal", worlds=8,
        token_budget=30_000_000, session_cap=768,
        split_seeds=SPLIT_SEEDS, repeats=(1, 2),
        checkpoints=(1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000),
        methods=_methods(w=8, k1=12, a_M=3, c_k1=96),
    ),
    # The formal phase as re-run under the fixed feedback encoding. Same arms,
    # seeds, blocks, checkpoints and ceilings as "formal" -- the parameters did
    # not change -- but its own key, so that run ids, the output directory, the
    # cost ledgers, the pause marker, the unseal pass and the report all carry
    # the v2 identity and cannot be confused with, or merge into, the v1 records
    # this phase replaces. v1 is kept whole and reported separately.
    "formal2": PhaseSpec(
        key="formal2", worlds=8,
        token_budget=30_000_000, session_cap=768,
        split_seeds=SPLIT_SEEDS, repeats=(1, 2),
        checkpoints=(1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000),
        methods=_methods(w=8, k1=12, a_M=3, c_k1=96),
    ),
}

#: Phases whose numbers are the experiment's result. ``pilot`` and ``formal``
#: are what was actually run and are never overwritten; ``formal2`` is the
#: continuation under the fixed feedback encoding.
PHASE_NAMES: tuple[str, ...] = tuple(PHASES)

#: Replay sweep and penalty, identical in both phases and every arm.
RHO_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
LIVE_RHO = 1.0
LAMBDA_WEIGHT = 0.1

#: Per-attempt ceilings, identical in both phases and every arm.
AGENT_TIMEOUT_S = 900.0
EVALUATOR_TIMEOUT_S = 120.0
POLICY_SELECT_TIMEOUT_S = 5.0
AGENT_MEMORY_MB = 4096
EVALUATOR_MEMORY_MB = 2048
POLICY_MEMORY_MB = 256

#: Fraction of a method's budget method A keeps back for policy development.
#: The design requires the reserve; the size is this protocol's choice, and it
#: is recorded so the report can say what it was.
DEVELOPMENT_RESERVE_FRACTION = 0.10

#: The design's practical threshold for calling A better than B, in S units.
PRACTICAL_THRESHOLD = 0.05

#: How many predetermined reshuffles of the final test pool's row order are used
#: to check that a frozen candidate returns the same materials regardless of
#: layout. Diagnostic only; never used to pick a more flattering result.
ROW_ORDER_PERMUTATIONS = 3


@dataclass(frozen=True)
class BlockPlan:
    """One paired block, one arm. The unit of execution and of statistics."""

    phase: str
    split_seed: int
    repeat: int
    method: str
    order: int

    @property
    def block_id(self) -> str:
        return f"{self.phase}-s{self.split_seed}-r{self.repeat}"

    @property
    def run_id(self) -> str:
        return f"mof-{self.phase}-s{self.split_seed}-r{self.repeat}-{self.method}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "block_id": self.block_id, "run_id": self.run_id}


def method_order(phase: str, split_seed: int, repeat: int) -> tuple[str, ...]:
    """A fixed, pre-registered execution order for one block's three arms.

    The design asks for the arm order inside a block to be randomised in advance
    to spread service-side drift. "In advance" has to mean *derivable*: the order
    is a hash of the block's identity, so it is not chosen after seeing anything,
    it is reproducible by a reviewer, and it is recorded in the block record.
    """
    digest = hashlib.sha256(
        f"dream-rsi-mof-order|{phase}|{split_seed}|{repeat}".encode("utf-8")
    ).digest()
    keys = ["A", "B", "C"]
    ordered = sorted(range(3), key=lambda index: (digest[index], keys[index]))
    return tuple(keys[index] for index in ordered)


def block_plans(phase: str) -> list[BlockPlan]:
    spec = PHASES[phase]
    plans: list[BlockPlan] = []
    for split_seed in spec.split_seeds:
        for repeat in spec.repeats:
            for order, method in enumerate(method_order(phase, split_seed, repeat)):
                plans.append(BlockPlan(phase=phase, split_seed=split_seed, repeat=repeat,
                                       method=method, order=order))
    return plans


@dataclass
class Protocol:
    """Everything above, plus the identity of the run that will use it."""

    phase: str
    prepared_root: str
    pi_version: str | None = None
    notes: Sequence[str] = field(default_factory=tuple)
    #: Acceptance-only replacement for the registered phase spec. A live phase
    #: never sets it; the digest covers whatever spec is actually used, so a
    #: shrunk smoke block can never be mistaken for a real one.
    spec_override: PhaseSpec | None = None

    def __post_init__(self) -> None:
        if self.phase not in PHASES:
            raise ValueError(f"unknown phase {self.phase!r}; known: {sorted(PHASES)}")

    @property
    def spec(self) -> PhaseSpec:
        return self.spec_override or PHASES[self.phase]

    def to_dict(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "schema": "dream-rsi-mof-protocol-v1",
            "phase": self.phase,
            # `total_token_cap` is deliberately absent: it is the product of
            # values already in here, so it carries no information the digest
            # does not have, and putting it in would move the digest of a phase
            # that has already run. A frozen digest that can be changed by
            # restating a derived number is not frozen.
            "phase_spec": {
                "worlds": spec.worlds, "token_budget": spec.token_budget,
                "session_cap": spec.session_cap, "split_seeds": list(spec.split_seeds),
                "repeats": list(spec.repeats), "checkpoints": list(spec.checkpoints),
                "blocks": spec.blocks,
                "methods": {key: asdict(m) for key, m in sorted(spec.methods.items())},
            },
            "dataset": DATASET,
            "model": MODEL,
            "thinking": THINKING,
            # `context_limit_tokens` is deliberately absent, for the same reason
            # as `total_token_cap`: it is a property of the pinned model, which
            # this dict already names, so it adds no information the digest does
            # not have -- and putting it in moved the digest of a phase that has
            # already run. The number is recorded in the module and in the
            # phase record instead.
            "pi_version": self.pi_version,
            "k": K,
            "candidate_seed": CANDIDATE_SEED,
            "rho_grid": list(RHO_GRID),
            "live_rho": LIVE_RHO,
            "lambda_weight": LAMBDA_WEIGHT,
            "timeouts": {
                "agent_s": AGENT_TIMEOUT_S, "evaluator_s": EVALUATOR_TIMEOUT_S,
                "policy_select_s": POLICY_SELECT_TIMEOUT_S,
                "agent_memory_mb": AGENT_MEMORY_MB,
                "evaluator_memory_mb": EVALUATOR_MEMORY_MB,
                "policy_memory_mb": POLICY_MEMORY_MB,
            },
            "development_reserve_fraction": DEVELOPMENT_RESERVE_FRACTION,
            "practical_threshold": PRACTICAL_THRESHOLD,
            "row_order_permutations": ROW_ORDER_PERMUTATIONS,
            "prepared_root": self.prepared_root,
            "notes": list(self.notes),
        }

    @property
    def digest(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def development_reserve(spec: PhaseSpec) -> int:
    """Tokens method A holds back so a long search cannot starve its developer."""
    return int(spec.token_budget * DEVELOPMENT_RESERVE_FRACTION)


__all__ = [
    "AGENT_MEMORY_MB", "AGENT_TIMEOUT_S", "BlockPlan", "CANDIDATE_SEED", "DATASET",
    "DEVELOPMENT_RESERVE_FRACTION", "EVALUATOR_MEMORY_MB", "EVALUATOR_TIMEOUT_S",
    "K", "LAMBDA_WEIGHT", "LIVE_RHO", "MODEL", "MethodSpec", "PHASES",
    "POLICY_MEMORY_MB", "POLICY_SELECT_TIMEOUT_S", "PRACTICAL_THRESHOLD",
    "CONTEXT_LIMIT_TOKENS", "IMPLEMENTATION_REVISION", "PHASE_NAMES", "PhaseSpec",
    "Protocol", "RHO_GRID",
    "ROW_ORDER_PERMUTATIONS", "SPLIT_SEEDS",
    "THINKING",
    "block_plans", "development_reserve", "method_order",
]
