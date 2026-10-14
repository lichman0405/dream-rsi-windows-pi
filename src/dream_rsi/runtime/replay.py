"""Offline historical replay (§3.3).

A replay rolls a policy over an *already sealed* world. It never executes a
discovery agent and never runs an evaluator: every score it sees was produced
during the original online rollout and is read back from the immutable world.

    discovery_agent_calls == 0
    evaluator_calls == 0

Those two counters are part of every :class:`ReplayEpisode` and are asserted in
the test suite. This module deliberately has no reference to any agent or
evaluator object, so there is nothing it *could* call.

Information barrier: the policy is handed a :class:`PrefixObservation` built
only from the revealed subtree. `SealedWorld` stays in the controller's memory
and is never passed across the policy boundary (which is a separate process in
production — see ``policy/runner.py``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..domain.errors import IntegrityError
from ..domain.models import Node, ReplayEpisode, StopReason, new_id
from ..policy.protocol import PrefixObservation
from ..policy.validation import validate_batch
from .trajectory import CURRENT_FORMAT, encode_decisions, is_supported_format, iter_batches

PolicyFn = Callable[[PrefixObservation, float], Sequence[str]]


class SealedWorldLike(Protocol):
    """Structural view of the immutable world a replay rolls over.

    Declared as a Protocol rather than an import so that ``runtime.replay``
    does not depend on ``domain.tree``; ``SealedWorld`` satisfies it exactly.
    """

    # Read-only on purpose: replay reads a sealed world and never mutates it,
    # so a world exposing these as properties must still satisfy the protocol.
    @property
    def world_id(self) -> str: ...
    @property
    def root_id(self) -> str: ...
    @property
    def s_best(self) -> float: ...

    def node_ids(self) -> Sequence[str]: ...
    def node(self, node_id: str) -> Node: ...
    def world_digest(self) -> str: ...
    def observation(
        self, revealed: Collection[str], completed_rounds: int, max_parallelism: int
    ) -> PrefixObservation: ...
    def next_child(self, node_id: str, revealed: Collection[str]) -> str | None: ...

@dataclass(frozen=True)
class ReplayCounters:
    """Instrumentation proving replay performed no external calls."""

    discovery_agent_calls: int = 0
    evaluator_calls: int = 0
    policy_select_calls: int = 0


def _episode_digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def replay_world(
    world: SealedWorldLike,
    policy: PolicyFn,
    *,
    policy_version_id: str,
    rho: float,
    max_parallelism: int,
    max_rounds: int,
) -> ReplayEpisode:
    """Roll ``policy`` over the revealed prefix of ``world`` for one ``rho``.

    ``world`` is a :class:`dream_rsi.domain.tree.SealedWorld`, seen through
    :class:`SealedWorldLike`.
    """
    if max_parallelism < 1:
        raise ValueError("max_parallelism (W) must be >= 1")
    if max_rounds < 0:
        raise ValueError("max_rounds (K2) must be >= 0")

    root_id: str = world.root_id
    total = len(world.node_ids())
    revealed: set[str] = {root_id}
    action_trace: list[tuple[str, ...]] = []
    rounds = 0
    no_progress = 0
    stop_reason = StopReason.MAX_ROUNDS  # the paper's K2 exhaustion
    select_calls = 0
    rich_steps: list[dict[str, object]] = []

    while rounds < max_rounds:
        if len(revealed) == total:
            stop_reason = StopReason.FULLY_REVEALED
            break
        observation = world.observation(revealed, rounds, max_parallelism)
        raw = policy(observation, rho)
        select_calls += 1
        batch = validate_batch(
            raw,
            legal_actions=observation.legal_actions,
            observed_ids=observation.node_ids,
            max_parallelism=max_parallelism,
        )
        if not batch:
            # The stopping decision is a step like any other and keeps its full
            # observation: a policy being asked to learn when to stop needs to
            # see what it stopped on.
            rich_steps.append({"observation": observation.to_dict(),
                               "batch": [], "revealed": []})
            stop_reason = StopReason.STOP_EMPTY_BATCH
            break
        # Simultaneous batch: every child is resolved against the SAME pre-round
        # revealed set, so batch members cannot see each other's reveals.
        children = [world.next_child(node_id, revealed) for node_id in batch]
        action_trace.append(batch)
        rounds += 1  # per §3 every NONEMPTY batch counts, including no-progress
        fresh = [c for c in children if c is not None]
        rich_steps.append({"observation": observation.to_dict(),
                           "batch": list(batch),
                           "revealed": [world.observation({c}, rounds, max_parallelism).nodes[0].to_dict()
                                        for c in fresh]})
        if not fresh:
            no_progress += 1
        revealed.update(fresh)
    else:
        if len(revealed) == total:
            stop_reason = StopReason.FULLY_REVEALED

    # Node records are written once and the steps refer to them by id; see
    # runtime/trajectory.py for why, and decode_decisions for the exact inverse.
    decisions = list(encode_decisions(rich_steps))

    revealed_order = tuple(nid for nid in world.node_ids() if nid in revealed)
    non_root = revealed_order[1:]
    best_score = max((world.node(nid).score for nid in non_root), default=0.0)
    s_best = float(world.s_best)
    # An unscoreable world cannot produce a quality reading. Eq(1) will refuse
    # it outright; we must not fabricate a ratio here either.
    attained = max(0.0, min(1.0, best_score / s_best)) if s_best > 0 else 0.0

    n = len(non_root)
    payload: dict[str, object] = {
        "world_id": world.world_id,
        "policy_version_id": policy_version_id,
        "rho": rho,
        "n": n,
        "kstar": rounds,
        "best_score": best_score,
        "stop_reason": stop_reason.value,
        "revealed_order": list(revealed_order),
        "selected_batches": [list(b) for b in action_trace],
        "world_digest": world.world_digest(),
        "trajectory_format": CURRENT_FORMAT,
        "decision_trace": decisions,
    }
    return ReplayEpisode(
        episode_id=new_id("replay"),
        world_id=world.world_id,
        policy_version_id=policy_version_id,
        rho=rho,
        n=n,
        kstar=rounds,
        best_score=best_score,
        attained_quality=attained,
        selected_batches=tuple(action_trace),
        revealed_order=revealed_order,
        stop_reason=stop_reason,
        replay_digest=_episode_digest(payload),
        no_progress_count=no_progress,
        discovery_agent_calls=0,
        evaluator_calls=0,
        trajectory_format=CURRENT_FORMAT,
        decision_trace=tuple(decisions),
    )


def replay_all_rhos(
    world: SealedWorldLike,
    policy: PolicyFn,
    *,
    policy_version_id: str,
    rho_grid: Sequence[float],
    max_parallelism: int,
    max_rounds: int,
) -> tuple[ReplayEpisode, ...]:
    """One replay episode per rho. Order follows ``rho_grid``."""
    return tuple(
        replay_world(
            world,
            policy,
            policy_version_id=policy_version_id,
            rho=rho,
            max_parallelism=max_parallelism,
            max_rounds=max_rounds,
        )
        for rho in rho_grid
    )


def validate_episode_trace(world: SealedWorldLike, episode: ReplayEpisode,
                           max_parallelism: int, max_rounds: int) -> None:
    """Reconstruct every decision from recorded actions, without running policy/agent."""
    if not is_supported_format(episode.trajectory_format):
        raise IntegrityError("legacy replay has no complete decision trace")
    batches = iter(iter_batches(episode))

    def recorded_action(_observation: PrefixObservation, _rho: float) -> Sequence[str]:
        try:
            return next(batches)
        except StopIteration as exc:
            raise IntegrityError("replay trace is missing a decision") from exc

    reproduced = replay_world(world, recorded_action,
        policy_version_id=episode.policy_version_id, rho=episode.rho,
        max_parallelism=max_parallelism, max_rounds=max_rounds)
    for name in ("decision_trace", "selected_batches", "revealed_order", "n", "kstar",
                 "best_score", "attained_quality", "stop_reason", "replay_digest"):
        if getattr(reproduced, name) != getattr(episode, name):
            raise IntegrityError(f"replay trace does not reproduce {name}")


__all__ = [
    "PolicyFn",
    "ReplayCounters",
    "SealedWorldLike",
    "replay_all_rhos",
    "replay_world",
]
