"""Online rollout (§3.2): grow one Discovery Tree for one outer round.

Per round: the fixed policy picks a batch of legal parents, every parent gets
exactly one real generation+evaluation, and the resulting children are attached
to the tree. Parent workspaces are immutable snapshots, so the batch is safe to
run concurrently — no attempt can observe another attempt's writes.

Failure handling follows §3 + the project state machine: an attempt that fails
still produces a node, carrying an explicit ``error_type`` and a score produced
by the **fixed evaluator** under the task protocol. Failures are never dropped
(which would desynchronise online and replay) and are never turned into a
fabricated success score.

``Node`` records carry a workspace *content digest*, never a path, so the
node→workspace association lives here in a :class:`WorkspaceMap` that the
controller persists alongside the tree. Keeping paths out of the world records
is what lets a world be rebuilt on another machine and still replay identically.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

from ..adapters.base import (
    ContinuationContext,
    DiscoveryBackend,
    GeneratedAttempt,
    HistoryEntry,
    TaskSpec,
)
from ..config import DIGEST_EXCLUDES
from ..domain.models import EvaluatorStatus, Node, new_id
from ..domain.tree import DiscoveryTree
from ..evaluator.protocol import EvaluationResult, Evaluator
from ..policy.protocol import PrefixObservation
from ..policy.validation import validate_batch
from ..storage.cas import ContentAddressedStore
from .context import OBSERVATION_FORMAT, branch_chain, canonical

BatchCallback = Callable[[int, DiscoveryTree, Sequence[Node]], None]


class CallCounter:
    """Thread-safe tally of the online rollout's real external calls.

    Replay reports *zero* agent calls and *zero* evaluator calls, and the whole
    replay gate rests on that contrast — so the online side has to be genuinely
    measured, not assumed. Deriving it from node counts would make the two
    numbers agree by construction and the comparison would prove nothing.
    """

    __slots__ = ("_count", "_lock")

    def __init__(self) -> None:
        self._count = 0
        self._lock = Lock()

    def bump(self) -> None:
        with self._lock:
            self._count += 1

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class CallBudget:
    """A hard ceiling on real model calls, shared by every path that makes one.

    Counting is not capping. :class:`CallCounter` reports what a run spent after
    the fact; this refuses to spend past a limit in the first place, which is a
    different guarantee and the one a budget needs.

    The unit is one **discovery-agent invocation** or one **policy-development
    invocation** -- one ``backend.generate`` or one ``developer.propose``, the
    same units the run report already counts. It is deliberately *not* a count
    of model turns: a Pi agent invocation is a whole session that may make many
    turns internally, and the loop cannot see or stop those. A budget in this
    unit therefore bounds the calls the loop initiates, and a report quoting it
    has to say so rather than implying it bounds tokens.

    Reserving is atomic, so parallel batches cannot jointly overshoot: each
    worker's claim is granted under the lock before the call is made.
    """

    __slots__ = ("_limit", "_lock", "_refusals", "_spent")

    def __init__(self, limit: int | None, initial_spent: int = 0) -> None:
        if limit is not None and limit < 0:
            raise ValueError("call budget must be >= 0")
        self._limit = limit
        # A resumed run has already made calls; the ceiling is for the run, not
        # for the process that happens to be continuing it.
        self._spent = max(0, int(initial_spent))
        self._refusals = 0
        self._lock = Lock()

    @property
    def limit(self) -> int | None:
        return self._limit

    @property
    def spent(self) -> int:
        with self._lock:
            return self._spent

    @property
    def refusals(self) -> int:
        """How many calls were asked for and refused. Nonzero means a cut."""
        with self._lock:
            return self._refusals

    @property
    def exhausted(self) -> bool:
        with self._lock:
            return self._limit is not None and self._spent >= self._limit

    def remaining(self) -> int | None:
        with self._lock:
            if self._limit is None:
                return None
            return max(0, self._limit - self._spent)

    def reserve(self, count: int = 1) -> int:
        """Claim up to ``count`` calls. Returns how many were actually granted.

        A partial grant is normal and is not an error: the caller is expected to
        shrink its batch to what it got. A grant of zero means the caller must
        not make the call at all.
        """
        if count < 0:
            raise ValueError("cannot reserve a negative number of calls")
        with self._lock:
            if self._limit is None:
                self._spent += count
                return count
            granted = max(0, min(count, self._limit - self._spent))
            self._spent += granted
            if granted < count:
                self._refusals += count - granted
            return granted

    def to_dict(self) -> dict[str, Any]:
        return {
            "limit": self._limit,
            "spent": self.spent,
            "refusals": self.refusals,
            "exhausted": self.exhausted,
            "unit": "discovery-or-developer invocations (not model turns)",
        }


class CountingBackend:
    """Transparent :class:`DiscoveryBackend` proxy that counts ``generate``."""

    def __init__(self, inner: DiscoveryBackend, counter: CallCounter) -> None:
        self._inner = inner
        self._counter = counter
        # Copied rather than proxied: the protocol declares these as plain
        # attributes, and a read-only property would not satisfy them.
        self.backend_id = getattr(inner, "backend_id", "unknown")
        self.is_live = bool(getattr(inner, "is_live", False))

    def generate(
        self,
        *,
        task: TaskSpec,
        parent_workspace: Path,
        history: Sequence[HistoryEntry],
        output_workspace: Path,
        invocation_id: str,
        context: ContinuationContext | None = None,
    ) -> GeneratedAttempt:
        self._counter.bump()
        return self._inner.generate(
            task=task,
            parent_workspace=parent_workspace,
            history=history,
            output_workspace=output_workspace,
            invocation_id=invocation_id,
            context=context,
        )


class CountingEvaluator:
    """Transparent :class:`Evaluator` proxy that counts ``evaluate``."""

    def __init__(self, inner: Evaluator, counter: CallCounter) -> None:
        self._inner = inner
        self._counter = counter

    @property
    def spec_digest(self) -> str:
        return self._inner.spec_digest

    def evaluate(self, candidate_dir: Path, *, invocation_id: str) -> EvaluationResult:
        self._counter.bump()
        return self._inner.evaluate(candidate_dir, invocation_id=invocation_id)


@dataclass
class WorkspaceMap:
    """Which directory holds each node's frozen workspace."""

    root_id: str
    _paths: dict[str, Path] = field(default_factory=dict)

    def set(self, node_id: str, path: Path) -> None:
        self._paths[node_id] = Path(path).resolve()

    def get(self, node_id: str) -> Path:
        try:
            return self._paths[node_id]
        except KeyError as exc:
            raise KeyError(f"no workspace recorded for node {node_id!r}") from exc

    def __contains__(self, node_id: object) -> bool:
        return node_id in self._paths

    def as_dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in self._paths.items()}


@dataclass(frozen=True)
class AttemptOutcome:
    parent_id: str
    node: Node
    invocation_id: str
    workspace_path: Path
    attempt_error_type: str | None
    backend_id: str
    is_live: bool
    duration_s: float


def _materialize(parent_workspace: Path, output_workspace: Path) -> None:
    """Guarantee the attempt workspace exists, even if a backend died early."""
    if output_workspace.exists():
        return
    output_workspace.mkdir(parents=True, exist_ok=True)
    if parent_workspace.is_dir():
        shutil.copytree(parent_workspace, output_workspace, dirs_exist_ok=True)


def _run_one_attempt(
    *,
    parent_id: str,
    parent_workspace: Path,
    round_index: int,
    branch_index: int,
    task: TaskSpec,
    history: Sequence[HistoryEntry],
    backend: DiscoveryBackend,
    evaluator: Evaluator,
    workspace_root: Path,
    tree: DiscoveryTree,
    store_digest: Callable[[Path], str],
    store: ContentAddressedStore,
    mutation_lock: Lock,
) -> AttemptOutcome:
    invocation_id = new_id("attempt")
    output_workspace = (workspace_root / invocation_id).resolve()
    parent = tree.node(parent_id)
    store.restore_tree(parent.workspace_snapshot_digest, output_workspace)
    context = ContinuationContext(
        parent_id=parent_id,
        branch_observations=branch_chain(tree.nodes(), parent_id, store),
        completed_worlds=tuple(entry.contents_json for entry in history),
        directory=store.root.parent / "ctx" / invocation_id[-16:],
    )
    context.materialize(store)
    context_digest = store.put_json(context.to_dict())
    assert context.directory is not None
    parent_workspace = context.directory / context.workspace_path(parent.workspace_snapshot_digest)

    attempt_error: str | None = None
    generated = None
    try:
        generated = backend.generate(
            task=task,
            parent_workspace=parent_workspace,
            history=history,
            output_workspace=output_workspace,
            invocation_id=invocation_id,
            context=context,
        )
        if not generated.succeeded:
            attempt_error = generated.error_type or "AGENT_FAILED"
    except Exception as exc:
        attempt_error = type(exc).__name__

    # The workspace must be gradeable even when the agent died, so a failed
    # attempt is measured as "produced nothing" — a real reading, not a
    # fabricated score.
    _materialize(parent_workspace, output_workspace)

    evaluation = evaluator.evaluate(output_workspace, invocation_id=invocation_id)
    if evaluation.status is EvaluatorStatus.SUCCESS:
        score = evaluation.score
    elif evaluation.status is EvaluatorStatus.FAILED:
        score = evaluation.score
        attempt_error = attempt_error or evaluation.error_type or "EVALUATION_FAILED"
    else:
        score = 0.0
        attempt_error = attempt_error or evaluation.error_type or "EVALUATION_ERROR"

    workspace_digest = store.put_tree(output_workspace, exclude=DIGEST_EXCLUDES)
    summary = (generated.summary if generated is not None else "") or ""
    if attempt_error:
        summary = f"[attempt error: {attempt_error}] {summary}".strip()

    node_id = new_id("node")
    evaluation_digest = store.put_json(
        {
            "format": "dream-rsi-evaluation-v2",
            "score": score,
            "status": evaluation.status.value,
            "error_type": evaluation.error_type,
            "detail": dict(evaluation.detail),
            "public_detail": dict(evaluation.public_detail),
            "original_result_digest": evaluation.result_digest,
        }
    )
    generation_digest = store.put_json(
        {
            "format": "dream-rsi-generation-v2",
            "invocation_id": invocation_id,
            "context_digest": context_digest,
            "summary": generated.summary if generated else "",
            "record": dict(generated.generation_record) if generated else {},
            "error_type": attempt_error,
        }
    )
    public = {
        "score": score,
        "status": evaluation.status.value,
        "error_type": attempt_error,
        "diagnostics": dict(evaluation.public_detail),
        "generation_summary": generated.summary if generated else "",
    }
    observation_digest = store.put_json(
        {
            "format": OBSERVATION_FORMAT,
            "world_id": tree.world_id,
            "node_id": node_id,
            "parent_id": parent_id,
            "parent_observation_digest": parent.observation_digest,
            "workspace_snapshot_digest": workspace_digest,
            "context_digest": context_digest,
            "generation_digest": generation_digest,
            "public": public,
        }
    )
    # Tree creation indices must be assigned atomically under parallel workers.
    with mutation_lock:
        node = tree.add_child(
            parent_id=parent_id,
            node_id=node_id,
            round_index=round_index,
            branch_index=branch_index,
            workspace_snapshot_digest=workspace_digest,
            observation_digest=observation_digest,
            score=score,
            evaluator_status=evaluation.status,
            summary=summary[:500],
            agent_session_digest=generation_digest,
            evaluator_result_digest=evaluation_digest,
            error_type=attempt_error,
            observation_format=2,
            public_observation_json=canonical(public),
        )
    return AttemptOutcome(
        parent_id=parent_id,
        node=node,
        invocation_id=invocation_id,
        workspace_path=output_workspace,
        attempt_error_type=attempt_error,
        backend_id=getattr(backend, "backend_id", "unknown"),
        is_live=bool(getattr(backend, "is_live", False)),
        duration_s=float(getattr(generated, "duration_s", 0.0)) if generated else 0.0,
    )


def run_online_rollout(
    *,
    world_id: str,
    outer_t: int,
    task: TaskSpec,
    policy_select: Callable[[PrefixObservation, float], Any],
    backend: DiscoveryBackend,
    evaluator: Evaluator,
    workspace_root: Path,
    root_workspace: Path,
    history: Sequence[HistoryEntry],
    max_workers: int,
    max_rounds: int,
    live_rho: float,
    store_digest: Callable[[Path], str],
    store: ContentAddressedStore | None = None,
    root_id: str = "root",
    on_batch_committed: BatchCallback | None = None,
    workspaces: WorkspaceMap | None = None,
    budget: CallBudget | None = None,
    on_budget_cut: Callable[[int], None] | None = None,
) -> tuple[DiscoveryTree, WorkspaceMap]:
    """Run one full online rollout; return the unsealed tree and its workspace map.

    ``max_workers`` is the paper's ``W``: it caps the batch size *and* is the
    ``max_parallelism`` the policy is told about, so a policy can never be
    surprised by a cap it was not shown.

    ``root_id`` labels this world's root. §3 says a world has exactly one
    parentless node; naming it after the world keeps that true while letting
    node ids be unique across worlds, which the registry requires.

    ``budget``, when given, bounds how many discovery calls this rollout may
    make. The world then seals *early* and short rather than complete -- an
    honest partial world, not a failed one.
    """
    if on_budget_cut is None:
        on_budget_cut = lambda _n: None  # noqa: E731
    workspace_root = Path(workspace_root).resolve()
    root_workspace = Path(root_workspace).resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    store = store or ContentAddressedStore(workspace_root / ".cas")
    mutation_lock = Lock()
    root_digest = store.put_tree(root_workspace, exclude=DIGEST_EXCLUDES)
    root_observation = store.put_json(
        {
            "format": OBSERVATION_FORMAT,
            "world_id": world_id,
            "node_id": root_id,
            "parent_id": None,
            "workspace_snapshot_digest": root_digest,
            "generation_digest": None,
            "public": {"score": 0.0, "status": "INITIAL", "diagnostics": {}},
        }
    )
    tree = DiscoveryTree.new_root(
        world_id=world_id,
        root_id=root_id,
        workspace_snapshot_digest=root_digest,
        observation_digest=root_observation,
        summary=f"root workspace for world t={outer_t}",
        observation_format=2,
        public_observation_json=canonical({"score": 0.0, "status": "INITIAL", "diagnostics": {}}),
    )
    ws_map = workspaces if workspaces is not None else WorkspaceMap(root_id=root_id)
    ws_map.root_id = root_id
    ws_map.set(root_id, root_workspace)

    for round_index in range(max_rounds):
        observation = tree.observation(set(tree.node_ids()), round_index, max_workers)
        batch = validate_batch(
            policy_select(observation, live_rho),
            legal_actions=observation.legal_actions,
            observed_ids=observation.node_ids,
            max_parallelism=max_workers,
        )
        if not batch:
            break

        # Freeze every selected parent's workspace at batch start: the paths are
        # resolved once, and each attempt writes only into its own directory.
        parents = [(pid, ws_map.get(pid)) for pid in batch]

        # The budget is checked here, before dispatch, and enforced by shrinking
        # the batch rather than by failing an attempt. A refused attempt would
        # still land as a node with a score of 0.0, which would read as "the
        # policy proposed something bad" when the truth is "we stopped paying".
        if budget is not None:
            granted = budget.reserve(len(parents))
            if granted < len(parents):
                on_budget_cut(len(parents) - granted)
            if granted == 0:
                break
            parents = parents[:granted]

        with ThreadPoolExecutor(max_workers=len(parents)) as pool:
            futures = [
                pool.submit(
                    _run_one_attempt,
                    parent_id=parent_id,
                    parent_workspace=parent_ws,
                    round_index=round_index,
                    branch_index=branch_index,
                    task=task,
                    history=history,
                    backend=backend,
                    evaluator=evaluator,
                    workspace_root=workspace_root,
                    tree=tree,
                    store_digest=store_digest,
                    store=store,
                    mutation_lock=mutation_lock,
                )
                for branch_index, (parent_id, parent_ws) in enumerate(parents)
            ]
            outcomes = [future.result() for future in futures]

        for outcome in outcomes:
            ws_map.set(outcome.node.node_id, outcome.workspace_path)
        if on_batch_committed is not None:
            on_batch_committed(round_index, tree, [o.node for o in outcomes])

    return tree, ws_map


__all__ = [
    "AttemptOutcome",
    "CallBudget",
    "CallCounter",
    "CountingBackend",
    "CountingEvaluator",
    "WorkspaceMap",
    "run_online_rollout",
]
