"""Run controller: the single writer of the run state machine.

One outer round ``t``:

1.  **Settle** ``H_t`` — the sealed, scoreable worlds from rounds ``< t``.
2.  **Select** ``pi_t``. At ``t = 0`` there is no history, so the incumbent is
    deployed without a contest and that fact is recorded. From ``t >= 1`` every
    candidate (incumbent included) is replayed on the *whole* of ``H_t`` across
    the full rho grid, scored with Eq(1), and the highest mean ``V`` wins.
3.  **Grow** world ``t`` online under ``pi_t`` for ``K1`` rounds, one real
    generation+evaluation per selected parent, batched in parallel.
4.  **Seal** world ``t``: ``B``, ``S``, a content digest, and a replay from the
    *persisted* records re-validated against §3.

Durability: nodes are committed per batch inside one SQLite transaction, and a
world only becomes replayable once sealed. A resumed run reuses sealed worlds
and quarantines — never silently reuses — a world that was interrupted.

Nothing here decides whether results are *good*: selection is
:mod:`dream_rsi.runtime.selector` and scoring is :mod:`dream_rsi.score`. The
controller's job is ordering, durability and refusing to proceed on a broken
invariant.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from ..adapters.base import DiscoveryBackend, HistoryEntry, TaskSpec
from ..domain.errors import BlockedModelConfiguration, IntegrityError, PolicyRejected
from ..domain.models import (
    AlgorithmMode,
    Attempt,
    AttemptState,
    Event,
    Node,
    PolicyState,
    PolicyVersion,
    Revision,
    Run,
    RunMode,
    RunState,
    ScoreRecord,
    World,
    WorldState,
    new_id,
)
from ..domain.tree import DiscoveryTree, SealedWorld
from ..evaluator.protocol import Evaluator
from ..policy.protocol import PrefixObservation
from ..policy.runner import SubprocessPolicyRunner
from ..score.equation1 import SCORE_FORMULA_VERSION, equation1
from ..storage.cas import ContentAddressedStore
from ..storage.sqlite_registry import Registry
from .context import canonical, export_world
from .online import (
    CallBudget,
    CallCounter,
    CountingBackend,
    CountingEvaluator,
    WorkspaceMap,
    run_online_rollout,
)
from .policy_development import (
    PolicyDeveloper,
    deploy_policy,
    validate_policy_source,
)
from .replay import replay_all_rhos
from .trajectory import render_episode_steps, render_node_table
from .selector import PolicyScore, Selection, assert_monotonic, select

#: A world with no positive best score cannot be scored by Eq(1) at all: the
#: denominator ``S`` is zero. Such a world is sealed as UNSCOREABLE and kept,
#: but it never enters ``H_t`` — substituting a fake ``S`` would corrupt every
#: downstream number.
MIN_REPLAY_WORLDS = 1


@dataclass(frozen=True)
class WorldSummary:
    world_id: str
    outer_t: int
    state: WorldState
    node_count: int
    b_count: int
    s_best: float
    world_digest: str | None
    policy_version_id: str
    replay_worlds_available: int


@dataclass(frozen=True)
class SelectionSummary:
    iteration_t: int
    selected_policy_id: str
    deployed_policy_id: str
    deployed_again: bool
    reason: str
    mean_values: tuple[tuple[str, float], ...]


@dataclass
class RunOutcome:
    run_id: str
    state: RunState
    mode: RunMode
    worlds: tuple[WorldSummary, ...] = ()
    selections: tuple[SelectionSummary, ...] = ()
    replay_count: int = 0
    online_agent_calls: int = 0
    online_evaluator_calls: int = 0
    developer_id: str = ""
    developer_is_live: bool = False
    policy_agent_calls: int = 0
    notes: tuple[str, ...] = ()
    budget: dict[str, Any] = field(default_factory=dict)
    final_policy_id: str = ""

    @property
    def live(self) -> bool:
        return self.mode is RunMode.LIVE


@dataclass
class ControllerDeps:
    backend: DiscoveryBackend
    evaluator: Evaluator
    registry: Registry
    store: ContentAddressedStore
    policy_runner: SubprocessPolicyRunner
    task: TaskSpec
    developer: PolicyDeveloper | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    notes: list[str] = field(default_factory=list)
    #: Hard ceiling on real model calls for the whole run, shared by the online
    #: rollouts and the policy developer. ``None`` means uncapped, which is what
    #: every existing caller gets -- the budget is opt-in so that adding it
    #: cannot silently change a run that was not given one.
    budget: CallBudget | None = None


class Controller:
    """Drives one run end to end. The only component that writes state."""

    def __init__(self, config: Any, deps: ControllerDeps) -> None:
        self.config = config
        self.deps = deps
        self.loop = config.loop

    # ------------------------------------------------------------------ helpers

    def _emit(self, run_id: str, kind: str, **payload: Any) -> None:
        event = Event(run_id=run_id, kind=kind, payload=payload)
        self.deps.registry.append_event(event)
        self._mirror_event(event)
        if self.deps.on_event is not None:
            self.deps.on_event(kind, payload)

    def _mirror_event(self, event: Event) -> None:
        """Append the event to the run's ``events.jsonl``.

        ``contracts/SCHEMAS.md`` names the event log as ``events.jsonl`` **and**
        SQLite WAL; the table alone is only half of that, and the half that is
        missing is the one that survives the database. The registry also holds
        mutable state — nodes, policy states that change as a run proceeds — so
        "the immutable record of what happened" currently lives in the same file
        as "the current state of things". The mirror separates them.

        Written *after* the row is committed, never inside the transaction: a
        rollback must not leave a line claiming something that never happened.
        The cost is that a hard crash can lose the last line or two, which is
        the safe direction for an audit trail to be wrong in. The table remains
        authoritative — this is a copy, and :meth:`list_events` is unaffected.

        A mirror that cannot be written is noted, not raised. The run's durable
        record is already complete by this point, and aborting a live run over a
        redundant copy of it would trade the result for a copy of the result.
        """
        path = Path(self.config.run_root(event.run_id)) / "events.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(
                event.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            # newline="\n" is deliberate: an explicit LF rather than the
            # platform default, so the log is byte-identical on every machine.
            with path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
        except OSError as exc:
            self._note(f"events.jsonl mirror failed for {event.kind}: {exc}")

    def _note(self, message: str) -> None:
        self.deps.notes.append(message)

    def _on_budget_cut(self, run_id: str, world_id: str, refused: int) -> None:
        """Record that the call budget, not the policy, ended a batch.

        This is deliberately loud. A world that sealed short because the money
        ran out looks identical in the tree to one that sealed short because the
        policy stopped asking, and those two must never be confused when the
        results are read back.
        """
        budget = self.deps.budget
        spent = budget.spent if budget else 0
        limit = budget.limit if budget else None
        message = (
            f"{world_id}: budget cut {refused} discovery call(s) in this batch; "
            f"{spent}/{limit} model calls spent"
        )
        self._note(message)
        self._emit(run_id, "budget.cut", world_id=world_id, refused=refused,
                   spent=spent, limit=limit)

    def _digest_tree(self, directory: Path) -> str:
        return self.deps.store.put_tree(directory)

    def _digest_path(self, path: Path) -> str:
        """Content digest of a directory *or* a file.

        Workspace snapshots are whole trees; an ``observation_digest`` is a
        single artifact. One helper handles both so a caller cannot silently
        hash the wrong kind of thing — a missing path hashes to ``""``, which
        is visibly empty rather than a plausible-looking digest.
        """
        target = Path(path)
        if target.is_dir():
            return self.deps.store.put_tree(target)
        if target.is_file():
            return self.deps.store.put_file(target)
        return ""

    # ------------------------------------------------------------------ run

    def _run_mode(self) -> RunMode:
        """``LIVE`` only when *every* component that produces results is live.

        This used to read the discovery backend alone, which meant a run whose
        policy revisions were hand-written constants could still be recorded as
        ``live`` — the exact confusion the "a fixture result must never be
        merged into live statistics" gate exists to prevent. A live backend
        plus an offline developer is a *hybrid*, and the vocabulary has no word
        for it, so it takes the conservative label and the run record names the
        developer so the mix is never invisible.
        """
        backend_live = bool(getattr(self.deps.backend, "is_live", False))
        developer = self.deps.developer
        developer_live = developer is None or bool(
            getattr(developer, "is_live", False)
        )
        return (
            RunMode.LIVE
            if backend_live and developer_live
            else RunMode.OFFLINE_FIXTURE
        )

    def run(self) -> RunOutcome:
        cfg = self.config
        run_id = cfg.run_id or new_id("run")
        cfg.ensure_dirs(run_id)

        existing = self.deps.registry.get_run(run_id)
        if existing is not None and not cfg.resume:
            raise IntegrityError(f"run {run_id} already exists; pass --resume to continue it")
        if existing is None and cfg.resume:
            raise IntegrityError(f"run {run_id} does not exist; there is nothing to resume")
        if existing is None:
            run = Run(
                run_id=run_id,
                mode=self._run_mode(),
                algorithm_mode=AlgorithmMode.STRICT_S3,
                task_spec_digest=self.deps.task.spec_digest,
                provider_id=str(cfg.provider or getattr(self.deps.backend, "backend_id", "unknown")),
                agent_model=str(cfg.model or "fixture"),
                fixed_evaluator_digest=self.deps.evaluator.spec_digest,
                cfg_digest=cfg.cfg_digest,
            )
            self.deps.registry.create_run(run)
        else:
            run = existing
            if run.state is RunState.BLOCKED_EXTERNAL:
                raise IntegrityError(
                    f"run {run_id} is externally blocked; start a fresh run after "
                    "fixing the model connection rather than reusing failed worlds"
                )
            if run.cfg_digest != cfg.cfg_digest:
                raise IntegrityError(
                    f"run {run_id} was created with cfg_digest {run.cfg_digest} but this "
                    f"config hashes to {cfg.cfg_digest}; resuming under different "
                    "parameters would make the results meaningless"
                )
            if (
                run.task_spec_digest != self.deps.task.spec_digest
                or run.fixed_evaluator_digest != self.deps.evaluator.spec_digest
                or run.mode != self._run_mode()
            ):
                raise IntegrityError(f"run {run_id} changed task, evaluator, or live mode")

        outcome = RunOutcome(run_id=run_id, state=run.state, mode=run.mode)
        # Who produced what, in the append-only log. The run's `mode` is coarse
        # (see `_run_mode`); this event is what makes a hybrid run legible after
        # the fact instead of having to be inferred from the code that ran.
        developer = self.deps.developer
        outcome.developer_id = str(getattr(developer, "developer_id", "none"))
        outcome.developer_is_live = bool(getattr(developer, "is_live", False))

        policy_root = cfg.policy_root(run_id)
        deployed_path = policy_root / "pi_0.py"
        if not deployed_path.exists():
            if existing is not None:
                raise IntegrityError(f"run {run_id} lost its incumbent policy file")
            shutil.copyfile(cfg.incumbent_policy_path, deployed_path)
        elif existing is not None and deployed_path.read_bytes() != cfg.incumbent_policy_path.read_bytes():
            raise IntegrityError(f"run {run_id} changed its incumbent policy file")
        deployed_digest = self.deps.store.put_file(deployed_path)

        incumbent_id = f"{run_id}-pi0"
        if self.deps.registry.get_policy(incumbent_id) is None:
            self.deps.registry.add_policy(PolicyVersion(
                version_id=incumbent_id,
                run_id=run_id,
                iteration_t=0,
                m=0,
                parent_version=None,
                source_sha256=deployed_digest,
                source_artifact_id=str(deployed_path),
                origin="incumbent",
                state=PolicyState.ELIGIBLE,
                static_validation_status="PASS",
                sandbox_validation_status="PENDING",
            ))
        deployed_policy_id = incumbent_id

        worlds: list[WorldSummary] = []
        selections: list[SelectionSummary] = []
        replay_count = 0
        online_agent_calls = 0
        online_evaluator_calls = 0
        start_t = 0
        resumed_selection_id: str | None = None
        if existing is not None:
            start_t, worlds, deployed_policy_id, deployed_path, resumed_selection_id = (
                self._resume_position(run_id, policy_root, incumbent_id)
            )
            selections = self._recorded_selections(run_id, worlds, incumbent_id)
            if start_t >= self.loop.outer_rounds and run.state is RunState.COMMITTED:
                outcome.worlds = tuple(worlds)
                outcome.selections = tuple(selections)
                outcome.state = RunState.COMMITTED
                outcome.final_policy_id = deployed_policy_id
                self._note(f"run {run_id} was already committed; no work repeated")
                outcome.notes = tuple(self.deps.notes)
                return outcome

        self._emit(run_id, "run.start", backend=self.deps.task.task_id,
                   cfg_digest=cfg.cfg_digest)
        self._emit(
            run_id, "run.components",
            backend_id=str(getattr(self.deps.backend, "backend_id", "unknown")),
            backend_is_live=bool(getattr(self.deps.backend, "is_live", False)),
            developer_id=outcome.developer_id,
            developer_is_live=outcome.developer_is_live,
            mode=run.mode.value,
        )

        for t in range(start_t, self.loop.outer_rounds):
            self.deps.registry.update_run_state(run_id, RunState.ONLINE_RUNNING)

            # -- 1/2. Settle H_t and choose pi_t -----------------------------
            history_worlds = self._settle_history(run_id, t)
            if t == start_t and resumed_selection_id is not None:
                self._emit(run_id, "policy.selection.reused", iteration_t=t,
                           selected=resumed_selection_id)
            elif t == 0 or not history_worlds:
                self._note(
                    f"t={t}: H_t is empty, so pi_t is the incumbent by definition, "
                    "not by winning a contest"
                )
                self._emit(run_id, "policy.no_contest", iteration_t=t,
                           reason="empty_history")
            else:
                selection, episodes = self._select_policy(
                    run_id=run_id,
                    iteration_t=t,
                    incumbent_id=deployed_policy_id,
                    incumbent_path=deployed_path,
                    history_worlds=history_worlds,
                )
                replay_count += episodes
                deployed_policy_id = selection.selected.policy_version_id
                deployed_path = policy_root / f"pi_{t}.py"
                src = self.deps.registry.get_policy(deployed_policy_id)
                if src is None:
                    raise IntegrityError(f"selected policy {deployed_policy_id} vanished")
                deploy_policy(Path(src.source_artifact_id), deployed_path)
                assert_monotonic(selection)
                selections.append(SelectionSummary(
                    iteration_t=t,
                    selected_policy_id=selection.selected.policy_version_id,
                    deployed_policy_id=selection.deployed.policy_version_id,
                    deployed_again=selection.deployed_again,
                    reason=selection.reason,
                    mean_values=tuple(
                        (c.policy_version_id, c.mean_value) for c in selection.ranking
                    ),
                ))
                self.deps.registry.update_run_state(run_id, RunState.POLICY_SELECTED,
                                                    note=selection.reason)
                self._emit(run_id, "policy.selected", iteration_t=t,
                           selected=selection.selected.policy_version_id,
                           deployed_again=selection.deployed_again,
                           reason=selection.reason)

            # -- 3. Grow world t ---------------------------------------------
            world_id = self._next_world_id(run_id, t)
            # §3 gives each world exactly one parentless root. Naming it after
            # the world keeps that invariant while making node ids unique across
            # worlds, which the registry's global node key requires.
            root_id = f"{world_id}-root"
            world = World(
                world_id=world_id,
                outer_t=t,
                run_id=run_id,
                root_id=root_id,
                policy_version_id=deployed_policy_id,
                state=WorldState.BUILDING,
            )
            self.deps.registry.add_world(world)
            self._emit(run_id, "world.build.start", world_id=world_id, outer_t=t,
                       policy_version_id=deployed_policy_id)

            agent_counter = CallCounter()
            evaluator_counter = CallCounter()
            tree, ws_map = run_online_rollout(
                world_id=world_id,
                outer_t=t,
                task=self.deps.task,
                policy_select=self._policy_select_fn(deployed_path),
                backend=CountingBackend(self.deps.backend, agent_counter),
                evaluator=CountingEvaluator(self.deps.evaluator, evaluator_counter),
                workspace_root=cfg.workspaces_root(run_id) / world_id,
                root_workspace=cfg.seed_workspace,
                history=self._history_entries(worlds),
                max_workers=self.loop.W,
                max_rounds=self.loop.K1,
                live_rho=self.loop.live_rho,
                store_digest=self._digest_path,
                store=self.deps.store,
                root_id=root_id,
                on_batch_committed=self._make_batch_committer(run_id, world_id),
                workspaces=WorkspaceMap(root_id=root_id),
                budget=self.deps.budget,
                on_budget_cut=partial(self._on_budget_cut, run_id, world_id),
            )
            online_agent_calls += agent_counter.count
            online_evaluator_calls += evaluator_counter.count
            self._persist_workspace_map(run_id, world_id, ws_map)

            # The root is a node too. Without it a sealed world cannot be read
            # back into a SealedWorld, because §3 requires exactly one parentless
            # node and the world would look rootless.
            root_node = tree.nodes()[0]
            if self.deps.registry.get_node(root_node.node_id) is None:
                self.deps.registry.add_node(root_node)

            # -- 4. Seal -----------------------------------------------------
            summary = self._seal_world(world, tree)
            worlds.append(summary)
            self.deps.registry.update_run_state(run_id, RunState.WORLD_SEALED,
                                                note=f"{world_id} {summary.state.value}")
            self._emit(run_id, "world.sealed", world_id=world_id,
                       state=summary.state.value, b_count=summary.b_count,
                       s_best=summary.s_best, digest=summary.world_digest)
            if run.mode is RunMode.LIVE and not any(
                node.error_type is None for node in tree.nodes()[1:]
            ):
                reason = f"all Pi discovery attempts failed in {world_id}"
                self.deps.registry.update_run_state(
                    run_id, RunState.BLOCKED_EXTERNAL, note=reason,
                )
                self._emit(run_id, "run.blocked", reason=reason)
                raise BlockedModelConfiguration(reason)

        # Complete the final online+offline iteration. Intermediate selections
        # run at the next iteration's start; this selection has no next rollout.
        final_t = self.loop.outer_rounds
        final_history = self._settle_history(run_id, final_t)
        if final_history and not any(s.iteration_t == final_t for s in selections):
            selection, episodes = self._select_policy(
                run_id=run_id, iteration_t=final_t,
                incumbent_id=deployed_policy_id, incumbent_path=deployed_path,
                history_worlds=final_history,
            )
            replay_count += episodes
            deployed_policy_id = selection.selected.policy_version_id
            selected = self.deps.registry.get_policy(deployed_policy_id)
            if selected is None:
                raise IntegrityError("final selected policy vanished")
            deployed_path = policy_root / f"pi_{final_t}.py"
            deploy_policy(Path(selected.source_artifact_id), deployed_path)
            assert_monotonic(selection)
            selections.append(SelectionSummary(
                iteration_t=final_t, selected_policy_id=deployed_policy_id,
                deployed_policy_id=selection.deployed.policy_version_id,
                deployed_again=selection.deployed_again, reason=selection.reason,
                mean_values=tuple((c.policy_version_id, c.mean_value) for c in selection.ranking),
            ))
            self._emit(run_id, "policy.selected", iteration_t=final_t,
                       selected=deployed_policy_id, deployed_again=selection.deployed_again,
                       reason=selection.reason, final=True)
        elif not final_history:
            self._emit(run_id, "policy.final.skipped", reason="no scoreable historical world")
        outcome.final_policy_id = deployed_policy_id

        # -- finish ---------------------------------------------------------
        state = RunState.COMMITTED
        self.deps.registry.update_run_state(run_id, state, finished=True)
        outcome.state = state
        outcome.worlds = tuple(worlds)
        outcome.selections = tuple(selections)
        outcome.replay_count = replay_count
        outcome.online_agent_calls = online_agent_calls
        outcome.online_evaluator_calls = online_evaluator_calls
        # Only a developer that records its own calls can be counted. A
        # developer that records none is reported as 0 rather than estimated
        # from the number of revisions, because revisions can be produced
        # without any call at all (the scripted one does exactly that).
        outcome.policy_agent_calls = len(getattr(self.deps.developer, "sessions", ()))
        outcome.notes = tuple(self.deps.notes)
        if self.deps.budget is not None:
            outcome.budget = self.deps.budget.to_dict()
            self._emit(run_id, "run.budget", **outcome.budget)
        self._emit(run_id, "run.finish", state=state.value, worlds=len(worlds))
        return outcome

    # ------------------------------------------------------------------ rollout glue

    def _policy_select_fn(self, policy_path: Path) -> Callable[[PrefixObservation, float], Any]:
        """A live session per batch: the policy is loaded fresh, never trusted in-process."""
        runner = self.deps.policy_runner

        def _select(observation: PrefixObservation, rho: float) -> Any:
            session = runner.open(
                policy_path=policy_path,
                policy_version_id=policy_path.stem,
                session_key="online",
            )
            try:
                return session.select(observation, rho)
            finally:
                session.close()

        return _select

    def _make_batch_committer(self, run_id: str, world_id: str) -> Callable[..., None]:
        registry = self.deps.registry

        def _commit(round_index: int, tree: DiscoveryTree, nodes: Sequence[Node]) -> None:
            with registry.transaction():
                registry.add_nodes(nodes)
                for node in nodes:
                    registry.add_attempt(Attempt(
                        attempt_id=f"{node.node_id}-attempt",
                        world_id=world_id,
                        parent_id=node.parent_id or tree.root_id,
                        round_index=round_index,
                        invocation_id=node.node_id,
                        state=AttemptState.FINISHED
                        if node.error_type is None
                        else AttemptState.ERROR_RECORDED,
                        workspace_snapshot_digest=node.workspace_snapshot_digest,
                        failure_reason=node.error_type,
                    ))
            self._emit(run_id, "world.batch.committed", world_id=world_id,
                       round_index=round_index, node_count=len(nodes))

        return _commit

    def _persist_workspace_map(self, run_id: str, world_id: str, ws_map: WorkspaceMap) -> None:
        path = self.config.run_root(run_id) / "workspaces" / f"{world_id}.map.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(ws_map.as_dict(), indent=2, sort_keys=True),
                        encoding="utf-8")

    # ------------------------------------------------------------------ worlds

    def _next_world_id(self, run_id: str, t: int) -> str:
        """Deterministic per (run, t); a retry after an interrupted world gets a suffix."""
        base = f"{run_id}-w{t}"
        existing = [w for w in self.deps.registry.list_worlds(run_id) if w.outer_t == t]
        usable = [w for w in existing if w.state.value in ("SEALED_SCOREABLE", "SEALED_UNSCOREABLE")]
        if usable:
            return usable[-1].world_id
        if not existing:
            return base
        return f"{base}-r{len(existing)}"

    def _resume_position(
        self, run_id: str, policy_root: Path, incumbent_id: str,
    ) -> tuple[int, list[WorldSummary], str, Path, str | None]:
        """Reuse sealed rounds; quarantine an interrupted world and retry it.

        A selection already recorded for the interrupted round is reused. If a
        crash happened *inside* candidate selection, its incomplete revisions
        cannot safely be repeated under the same IDs, so resume fails closed.
        """
        all_worlds = self.deps.registry.list_worlds(run_id)
        summaries: list[WorldSummary] = []
        for t in range(self.loop.outer_rounds):
            sealed = [
                w for w in all_worlds
                if w.outer_t == t and w.state in (
                    WorldState.SEALED_SCOREABLE, WorldState.SEALED_UNSCOREABLE,
                )
            ]
            if len(sealed) > 1:
                raise IntegrityError(f"run {run_id} has multiple sealed worlds at t={t}")
            if not sealed:
                break
            world = sealed[0]
            policy = self.deps.registry.get_policy(world.policy_version_id)
            source = Path(policy.source_artifact_id) if policy is not None else None
            deployment = policy_root / f"pi_{t}.py"
            if source is None or not source.is_file() or not deployment.is_file():
                raise IntegrityError(f"run {run_id} lost its deployed policy at t={t}")
            if (deployment.read_bytes() != source.read_bytes()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != policy.source_sha256):  # type: ignore[union-attr]
                raise IntegrityError(f"run {run_id} changed its deployed policy at t={t}")
            nodes = self.deps.registry.list_nodes(world.world_id)
            restored = SealedWorld.from_nodes(world.world_id, nodes, root_id=world.root_id)
            if (
                restored.world_digest() != world.world_digest
                or restored.b_count != world.b_count
                or restored.s_best != world.s_best
            ):
                raise IntegrityError(f"sealed world {world.world_id} changed before resume")
            summaries.append(WorldSummary(
                world_id=world.world_id, outer_t=t, state=world.state,
                node_count=len(nodes), b_count=restored.b_count,
                s_best=restored.s_best, world_digest=world.world_digest,
                policy_version_id=world.policy_version_id, replay_worlds_available=0,
            ))

        start_t = len(summaries)
        if self.deps.registry.get_run(run_id).state is RunState.COMMITTED and start_t < self.loop.outer_rounds:  # type: ignore[union-attr]
            raise IntegrityError(f"run {run_id} is marked committed but has missing worlds")
        if any(w.outer_t > start_t and w.state is not WorldState.QUARANTINED for w in all_worlds):
            raise IntegrityError(f"run {run_id} has a gap in completed outer rounds")
        events = [
            e for e in self.deps.registry.list_events(run_id)
            if e.kind == "policy.selected" and e.payload.get("iteration_t") == start_t
        ]
        if len(events) > 1:
            raise IntegrityError(f"run {run_id} has multiple selections at t={start_t}")
        selected_id = str(events[0].payload["selected"]) if events else None
        if (start_t == self.loop.outer_rounds and selected_id is None
                and self.deps.registry.get_run(run_id).state is RunState.COMMITTED  # type: ignore[union-attr]
                and any(w.state is WorldState.SEALED_SCOREABLE for w in summaries)):
            raise IntegrityError("committed v2 run is missing its final policy selection")
        if start_t > 0 and selected_id is None and self.deps.registry.list_revisions(run_id, start_t):
            raise IntegrityError(
                f"run {run_id} stopped during policy selection at t={start_t}; "
                "candidate calls may have uncertain outcomes, so automatic replay is unsafe"
            )
        partial = [
            w for w in all_worlds
            if w.outer_t == start_t and w.state is WorldState.BUILDING
        ]
        if partial and start_t > 0 and selected_id is None:
            raise IntegrityError(f"run {run_id} has a building world without a selected policy")
        for world in partial:
            self.deps.registry.quarantine_world(world.world_id, "interrupted before seal; retry fresh")
            self._emit(run_id, "world.quarantined", world_id=world.world_id,
                       reason="interrupted before seal")

        if selected_id is not None:
            policy = self.deps.registry.get_policy(selected_id)
            path = policy_root / f"pi_{start_t}.py"
            if policy is None or not path.is_file():
                raise IntegrityError(f"selected policy {selected_id} is missing on resume")
            source = Path(policy.source_artifact_id)
            if (not source.is_file() or path.read_bytes() != source.read_bytes()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != policy.source_sha256):
                raise IntegrityError(f"selected policy {selected_id} changed before resume")
            return start_t, summaries, selected_id, path, selected_id

        previous_id = summaries[-1].policy_version_id if summaries else incumbent_id
        previous_path = policy_root / f"pi_{start_t-1}.py" if summaries else policy_root / "pi_0.py"
        if not previous_path.is_file():
            raise IntegrityError(f"deployed policy for t={start_t-1} is missing on resume")
        return start_t, summaries, previous_id, previous_path, None

    def _recorded_selections(
        self, run_id: str, worlds: Sequence[WorldSummary], incumbent_id: str,
    ) -> list[SelectionSummary]:
        """Rebuild prior selection summaries for a resumed run's report."""
        by_t = {w.outer_t: w for w in worlds}
        selections: list[SelectionSummary] = []
        for event in self.deps.registry.list_events(run_id):
            if event.kind != "policy.selected":
                continue
            t = int(event.payload["iteration_t"])
            if t < 1 or t > self.loop.outer_rounds:
                raise IntegrityError(f"invalid recorded selection round {t}")
            selected_id = str(event.payload["selected"])
            previous_id = by_t[t - 1].policy_version_id if t - 1 in by_t else incumbent_id
            history = [w for w in worlds if w.outer_t < t and w.state is WorldState.SEALED_SCOREABLE]
            ids = [previous_id, *(
                r.output_policy_id for r in self.deps.registry.list_revisions(run_id, t)
                if r.output_policy_id is not None
            )]
            values: list[tuple[str, float]] = []
            for policy_id in ids:
                records = [
                    self.deps.registry.list_scores(policy_version_id=policy_id, world_id=w.world_id)
                    for w in history
                ]
                if records and all(len(rows) == 1 for rows in records):
                    values.append((policy_id, sum(rows[0].value for rows in records) / len(records)))
            values.sort(key=lambda pair: -pair[1])
            selections.append(SelectionSummary(
                iteration_t=t, selected_policy_id=selected_id,
                deployed_policy_id=previous_id,
                deployed_again=bool(event.payload["deployed_again"]),
                reason=str(event.payload["reason"]), mean_values=tuple(values),
            ))
        return selections

    def _seal_world(self, world: World, tree: DiscoveryTree) -> WorldSummary:
        b_count, s_best = tree.seal()
        digest = tree.world_digest()
        if b_count <= 0:
            state = WorldState.SEALED_UNSCOREABLE
            note = "world has no non-root nodes; B=0 so Eq(1) is undefined"
            self._note(f"{world.world_id}: {note}")
        elif not s_best > 0:
            state = WorldState.SEALED_UNSCOREABLE
            note = f"best non-root score is {s_best!r}; S<=0 makes Eq(1) undefined"
            self._note(f"{world.world_id}: {note}")
        else:
            state = WorldState.SEALED_SCOREABLE
            note = None

        self.deps.registry.seal_world(
            world.world_id, state=state, world_digest=digest,
            b_count=b_count, s_best=s_best, note=note,
        )

        # Read the world back and re-validate it before it may ever be replayed.
        persisted = self.deps.registry.list_nodes(world.world_id)
        sealed = SealedWorld.from_nodes(world.world_id, persisted, root_id=world.root_id)
        if sealed.world_digest() != digest:
            raise IntegrityError(
                f"world {world.world_id}: persisted records hash to {sealed.world_digest()} "
                f"but the live tree hashed to {digest}"
            )
        if (sealed.b_count, sealed.s_best) != (b_count, s_best):
            raise IntegrityError(f"world {world.world_id}: (B, S) changed on round trip")

        return WorldSummary(
            world_id=world.world_id,
            outer_t=world.outer_t,
            state=state,
            node_count=len(persisted),
            b_count=b_count,
            s_best=s_best,
            world_digest=digest,
            policy_version_id=world.policy_version_id,
            replay_worlds_available=0,
        )

    def _settle_history(self, run_id: str, t: int) -> list[SealedWorld]:
        """Sealed, scoreable worlds with ``outer_t < t``, oldest first."""
        out: list[SealedWorld] = []
        for world in sorted(self.deps.registry.list_worlds(run_id), key=lambda w: w.outer_t):
            if world.outer_t >= t:
                continue
            if world.state is not WorldState.SEALED_SCOREABLE:
                continue
            nodes = self.deps.registry.list_nodes(world.world_id)
            sealed = SealedWorld.from_nodes(world.world_id, nodes, root_id=world.root_id)
            if sealed.world_digest() != world.world_digest:
                raise IntegrityError(
                    f"sealed world {world.world_id} changed after sealing: "
                    f"stored digest={world.world_digest}, actual={sealed.world_digest()}"
                )
            if (sealed.b_count, sealed.s_best) != (world.b_count, world.s_best):
                raise IntegrityError(f"sealed world {world.world_id} has inconsistent B or S")
            out.append(sealed)
        return out

    def _history_entries(self, worlds: Sequence[WorldSummary]) -> tuple[HistoryEntry, ...]:
        """Sanitized public view of finished worlds. No hidden node is exposed."""
        return tuple(
            HistoryEntry(
                outer_t=w.outer_t,
                world_digest=w.world_digest or "",
                node_count=w.node_count,
                best_score=w.s_best,
                policy_version_id=w.policy_version_id,
                summary=f"sealed {w.state.value}",
                contents_json=canonical(export_world(self.deps.store,
                    self.deps.registry.list_nodes(w.world_id), world_id=w.world_id,
                    outer_t=w.outer_t, world_digest=w.world_digest)),
            )
            for w in worlds
        )

    # ------------------------------------------------------------------ selection

    def _revision_candidates(
        self, *, run_id: str, iteration_t: int, incumbent_path: Path,
        incumbent_id: str, feedback: Sequence[str], revision_m: int,
    ) -> tuple[list[PolicyVersion], list[tuple[str, str]]]:
        """Propose and screen one revision of the just-scored policy.

        Returns the eligible versions and the ``(version_id, reason)`` pairs of
        everything that was rejected — rejections are recorded, never dropped.
        """
        developer = self.deps.developer
        if developer is None or self.loop.M <= 1:
            return [], []

        # A revision is a real model call and comes out of the same budget as
        # the discovery calls. Refusing here leaves the incumbent in place,
        # which is the honest outcome of running out of budget mid-selection:
        # the loop keeps what it had rather than inventing a revision.
        if self.deps.budget is not None and self.deps.budget.reserve(1) == 0:
            reason = (
                f"model call budget exhausted ({self.deps.budget.spent}/"
                f"{self.deps.budget.limit}); no revision attempted at t={iteration_t}"
            )
            self._note(reason)
            self.deps.registry.add_revision(Revision(
                iteration_t=iteration_t, m=revision_m, input_policy_id=incumbent_id,
                output_policy_id=None, status="BUDGET_EXHAUSTED",
                feedback_digest=self._digest_text("\n".join(feedback)),
                failure_reason=reason[:500],
            ))
            return [], [(incumbent_id, reason)]

        out_dir = self.config.policy_root(run_id) / f"proposals_t{iteration_t}"
        try:
            proposals = list(developer.propose(
                incumbent_path=incumbent_path,
                feedback=feedback,
                output_dir=out_dir,
                count=1,
                iteration_t=iteration_t,
                start_m=revision_m,
            ))
        except Exception as exc:
            reason = f"developer failed: {type(exc).__name__}: {exc}"
            # A revision we declined to send is not a broken connection. The
            # first formal A arm died because its feedback outgrew the model's
            # context window and the resulting agent error was classified as an
            # external failure, which blocked the run. The honest reading is that
            # this revision could not be afforded: keep the incumbent, record why,
            # and carry on, exactly as a token-budget cut does.
            context_bound = "CONTEXT_LIMIT_EXCEEDED" in reason
            self.deps.registry.add_revision(Revision(
                iteration_t=iteration_t, m=revision_m, input_policy_id=incumbent_id,
                output_policy_id=None,
                status="CONTEXT_LIMIT" if context_bound else "DEVELOPER_ERROR",
                feedback_digest=self._digest_text("\n".join(feedback)),
                failure_reason=reason[:500],
            ))
            self._note(f"t={iteration_t}: {reason}")
            self._emit(run_id, "policy.development.skipped" if context_bound
                       else "policy.development.failed",
                       iteration_t=iteration_t, m=revision_m, reason=reason[:500])
            if context_bound:
                return [], [(incumbent_id, reason)]
            if bool(getattr(developer, "is_live", False)) and "call failed" in reason:
                self.deps.registry.update_run_state(
                    run_id, RunState.BLOCKED_EXTERNAL, note=reason[:500],
                )
                self._emit(run_id, "run.blocked", reason=reason[:500])
                raise BlockedModelConfiguration(reason) from exc
            return [], [(incumbent_id, reason)]

        eligible: list[PolicyVersion] = []
        rejected: list[tuple[str, str]] = []
        for proposal in proposals[:1]:
            if proposal.m != revision_m:
                raise IntegrityError(
                    f"developer returned revision m={proposal.m}, expected m={revision_m}"
                )
            version_id = f"{run_id}-t{iteration_t}-m{proposal.m}"
            check = validate_policy_source(proposal.path)
            revision = Revision(
                iteration_t=iteration_t,
                m=proposal.m,
                input_policy_id=incumbent_id,
                output_policy_id=version_id,
                status="PENDING",
                feedback_digest=self._digest_text("\n".join(feedback)),
                proposal_digest=check.digest,
            )
            if not check.ok:
                self.deps.registry.add_policy(PolicyVersion(
                    version_id=version_id, run_id=run_id, iteration_t=iteration_t,
                    m=proposal.m, parent_version=incumbent_id,
                    source_sha256=check.digest, source_artifact_id=str(proposal.path),
                    origin="revision", state=PolicyState.REJECTED,
                    static_validation_status="FAIL",
                    failure_reason=check.reason,
                ))
                self.deps.registry.add_revision(Revision(
                    iteration_t=iteration_t, m=proposal.m, input_policy_id=incumbent_id,
                    output_policy_id=version_id, status="STATIC_REJECTED",
                    feedback_digest=revision.feedback_digest,
                    proposal_digest=check.digest, failure_reason=check.reason,
                ))
                rejected.append((version_id, f"static: {check.reason}"))
                self._emit(run_id, "policy.rejected", version_id=version_id,
                           stage="static", reason=check.reason)
                continue

            sandbox_error = self._probe_policy(proposal.path)
            state = PolicyState.ELIGIBLE if sandbox_error is None else PolicyState.REJECTED
            self.deps.registry.add_policy(PolicyVersion(
                version_id=version_id, run_id=run_id, iteration_t=iteration_t,
                m=proposal.m, parent_version=incumbent_id,
                source_sha256=check.digest, source_artifact_id=str(proposal.path),
                origin="revision", state=state,
                static_validation_status="PASS",
                sandbox_validation_status="PASS" if sandbox_error is None else "FAIL",
                failure_reason=sandbox_error,
            ))
            self.deps.registry.add_revision(Revision(
                iteration_t=iteration_t, m=proposal.m, input_policy_id=incumbent_id,
                output_policy_id=version_id,
                status="ELIGIBLE" if sandbox_error is None else "SANDBOX_REJECTED",
                feedback_digest=revision.feedback_digest,
                proposal_digest=check.digest, failure_reason=sandbox_error,
            ))
            if sandbox_error is None:
                eligible.append(self.deps.registry.get_policy(version_id))  # type: ignore[arg-type]
            else:
                rejected.append((version_id, f"sandbox: {sandbox_error}"))
                self._emit(run_id, "policy.rejected", version_id=version_id,
                           stage="sandbox", reason=sandbox_error)

        return eligible, rejected

    def _record_replays(self, episodes: Sequence[Any]) -> None:
        """Persist replay episodes, insisting that a replay is reproducible.

        ``H_t`` is frozen and a policy is deterministic, so replaying the same
        (policy, world, rho) **must** yield the same episode. Re-recording an
        identical episode is a no-op — this happens legitimately, because the
        incumbent is re-scored on the growing ``H_t`` each round. Recording a
        *different* episode for the same triple means replay is nondeterministic,
        which would invalidate every cross-round comparison, so it is a hard
        failure rather than an overwrite.
        """
        existing = {
            (e.world_id, e.policy_version_id, e.rho): e
            for e in self.deps.registry.list_replays()
        }
        for episode in episodes:
            key = (episode.world_id, episode.policy_version_id, episode.rho)
            previous = existing.get(key)
            if previous is None:
                self.deps.registry.add_replay(episode)
                existing[key] = episode
                continue
            if previous.replay_digest != episode.replay_digest:
                raise IntegrityError(
                    f"replay is not deterministic: {key} produced digest "
                    f"{episode.replay_digest} but {previous.replay_digest} was already "
                    "recorded. A fixed H_t must replay identically."
                )

    def _record_score(self, record: ScoreRecord) -> None:
        """Persist an Eq(1) result under the same reproducibility rule."""
        previous = next(
            (
                s
                for s in self.deps.registry.list_scores(
                    policy_version_id=record.policy_version_id,
                    world_id=record.world_id,
                )
            ),
            None,
        )
        if previous is None:
            self.deps.registry.add_score(record)
            return
        if (
            previous.value != record.value
            or previous.hv != record.hv
            or previous.parallel_penalty != record.parallel_penalty
        ):
            raise IntegrityError(
                f"Eq(1) is not reproducible for policy {record.policy_version_id} on "
                f"world {record.world_id}: V={record.value} but {previous.value} was "
                "already recorded"
            )

    def _probe_policy(self, path: Path) -> str | None:
        """Load the policy in an isolated worker and call it once on a trivial prefix."""
        from ..policy.protocol import PrefixObservation, PublicNode

        probe = PrefixObservation(
            nodes=(PublicNode(node_id="root", parent_id=None, creation_index=0, depth=0,
                              score=0.0, evaluator_status="SUCCESS",
                              error_type=None, summary=""),),
            legal_actions=("root",),
            completed_rounds=0,
            max_parallelism=self.loop.W,
        )
        try:
            session = self.deps.policy_runner.open(
                policy_path=path, policy_version_id="probe", session_key="probe",
            )
        except Exception as exc:
            return f"worker did not start: {type(exc).__name__}: {exc}"
        try:
            result = session.select(probe, 0.0)
        except PolicyRejected as exc:
            return str(exc)[:300]
        except Exception as exc:
            return f"{type(exc).__name__}: {exc}"[:300]
        finally:
            session.close()
        if not isinstance(result, (list, tuple)):
            return f"select() returned {type(result).__name__}, not a list"
        return None

    def _select_policy(
        self, *, run_id: str, iteration_t: int, incumbent_id: str,
        incumbent_path: Path, history_worlds: Sequence[SealedWorld],
    ) -> tuple[Selection, int]:
        incumbent_policy = self.deps.registry.get_policy(incumbent_id)
        if incumbent_policy is None:
            raise IntegrityError(f"incumbent policy {incumbent_id} is missing from the registry")

        incumbent_score, episodes_total = self._score_policy(
            version=incumbent_policy, path=incumbent_path, worlds=history_worlds, m=0,
        )
        if incumbent_score is None:
            raise IntegrityError(
                "the incumbent failed to replay on H_t; without it the selection has no "
                "reference point and monotonicity cannot be guaranteed"
            )
        scored: list[PolicyScore] = [incumbent_score]
        current_version = incumbent_policy
        current_path = incumbent_path
        current_score = incumbent_score
        tables_emitted: set[str] = set()
        feedback = list(self._feedback(history_worlds, current_score, tables_emitted))

        # Paper §3: evaluate pi_t^m before asking the developer to write
        # pi_t^(m+1). Each call sees the current version's trajectories and
        # scores plus the feedback from all earlier revisions.
        for m in range(1, self.loop.M):
            eligible, rejected = self._revision_candidates(
                run_id=run_id, iteration_t=iteration_t,
                incumbent_path=current_path, incumbent_id=current_version.version_id,
                feedback=feedback, revision_m=m,
            )
            for version_id, reason in rejected:
                self._note(f"t={iteration_t}: revision {version_id} rejected ({reason})")
                feedback.append(f"revision m={m} rejected: {reason}")
            if not eligible:
                continue
            version = eligible[0]
            path = Path(version.source_artifact_id)
            score, episodes = self._score_policy(
                version=version, path=path, worlds=history_worlds, m=m,
            )
            episodes_total += episodes
            if score is None:
                self._note(
                    f"t={iteration_t}: policy {version.version_id} failed replay on H_t "
                    "and is not a candidate"
                )
                feedback.append(f"revision m={m} failed replay on H_t")
                continue
            scored.append(score)
            current_version, current_path, current_score = version, path, score
            feedback.extend(self._feedback(history_worlds, current_score, tables_emitted))

        selection = select(deployed=incumbent_score, candidates=[
            s for s in scored if s.policy_version_id != incumbent_id
        ])
        return selection, episodes_total

    def _score_policy(
        self, *, version: PolicyVersion, path: Path, worlds: Sequence[SealedWorld], m: int,
    ) -> tuple[PolicyScore | None, int]:
        values: list[float] = []
        digests: list[str] = []
        episodes_total = 0
        for world in worlds:
            try:
                episodes = replay_all_rhos(
                    world, self._policy_select_fn(path),
                    policy_version_id=version.version_id,
                    rho_grid=self.loop.rho_grid,
                    max_parallelism=self.loop.W,
                    max_rounds=self.loop.K2,
                )
            except PolicyRejected as exc:
                self._emit(version.run_id, "policy.replay_failed",
                           version_id=version.version_id, world_id=world.world_id,
                           reason=str(exc)[:300])
                return None, episodes_total
            for episode in episodes:
                if episode.discovery_agent_calls or episode.evaluator_calls:
                    raise IntegrityError(
                        f"replay of {version.version_id} on {world.world_id} at rho="
                        f"{episode.rho} reported agent={episode.discovery_agent_calls} "
                        f"evaluator={episode.evaluator_calls}; replay must touch neither"
                    )
            self._record_replays(episodes)
            episodes_total += len(episodes)

            result = equation1(
                episodes,
                b_count=world.b_count,
                s_best=world.s_best,
                lambda_weight=self.loop.lambda_weight,
                expected_rhos=self.loop.rho_grid,
            )
            self._record_score(ScoreRecord(
                policy_version_id=version.version_id,
                world_id=world.world_id,
                b_count=world.b_count,
                s_best=world.s_best,
                five_points=tuple((int(s.n), float(s.attained_quality)) for s in episodes),
                frontier=tuple(result.frontier),
                hv=result.hv,
                parallel_penalty=result.penalty,
                lambda_weight=self.loop.lambda_weight,
                value=result.value,
                score_formula_version=SCORE_FORMULA_VERSION,
            ))
            values.append(result.value)
            digests.append(world.world_digest())

        if not values:
            return None, episodes_total
        self.deps.registry.update_policy_state(
            version.version_id, PolicyState.REPLAYED, sandbox_validation_status="PASS",
        )
        return PolicyScore(
            policy_version_id=version.version_id,
            m=m,
            module_digest=version.source_sha256,
            per_world_values=tuple(values),
            per_world_digests=tuple(digests),
        ), episodes_total

    def _feedback(
        self, worlds: Sequence[SealedWorld], score: PolicyScore,
        tables_emitted: set[str] | None = None,
    ) -> tuple[str, ...]:
        """Current version's measured replay scores and revealed trajectories.

        ``tables_emitted`` carries the worlds whose node table has already been
        written into this selection round's feedback. The same records do not
        depend on the policy or the rho, so writing them once per world instead
        of once per episode is what keeps the developer's prompt inside the
        model's context window; passing ``None`` restores self-contained output
        for a caller that renders one block in isolation.
        """
        lines = [
            f"policy {score.policy_version_id} (m={score.m}) mean V={score.mean_value:.12f}"
        ]
        for world in worlds:
            records = self.deps.registry.list_scores(
                policy_version_id=score.policy_version_id, world_id=world.world_id,
            )
            if len(records) != 1:
                raise IntegrityError(
                    f"missing Eq(1) feedback for {score.policy_version_id} on {world.world_id}"
                )
            record = records[0]
            lines.append(
                f"world {world.world_id}: B={world.b_count}, S={world.s_best:.6f}, "
                f"HV={record.hv:.12f}, penalty={record.parallel_penalty:.12f}, "
                f"V={record.value:.12f}, frontier={record.frontier}"
            )
            episodes = self.deps.registry.list_replays(
                policy_version_id=score.policy_version_id, world_id=world.world_id,
            )
            if len(episodes) != len(self.loop.rho_grid):
                raise IntegrityError(
                    f"missing replay trajectories for {score.policy_version_id} on {world.world_id}"
                )
            ordered = sorted(episodes, key=lambda e: e.rho)
            if tables_emitted is None or world.world_id not in tables_emitted:
                lines.extend(render_node_table(ordered))
                if tables_emitted is not None:
                    tables_emitted.add(world.world_id)
            for episode in ordered:
                lines.append(
                    f"  rho={episode.rho}: n={episode.n}, k={episode.kstar}, "
                    f"best={episode.best_score:.6f}, a={episode.attained_quality:.6f}, "
                    f"stop={episode.stop_reason.value}, batches={episode.selected_batches}, "
                    f"revealed={episode.revealed_order}"
                )
                lines.extend(render_episode_steps(episode))
        return tuple(lines)

    @staticmethod
    def _digest_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "Controller", "ControllerDeps", "RunOutcome", "SelectionSummary", "WorldSummary",
]
