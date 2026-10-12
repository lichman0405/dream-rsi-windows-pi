"""Run one phase of the MOF screening experiment, block by block.

A *block* is one ``(phase, split seed, repeat)`` pair and it runs three arms --
A (Dream-RSI), B (frozen initial policy), C (frozen serial policy) -- one after
another. The block is also the statistical unit: the design counts paired search
runs, never nodes or materials, and this module is careful to keep that
distinction visible in what it writes.

What one block does:

1. builds the arm's :class:`~dream_rsi.config.Config`, a :class:`TokenBudget`
   with the arm's hard ceiling and its own cost ledger, the MOF evaluator (which
   holds the development labels in memory) and the Pi backend in no-tools,
   response-only mode;
2. scores the *starting* program once, so the baseline is a candidate too;
3. runs the controller -- worlds, policy selection, replay, sealing;
4. reads the sealed tree back and turns it into an ordered list of findings,
   each joined to the ledger row that paid for it;
5. freezes the best completed candidate at every budget checkpoint, and the
   block's final candidate at the arm's ceiling.

Nothing here computes an effect. Selection uses the development score only, the
final test pool is not opened, and ``summary.json`` says so in as many words.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..adapters.base import TaskSpec
from ..adapters.pi import PiDiscoveryBackend
from ..config import Config, LoopConfig, Timeouts, load_config
from ..domain.models import RunState, WorldState
from ..runtime.controller import Controller, ControllerDeps
from ..runtime.llm_policy_developer import LlmPolicyDeveloper
from ..runtime.online import CallBudget
from ..policy.runner import SubprocessPolicyRunner
from ..runtime.token_budget import TokenBudget
from ..storage.cas import ContentAddressedStore
from ..storage.sqlite_registry import Registry
from .protocol import (
    AGENT_MEMORY_MB,
    AGENT_TIMEOUT_S,
    BlockPlan,
    EVALUATOR_MEMORY_MB,
    EVALUATOR_TIMEOUT_S,
    K,
    LAMBDA_WEIGHT,
    LIVE_RHO,
    MODEL,
    POLICY_MEMORY_MB,
    POLICY_SELECT_TIMEOUT_S,
    Protocol,
    development_reserve,
)
from .scoring import MofPool, MofScreeningEvaluator

#: Reply ceiling for one Pi call. A candidate file plus its explanation fits
#: comfortably; the ceiling exists so a runaway generation is cut off rather
#: than billed to the end of the context window.
MAX_OUTPUT_BYTES = 32768

#: Replay rounds per episode (the paper's K2). Fixed here and recorded in the
#: block record; the design fixes the rho grid but leaves this to the runner.
K2 = 6


@dataclass
class Finding:
    """One non-root candidate: what it scored, and what it cost to produce."""

    node_id: str
    world_id: str
    outer_t: int
    round_index: int
    score: float
    error_type: str | None
    candidate_sha256: str
    workspace: str
    ledger_seq: int | None
    charged: int
    cumulative: int
    usage_reported: bool

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class FrozenCandidate:
    """A candidate the block may evaluate on the sealed pool, frozen by content."""

    checkpoint: int
    origin: str                     # "baseline" | node id
    node_id: str | None
    development_S: float
    cumulative_tokens: int
    candidate_sha256: str
    source_path: str
    usage_reported: bool
    error_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class BlockRecord:
    plan: dict[str, Any]
    run_id: str
    status: str
    protocol_digest: str
    config_digest: str | None = None
    evaluator_digest: str | None = None
    baseline_S: float | None = None
    backend_id: str = ""
    budget_limit: int = 0
    budget: dict[str, Any] = field(default_factory=dict)
    ledger: dict[str, Any] = field(default_factory=dict)
    findings: list[dict[str, Any]] = field(default_factory=list)
    frozen: list[dict[str, Any]] = field(default_factory=list)
    final_candidate: dict[str, Any] | None = None
    worlds: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _read_ledger(path: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Index the cost ledger by settle order, and return its raw rows."""
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return {}, rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(row)
    return {int(row["seq"]): row for row in rows if row.get("event") == "request"}, rows


def _ledger_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Totals a report can quote, with unknown usage kept visible as unknown."""
    requests = [row for row in rows if row.get("event") == "request"]
    refused = [row for row in rows if row.get("event") == "refused"]
    unreported = [row for row in requests if not row.get("usage_reported")]
    tokens = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "reasoning": 0}
    for row in requests:
        block = row.get("tokens") or {}
        for key in tokens:
            tokens[key] += int(block.get(key) or 0)
    roles: dict[str, int] = {}
    for row in requests:
        roles[str(row.get("role"))] = roles.get(str(row.get("role")), 0) + 1
    costs = [row.get("cost") or {} for row in requests]
    known_costs = [cost for cost in costs if cost.get("known")]
    return {
        "requests": len(requests),
        "refusals": len(refused),
        "requests_without_reported_usage": len(unreported),
        "tokens": tokens,
        "charged_total": sum(int(row.get("charged") or 0) for row in requests),
        "requests_by_role": roles,
        "duration_s_total": round(sum(float(row.get("duration_s") or 0.0) for row in requests), 3),
        "cost": {
            "known": bool(known_costs) and len(known_costs) == len(requests),
            "currency": known_costs[0].get("currency") if known_costs else None,
            "amount": (sum(float(cost.get("amount") or 0.0) for cost in known_costs)
                       if known_costs else None),
            "requests_without_cost": len(requests) - len(known_costs),
            "note": ("money is reported only where the service reported it; no price "
                     "table is assumed here"),
        },
    }


class MofExperimentRunner:
    """Drives every block of one phase inside a single vaulted process."""

    def __init__(
        self,
        *,
        project_root: Path,
        protocol: Protocol,
        task_dir: Path,
        runner_script: Path,
        out_root: Path,
        dry_run: bool = False,
        backend_factory: Any | None = None,
        budget_scale: float = 1.0,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.protocol = protocol
        self.task_dir = Path(task_dir).resolve()
        self.runner_script = Path(runner_script).resolve()
        self.out_root = Path(out_root).resolve()
        self.dry_run = dry_run
        #: Test seam only. The acceptance suite injects a factory here so the
        #: whole block pipeline -- ledger, findings, freezing -- can be exercised
        #: without a live model. A live phase never sets it.
        self.backend_factory = backend_factory
        #: Test seam: shrinks the per-arm ceiling so a smoke block finishes. The
        #: applied ceiling is written into the block record either way.
        self.budget_scale = float(budget_scale)
        self.out_root.mkdir(parents=True, exist_ok=True)
        self.prepared_root = Path(protocol.prepared_root)

    # ------------------------------------------------------------------ config

    def _config_for(self, plan: BlockPlan) -> Config:
        spec = self.protocol.spec
        method = spec.methods[plan.method]
        base = load_config(
            project_root=self.project_root,
            backend="pi",
            developer="llm" if method.uses_developer else "scripted",
            provider=MODEL["provider"],
            model=MODEL["model"],
            task="mof_screening",
            run_id=plan.run_id,
            pi_no_tools=True,
            pi_response_only=True,
        )
        loop = LoopConfig(
            K1=method.K1, K2=K2, W=method.W, M=method.M,
            lambda_weight=LAMBDA_WEIGHT, outer_rounds=spec.worlds,
            live_rho=LIVE_RHO,
        )
        timeouts = Timeouts(
            agent_s=AGENT_TIMEOUT_S, evaluator_s=EVALUATOR_TIMEOUT_S,
            policy_select_s=POLICY_SELECT_TIMEOUT_S,
            agent_memory_mb=AGENT_MEMORY_MB, evaluator_memory_mb=EVALUATOR_MEMORY_MB,
            policy_memory_mb=POLICY_MEMORY_MB,
        )
        return replace(
            base, loop=loop, timeouts=timeouts,
            incumbent_policy_path=(self.project_root / method.incumbent_policy).resolve(),
        )

    def _pool(self, split_seed: int) -> MofPool:
        return MofPool.development(self.prepared_root, split_seed)

    # ------------------------------------------------------------------ one block

    def run_block(
        self, plan: BlockPlan, labels: Mapping[str, float] | None = None,
    ) -> BlockRecord:
        config = self._config_for(plan)
        spec = self.protocol.spec
        method = spec.methods[plan.method]
        record = BlockRecord(
            plan=plan.to_dict(), run_id=plan.run_id, status="RUNNING",
            protocol_digest=self.protocol.digest,
        )
        run_root = config.run_root(plan.run_id)
        run_root.mkdir(parents=True, exist_ok=True)

        budget = TokenBudget(
            limit=max(1, int(spec.token_budget * self.budget_scale)),
            ledger_path=run_root / "cost-ledger.jsonl",
            context={"run_id": plan.run_id, "phase": plan.phase, "method": plan.method,
                     "split_seed": plan.split_seed, "repeat": plan.repeat},
            development_reserve=development_reserve(spec) if method.uses_developer else 0,
        )

        pool = self._pool(plan.split_seed)
        pool_labels = dict(labels) if labels is not None else pool.load_labels()
        evaluator = MofScreeningEvaluator(
            pool=pool, labels=pool_labels, runner_script=self.runner_script,
            workdir_root=run_root / "eval", k=K, timeout_s=EVALUATOR_TIMEOUT_S,
            memory_limit_mb=EVALUATOR_MEMORY_MB,
        )
        record.evaluator_digest = evaluator.spec_digest
        record.config_digest = config.cfg_digest

        # The starting program is a candidate in its own right: every arm is
        # free to end where it began, and a report that hid that would make a
        # flat arm look like a missing measurement.
        baseline = evaluator.evaluate(config.seed_workspace, invocation_id=f"{plan.run_id}-baseline")
        record.baseline_S = baseline.score

        if self.dry_run:
            record.status = "DRY_RUN"
            record.budget = budget.snapshot()
            _write_json(run_root / "block-summary.json", record.to_dict())
            return record

        backend = self.backend_factory(config, plan, budget) if self.backend_factory else (
            PiDiscoveryBackend(
                workdir_root=run_root / "agent",
                provider=MODEL["provider"], model=MODEL["model"],
                timeout_s=AGENT_TIMEOUT_S, memory_limit_mb=AGENT_MEMORY_MB,
                no_tools=True, response_only=True, standard_library_only=False,
                token_budget=budget, max_output_bytes=MAX_OUTPUT_BYTES,
            )
        )
        record.backend_id = str(getattr(backend, "backend_id", "unknown"))
        record.budget_limit = budget.limit
        developer = (
            LlmPolicyDeveloper(
                workdir_root=run_root / "policy_agent",
                provider=MODEL["provider"], model=MODEL["model"],
                timeout_s=AGENT_TIMEOUT_S, memory_limit_mb=AGENT_MEMORY_MB,
                token_budget=budget, max_output_bytes=MAX_OUTPUT_BYTES,
            )
            if method.uses_developer else None
        )

        registry = Registry(config.db_path)
        store = ContentAddressedStore(config.cas_root)
        policy_runner = SubprocessPolicyRunner(
            workdir_root=run_root / "policy_sessions",
            timeout_s=POLICY_SELECT_TIMEOUT_S, memory_limit_mb=POLICY_MEMORY_MB,
        )
        deps = ControllerDeps(
            backend=backend, evaluator=evaluator, registry=registry, store=store,
            policy_runner=policy_runner, task=_task_spec(config), developer=developer,
            budget=CallBudget(spec.session_cap),
        )
        try:
            outcome = Controller(config, deps).run()
            record.status = outcome.state.value
            record.notes = list(outcome.notes)
            record.worlds = [w.__dict__ for w in outcome.worlds]
        except Exception as exc:  # noqa: BLE001 - a dead block is data, not a crash
            record.status = "FAILED"
            record.error = f"{type(exc).__name__}: {exc}"[:600]
            record.notes = list(deps.notes)
        finally:
            registry.close()

        charge_by_seq, ledger_rows = _read_ledger(run_root / "cost-ledger.jsonl")
        record.ledger = _ledger_summary(ledger_rows)
        record.budget = budget.snapshot()

        findings = self._collect_findings(config, charge_by_seq)
        record.findings = [finding.to_dict() for finding in findings]
        frozen = self._freeze(plan, config, baseline.score, findings)
        record.frozen = [candidate.to_dict() for candidate in frozen]
        if frozen:
            record.final_candidate = frozen[-1].to_dict()
        _write_json(run_root / "block-summary.json", record.to_dict())
        return record

    # ------------------------------------------------------------------ findings

    def _collect_findings(
        self, config: Config, charge_by_seq: Mapping[int, Mapping[str, Any]],
    ) -> list[Finding]:
        """Every non-root candidate, ordered by the ledger row that paid for it.

        The order is *settle* order, which is the order in which the loop could
        first act on each score. Ordering by wall clock instead would place a
        candidate's tokens before its own result was knowable, and ordering by
        tree position would ignore that eight workers finish out of order.
        """
        store = ContentAddressedStore(config.cas_root)
        registry = Registry(config.db_path)
        findings: list[Finding] = []
        try:
            for world in registry.list_worlds(config.run_id or ""):
                if world.state not in (WorldState.SEALED_SCOREABLE, WorldState.SEALED_UNSCOREABLE):
                    continue
                map_path = (config.run_root(config.run_id or "")
                            / "workspaces" / f"{world.world_id}.map.json")
                workspaces: dict[str, str] = {}
                if map_path.is_file():
                    workspaces = json.loads(map_path.read_text(encoding="utf-8"))
                for node in registry.list_nodes(world.world_id):
                    if node.node_id == world.root_id:
                        continue
                    generation = store.get_json(node.agent_session_digest) if (
                        node.agent_session_digest
                    ) else {}
                    generation_record = (generation or {}).get("record") or {}
                    seq = generation_record.get("ledger_seq")
                    seq = int(seq) if isinstance(seq, int) else None
                    row = charge_by_seq.get(seq) if seq is not None else None
                    charged = int(row.get("charged") or 0) if row else 0
                    usage_reported = bool(row.get("usage_reported")) if row else False
                    workspace = workspaces.get(node.node_id, "")
                    sha = ""
                    candidate = Path(workspace) / "candidate.py" if workspace else None
                    if candidate is not None and candidate.is_file():
                        sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
                    findings.append(Finding(
                        node_id=node.node_id, world_id=world.world_id,
                        outer_t=world.outer_t, round_index=node.round_index,
                        score=float(node.score), error_type=node.error_type,
                        candidate_sha256=sha, workspace=workspace, ledger_seq=seq,
                        charged=charged, cumulative=0, usage_reported=usage_reported,
                    ))
        finally:
            registry.close()

        findings.sort(key=lambda f: (f.ledger_seq is None, f.ledger_seq or 0, f.node_id))
        running = 0
        for finding in findings:
            running += finding.charged
            finding.cumulative = running
        return findings

    # ------------------------------------------------------------------ freezing

    def _freeze(
        self, plan: BlockPlan, config: Config, baseline_S: float, findings: Sequence[Finding],
    ) -> list[FrozenCandidate]:
        """Best completed candidate at every checkpoint, then the block's final pick.

        Ties on the development score go to the candidate that finished first, as
        the design fixes. The starting program competes on equal terms: it is the
        only entry at zero tokens, and an arm that never beat it selects it.
        """
        spec = self.protocol.spec
        baseline_source = (config.seed_workspace / "candidate.py").read_text(encoding="utf-8")
        baseline_sha = hashlib.sha256(baseline_source.encode("utf-8")).hexdigest()

        frozen: list[FrozenCandidate] = []
        for checkpoint in spec.checkpoints:
            eligible = [f for f in findings if f.cumulative <= checkpoint]
            best = _best(findings=eligible, baseline_S=baseline_S)
            frozen.append(self._materialize(
                plan, config, checkpoint, best, baseline_S, baseline_sha, baseline_source,
            ))
        return frozen

    def _materialize(
        self, plan: BlockPlan, config: Config, checkpoint: int, best: Finding | None,
        baseline_S: float, baseline_sha: str, baseline_source: str,
    ) -> FrozenCandidate:
        """Copy the chosen candidate's source into the block's freeze area."""
        target_dir = config.run_root(plan.run_id) / "frozen" / f"budget-{checkpoint}"
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / "candidate.py"
        if best is None or not best.candidate_sha256:
            target.write_text(baseline_source, encoding="utf-8", newline="\n")
            return FrozenCandidate(
                checkpoint=checkpoint, origin="baseline", node_id=None,
                development_S=baseline_S, cumulative_tokens=0,
                candidate_sha256=baseline_sha, source_path=str(target),
                usage_reported=True,
            )
        source = Path(best.workspace) / "candidate.py"
        if not source.is_file():
            target.write_text(baseline_source, encoding="utf-8", newline="\n")
            return FrozenCandidate(
                checkpoint=checkpoint, origin="baseline", node_id=None,
                development_S=baseline_S, cumulative_tokens=0,
                candidate_sha256=baseline_sha, source_path=str(target),
                usage_reported=True,
                error_type="CHOSEN_CANDIDATE_SOURCE_MISSING",
            )
        shutil.copyfile(source, target)
        return FrozenCandidate(
            checkpoint=checkpoint, origin=best.node_id, node_id=best.node_id,
            development_S=best.score, cumulative_tokens=best.cumulative,
            candidate_sha256=best.candidate_sha256, source_path=str(target),
            usage_reported=best.usage_reported, error_type=best.error_type,
        )

    # ------------------------------------------------------------------ phase

    def run_phase(self, plans: Sequence[BlockPlan]) -> dict[str, Any]:
        """Run every block, writing one record file per block plus a phase index."""
        labels_by_split: dict[int, dict[str, float]] = {}
        for plan in plans:
            if plan.split_seed not in labels_by_split:
                labels_by_split[plan.split_seed] = self._pool(plan.split_seed).load_labels()

        records: list[dict[str, Any]] = []
        index_path = self.out_root / f"{self.protocol.phase}-blocks.jsonl"
        for plan in plans:
            record = self.run_block(plan, labels=labels_by_split[plan.split_seed])
            records.append(record.to_dict())
            with index_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record.to_dict(), sort_keys=True, default=str) + "\n")
        return {
            "phase": self.protocol.phase,
            "protocol_digest": self.protocol.digest,
            "blocks": records,
        }


def _best(*, findings: Sequence[Finding], baseline_S: float) -> Finding | None:
    """Highest development score; ties to the earlier finisher; baseline at 0 tokens."""
    if not findings:
        return None
    ranked = sorted(findings, key=lambda f: (-f.score, f.ledger_seq is None,
                                             f.ledger_seq or 0, f.node_id))
    winner = ranked[0]
    if winner.score <= baseline_S:
        return None
    return winner


def _task_spec(config: Config) -> TaskSpec:
    instructions = config.task_md.read_text(encoding="utf-8") if config.task_md.is_file() else ""
    return TaskSpec(
        task_id=config.task_dir.name, entry_filename=config.entry_filename,
        instructions=instructions, seed_workspace=config.seed_workspace,
        is_paper_benchmark=False,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


__all__ = [
    "BlockRecord", "Finding", "FrozenCandidate", "K2", "MAX_OUTPUT_BYTES",
    "MofExperimentRunner",
]
