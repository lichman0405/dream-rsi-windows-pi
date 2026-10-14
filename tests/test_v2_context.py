"""v2 context inheritance, complete history, storage and replay regression evidence."""
from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from dream_rsi.adapters._pi_worker import HistoryTrace
from dream_rsi.adapters.base import GeneratedAttempt, HistoryEntry, TaskSpec
from dream_rsi.adapters.dsh import build_prompt
from dream_rsi.cli import _build_controller
from dream_rsi.config import Config, LoopConfig
from dream_rsi.domain.errors import IntegrityError
from dream_rsi.domain.models import EvaluatorStatus
from dream_rsi.domain.tree import SealedWorld
from dream_rsi.evaluator.protocol import EvaluationResult
from dream_rsi.policy.protocol import PrefixObservation
from dream_rsi.runtime.context import (
    ContinuationContext,
    canonical,
    export_world,
    load_observation,
    validate_world_artifacts,
)
from dream_rsi.runtime.online import run_online_rollout
from dream_rsi.runtime.policy_development import validate_policy_source
from dream_rsi.runtime.replay import replay_all_rhos, replay_world, validate_episode_trace
from dream_rsi.runtime.trajectory import decode_any, node_table
from dream_rsi.storage.cas import ContentAddressedStore


class DiagnosticBackend:
    backend_id = "diagnostic-fixture"
    is_live = False

    def __init__(self) -> None:
        self.contexts: list[ContinuationContext] = []

    def generate(self, *, task: TaskSpec, parent_workspace: Path,
                 history: Any, output_workspace: Path, invocation_id: str,
                 context: ContinuationContext | None = None) -> GeneratedAttempt:
        assert context is not None and context.directory is not None
        self.contexts.append(context)
        # The new candidate needs the parent's measured diagnostic, not its file.
        branch = context.to_dict()["branch"]
        value = branch[-1]["public"].get("diagnostics", {}).get("next_value", 2)
        (output_workspace / task.entry_filename).write_text(str(value), encoding="utf-8")
        index = json.loads((context.directory / "index.json").read_text(encoding="utf-8"))
        assert index == context.to_dict()
        assert "PRIVATE_TEST_INPUT" not in canonical(index)
        return GeneratedAttempt(invocation_id=invocation_id, backend_id=self.backend_id,
            model="fixture", workspace_path=output_workspace, entry_file_digest=None,
            succeeded=True, duration_s=0, summary=f"produced {value}",
            generation_record={"complete_message": "x" * 7000})


class DiagnosticEvaluator:
    spec_digest = "fixed"

    def evaluate(self, candidate_dir: Path, *, invocation_id: str) -> EvaluationResult:
        value = int((candidate_dir / "candidate.py").read_text(encoding="utf-8"))
        return EvaluationResult(score=value / 10, status=EvaluatorStatus.SUCCESS,
            duration_s=0, result_digest="untrusted-adapter-hash",
            detail={"PRIVATE_TEST_INPUT": "never expose this"},
            public_detail={"next_value": value + 1})


def _grow(tmp_path: Path, store: ContentAddressedStore, backend: Any,
          history: tuple[HistoryEntry, ...] = (), t: int = 0,
          rounds: int = 3, policy: Any = None) -> Any:
    seed = tmp_path / "seed"
    seed.mkdir(exist_ok=True)
    (seed / "candidate.py").write_text("0", encoding="utf-8")
    return run_online_rollout(world_id=f"w{t}", outer_t=t, task=TaskSpec(
        task_id="fixture", instructions="read diagnostics", entry_filename="candidate.py", seed_workspace=seed),
        policy_select=policy or (lambda obs, rho: [obs.legal_actions[-1]]),
        backend=backend, evaluator=DiagnosticEvaluator(), workspace_root=tmp_path / f"w{t}",
        root_workspace=seed, history=history, max_workers=2, max_rounds=rounds,
        live_rho=.5, store_digest=store.put_tree, store=store, root_id=f"w{t}-root")


def test_branch_inherits_measured_feedback_and_full_generation(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    backend = DiagnosticBackend()
    tree, _ = _grow(tmp_path, store, backend)
    assert [n.score for n in tree.nodes()] == [0, .2, .3, .4]
    assert [len(context.branch_observations) for context in backend.contexts] == [1, 2, 3]
    validate_world_artifacts(store, tree.nodes(), [])
    parent = tree.nodes()[1]
    generation = store.get_json(parent.agent_session_digest or "")
    assert len(generation["record"]["complete_message"]) == 7000
    assert store.get_json(parent.evaluator_result_digest or "")["detail"]["PRIVATE_TEST_INPUT"]


def test_parallel_branches_do_not_inherit_sibling_observations(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    backend = DiagnosticBackend()
    def policy(obs: PrefixObservation, rho: float) -> list[str]:
        return [obs.legal_actions[0]] if obs.completed_rounds == 0 else list(obs.legal_actions)
    tree, _ = _grow(tmp_path, store, backend, rounds=2, policy=policy)
    for context in backend.contexts:
        branch = context.to_dict()["branch"]
        assert branch[-1]["node_id"] == context.parent_id
        assert len(branch) == (1 if context.parent_id == tree.root_id else 2)
    assert len(tree) == 4
    assert sorted(n.creation_index for n in tree.nodes()) == list(range(4))
    validate_world_artifacts(store, tree.nodes(), [])


def test_all_seven_completed_worlds_remain_readable(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    backend = DiagnosticBackend()
    history: tuple[HistoryEntry, ...] = ()
    for t in range(8):
        tree, _ = _grow(tmp_path, store, backend, history, t=t, rounds=1)
        exports = [json.loads(entry.contents_json) for entry in history]
        validate_world_artifacts(store, tree.nodes(), exports)
        contents = export_world(store, tree.nodes(), world_id=tree.world_id,
                                outer_t=t, world_digest=tree.world_digest())
        history += (HistoryEntry(outer_t=t, world_digest=tree.world_digest(),
            node_count=len(tree), best_score=.2, policy_version_id="fixture",
            contents_json=canonical(contents)),)
    last = backend.contexts[-1]
    assert [w["outer_t"] for w in last.to_dict()["completed_worlds"]] == list(range(7))
    prompt = build_prompt(TaskSpec("fixture", "candidate.py", "task", tmp_path / "seed"), history[:7], context=last)
    assert "world t=0:" in prompt and "world t=6:" in prompt
    earliest = last.to_dict()["completed_worlds"][0]["nodes"][1]["observation"]
    assert last.directory is not None
    restored = last.directory / last.workspace_path(earliest["workspace_snapshot_digest"])
    assert (restored / "candidate.py").read_text(encoding="utf-8") == "2"


def test_snapshot_restores_files_empty_directories_and_checks_blobs(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    source = tmp_path / "source"
    (source / "empty").mkdir(parents=True)
    (source / "candidate.py").write_text("original", encoding="utf-8")
    snapshot = store.put_tree(source)
    shutil.rmtree(source)
    destination = tmp_path / "restored"
    store.restore_tree(snapshot, destination)
    assert (destination / "candidate.py").read_text(encoding="utf-8") == "original"
    assert (destination / "empty").is_dir()
    assert store.put_tree(destination) == snapshot
    file_digest = next(marker for name, marker in store.tree_entries(snapshot) if name == "candidate.py")
    store.path_for(file_digest).write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="corrupt"):
        store.restore_tree(snapshot, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("name", ["../escape", "C:/escape", "x\\escape", "a./b", "/escape"])
def test_restore_rejects_manifest_path_escape(tmp_path: Path, name: str) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    data = store.put_bytes(b"data")
    manifest = store.put_bytes(f"TREE2\n{name}\0{data}\n".encode())
    with pytest.raises(IntegrityError):
        store.restore_tree(manifest, tmp_path / "out")


def test_legacy_hash_manifest_and_legacy_observation_are_not_complete(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    with pytest.raises(IntegrityError, match="legacy"):
        store.restore_tree(store.put_bytes(b"candidate.py\0" + b"a" * 64 + b"\n"), tmp_path / "out")
    with pytest.raises(IntegrityError, match="legacy"):
        load_observation(store, store.put_json({"old": "summary"}))


def test_replay_trace_reconstructs_observations_without_external_calls(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    backend = DiagnosticBackend()
    tree, _ = _grow(tmp_path, store, backend)
    world = SealedWorld.from_nodes(tree.world_id, tree.nodes(), root_id=tree.root_id)
    before = len(backend.contexts)
    episode = replay_world(world, lambda obs, rho: [obs.legal_actions[-1]],
        policy_version_id="fixture", rho=.5, max_parallelism=2, max_rounds=4)
    validate_episode_trace(world, episode, 2, 4)
    assert len(backend.contexts) == before
    assert episode.discovery_agent_calls == episode.evaluator_calls == 0
    # The trace stores each node record once and refers to it by id, so a step
    # is read through the decoder rather than by index into a self-contained
    # blob. What the step *contains* is unchanged.
    step = decode_any(episode)[1]
    assert step["observation"]["nodes"][-1]["observation_json"] == tree.nodes()[1].public_observation_json
    assert "PRIVATE_TEST_INPUT" not in canonical(step)
    # The trace must expose exactly what the episode observed and nothing else.
    # The old form of this check looked for one particular node id in the first
    # step, which only held because the first step exposes the root alone; this
    # states the property instead of one instance of it.
    exposed = set(node_table(episode.decision_trace))
    observed = {world.root_id} | set(episode.revealed_order)
    assert exposed == observed, (
        f"trace exposes {sorted(exposed - observed)} that were never observed, and "
        f"omits {sorted(observed - exposed)}"
    )
    with pytest.raises(IntegrityError):
        validate_episode_trace(world, replace(episode, n=999), 2, 4)
    with pytest.raises(IntegrityError):
        validate_episode_trace(world, replace(episode, decision_trace=()), 2, 4)


def test_complete_history_trace_is_distinct_from_bounded_debug_log(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    trace = HistoryTrace(path).open()
    record = {"type": "tool_execution_end", "result": "x" * 15000}
    trace.write(record)
    trace.write({"type": "text_delta", "delta": "redundant"})
    trace.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1 and rows[0]["event"] == record


def test_incomplete_branch_context_is_rejected_even_with_valid_blob_hashes(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    tree, _ = _grow(tmp_path, store, DiagnosticBackend(), rounds=2)
    node = tree.nodes()[-1]
    observation = load_observation(store, node.observation_digest)
    context = store.get_json(observation["context_digest"])
    context["branch"] = context["branch"][-1:]
    observation["context_digest"] = store.put_json(context)
    bad = replace(node, observation_digest=store.put_json(observation))
    with pytest.raises(IntegrityError, match="incomplete/incorrect branch"):
        validate_world_artifacts(store, [*tree.nodes()[:-1], bad], [])


def test_public_observation_mapping_cannot_mutate_world_data(tmp_path: Path) -> None:
    store = ContentAddressedStore(tmp_path / "cas")
    tree, _ = _grow(tmp_path, store, DiagnosticBackend(), rounds=1)
    observation = tree.observation(set(tree.node_ids()), 1, 2)
    public = observation.nodes[-1].public_observation
    public["diagnostics"]["next_value"] = 999
    assert observation.nodes[-1].public_observation["diagnostics"]["next_value"] == 3


def _controller_config(root: Path, *, resume: bool = False) -> Config:
    repo = Path(__file__).resolve().parents[1]
    return Config(project_root=root, task_dir=repo / "tasks" / "knapsack",
        incumbent_policy_path=repo / "policies" / "incumbent.py",
        loop=LoopConfig(outer_rounds=1, K1=1, M=1, W=1), run_id="v2", resume=resume)


def test_stateful_policy_resets_each_decision_and_rho(tmp_path: Path) -> None:
    config = _controller_config(tmp_path / "project")
    controller, registry = _build_controller(config, "v2", with_developer=False)
    path = tmp_path / "stateful.py"
    path.write_text("calls = [0]\ndef select(observation, rho):\n"
                    "    calls[0] += 1\n    if calls[0] > 1:\n        return []\n"
                    "    return [observation.legal_actions[-1]]\n", encoding="utf-8", newline="\n")
    assert validate_policy_source(path).ok
    try:
        policy = controller._policy_select_fn(path)
        store = controller.deps.store
        tree, _ = _grow(tmp_path, store, DiagnosticBackend(), policy=policy)
        world = SealedWorld.from_nodes(tree.world_id, tree.nodes(), root_id=tree.root_id)
        # Any accidental external call during dreaming fails immediately.
        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("replay must not call discovery or evaluator")
        controller.deps.backend.generate = forbidden
        controller.deps.evaluator.evaluate = forbidden
        options = {"policy_version_id": "stateful", "max_parallelism": 2, "max_rounds": 4}
        forward = replay_all_rhos(world, policy, rho_grid=(0, .25, .5, .75, 1), **options)
        backward = replay_all_rhos(world, policy, rho_grid=(1, .75, .5, .25, 0), **options)
        assert [episode.n for episode in forward] == [3] * 5
        assert {e.rho: e.replay_digest for e in forward} == {e.rho: e.replay_digest for e in backward}
    finally:
        registry.close()


def test_resume_after_final_seal_completes_final_selection_once(tmp_path: Path) -> None:
    config = _controller_config(tmp_path / "project")
    controller, registry = _build_controller(config, "v2", with_developer=False)
    original = controller._emit
    def interrupted(run_id: str, kind: str, **payload: Any) -> None:
        original(run_id, kind, **payload)
        if kind == "world.sealed":
            raise RuntimeError("simulated crash after final seal")
    controller._emit = interrupted
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            controller.run()
    finally:
        registry.close()
    for _attempt in range(2):
        resumed, registry = _build_controller(replace(config, resume=True), "v2", with_developer=False)
        def forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("sealed discovery must not be repeated")
        resumed.deps.backend.generate = forbidden
        try:
            result = resumed.run()
            assert [selection.iteration_t for selection in result.selections] == [1]
            assert result.final_policy_id == result.selections[0].selected_policy_id
            assert len([e for e in registry.list_events("v2") if e.kind == "policy.selected"]) == 1
        finally:
            registry.close()
