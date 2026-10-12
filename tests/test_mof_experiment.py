"""Acceptance tests for the MOF screening pipeline.

These exercise the parts that a live run would otherwise be the first to try:
the block loop with a scripted backend, the cost ledger's join to node records,
checkpoint freezing, the vault's promise that labels are gone while a candidate
runs, and the grader's contract handling. Nothing here makes a model call, and
nothing here decides an experimental result -- the point is that the machinery
is correct before it is pointed at a paid service.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dream_rsi.mof.protocol import (
    MethodSpec,
    PhaseSpec,
    Protocol,
    block_plans,
    method_order,
)
from dream_rsi.mof.scoring import (
    MofPool,
    MofScreeningEvaluator,
    score_ranking,
    screen_candidate_source,
)

PREPARED = Path(".runtime/mof_experiment/prepared/CRAFTED-2.0.1-CO2-298-UFF-DDEC-v1")
RUNNER = Path("tasks/mof_screening/candidate_runner.py")
BASELINE = Path("tasks/mof_screening/initial")

pytestmark = pytest.mark.skipif(
    not PREPARED.is_dir(), reason="prepared CRAFTED splits are not present in .runtime"
)


# --------------------------------------------------------------------- protocol


def test_method_order_is_deterministic_and_complete() -> None:
    first = method_order("pilot", 101, 1)
    assert first == method_order("pilot", 101, 1)
    assert set(first) == {"A", "B", "C"}
    # Different blocks should not all land on the same order, or the
    # randomisation the design asks for would be decorative.
    orders = {method_order("formal", seed, repeat)
              for seed in (101, 211, 307, 401, 503) for repeat in (1, 2)}
    assert len(orders) > 1


def test_block_plans_cover_every_arm_once_per_block() -> None:
    plans = block_plans("pilot")
    assert len(plans) == 2 * 2 * 3
    by_block: dict[str, list[str]] = {}
    for plan in plans:
        by_block.setdefault(plan.block_id, []).append(plan.method)
    for methods in by_block.values():
        assert sorted(methods) == ["A", "B", "C"]
        assert len(methods) == 3


def test_protocol_digest_tracks_the_parameters_it_covers() -> None:
    base = Protocol(phase="pilot", prepared_root=str(PREPARED))
    same = Protocol(phase="pilot", prepared_root=str(PREPARED))
    assert base.digest == same.digest
    changed = Protocol(phase="formal", prepared_root=str(PREPARED))
    assert changed.digest != base.digest
    spec = base.spec
    shrunk = Protocol(
        phase="pilot", prepared_root=str(PREPARED),
        spec_override=replace(spec, token_budget=spec.token_budget - 1),
    )
    assert shrunk.digest != base.digest


# ---------------------------------------------------------------------- grader


def test_screen_rejects_network_and_process_imports() -> None:
    assert "socket" in screen_candidate_source("import socket\n")
    assert "subprocess" in screen_candidate_source("from subprocess import run\n")
    assert screen_candidate_source("import numpy, pandas, sklearn\n") == ()
    assert screen_candidate_source("def broken(:\n") == ("SYNTAX_ERROR",)


def test_score_ranking_uses_ids_to_break_ties() -> None:
    labels = {"a": 1.0, "b": 1.0, "c": 0.5, "d": 0.1}
    _q, _q_star, s, recommended, top = score_ranking(["a", "b"], labels, k=2)
    assert recommended == [1.0, 1.0]
    assert top == [1.0, 1.0]          # ties resolved by id: a then b
    assert s == pytest.approx(1.0)


def test_grader_scores_the_baseline_and_reports_failures_distinctly() -> None:
    pool = MofPool.development(PREPARED, 101)
    evaluator = MofScreeningEvaluator(
        pool=pool, labels=pool.load_labels(), runner_script=RUNNER,
        workdir_root=Path(".runtime/pytest-mof/eval"), k=10,
    )
    good = evaluator.evaluate(BASELINE, invocation_id="baseline")
    assert good.status.value == "SUCCESS"
    assert 0.0 < good.score < 1.0
    # The pool's private numbers must not appear in the public half.
    assert set(good.public_detail) == {"contract_status", "k", "evaluated_materials"}

    broken = Path(".runtime/pytest-mof/broken")
    broken.mkdir(parents=True, exist_ok=True)
    (broken / "candidate.py").write_text(
        "def rank(train_X, train_y, pool_X, *, k, seed):\n"
        "    return [f'not-a-material-{i}' for i in range(k)]\n",
        encoding="utf-8",
    )
    bad = evaluator.evaluate(broken, invocation_id="broken")
    assert bad.status.value == "FAILED"
    assert bad.score == 0.0
    assert bad.error_type == "OUT_OF_POOL_ID"

    duplicated = Path(".runtime/pytest-mof/duplicated")
    duplicated.mkdir(parents=True, exist_ok=True)
    (duplicated / "candidate.py").write_text(
        "def rank(train_X, train_y, pool_X, *, k, seed):\n"
        "    return [pool_X['material_id'].iloc[0]] * k\n",
        encoding="utf-8",
    )
    dup = evaluator.evaluate(duplicated, invocation_id="duplicated")
    assert dup.status.value == "FAILED" and dup.error_type == "DUPLICATE_ID"


def test_grader_is_indifferent_to_pool_row_order() -> None:
    """The baseline must return the same set when the pool rows are reshuffled."""
    import csv
    import random

    pool = MofPool.development(PREPARED, 101)
    labels = pool.load_labels()
    base = MofScreeningEvaluator(pool=pool, labels=labels, runner_script=RUNNER,
                                 workdir_root=Path(".runtime/pytest-mof/order"), k=10)
    first = base.evaluate(BASELINE, invocation_id="order-0")

    with pool.pool_features.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    shuffled = list(rows[1:])
    random.Random(7).shuffle(shuffled)
    variant_dir = Path(".runtime/pytest-mof/order-variant")
    variant_dir.mkdir(parents=True, exist_ok=True)
    variant_features = variant_dir / "pool_features.csv"
    with variant_features.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(rows[0])
        writer.writerows(shuffled)

    variant = MofPool(label="variant", train_features=pool.train_features,
                      train_targets=pool.train_targets, pool_features=variant_features,
                      private_targets=pool.private_targets, payload_seed=pool.payload_seed)
    second = MofScreeningEvaluator(pool=variant, labels=labels, runner_script=RUNNER,
                                   workdir_root=variant_dir, k=10).evaluate(
        BASELINE, invocation_id="order-1")
    assert set(first.detail["ranking"]) == set(second.detail["ranking"])
    assert first.score == pytest.approx(second.score)


# ----------------------------------------------------------------------- vault


def test_vault_hides_labels_and_restores_them_exactly() -> None:
    from dream_rsi.mof.vault import FileVault

    root = Path(".runtime/pytest-mof/vault")
    secret = root / "evaluator_private"
    secret.mkdir(parents=True, exist_ok=True)
    (secret / "dev_targets.private.csv").write_text("material_id,target_mol_kg\nx,1.0\n",
                                                    encoding="utf-8")
    before = (secret / "dev_targets.private.csv").read_bytes()

    vault = FileVault(manifest_path=root / "vault.jsonl")
    vault.seal([secret])
    verification = vault.verify_absent()
    assert verification["ok"], verification
    with pytest.raises(FileNotFoundError):
        (secret / "dev_targets.private.csv").read_bytes()
    vault.release()
    assert (secret / "dev_targets.private.csv").read_bytes() == before
    assert vault.verify_restored()["ok"]


# ------------------------------------------------------------------- block loop


class _ScriptedMofBackend:
    """A discovery backend that rewrites the candidate without calling a model.

    It exists to prove the *plumbing*: that a batch produces nodes, that the
    ledger join survives eight-way concurrency, and that freezing picks by
    development score. It is never used for a reported result.
    """

    backend_id = "scripted-mof-acceptance"
    is_live = False

    def __init__(self, token_budget) -> None:
        self.token_budget = token_budget
        self.prompts: list[str] = []

    def generate(self, *, task, parent_workspace, history, output_workspace,
                 invocation_id, context=None):
        from dream_rsi.adapters.base import GeneratedAttempt, digest_file

        source = (Path(parent_workspace) / task.entry_filename).read_text(encoding="utf-8")
        self.prompts.append(task.instructions[:64])
        grant = self.token_budget.reserve(
            role="discovery", prompt=source[:4000], system_prompt="", max_output_bytes=4096,
        )
        assert grant is not None
        settlement = self.token_budget.settle(
            grant, {"input": 120, "output": 30, "total": 150, "reported_messages": 1},
            role="discovery", duration_s=0.01, ok=True, provider="scripted",
            model="scripted",
        )
        output_workspace = Path(output_workspace)
        output_workspace.mkdir(parents=True, exist_ok=True)
        target = output_workspace / task.entry_filename
        # A real, small edit: a different ridge penalty. The grader decides
        # whether that helped; the backend never reports a score.
        alpha = 1.0 + (hash(invocation_id) % 7)
        target.write_text(
            source.replace("ALPHA = 1.0", f"ALPHA = {alpha}.0")
            if "ALPHA = 1.0" in source else source + "\n# variant\n",
            encoding="utf-8",
        )
        return GeneratedAttempt(
            invocation_id=invocation_id, backend_id=self.backend_id, model="scripted",
            workspace_path=output_workspace, entry_file_digest=digest_file(target),
            succeeded=True, duration_s=0.01,
            generation_record={"ledger_seq": settlement.seq,
                               "token_charged": settlement.charged,
                               "token_usage": {"total": 150}},
        )


def test_block_pipeline_records_ledger_findings_and_freezing(tmp_path: Path) -> None:
    from dream_rsi.mof.experiment import MofExperimentRunner

    spec = PhaseSpec(
        key="acceptance", worlds=1, token_budget=200_000, session_cap=32,
        split_seeds=(101,), repeats=(1,),
        checkpoints=(1_000, 100_000),
        methods={
            "A": MethodSpec(key="A", name="dream_rsi",
                            incumbent_policy="policies/incumbent.py",
                            uses_developer=False, W=2, K1=2, M=1),
            "B": MethodSpec(key="B", name="frozen_initial",
                            incumbent_policy="policies/incumbent.py",
                            uses_developer=False, W=2, K1=2, M=1),
            "C": MethodSpec(key="C", name="frozen_serial",
                            incumbent_policy="policies/serial.py",
                            uses_developer=False, W=2, K1=2, M=1),
        },
    )
    protocol = Protocol(phase="pilot", prepared_root=str(PREPARED), spec_override=spec)
    holder: dict[str, _ScriptedMofBackend] = {}

    def factory(config, plan, budget):  # noqa: ANN001 - test seam
        holder["backend"] = _ScriptedMofBackend(budget)
        return holder["backend"]

    runner = MofExperimentRunner(
        project_root=Path.cwd(), protocol=protocol, task_dir=Path("tasks/mof_screening"),
        runner_script=RUNNER, out_root=tmp_path, backend_factory=factory,
        # The live pilot uses `mof-pilot-s101-r1-A`; a rehearsal must not, and it
        # must not write into the project's registry either.
        run_namespace="acceptance-", runtime_root=tmp_path / "runtime",
    )
    plan = block_plans("pilot")[0]
    plan = replace(plan, phase="pilot")
    record = runner.run_block(plan, labels=MofPool.development(PREPARED, 101).load_labels())

    assert record.run_id.startswith("acceptance-")
    assert record.status in ("COMMITTED", "FAILED"), record.error
    assert record.baseline_S is not None and 0.0 < record.baseline_S < 1.0
    assert record.ledger["requests"] > 0, "the ledger must have rows for real calls"
    assert record.ledger["requests_without_reported_usage"] == 0
    assert record.findings, "the rollout must have produced candidates"
    assert all(f["charged"] > 0 for f in record.findings)
    cumulative = [f["cumulative"] for f in record.findings]
    assert cumulative == sorted(cumulative), "cumulative tokens must be monotone"
    assert record.final_candidate is not None
    frozen_source = Path(record.final_candidate["source_path"])
    assert frozen_source.is_file()
    import hashlib
    assert record.final_candidate["candidate_sha256"] == hashlib.sha256(
        frozen_source.read_bytes()).hexdigest()
    # Every checkpoint freezes something, and the pick never scores below the start.
    assert len(record.frozen) == len(spec.checkpoints)
    assert all(c["development_S"] >= record.baseline_S for c in record.frozen)
