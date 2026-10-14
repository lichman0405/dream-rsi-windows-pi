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
import shutil
from pathlib import Path

import pytest

from dream_rsi.mof.analysis import UnsealResult
from dream_rsi.mof.protocol import (
    PHASES,
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


@pytest.fixture(scope="session")
def split_101(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A private copy of split 101's mounted files.

    A live phase deliberately takes the held-out files off disk, so a test that
    read them from `.runtime` would fail for the whole duration of a real run --
    and would be testing the vault's schedule rather than the grader. The split
    is three small CSVs; copying them costs nothing and makes the suite
    independent of whether a search is in flight.
    """
    source = PREPARED / "splits" / "101"
    destination = tmp_path_factory.mktemp("split-101")
    if not (source / "evaluator_private").is_dir():
        pytest.skip("split 101 is sealed by a running phase; re-run after it releases")
    shutil.copytree(source / "agent", destination / "agent")
    shutil.copytree(source / "evaluator_private", destination / "evaluator_private")
    return destination


def _pool_from(copy_root: Path, label: str) -> MofPool:
    return MofPool(
        label=label,
        train_features=copy_root / "agent" / "train_features.csv",
        train_targets=copy_root / "agent" / "train_targets.csv",
        pool_features=copy_root / "agent" / "dev_features.csv",
        private_targets=copy_root / "evaluator_private" / "dev_targets.private.csv",
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


def test_grader_scores_the_baseline_and_reports_failures_distinctly(split_101: Path) -> None:
    pool = _pool_from(split_101, "development-101")
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


def test_grader_is_indifferent_to_pool_row_order(split_101: Path) -> None:
    """The baseline must return the same set when the pool rows are reshuffled."""
    import csv
    import random

    pool = _pool_from(split_101, "development-101")
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


def test_block_pipeline_records_ledger_findings_and_freezing(
    tmp_path: Path, split_101: Path,
) -> None:
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
    record = runner.run_block(plan, labels=_pool_from(split_101, "dev-101").load_labels(),
                              pool_digest=_pool_from(split_101, "dev-101").data_digest())

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


# -------------------------------------------------------------------- figures


def test_every_figure_renders_with_its_source_data(tmp_path: Path) -> None:
    """Six figures, each with a PNG, an SVG and the CSV behind it.

    The report is a deliverable, not a side effect, so it gets a test that fails
    when a figure stops rendering rather than being discovered after a phase has
    already run. The numbers here are synthetic: this checks the plumbing, and
    nothing in it is a result.
    """
    from dream_rsi.mof import figures
    from dream_rsi.mof.analysis import UnsealResult, paired_table

    blocks = [{
        "run_id": f"mof-pilot-s101-r1-{method}",
        "plan": {"method": method, "split_seed": 101, "repeat": 1, "block_id": "b"},
        "baseline_S": 0.58,
        "budget_limit": 5_000_000,
        "ledger": {"charged_total": 4_100_000 - 100_000 * index,
                   "requests": 60 + index, "tokens": {"total": 4_100_000}},
        "findings": [{"cumulative": (index + 1) * 5300, "score": 0.58 + 0.02 * index,
                      "outer_t": index // 6, "round_index": index % 6,
                      "ledger_seq": index + 1, "charged": 5300} for index in range(12)],
        "worlds": [{"outer_t": t, "s_best": 0.6 + 0.02 * t, "b_count": 5,
                    "state": "SEALED_SCOREABLE", "node_count": 6, "world_digest": "x",
                    "policy_version_id": "p", "replay_worlds_available": 0} for t in range(2)],
    } for index, method in enumerate(("A", "B", "C"))]

    unsealed = [
        UnsealResult(
            run_id=f"r-{method}-{seed}-{repeat}", method=method, split_seed=seed,
            repeat=repeat, checkpoint=5_000_000, is_final=True, origin="baseline",
            candidate_sha256=f"{method}{seed}{repeat}", development_S=0.7,
            test_S=0.7 if method != "A" else 0.74, test_Q=5.2, test_Q_star=7.5,
            recommended_uptakes=[5.2] * 10, top_k_uptakes=[7.5] * 10,
            high_performer_hits=4, high_performer_cutoff=14, contract_status="OK",
            error_type=None, duration_s=3.0, permutation_agreement=[True] * 3,
        )
        for method in ("A", "B", "C") for seed in (101, 211) for repeat in (1, 2)
    ]
    table = paired_table(unsealed)
    out = tmp_path / "figures"

    produced = {
        "staircase": figures.quality_budget_staircase(blocks, out),
        "dynamics": figures.recursive_dynamics(
            [{"split_seed": row.split_seed, "method": row.method, "repeat": row.repeat,
              "run_id": row.run_id, "points": [{"checkpoint": row.checkpoint,
                                                "test_S": row.test_S, "test_Q": row.test_Q,
                                                "development_S": row.development_S,
                                                "origin": row.origin}]} for row in unsealed],
            out),
        "paired": figures.paired_final_comparison(table, out),
        "recommendation": figures.recommendation_explanation(
            [row.to_dict() for row in unsealed],
            {"random": {"mean_S": 0.43}, "ridge": {"Q": 4.9}, "random_forest": {"Q": 5.8}}, out),
        "mechanism": figures.internal_mechanism(blocks, out),
        "cost": figures.cost_comparison(blocks, out),
    }
    for name, paths in produced.items():
        for kind in ("png", "svg", "csv"):
            assert Path(paths[kind]).is_file(), f"{name} has no {kind}"
            assert Path(paths[kind]).stat().st_size > 0, f"{name} {kind} is empty"


def test_paired_table_uses_one_candidate_per_run(tmp_path: Path) -> None:
    """Checkpoint candidates belong in the curve, not in the paired difference."""
    from dream_rsi.mof.analysis import UnsealResult, paired_table

    def row(checkpoint: int, is_final: bool, test_S: float) -> UnsealResult:
        return UnsealResult(
            run_id="r", method="A", split_seed=101, repeat=1, checkpoint=checkpoint,
            is_final=is_final, origin="baseline", candidate_sha256=f"c{checkpoint}",
            development_S=0.5, test_S=test_S, test_Q=1.0, test_Q_star=2.0,
            recommended_uptakes=[1.0], top_k_uptakes=[2.0], high_performer_hits=0,
            high_performer_cutoff=1, contract_status="OK", error_type=None,
            duration_s=1.0, permutation_agreement=[],
        )

    table = paired_table([row(500_000, False, 0.10), row(5_000_000, True, 0.80)])
    assert len(table["rows"]) == 1
    assert table["rows"][0]["A_test_S"] == pytest.approx(0.80)


# ------------------------------------------- unfrozen-code isolation at unseal


def _synthetic_pool(root: Path, *, label: str = "synthetic") -> MofPool:
    """A tiny pool with otherwise-real files, for isolation tests only.

    Nothing here is CRAFTED data: the point is to exercise the machinery that
    keeps held-out labels away from model-written code without reading, sealing
    or depending on the real held-out files.
    """
    agent = root / "agent"
    private = root / "private"
    agent.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    ids = [f"mof_synthetic_{index:03d}" for index in range(40)]
    with (agent / "train_features.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("material_id,f0,f1\n")
        for index, material in enumerate(ids[:26]):
            handle.write(f"{material},{index},{index * 2}\n")
    with (agent / "train_targets.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("material_id,target_mol_kg\n")
        for index, material in enumerate(ids[:26]):
            handle.write(f"{material},{1.0 + index * 0.1}\n")
    with (agent / "dev_features.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("material_id,f0,f1\n")
        for index, material in enumerate(ids[26:]):
            handle.write(f"{material},{index},{index * 3}\n")
    with (private / "dev_targets.private.csv").open("w", newline="", encoding="utf-8") as handle:
        handle.write("material_id,target_mol_kg,uncertainty_mol_kg\n")
        for index, material in enumerate(ids[26:]):
            handle.write(f"{material},{2.0 + index * 0.2},0.1\n")
    return MofPool(
        label=label, train_features=agent / "train_features.csv",
        train_targets=agent / "train_targets.csv",
        pool_features=agent / "dev_features.csv",
        private_targets=private / "dev_targets.private.csv",
    )


def test_candidate_cannot_read_the_labels_while_it_is_being_graded(tmp_path: Path) -> None:
    """The property the whole search-phase arrangement exists for.

    The candidate is given a source that does the one thing a cheating program
    would do -- read the pool's true labels from a path -- and the assertion is
    that the path is not there while it runs. The path is guessed rather than
    discovered, which is the realistic attack: a candidate cannot learn the
    layout during the search, when the file is absent, and hope to find it later
    only if it survives both passes.
    """
    from dream_rsi.mof.vault import FileVault

    pool = _synthetic_pool(tmp_path / "pool")
    labels = pool.load_labels()
    evaluator = MofScreeningEvaluator(
        pool=pool, labels=labels, runner_script=RUNNER,
        workdir_root=tmp_path / "eval", k=3,
    )
    # Not a real path and never a real label: it only has to be the path the
    # candidate would try, so that the test shows the file's absence rather
    # than the candidate's good manners.
    target = pool.private_targets.resolve()
    cheater = tmp_path / "cheater"
    cheater.mkdir()
    (cheater / "candidate.py").write_text(
        "import pandas as pd\n"
        f"TARGET = r'{target}'\n"
        "def rank(train_X, train_y, pool_X, *, k, seed):\n"
        "    try:\n"
        "        truth = pd.read_csv(TARGET)\n"
        "        order = truth.sort_values('target_mol_kg', ascending=False)\n"
        "        return order['material_id'].tolist()[:k]\n"
        "    except Exception as exc:\n"
        "        raise RuntimeError('labels unreachable: ' + type(exc).__name__) from exc\n",
        encoding="utf-8",
    )

    vault = FileVault(manifest_path=tmp_path / "vault.jsonl")
    vault.seal([pool.private_targets])
    try:
        assert vault.verify_absent()["ok"]
        verdict = evaluator.evaluate(cheater, invocation_id="cheater")
        assert verdict.score == 0.0, "a candidate that reads the labels must not score"
        assert verdict.error_type == "CANDIDATE_RAISED"
        assert "labels unreachable: FileNotFoundError" in str(verdict.detail.get("error", ""))
    finally:
        vault.release()
    assert vault.verify_restored()["ok"]

    # With the vault open the same program succeeds, which is what makes the
    # assertion above about the seal rather than about the program.
    open_verdict = evaluator.evaluate(cheater, invocation_id="cheater-open")
    assert open_verdict.score > 0.9, (
        "the probe must actually be able to read the file when it is present, "
        "or the test above proves nothing"
    )


def test_the_grader_can_run_while_its_labels_are_off_disk(tmp_path: Path) -> None:
    """Preloaded labels let the unseal pass run under the same arrangement."""
    from dream_rsi.mof.vault import FileVault

    pool = _synthetic_pool(tmp_path / "pool2", label="synthetic-sealed")
    labels = pool.load_labels()
    evaluator = MofScreeningEvaluator(
        pool=pool, labels=labels, runner_script=RUNNER,
        workdir_root=tmp_path / "eval2", k=3,
    )
    vault = FileVault(manifest_path=tmp_path / "vault2.jsonl")
    vault.seal([pool.private_targets])
    try:
        assert vault.verify_absent()["ok"]
        verdict = evaluator.evaluate(BASELINE, invocation_id="sealed")
        assert verdict.status.value in ("SUCCESS", "FAILED"), verdict.detail
        assert float(verdict.detail["Q_star"]) > 0
    finally:
        vault.release()
    assert vault.verify_restored()["ok"]


# ------------------------------------------------------------------- resume


def test_a_block_that_was_already_run_resumes_instead_of_respending(
    tmp_path: Path, split_101: Path,
) -> None:
    """A restart must not buy the arm a second budget.

    This is the local check that a phase can be interrupted and continued. The
    scripted backend replaces the model, so the cost of a resumed block is
    deterministic and the assertions are about accounting, not about a service.
    """
    from dream_rsi.mof.experiment import MofExperimentRunner
    from dream_rsi.mof.protocol import MethodSpec, PhaseSpec, Protocol, block_plans

    spec = PhaseSpec(
        key="acceptance", worlds=1, token_budget=200_000, session_cap=32,
        split_seeds=(101,), repeats=(1,), checkpoints=(1_000, 100_000),
        methods={
            method: MethodSpec(key=method, name=method,
                               incumbent_policy="policies/incumbent.py",
                               uses_developer=False, W=2, K1=2, M=1)
            for method in ("A", "B", "C")
        },
    )
    protocol = Protocol(phase="pilot", prepared_root=str(PREPARED), spec_override=spec)
    runtime = tmp_path / "runtime"

    def factory(config, plan, budget):  # noqa: ANN001 - test seam
        return _ScriptedMofBackend(budget)

    runner = MofExperimentRunner(
        project_root=Path.cwd(), protocol=protocol, task_dir=Path("tasks/mof_screening"),
        runner_script=RUNNER, out_root=tmp_path, backend_factory=factory,
        run_namespace="resume-", runtime_root=runtime,
    )
    plan = block_plans("pilot")[0]
    pool = _pool_from(split_101, "resume-101")

    first = runner.run_block(plan, labels=pool.load_labels(),
                             pool_digest=pool.data_digest())
    spent_first = int(first.ledger["charged_total"])
    first_nodes = {finding["node_id"] for finding in first.findings}
    assert spent_first > 0

    second = runner.run_block(plan, labels=pool.load_labels(),
                              pool_digest=pool.data_digest())
    spent_second = int(second.ledger["charged_total"])

    assert any("no work repeated" in note for note in second.notes), second.notes
    assert spent_second == spent_first, (
        "a resumed block must not spend again: it reuses sealed worlds, and its "
        "ceiling is seeded from the ledger it already wrote"
    )
    assert {finding["node_id"] for finding in second.findings} == first_nodes, (
        "resuming must not grow a second tree"
    )
    assert second.budget["limit"] == first.budget["limit"], (
        "the ceiling belongs to the run, not to the process continuing it"
    )


def test_a_pause_marker_stops_the_phase_at_a_block_boundary(
    tmp_path: Path, split_101: Path,
) -> None:
    """There has to be a way to stop a phase that is not killing it.

    Killing a run that holds a sealed vault is the thing the vault exists to
    survive, not something to do on purpose; `--block-limit` is read at start.
    The marker is checked between blocks, so a phase stops with every finished
    block committed and the vault closed in the ordinary way.
    """
    from dream_rsi.mof.experiment import MofExperimentRunner
    from dream_rsi.mof.protocol import MethodSpec, PhaseSpec, Protocol, block_plans

    spec = PhaseSpec(
        key="acceptance", worlds=1, token_budget=200_000, session_cap=32,
        split_seeds=(101,), repeats=(1, 2), checkpoints=(1_000,),
        methods={
            "A": MethodSpec(key="A", name="A", incumbent_policy="policies/incumbent.py",
                            uses_developer=False, W=2, K1=1, M=1),
            "B": MethodSpec(key="B", name="B", incumbent_policy="policies/incumbent.py",
                            uses_developer=False, W=2, K1=1, M=1),
            "C": MethodSpec(key="C", name="C", incumbent_policy="policies/serial.py",
                            uses_developer=False, W=1, K1=1, M=1),
        },
    )
    protocol = Protocol(phase="pilot", prepared_root=str(PREPARED), spec_override=spec)
    runner = MofExperimentRunner(
        project_root=Path.cwd(), protocol=protocol, task_dir=Path("tasks/mof_screening"),
        runner_script=RUNNER, out_root=tmp_path,
        backend_factory=lambda config, plan, budget: _ScriptedMofBackend(budget),
        run_namespace="pause-", runtime_root=tmp_path / "runtime",
    )
    plans = block_plans("pilot")          # two blocks of three arms
    labels = _pool_from(split_101, "pause-101").load_labels()
    digest = _pool_from(split_101, "pause-101").data_digest()

    # Stop the real runner's phase loop by giving every block the same body but
    # marking the pause before the second block would start.
    runner.pause_marker.write_text("stop", encoding="utf-8")
    summary = runner.run_phase(plans, protect=[])
    assert summary["blocks"] == [], (
        "with the marker already present the phase must start nothing"
    )
    assert any("paused before" in note for note in summary["notes"]), summary["notes"]

    runner.pause_marker.unlink()
    first = runner.run_block(plans[0], labels=labels, pool_digest=digest)
    assert first.status == "COMMITTED"
    runner.pause_marker.write_text("stop", encoding="utf-8")
    summary = runner.run_phase([plans[-1]], protect=[])
    assert summary["blocks"] == [], "the marker must stop the next block too"
    assert any("paused before" in note for note in summary["notes"])


# --------------------------------------------------------------- formal v2 identity


def test_formal2_is_a_separate_identity_from_formal() -> None:
    """One v2 label for run ids, output dir, marker, digests and reports.

    Reusing "formal" would have put v2's runs into v1's registry namespace and
    its records into v1's output directory, where a reader could not tell which
    implementation produced which number.
    """
    from dream_rsi.mof.protocol import PHASES, block_plans

    v1, v2 = PHASES["formal"], PHASES["formal2"]
    assert v2.worlds == v1.worlds and v2.token_budget == v1.token_budget
    assert v2.split_seeds == v1.split_seeds and v2.repeats == v1.repeats
    assert v2.checkpoints == v1.checkpoints
    assert {k: (m.W, m.K1, m.M) for k, m in v2.methods.items()} == {
        k: (m.W, m.K1, m.M) for k, m in v1.methods.items()
    }, "v2 must keep v1's parameters; only the implementation changed"

    ids_v1 = {plan.run_id for plan in block_plans("formal")}
    ids_v2 = {plan.run_id for plan in block_plans("formal2")}
    assert ids_v1.isdisjoint(ids_v2)
    assert all(run_id.startswith("mof-formal2-") for run_id in ids_v2)


def test_the_two_formal_phases_have_distinct_digests() -> None:
    from dream_rsi.mof.cli import _prepared
    from dream_rsi.mof.protocol import Protocol

    # The digest covers the prepared_root string, so it is only reproducible
    # when it is built the way the command line builds it.
    prepared = str(_prepared(Path.cwd()))
    v1 = Protocol(phase="formal", prepared_root=prepared)
    v2 = Protocol(phase="formal2", prepared_root=prepared)
    assert v1.digest != v2.digest
    # The historical digests still reproduce from this code, so the records of
    # the phases that already ran stay checkable.
    assert v1.digest == (
        "10d9a668af73d64281bc3301b90cac7e8b44cb18bee7346fe9769ece9cb2c603"
    )
    assert Protocol(phase="pilot", prepared_root=prepared).digest == (
        "9f5940ed36ffbedf3b4ce3a6f43a4d65ee07435df1a4727f15e6c9620430b0f8"
    )


def test_the_pause_marker_is_checked_at_every_block_boundary(
    tmp_path: Path, split_101: Path,
) -> None:
    """Not just before the first block: a phase must stop at any boundary."""
    from dream_rsi.mof.experiment import MofExperimentRunner
    from dream_rsi.mof.protocol import MethodSpec, PhaseSpec, Protocol, block_plans

    spec = PhaseSpec(
        key="acceptance", worlds=1, token_budget=200_000, session_cap=32,
        split_seeds=(101,), repeats=(1, 2, 3), checkpoints=(1_000,),
        methods={
            method: MethodSpec(key=method, name=method,
                               incumbent_policy="policies/incumbent.py",
                               uses_developer=False, W=2, K1=1, M=1)
            for method in ("A", "B", "C")
        },
    )
    protocol = Protocol(phase="pilot", prepared_root=str(PREPARED), spec_override=spec)
    runner = MofExperimentRunner(
        project_root=Path.cwd(), protocol=protocol, task_dir=Path("tasks/mof_screening"),
        runner_script=RUNNER, out_root=tmp_path,
        backend_factory=lambda config, plan, budget: _ScriptedMofBackend(budget),
        run_namespace="boundary-", runtime_root=tmp_path / "runtime",
    )
    labels = _pool_from(split_101, "boundary-101").load_labels()
    digest = _pool_from(split_101, "boundary-101").data_digest()
    plans = block_plans("pilot")
    block_ids = sorted({plan.block_id for plan in plans})

    # Run block 1, then raise the marker: the phase must stop before block 2.
    first = [p for p in plans if p.block_id == block_ids[0]]
    rest = [p for p in plans if p.block_id != block_ids[0]]
    for plan in first:
        assert runner.run_block(plan, labels=labels, pool_digest=digest).status == "COMMITTED"

    runner.pause_marker.write_text("stop", encoding="utf-8")
    summary = runner.run_phase(rest, protect=[])
    assert summary["blocks"] == [], "the marker must stop the phase at this boundary"
    assert any("paused before" in note for note in summary["notes"]), summary["notes"]

    # Clearing it lets the same call proceed, so the marker is the only cause.
    runner.pause_marker.unlink()
    summary = runner.run_phase(rest, protect=[])
    assert len(summary["blocks"]) == len(rest)


# ----------------------------------------------------------- the two-arm phase


def _unseal_row(method: str, split_seed: int, *, test_S: float,
                development_S: float = 0.80) -> UnsealResult:
    """One final candidate as the unseal pass would record it."""
    return UnsealResult(
        run_id=f"mof-ab-s{split_seed}-r1-{method}", method=method,
        split_seed=split_seed, repeat=1, checkpoint=30_000_000, is_final=True,
        origin="baseline", candidate_sha256="0" * 64, development_S=development_S,
        test_S=test_S, test_Q=1.0, test_Q_star=2.0, recommended_uptakes=[1.0],
        top_k_uptakes=[1.0], high_performer_hits=0, high_performer_cutoff=1,
        contract_status="OK", error_type=None, duration_s=1.0, permutation_agreement=[],
    )


def test_the_ab_phase_runs_two_arms_of_the_frozen_design() -> None:
    """The small phase may shrink replication; it may not reshape an arm.

    A and B must be the same programs under the same parameters as in the full
    phase, or the two would not be measuring the same comparison.
    """
    from dream_rsi.mof.protocol import PHASES

    ab, formal = PHASES["ab"], PHASES["formal"]
    assert sorted(ab.methods) == ["A", "B"], "this phase registers no auxiliary arm"
    for arm in ("A", "B"):
        got, want = ab.methods[arm], formal.methods[arm]
        assert (got.W, got.K1, got.M) == (want.W, want.K1, want.M)
        assert got.incumbent_policy == want.incumbent_policy
        assert got.uses_developer == want.uses_developer
    assert ab.worlds == formal.worlds == 8
    assert ab.methods["A"].M == 3
    assert ab.token_budget == formal.token_budget == 30_000_000
    assert ab.checkpoints == formal.checkpoints
    assert ab.split_seeds == (101, 211) and ab.repeats == (1,)
    assert ab.blocks == 2
    assert ab.session_cap == ab.worlds * ab.methods["A"].K1 * ab.methods["A"].W


def test_the_ab_phase_has_its_own_identity() -> None:
    from dream_rsi.mof.cli import _prepared
    from dream_rsi.mof.protocol import PHASES

    prepared = str(_prepared(Path.cwd()))
    ids = {phase: {plan.run_id for plan in block_plans(phase)}
           for phase in ("pilot", "formal", "formal2", "ab")}
    assert all(run_id.startswith("mof-ab-") for run_id in ids["ab"])
    for other in ("pilot", "formal", "formal2"):
        assert ids["ab"].isdisjoint(ids[other])

    digests = {phase: Protocol(phase=phase, prepared_root=prepared).digest
               for phase in ids}
    assert digests["ab"] != digests["formal2"]
    # The phases that already produced records still reproduce from this code.
    assert digests["pilot"] == (
        "9f5940ed36ffbedf3b4ce3a6f43a4d65ee07435df1a4727f15e6c9620430b0f8"
    )
    assert digests["formal"] == (
        "10d9a668af73d64281bc3301b90cac7e8b44cb18bee7346fe9769ece9cb2c603"
    )
    assert PHASES["ab"].reading_mode == "descriptive"


def test_the_reading_mode_changes_the_narrative_and_not_the_digest() -> None:
    """It is deliberately outside the digest: it never changes what was run."""
    base = Protocol(phase="ab", prepared_root=str(PREPARED))
    flipped = Protocol(
        phase="ab", prepared_root=str(PREPARED),
        spec_override=replace(base.spec, reading_mode="preregistered"),
    )
    assert flipped.digest == base.digest


def test_a_two_arm_phase_orders_only_the_arms_it_has() -> None:
    for phase in ("pilot", "formal", "formal2", "ab"):
        expected = sorted(PHASES[phase].methods)
        by_block: dict[str, list[str]] = {}
        for plan in block_plans(phase):
            by_block.setdefault(plan.block_id, []).append(plan.method)
        for methods in by_block.values():
            assert sorted(methods) == expected
    assert sorted(method_order("ab", 101, 1)) == ["A", "B"]


def test_a_block_is_complete_against_its_own_arm_set() -> None:
    """A missing-by-design arm is not a missing-by-failure one."""
    from dream_rsi.mof.analysis import paired_table

    rows = [
        _unseal_row("A", 101, test_S=0.90), _unseal_row("B", 101, test_S=0.80),
        _unseal_row("A", 211, test_S=0.88), _unseal_row("B", 211, test_S=0.86),
    ]
    table = paired_table(rows, expected_arms=("A", "B"), reading_mode="descriptive")
    assert table["blocks"] == 2
    assert table["complete_blocks"] == 2
    assert table["expected_arms"] == ["A", "B"]
    assert [row["A_minus_B"] for row in table["rows"]] == pytest.approx([0.10, 0.02])
    assert table["A_minus_B"]["mean"] == pytest.approx(0.06)
    assert table["A_minus_B"]["splits"] == [101, 211]
    assert not table["A_minus_C"]["available"], "no C ran, so there is no A-C contrast"


def test_the_small_phase_cannot_earn_the_preregistered_verdict() -> None:
    """The guard is on the split count as well as on the declared mode.

    A "4/5 splits" rule applied to two splits would read as a pre-registered
    test that came out negative, when no such test was run -- and a
    ``worth_further_validation`` on two blocks would be worse.
    """
    from dream_rsi.mof.analysis import verdict_from_ci

    flattering = {"available": True, "mean": 0.09, "ci95": [0.02, 0.16],
                  "consistent_direction": 2, "splits": [101, 211]}
    small = verdict_from_ci(flattering)
    assert small["verdict"] == "descriptive_only"
    assert small["registered_rule_applies"] is False
    assert small["splits_total"] == 2
    # The arithmetic is still reported: a reader gets the numbers, not a verdict.
    assert small["mean"] == pytest.approx(0.09)
    assert small["ci95"] == [0.02, 0.16]
    assert small["practical_threshold"] == 0.05

    five = {**flattering, "consistent_direction": 5,
            "splits": [101, 211, 307, 401, 503]}
    assert verdict_from_ci(five)["verdict"] == "worth_further_validation"
    assert verdict_from_ci(five)["registered_rule_applies"] is True


def test_a_two_arm_block_runs_end_to_end_and_never_invents_a_third_arm(
    tmp_path: Path, split_101: Path,
) -> None:
    """The runner's own view of a block, with no C anywhere in the record."""
    from dream_rsi.mof.experiment import MofExperimentRunner

    spec = PhaseSpec(
        key="acceptance", worlds=1, token_budget=200_000, session_cap=32,
        split_seeds=(101,), repeats=(1,), checkpoints=(1_000,),
        methods={
            method: MethodSpec(key=method, name=method,
                               incumbent_policy="policies/incumbent.py",
                               uses_developer=False, W=2, K1=1, M=1)
            for method in ("A", "B")
        },
        reading_mode="descriptive",
    )
    protocol = Protocol(phase="ab", prepared_root=str(PREPARED), spec_override=spec)
    runner = MofExperimentRunner(
        project_root=Path.cwd(), protocol=protocol, task_dir=Path("tasks/mof_screening"),
        runner_script=RUNNER, out_root=tmp_path,
        backend_factory=lambda config, plan, budget: _ScriptedMofBackend(budget),
        run_namespace="two-arm-", runtime_root=tmp_path / "runtime",
    )
    pool = _pool_from(split_101, "two-arm-101")
    labels, digest = pool.load_labels(), pool.data_digest()
    plans = [p for p in block_plans("ab") if p.block_id == "ab-s101-r1"]
    assert sorted(p.method for p in plans) == ["A", "B"]

    records = [runner.run_block(plan, labels=labels, pool_digest=digest) for plan in plans]
    assert [record.status for record in records] == ["COMMITTED", "COMMITTED"]
    assert {record.plan["method"] for record in records} == {"A", "B"}
    assert all(record.budget_limit == 200_000 for record in records)
    assert all(len(record.worlds) == 1 for record in records)
    # The scripted backend's candidate never scores above zero, so the world
    # seals UNSCOREABLE rather than SCOREABLE; either way it is sealed, which is
    # what says the two-arm block ran to its end.
    assert all(world["state"] in ("SEALED_SCOREABLE", "SEALED_UNSCOREABLE")
               for record in records for world in record.worlds)
    assert not any(record.plan["method"] == "C" for record in records)


# --------------------------------------------------------------- ledger sessions


def test_a_resumed_ledger_resolves_each_world_to_the_session_that_built_it() -> None:
    """A ledger's ``seq`` is unique per session, and a resume can repeat numbers.

    A process that restarts the search appends to a ledger another process
    already wrote, and the older ledgers repeat the first session's sequence
    numbers. Resolving a finding through a flat ``{seq: row}`` index would then
    hand it another candidate's tokens -- so rows are split at every reset of
    the number, and a finding is resolved through the session that was writing
    when its world was created.
    """
    from dream_rsi.mof.experiment import LedgerIndex

    rows = [
        {"event": "request", "seq": 1, "at": "2026-10-14T15:27:12+0800",
         "charged": 4294, "call_id": "first-1"},
        {"event": "request", "seq": 2, "at": "2026-10-14T15:27:46+0800",
         "charged": 4544, "call_id": "first-2"},
        {"event": "request", "seq": 1, "at": "2026-10-14T17:35:17+0800",
         "charged": 3862, "call_id": "second-1"},
        {"event": "request", "seq": 2, "at": "2026-10-14T17:35:56+0800",
         "charged": 3937, "call_id": "second-2"},
        {"event": "request", "seq": 3, "at": "2026-10-14T17:41:00+0800",
         "charged": 8787, "call_id": "second-3"},
    ]
    index = LedgerIndex(rows)
    assert index.session_count == 2
    before_reset = "2026-10-14T07:26:44.846407+00:00"    # 15:26:44 +0800
    after_reset = "2026-10-14T09:35:00.553165+00:00"     # 17:35:00 +0800

    assert index.charge_row(1, world_created_at=before_reset)["call_id"] == "first-1"
    assert index.charge_row(2, world_created_at=before_reset)["call_id"] == "first-2"
    assert index.charge_row(1, world_created_at=after_reset)["call_id"] == "second-1"
    assert index.charge_row(3, world_created_at=after_reset)["call_id"] == "second-3"
    # A number only one session wrote resolves for that session and nowhere else.
    assert index.charge_row(3, world_created_at=before_reset) is None
    assert index.charge_row(None, world_created_at=before_reset) is None

    # A ledger that never restarted is exactly the flat index it replaces.
    flat = LedgerIndex(rows[:2])
    assert flat.session_count == 1
    assert flat.charge_row(2, world_created_at=after_reset)["call_id"] == "first-2"
    assert flat.charge_row(3, world_created_at=after_reset) is None
