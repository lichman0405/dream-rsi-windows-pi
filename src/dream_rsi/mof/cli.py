"""Command-line entry points for the MOF screening experiment.

``mof-acceptance``  implementation checks only; makes no model call and states
                   no result. It has to pass before a live block runs.
``mof-run``        one phase, block by block, under the vault.
``mof-unseal``     opens the sealed test pool once, for every frozen candidate.
``mof-report``     figures, tables and the reader-facing ``summary.md``.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .analysis import (
    UnsealResult,
    development_test_gap,
    load_jsonl,
    paired_table,
    unseal_block,
    write_json,
)
from .experiment import MofExperimentRunner, default_protected_paths
from .figures import (
    internal_mechanism,
    paired_final_comparison,
    quality_budget_staircase,
    recommendation_explanation,
    recursive_dynamics,
)
from .protocol import K, MODEL, THINKING, Protocol, block_plans
from .scoring import MofPool, MofScreeningEvaluator, material_references
from .vault import FileVault

PREPARED_DIR = Path(".runtime/mof_experiment/prepared/CRAFTED-2.0.1-CO2-298-UFF-DDEC-v1")
DATA_DIR = Path(".runtime/mof_experiment/data")
RAW_DIR = Path(".runtime/mof_experiment/raw")
OUT_DIR = Path(".runtime/mof_experiment/experiment")


def _status(token: str, **extra: Any) -> None:
    print("STATUS: " + json.dumps({"status": token, **extra}, sort_keys=True, default=str))


def _pi_version() -> str | None:
    try:
        result = subprocess.run(["pi", "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip() or None


def _prepared(project_root: Path) -> Path:
    return project_root / PREPARED_DIR


# --------------------------------------------------------------------- acceptance


def cmd_mof_acceptance(args: Any) -> int:
    """Check the implementation, not the result. No model call is made."""
    project_root = Path(args.project_root).resolve()
    prepared = _prepared(project_root)
    runner_script = project_root / "tasks" / "mof_screening" / "candidate_runner.py"
    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str, **extra: Any) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail, **extra})

    pool = MofPool.development(prepared, 101)
    labels = pool.load_labels()
    workdir = project_root / ".runtime" / "mof_experiment" / "acceptance"
    evaluator = MofScreeningEvaluator(
        pool=pool, labels=labels, runner_script=runner_script, workdir_root=workdir, k=K,
    )

    # 1. The shipped starting program scores, and the score is in range.
    verdict = evaluator.evaluate(
        project_root / "tasks" / "mof_screening" / "initial", invocation_id="baseline")
    record("grader.baseline_scores", verdict.status.value == "SUCCESS" and 0.0 < verdict.score < 1.0,
           f"starting program S={verdict.score:.6f} on split 101 development pool",
           Q=verdict.detail.get("Q"), Q_star=verdict.detail.get("Q_star"))

    # 2. The private half of the result never carries the labels or the reference.
    leaked = sorted(set(verdict.public_detail) & {"Q", "Q_star", "labels", "ranking",
                                                  "top_k_uptakes", "recommended_uptakes"})
    record("grader.public_half_is_clean", not leaked,
           "public diagnostics carry only the score, a status and a contract class",
           leaked=leaked)

    # 3. Contract failures are distinguished from bad-but-valid programs.
    probe = workdir / "contract-probe"
    probe.mkdir(parents=True, exist_ok=True)
    outcomes: dict[str, str] = {}
    for name, source in (
        ("forbidden_import", "import socket\ndef rank(*a, **k): return []\n"),
        ("missing_entry", "x = 1\n"),
        ("wrong_count",
         "def rank(tX, ty, pX, *, k, seed):\n    return pX['material_id'].tolist()[:k-1]\n"),
        ("raises",
         "def rank(tX, ty, pX, *, k, seed):\n    raise RuntimeError('boom')\n"),
    ):
        (probe / "candidate.py").write_text(source, encoding="utf-8")
        outcomes[name] = str(evaluator.evaluate(probe, invocation_id=name).error_type)
    record("grader.contract_failures_are_labelled",
           set(outcomes.values()) == {"FORBIDDEN_IMPORT", "MISSING_ENTRY_POINT",
                                      "WRONG_COUNT", "CANDIDATE_RAISED"},
           "each failure mode reports its own contract error", outcomes=outcomes)

    # 4. Isolation: the vault really removes the labels, and puts them back.
    vault = FileVault(manifest_path=workdir / "vault-manifest.jsonl")
    protected = default_protected_paths(prepared, dataset_root=project_root / DATA_DIR,
                                        raw_root=project_root / RAW_DIR)
    sealed = vault.seal(protected)
    verified = vault.verify_absent()
    record("isolation.protected_paths_are_absent", bool(verified["ok"]),
           "every protected path was opened after sealing; none was readable",
           files=sealed["files"], readable=verified["still_readable"])
    restored = None
    try:
        # 5. Two independent routes to the network, both closed.
        #    (a) the grader's own screen reads the syntax tree, so a name built
        #        by concatenation is refused because `__import__` is refused.
        guard_probe = workdir / "guard-probe"
        guard_probe.mkdir(parents=True, exist_ok=True)
        (guard_probe / "candidate.py").write_text(
            "def rank(tX, ty, pX, *, k, seed):\n"
            "    m = __import__('sock' + 'et')\n"
            "    return pX['material_id'].tolist()[:k]\n",
            encoding="utf-8")
        blocked = evaluator.evaluate(guard_probe, invocation_id="guard")
        record("isolation.dynamic_import_is_refused_statically",
               blocked.error_type == "FORBIDDEN_IMPORT",
               "a name assembled at run time is refused by the source screen",
               error_type=blocked.error_type)
        #    (b) and if a route past the screen ever exists, the runner's import
        #        guard is armed inside the candidate's own process. This runs the
        #        guard directly rather than through a candidate, because a route
        #        the screen already closes cannot demonstrate it.
        probe = (
            "import sys\n"
            f"sys.path.insert(0, {str(runner_script.parent)!r})\n"
            "import candidate_runner as runner\n"
            "runner.install_import_guard()\n"
            "try:\n"
            "    import socket\n"
            "except ImportError as exc:\n"
            "    print('BLOCKED:' + str(exc))\n"
            "else:\n"
            "    print('REACHED')\n"
        )
        probe_path = workdir / "guard_probe.py"
        probe_path.write_text(probe, encoding="utf-8")
        guard_run = subprocess.run([sys.executable, "-I", str(probe_path)],
                                   capture_output=True, text=True, timeout=180)
        record("isolation.import_guard_is_armed_in_the_candidate_process",
               guard_run.stdout.strip().startswith("BLOCKED:"),
               "the runtime guard refuses `import socket` inside the runner",
               stdout=guard_run.stdout.strip()[:160])
    finally:
        vault.release()
        restored = vault.verify_restored()
    record("isolation.protected_paths_are_restored", bool(restored["ok"]),
           "every sealed file is back and byte-identical",
           missing=restored["missing"], mismatched=restored["sha256_mismatch"])

    # 6. Labels are not reachable from a candidate's working directory.
    scratch = workdir / "baseline" / "run"
    reachable = [p.name for p in scratch.glob("*.csv")] if scratch.is_dir() else []
    record("isolation.scratch_has_no_labels",
           "dev_targets.private.csv" not in reachable
           and "final_test_targets.private.csv" not in reachable,
           "a candidate's scratch directory holds features and training targets only",
           files=sorted(reachable))

    # 7. The runtime guard does not break the allowlisted stack.
    record("isolation.allowlisted_stack_still_works",
           verdict.status.value == "SUCCESS",
           "the baseline fits under the guard, so the guard is not a blanket ban")

    # 8. The protocol is frozen and self-describing.
    protocol = Protocol(phase=args.phase, prepared_root=str(prepared), pi_version=_pi_version())
    record("protocol.digest_is_stable",
           protocol.digest == Protocol(phase=args.phase, prepared_root=str(prepared),
                                       pi_version=_pi_version()).digest,
           f"protocol digest {protocol.digest[:16]}",
           thinking=THINKING, model=MODEL)

    passed = all(check["ok"] for check in checks)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {"system": platform.system(), "release": platform.release()},
        "python": sys.version.split()[0],
        "protocol_digest": protocol.digest,
        "checks": checks,
        "passed": passed,
    }
    write_json(project_root / ".runtime" / "mof_experiment" / "acceptance.json", report)
    for check in checks:
        print(f"  [{'ok ' if check['ok'] else 'FAIL'}] {check['check']}: {check['detail']}")
    _status("PASSED" if passed else "FAILED",
            report=str(project_root / ".runtime" / "mof_experiment" / "acceptance.json"))
    return 0 if passed else 1


# --------------------------------------------------------------------- run


def cmd_mof_run(args: Any) -> int:
    project_root = Path(args.project_root).resolve()
    prepared = _prepared(project_root)
    protocol = Protocol(phase=args.phase, prepared_root=str(prepared), pi_version=_pi_version())
    plans = block_plans(args.phase)
    if args.only_split_seed is not None:
        plans = [p for p in plans if p.split_seed == args.only_split_seed]
    if args.only_repeat is not None:
        plans = [p for p in plans if p.repeat == args.only_repeat]
    if args.block_limit is not None:
        keep: list[Any] = []
        seen: list[str] = []
        for plan in plans:
            if plan.block_id not in seen:
                seen.append(plan.block_id)
            if seen.index(plan.block_id) >= args.block_limit:
                continue
            keep.append(plan)
        plans = keep

    protect = [] if args.no_isolation else default_protected_paths(
        prepared, dataset_root=project_root / DATA_DIR, raw_root=project_root / RAW_DIR)
    runner = MofExperimentRunner(
        project_root=project_root, protocol=protocol,
        task_dir=project_root / "tasks" / "mof_screening",
        runner_script=project_root / "tasks" / "mof_screening" / "candidate_runner.py",
        out_root=project_root / OUT_DIR / args.phase,
    )
    print(f"phase {args.phase}: {len(plans)} arm-runs, digest {protocol.digest[:16]}, "
          f"isolation {'off' if args.no_isolation else 'on'}")
    summary = runner.run_phase(plans, protect=protect)
    blocks = summary["blocks"]
    complete = sum(1 for b in blocks if b["status"] == "COMMITTED")
    spent = sum(int(b["ledger"].get("charged_total") or 0) for b in blocks)
    _status("COMPLETED", phase=args.phase, arm_runs=len(blocks), committed=complete,
            tokens=spent, out=str(project_root / OUT_DIR / args.phase))
    return 0


# --------------------------------------------------------------------- unseal


def _latest_blocks(path: Path) -> list[dict[str, Any]]:
    """One record per arm-run: the last one written.

    A resumed phase appends another line for a block it continued, and the later
    line is the one carrying the completed tree and the full ledger. Deduping by
    run id keeps a re-run from unsealing the same candidate twice.
    """
    latest: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        latest[str(row.get("run_id", ""))] = row
    return list(latest.values())


def _unseal_dir(project_root: Path, phase: str) -> Path:
    return project_root / OUT_DIR / phase / "unsealed"


def cmd_mof_unseal(args: Any) -> int:
    """Open the sealed pool once, for candidates that were frozen in advance.

    ``--references-only`` recomputes the fixed screening references on the same
    sealed pool without re-grading any candidate. It exists because the
    references are a pure function of the pool and the training set, so a later
    fix to how they are computed does not invalidate the frozen candidates and
    must not force a second pass over them.
    """
    project_root = Path(args.project_root).resolve()
    prepared = _prepared(project_root)
    runner_script = project_root / "tasks" / "mof_screening" / "candidate_runner.py"
    records = _latest_blocks(project_root / OUT_DIR / args.phase / f"{args.phase}-blocks.jsonl")
    if getattr(args, "references_only", False):
        target = _unseal_dir(project_root, args.phase)
        references: dict[str, Any] = {}
        for split_seed in sorted({int(r["plan"]["split_seed"]) for r in records}):
            pool = MofPool.final_test(prepared, split_seed=split_seed)
            references[str(split_seed)] = material_references(
                pool_features=pool.pool_features, labels=pool.load_labels(),
                train_features=pool.train_features, train_targets=pool.train_targets, k=K,
            )
        write_json(target / "references.json", references)
        _status("COMPLETED", phase=args.phase, references=len(references), out=str(target))
        return 0
    if not records:
        _status("FAILED", reason=f"no block records under {OUT_DIR / args.phase}")
        return 1
    target = _unseal_dir(project_root, args.phase)
    results: list[dict[str, Any]] = []
    for record in records:
        if record.get("status") != "COMMITTED" and not record.get("final_candidate"):
            continue
        try:
            verdicts = unseal_block(
                record=record, prepared_root=prepared, runner_script=runner_script,
                workdir_root=target / "work",
            )
        except Exception as exc:  # noqa: BLE001 - a failed unseal is a recorded outcome
            results.append({"run_id": record["run_id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        results.extend(verdict.to_dict() for verdict in verdicts)
    write_json(target / "unsealed.json", results)

    # The references use the same sealed pool, so they are computed here and
    # nowhere earlier: before unsealing there is no test label in this process.
    references: dict[str, Any] = {}
    for split_seed in sorted({int(r["plan"]["split_seed"]) for r in records}):
        pool = MofPool.final_test(prepared, split_seed=split_seed)
        references[str(split_seed)] = material_references(
            pool_features=pool.pool_features, labels=pool.load_labels(),
            train_features=pool.train_features, train_targets=pool.train_targets, k=K,
        )
    write_json(target / "references.json", references)
    _status("COMPLETED", phase=args.phase, unsealed=len(results), out=str(target))
    return 0


# --------------------------------------------------------------------- report


def _load_unsealed(target: Path) -> list[UnsealResult]:
    rows = json.loads((target / "unsealed.json").read_text(encoding="utf-8"))
    return [UnsealResult(**{k: v for k, v in row.items() if k in UnsealResult.__dataclass_fields__})
            for row in rows if "error" not in row]


def cmd_mof_report(args: Any) -> int:
    project_root = Path(args.project_root).resolve()
    prepared = _prepared(project_root)
    phase_dir = project_root / OUT_DIR / args.phase
    target = _unseal_dir(project_root, args.phase)
    fig_dir = target / "figures"
    blocks = _latest_blocks(phase_dir / f"{args.phase}-blocks.jsonl")
    unsealed = _load_unsealed(target) if (target / "unsealed.json").is_file() else []
    references = (json.loads((target / "references.json").read_text(encoding="utf-8"))
                  if (target / "references.json").is_file() else {})

    figures: dict[str, Any] = {}
    if blocks:
        figures["quality_budget_staircase"] = quality_budget_staircase(blocks, fig_dir)
        figures["internal_mechanism"] = internal_mechanism(blocks, fig_dir)
    table: dict[str, Any] = {}
    if unsealed:
        table = paired_table(unsealed)
        checkpoints = _checkpoint_series(blocks, unsealed)
        figures["recursive_dynamics"] = recursive_dynamics(checkpoints, fig_dir)
        figures["paired_final_comparison"] = paired_final_comparison(table, fig_dir)
        figures["recommendation_explanation"] = recommendation_explanation(
            [row.to_dict() for row in unsealed if row.is_final],
            _average_references(references), fig_dir)

    report = {
        "phase": args.phase,
        "protocol_digest": blocks[0]["protocol_digest"] if blocks else None,
        "blocks": len(blocks),
        "arm_runs": len(blocks),
        "unsealed": sum(1 for row in unsealed if row.is_final),
        "unsealed_checkpoint_candidates": len(unsealed),
        "paired": table,
        "gap": development_test_gap(unsealed) if unsealed else {},
        "references": references,
        "figures": figures,
        "ledger": _ledger_totals(blocks),
    }
    write_json(target / "report.json", report)
    _write_summary(target / "summary.md", report, blocks, unsealed, references)
    _status("COMPLETED", phase=args.phase, figures=len(figures), out=str(target / "summary.md"))
    return 0


def _checkpoint_series(
    blocks: Sequence[dict[str, Any]], unsealed: Sequence[UnsealResult],
) -> list[dict[str, Any]]:
    """Sealed-pool S at each budget checkpoint, via each run's frozen candidates.

    A checkpoint's candidate is whatever the block froze for that checkpoint --
    chosen on the development score before the pool was opened -- and its sealed
    score is looked up from the unseal pass rather than recomputed here.
    """
    by_source = {row.candidate_sha256: row for row in unsealed}
    series: list[dict[str, Any]] = []
    for record in blocks:
        plan = record["plan"]
        points = []
        for frozen in record.get("frozen", []):
            match = by_source.get(str(frozen["candidate_sha256"]))
            if match is None:
                continue
            points.append({
                "checkpoint": int(frozen["checkpoint"]),
                "test_S": match.test_S, "test_Q": match.test_Q,
                "development_S": float(frozen["development_S"]),
                "origin": frozen["origin"],
            })
        if points:
            series.append({"split_seed": plan["split_seed"], "repeat": plan["repeat"],
                           "method": plan["method"], "run_id": record["run_id"],
                           "points": points})
    return series


def _average_references(references: Mapping[str, Any]) -> dict[str, Any]:
    """Mean the fixed references across the splits' training sets.

    Every split scores the *same* sealed pool but fits its references on its own
    training set, so there is one reference per split, not one reference. The
    figure shows the mean of what was actually computed and the report keeps the
    per-split table; showing a single split would overstate how fixed they are.
    """
    keys = ("random", "ridge", "random_forest")
    out: dict[str, Any] = {}
    for key in keys:
        blocks = [value[key] for value in references.values()
                  if isinstance(value, dict) and key in value]
        if not blocks:
            continue
        merged: dict[str, Any] = {}
        for name in ("S", "Q", "Q_star", "mean_S"):
            values = [float(block[name]) for block in blocks if name in block]
            if values:
                merged[name] = sum(values) / len(values)
        merged["splits_averaged"] = len(blocks)
        out[key] = merged
    if any("pore_volume" in (value or {}) for value in references.values()):
        out["pore_volume"] = {"available": False,
                              "reason": "no accessible-pore-volume column exists in the "
                                        "pinned CRAFTED 2.0.1 descriptor frame"}
    return out


def _ledger_totals(blocks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = {"requests": 0, "refusals": 0, "charged_total": 0,
             "requests_without_reported_usage": 0,
             "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0,
                        "reasoning": 0}}
    cost_amount = 0.0
    cost_known = 0
    for record in blocks:
        ledger = record.get("ledger") or {}
        total["requests"] += int(ledger.get("requests") or 0)
        total["refusals"] += int(ledger.get("refusals") or 0)
        total["charged_total"] += int(ledger.get("charged_total") or 0)
        total["requests_without_reported_usage"] += int(
            ledger.get("requests_without_reported_usage") or 0)
        for key, value in (ledger.get("tokens") or {}).items():
            total["tokens"][key] = total["tokens"].get(key, 0) + int(value or 0)
        cost = ledger.get("cost") or {}
        if cost.get("known") and cost.get("amount") is not None:
            cost_amount += float(cost["amount"])
            cost_known += 1
    total["cost"] = {
        "known": cost_known > 0 and cost_known == len(blocks),
        "blocks_with_cost": cost_known,
        "amount": cost_amount if cost_known else None,
        "note": ("the service reports no per-request cost on this provider; the "
                 "report gives tokens and says the money figure is unknown"
                 if not cost_known else ""),
    }
    return total


def _write_summary(
    path: Path, report: dict[str, Any], blocks: Sequence[dict[str, Any]],
    unsealed: Sequence[UnsealResult], references: dict[str, Any],
) -> None:
    phase = report["phase"]
    ledger = report["ledger"]
    lines: list[str] = []
    lines.append(f"# MOF screening experiment — {phase} phase\n")
    lines.append(f"- Protocol digest: `{report['protocol_digest']}`")
    lines.append(f"- Model: `{MODEL['provider']}` / `{MODEL['model']}`, thinking `{THINKING}`")
    lines.append(f"- Arm-runs recorded: {report['arm_runs']}; unsealed candidates: "
                 f"{report['unsealed']}")
    lines.append("")
    lines.append("## Cost\n")
    lines.append(f"- Service requests: **{ledger['requests']}** "
                 f"(refused before sending: {ledger['refusals']})")
    lines.append(f"- Tokens charged: **{ledger['charged_total']:,}** "
                 f"(input {ledger['tokens']['input']:,}, output {ledger['tokens']['output']:,}, "
                 f"reasoning {ledger['tokens']['reasoning']:,})")
    lines.append(f"- Requests with no reported usage: "
                 f"{ledger['requests_without_reported_usage']}")
    lines.append(f"- Money: {'unknown' if not ledger['cost']['known'] else ledger['cost']['amount']}"
                 " — the provider reports no per-request cost, so no currency figure is claimed.")
    lines.append("")
    if report["paired"]:
        verdict = report["paired"]["A_minus_B_verdict"]
        ab = report["paired"]["A_minus_B"]
        lines.append("## Main result\n")
        lines.append(f"- Paired blocks: {report['paired']['complete_blocks']} complete of "
                     f"{report['paired']['blocks']}")
        lines.append(f"- **A − B** on sealed-pool S: mean {ab['mean']:+.4f}, "
                     f"exploratory 95% [{ab['ci95'][0]:+.4f}, {ab['ci95'][1]:+.4f}], "
                     f"{ab['consistent_direction']}/{len(ab['splits'])} splits favouring A")
        lines.append(f"- Pre-registered reading: **{verdict['verdict']}** "
                     f"({verdict['rules']})")
        lines.append("")
        if report.get("gap"):
            lines.append("## Development against sealed pool\n")
            lines.append("| arm | runs | mean dev S | mean test S | mean gap | failures |")
            lines.append("| --- | --- | --- | --- | --- | --- |")
            for arm in sorted(report["gap"]):
                row = report["gap"][arm]
                lines.append(f"| {arm} | {row['n']} | {row['mean_development_S']:.4f} | "
                             f"{row['mean_test_S']:.4f} | {row['mean_gap']:+.4f} | "
                             f"{row['failures']} |")
            lines.append("")
    else:
        lines.append("## Main result\n")
        lines.append("No candidates were unsealed, so this phase reports no effect. "
                     "The sealed pool was not opened.\n")
    lines.append("## Figures\n")
    for name, paths in sorted(report.get("figures", {}).items()):
        lines.append(f"- `{name}`: PNG `{paths['png']}`, SVG `{paths['svg']}`, "
                     f"source data `{paths['csv']}`")
    lines.append("")
    lines.append("## Limits\n")
    lines.append("- The development pool is a finite sample of the same material library the "
                 "sealed pool comes from, and the five splits share that pool. Intervals are "
                 "exploratory and are not a claim about MOFs in general.")
    lines.append("- The provider reports no per-request money cost, so the report gives tokens "
                 "and leaves the currency figure unknown rather than estimating one.")
    lines.append("- Extended reasoning was fixed `off` for every call in every arm; the probe "
                 "measurements behind that choice are recorded in `src/dream_rsi/mof/protocol.py`.")
    lines.append("- Internal Eq(1) `V` is a policy-selection signal. It is shown for mechanism "
                 "only and is never used as evidence that screening improved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


__all__ = [
    "cmd_mof_acceptance", "cmd_mof_report", "cmd_mof_run", "cmd_mof_unseal",
]
