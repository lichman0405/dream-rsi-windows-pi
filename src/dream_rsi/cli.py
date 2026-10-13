"""``dream-rsi`` command line.

The subcommands mirror the work order, in the order they are meant to be run:

``doctor``        environment facts, written to ``reports/ENV_STATUS.json``
``bootstrap``     delegates to ``tools/bootstrap_windows.ps1``
``offline-test``  the whole loop with no model in it, asserting replay is inert
``model-check``   a **real** API request to DSH or Pi; never a stub
``run``           the actual loop (``--backend fixture`` or a live backend)
``report``        reads a finished run back out of the registry

Every command prints a machine-readable ``STATUS:`` line so a caller (or a
script) can branch without parsing prose.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .adapters.base import DiscoveryBackend, TaskSpec
from .config import (
    BACKEND_IDENTITY_ENV,
    KNOWN_TASKS,
    Config,
    load_config,
    standard_library_only_for,
)
from .domain.errors import BlockedModelConfiguration, DreamRsiError
from .domain.models import new_id
from .evaluator.command import FixedCommandEvaluator
from .policy.runner import SubprocessPolicyRunner
from .runtime.controller import Controller, ControllerDeps
from .runtime.llm_policy_developer import LlmPolicyDeveloper
from .runtime.online import CallBudget
from .runtime.policy_development import PolicyDeveloper, ScriptedVariantDeveloper
from .storage.cas import ContentAddressedStore
from .storage.sqlite_registry import Registry

REPORTS = "reports"


def _status(token: str, **extra: Any) -> None:
    payload = {"status": token, **extra}
    print("STATUS: " + json.dumps(payload, sort_keys=True))


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def _task_spec(config: Config) -> TaskSpec:
    instructions = ""
    if config.task_md.is_file():
        instructions = config.task_md.read_text(encoding="utf-8")
    return TaskSpec(
        task_id=config.task_dir.name,
        entry_filename=config.entry_filename,
        instructions=instructions,
        seed_workspace=config.seed_workspace,
        is_paper_benchmark=False,
    )


def _require_live_identity(config: Config, backend: str) -> tuple[str, str]:
    """Provider and model must be named explicitly for a real backend.

    There is no safe default: guessing would spend the user's tokens on a model
    they did not pick, and the resulting numbers would be attributed to the
    wrong system. An unset value is a configuration error, not a default.
    """
    missing = [
        name
        for name, value in (("provider", config.provider), ("model", config.model))
        if not value
    ]
    if missing:
        raise BlockedModelConfiguration(
            f"backend {backend!r}: {' and '.join(missing)} not configured. Pass "
            "--provider/--model, or set the backend's own environment variables; "
            "this project never reuses another tool's credentials."
        )
    return str(config.provider), str(config.model)


def _build_backend(config: Config, run_id: str) -> DiscoveryBackend:
    kinds = {
        "fixture": lambda: _fixture_backend(),
        "dsh": lambda: _dsh_backend(config, run_id),
        "pi": lambda: _pi_backend(config, run_id),
    }
    return kinds[config.backend]()


def _fixture_backend() -> DiscoveryBackend:
    from .adapters.fixture import FixtureDiscoveryBackend
    return FixtureDiscoveryBackend()


def _dsh_backend(config: Config, run_id: str) -> DiscoveryBackend:
    from .adapters.dsh import DshDiscoveryBackend
    provider, model = _require_live_identity(config, "dsh")
    return DshDiscoveryBackend(
        workdir_root=config.run_root(run_id) / "agent",
        provider=provider,
        model=model,
        dsh_home_root=config.dsh_home_root(run_id),
        timeout_s=config.timeouts.agent_s,
        memory_limit_mb=config.timeouts.agent_memory_mb,
    )


def _pi_backend(config: Config, run_id: str) -> DiscoveryBackend:
    from .adapters.pi import PiDiscoveryBackend
    # Naming the model is mandatory, exactly as for DSH. Pi does have its own
    # global default, but it is not necessarily the model the user wired up, and
    # a run that silently used a different one would have its results attributed
    # to a model that never ran.
    provider, model = _require_live_identity(config, "pi")
    return PiDiscoveryBackend(
        workdir_root=config.run_root(run_id) / "agent",
        provider=provider,
        model=model,
        timeout_s=config.timeouts.agent_s,
        memory_limit_mb=config.timeouts.agent_memory_mb,
        no_tools=config.pi_no_tools,
        response_only=config.pi_response_only,
        standard_library_only=standard_library_only_for(config.task_dir.name),
    )


def _build_developer(config: Config, run_id: str) -> PolicyDeveloper | None:
    """The policy-development step, or ``None`` when M=1 (no revisions).

    ``llm`` is the only setting that makes a run live end to end: it is the one
    that puts a real model on the *policy* side of the loop. ``scripted`` is the
    offline fixture and is what makes a live-discovery run a hybrid — which is
    why the controller refuses to label such a run ``LIVE`` on its own.
    """
    if config.loop.M <= 1:
        return None
    if config.developer == "llm":
        provider, model = _require_live_identity(config, config.backend)
        return LlmPolicyDeveloper(
            workdir_root=config.run_root(run_id) / "policy_agent",
            provider=provider,
            model=model,
            timeout_s=config.timeouts.agent_s,
            memory_limit_mb=config.timeouts.agent_memory_mb,
        )
    return ScriptedVariantDeveloper()


def _build_controller(
    config: Config, run_id: str, *, with_developer: bool,
    max_model_calls: int | None = None,
) -> tuple[Controller, Registry]:
    task = _task_spec(config)
    if not config.seed_workspace.is_dir():
        raise DreamRsiError(f"seed workspace is missing: {config.seed_workspace}")
    if not config.evaluator_harness.is_file():
        raise DreamRsiError(f"fixed evaluator harness is missing: {config.evaluator_harness}")

    backend = _build_backend(config, run_id)
    evaluator = FixedCommandEvaluator(
        harness_path=config.evaluator_harness,
        workdir_root=config.run_root(run_id) / "eval",
        timeout_s=config.timeouts.evaluator_s,
        memory_limit_mb=config.timeouts.evaluator_memory_mb,
    )
    registry = Registry(config.db_path)
    store = ContentAddressedStore(config.cas_root)
    policy_runner = SubprocessPolicyRunner(
        workdir_root=config.run_root(run_id) / "policy_sessions",
        timeout_s=config.timeouts.policy_select_s,
        memory_limit_mb=config.timeouts.policy_memory_mb,
    )

    developer = _build_developer(config, run_id) if with_developer else None

    deps = ControllerDeps(
        backend=backend, evaluator=evaluator, registry=registry, store=store,
        policy_runner=policy_runner, task=task, developer=developer,
        budget=CallBudget(max_model_calls),
    )
    return Controller(config, deps), registry


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------


def _which(name: str) -> str | None:
    return shutil.which(name)


def _dsh_sdk_available() -> tuple[bool, str]:
    try:
        import deepseek_harness  # noqa: F401
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, "importable"


def _model_config_present(config: Config) -> dict[str, Any]:
    """Detect credentials WITHOUT reading anyone else's config files.

    Only environment variables that the user themselves would set for DSH/Pi are
    consulted. Claude Code's own configuration is never opened.
    """
    dsh_key = bool(os.environ.get("DEEPSEEK_API_KEY"))
    dsh_base = bool(os.environ.get("DEEPSEEK_BASE_URL"))
    pi_env = sorted(k for k in os.environ if k.upper().startswith("PI_"))
    pi_auth = Path(os.environ.get("USERPROFILE") or Path.home()) / ".pi" / "agent" / "auth.json"
    return {
        "dsh_api_key_set": dsh_key,
        "dsh_base_url_set": dsh_base,
        "dsh_configured": dsh_key and dsh_base,
        "pi_env_vars": pi_env,
        "pi_auth_file_present": pi_auth.is_file(),
        "pi_configured": bool(pi_env) or pi_auth.is_file(),
        "provider_override": config.provider,
        "model_override": config.model,
    }


def cmd_doctor(args: argparse.Namespace) -> int:
    config = load_config(project_root=args.project_root, backend=args.backend,
                         task=getattr(args, "task", None))
    from .runtime.winjob import job_support_report

    sdk_ok, sdk_detail = _dsh_sdk_available()
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "is_windows": sys.platform == "win32",
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "in_venv": sys.prefix != getattr(sys, "base_prefix", sys.prefix),
        },
        "tools": {
            "dsh": _which("dsh"),
            "pi": _which("pi"),
            "pwsh": _which("pwsh"),
            "bash": _which("bash"),
            "git": _which("git"),
        },
        "dsh_sdk": {"available": sdk_ok, "detail": sdk_detail},
        "job_objects": job_support_report(),
        "model_configuration": _model_config_present(config),
        "paths": {
            "project_root": str(config.project_root),
            "runtime_root": str(config.runtime_root),
            "seed_workspace": str(config.seed_workspace),
            "evaluator_harness": str(config.evaluator_harness),
            "seed_workspace_present": config.seed_workspace.is_dir(),
            "evaluator_harness_present": config.evaluator_harness.is_file(),
        },
        "cfg_digest": config.cfg_digest,
    }

    blockers: list[str] = []
    if not report["platform"]["is_windows"]:
        blockers.append("NOT_WINDOWS")
    if not report["paths"]["seed_workspace_present"]:
        blockers.append("MISSING_SEED_WORKSPACE")
    if not report["paths"]["evaluator_harness_present"]:
        blockers.append("MISSING_EVALUATOR_HARNESS")
    if report["tools"]["dsh"] is None:
        blockers.append("DSH_NOT_ON_PATH")
    if report["tools"]["pi"] is None:
        blockers.append("PI_NOT_ON_PATH")
    if not sdk_ok:
        blockers.append("DSH_SDK_NOT_IMPORTABLE")
    if not report["job_objects"]["available"]:
        blockers.append("JOB_OBJECTS_UNAVAILABLE")
    report["blockers"] = blockers
    report["offline_ready"] = not any(
        b in blockers for b in ("NOT_WINDOWS", "MISSING_SEED_WORKSPACE", "MISSING_EVALUATOR_HARNESS")
    )
    report["live_ready"] = bool(
        report["model_configuration"]["dsh_configured"]
        or report["model_configuration"]["pi_configured"]
    )

    out = config.project_root / REPORTS / "ENV_STATUS.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    _status(
        "OK" if report["offline_ready"] else "BLOCKED_ENVIRONMENT",
        env_status=str(out), blockers=blockers, live_ready=report["live_ready"],
    )
    return 0 if report["offline_ready"] else 3


def cmd_bootstrap(args: argparse.Namespace) -> int:
    script = Path(args.project_root or ".").resolve() / "tools" / "bootstrap_windows.ps1"
    if not script.is_file():
        _status("FAILED", reason=f"bootstrap script not found at {script}")
        return 3
    shell = _which("pwsh") or _which("powershell")
    if shell is None:
        _status("FAILED", reason="neither pwsh nor powershell is on PATH")
        return 3
    proc = subprocess.run(
        [shell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        cwd=str(script.parent.parent), check=False,
    )
    _status("OK" if proc.returncode == 0 else "FAILED", exit_code=proc.returncode)
    return proc.returncode


# --------------------------------------------------------------------------
# model-check  (always a real request)
# --------------------------------------------------------------------------


def _probe_environment(config: Config, backend: str) -> dict[str, str]:
    """The environment the model probe has to see.

    The probe reads provider and model from the backend's own variables
    (``config.BACKEND_IDENTITY_ENV``), not from its argv -- its only arguments
    are ``--backend`` and ``--root``. So forwarding the CLI's
    ``--provider``/``--model`` means exporting them here.

    Omitting this does not fail loudly, which is the whole problem: the probe
    falls back to the backend's globally configured default model, validates
    *that*, and reports the result as if it were about the model the caller
    named. A working provider then shows up as ``AUTHENTICATION_FAILED``.
    """
    env = dict(os.environ)
    names = BACKEND_IDENTITY_ENV.get(backend)
    if names:
        if config.provider:
            env[names[0]] = config.provider
        if config.model:
            env[names[1]] = config.model
    return env


def cmd_model_check(args: argparse.Namespace) -> int:
    config = load_config(
        project_root=args.project_root, backend=args.backend,
        provider=args.provider, model=args.model,
        task=getattr(args, "task", None),
    )
    probe = config.project_root / "tools" / "model_probe.py"
    if not probe.is_file():
        _status("FAILED", reason=f"model probe not found at {probe}")
        return 3

    report = _model_config_present(config)
    if args.backend == "dsh" and not report["dsh_configured"]:
        _status(
            "BLOCKED_MODEL_CONFIGURATION",
            backend="dsh",
            reason="DEEPSEEK_API_KEY and DEEPSEEK_BASE_URL are not both set in this process",
            how_to_fix=(
                "set the two variables in the shell that launches dream-rsi; this project "
                "never reads another tool's credentials"
            ),
        )
        return 4
    if args.backend == "pi" and not report["pi_configured"]:
        _status(
            "BLOCKED_MODEL_CONFIGURATION",
            backend="pi",
            reason="no PI_* environment variable and no ~/.pi/agent/auth.json",
            how_to_fix="run `pi /login` (or export PI_* settings) before model-check",
        )
        return 4

    started = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-I", str(probe), "--backend", args.backend,
         "--root", str(config.state_root)],
        cwd=str(config.project_root), check=False,
        env=_probe_environment(config, args.backend),
    )
    elapsed = round(time.monotonic() - started, 3)
    _status(
        "OK" if proc.returncode == 0 else "FAILED",
        backend=args.backend, exit_code=proc.returncode, duration_s=elapsed,
    )
    return proc.returncode


# --------------------------------------------------------------------------
# run / offline-test
# --------------------------------------------------------------------------


def _run(args: argparse.Namespace, *, live_required: bool) -> int:
    overrides: dict[str, Any] = {}
    if args.outer_rounds is not None:
        overrides["outer_rounds"] = args.outer_rounds
    if args.k1 is not None:
        overrides["K1"] = args.k1
    if args.k2 is not None:
        overrides["K2"] = args.k2
    if args.workers is not None:
        overrides["W"] = args.workers
    if args.versions is not None:
        overrides["M"] = args.versions

    config = load_config(
        project_root=args.project_root, backend=args.backend,
        provider=args.provider, model=args.model, overrides=overrides,
        run_id=args.run_id, resume=args.resume,
        developer=getattr(args, "developer", "scripted"),
        task=args.task,
        agent_timeout_s=getattr(args, "agent_timeout_s", None),
        pi_no_tools=getattr(args, "pi_no_tools", False),
        pi_response_only=getattr(args, "pi_response_only", False),
    )
    # The run id must exist before any path is derived from it: the controller,
    # the evaluator workdir and the policy workdirs all hang off it, and they
    # must agree or a resumed run would look in the wrong directory.
    run_id: str = config.run_id or new_id("run")
    if config.run_id is None:
        from dataclasses import replace as _replace
        config = _replace(config, run_id=run_id)
    config.ensure_dirs(run_id)

    if live_required:
        report = _model_config_present(config)
        key = "dsh_configured" if args.backend == "dsh" else "pi_configured"
        if args.backend != "fixture" and not report[key]:
            _status(
                "BLOCKED_MODEL_CONFIGURATION", backend=args.backend,
                reason="no credentials configured for this backend",
            )
            return 4

    # Construction is inside the try: building a live backend can itself refuse
    # (an unnamed provider or model), and that refusal must be reported as a
    # configuration blocker rather than escaping as a traceback.
    registry = None
    try:
        controller, registry = _build_controller(
            config, run_id, with_developer=True,
            max_model_calls=getattr(args, "max_model_calls", None),
        )
        outcome = controller.run()
    except BlockedModelConfiguration as exc:
        _status("BLOCKED_MODEL_CONFIGURATION", reason=str(exc))
        return 4
    except DreamRsiError as exc:
        _status("FAILED", error_type=type(exc).__name__, reason=str(exc)[:400])
        return 5
    finally:
        # May be None when construction refused before opening the store.
        if registry is not None:
            registry.close()

    run_registry = Registry(config.db_path)
    try:
        replays = _replays_for_run(run_registry, outcome.run_id)
    finally:
        run_registry.close()

    agent_calls = sum(e.discovery_agent_calls for e in replays)
    evaluator_calls = sum(e.evaluator_calls for e in replays)

    print(json.dumps({
        "run_id": outcome.run_id,
        "mode": outcome.mode.value,
        "state": outcome.state.value,
        "replay_episodes": len(replays),
        "replay_agent_calls": agent_calls,
        "replay_evaluator_calls": evaluator_calls,
        # The online side of the same ledger. Replay's zeros only mean something
        # next to the nonzero number the online phase actually incurred.
        "online_agent_calls": outcome.online_agent_calls,
        "online_evaluator_calls": outcome.online_evaluator_calls,
        # Both sides of the loop have to be live for `mode` to be live, so the
        # report states which developer ran rather than leaving it to be
        # inferred from the mode.
        "developer_id": outcome.developer_id,
        "developer_is_live": outcome.developer_is_live,
        "policy_agent_calls": outcome.policy_agent_calls,
        "budget": outcome.budget,
        "worlds": [
            {"world_id": w.world_id, "outer_t": w.outer_t, "state": w.state.value,
             "nodes": w.node_count, "B": w.b_count, "S": w.s_best}
            for w in outcome.worlds
        ],
        "final_policy_id": outcome.final_policy_id,
        "selections": [
            {"t": s.iteration_t, "selected": s.selected_policy_id,
             "deployed_again": s.deployed_again, "reason": s.reason,
             "mean_values": s.mean_values}
            for s in outcome.selections
        ],
        "notes": outcome.notes,
    }, indent=2, default=str))

    if agent_calls or evaluator_calls:
        _status("FAILED", reason="replay touched a discovery agent or the evaluator")
        return 6
    _status("OK", run_id=outcome.run_id, mode=outcome.mode.value, state=outcome.state.value)
    return 0


def _replays_for_run(registry: Registry, run_id: str) -> list[Any]:
    """Every replay belonging to ``run_id``.

    Replays are keyed by (world, policy, rho), not by run, so a run's episodes
    are collected through the policies it created — the registry has no direct
    run→replay path and inventing one would mean duplicating its schema.
    """
    policy_ids = {p.version_id for p in registry.list_policies(run_id)}
    return [e for e in registry.list_replays() if e.policy_version_id in policy_ids]


def cmd_offline_test(args: argparse.Namespace) -> int:
    """The whole loop with no model in it. A pass here is OFFLINE_PASS, not a live pass."""
    args.backend = "fixture"
    code = _run(args, live_required=False)
    if code == 0:
        _status("OFFLINE_PASS")
    return code


def cmd_run(args: argparse.Namespace) -> int:
    return _run(args, live_required=True)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    config = load_config(project_root=args.project_root, task=getattr(args, "task", None))
    registry = Registry(config.db_path)
    try:
        runs = registry.list_runs()
        if args.run_id:
            runs = [r for r in runs if r.run_id == args.run_id]
        payload = []
        for run in runs:
            replays = _replays_for_run(registry, run.run_id)
            payload.append({
                "run": run.to_dict(),
                "worlds": [w.to_dict() for w in registry.list_worlds(run.run_id)],
                "policies": [p.to_dict() for p in registry.list_policies(run.run_id)],
                "replays": len(replays),
                "replay_agent_calls": sum(e.discovery_agent_calls for e in replays),
                "replay_evaluator_calls": sum(e.evaluator_calls for e in replays),                "events": [e.to_dict() for e in registry.list_events(run.run_id)],
            })
    finally:
        registry.close()
    print(json.dumps(payload, indent=2, default=str))
    _status("OK", runs=len(payload))
    return 0


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def _mof(name: str) -> Any:
    """Resolve a MOF command lazily so importing the CLI stays cheap."""
    def _run(args: argparse.Namespace) -> int:
        from .mof import cli as mof_cli
        return int(getattr(mof_cli, name)(args))
    return _run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dream-rsi", description=(__doc__ or "dream-rsi CLI").splitlines()[0])
    parser.add_argument("--project-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment facts -> reports/ENV_STATUS.json")
    p.add_argument("--backend", default="fixture", choices=("fixture", "dsh", "pi"))
    p.add_argument("--task", default=None, choices=KNOWN_TASKS)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("bootstrap", help="install DSH/Pi (delegates to the PowerShell script)")
    p.set_defaults(func=cmd_bootstrap)

    for name, fn, helptext in (
        ("offline-test", cmd_offline_test, "full loop with no model; asserts replay is inert"),
        ("run", cmd_run, "run the loop (fixture or a live backend)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--backend", default="fixture", choices=("fixture", "dsh", "pi"))
        p.add_argument("--developer", default="scripted", choices=("scripted", "llm"),
                       help="who writes policy revisions; only 'llm' makes the "
                            "whole loop live (default: scripted, offline)")
        p.add_argument("--provider", default=None)
        p.add_argument("--model", default=None)
        p.add_argument("--outer-rounds", type=int, default=None)
        p.add_argument("--k1", type=int, default=None)
        p.add_argument("--k2", type=int, default=None)
        p.add_argument("--workers", type=int, default=None)
        p.add_argument("--versions", type=int, default=None)
        p.add_argument("--run-id", default=None)
        p.add_argument("--resume", action="store_true")
        p.add_argument(
            "--pi-no-tools", action="store_true",
            help="disable Pi tools, extensions, MCP, skills, and project context",
        )
        p.add_argument(
            "--pi-response-only", action="store_true",
            help="require Pi to return the complete candidate between explicit markers",
        )
        p.add_argument("--task", default=None, choices=KNOWN_TASKS,
                       help="which task under tasks/ to run (default: the bundled knapsack task)")
        p.add_argument(
            "--max-model-calls", type=int, default=None, metavar="N",
            help="hard ceiling on model calls for the whole run: one per discovery "
                 "generation plus one per policy revision. The run stops growing "
                 "worlds when it is reached and still commits. Default: uncapped.",
        )
        p.add_argument(
            "--agent-timeout-s", type=float, default=None, metavar="SECONDS",
            help="per-attempt wall-clock ceiling for a discovery agent. Default: "
                 "900. Raise it when the agent is legitimately slow: a call that "
                 "is cut off writes no candidate, and on the held-out workspace "
                 "the unchanged parent template is then what gets scored.",
        )
        p.set_defaults(func=fn)

    p = sub.add_parser("model-check", help="real API request to DSH or Pi")
    p.add_argument("backend", choices=("dsh", "pi"))
    p.add_argument("--task", default=None, choices=KNOWN_TASKS)
    p.add_argument("--provider", default=None)
    p.add_argument("--model", default=None)
    p.set_defaults(func=cmd_model_check)

    p = sub.add_parser(
        "mof-acceptance",
        help="check the MOF screening implementation; makes no model call",
    )
    p.add_argument("--phase", default="pilot", choices=("pilot", "formal"))
    p.set_defaults(func=_mof("cmd_mof_acceptance"))

    p = sub.add_parser("mof-run", help="run one phase of the MOF screening experiment")
    p.add_argument("--phase", default="pilot", choices=("pilot", "formal"))
    p.add_argument("--only-split-seed", type=int, default=None)
    p.add_argument("--only-repeat", type=int, default=None)
    p.add_argument("--block-limit", type=int, default=None,
                   help="cap the number of paired blocks (each still runs all three arms)")
    p.add_argument("--no-isolation", action="store_true",
                   help="skip the search-phase vault; only for an offline rehearsal")
    p.set_defaults(func=_mof("cmd_mof_run"))

    p = sub.add_parser("mof-unseal", help="open the sealed test pool for frozen candidates")
    p.add_argument("--phase", default="pilot", choices=("pilot", "formal"))
    p.add_argument("--references-only", action="store_true",
                   help="recompute the fixed screening references only; re-grades no candidate")
    p.add_argument("--no-isolation", action="store_true",
                   help="do not take the held-out data off disk while frozen candidates run")
    p.set_defaults(func=_mof("cmd_mof_unseal"))

    p = sub.add_parser("mof-report", help="figures, tables and summary.md for a phase")
    p.add_argument("--phase", default="pilot", choices=("pilot", "formal"))
    p.set_defaults(func=_mof("cmd_mof_report"))

    p = sub.add_parser("report", help="read a finished run back out")
    p.add_argument("--run-id", default=None)
    p.add_argument("--task", default=None, choices=KNOWN_TASKS)
    p.set_defaults(func=cmd_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.project_root is None:
        args.project_root = Path(__file__).resolve().parents[2]
    try:
        return int(args.func(args))
    except DreamRsiError as exc:
        _status("FAILED", error_type=type(exc).__name__, reason=str(exc)[:500])
        return 5
    except KeyboardInterrupt:
        _status("FAILED", reason="interrupted")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
