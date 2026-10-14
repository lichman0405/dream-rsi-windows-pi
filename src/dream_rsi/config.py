"""Runtime configuration: every number the loop obeys, in one place.

Nothing in the algorithm modules reads the environment or invents a default at
call time — the controller is handed one frozen :class:`Config` and the whole
run is reproducible from it (plus the task file and the policy source). The
config digest is persisted on the ``Run`` record, so a resumed run can refuse to
continue under different parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .score.equation1 import RHO_GRID

#: Directory names that must never enter a workspace digest.
DIGEST_EXCLUDES: tuple[str, ...] = (
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".git",
    ".env", ".env.*", "*.key", "*.pem",
)

BACKENDS = ("fixture", "dsh", "pi")

#: Who writes the candidate policy revisions. `scripted` emits bundled
#: constants and is offline; `llm` makes a real model call per revision and is
#: the only setting under which the whole loop is live.
DEVELOPERS = ("scripted", "llm")


@dataclass(frozen=True)
class LoopConfig:
    """The paper's loop parameters.

    ``K1``/``K2``/``W``/``M``/``lambda`` are *configuration*, not paper
    constants: the spec fixes the algorithm's shape and leaves the budget to the
    experimenter. They are recorded on every run so results stay comparable.
    """

    #: Online rounds per world (§3.2 `k < K1`).
    K1: int = 3
    #: Replay rounds per episode (§3.3 `k < K2`).
    K2: int = 6
    #: Worker cap W: batch size ceiling and the parallelism told to policies.
    W: int = 4
    #: Policy versions per selection round, incumbent included (m = 0..M-1).
    M: int = 4
    #: Parallel-penalty weight in Eq(1).
    lambda_weight: float = 0.1
    #: Outer discovery/self-improvement rounds. t=0 grows the first world with
    #: no history; from t=1 on, selection runs against the sealed H_t.
    outer_rounds: int = 3
    #: The rho used to grow the *live* world. Replay sweeps the full grid.
    live_rho: float = 0.5
    rho_grid: tuple[float, ...] = RHO_GRID

    def __post_init__(self) -> None:
        if self.K1 < 1:
            raise ValueError("K1 (online rounds) must be >= 1")
        if self.K2 < 0:
            raise ValueError("K2 (replay rounds) must be >= 0")
        if self.W < 1:
            raise ValueError("W (worker cap) must be >= 1")
        if self.M < 1:
            raise ValueError("M (policy versions) must be >= 1")
        if self.lambda_weight < 0:
            raise ValueError("lambda_weight must be >= 0")
        if self.outer_rounds < 1:
            raise ValueError("outer_rounds must be >= 1")
        if not 0.0 <= self.live_rho <= 1.0:
            raise ValueError("live_rho must be in [0, 1]")
        if tuple(self.rho_grid) != RHO_GRID:
            raise ValueError(
                f"the rho grid is fixed at {RHO_GRID} by §3.3; got {tuple(self.rho_grid)}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "K1": self.K1, "K2": self.K2, "W": self.W, "M": self.M,
            "lambda_weight": self.lambda_weight,
            "outer_rounds": self.outer_rounds,
            "live_rho": self.live_rho,
            "rho_grid": list(self.rho_grid),
        }


@dataclass(frozen=True)
class Timeouts:
    """Wall-clock and memory ceilings. A timeout is a recorded failure, not a retry."""

    agent_s: float = 900.0
    evaluator_s: float = 120.0
    policy_select_s: float = 5.0
    agent_memory_mb: int = 4096
    evaluator_memory_mb: int = 2048
    policy_memory_mb: int = 256

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_s": self.agent_s, "evaluator_s": self.evaluator_s,
            "policy_select_s": self.policy_select_s,
            "agent_memory_mb": self.agent_memory_mb,
            "evaluator_memory_mb": self.evaluator_memory_mb,
            "policy_memory_mb": self.policy_memory_mb,
        }


@dataclass(frozen=True)
class Config:
    """Everything one run needs, resolved to absolute paths."""

    project_root: Path
    task_dir: Path
    incumbent_policy_path: Path
    backend: str = "fixture"
    developer: str = "scripted"
    loop: LoopConfig = field(default_factory=LoopConfig)
    timeouts: Timeouts = field(default_factory=Timeouts)
    provider: str | None = None
    model: str | None = None
    pi_no_tools: bool = False
    pi_response_only: bool = False
    run_id: str | None = None
    resume: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        object.__setattr__(self, "task_dir", Path(self.task_dir).resolve())
        object.__setattr__(
            self, "incumbent_policy_path", Path(self.incumbent_policy_path).resolve()
        )
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.developer not in DEVELOPERS:
            raise ValueError(
                f"developer must be one of {DEVELOPERS}, got {self.developer!r}"
            )

    # ------------------------------------------------------------------ paths

    @property
    def runtime_root(self) -> Path:
        return self.project_root / ".runtime"

    @property
    def state_root(self) -> Path:
        return self.project_root / "state"

    @property
    def db_path(self) -> Path:
        return self.runtime_root / "registry.sqlite3"

    @property
    def cas_root(self) -> Path:
        return self.runtime_root / "cas"

    def run_root(self, run_id: str) -> Path:
        return self.runtime_root / "runs" / run_id

    def workspaces_root(self, run_id: str) -> Path:
        return self.run_root(run_id) / "workspaces"

    def policy_root(self, run_id: str) -> Path:
        return self.run_root(run_id) / "policies"

    def dsh_home_root(self, run_id: str) -> Path:
        """``DSH_HOME`` for this run — deliberately *not* ``~/.dsh``.

        The SDK keeps session state under ``DSH_HOME``. Pointing it at the
        user's real home would let one experiment's state leak into the next
        and would mutate a directory this project does not own, so every run
        gets its own.
        """
        return self.run_root(run_id) / "dsh_home"

    def pi_home_root(self, run_id: str) -> Path:
        """Project-local ``PI_HOME``, for the same reason as ``dsh_home_root``."""
        return self.run_root(run_id) / "pi_home"

    @property
    def seed_workspace(self) -> Path:
        return self.task_dir / "initial"

    @property
    def evaluator_harness(self) -> Path:
        return self.task_dir / "evaluator_harness.py"

    @property
    def task_md(self) -> Path:
        return self.task_dir / "TASK.md"

    @property
    def entry_filename(self) -> str:
        return "candidate.py"

    def ensure_dirs(self, run_id: str) -> None:
        for path in (
            self.runtime_root, self.state_root, self.cas_root,
            self.run_root(run_id), self.workspaces_root(run_id), self.policy_root(run_id),
            self.dsh_home_root(run_id), self.pi_home_root(run_id),
        ):
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "algorithm_revision": "dream-rsi-2609.14858v2-context2",
            # In the digest on purpose: swapping the developer changes what
            # produced every revision, so a resume must not be allowed to
            # continue a run under a different one.
            "developer": self.developer,
            "provider": self.provider,
            "model": self.model,
            "pi_no_tools": self.pi_no_tools,
            "pi_response_only": self.pi_response_only,
            "loop": self.loop.to_dict(),
            "timeouts": self.timeouts.to_dict(),
            "task_dir": self.task_dir.name,
            "entry_filename": self.entry_filename,
        }

    @property
    def cfg_digest(self) -> str:
        """Stable over the *parameters*, not the machine's absolute paths."""
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def default_project_root() -> Path:
    """The repository root, derived from this file's location."""
    return Path(__file__).resolve().parents[2]


# Which environment variables name a backend's provider and model. These are
# the *same* names `tools/model_probe.py` reads, deliberately: the probe is
# supposed to validate the configuration the real adapter will actually use,
# and it cannot do that if the two disagree about where to look.
BACKEND_IDENTITY_ENV: dict[str, tuple[str, str]] = {
    "dsh": ("DSH_PROVIDER", "DSH_MODEL"),
    "pi": ("PI_PROVIDER", "PI_MODEL"),
}


def resolve_identity(
    backend: str, *, provider: str | None = None, model: str | None = None
) -> tuple[str | None, str | None]:
    """Work out which provider and model a backend should use.

    Most specific wins:

    1. an explicit argument (the CLI's ``--provider``/``--model``),
    2. the generic ``DREAM_RSI_PROVIDER``/``DREAM_RSI_MODEL``,
    3. the backend's own variables, e.g. ``PI_PROVIDER``/``PI_MODEL``.

    There is deliberately no fourth step that guesses. A live backend that
    cannot name its model must fail loudly rather than fall back to whatever
    happens to be configured globally — a silent fallback is how a run gets
    attributed to a model that never produced it.
    """
    names = BACKEND_IDENTITY_ENV.get(backend, ())
    for source in (
        (provider, model),
        (os.environ.get("DREAM_RSI_PROVIDER"), os.environ.get("DREAM_RSI_MODEL")),
        (os.environ.get(names[0]) if names else None,
         os.environ.get(names[1]) if names else None),
    ):
        provider = provider or source[0]
        model = model or source[1]
    return provider, model


# ``tasks/<name>/`` directories a run may be pointed at. Named explicitly rather
# than discovered by listing ``tasks/``: discovery would let a stray directory
# become a runnable task by accident, and the task name is part of ``cfg_digest``,
# so it has to mean the same thing on every machine.
KNOWN_TASKS: tuple[str, ...] = ("knapsack", "mof_screening")
DEFAULT_TASK = "knapsack"

#: Tasks whose entry file may use third-party numerical libraries. The knapsack
#: task is deliberately standard-library-only; ``mof_screening`` is not, because
#: its whole point is that a discovery agent improves a *learning* program. The
#: prompt a worker receives has to state the rule that actually applies, which is
#: why this lives here and not inside the shared prompt builder.
THIRD_PARTY_TASKS: frozenset[str] = frozenset({"mof_screening"})


def standard_library_only_for(task_name: str) -> bool:
    return task_name not in THIRD_PARTY_TASKS


def resolve_task(task: str | None = None) -> str:
    """Most specific wins: argument, then ``DREAM_RSI_TASK``, then the default.

    An unknown name raises instead of falling back. A run that silently used a
    different task than the one asked for would produce numbers attributed to the
    wrong task, which is worse than not running.
    """
    name = task or os.environ.get("DREAM_RSI_TASK") or DEFAULT_TASK
    if name not in KNOWN_TASKS:
        raise ValueError(
            f"unknown task {name!r}; known tasks: {', '.join(KNOWN_TASKS)}"
        )
    return name


def load_config(
    *,
    project_root: Path | None = None,
    backend: str = "fixture",
    developer: str = "scripted",
    provider: str | None = None,
    model: str | None = None,
    overrides: Mapping[str, Any] | None = None,
    resume: bool = False,
    run_id: str | None = None,
    task: str | None = None,
    agent_timeout_s: float | None = None,
    pi_no_tools: bool = False,
    pi_response_only: bool = False,
) -> Config:
    """Build a :class:`Config` for one of the tasks in :data:`KNOWN_TASKS`.

    ``task`` selects ``tasks/<name>/``; it defaults to the bundled knapsack task,
    so every existing caller keeps the behaviour it had. The name reaches
    ``cfg_digest`` through ``task_dir.name``, which means a run's checker can tell
    which task produced it and cannot confuse results from different tasks.

    ``overrides`` are applied to :class:`LoopConfig` by field name; an unknown
    key is an error rather than a silent no-op, so a typo cannot quietly change
    what a run means.

    ``agent_timeout_s`` raises the per-attempt wall-clock ceiling. It defaults to
    ``None``, which leaves :class:`Timeouts` untouched, so every existing caller
    keeps its current digest. It is a *ceiling on the agent*, not an acceptance
    threshold: raising it can only let an attempt finish that would otherwise be
    cut off, and a run that leaves it unset is bit-identical to before.
    """
    root = Path(project_root).resolve() if project_root else default_project_root()
    task_name = resolve_task(task)
    loop = LoopConfig()
    if overrides:
        known = set(LoopConfig.__dataclass_fields__)
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValueError(f"unknown LoopConfig override(s): {', '.join(unknown)}")
        loop = replace(loop, **dict(overrides))

    resolved_provider, resolved_model = resolve_identity(
        backend, provider=provider, model=model
    )
    timeouts = Timeouts()
    if agent_timeout_s is not None:
        if agent_timeout_s <= 0:
            raise ValueError("agent_timeout_s must be positive")
        timeouts = replace(timeouts, agent_s=float(agent_timeout_s))
    return Config(
        project_root=root,
        task_dir=root / "tasks" / task_name,
        incumbent_policy_path=root / "policies" / "incumbent.py",
        backend=backend,
        developer=developer,
        loop=loop,
        timeouts=timeouts,
        provider=resolved_provider,
        model=resolved_model,
        pi_no_tools=pi_no_tools,
        pi_response_only=pi_response_only,
        run_id=run_id,
        resume=resume,
    )


__all__ = [
    "BACKENDS", "BACKEND_IDENTITY_ENV", "DEFAULT_TASK", "DEVELOPERS", "DIGEST_EXCLUDES",
    "KNOWN_TASKS", "THIRD_PARTY_TASKS", "Config", "LoopConfig", "Timeouts",
    "default_project_root", "load_config", "resolve_identity", "resolve_task",
    "standard_library_only_for",
]
