"""Live discovery backend backed by the Pi coding agent (JSONL RPC).

This is the project's primary live backend. It is a real backend rather than a
configured one: an installation check is not an integration check, and this
adapter only reports success when a prompt actually reached ``agent_settled``
**and** the workspace artifact changed.

Pi owns its own credentials (``~/.pi/agent/auth.json`` / ``pi /login``). This
adapter never copies another agent's keys: it forwards the ordinary OS
environment plus the user's own ``PI_*`` settings, and strips
Anthropic/DeepSeek credential variables.

The transport lives in :mod:`dream_rsi.adapters._pi_session` because the policy
developer needs the identical round trip. Two copies of "start Pi under a job
object, wait for settle, read the result file" would be two chances to get the
await right, and an await bug is exactly the one that fabricates successes.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from ..domain.errors import BlockedModelConfiguration
from ..runtime.token_budget import TokenBudget
from ._pi_session import run_pi_prompt
from .base import (
    ContinuationContext,
    GeneratedAttempt,
    HistoryEntry,
    TaskSpec,
    digest_file,
    digest_text,
)
from .dsh import build_prompt

#: How much longer the parent waits than the child's own ceiling.
#:
#: The child stops waiting for ``agent_settled`` at ``settle_s`` and then *writes
#: down why*: the event types it saw, the session id, whatever text arrived. The
#: parent waits on the process, and when the two numbers are equal the parent
#: always wins -- it started its clock first -- so it kills the child in the
#: instant before the diagnosis lands and records a bare ``TIMEOUT``. A call that
#: produced nothing then explains nothing, which is the worst combination for a
#: budget the work order says must be reported rather than exceeded.
#:
#: The margin buys the child time to report. It does not extend the agent's
#: deadline: the agent is still cut off at ``settle_s``, and this only decides
#: whether the record of that cut-off survives.
SETTLE_REPORT_MARGIN_S = 120.0


class PiDiscoveryBackend:
    """Real model calls through Pi's RPC mode. ``is_live`` is True."""

    backend_id = "pi"
    is_live = True

    def __init__(
        self,
        *,
        workdir_root: Path,
        provider: str | None = None,
        model: str | None = None,
        timeout_s: float = 900.0,
        python_executable: str | None = None,
        memory_limit_mb: int | None = 4096,
        no_tools: bool = False,
        response_only: bool = False,
        standard_library_only: bool = True,
        token_budget: TokenBudget | None = None,
        max_output_bytes: int = 32768,
        thinking: str | None = None,
    ) -> None:
        self.workdir_root = Path(workdir_root)
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.python_executable = python_executable or sys.executable
        self.memory_limit_mb = memory_limit_mb
        self.no_tools = no_tools
        self.response_only = response_only
        self.standard_library_only = standard_library_only
        self.token_budget = token_budget
        self.max_output_bytes = int(max_output_bytes)
        self.thinking = thinking

    def preflight(self) -> None:
        if not shutil.which("pi"):
            raise BlockedModelConfiguration(
                "`pi` is not on PATH. Install Pi using its upstream instructions "
                "and verify that the `pi` command is available."
            )

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
        started = time.monotonic()
        parent_workspace = Path(parent_workspace).resolve()
        output_workspace = Path(output_workspace).resolve()
        output_workspace.mkdir(parents=True, exist_ok=True)
        if parent_workspace.is_dir() and parent_workspace != output_workspace:
            shutil.copytree(parent_workspace, output_workspace, dirs_exist_ok=True)

        entry_file = output_workspace / task.entry_filename
        before = digest_file(entry_file)
        parent_source = entry_file.read_text(encoding="utf-8") if entry_file.is_file() else ""

        run_dir = (self.workdir_root / invocation_id).resolve()
        result = run_pi_prompt(
            request={
                # The worker is told the ceiling it is actually under, taken
                # from the same attribute that stops the call.
                "prompt": build_prompt(
                    task, history, time_limit_s=self.timeout_s, context=context,
                    response_only=self.response_only,
                    parent_source=parent_source if self.response_only else None,
                    standard_library_only=self.standard_library_only,
                ),
                "cwd": str(output_workspace),
                "provider": self.provider,
                "model": self.model,
                # The worker waits for `agent_settled` on its own clock, and that
                # wait is what actually bounds a call. It used to be hard-coded at
                # 900s inside the worker, so a caller who raised `timeout_s` got a
                # longer subprocess wait and the same 900s cut-off -- a defect that
                # only showed up as calls producing no candidate. Passing the
                # ceiling through makes the two the same number.
                "settle_s": self.timeout_s,
                "no_tools": self.no_tools,
                "thinking": self.thinking,
            },
            run_dir=run_dir,
            # Longer than `settle_s` by a margin, so the child outlives its own
            # deadline far enough to write down what it saw. See
            # SETTLE_REPORT_MARGIN_S.
            timeout_s=self.timeout_s + SETTLE_REPORT_MARGIN_S,
            python_executable=self.python_executable,
            memory_limit_mb=self.memory_limit_mb,
            token_budget=self.token_budget,
            max_output_bytes=self.max_output_bytes,
        )

        duration = time.monotonic() - started
        error_type = result.error_type
        if error_type is None and self.response_only:
            source = _extract_candidate(result.final_text)
            if source is None:
                error_type = "MISSING_CANDIDATE_ARTIFACT"
            else:
                entry_file.write_text(source, encoding="utf-8", newline="\n")
        after = digest_file(entry_file)
        # A settled agent that touched nothing has not done the job. Without
        # this the attempt would be recorded as a success with an unchanged
        # artifact, which is the most flattering possible reading of "no work".
        if error_type is None and before == after:
            error_type = "NO_ARTIFACT_CHANGE"

        return GeneratedAttempt(
            invocation_id=invocation_id,
            backend_id=self.backend_id,
            model=self.model or "pi-default",
            workspace_path=output_workspace,
            entry_file_digest=after,
            succeeded=error_type is None,
            duration_s=duration,
            error_type=error_type,
            summary=result.final_text.strip()[:500],
            session_id=result.session_id,
            raw_response_digest=digest_text(result.final_text) or None,
            generation_record={"ledger_seq": result.token_seq,
                               "token_charged": result.token_charged,
                               "request": json.loads((run_dir / "pi_request.json").read_text(encoding="utf-8"))
                               if (run_dir / "pi_request.json").exists() else {},
                               "final_response": result.final_text,
                               "token_usage": result.token_usage,
                               "events": [json.loads(line) for line in
                                          (run_dir / "pi_request.json.history.jsonl").read_text(encoding="utf-8").splitlines()]
                               if (run_dir / "pi_request.json.history.jsonl").exists() else [],
                               "error_type": result.error_type},
            extra={
                "is_live": True,
                "event_types": list(result.event_types),
                "entry_changed": before != after,
                "stderr_tail": result.stderr_tail,
                "worker_error": result.error,
                "token_usage": result.token_usage,
                "token_charged": result.token_charged,
                "token_seq": result.token_seq,
            },
        )


_CANDIDATE_BLOCK = re.compile(
    r"=== CANDIDATE BEGIN ===\s*\r?\n(.*?)\r?\n=== CANDIDATE END ===",
    flags=re.DOTALL,
)


def _extract_candidate(response: str) -> str | None:
    """Accept exactly one explicit, bounded candidate artifact from text mode."""
    matches = list(_CANDIDATE_BLOCK.finditer(response))
    if len(matches) != 1:
        return None
    source = matches[0].group(1)
    return source if source.strip() else None


__all__ = ["PiDiscoveryBackend", "_extract_candidate"]
