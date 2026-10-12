"""``LlmPolicyDeveloper`` — real model calls that rewrite the policy (§3.3).

The scripted developer next door emits fixed constants. This one asks a live
model to rewrite the incumbent policy given *measured* feedback, and it is the
only component in the project that closes the self-improvement loop with a
model on both sides (discovery **and** policy development).

Three things make this honest rather than decorative:

1. **It never validates its own output.** The controller screens every file
   this returns through the same :func:`validate_policy_source` gate and the
   same isolated load-and-call probe that the scripted variants face. This
   module's own pre-screen exists only to avoid burning a live call on a
   rejection it can already foresee — it is a retry trigger, not an authority.
2. **A failed call is not a proposal.** If the model errors, times out, or
   returns something with no extractable policy, no file is written and no
   revision is claimed. If *every* call fails the developer raises, so the
   controller records ``DEVELOPER_ERROR`` with the real reason instead of
   silently selecting the incumbent.
3. **Each revision carries its own provenance** — model, provider, session id
   and attempt number — in its header, so a policy on disk can always be traced
   back to the call that produced it.

One call per revision, not one call for all revisions: the rounds want
*candidates at different points of the ``(n, a)`` plane*, and asking a single
call for three near-identical rewrites reliably produces three near-identical
rewrites, which makes the selection step look busy while measuring nothing.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ..adapters._pi_session import PiSessionResult, run_pi_prompt
from .policy_development import Proposal, StaticCheck, validate_policy_source
from .token_budget import ROLE_POLICY_DEVELOPMENT, TokenBudget

#: Delimiters the model is told to use. Plain sentinels rather than a Markdown
#: fence because a model that also writes prose about its change will happily
#: emit two or three fences, and guessing which one is the policy is how a
#: truncated or explanatory fragment ends up deployed as a policy.
BEGIN = "=== POLICY BEGIN ==="
END = "=== POLICY END ==="

#: Second attempt is a repair, not a retry: it is only spent when the first
#: candidate was produced and then *rejected*, and its prompt carries the exact
#: rejection reason. A transport failure is never retried here.
MAX_ATTEMPTS_PER_REVISION = 2


@dataclass(frozen=True)
class ExplorationRole:
    """The exploration stance one revision is asked to take."""

    key: str
    instruction: str


#: Deliberately mirrored on ``ScriptedVariantDeveloper``'s three modes. The
#: point of the loop is to spread candidates across the (n, a) plane; three
#: revisions that all "improve the policy" converge on the same corner and the
#: argmax over them is noise. So each revision is given a different stance.
ROLES: tuple[ExplorationRole, ...] = (
    ExplorationRole(
        "exploit",
        "EXPLOIT. Push toward the highest-scoring frontier. Expand the node "
        "that looks most promising and keep the batch narrow, so the episode "
        "reaches a high score in few rounds. Accept fewer discoveries.",
    ),
    ExplorationRole(
        "explore",
        "EXPLORE. Push toward coverage. Keep the root in the batch often so "
        "sibling subtrees stay reachable, and prefer breadth over depth. "
        "Accept a lower peak score in exchange for a wider tree.",
    ),
    ExplorationRole(
        "balance",
        "BALANCE. Trade the two off explicitly as a function of rho: at low "
        "rho exploit, at high rho explore. Make the transition a smooth "
        "function of rho rather than a hard threshold, so the rho sweep "
        "produces a genuinely different curve from either of the other two.",
    ),
)


def build_developer_prompt(
    *,
    incumbent_source: str,
    incumbent_digest: str,
    feedback: Sequence[str],
    role: ExplorationRole,
    iteration_t: int,
    repair_reason: str | None = None,
) -> str:
    """The policy-development prompt: contract + incumbent + measured feedback."""
    parts = [
        "You are the policy-development step of an automated search loop.",
        "You do not run anything. You rewrite one Python file, and an external",
        "harness measures it. Your opinion of your own work is not evidence.",
        "",
        "## What the file must contain",
        "A module-level function with exactly this signature:",
        "",
        "    def select(observation: Any, rho: float) -> list[str]:",
        "",
        "It returns a subset of `observation.legal_actions` — node id strings —",
        "with no duplicates and at most `observation.max_parallelism` entries.",
        "Returning `[]` ends the episode. It must be deterministic: the same",
        "`(observation, rho)` always returns the same list in the same order.",
        "",
        "The observation exposes:",
        "- `legal_actions`: tuple of node ids that may be expanded this round.",
        "- `max_parallelism`: the hard ceiling on the returned list length.",
        "- `nodes`: objects with `.node_id`, `.score` (float), `.depth` (int),",
        "  `.creation_index` (int, monotone creation order).",
        "  `.public_observation` is a dict of measured status, public evaluator",
        "  diagnostics and generation summary; use it without importing json.",
        "- `completed_rounds`: the number of completed nonempty batches.",
        "- `node_ids`: the tree's node ids in creation order; the first is the root.",
        "",
        "## Hard rules — a file that breaks any of these is rejected unread",
        "- You may import ONLY these modules, and nothing else:",
        "  `__future__`, `typing`, `dataclasses`, `math`, `itertools`,",
        "  `functools`, `collections`, `abc`, `enum`. Any other import is",
        "  rejected, including ordinary ones like `json` or `re`. `from",
        "  __future__ import annotations` and `from typing import Any` are",
        "  expected. Import nothing at all if you do not need it.",
        "- MUST NOT call: open, exec, eval, compile, __import__, input,",
        "  breakpoint, globals, locals, vars, memoryview, getattr, setattr,",
        "  delattr, type, dir, super, object.",
        "- MUST NOT write `__` anywhere in a string. A double underscore in a",
        "  string is only ever a way to name a Python internal, so the gate",
        "  rejects it. This does NOT apply to your docstring or comments, where",
        "  you may discuss anything freely.",
        "- `select` must not be decorated, and must not be nested in a class.",
        "- Line endings must be LF, not CRLF.",
        "- No network, no filesystem, no clock, no randomness. The policy is a",
        "  pure function of the observation it is handed.",
        "",
        f"## Your assigned stance for this revision: {role.key.upper()}",
        role.instruction,
        "",
        f"## Current incumbent (sha256 {incumbent_digest[:16]}…)",
        "This is the policy your revision replaces. Read it for the exact API",
        "in use. Do not copy it unchanged — an unchanged file is a wasted round.",
        "",
        "```python",
        incumbent_source.rstrip(),
        "```",
        "",
        "## Measured feedback on the current policy",
        "These are real measurements from the frozen history, not opinions.",
        *([f"- {line}" for line in feedback] if feedback else ["- (no history yet)"]),
        "",
        "`B` is the number of non-root nodes in the sealed world; `S` is its",
        "best recorded score. `n` is the number of nodes this policy revealed",
        "in one replay. Compare policies by their measured Eq(1) `V` on the",
        "same worlds; a higher historical `V` does not prove better future",
        "online discovery.",
        "",
        f"This is outer round t={iteration_t}.",
    ]
    if repair_reason:
        parts += [
            "",
            "## Your previous attempt was REJECTED by the static gate",
            f"Reason: {repair_reason}",
            "Fix exactly that problem and return the corrected file.",
        ]
    parts += [
        "",
        "## Output format — follow exactly",
        "Reply with the complete file and nothing else, wrapped like this:",
        "",
        BEGIN,
        "<the entire Python file>",
        END,
        "",
        "No commentary before or after. No Markdown fences. The block is written",
        "to disk verbatim and screened, so anything you put inside it must be",
        "valid, self-contained Python.",
    ]
    return "\n".join(parts)


def extract_policy(text: str) -> tuple[str | None, str | None]:
    """Pull the sentinel-delimited policy out of a model response.

    Returns ``(source, problem)``; exactly one is non-``None``. A response that
    lost its beginning — the transport truncates long replies — is reported as
    such rather than silently returned from its first ``def``, because a policy
    missing its header is a different program from the one the model wrote.
    """
    if not text.strip():
        return None, "model returned no text"

    start = text.find(BEGIN)
    if start < 0:
        return None, f"response contained no {BEGIN} marker"
    end = text.find(END, start + len(BEGIN))
    if end < 0:
        return None, (
            f"response had {BEGIN} but no {END}; it was most likely truncated "
            "by the transport, so the recovered text would be a fragment"
        )

    body = text[start + len(BEGIN) : end]
    # CRLF is the single most likely rejection that is purely mechanical, and
    # normalizing it does not weaken the gate: the bytes we write are the bytes
    # that get screened.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    lines = body.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    body = "\n".join(lines)
    if not body.strip():
        return None, "the delimited block was empty"

    # Tolerate one wrapping fence even though the prompt forbids it — but only
    # a wrapper, never a fence somewhere in the middle.
    if body.lstrip().startswith("```"):
        stripped = body.lstrip()
        first_newline = stripped.find("\n")
        if first_newline > 0 and stripped.rstrip().endswith("```"):
            body = stripped[first_newline + 1 : stripped.rstrip().rfind("```")].strip("\n")
        else:
            return None, "the block was wrapped in an unterminated Markdown fence"
    return body + "\n", None


class LlmPolicyDeveloper:
    """Live: one real Pi model call per candidate revision.

    Satisfies :class:`~dream_rsi.runtime.policy_development.PolicyDeveloper`.
    """

    developer_id = "llm-pi"
    is_live = True

    def __init__(
        self,
        *,
        workdir_root: Path,
        provider: str,
        model: str,
        timeout_s: float = 900.0,
        python_executable: str | None = None,
        memory_limit_mb: int | None = 4096,
        token_budget: TokenBudget | None = None,
        max_output_bytes: int = 32768,
        thinking: str | None = None,
    ) -> None:
        if not provider or not model:
            raise ValueError(
                "LlmPolicyDeveloper needs an explicit provider and model: a "
                "developer that cannot name the model it used cannot be "
                "distinguished from one that used no model at all"
            )
        self.workdir_root = Path(workdir_root).resolve()
        self.provider = provider
        self.model = model
        self.timeout_s = float(timeout_s)
        self.python_executable = python_executable
        self.memory_limit_mb = memory_limit_mb
        self.token_budget = token_budget
        self.max_output_bytes = int(max_output_bytes)
        self.thinking = thinking
        #: One entry per call, kept so a run can report what it actually spent.
        self.sessions: list[dict[str, object]] = []

    # ------------------------------------------------------------------ call

    def _session_dir(self, iteration_t: int, m: int, attempt: int) -> Path:
        """An empty directory Pi runs in. Never the repository.

        The incumbent source is pasted into the prompt, so the model has
        everything it needs without filesystem access; giving it a cwd inside
        the project would let an agentic detour read and rewrite real project
        files, which is not what this step is for.
        """
        return self.workdir_root / f"t{iteration_t}_m{m}_a{attempt}"

    def _call(
        self, *, prompt: str, iteration_t: int, m: int, attempt: int
    ) -> PiSessionResult:
        session_dir = self._session_dir(iteration_t, m, attempt)
        result = run_pi_prompt(
            request={
                "cwd": str(session_dir),
                "provider": self.provider,
                "model": self.model,
                "prompt": prompt,
                "no_tools": True,
                "thinking": self.thinking,
            },
            run_dir=session_dir,
            timeout_s=self.timeout_s,
            python_executable=self.python_executable,
            memory_limit_mb=self.memory_limit_mb,
            token_budget=self.token_budget,
            role=ROLE_POLICY_DEVELOPMENT,
            max_output_bytes=self.max_output_bytes,
        )
        self.sessions.append({
            "iteration_t": iteration_t, "m": m, "attempt": attempt,
            "ok": result.ok, "error_type": result.error_type,
            "session_id": result.session_id, "duration_s": round(result.duration_s, 3),
            "event_types": list(result.event_types[-12:]),
            "token_usage": result.token_usage,
            "token_charged": result.token_charged,
        })
        return result

    # ------------------------------------------------------------- interface

    def propose(
        self,
        *,
        incumbent_path: Path,
        feedback: Sequence[str],
        output_dir: Path,
        count: int,
        iteration_t: int,
        start_m: int = 1,
    ) -> Sequence[Proposal]:
        incumbent_bytes = Path(incumbent_path).read_bytes()
        incumbent_digest = hashlib.sha256(incumbent_bytes).hexdigest()
        incumbent_source = incumbent_bytes.decode("utf-8")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        proposals: list[Proposal] = []
        failures: list[str] = []

        for m in range(start_m, start_m + max(0, count)):
            role = ROLES[(m - 1) % len(ROLES)]
            proposal, failure = self._propose_one(
                m=m, role=role, iteration_t=iteration_t,
                incumbent_source=incumbent_source,
                incumbent_digest=incumbent_digest,
                feedback=feedback, output_dir=output_dir,
            )
            if proposal is not None:
                proposals.append(proposal)
            else:
                failures.append(f"m={m} ({role.key}): {failure}")

        if not proposals and failures:
            # Every call failed: this is a provider/configuration problem, not a
            # weak model. Raising routes it to the controller's DEVELOPER_ERROR
            # path, which records the reason instead of quietly re-selecting the
            # incumbent and reporting a successful round.
            raise RuntimeError(
                "every policy-development call failed: " + "; ".join(failures)
            )
        return proposals

    def _propose_one(
        self,
        *,
        m: int,
        role: ExplorationRole,
        iteration_t: int,
        incumbent_source: str,
        incumbent_digest: str,
        feedback: Sequence[str],
        output_dir: Path,
    ) -> tuple[Proposal | None, str | None]:
        """Produce one revision, with at most one static-gate repair round."""
        repair_reason: str | None = None
        last_problem = "no attempt was made"

        for attempt in range(1, MAX_ATTEMPTS_PER_REVISION + 1):
            prompt = build_developer_prompt(
                incumbent_source=incumbent_source,
                incumbent_digest=incumbent_digest,
                feedback=feedback,
                role=role,
                iteration_t=iteration_t,
                repair_reason=repair_reason,
            )
            result = self._call(
                prompt=prompt, iteration_t=iteration_t, m=m, attempt=attempt
            )
            if result.failed:
                return None, f"call failed ({result.error_type}): {result.error}"

            source, problem = extract_policy(result.final_text)
            if source is None:
                last_problem = f"unusable response: {problem}"
                # A response with no policy in it is not something a "please
                # fix X" follow-up reliably repairs, so it is not retried.
                return None, last_problem

            path = output_dir / f"llm_t{iteration_t}_m{m}_{role.key}.py"
            header = (
                f"# revision m={m} at iteration t={iteration_t}; "
                f"incumbent sha256={incumbent_digest}\n"
                f"# developer={self.developer_id} provider={self.provider} "
                f"model={self.model} role={role.key} attempt={attempt}\n"
            )
            path.write_text(header + source, encoding="utf-8", newline="\n")

            check: StaticCheck = validate_policy_source(path)
            if check.ok:
                return Proposal(
                    m=m, path=path, developer=self.developer_id, is_live=True,
                    summary=f"live revision role={role.key} model={self.model} "
                            f"attempt={attempt} session={result.session_id or 'unknown'}",
                ), None
            last_problem = f"static gate rejected it: {check.reason}"
            repair_reason = check.reason

        return None, last_problem


__all__ = [
    "BEGIN",
    "END",
    "MAX_ATTEMPTS_PER_REVISION",
    "ROLES",
    "ExplorationRole",
    "LlmPolicyDeveloper",
    "build_developer_prompt",
    "extract_policy",
]
