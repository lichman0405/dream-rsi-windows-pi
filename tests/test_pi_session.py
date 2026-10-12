"""Transport contract for the shared Pi RPC round trip.

No model is called here: ``subprocess.Popen`` is replaced with a stub, so these
tests pin the *parent-side* contract — which paths are handed to the child and
which environment it is given — and say nothing about any provider.

The regression these exist for is quiet and was found by running it, not by
reading it. The child runs with ``cwd`` set to its own scratch directory, so a
relative request path handed to it resolves against the *child's* directory.
The child then finds no request, writes its error payload to the same wrong
place, and still exits 0 — so the parent sees only "no result file" and the real
reason is lost. Resolving the path in the parent is what prevents it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dream_rsi.adapters import _pi_session
from dream_rsi.adapters._pi_session import pi_child_env, run_pi_prompt


class _RecordingProc:
    """Stands in for the worker process. Records how it was launched."""

    last: _RecordingProc | None = None

    def __init__(self, argv: list[str], **kwargs: Any) -> None:
        self.argv = argv
        self.cwd = kwargs.get("cwd")
        self.env = kwargs.get("env") or {}
        self.returncode = 0
        _RecordingProc.last = self

    def communicate(self, timeout: float | None = None) -> tuple[bytes, bytes]:
        return b"", b""


class _NullJob:
    """A job object that contains nothing, so no child is needed."""

    def __init__(self, **_: Any) -> None:
        pass

    def __enter__(self) -> _NullJob:
        return self

    def __exit__(self, *_: Any) -> bool:
        return False

    def assign(self, proc: Any) -> None:
        pass


@pytest.fixture()
def stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_pi_session, "WindowsJob", _NullJob)
    monkeypatch.setattr(_pi_session.subprocess, "Popen", _RecordingProc)


def test_the_request_path_handed_to_the_child_is_absolute(stubbed: None, tmp_path: Path) -> None:
    """A relative path here is the bug this file exists for: the child resolves
    it against its own cwd, never finds the request, and reports nothing."""
    run_pi_prompt(
        request={"cwd": str(tmp_path), "prompt": "hi"},
        run_dir=Path("relative/scratch"),  # deliberately relative
        timeout_s=5.0,
    )
    proc = _RecordingProc.last
    assert proc is not None
    handed = Path(proc.argv[-1])
    assert handed.is_absolute(), f"child was handed a relative path: {handed}"
    assert handed.is_file(), "the child must be pointed at the file that exists"
    assert Path(proc.cwd).is_absolute()


def test_the_child_never_inherits_another_agents_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is what keeps a Pi result from being Claude Code's result wearing a
    Pi label — the failure that would make every live claim worthless."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-ours")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-claude-not-ours")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-not-ours")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-not-ours")
    monkeypatch.setenv("PI_PROVIDER", "cc-switch-deep-seek")
    monkeypatch.setenv("PI_MODEL", "deepseek-v4-pro")

    env = pi_child_env()

    for leaked in ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "DEEPSEEK_API_KEY",
                   "OPENAI_API_KEY"):
        assert leaked not in env, f"{leaked} reached the Pi child"
    # The user's own Pi settings must survive, or Pi cannot name its model.
    assert env["PI_PROVIDER"] == "cc-switch-deep-seek"
    assert env["PI_MODEL"] == "deepseek-v4-pro"


def test_a_missing_result_file_names_the_exit_code(stubbed: None, tmp_path: Path) -> None:
    """The stub never writes a result, which is exactly the shape of a worker
    that died before reporting. The failure must name the exit code rather than
    be reported as a generic agent error."""
    result = run_pi_prompt(
        request={"cwd": str(tmp_path), "prompt": "hi"},
        run_dir=tmp_path / "scratch",
        timeout_s=5.0,
    )
    assert result.failed
    assert result.error_type == "WORKER_NO_RESULT_EXIT_0"
    assert result.final_text == ""


def test_the_request_file_is_the_only_thing_carrying_the_settle_ceiling(
    stubbed: None, tmp_path: Path,
) -> None:
    """The agent's deadline must reach the worker, or it does not exist.

    The worker waits for ``agent_settled`` on its own clock, and that wait -- not
    the parent's subprocess timeout -- is what actually cuts a call off. It was
    hard-coded at 900s inside the child, so raising ``--agent-timeout-s`` bought
    a longer *wait* and the same 900s *ceiling*; the only visible symptom was a
    call that produced no candidate. The ceiling now travels in the request, and
    this pins that it travels.
    """
    import json

    run_pi_prompt(
        request={"cwd": str(tmp_path), "prompt": "hi", "settle_s": 1800.0},
        run_dir=tmp_path / "scratch",
        timeout_s=5.0,
    )
    proc = _RecordingProc.last
    assert proc is not None
    payload = json.loads(Path(proc.argv[-1]).read_text(encoding="utf-8"))
    assert payload["settle_s"] == 1800.0


def test_the_parent_waits_longer_than_the_ceiling_it_hands_the_child(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A diagnosis cannot survive a kill that lands in the same instant.

    The child stops waiting at ``settle_s`` and then writes down what it saw --
    event types, session id, any text. The parent waits on the process. When both
    numbers are equal the parent wins, because it started its clock first: it
    kills the child just before the payload lands and records a bare ``TIMEOUT``,
    so the call that produced nothing is also the call that explains nothing.
    Raising the ceiling alone does not fix that, and it is why the first MOF
    timeout arrived with no ``event_types`` at all.

    This pins the order rather than the constant: whatever the margin becomes,
    the outer wait has to stay strictly longer.
    """
    from dream_rsi.adapters import pi as pi_module
    from dream_rsi.adapters.base import TaskSpec
    from dream_rsi.adapters.pi import PiDiscoveryBackend

    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _Result:
            ok = False
            error_type = "TIMEOUT"
            final_text = ""
            session_id = None
            event_types: tuple[str, ...] = ()
            stderr_tail = ""
            error = ""
            token_usage: dict[str, int] = {}
            token_charged: int | None = None
            token_seq: int | None = None

        return _Result()

    monkeypatch.setattr(pi_module, "run_pi_prompt", _capture)

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.py").write_text("def propose(pool, train):\n    return pool[0]\n",
                                       encoding="utf-8")
    task = TaskSpec(
        task_id="t", entry_filename="candidate.py", instructions="i", seed_workspace=seed,
    )
    backend = PiDiscoveryBackend(workdir_root=tmp_path / "run", timeout_s=1800.0)
    backend.generate(
        task=task,
        parent_workspace=seed,
        history=(),
        output_workspace=tmp_path / "out",
        invocation_id="call-1",
    )

    settle = captured["request"]["settle_s"]
    outer = captured["timeout_s"]
    assert settle == 1800.0, "the agent's own ceiling must be the configured one"
    assert outer > settle, (
        f"the parent gives up at {outer}s and the child reports at {settle}s, so the "
        "report is always lost"
    )


def test_the_prompt_announces_the_ceiling_that_will_actually_stop_the_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A worker on a clock it was not told about cannot plan against it.

    Two MOF calls were stopped mid-computation after writing pure-Python
    numerical sweeps, and nothing in either prompt said a limit existed. A slow
    call and a dead one look identical from inside the shell -- a long tool call
    prints nothing while it runs -- so the only available defence is to know the
    number and size the work against it.

    This pins the *wiring*, not the wording: the number in the prompt has to be
    the same attribute that stops the call, so raising ``--agent-timeout-s``
    cannot leave the worker planning against the old one.
    """
    from dream_rsi.adapters import pi as pi_module
    from dream_rsi.adapters.base import TaskSpec
    from dream_rsi.adapters.pi import PiDiscoveryBackend

    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> Any:
        captured.update(kwargs)

        class _Result:
            ok = False
            error_type = "TIMEOUT"
            final_text = ""
            session_id = None
            event_types: tuple[str, ...] = ()
            stderr_tail = ""
            error = ""
            token_usage: dict[str, int] = {}
            token_charged: int | None = None
            token_seq: int | None = None

        return _Result()

    monkeypatch.setattr(pi_module, "run_pi_prompt", _capture)

    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "candidate.py").write_text("def propose(pool, train):\n    return pool[0]\n",
                                       encoding="utf-8")
    task = TaskSpec(
        task_id="t", entry_filename="candidate.py", instructions="i", seed_workspace=seed,
    )
    backend = PiDiscoveryBackend(workdir_root=tmp_path / "run", timeout_s=1234.0)
    backend.generate(
        task=task,
        parent_workspace=seed,
        history=(),
        output_workspace=tmp_path / "out",
        invocation_id="call-1",
    )

    settle = captured["request"]["settle_s"]
    assert settle == 1234.0, "the ceiling that stops the call is the configured one"
    assert f"{settle:.0f} seconds" in captured["request"]["prompt"], (
        "the prompt names a limit that is not the one that will stop the call"
    )


def test_the_standard_library_rule_is_scoped_to_the_graded_file() -> None:
    """The rule is a fact about the grader's interpreter, and reads as a ban on
    the whole environment when it is not scoped.

    The evaluator runs the entry file under the project virtualenv, which has no
    numpy; the shell the worker explores in is a different interpreter that does.
    Stated unscoped -- "standard library only; do not add third-party
    dependencies" -- a worker applies it to its own scratch scripts and writes
    pure-Python numerics for work one numpy call would finish, which is how two
    real calls were lost to the clock. The scope, not the rule, is the fix.
    """
    from dream_rsi.adapters.base import TaskSpec
    from dream_rsi.adapters.dsh import build_prompt

    task = TaskSpec(
        task_id="t", entry_filename="candidate.py", instructions="i",
        seed_workspace=Path("."),
    )
    prompt = build_prompt(task, ())

    assert "`candidate.py`" in prompt
    assert "never graded" in prompt, "scratch scripts must be explicitly out of scope"
    assert "- Standard library only;" not in prompt, (
        "the unscoped form is the defect: it reads as a ban on the whole shell"
    )


def test_a_prompt_with_no_ceiling_does_not_invent_one() -> None:
    """The other direction: an unannounced limit must not be stated as a fact."""
    from dream_rsi.adapters.base import TaskSpec
    from dream_rsi.adapters.dsh import build_prompt

    task = TaskSpec(
        task_id="t", entry_filename="candidate.py", instructions="i",
        seed_workspace=Path("."),
    )
    assert "stopped after" not in build_prompt(task, ())


def _trace_events(path: Path) -> list[dict[str, Any]]:
    """Read a trace back as its event payloads, with the `trace_s` wrapper off."""
    import json

    return [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_a_hanging_call_leaves_a_parseable_trace_behind(tmp_path: Path) -> None:
    """The evidence has to be on disk *before* the kill, not written at the end.

    A worker that reports once, at the end, cannot explain a call that never
    reaches the end: the timeout kills it first and the only surviving fact is
    "no candidate", which is the same fact for a stuck provider, an agent
    looping on its own tool output, and a model that is merely slow. The trace
    is flushed per record so that killing the process cannot take it away.
    """
    from dream_rsi.adapters._pi_worker import EventTrace

    path = tmp_path / "nested" / "pi_request.json.events.jsonl"
    trace = EventTrace(path).open()
    trace.write({"type": "message_start"})
    trace.write({"type": "tool_call", "name": "read", "arguments": {"path": "pool.json"}})
    # Read it back while the writer is still open: this is exactly what an
    # observer does to a call that has not finished.
    import json

    live = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"]["type"] for row in live] == ["message_start", "tool_call"]
    assert all(isinstance(row["trace_s"], float) for row in live), (
        "a trace without times cannot say how long anything took"
    )
    trace.close()


def test_a_call_that_only_generates_text_still_shows_that_it_generated(
    tmp_path: Path,
) -> None:
    """Deltas are counted, not stored -- a call emits thousands and each says a
    few characters, so storing them would crowd out the tool calls that are the
    actual evidence. Counting them is what keeps 'the model was writing for
    forty minutes' distinguishable from 'nothing happened for forty minutes'."""
    from dream_rsi.adapters._pi_worker import TRACE_STREAM_SUMMARY_EVERY, EventTrace

    path = tmp_path / "events.jsonl"
    trace = EventTrace(path).open()
    for _ in range(TRACE_STREAM_SUMMARY_EVERY * 2):
        trace.write({
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "ab"},
        })
    trace.write({"type": "tool_execution_start", "toolName": "bash"})
    trace.close()

    events = _trace_events(path)
    assert events[-1]["type"] == "tool_execution_start", "the tool call must survive"
    summaries = [e for e in events if e["type"] == "trace_stream_summary"]
    # Two full bursts plus the tail flushed at close.
    assert [s["deltas"] for s in summaries] == [
        TRACE_STREAM_SUMMARY_EVERY, TRACE_STREAM_SUMMARY_EVERY,
    ]
    assert [s["chars"] for s in summaries] == [TRACE_STREAM_SUMMARY_EVERY * 2] * 2
    assert all(s["kinds"] == {"text_delta": TRACE_STREAM_SUMMARY_EVERY} for s in summaries)


def test_an_oversized_event_stays_valid_jsonl(tmp_path: Path) -> None:
    """A single tool result can be megabytes. Truncating the *string* would
    leave a line that no JSONL reader can parse, so the file would be
    unreadable exactly when the big tool result is the thing under
    investigation."""
    from dream_rsi.adapters._pi_worker import MAX_TRACE_RECORD_CHARS, EventTrace

    path = tmp_path / "events.jsonl"
    trace = EventTrace(path).open()
    trace.write({"type": "tool_result", "output": "x" * (MAX_TRACE_RECORD_CHARS * 4)})
    trace.close()

    record = _trace_events(path)[0]
    assert record["type"] == "tool_result", "the truncated record still names its kind"
    assert record["trace_truncated_record_chars"] > MAX_TRACE_RECORD_CHARS
    assert record["head"].startswith('{"type": "tool_result"')
    assert "x" * 100 in record["tail"]


def test_a_runaway_agent_cannot_fill_the_disk(tmp_path: Path) -> None:
    """The trace is written by the process that is misbehaving, so its size
    has to be bounded by the observer rather than trusted to the observed."""
    import json

    from dream_rsi.adapters._pi_worker import MAX_TRACE_RECORDS, EventTrace

    path = tmp_path / "events.jsonl"
    trace = EventTrace(path).open()
    for i in range(MAX_TRACE_RECORDS + 50):
        trace.write({"type": "tool_call", "n": i})
    trace.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    # The cap, plus the one line that says the cap was reached.
    assert len(lines) == MAX_TRACE_RECORDS + 1
    assert json.loads(lines[-1])["event"] == {
        "type": "trace_truncated_after", "records": MAX_TRACE_RECORDS,
    }


def test_the_worker_still_defaults_to_nine_hundred_seconds() -> None:
    """An old request file must behave exactly as it did.

    The default is not a preference: it is what keeps every archived request
    reproducible, since a request written before the field existed carried no
    ceiling and was bounded at 900 by the literal in the child.
    """
    from dream_rsi.adapters import _pi_worker

    assert _pi_worker.SETTLE_DEFAULT_S == 900.0
    source = (_pi_worker.__file__ and Path(_pi_worker.__file__).read_text(encoding="utf-8")) or ""
    assert 'float(request.get("settle_s") or SETTLE_DEFAULT_S)' in source, (
        "the worker no longer reads the ceiling out of the request"
    )
