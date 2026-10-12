#!/usr/bin/env python3
"""Child process that drives ONE Pi RPC prompt and reports JSON.

Pi speaks strict UTF-8 + LF **JSONL** over stdin/stdout when started with
``pi --mode rpc --no-session``. The critical detail, straight from the Pi RPC
docs: a successful ``prompt`` **ack is not completion**. The response merely
acknowledges that the command was accepted. Completion is the later
``agent_settled`` event. Treating the ack as "done" is the classic way to
record a fake success, so this worker waits for ``agent_settled`` and fails on
EOF, timeout, ``agent_error``, or an aborted stop reason.

Invoked as::

    python -m dream_rsi.adapters._pi_worker <request.json>

Result JSON is written next to the request file as ``<request>.result.json``.
"""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

#: Ceiling on the assistant text carried back in the result JSON.
#:
#: This used to be 8000, which is smaller than a verbose policy file plus the
#: prose a model writes around it. The truncation keeps the *tail*, so a long
#: reply loses its opening — and for a policy reply the opening is where the
#: ``=== POLICY BEGIN ===`` marker lives. The developer then rejects the reply
#: as marker-less, which is a wasted call rather than a wrong result, but it is
#: still a real call spent to learn nothing. 64 KiB is far above any plausible
#: reply and still bounded.
MAX_FINAL_RESPONSE_CHARS = 65536

#: How long to wait for ``agent_settled`` when the request does not say.
#:
#: The caller threads its own agent deadline through ``settle_s``. This default
#: exists so that an older request file -- one written before the field existed --
#: behaves exactly as it did: bounded at 900 seconds.
SETTLE_DEFAULT_S = 900.0

#: Bounds on the per-call event trace. See :class:`EventTrace`.
MAX_TRACE_RECORDS = 20000
MAX_TRACE_RECORD_CHARS = 3000

#: How many coalesced streaming deltas between summary lines.
TRACE_STREAM_SUMMARY_EVERY = 250


class EventTrace:
    """Append-only record of everything Pi said, written while it is saying it.

    A call that hangs used to leave nothing behind. The worker reports once, at
    the end, and a call that reaches its ceiling is killed before that report
    exists -- so the only evidence was "no candidate", which cannot tell a stuck
    provider from an agent looping on its own tool output from a model that is
    simply thinking for an hour. The trace is written record by record, with a
    flush after each, so the shape of a hang survives the hang.

    Every line carries ``trace_s``, seconds since the first record. Without it a
    trace says what happened but not how long anything took, which is the whole
    question when the symptom is a call that never returns.

    Token-by-token ``*_delta`` events are *counted*, not stored: a single call
    emits thousands of them and each says a few characters, so storing them
    would fill the file with the least informative events and crowd out the tool
    calls. They are summarised in place every ``TRACE_STREAM_SUMMARY_EVERY``.

    Bounded at both ends, because the file is written by the process that is
    misbehaving: at most ``MAX_TRACE_RECORDS`` records, each serialised to at
    most ``MAX_TRACE_RECORD_CHARS``. An over-long record is replaced by a valid
    JSON object holding its head and tail, so the file stays parseable as JSONL
    and a huge tool result stays legible at both ends.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self.written = 0
        self._handle: Any = None
        self._started = 0.0
        self._truncation_noted = False
        self._stream_count = 0
        self._stream_chars = 0
        self._stream_kinds: Counter[str] = Counter()

    def open(self) -> EventTrace:
        self._started = time.monotonic()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        return self

    def close(self) -> None:
        # The last partial burst of deltas is summarised too, or a call that
        # spent its whole ceiling generating one message would show no
        # generation at all.
        self._summary()
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write(self, record: dict[str, Any]) -> None:
        if self._handle is None:
            return
        inner = record.get("assistantMessageEvent")
        if record.get("type") == "message_update" and isinstance(inner, dict):
            kind = str(inner.get("type", ""))
            if kind.endswith("_delta"):
                delta = inner.get("delta")
                self._stream_count += 1
                self._stream_chars += len(delta) if isinstance(delta, str) else 0
                self._stream_kinds[kind] += 1
                if self._stream_count % TRACE_STREAM_SUMMARY_EVERY == 0:
                    self._summary()
                return
        self._emit(record)

    def _summary(self) -> None:
        if not self._stream_count or self._handle is None:
            return
        self._emit(
            {
                "type": "trace_stream_summary",
                "deltas": self._stream_count,
                "chars": self._stream_chars,
                "kinds": dict(self._stream_kinds),
            }
        )
        self._stream_count = 0
        self._stream_chars = 0
        self._stream_kinds.clear()

    def _emit(self, record: dict[str, Any]) -> None:
        if self.written >= MAX_TRACE_RECORDS:
            if not self._truncation_noted:
                self._truncation_noted = True
                self._write_line({"type": "trace_truncated_after", "records": self.written})
            return
        text = json.dumps(record, ensure_ascii=False, default=str)
        if len(text) > MAX_TRACE_RECORD_CHARS:
            keep = MAX_TRACE_RECORD_CHARS // 2
            shortened: dict[str, Any] = {
                "type": record.get("type"),
                "trace_truncated_record_chars": len(text),
                "head": text[:keep],
                "tail": text[-keep:],
            }
            # A long message is exactly the case where the accounting matters
            # most, and truncation used to throw the whole `message` object away
            # with it -- so a 140 kB reply lost its usage and the ledger recorded
            # a paid call as unreported. The small fields survive on their own.
            message = record.get("message")
            if isinstance(message, dict):
                shortened["message"] = {
                    key: message.get(key)
                    for key in ("role", "stopReason", "usage", "model", "provider")
                    if key in message
                }
            text = json.dumps(shortened, ensure_ascii=False)
        self._write_line(text)

    def _write_line(self, line: str | dict[str, Any]) -> None:
        if isinstance(line, dict):
            line = json.dumps(line, ensure_ascii=False, default=str)
        # The timestamp is injected as text rather than merged into the record,
        # so that a record Pi sends with a `trace_s` key of its own keeps it.
        elapsed = round(time.monotonic() - self._started, 3)
        self._handle.write(f'{{"trace_s": {elapsed}, "event": {line}}}\n')
        self._handle.flush()
        self.written += 1


def _collect_assistant_text(record: dict[str, Any]) -> str:
    message = record.get("message") or {}
    if message.get("role") != "assistant":
        return ""
    content = message.get("content", [])
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


class HistoryTrace(EventTrace):
    """Complete semantic records for inheritance; streaming deltas are redundant."""

    def write(self, record: dict[str, Any]) -> None:
        if record.get("type") in {
            "message_end", "tool_execution_end", "agent_error", "extension_error", "error",
        }:
            self._write_line(json.dumps(record, ensure_ascii=False))


def run_request(
    request: dict[str, Any], trace: EventTrace | None = None,
    history: EventTrace | None = None,
) -> dict[str, Any]:
    exe = shutil.which("pi")
    if not exe:
        return {"ok": False, "error_type": "PI_NOT_INSTALLED",
                "error": "`pi` is not on PATH; install Pi using its upstream instructions."}

    cwd = Path(str(request["cwd"]))
    cwd.mkdir(parents=True, exist_ok=True)
    args = [exe, "--mode", "rpc", "--no-session"]
    if request.get("no_tools"):
        # Experiments that return a bounded text artifact must not give the
        # model filesystem, shell, MCP, skill, or project-instruction access.
        # This keeps the worker's authority at the prompt boundary.
        args += [
            "--no-tools", "--no-mcp", "--no-extensions", "--no-skills",
            "--no-context-files", "--no-prompt-templates", "--no-themes",
            "--offline",
        ]
        if request.get("system_prompt"):
            args += ["--system-prompt", str(request["system_prompt"])]
        if request.get("thinking"):
            # Recorded on every request: the reasoning setting changes what the
            # service is asked to do, so an experiment that fixed it in one arm
            # and not another would not be comparing the same thing.
            args += ["--thinking", str(request["thinking"])]
    if request.get("provider"):
        args += ["--provider", str(request["provider"])]
    if request.get("model"):
        args += ["--model", str(request["model"])]

    proc = subprocess.Popen(
        args, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, bufsize=0, shell=False,
    )
    events: queue.Queue[dict[str, Any]] = queue.Queue()
    seen: list[str] = []
    state = {"agent_error": False, "settled_aborted": False, "bad_json": 0,
             "output_limit": False, "output_bytes": 0}
    texts: list[str] = []

    def read_stdout() -> None:
        assert proc.stdout is not None
        for raw in iter(proc.stdout.readline, b""):
            try:
                events.put(json.loads(raw.decode("utf-8").rstrip("\r\n")))
            except (UnicodeError, json.JSONDecodeError):
                state["bad_json"] += 1

    def drain_stderr() -> None:
        assert proc.stderr is not None
        while proc.stderr.read(4096):
            pass

    threading.Thread(target=read_stdout, daemon=True).start()
    threading.Thread(target=drain_stderr, daemon=True).start()

    def send(command: dict[str, Any]) -> None:
        assert proc.stdin is not None
        proc.stdin.write((json.dumps(command, ensure_ascii=False) + "\n").encode("utf-8"))
        proc.stdin.flush()

    def observe(record: dict[str, Any]) -> None:
        if trace is not None:
            trace.write(record)
        if history is not None:
            history.write(record)
        kind = str(record.get("type", ""))
        inner = record.get("assistantMessageEvent")
        if kind == "message_update" and isinstance(inner, dict):
            delta = inner.get("delta")
            # Only the *deliverable* counts against the ceiling. Thinking deltas
            # arrive on the same channel with the same shape, and a call cut off
            # 32 KiB into its reasoning loses the answer it was about to write
            # while looking, from the outside, exactly like a runaway: the first
            # live MOF probe died that way, having emitted no text at all.
            if isinstance(delta, str) and str(inner.get("type", "")) == "text_delta":
                state["output_bytes"] += len(delta.encode("utf-8"))
                cap = int(request.get("max_output_bytes") or 0)
                if cap and state["output_bytes"] > cap and not state["output_limit"]:
                    state["output_limit"] = True
                    send({"id": "abort-output-limit", "type": "abort"})
        seen.append(kind)
        if kind in ("agent_error", "extension_error", "error"):
            state["agent_error"] = True
        if kind == "message_end":
            message = record.get("message") or {}
            if message.get("role") == "assistant":
                if message.get("stopReason") in ("error", "aborted"):
                    state["agent_error"] = True
                text = _collect_assistant_text(record)
                if text:
                    texts.append(text)
        if kind == "agent_settled" and record.get("aborted"):
            state["settled_aborted"] = True

    def recv_until(predicate: Any, seconds: float) -> dict[str, Any]:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            try:
                record = events.get(timeout=max(0.01, min(0.5, deadline - time.monotonic())))
            except queue.Empty:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"Pi exited early (code {proc.returncode}) before the awaited event"
                    ) from None
                continue
            observe(record)
            if predicate(record):
                return record
        raise TimeoutError(f"timed out waiting for Pi event after {seconds}s")

    try:
        send({"id": "state-1", "type": "get_state"})
        handshake = recv_until(
            lambda m: m.get("type") == "response" and m.get("id") == "state-1", 20
        )
        if not handshake.get("success"):
            return {"ok": False, "error_type": "PI_HANDSHAKE_FAILED",
                    "error": "Pi RPC get_state was rejected"}

        send({"id": "prompt-1", "type": "prompt", "message": str(request["prompt"])})
        ack = recv_until(
            lambda m: m.get("type") == "response" and m.get("id") == "prompt-1", 30
        )
        if not ack.get("success"):
            detail = ack.get("error")
            if isinstance(detail, dict):
                detail = detail.get("message") or detail.get("code")
            return {"ok": False, "error_type": "PI_PROMPT_REJECTED",
                    "error": f"Pi rejected the prompt command: {str(detail)[:300]}"}
        if (ack.get("data") or {}).get("disposition") == "handled":
            return {"ok": False, "error_type": "PI_PROMPT_NOT_RUN",
                    "error": "prompt was handled locally without a model run"}
        # The ack only means "accepted". Completion is agent_settled.
        #
        # `settle_s` is threaded through the request rather than hard-coded. It
        # used to be the literal 900 here, which silently made the caller's
        # `--agent-timeout-s` a ceiling on the *subprocess wait* and nothing
        # else: a run configured for 1800s was still cut off at 900, and the
        # only sign was a call that produced no candidate. Defaulting to 900
        # keeps a request that does not carry the field behaving exactly as it
        # did before.
        settled = recv_until(
            lambda m: m.get("type") == "agent_settled",
            float(request.get("settle_s") or SETTLE_DEFAULT_S),
        )
        if state["output_limit"]:
            return {"ok": False, "error_type": "PI_OUTPUT_LIMIT",
                    "error": f"assistant output exceeded {request.get('max_output_bytes')} bytes",
                    "event_types": seen[:80], "output_bytes": state["output_bytes"]}
        if state["bad_json"] or state["agent_error"] or state["settled_aborted"]:
            return {"ok": False, "error_type": "PI_AGENT_ERROR",
                    "error": "Pi reported an agent/provider error during the run"}

        send({"id": "state-2", "type": "get_state"})
        recv_until(lambda m: m.get("type") == "response" and m.get("id") == "state-2", 20)

        return {
            "ok": True,
            "final_response": "\n".join(texts)[-MAX_FINAL_RESPONSE_CHARS:],
            "session_id": str(settled.get("sessionId") or ""),
            "error_type": None,
            "error": None,
            "event_types": seen[:80],
        }
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:600],
                "event_types": seen[:80]}
    finally:
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: _pi_worker.py <request.json>", file=sys.stderr)
        return 2
    request_path = Path(argv[1]).resolve()
    result_path = Path(str(request_path) + ".result.json")
    trace_path = Path(str(request_path) + ".events.jsonl")
    result_path.write_text(
        json.dumps(_dispatch(request_path, trace_path), indent=2), encoding="utf-8"
    )
    return 0


def _dispatch(request_path: Path, trace_path: Path) -> dict[str, Any]:
    """One request in, one result payload out. Never raises.

    The worker has to leave a result file behind whatever happens, because the
    parent distinguishes "the agent failed" from "the worker never reported" by
    whether that file exists -- and the second reading sends the investigation
    to the wrong process.
    """
    trace = EventTrace(trace_path).open()
    history = HistoryTrace(Path(str(request_path) + ".history.jsonl")).open()
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
        return run_request(request, trace, history)
    except Exception as exc:
        return {"ok": False, "error_type": type(exc).__name__, "error": str(exc)[:600]}
    finally:
        trace.close()
        history.close()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
