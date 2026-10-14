#!/usr/bin/env python
"""Keep the frozen ``ab`` phase moving when its controller is killed from outside.

The controller records a pause marker for every defect it can see and exits
cleanly on its own policy stops, so most ways of stopping it already leave a
reason behind. What it cannot survive is being killed outright -- the observed
failure of 2026-10-14, when the controller and its vault keeper were both
terminated mid-arm and the sealed files had to be regenerated. This supervisor
answers that one case and nothing else:

* it never kills, signals, edits or scores anything;
* it starts ``mof_ab_continue.py --execute`` only when no live controller holds
  the lock, no pause marker exists, the phase has not finished and no unsealed
  results exist yet;
* it stops on the first clean exit, the first pause marker and the first
  completed unseal, and it will not start more than ``MAX_ATTEMPTS`` attempts;
* each attempt, exit and decision is appended to its own JSONL log beside the
  phase.

It carries no policy of its own: the phase digest, the arms, the ceilings, the
gate and the freeze all belong to the controller, and a restart re-enters
exactly the command a human would run by hand.

    python tools/mof_ab_supervise.py --check-only
    python tools/mof_ab_supervise.py --run
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from mof_ab_continue import (  # noqa: E402
    LOCK_PATH,
    OUT_ROOT,
    PAUSE_MARKER,
    STATUS_PATH,
    UNSEALED,
    live_controller_pids,
    read_json,
)

PHASE = "ab"
CONTROLLER = REPO / "tools" / "mof_ab_continue.py"
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
if not PYTHON.is_file():
    PYTHON = Path(sys.executable)
CHINA = timezone(timedelta(hours=8))

#: Bounded on purpose. A controller that dies eight times without writing a
#: reason is not a transient problem, and spending a ninth arm's budget on it
#: would be a decision no supervisor should make on its own.
MAX_ATTEMPTS = 8
RETRY_DELAY_S = 120


def now_iso() -> str:
    return datetime.now(CHINA).isoformat(timespec="seconds")


class Log:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, event: str, **fields: object) -> None:
        record = {"event": event, "at": now_iso(), **fields}
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        print("SUPERVISOR: " + json.dumps(record, sort_keys=True, default=str)[:600], flush=True)


def finished() -> str | None:
    """Why the phase needs no further controller run, if it does not."""
    if UNSEALED.is_dir() and any(UNSEALED.iterdir()):
        return "unsealed results already exist; refusing to score twice"
    status = read_json(STATUS_PATH) or {}
    if status.get("stage") == "complete":
        return "the controller reports the phase complete"
    return None


def delay(seconds: int) -> None:
    """Sleep, but stop early if a pause marker appears while waiting."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if PAUSE_MARKER.exists():
            return
        time.sleep(5)


def next_action() -> tuple[str, str]:
    """What one cycle would do next: ("stop"|"wait"|"run", reason)."""
    stop = finished()
    if stop:
        return "stop", stop
    if PAUSE_MARKER.exists():
        return "stop", f"{PAUSE_MARKER.name} is present"
    live = live_controller_pids()
    if live:
        return "wait", f"controller pids {sorted(live)} are running"
    return "run", "no live controller, no pause marker, phase unfinished"


def run(log: Log) -> int:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        action, reason = next_action()
        if action == "stop":
            log("stop", reason=reason, attempts=attempt - 1)
            return 0
        if action == "wait":
            log("wait", reason=reason, attempt=attempt)
            delay(RETRY_DELAY_S)
            continue

        log("attempt", n=attempt, reason=reason,
            lock_present=LOCK_PATH.exists(), controller=str(CONTROLLER))
        started = time.monotonic()
        result = subprocess.run(
            [str(PYTHON), "-u", str(CONTROLLER), "--execute"],
            cwd=str(REPO), capture_output=True, text=True,
            encoding="utf-8", errors="replace")
        seconds = round(time.monotonic() - started, 1)
        tail = "\n".join((result.stdout or "").strip().splitlines()[-3:])
        log("exit", n=attempt, code=result.returncode, seconds=seconds, tail=tail[:800])

        if result.returncode == 0:
            log("stop", reason="the controller exited cleanly", attempts=attempt)
            return 0
        if PAUSE_MARKER.exists():
            log("stop", reason=f"{PAUSE_MARKER.name} was written; a human decides next",
                attempts=attempt)
            return 0
        if attempt < MAX_ATTEMPTS:
            log("retry", reason="the controller died without recording a reason",
                n=attempt, delay_s=RETRY_DELAY_S)
            delay(RETRY_DELAY_S)
    log("stop", reason=f"{MAX_ATTEMPTS} attempts reached without a clean exit")
    return 2


def check_only() -> int:
    action, reason = next_action()
    print(json.dumps({
        "mode": "check-only",
        "would": action,
        "reason": reason,
        "pause_marker": PAUSE_MARKER.exists(),
        "lock_present": LOCK_PATH.exists(),
        "unsealed_present": UNSEALED.is_dir() and any(UNSEALED.iterdir()),
        "max_attempts": MAX_ATTEMPTS,
        "wrote": "nothing", "model_calls": 0, "controller_started": False,
    }, indent=2))
    return 0


def main(argv: list[str]) -> int:
    known = {"--check-only", "--run"}
    unknown = [arg for arg in argv if arg not in known]
    if unknown or len(set(argv)) != len(argv) or ("--check-only" in argv) == ("--run" in argv):
        print(f"usage: {Path(__file__).name} --check-only | --run", flush=True)
        return 2
    if "--check-only" in argv:
        return check_only()
    stamp = datetime.now(CHINA).strftime("%Y%m%dT%H%M%SZ")
    log = Log(OUT_ROOT / f"supervisor-{stamp}.jsonl")
    log("start", pid=os.getpid(), max_attempts=MAX_ATTEMPTS)
    return run(log)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("SUPERVISOR INTERRUPTED", flush=True)
        sys.exit(2)
    except Exception as exc:
        print(f"SUPERVISOR ERROR: {type(exc).__name__}: {exc}", flush=True)
        sys.exit(2)
