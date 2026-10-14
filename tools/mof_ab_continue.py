#!/usr/bin/env python
"""Controller: continue the frozen ``ab`` phase, one arm at a time.

This schedules the arms that are still outstanding in the phase that is already
frozen. It is not a new phase, a new identity or a new configuration: the phase
name, the protocol digest, the plans, the ceilings and the prepared root are all
read from the frozen protocol, and no parameter is set here.

It owns no earlier process. It never signals, kills or restarts anything, and it
never clears a pause marker -- it only writes one, and only on a defect.

    python tools/mof_ab_continue.py --check-only
    python tools/mof_ab_continue.py --execute

``--check-only`` touches nothing: no lock, no model call, no Pi call, no label
read or hash, no scoring, no gate, and no write of any kind -- including no write
to the live registry, which is opened read-only. It reports what would run and
what it could verify statically.

``--execute`` refuses to start unless the pause marker is absent, the initial
``ab-s101-r1`` B arm is ``COMMITTED`` with every world sealed and scoreable, and
the previous phase record says its vault was verified restored. It then runs the
remaining arms in frozen plan order, one at a time, verifying each from the
records before moving on, and runs the real gate when a paired block completes.
Any defect writes the pause marker with a short reason and stops.

Exit codes: 0 = the requested stage finished cleanly (a policy stop such as the
24h budget also exits 0, with its own status token); 2 = blocked, nothing more
was started.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))

from mof_ledger_audit import ApprovedFailure, audit_ledger  # noqa: E402

from dream_rsi.mof.cli import DATA_DIR, OUT_DIR, RAW_DIR, _pi_version, _prepared  # noqa: E402
from dream_rsi.mof.experiment import K2, MofExperimentRunner, default_protected_paths  # noqa: E402
from dream_rsi.mof.protocol import PHASES, Protocol, block_plans  # noqa: E402

PHASE = "ab"
PHASE_DIGEST = "a07196279bf61523954a763447ceaef8006f0a4d1766074214ec1fd9d250422d"

RUNS = REPO / ".runtime" / "runs"
REGISTRY_DB = REPO / ".runtime" / "registry.sqlite3"
OUT_ROOT = REPO / OUT_DIR / PHASE
BLOCKS_INDEX = OUT_ROOT / f"{PHASE}-blocks.jsonl"
PHASE_JSON = OUT_ROOT / f"{PHASE}-phase.json"
PAUSE_MARKER = OUT_ROOT / f"{PHASE}.pause"
LOCK_PATH = OUT_ROOT / f"{PHASE}.controller.lock"
STATUS_PATH = OUT_ROOT / "controller-status.json"
FREEZE_MANIFEST = OUT_ROOT / "final-freeze-manifest.json"
CLI_LOGS = OUT_ROOT / "cli-logs"
UNSEALED = OUT_ROOT / "unsealed"
GATE_SCRIPT = REPO / "tools" / "mof_gate.py"
TASK_DIR = REPO / "tasks" / "mof_screening"
RUNNER_SCRIPT = TASK_DIR / "candidate_runner.py"

#: A frozen candidate program belongs to its own run, under that run's directory:
#: ``.runtime/runs/<run_id>/frozen/budget-<checkpoint>/candidate.py``. Nothing
#: outside ``frozen/`` and nothing that is not ``candidate.py`` is read, so no
#: dataset, label or prompt file can be opened by the hash sweep. Every record is
#: expected to carry the same six checkpoints the runner froze.
FROZEN_DIRNAME = "frozen"
FROZEN_FILENAME = "candidate.py"
EXPECTED_FROZEN = 6

#: The single failure a human approved: one call, one run, matched on the
#: fingerprint of the identity the ledger row itself carries.
APPROVED_RUN_ID = "mof-ab-s101-r1-B"
APPROVED_FAILURE_SHA256 = \
    "330566cf47932d61160173329075d5ebe50ea250b0710432466545df56f17dc9"

INITIAL_B_START = "2026-10-14T12:14:19+08:00"
MAX_ELAPSED = timedelta(hours=24)

#: The pinned dataset archive, and the digest the protocol names for it. The
#: vault can only take what is on disk; a protected set that lost its archive
#: would still "seal" -- over the wrong files -- so its presence is asserted
#: before any arm starts.
PINNED_ARCHIVE = REPO / RAW_DIR / "CRAFTED-2.0.1.verified.tar.xz"
PINNED_ARCHIVE_SHA256 = \
    "f29f7b3d6b7d6c51fd113138bc4a811c448d4dcce66ed37fddbfa4141a238dde"

#: The record of what was lost and how it was restored, written once by the
#: recovery pass. When it exists the protected set must match it exactly.
RECOVERY_RECORD = OUT_ROOT / "vault-recovery.json"

#: An approval to resume one partially run arm in place. Without an entry here
#: that matches the arm's own partial state, a partially run arm still blocks
#: the phase -- silence is not permission.
RESUME_APPROVALS = OUT_ROOT / f"{PHASE}.resume-approvals.json"
LOCK_ARCHIVE = OUT_ROOT / f"{PHASE}.controller.lock.stale"

CHINA = timezone(timedelta(hours=8))
INDEX_BAD_LINES: list[int] = []
STATE: dict = {
    "phase": PHASE,
    "protocol_digest": PHASE_DIGEST,
    "executed_replay_K2": K2,
    "original_design_replay_K2": 12,
    "initial_b_start": INITIAL_B_START,
    "completed_arms": [],
    "current_arm": None,
    "phase_recorded_tokens": 0,
    "conservative_unknown_reserve": 0,
    "isolation_cycles": [],
    "gate_results": {},
}


# --------------------------------------------------------------------- plumbing


def now_iso() -> str:
    return datetime.now(CHINA).isoformat(timespec="seconds")


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
                    encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def emit(stage: str, **extra) -> None:
    """Write and print the bounded controller status. Nothing else is printed."""
    STATE["stage"] = stage
    STATE["at"] = now_iso()
    STATE.update(extra)
    write_json(STATUS_PATH, STATE)
    print("STATUS: " + json.dumps(STATE, sort_keys=True, default=str)[:1200], flush=True)


def run_id_for(block_id: str, method: str) -> str:
    return f"mof-{PHASE}-{block_id.split('-', 1)[1]}-{method}"


def latest_records() -> dict[str, dict]:
    """Latest record per run id from the phase index; the index is append-only."""
    INDEX_BAD_LINES.clear()
    records: dict[str, dict] = {}
    if not BLOCKS_INDEX.is_file():
        return records
    for number, line in enumerate(BLOCKS_INDEX.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            INDEX_BAD_LINES.append(number)
            continue
        if isinstance(row, dict) and row.get("run_id"):
            records[str(row["run_id"])] = row
    return records


def approved_failure_for(run_id: str) -> ApprovedFailure | None:
    if run_id != APPROVED_RUN_ID:
        return None
    return ApprovedFailure(sha256=APPROVED_FAILURE_SHA256, run_id=run_id)


def pause(reason: str) -> None:
    """Write the pause marker. Never removed by this script."""
    PAUSE_MARKER.parent.mkdir(parents=True, exist_ok=True)
    PAUSE_MARKER.write_text(f"{now_iso()} continue-ab: {reason}\n", encoding="utf-8")


def blocker(operation: str) -> int:
    pause(operation)
    emit("blocked", reason=operation[:300])
    return 2


# ------------------------------------------------------------------------- lock


def pid_exists(pid: int) -> bool:
    """Whether a process with this pid is running. Never signals anything."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    import ctypes
    SYNCHRONIZE = 0x00100000
    WAIT_TIMEOUT = 0x00000102
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
    finally:
        kernel32.CloseHandle(handle)


def live_controller_pids() -> set[int] | None:
    """Pids whose command line is this controller running ``--execute``.

    ``None`` means the probe could not answer, and every caller must then treat
    the lock as live: an unanswerable question is not evidence of absence.
    """
    if os.name != "nt":
        return None
    script = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^py' -and "
        "$_.CommandLine -match 'mof_ab_continue\\.py' -and "
        "$_.CommandLine -match '--execute' } | "
        "ForEach-Object { $_.ProcessId }")
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=90)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return {int(token) for token in (result.stdout or "").split() if token.isdigit()}


def lock_owner_alive(data: dict) -> bool:
    pid = data.get("pid")
    if not isinstance(pid, int) or isinstance(pid, bool):
        return False
    live = live_controller_pids()
    if live is None:
        return pid_exists(pid)
    return pid in live


def archive_stale_lock(data: dict) -> None:
    """Move a dead controller's lock aside, bytes kept, reason recorded."""
    stamp = datetime.now(CHINA).strftime("%Y%m%dT%H%M%S")
    target = LOCK_ARCHIVE.with_name(f"{LOCK_ARCHIVE.name}-{stamp}")
    try:
        LOCK_PATH.replace(target)
    except OSError:
        return
    write_json(target.with_name(target.name + ".json"),
               {"reason": "the recorded controller pid is not a live controller",
                "archived_at": now_iso(), "lock": data})
    STATE.setdefault("stale_locks", []).append(
        {"at": now_iso(), "file": str(target), "lock": data})


def acquire_lock() -> str | None:
    """Atomically take the controller lock; a dead controller's lock is archived first.

    A lock whose controller is gone is not authority to stay blocked -- the
    process it recorded is not coming back -- and it is not deleted either: its
    bytes are archived beside the phase with the reason, so the interruption it
    records stays readable. A lock whose controller is alive still refuses.
    """
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        token = uuid.uuid4().hex
        try:
            handle = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            data = read_json(LOCK_PATH) or {}
            if lock_owner_alive(data):
                return None
            archive_stale_lock(data)
            continue
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"token": token, "pid": os.getpid(), "at": now_iso()}) + "\n")
        return token
    return None


def release_lock(token: str) -> None:
    """Remove the lock only if it is still the one this process created."""
    try:
        data = read_json(LOCK_PATH) or {}
    except OSError:
        return
    if data.get("token") != token:
        return
    with suppress(OSError):
        LOCK_PATH.unlink()


# ------------------------------------------------------------------ inspection


def registry_column(query: str, run_id: str) -> list:
    """Read the live registry read-only: it is never created, migrated or written."""
    connection = sqlite3.connect(REGISTRY_DB.as_uri() + "?mode=ro", uri=True)
    try:
        return connection.execute(query, (run_id,)).fetchall()
    finally:
        connection.close()


def world_report(run_id: str, method: str) -> tuple[list[str], dict]:
    """World, round and revision facts for one arm, read without touching the database.

    A world row is not a round. An arm interrupted before a world sealed is
    resumed by quarantining that world and building its round again under a new
    id (``-r1``), which is the resume path's own design -- so the row count can
    exceed the round count for a reason the phase created, not hid. What the
    phase requires is unchanged: exactly one sealed, scoreable world for each of
    its outer rounds, no gaps, and nothing left building. A quarantined retry is
    reported as a recorded fact rather than counted as a missing or extra world.
    """
    spec = PHASES[PHASE]
    problems: list[str] = []
    facts: dict = {"worlds": 0, "sealed": 0, "rounds": 0, "quarantined": 0, "revisions": 0}
    try:
        rows = [(str(row[0]), int(row[1]), str(row[2])) for row in registry_column(
            "select world_id, outer_t, state from worlds where run_id = ?", run_id)]
        payloads = [row[0] for row in
                    registry_column("select payload from revisions where run_id = ?", run_id)]
    except (sqlite3.Error, OSError, ValueError, TypeError):
        return [f"{run_id}: the registry could not be read read-only"], facts

    by_round: dict[int, list[tuple[str, str]]] = {}
    for world_id, outer_t, state in rows:
        by_round.setdefault(outer_t, []).append((world_id, state))
    facts["worlds"] = len(rows)
    facts["revisions"] = len(payloads)
    facts["quarantined"] = sum(1 for _, _, state in rows if state == "QUARANTINED")
    facts["quarantine_detail"] = [{"world_id": world_id, "outer_t": outer_t}
                                  for world_id, outer_t, state in rows
                                  if state == "QUARANTINED"]

    sealed_rounds = 0
    for outer_t in sorted(by_round):
        entries = by_round[outer_t]
        sealed_here = [world_id for world_id, state in entries if state == "SEALED_SCOREABLE"]
        unexpected = [(world_id, state) for world_id, state in entries
                      if state not in ("SEALED_SCOREABLE", "QUARANTINED")]
        if not 0 <= outer_t < spec.worlds:
            problems.append(f"{run_id}: a world sits at outer round {outer_t}")
        if len(sealed_here) > 1:
            problems.append(f"{run_id}: {len(sealed_here)} sealed worlds at round {outer_t}")
        if sealed_here:
            sealed_rounds += 1
        if unexpected:
            problems.append(f"{run_id}: {unexpected[0][0]} is {unexpected[0][1]} at round {outer_t}")
    facts["sealed"] = sealed_rounds
    facts["rounds"] = sum(1 for outer_t in by_round if 0 <= outer_t < spec.worlds)
    if sealed_rounds != spec.worlds:
        problems.append(f"{run_id}: {sealed_rounds} of {spec.worlds} outer rounds sealed scoreable")

    if method == "A":
        statuses = []
        for payload in payloads:
            try:
                statuses.append((json.loads(payload) or {}).get("status"))
            except (TypeError, ValueError):
                problems.append(f"{run_id}: a revision payload could not be decoded")
        if "CONTEXT_LIMIT" in statuses:
            problems.append(f"{run_id}: a revision was cut by the context ceiling")
        if "DEVELOPER_ERROR" in statuses:
            problems.append(f"{run_id}: a revision ended in a developer error")
    return problems, facts


def ledger_report(run_id: str, row: dict) -> tuple[list[str], dict]:
    """Ledger audit for one arm, with the approval only where it was granted."""
    spec = PHASES[PHASE]
    problems: list[str] = []
    ledger_path = RUNS / run_id / "cost-ledger.jsonl"
    if not ledger_path.is_file():
        return [f"{run_id}: no cost ledger"], {"ledger": str(ledger_path), "exists": False}

    approval = approved_failure_for(run_id)
    audit = audit_ledger(ledger_path, approved=approval)
    if approval is not None and audit.approved_failed != 1:
        problems.append(f"{run_id}: the approved failure is not exactly one row "
                        f"({audit.approved_failed})")
    if audit.failed != audit.approved_failed:
        problems.append(f"{run_id}: failed calls {audit.failed} "
                        f"(approved {audit.approved_failed})")
    if audit.refused:
        problems.append(f"{run_id}: {audit.refused} refused call(s)")
    if audit.unreported:
        problems.append(f"{run_id}: {audit.unreported} call(s) without a usage report")
    if audit.unreadable:
        problems.append(f"{run_id}: {audit.unreadable} unreadable ledger row(s)")
    if audit.outstanding:
        problems.append(f"{run_id}: {audit.outstanding} reservation(s) left unsettled")

    charged = int((row.get("ledger") or {}).get("charged_total") or 0)
    if charged + audit.conservative_unknown_reserve > spec.token_budget:
        problems.append(f"{run_id}: recorded plus reserve exceeds the arm ceiling")

    facts = {
        "ledger": str(ledger_path),
        "exists": True,
        "rows": audit.rows,
        "requests": audit.requests,
        "failed": audit.failed,
        "approved_failed": audit.approved_failed,
        "refused": audit.refused,
        "unreported": audit.unreported,
        "unreadable": audit.unreadable,
        "outstanding": audit.outstanding,
        "zero_charge": audit.zero_charge,
        "conservative_unknown_reserve": audit.conservative_unknown_reserve,
        "sha256": sha256_file(ledger_path),
    }
    return problems, facts


def frozen_source_report(run_id: str, row: dict) -> tuple[list[str], list[dict]]:
    """Hash every frozen candidate program. Never executes one, never reads data."""
    problems: list[str] = []
    checked: list[dict] = []
    frozen_root = (RUNS / run_id / FROZEN_DIRNAME).resolve()
    entries = row.get("frozen") or []
    if not entries:
        return [f"{run_id}: the record lists no frozen candidate"], checked
    if len(entries) != EXPECTED_FROZEN:
        problems.append(f"{run_id}: {len(entries)} frozen candidates of {EXPECTED_FROZEN}")

    seen: list[int] = []
    for entry in entries:
        if not isinstance(entry, dict):
            problems.append(f"{run_id}: a frozen entry is not a record")
            continue
        checkpoint = entry.get("checkpoint")
        if not isinstance(checkpoint, int) or isinstance(checkpoint, bool):
            problems.append(f"{run_id}: a frozen entry carries no checkpoint number")
            continue
        seen.append(checkpoint)
        source, expected = entry.get("source_path"), entry.get("candidate_sha256")
        if not source or not expected:
            problems.append(f"{run_id}: checkpoint {checkpoint} carries no source or digest")
            continue
        path = Path(str(source))
        if not path.is_absolute():
            path = REPO / path
        try:
            resolved = path.resolve()
        except OSError:
            problems.append(f"{run_id}: checkpoint {checkpoint} source cannot be resolved")
            continue
        if frozen_root not in resolved.parents:
            problems.append(f"{run_id}: checkpoint {checkpoint} source is outside "
                            f"{FROZEN_DIRNAME}/")
            continue
        if resolved.name != FROZEN_FILENAME or resolved.suffix != ".py":
            problems.append(f"{run_id}: checkpoint {checkpoint} source is not "
                            f"{FROZEN_FILENAME}")
            continue
        if not resolved.is_file():
            problems.append(f"{run_id}: checkpoint {checkpoint} source is missing")
            continue
        actual = sha256_file(resolved)
        checked.append({"run_id": run_id, "checkpoint": checkpoint, "source": str(resolved),
                        "sha256": actual, "ok": actual == expected})
        if actual != expected:
            problems.append(f"{run_id}: checkpoint {checkpoint} digest differs from the record")
    if len(set(seen)) != len(seen):
        problems.append(f"{run_id}: a checkpoint number is listed twice")
    return problems, checked


def isolation_problems(isolation: dict, label: str) -> list[str]:
    """The three facts an arm's own vault cycle must state, as values."""
    problems: list[str] = []
    if not isolation.get("sealed"):
        problems.append(f"{label}: the vault did not report itself sealed")
    if (isolation.get("verified_absent") or {}).get("ok") is not True:
        problems.append(f"{label}: the protected paths were not verified absent")
    if (isolation.get("verified_restored") or {}).get("ok") is not True:
        problems.append(f"{label}: the vault was not verified restored")
    return problems


def partial_fingerprint(run_id: str) -> str | None:
    """A digest of the partial state an in-place resume would reuse or discard.

    Approval is granted to a state, not to a name: the worlds, their rounds and
    states, the revisions and the run's own state are all hashed, so resuming
    after the state moved on is a different question and needs a different
    answer. Returns None when the state cannot be read.
    """
    try:
        worlds = registry_column(
            "select world_id, outer_t, state from worlds where run_id = ?", run_id)
        revisions = registry_column(
            "select iteration_t, m, status from revisions where run_id = ?", run_id)
        runs = registry_column("select state from runs where run_id = ?", run_id)
    except (sqlite3.Error, OSError, ValueError):
        return None
    payload = {
        "run_id": run_id,
        "run_state": str(runs[0][0]) if runs else None,
        "worlds": sorted([[str(w[0]), int(w[1]), str(w[2])] for w in worlds]),
        "revisions": sorted([[int(r[0]), int(r[1]), str(r[2])] for r in revisions]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def partial_state_facts(run_id: str) -> dict:
    """A read-only summary of an interrupted arm's state, without judging it.

    An arm that has not finished has no block record, so the record-based
    verification is not available for it; this is what can be said instead, and
    every value is read, never inferred.
    """
    try:
        worlds = registry_column(
            "select world_id, outer_t, state from worlds where run_id = ?", run_id)
        revisions = registry_column("select status from revisions where run_id = ?", run_id)
        runs = registry_column("select state from runs where run_id = ?", run_id)
    except (sqlite3.Error, OSError, ValueError):
        return {"run_id": run_id, "readable": False}
    return {
        "run_id": run_id,
        "readable": True,
        "run_state": str(runs[0][0]) if runs else None,
        "worlds": [{"world_id": str(world[0]), "outer_t": int(world[1]), "state": str(world[2])}
                   for world in worlds],
        "revision_statuses": sorted(str(row[0]) for row in revisions),
        "fingerprint": partial_fingerprint(run_id),
    }


def resume_approvals() -> dict[str, dict]:
    """Recorded approvals to resume one partially run arm in place, by run id."""
    data = read_json(RESUME_APPROVALS) or {}
    approvals: dict[str, dict] = {}
    entries = data.get("approvals")
    if isinstance(entries, list):
        for item in entries:
            if isinstance(item, dict) and item.get("run_id"):
                approvals[str(item["run_id"])] = item
    return approvals


def protected_paths() -> list[Path]:
    """The same expansion the vault seals: every protected path that exists."""
    return default_protected_paths(_prepared(REPO), dataset_root=REPO / DATA_DIR,
                                   raw_root=REPO / RAW_DIR)


def protected_set_report(*, hash_archive: bool) -> tuple[list[str], dict]:
    """What the vault would actually take, checked before any arm starts.

    Sealing takes what is on disk, so a half-restored protected set would seal
    quietly -- over fewer files -- and the arm would run with labels and the
    pinned archive missing rather than with an unsealed vault. This is the check
    that refuses that: every protected family must hold files, the pinned
    archive must be present (and, in execute mode, hash-verified), and when a
    recovery record exists the set must match it exactly.
    """
    problems: list[str] = []
    facts: dict = {"families": [], "files": 0, "bytes": 0}
    files: list[Path] = []
    for path in protected_paths():
        if path.is_dir():
            members = sorted(entry for entry in path.rglob("*") if entry.is_file())
        else:
            members = [path]
        family = {"path": str(path), "files": len(members),
                  "bytes": sum(entry.stat().st_size for entry in members)}
        facts["families"].append(family)
        facts["files"] += family["files"]
        facts["bytes"] += family["bytes"]
        files.extend(members)
        if not members:
            problems.append(f"{path}: no files; the vault would seal this family empty")

    if not PINNED_ARCHIVE.is_file():
        problems.append(f"the pinned archive is missing: {PINNED_ARCHIVE}")
    elif hash_archive:
        actual = sha256_file(PINNED_ARCHIVE)
        facts["pinned_archive_sha256"] = actual
        if actual != PINNED_ARCHIVE_SHA256:
            problems.append("the pinned archive's digest differs from the protocol's")

    record = read_json(RECOVERY_RECORD)
    if record is not None:
        facts["recovery_record"] = str(RECOVERY_RECORD)
        if record.get("files") != facts["files"] or record.get("bytes") != facts["bytes"]:
            problems.append(
                f"the protected set holds {facts['files']} files / {facts['bytes']} bytes; "
                f"the recovery record holds {record.get('files')} / {record.get('bytes')}")
    return problems, facts


def arm_report(run_id: str, method: str) -> tuple[list[str], dict]:
    """Everything the controller asserts about one arm. Read-only."""
    problems: list[str] = []
    records = latest_records()
    if INDEX_BAD_LINES:
        problems.append(f"the phase index has {len(INDEX_BAD_LINES)} unreadable line(s)")
    row = records.get(run_id)
    if row is None:
        return [f"{run_id}: no block record"], {"run_id": run_id, "record": False}

    facts: dict = {"run_id": run_id, "record": True, "status": row.get("status")}
    if row.get("status") != "COMMITTED":
        problems.append(f"{run_id}: status {row.get('status')}")
    if row.get("protocol_digest") != PHASE_DIGEST:
        problems.append(f"{run_id}: protocol digest {str(row.get('protocol_digest'))[:16]}")

    world_problems, world_facts = world_report(run_id, method)
    ledger_problems, ledger_facts = ledger_report(run_id, row)
    source_problems, sources = frozen_source_report(run_id, row)
    problems += world_problems + ledger_problems + source_problems
    facts.update(world_facts)
    facts["ledger_facts"] = ledger_facts
    facts["sources"] = sources
    return problems, facts


# ------------------------------------------------------------------------ gate


def run_gate(block_id: str) -> tuple[bool, str]:
    """The real gate, as a fixed subprocess. Its exit code is the verdict."""
    result = subprocess.run(
        [sys.executable, str(GATE_SCRIPT), PHASE, block_id],
        cwd=str(REPO), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=900)
    tail = "\n".join((result.stdout or "").strip().splitlines()[-12:])
    if tail:
        print(f"GATE {block_id}:\n{tail}", flush=True)
    return result.returncode == 0, f"exit {result.returncode}"


def gate_block(block_id: str, plans: list, records: dict) -> list[str]:
    """Gate a paired block, but only once both of its arms are committed."""
    run_ids = [run_id_for(block_id, plan.method) for plan in plans
               if plan.block_id == block_id]
    if not run_ids or any((records.get(rid) or {}).get("status") != "COMMITTED"
                          for rid in run_ids):
        return []
    ok, detail = run_gate(block_id)
    STATE["gate_results"][block_id] = detail
    if not ok:
        return [f"gate {block_id} failed ({detail})"]
    return []


def run_experiment_cli(argv: list[str], name: str) -> int:
    """Run one of the experiment's own CLIs. Its output is not echoed here."""
    result = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=7200)
    CLI_LOGS.mkdir(parents=True, exist_ok=True)
    (CLI_LOGS / f"{name}.log").write_text(
        f"argv: {argv}\nexit: {result.returncode}\n{result.stdout or ''}\n"
        f"{result.stderr or ''}\n", encoding="utf-8")
    print(f"CLI {name}: exit {result.returncode}", flush=True)
    return result.returncode


# ------------------------------------------------------------------- preflight


def plan_rows(plans: list, records: dict) -> list[dict]:
    rows = []
    for plan in plans:
        run_id = run_id_for(plan.block_id, plan.method)
        row = records.get(run_id)
        if row is None:
            state = "partial" if (RUNS / run_id / "cost-ledger.jsonl").is_file() else "pending"
        else:
            state = "committed" if row.get("status") == "COMMITTED" else "partial"
        rows.append({"block_id": plan.block_id, "method": plan.method,
                     "run_id": run_id, "state": state})
    return rows


def phase_preconditions(plans: list) -> tuple[list[str], dict[str, dict], list[dict]]:
    """What both modes assert before anything starts, read from the records."""
    problems: list[str] = []
    if PAUSE_MARKER.exists():
        problems.append(f"{PAUSE_MARKER.name} is present")

    records = latest_records()
    if INDEX_BAD_LINES:
        problems.append(f"the phase index has {len(INDEX_BAD_LINES)} unreadable line(s)")

    planned = {run_id_for(plan.block_id, plan.method) for plan in plans}
    for run_id in sorted(set(records) - planned):
        problems.append(f"{run_id}: a run in the index is not in the frozen plan")

    rows = plan_rows(plans, records)
    approvals = resume_approvals()
    for row in rows:
        if row["state"] != "partial":
            continue
        approval = approvals.get(row["run_id"])
        if approval is None:
            problems.append(f"{row['run_id']}: partially run; not resumed")
            continue
        actual = partial_fingerprint(row["run_id"])
        if actual is None:
            problems.append(f"{row['run_id']}: the partial state could not be read")
        elif approval.get("partial_fingerprint") != actual:
            problems.append(f"{row['run_id']}: the partial state differs from the approved "
                            f"resume fingerprint")
        else:
            row["resume_approved"] = True
            row["resume_approval"] = approval.get("authority")

    previous = read_json(PHASE_JSON)
    if previous is None:
        problems.append(f"{PHASE_JSON.name} is absent")
    else:
        problems += isolation_problems(previous.get("isolation") or {},
                                       f"the previous phase record of {PHASE}")
    return problems, records, rows


def preflight(plans: list) -> tuple[list[str], dict[str, dict], list[dict]]:
    problems, records, rows = phase_preconditions(plans)
    if (records.get(APPROVED_RUN_ID) or {}).get("status") != "COMMITTED":
        problems.append(f"{APPROVED_RUN_ID}: initial B is not committed")
    protected_problems, protected_facts = protected_set_report(hash_archive=True)
    STATE["protected_set"] = protected_facts
    problems += protected_problems
    return problems, records, rows


def phase_totals() -> tuple[int, int]:
    """Recorded tokens (latest record per run id) and the separate reserve."""
    records = latest_records()
    recorded = sum(int((row.get("ledger") or {}).get("charged_total") or 0)
                   for row in records.values())
    reserve = 0
    row = records.get(APPROVED_RUN_ID)
    if row is not None:
        _, facts = ledger_report(APPROVED_RUN_ID, row)
        if facts.get("exists"):
            reserve = int(facts.get("conservative_unknown_reserve") or 0)
    STATE["phase_recorded_tokens"] = recorded
    STATE["conservative_unknown_reserve"] = reserve
    return recorded, reserve


def elapses_past_budget() -> bool:
    started = STATE.get("initial_b_start") or INITIAL_B_START
    try:
        origin = datetime.fromisoformat(str(started))
    except ValueError:
        origin = datetime.fromisoformat(INITIAL_B_START)
    return datetime.now(CHINA) - origin > MAX_ELAPSED


# ------------------------------------------------------------------- execution


def build_protocol() -> Protocol:
    """The frozen protocol, with the Pi version the original records carry."""
    if K2 != 6:
        raise RuntimeError("replay horizon differs from the completed initial B")
    return Protocol(phase=PHASE, prepared_root=str(_prepared(REPO)), pi_version=_pi_version())


def run_one_arm(plan, protocol: Protocol) -> dict:
    """One arm, one fresh runner, one vault cycle. The runner restores in finally."""
    prepared = _prepared(REPO)
    protect = default_protected_paths(prepared, dataset_root=REPO / DATA_DIR,
                                      raw_root=REPO / RAW_DIR)
    runner = MofExperimentRunner(
        project_root=REPO, protocol=protocol, task_dir=TASK_DIR,
        runner_script=RUNNER_SCRIPT, out_root=OUT_ROOT)
    return runner.run_phase([plan], protect=protect)


def keep_phase_snapshot(run_id: str) -> bool:
    """Copy a run's own phase record aside. run_phase overwrites it each arm.

    Nothing is copied under a run id whose records are not in the file, and an
    existing snapshot is never overwritten.
    """
    if not PHASE_JSON.is_file():
        return False
    target = OUT_ROOT / f"{PHASE}-phase-{run_id}.json"
    if target.exists():
        return False
    try:
        data = json.loads(PHASE_JSON.read_text(encoding="utf-8"))
    except ValueError:
        return False
    blocks = data.get("blocks") if isinstance(data, dict) else None
    if not blocks or any((row or {}).get("run_id") != run_id for row in blocks):
        return False
    target.write_bytes(PHASE_JSON.read_bytes())
    return True


def cycle_facts(run_id: str, isolation: dict) -> dict:
    return {
        "run_id": run_id,
        "at": now_iso(),
        "file": f"{PHASE}-phase-{run_id}.json",
        "sealed": bool(isolation.get("sealed")),
        "verified_absent_ok": (isolation.get("verified_absent") or {}).get("ok"),
        "released": bool(isolation.get("released")),
        "verified_restored_ok": (isolation.get("verified_restored") or {}).get("ok"),
    }


def aggregate_phase(cycles: list[dict]) -> dict:
    """Truthful final phase record: latest unique arm records, no double count."""
    spec = PHASES[PHASE]
    records = latest_records()
    ordered = [records[run_id] for run_id in sorted(records)]
    recorded = sum(int((row.get("ledger") or {}).get("charged_total") or 0) for row in ordered)
    reserve = int(STATE.get("conservative_unknown_reserve") or 0)
    last_cycle = read_json(PHASE_JSON) or {}
    summary = {
        "phase": PHASE,
        "protocol_digest": PHASE_DIGEST,
        "implementation_revision": max(int(row.get("implementation_revision") or 0)
                                       for row in ordered) if ordered else 0,
        # One vault cycle ran per arm, the first of them owned by the runner that
        # was already alive. This field is the last cycle, not a single keeper
        # that sealed the whole calendar time; every cycle, from the initial B
        # record onwards, is preserved in isolation_cycles.
        "isolation": last_cycle.get("isolation") or {},
        "isolation_note": ("one vault cycle per arm-run, preserved in isolation_cycles; "
                           "the field above is the last cycle only"),
        "isolation_cycles": cycles,
        "phase_token_cap": spec.total_token_cap,
        "executed_replay_K2": K2,
        "original_design_replay_K2": 12,
        "phase_tokens_committed": recorded,
        "conservative_unknown_reserve": reserve,
        "recorded_plus_reserve": recorded + reserve,
        "accounting_exceptions": accounting_exceptions(),
        "notes": [
            f"{len(ordered)} arm-run(s) recorded for this phase; the initial "
            f"{APPROVED_RUN_ID} record was kept, not re-run",
            "one discovery call of the initial B arm was cut off by the output limit; "
            "its billed tokens are unknown and its reservation upper bound is carried "
            "as a separate conservative reserve, never as recorded usage",
            "recorded token positions were not rewritten",
            "exploratory: 2 paired blocks, one search each, on the pilot pool that was "
            "already opened; no C arm is registered and no confirmatory verdict is claimed",
        ],
        "blocks": ordered,
    }
    write_json(PHASE_JSON, summary)
    return summary


def accounting_exceptions() -> list[dict]:
    return [{
        "run_id": APPROVED_RUN_ID,
        "phase": PHASE,
        "call_id": "229d0d5d413b4ec988dfcaf79b1d029c",
        "seq": 60,
        "role": "discovery",
        "error_type": "PI_OUTPUT_LIMIT",
        "charged_tokens": 0,
        "usage_reported": True,
        "reservation_upper_bound": 23312,
        "billed_tokens": "unknown",
        "authorization": "human user, this one request only",
        "note": ("aborted output-limit discovery call; the billed tokens are unknown, "
                 "23312 is a separate conservative reservation and is not measured or "
                 "billed tokens"),
    }]


def execute() -> int:
    token = acquire_lock()
    if token is None:
        print(f"STATUS: {json.dumps({'status': 'LOCKED', 'lock': str(LOCK_PATH)})}", flush=True)
        return 2

    # A previous controller's status is carried forward, so a restart keeps the
    # same clock and the same isolated accounting rather than starting over.
    previous = read_json(STATUS_PATH)
    if isinstance(previous, dict):
        for key in ("initial_b_start", "completed_arms", "isolation_cycles", "gate_results"):
            if previous.get(key):
                STATE[key] = previous[key]

    retained = False
    try:
        code = _execute_locked()
    except Exception as exc:
        # The interrupted lock is evidence: it is deliberately left in place, and
        # the pause marker and the status record say what stopped.
        retained = True
        pause(f"unexpected controller error: {type(exc).__name__}")
        emit("controller-error", error_type=type(exc).__name__)
        code = 2
    if not retained:
        release_lock(token)
    return code


def _execute_locked() -> int:
    plans = block_plans(PHASE)
    if not plans:
        return blocker("the frozen phase has no plans")

    problems, _, _ = preflight(plans)
    if problems:
        return blocker("preflight: " + "; ".join(problems[:4]))

    spec = PHASES[PHASE]
    # The initial B arm is skipped entirely, but its vault cycle is retained
    # first, before any later arm overwrites the record it was written into.
    cycles = [c for c in (STATE.get("isolation_cycles") or []) if isinstance(c, dict)]
    if not any(c.get("run_id") == APPROVED_RUN_ID for c in cycles):
        initial = read_json(PHASE_JSON) or {}
        cycles.insert(0, cycle_facts(APPROVED_RUN_ID, initial.get("isolation") or {}))
    keep_phase_snapshot(APPROVED_RUN_ID)
    STATE["isolation_cycles"] = cycles

    completed = list(STATE.get("completed_arms") or [])
    _, reserve = phase_totals()
    initial_problems, _ = arm_report(APPROVED_RUN_ID, "B")
    if initial_problems:
        return blocker("initial B: " + "; ".join(initial_problems[:3]))
    if APPROVED_RUN_ID not in completed:
        completed.append(APPROVED_RUN_ID)
    emit("ready", completed_arms=completed)

    for plan in plans:
        run_id = run_id_for(plan.block_id, plan.method)
        records = latest_records()
        if (records.get(run_id) or {}).get("status") == "COMMITTED":
            if run_id not in completed:
                completed.append(run_id)
            continue

        if PAUSE_MARKER.exists():
            return blocker(f"{PAUSE_MARKER.name} appeared before {run_id}")
        if elapses_past_budget():
            pause(f"24h elapsed since the initial B start before {run_id}")
            emit("paused_time_budget", current_arm=None)
            return 0

        recorded, reserve = phase_totals()
        if recorded + reserve + spec.token_budget > spec.total_token_cap:
            return blocker(f"no headroom for {run_id}: {recorded:,} recorded + "
                           f"{reserve:,} reserve + {spec.token_budget:,} ceiling")

        # The digest is checked before the arm's model call, every arm.
        protocol = build_protocol()
        if protocol.digest != PHASE_DIGEST:
            return blocker(f"{run_id}: the protocol digest does not match the frozen digest")

        emit("running", current_arm=run_id, completed_arms=completed,
             phase_recorded_tokens=recorded, conservative_unknown_reserve=reserve)
        try:
            summary = run_one_arm(plan, protocol)
        except Exception as exc:
            return blocker(f"{run_id} raised {type(exc).__name__}")

        # Checked as soon as the arm returns, before another arm can start.
        isolation = summary.get("isolation") or {}
        keep_phase_snapshot(run_id)
        cycles.append(cycle_facts(run_id, isolation))
        STATE["isolation_cycles"] = cycles
        problems = isolation_problems(isolation, run_id)
        arm_problems, facts = arm_report(run_id, plan.method)
        problems += arm_problems

        recorded, reserve = phase_totals()
        emit("arm-complete" if not problems else "arm-defect", current_arm=run_id,
             completed_arms=completed + ([run_id] if not problems else []),
             phase_recorded_tokens=recorded, conservative_unknown_reserve=reserve,
             arm={key: facts.get(key) for key in
                  ("status", "worlds", "sealed", "revisions", "sources")})
        if problems:
            return blocker(f"{run_id}: " + "; ".join(problems[:4]))
        completed.append(run_id)
        STATE["completed_arms"] = completed

        gate_problems = gate_block(plan.block_id, plans, latest_records())
        if gate_problems:
            return blocker("; ".join(gate_problems))

    STATE["completed_arms"] = completed
    STATE["isolation_cycles"] = cycles
    aggregate_phase(cycles)
    emit("phase-aggregated", completed_arms=completed, current_arm=None)
    return finalise(plans)


# --------------------------------------------------------------- freeze/unseal


def finalise(plans: list) -> int:
    """Last gate pass, freeze manifest, then unseal and report. No Pi after here."""
    if UNSEALED.is_dir() and any(UNSEALED.iterdir()):
        return blocker("unsealed results already exist; refusing to score twice")

    problems: list[str] = []
    checked_sources: list[dict] = []
    ledger_hashes: dict[str, dict] = {}
    for plan in plans:
        run_id = run_id_for(plan.block_id, plan.method)
        arm_problems, facts = arm_report(run_id, plan.method)
        problems += arm_problems
        checked_sources += facts.get("sources") or []
        ledger_hashes[run_id] = facts.get("ledger_facts") or {}

    for block_id in sorted({plan.block_id for plan in plans}):
        ok, detail = run_gate(block_id)
        STATE["gate_results"][block_id] = detail
        if not ok:
            problems.append(f"gate {block_id} failed ({detail})")

    if problems:
        return blocker("freeze: " + "; ".join(problems[:4]))

    recorded, reserve = phase_totals()
    write_json(FREEZE_MANIFEST, {
        "phase": PHASE,
        "protocol_digest": PHASE_DIGEST,
        "at": now_iso(),
        "gates": STATE.get("gate_results") or {},
        "arms": sorted(ledger_hashes),
        "frozen_checkpoints_per_arm": EXPECTED_FROZEN,
        "frozen_sources": checked_sources,
        "ledgers": ledger_hashes,
        "approved_failure": {
            "run_id": APPROVED_RUN_ID,
            "ledger_sha256": (ledger_hashes.get(APPROVED_RUN_ID) or {}).get("sha256"),
            "approved_rows_present": (ledger_hashes.get(APPROVED_RUN_ID) or {})
            .get("approved_failed"),
            "sha256": APPROVED_FAILURE_SHA256,
        },
        "phase_recorded_tokens": recorded,
        "conservative_unknown_reserve": reserve,
        "recorded_plus_reserve": recorded + reserve,
        "accounting_exceptions": accounting_exceptions(),
    })
    emit("frozen", phase_recorded_tokens=recorded, conservative_unknown_reserve=reserve)

    for name in ("mof-unseal", "mof-report"):
        argv = [sys.executable, "-m", "dream_rsi.cli", name, "--phase", PHASE]
        if run_experiment_cli(argv, name) != 0:
            return blocker(f"{name} did not exit cleanly")
        emit("cli-complete", cli=name)

    annotate_unsealed()
    emit("complete", current_arm=None)
    return 0


def recovery_section() -> str:
    """The vault-loss disclosure, from the recovery record when one exists."""
    record = read_json(RECOVERY_RECORD)
    if record is None:
        return ""
    lines = [
        "\n## Vault recovery (2026-10-14)\n",
        "- The search controller was killed by an external event on 2026-10-14 "
        "(last write 16:31:45 +08:00); the vault keeper died with it, so the "
        "sealed files were not restored by the normal path.",
        f"- Recovery method: the protected set was regenerated and verified, not "
        f"restored from the keeper's memory. {record.get('method', '')}",
        f"- Restored protected set: {record.get('files')} files, "
        f"{record.get('bytes')} bytes"
        + (f"; the pre-incident set was {record.get('incident_files')} files, "
           f"{record.get('incident_bytes')} bytes" if record.get("incident_files") else "")
        + ".",
    ]
    for item in record.get("unrecoverable") or []:
        lines.append(f"- Not restorable: {item}")
    for item in record.get("checks") or []:
        lines.append(f"- Check: {item}")
    lines.append(f"- Record: `{RECOVERY_RECORD.name}` beside this file; every hash "
                 f"above was recomputed, not copied.")
    return "\n".join(lines) + "\n"


def annotate_unsealed() -> None:
    """Attach the accounting caveat to the unsealed outputs. Values untouched."""
    exceptions = accounting_exceptions()
    UNSEALED.mkdir(parents=True, exist_ok=True)

    summary = UNSEALED / "summary.md"
    with summary.open("a", encoding="utf-8") as handle:
        handle.write(recovery_section())
        handle.write(
            "\n## Parameter deviation\n\n"
            "The executed replay horizon was K2=6 for every arm, matching the existing "
            "runner and initial B. The original design tables stated K2=12; that value "
            "was not implemented. The protocol digest did not include K2. This run "
            "has an explicit design deviation; no parameter was changed during "
            "continuation.\n"
            "\n## Accounting exceptions\n\n"
            f"- `{APPROVED_RUN_ID}` has one aborted output-limit discovery call "
            "(seq 60, `PI_OUTPUT_LIMIT`). The billed tokens for it are unknown.\n"
            "- Its reservation upper bound, 23312 tokens, is carried as a separate "
            "conservative reservation. It is not measured tokens and not billed tokens, "
            "and it is never added to recorded usage.\n"
            "- The recorded-token curve positions were not rewritten, and no ledger row "
            "was removed or edited; the exception is an annotation over unchanged records.\n"
            "- Exploratory reading: 2 paired blocks, one search each, on the pilot pool "
            "that was already opened. No C arm is registered, and no confirmatory verdict "
            "is claimed.\n")

    report = UNSEALED / "report.json"
    if report.is_file():
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except ValueError:
            data = None
        if isinstance(data, dict):
            data["accounting_exceptions"] = exceptions
            recovery = read_json(RECOVERY_RECORD)
            if recovery is not None:
                data["vault_recovery"] = recovery
            data["execution_parameter_deviations"] = [{
                "parameter": "K2", "planned": 12, "executed": K2,
                "scope": "all ab arms and existing runner",
                "protocol_digest_covers_parameter": False,
            }]
            report.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n",
                              encoding="utf-8")

    csv_path = UNSEALED / "accounting-exceptions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "call_id", "seq", "role", "error_type",
                         "charged_tokens", "usage_reported", "conservative_reserve",
                         "billed_tokens", "recorded_tokens_rewritten", "note"])
        for item in exceptions:
            writer.writerow([item["run_id"], item["call_id"], item["seq"], item["role"],
                             item["error_type"], item["charged_tokens"],
                             item["usage_reported"], item["reservation_upper_bound"],
                             item["billed_tokens"], "no", item["note"]])


# ------------------------------------------------------------------ check only


def recorded_pi_version(records: dict) -> str | None:
    """The Pi version the earlier records carry, if they carry one at all."""
    for row in records.values():
        for holder in (row, row.get("plan"), row.get("budget"), row.get("config")):
            if isinstance(holder, dict) and isinstance(holder.get("pi_version"), str):
                return holder["pi_version"]
    return None


def check_only() -> int:
    """Report what would run and what can be verified without touching anything."""
    plans = block_plans(PHASE)
    problems, records, rows = phase_preconditions(plans)
    deferred: list[str] = []

    # The digest is over the frozen protocol, which includes the Pi version. This
    # mode must not invoke Pi, so a recorded version is used when one exists and
    # the comparison is otherwise deferred -- execution re-checks it, with the
    # live version, before every arm's model call.
    pi_version = recorded_pi_version(records)
    protocol = Protocol(phase=PHASE, prepared_root=str(_prepared(REPO)), pi_version=pi_version)
    if protocol.digest == PHASE_DIGEST:
        digest_state = "frozen digest reproduced"
    elif pi_version is None:
        deferred.append("protocol digest not validated: no recorded Pi version, and this "
                        "mode does not call Pi. Execution re-checks it before every arm.")
        digest_state = "digest deferred"
    else:
        problems.append("the protocol digest does not match the frozen digest")
        digest_state = "digest mismatch"

    arms: list[dict] = []
    for row in rows:
        entry: dict = {"run_id": row["run_id"], "block": row["block_id"],
                       "method": row["method"], "state": row["state"],
                       "resume_approved": bool(row.get("resume_approved"))}
        if row["state"] == "partial":
            # No block record exists yet -- that is what "partial" means -- so
            # the read-only state summary stands in for it, alongside the
            # approval that is required before it may be resumed in place.
            entry["partial_state"] = partial_state_facts(row["run_id"])
            entry["resume_approval_authority"] = row.get("resume_approval")
            arms.append(entry)
            continue
        if row["state"] != "pending":
            arm_problems, facts = arm_report(row["run_id"], row["method"])
            ledger = facts.get("ledger_facts") or {}
            entry["problems"] = arm_problems
            entry["worlds"] = f"{facts.get('sealed')}/{facts.get('worlds')} sealed"
            entry["ledger"] = {key: ledger.get(key) for key in
                               ("failed", "approved_failed", "refused", "unreported",
                                "unreadable", "outstanding", "conservative_unknown_reserve")}
            entry["frozen_sources"] = len(facts.get("sources") or [])
            problems += arm_problems
        arms.append(entry)

    protected_problems, protected_facts = protected_set_report(hash_archive=False)
    problems += protected_problems
    deferred.append("the pinned archive's digest is not computed in this mode; "
                    "execution verifies it before any arm")

    report = {
        "mode": "check-only",
        "phase": PHASE,
        "frozen_digest": PHASE_DIGEST,
        "digest_state": digest_state,
        "pi_version_recorded": pi_version,
        "pause_marker": PAUSE_MARKER.exists(),
        "lock_present": LOCK_PATH.exists(),
        "arm_plan": arms,
        "arms_that_would_run": [a["run_id"] for a in arms
                                if a["state"] in ("pending", "partial")],
        "resume_approvals_recorded": sorted(resume_approvals()),
        "protected_set": protected_facts,
        "problems": problems,
        "deferred": deferred,
        "wrote": "nothing", "model_calls": 0, "pi_calls": 0, "lock_acquired": False,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print("CHECK: " + ("blocked" if problems
                       else "ready (with deferred items)" if deferred else "ready"), flush=True)
    return 2 if problems else 0


# ------------------------------------------------------------------------ main


def main(argv: list[str]) -> int:
    known = {"--check-only", "--execute"}
    unknown = [arg for arg in argv if arg not in known]
    usage = f"usage: {Path(__file__).name} --check-only | --execute"
    if unknown or len(set(argv)) != len(argv) or ("--check-only" in argv) == ("--execute" in argv):
        print(usage, flush=True)
        return 2
    return check_only() if "--check-only" in argv else execute()


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except KeyboardInterrupt:
        print("CONTROLLER INTERRUPTED", flush=True)
        sys.exit(2)
    except Exception as exc:
        print(f"CONTROLLER ERROR: {type(exc).__name__}", flush=True)
        sys.exit(2)
