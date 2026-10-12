#!/usr/bin/env python3
"""Child process that takes protected files off disk and puts them back.

The problem this solves: the search phase runs model-written programs, and on a
single Windows user account every file is readable by every process of that
user. A directory boundary is therefore not a boundary, and Windows offers no
process-level confinement here either -- creating a Low-integrity or otherwise
restricted child needs ``SeAssignPrimaryTokenPrivilege`` or
``SeImpersonatePrivilege``, and this account holds neither (see
``docs/SECURITY.md``).

What *is* enforceable without any privilege is absence. A file that has been
removed from the filesystem cannot be read by anybody. This keeper is the
process that makes that safe: it receives the protected files over a private
pipe, the parent then deletes them, and the keeper writes them back -- with
their original SHA-256 verified -- the moment the parent's pipe closes, whether
that parent exited normally, crashed, or was killed.

The keeper deliberately holds no key and writes nothing secret anywhere. Its
whole content is the bytes it was handed, so its own crash loses data that the
pinned preparation script can regenerate (see ``docs/experiments``).

Protocol on stdin, all little-endian and length-prefixed:

1. one JSON header line, newline-terminated::

       {"version": 1, "files": [{"path": ..., "sha256": ..., "size": ...}, ...]}

2. for each entry, exactly ``size`` raw bytes, in header order.

The keeper replies ``{"ok": true, ...}\\n`` on stdout once every byte has
arrived and been removed from disk, then blocks on stdin. On EOF it restores
and replies once more.

Invoked as::

    python -I -m dream_rsi.mof.vault_keeper <manifest_json_path>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _reply(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def _read_exactly(stream, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining > 0:
        chunk = stream.read(min(remaining, 1 << 20))
        if not chunk:
            raise EOFError(f"stdin closed after {count - remaining} of {count} bytes")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _restore(entries: list[dict[str, object]]) -> list[str]:
    """Rewrite every protected file, verifying the digest it was taken with."""
    problems: list[str] = []
    for entry in entries:
        path = Path(str(entry["path"]))
        data: bytes = entry["data"]  # type: ignore[assignment]
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"]:
            problems.append(f"{path}: in-memory sha256 {digest} != recorded {entry['sha256']}")
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write through a sibling temp file so a crash mid-restore cannot
            # leave a half-written file that then fails its own digest check.
            staging = path.with_name(path.name + ".vault-restore")
            staging.write_bytes(data)
            os.replace(staging, path)
        except OSError as exc:
            problems.append(f"{path}: {type(exc).__name__}: {exc}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: vault_keeper.py <manifest_json_path>", file=sys.stderr)
        return 2
    manifest_path = Path(argv[1])
    stream = sys.stdin.buffer
    entries: list[dict[str, object]] = []
    try:
        header_line = stream.readline()
        if not header_line:
            _reply({"ok": False, "error": "no header on stdin"})
            return 3
        header = json.loads(header_line.decode("utf-8"))
        if int(header.get("version", 0)) != 1:
            _reply({"ok": False, "error": f"unsupported protocol version {header.get('version')!r}"})
            return 3
        for spec in header.get("files", []):
            size = int(spec["size"])
            data = _read_exactly(stream, size)
            digest = hashlib.sha256(data).hexdigest()
            if digest != spec["sha256"]:
                _reply({"ok": False, "error": f"{spec['path']}: streamed bytes do not match sha256"})
                return 3
            entries.append({"path": str(spec["path"]), "sha256": str(spec["sha256"]), "data": data})
            _remove(Path(str(spec["path"])))
    except Exception as exc:  # noqa: BLE001 - the keeper must always answer
        _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 3

    _reply({"ok": True, "held_files": len(entries),
            "bytes": sum(len(e["data"]) for e in entries)})  # type: ignore[arg-type]

    # Block until the parent's pipe closes. Anything else it wants to say first
    # is ignored on purpose: no message from the parent can talk the keeper out
    # of restoring.
    while True:
        chunk = stream.read(4096)
        if not chunk:
            break

    problems = _restore(entries)
    record = {
        "restored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "files": len(entries),
        "problems": problems,
    }
    try:
        with manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps({"event": "vault.restored", **record}, sort_keys=True) + "\n")
    except OSError:
        pass
    _reply({"ok": not problems, **record})
    return 0 if not problems else 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
