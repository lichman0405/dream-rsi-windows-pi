"""Parent side of the search-phase vault: take labels off disk while models run.

Why this exists, in one sentence: on this machine a directory boundary is not a
permission boundary, so the only enforceable way to keep the private labels away
from model-written code is for the labels not to be on the filesystem while that
code runs.

The mechanism is in :mod:`dream_rsi.mof.vault_keeper`: the files are streamed to
a child process, deleted from disk, and rewritten -- digest-verified -- when the
child's stdin closes, which happens on a clean exit, a crash, or a kill.
:class:`FileVault` is the parent half, plus the honesty checks that make the
claim testable:

* :meth:`FileVault.verify_absent` opens every protected path and fails loudly if
  anything is still readable. A caller runs it after :meth:`seal` and records
  the answer; "we sealed it" is not evidence.
* every seal/unseal is appended to a manifest, so the window during which the
  data was off disk is a recorded interval rather than a claim.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class VaultEntry:
    path: Path
    sha256: str
    size: int


@dataclass
class FileVault:
    """Holds a set of files off disk for the duration of a search phase."""

    manifest_path: Path
    keeper_module: str = "dream_rsi.mof.vault_keeper"
    python_executable: str = field(default_factory=lambda: sys.executable)
    _proc: subprocess.Popen[bytes] | None = None
    _entries: list[VaultEntry] = field(default_factory=list)
    _log: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ state

    @property
    def sealed(self) -> bool:
        return self._proc is not None

    @property
    def entries(self) -> tuple[VaultEntry, ...]:
        return tuple(self._entries)

    @property
    def log(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._log)

    # ------------------------------------------------------------------ seal

    def seal(self, paths: Iterable[Path]) -> dict[str, Any]:
        """Stream every path to the keeper, then delete it from disk.

        A path may be a file or a directory; directories are expanded to their
        contained files. Symlinks and reparse points are refused rather than
        followed: deleting the target of a link would take away something the
        caller never named.
        """
        if self.sealed:
            raise RuntimeError("this vault is already sealed")
        files: list[Path] = []
        for raw in paths:
            path = Path(raw)
            if path.is_symlink():
                raise ValueError(f"refusing to vault a symlink: {path}")
            if path.is_dir():
                files.extend(sorted(p for p in path.rglob("*") if p.is_file()))
            elif path.is_file():
                files.append(path)

        payloads: list[tuple[Path, bytes]] = []
        for path in files:
            payloads.append((path, path.read_bytes()))

        header = {
            "version": 1,
            "files": [
                {"path": str(path.resolve()), "sha256": hashlib.sha256(data).hexdigest(),
                 "size": len(data)}
                for path, data in payloads
            ],
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            [self.python_executable, "-I", "-m", self.keeper_module, str(self.manifest_path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            shell=False, cwd=str(self.manifest_path.parent),
        )
        assert proc.stdin is not None and proc.stdout is not None
        started = time.monotonic()
        proc.stdin.write((json.dumps(header) + "\n").encode("utf-8"))
        for _path, data in payloads:
            proc.stdin.write(data)
        proc.stdin.flush()
        reply = json.loads(proc.stdout.readline().decode("utf-8") or "{}")
        if not reply.get("ok"):
            proc.kill()
            raise RuntimeError(f"vault keeper refused the payload: {reply.get('error')}")
        self._proc = proc
        self._entries = [
            VaultEntry(Path(spec["path"]), spec["sha256"], spec["size"])
            for spec in header["files"]
        ]
        record = {
            "event": "vault.sealed",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "files": len(self._entries),
            "bytes": sum(e.size for e in self._entries),
            "seconds": round(time.monotonic() - started, 3),
            "keeper_pid": proc.pid,
        }
        self._append(record)
        return record

    # ------------------------------------------------------------------ check

    def verify_absent(self) -> dict[str, Any]:
        """Prove the protected paths are gone. Reads are attempted, not assumed."""
        readable: list[str] = []
        for entry in self._entries:
            try:
                entry.path.read_bytes()
            except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
                continue
            readable.append(str(entry.path))
        record = {
            "event": "vault.verified_absent",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checked": len(self._entries),
            "still_readable": readable,
            "ok": not readable,
        }
        self._append(record)
        return record

    # ------------------------------------------------------------------ release

    def release(self) -> dict[str, Any]:
        """Close the keeper's stdin and wait for it to restore every file."""
        proc = self._proc
        if proc is None:
            return {"event": "vault.release_skipped", "reason": "not sealed"}
        started = time.monotonic()
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            assert proc.stdout is not None
            reply = json.loads(proc.stdout.readline().decode("utf-8") or "{}")
            proc.wait(timeout=120)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        finally:
            self._proc = None
        record = {
            "event": "vault.released",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "seconds": round(time.monotonic() - started, 3),
            **{k: v for k, v in reply.items() if k != "event"},
        }
        self._append(record)
        return record

    def verify_restored(self) -> dict[str, Any]:
        """Every sealed file is back, byte-identical."""
        mismatched: list[str] = []
        missing: list[str] = []
        for entry in self._entries:
            try:
                data = entry.path.read_bytes()
            except OSError:
                missing.append(str(entry.path))
                continue
            if hashlib.sha256(data).hexdigest() != entry.sha256:
                mismatched.append(str(entry.path))
        record = {
            "event": "vault.verified_restored",
            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "checked": len(self._entries),
            "missing": missing,
            "sha256_mismatch": mismatched,
            "ok": not missing and not mismatched,
        }
        self._append(record)
        return record

    def __enter__(self) -> FileVault:
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.sealed:
            self.release()

    # ------------------------------------------------------------------ io

    def _append(self, record: dict[str, Any]) -> None:
        self._log.append(record)
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with self.manifest_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        except OSError:
            pass


def load_sealed(entries: Iterable[VaultEntry]) -> dict[str, bytes]:
    """Read a set of files that the caller is about to vault, keyed by path."""
    return {str(entry.path): entry.path.read_bytes() for entry in entries}


__all__ = ["FileVault", "VaultEntry", "load_sealed"]
