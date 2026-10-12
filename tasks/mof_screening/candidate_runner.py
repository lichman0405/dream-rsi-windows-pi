#!/usr/bin/env python3
"""Untrusted-code process: import ``candidate.py`` and call ``rank`` exactly once.

This file is deliberately the only place a discovery candidate is ever
executed. It is started by the fixed grader as

    python -I candidate_runner.py <scratch_dir>

from a working directory that holds nothing but the candidate and its inputs,
and it reports through ``<scratch_dir>/result.json`` and nowhere else. It never
prints the outcome, never raises, and never decides a score: the grader owns
scoring and receives only the ordered list of material ids.

What it does not do is as important as what it does. There is no network client,
no subprocess, and no filesystem enumeration in here, so a candidate has nothing
to borrow. The candidate module is loaded from an explicit path rather than the
import system, so it cannot shadow a module the runner itself uses.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

MAX_ERROR_CHARS = 400


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("mof_candidate_under_test", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build an import spec for {path.name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(scratch: Path) -> dict[str, object]:
    call = json.loads((scratch / "call.json").read_text(encoding="utf-8"))
    import pandas as pd

    train_X = pd.read_csv(scratch / "train_features.csv")
    train_y = pd.read_csv(scratch / "train_targets.csv")["target_mol_kg"]
    pool_X = pd.read_csv(scratch / "pool_features.csv")

    module = _load_candidate(scratch / "candidate.py")
    entry = getattr(module, "rank", None)
    if not callable(entry):
        return {"ok": False, "error_type": "MISSING_ENTRY_POINT",
                "error": "candidate.py must define a callable `rank`"}

    started = time.monotonic()
    try:
        result = entry(train_X, train_y, pool_X, k=int(call["k"]), seed=int(call["seed"]))
    except Exception as exc:  # noqa: BLE001 - reported as a contract failure
        return {
            "ok": False, "error_type": "CANDIDATE_RAISED",
            "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS],
            "traceback": traceback.format_exc(limit=4)[-1200:],
            "duration_s": round(time.monotonic() - started, 4),
        }

    if isinstance(result, (str, bytes)) or not isinstance(result, (list, tuple)):
        return {"ok": False, "error_type": "BAD_RETURN_TYPE",
                "error": f"rank() returned {type(result).__name__}, expected list[str]",
                "duration_s": round(time.monotonic() - started, 4)}
    return {
        "ok": True,
        "ids": [value if isinstance(value, str) else None for value in result],
        "duration_s": round(time.monotonic() - started, 4),
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: candidate_runner.py <scratch_dir>", file=sys.stderr)
        return 2
    scratch = Path(argv[1]).resolve()
    result_path = scratch / "result.json"
    try:
        payload = run(scratch)
    except BaseException as exc:  # noqa: BLE001 - e.g. a candidate that kills the interpreter
        payload = {"ok": False, "error_type": "RUNNER_FAILED",
                   "error": f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]}
    try:
        _write(result_path, payload)
    except OSError:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
