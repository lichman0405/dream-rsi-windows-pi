"""The fixed MOF grader: rank a candidate's recommendations against held labels.

One evaluation is: screen the candidate's source, hand it the training set and
the pool *features* in a scratch directory, run it once in its own process, and
score the ids it returns against labels this process holds in memory and never
writes anywhere.

    Q(A)  = mean true uptake over the K recommended materials        (mol/kg)
    Q*(D) = mean true uptake of the pool's true best K               (private)
    S     = Q(A) / Q*(D)

Correctness properties this module is responsible for, each of which is also an
acceptance test:

* ``S`` is computed here, from labels loaded here. Nothing in the candidate's
  process can see, set, or approximate it.
* The pool's labels and ``Q*`` never appear in ``public_detail``; only the score
  (which the loop already publishes), a success/failure status and a contract
  error class do.
* A program that runs and ranks badly keeps its real, low score. A program that
  breaks the contract -- missing entry point, wrong type, wrong count,
  duplicates, ids outside the pool, exception, timeout -- is ``FAILED`` with
  ``S = 0``. The two are never conflated, and neither is turned into ``ERROR``:
  ``ERROR`` is reserved for a grader that could not run.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..domain.models import EvaluatorStatus
from ..evaluator.protocol import EvaluationResult
from ..runtime.winjob import WindowsJob, terminate_tree

#: The only third-party packages a candidate may use, as stated in TASK.md.
ALLOWED_THIRD_PARTY: frozenset[str] = frozenset({
    "numpy", "pandas", "sklearn", "scipy", "joblib",
})

#: Standard-library modules a *ranking program* has a reason to import: pure
#: computation, typing and small data plumbing. Everything not named here and not
#: in :data:`ALLOWED_THIRD_PARTY` is refused, so the rule a candidate is given
#: ("these libraries and no others") is the rule that is enforced, rather than a
#: denylist that silently permits whatever it forgot to name.
ALLOWED_STDLIB: frozenset[str] = frozenset({
    "__future__", "abc", "array", "bisect", "collections", "contextlib", "copy",
    "dataclasses", "decimal", "enum", "fractions", "functools", "heapq",
    "itertools", "math", "numbers", "operator", "random", "re", "statistics",
    "string", "types", "typing", "unittest", "warnings",
})

#: Capability modules that must never load, whatever route is taken to them.
#: The static screen below cannot see ``__import__("sock" + "et")``; the runner's
#: runtime guard can, because it sits on the import machinery itself. This list
#: is deliberately limited to what a ranking program cannot need and an attacker
#: would: the network and the ability to reach into another process.
CAPABILITY_MODULES: frozenset[str] = frozenset({
    "socket", "ssl", "urllib", "http", "requests", "httpx", "aiohttp",
    "ftplib", "smtplib", "telnetlib", "poplib", "imaplib", "xmlrpc",
    "asyncio", "socketserver", "webbrowser",
    "subprocess", "ctypes", "_ctypes", "multiprocessing", "pty", "winreg",
    "msvcrt", "_winapi", "nt", "posix",
})

#: Kept as the name the task text and earlier code use; the enforced rule is now
#: the pair of allowlists above plus the runtime guard.
FORBIDDEN_IMPORTS: frozenset[str] = CAPABILITY_MODULES | frozenset({
    "glob", "tempfile", "shutil", "pickle", "marshal", "shelve", "sqlite3",
    "distutils", "setuptools", "pip", "os", "sys", "io", "pathlib", "importlib",
    "builtins", "runpy", "pkgutil", "inspect", "code", "codeop", "site",
})

#: Every module name a candidate's own source may import, at top level.
ALLOWED_IMPORTS: frozenset[str] = ALLOWED_STDLIB | ALLOWED_THIRD_PARTY

ID_COLUMN = "material_id"
TARGET_COLUMN = "target_mol_kg"
K_DEFAULT = 10


def _digest(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def screen_candidate_source(source: str) -> tuple[str, ...]:
    """Return the imports a candidate's source may not make, sorted.

    This is an **allowlist**, not a denylist. The task text tells a candidate it
    may use one standard library plus NumPy, pandas and scikit-learn, so the
    check is "is this one of those", and a module nobody thought to forbid is
    refused rather than permitted by omission.

    A source that does not parse is reported as ``SYNTAX_ERROR`` rather than
    raising: an unparseable candidate is a failed candidate, and the grader has
    to be able to say so.

    The screen reads the syntax tree, so it cannot see ``__import__("sock" +
    "et")``. That route is closed at run time instead, by the import guard in
    ``tasks/mof_screening/candidate_runner.py``; the two are complementary and
    neither is asked to do the other's job.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ("SYNTAX_ERROR",)
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    found.add(root)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import: nothing legitimate resolves here
                found.add(".")
                continue
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_IMPORTS:
                found.add(root)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"__import__", "eval", "exec", "compile", "open"}:
                found.add(node.func.id)
    return tuple(sorted(found))


def score_ranking(
    ranking: Sequence[str], labels: Mapping[str, float], *, k: int,
) -> tuple[float, float, float, list[float], list[float]]:
    """``(Q, Q*, S, recommended_uptakes, top_k_uptakes)`` for one ranking."""
    if len(ranking) != k:
        raise ValueError(f"ranking must hold exactly {k} ids, got {len(ranking)}")
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking holds duplicate ids")
    missing = [material for material in ranking if material not in labels]
    if missing:
        raise ValueError(f"ranking holds ids outside the pool: {missing[:3]}")
    recommended = [float(labels[material]) for material in ranking]
    # Ties on the true value are broken by material id, so the private reference
    # is a function of the pool alone and never of a candidate's ordering.
    best = sorted(labels.items(), key=lambda item: (-float(item[1]), item[0]))[:k]
    top = [float(value) for _material, value in best]
    q_star = sum(top) / k
    if not q_star > 0.0:
        raise ValueError(f"pool reference Q*={q_star!r} is not positive; task is unusable")
    q = sum(recommended) / k
    return q, q_star, q / q_star, recommended, top


@dataclass(frozen=True)
class MofPool:
    """One gradable pool, with its training set and its held-out labels.

    Two kinds of pool exist and they differ only in which files they name: a
    **development pool** (the split's ``dev_features.csv`` against its
    ``dev_targets.private.csv``) which is the search-time signal, and the
    **final test pool**, which is the same training set scored against
    ``final_test_features.private.csv`` and never touched until every candidate
    is frozen.
    """

    label: str
    train_features: Path
    train_targets: Path
    pool_features: Path
    private_targets: Path
    payload_seed: int = 20261012

    @classmethod
    def development(cls, prepared_root: Path, seed: int) -> MofPool:
        agent = Path(prepared_root) / "splits" / str(seed) / "agent"
        return cls(
            label=f"split-{seed}-development",
            train_features=agent / "train_features.csv",
            train_targets=agent / "train_targets.csv",
            pool_features=agent / "dev_features.csv",
            private_targets=(Path(prepared_root) / "splits" / str(seed)
                             / "evaluator_private" / "dev_targets.private.csv"),
            payload_seed=20261012 + seed,
        )

    @classmethod
    def final_test(cls, prepared_root: Path, *, split_seed: int) -> MofPool:
        """The sealed pool. ``split_seed`` only selects which training set it pairs with."""
        agent = Path(prepared_root) / "splits" / str(split_seed) / "agent"
        private = Path(prepared_root) / "private"
        return cls(
            label=f"split-{split_seed}-final-test",
            train_features=agent / "train_features.csv",
            train_targets=agent / "train_targets.csv",
            pool_features=private / "final_test_features.private.csv",
            private_targets=private / "final_test_targets.private.csv",
            payload_seed=20261012 + split_seed,
        )

    def load_labels(self) -> dict[str, float]:
        """Pool labels, read once and kept in this process's memory."""
        labels: dict[str, float] = {}
        with self.private_targets.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                labels[str(row[ID_COLUMN])] = float(row[TARGET_COLUMN])
        return labels

    def data_digest(self) -> str:
        material = {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("train_features", self.train_features),
                ("train_targets", self.train_targets),
                ("pool_features", self.pool_features),
                ("private_targets", self.private_targets),
            )
        }
        return _digest({"label": self.label, "files": material})


class MofScreeningEvaluator:
    """Fixed grader for ``mof_screening``. Holds the pool labels in memory."""

    def __init__(
        self,
        *,
        pool: MofPool,
        labels: Mapping[str, float],
        runner_script: Path,
        workdir_root: Path,
        k: int = K_DEFAULT,
        timeout_s: float = 120.0,
        python_executable: str | None = None,
        memory_limit_mb: int | None = 2048,
        pool_digest: str | None = None,
    ) -> None:
        self.pool = pool
        self.labels: dict[str, float] = dict(labels)
        self._pool_digest = pool_digest
        self.runner_script = Path(runner_script).resolve()
        self.workdir_root = Path(workdir_root).resolve()
        self.k = int(k)
        self.timeout_s = float(timeout_s)
        self.python_executable = python_executable or sys.executable
        self.memory_limit_mb = memory_limit_mb
        if not self.runner_script.is_file():
            raise FileNotFoundError(f"candidate runner not found: {self.runner_script}")

        # The private targets are deliberately absent while the search phase is
        # running: the vault has taken them off disk and the caller is holding
        # the labels in memory. Their *digest* is pinned before sealing instead,
        # so a grader built mid-search still names the data it is grading
        # against rather than whichever files happen to be present.
        required = [
            name for name, path in (
                ("train_features", pool.train_features),
                ("train_targets", pool.train_targets),
                ("pool_features", pool.pool_features),
            ) if not path.is_file()
        ]
        if required:
            raise FileNotFoundError(f"pool {pool.label} is missing {', '.join(required)}")
        if not pool.private_targets.is_file() and pool_digest is None:
            raise FileNotFoundError(
                f"pool {pool.label}: private targets are not on disk and no digest was "
                "pinned before they were sealed"
            )
        self._pool_ids = self._read_pool_ids()
        uncovered = [material for material in self._pool_ids if material not in self.labels]
        if uncovered:
            raise ValueError(
                f"pool {pool.label}: {len(uncovered)} pool ids have no private label"
            )
        if len(self.labels) != len(self._pool_ids):
            raise ValueError(
                f"pool {pool.label}: {len(self.labels)} labels for "
                f"{len(self._pool_ids)} pool rows"
            )

    # ------------------------------------------------------------------ identity

    @property
    def spec_digest(self) -> str:
        """Pins the grader: its own source, the runner, the split, K and the limits."""
        material = {
            "kind": "mof_screening_fixed",
            "grader_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "runner_sha256": hashlib.sha256(self.runner_script.read_bytes()).hexdigest(),
            "pool_data_digest": self._pool_digest or self.pool.data_digest(),
            "k": self.k,
            "timeout_s": self.timeout_s,
            "forbidden_imports": sorted(FORBIDDEN_IMPORTS),
        }
        return _digest(material)

    def _read_pool_ids(self) -> list[str]:
        with self.pool.pool_features.open("r", newline="", encoding="utf-8-sig") as handle:
            return [str(row[ID_COLUMN]) for row in csv.DictReader(handle)]

    # ------------------------------------------------------------------ grading

    def evaluate(self, candidate_dir: Path, *, invocation_id: str) -> EvaluationResult:
        started = time.monotonic()
        candidate_dir = Path(candidate_dir).resolve()
        candidate_file = candidate_dir / "candidate.py"
        workdir = self.workdir_root / invocation_id
        scratch = workdir / "run"
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True, exist_ok=True)

        base: dict[str, Any] = {"harness": "mof_screening", "pool": self.pool.label,
                                "invocation_id": invocation_id}
        if not candidate_file.is_file():
            return self._failure(base, started, "MISSING_CANDIDATE_FILE",
                                 "workspace has no candidate.py", workdir)

        try:
            source = candidate_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return self._failure(base, started, "UNREADABLE_CANDIDATE",
                                 f"{type(exc).__name__}: {exc}", workdir)

        violations = screen_candidate_source(source)
        if violations:
            return self._failure(
                {**base, "contract": list(violations)}, started, "FORBIDDEN_IMPORT",
                f"candidate uses forbidden modules: {', '.join(violations)}", workdir,
            )

        shutil.copyfile(candidate_file, scratch / "candidate.py")
        shutil.copyfile(self.pool.train_features, scratch / "train_features.csv")
        shutil.copyfile(self.pool.train_targets, scratch / "train_targets.csv")
        shutil.copyfile(self.pool.pool_features, scratch / "pool_features.csv")
        (scratch / "call.json").write_text(
            json.dumps({"k": self.k, "seed": self.pool.payload_seed}), encoding="utf-8"
        )

        timed_out = False
        argv = [self.python_executable, "-I", str(self.runner_script), str(scratch)]
        with WindowsJob(memory_limit_mb=self.memory_limit_mb, active_process_limit=4) as job:
            proc = subprocess.Popen(
                argv, cwd=str(scratch), env=_runner_env(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            job.assign(proc)
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                terminate_tree(proc, job)
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except (subprocess.TimeoutExpired, ValueError, OSError):
                    stdout, stderr = b"", b""

        detail = {
            **base,
            "candidate_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "timed_out": timed_out,
            "runner_returncode": proc.returncode,
            "runner_stdout_bytes": len(stdout),
            "runner_stderr_tail": stderr[-1500:].decode("utf-8", "replace"),
        }
        if timed_out:
            return self._failure(detail, started, "TIMEOUT",
                                 f"candidate exceeded {self.timeout_s:g}s", workdir)

        result_path = scratch / "result.json"
        if not result_path.is_file():
            return self._failure(detail, started, "NO_RESULT_FILE",
                                 f"runner exited {proc.returncode} without a result", workdir)
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return self._failure(detail, started, "UNREADABLE_RESULT",
                                 f"{type(exc).__name__}: {exc}", workdir)
        if not payload.get("ok"):
            return self._failure({**detail, "runner_payload": payload}, started,
                                 str(payload.get("error_type") or "CANDIDATE_FAILED"),
                                 str(payload.get("error") or "")[:400], workdir)

        ids = payload.get("ids")
        problem = self._contract_problem(ids)
        if problem is not None:
            error_type, message = problem
            return self._failure({**detail, "ranking": ids}, started, error_type, message, workdir)

        ranking = [str(value) for value in ids]  # type: ignore[union-attr]
        q, q_star, s, recommended, top = score_ranking(ranking, self.labels, k=self.k)
        duration = time.monotonic() - started
        private = {
            **detail,
            "ranking": ranking,
            "Q": q, "Q_star": q_star, "S": s,
            "recommended_uptakes": recommended, "top_k_uptakes": top,
            "candidate_duration_s": payload.get("duration_s"),
        }
        status = EvaluatorStatus.SUCCESS if s > 0.0 else EvaluatorStatus.FAILED
        return EvaluationResult(
            status=status,
            score=s,
            duration_s=duration,
            result_digest=_digest({"pool": self.pool.label, "ranking": ranking,
                                   "S": s, "Q": q, "Q_star": q_star}),
            detail=private,
            error_type=None if status is EvaluatorStatus.SUCCESS else "ZERO_SCORE",
            # The pool's labels, its true top K and Q* are NOT here. Only what the
            # design says a run may publish.
            public_detail={
                "contract_status": "OK",
                "k": self.k,
                "evaluated_materials": len(self._pool_ids),
            },
        )

    # ------------------------------------------------------------------ helpers

    def _contract_problem(self, ids: Any) -> tuple[str, str] | None:
        if not isinstance(ids, list):
            return "BAD_RETURN_TYPE", f"runner reported ids of type {type(ids).__name__}"
        if any(value is None for value in ids):
            return "NON_STRING_ID", "rank() returned an entry that is not a string"
        if len(ids) != self.k:
            return "WRONG_COUNT", f"rank() returned {len(ids)} ids, expected {self.k}"
        if len(set(ids)) != len(ids):
            return "DUPLICATE_ID", "rank() returned the same material more than once"
        known = set(self._pool_ids)
        outside = [value for value in ids if value not in known]
        if outside:
            return "OUT_OF_POOL_ID", f"rank() returned {len(outside)} ids outside the pool"
        return None

    def _failure(
        self, detail: Mapping[str, Any], started: float, error_type: str,
        message: str, workdir: Path,
    ) -> EvaluationResult:
        duration = time.monotonic() - started
        payload = {**detail, "error_type": error_type, "error": message}
        return EvaluationResult(
            status=EvaluatorStatus.FAILED,
            score=0.0,
            duration_s=duration,
            result_digest=_digest(payload),
            detail=payload,
            error_type=error_type,
            public_detail={"contract_status": error_type, "k": self.k},
        )


def _runner_env() -> dict[str, str]:
    """A minimal child environment: no credentials, no inherited surprises."""
    keep = ("PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC",
            "TEMP", "TMP", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
            "PYTHONIOENCODING")
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def material_references(
    *, pool_features: Path, labels: Mapping[str, float], train_features: Path,
    train_targets: Path, k: int = K_DEFAULT, seeds: Iterable[int] = range(1, 101),
) -> dict[str, Any]:
    """The four material-screening references the design asks for.

    These say how hard the task is; they are *not* Pi comparisons with a matched
    model-call budget, and the report has to keep those two claims apart. The
    pore-volume reference named in the design cannot be computed here: the
    pinned CRAFTED 2.0.1 descriptor frame contains the RAC family only, and the
    design forbids pulling descriptors from outside the pinned version.
    """
    import numpy as np
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Ridge

    pool = pd.read_csv(pool_features)
    train = pd.read_csv(train_features)
    targets = pd.read_csv(train_targets)
    pool_ids = [str(value) for value in pool[ID_COLUMN].tolist()]
    y = targets[TARGET_COLUMN].to_numpy(dtype=float)

    columns = [name for name in train.columns if name != ID_COLUMN]
    train_x = train[columns]
    pool_x = pool.reindex(columns=columns)
    medians = train_x.median(axis=0)
    train_filled = train_x.fillna(medians)
    pool_filled = pool_x.fillna(medians)
    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0).replace(0.0, 1.0).fillna(1.0)
    train_scaled = ((train_filled - means) / scales).to_numpy(dtype=float)
    pool_scaled = ((pool_filled - means) / scales).to_numpy(dtype=float)

    out: dict[str, Any] = {"pool_materials": len(pool_ids), "k": k}

    random_scores: list[float] = []
    rng = np.random.default_rng(20261012)
    for _seed in seeds:
        order = rng.permutation(len(pool_ids))[:k]
        ranking = [pool_ids[index] for index in order]
        random_scores.append(score_ranking(ranking, labels, k=k)[2])
    out["random"] = {
        "draws": len(random_scores),
        "mean_S": float(np.mean(random_scores)),
        "sd_S": float(np.std(random_scores, ddof=1)),
        "min_S": float(np.min(random_scores)),
        "max_S": float(np.max(random_scores)),
    }

    ridge = Ridge(alpha=1.0, random_state=20261012).fit(train_scaled, y)
    out["ridge"] = _reference_from_scores(ridge.predict(pool_scaled), pool_ids, labels, k)

    forest = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, max_features=1.0,
        random_state=20261012, n_jobs=2,
    ).fit(train_filled.to_numpy(dtype=float), y)
    out["random_forest"] = _reference_from_scores(
        forest.predict(pool_filled.to_numpy(dtype=float)), pool_ids, labels, k
    )
    out["pore_volume"] = {
        "available": False,
        "reason": ("the pinned CRAFTED 2.0.1 descriptor frame carries the RAC families "
                   "only; no accessible-pore-volume column exists in the frozen inputs"),
    }
    return out


def _reference_from_scores(
    predictions: Any, pool_ids: Sequence[str], labels: Mapping[str, float], k: int,
) -> dict[str, float]:
    order = sorted(range(len(pool_ids)),
                   key=lambda index: (-float(predictions[index]), pool_ids[index]))
    ranking = [pool_ids[index] for index in order[:k]]
    q, q_star, s, recommended, _top = score_ranking(ranking, labels, k=k)
    return {"S": s, "Q": q, "Q_star": q_star,
            "mean_recommended_uptake": float(sum(recommended) / k)}


__all__ = [
    "ALLOWED_THIRD_PARTY", "FORBIDDEN_IMPORTS", "ID_COLUMN", "K_DEFAULT",
    "MofPool", "MofScreeningEvaluator", "TARGET_COLUMN",
    "material_references", "score_ranking", "screen_candidate_source",
]
