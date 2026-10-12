"""Unseal the final test pool once, and turn frozen blocks into the result tables.

Two rules govern everything here, and both are about not fooling ourselves.

**The test pool is opened once.** :func:`unseal_block` grades a block's frozen
candidate on the sealed pool and nothing else -- it does not re-grade the tree,
does not search for a better node, and does not feed anything back. Selection
happened on the development score, before this module ran, and that ordering is
what makes the test number evidence rather than a second look at the same data.

**The statistics are at the level the design fixes: the paired search run.**
Five split seeds x two repeats is ten paired runs per arm, and the exploratory
interval is a cluster bootstrap that resamples *splits*, keeping each split's
repeats together. Nodes, materials and rho values are not independent
observations and are never treated as such.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import K, PRACTICAL_THRESHOLD, ROW_ORDER_PERMUTATIONS
from .scoring import MofPool, MofScreeningEvaluator, score_ranking

#: Resamples for the exploratory paired interval. Fixed so the report is
#: reproducible; the interval is exploratory either way and says so.
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261012


@dataclass(frozen=True)
class UnsealResult:
    """One frozen candidate, measured once on the sealed pool."""

    run_id: str
    method: str
    split_seed: int
    repeat: int
    origin: str
    candidate_sha256: str
    development_S: float
    test_S: float
    test_Q: float
    test_Q_star: float
    recommended_uptakes: list[float]
    top_k_uptakes: list[float]
    high_performer_hits: int
    high_performer_cutoff: int
    contract_status: str
    error_type: str | None
    duration_s: float
    permutation_agreement: list[bool]

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _high_performer_cutoff(pool_size: int) -> int:
    """The design's ``ceil(0.1 * materials)`` band, ties broken by material id."""
    import math
    return max(1, math.ceil(0.1 * pool_size))


def unseal_block(
    *,
    record: Mapping[str, Any],
    prepared_root: Path,
    runner_script: Path,
    workdir_root: Path,
    permutations: int = ROW_ORDER_PERMUTATIONS,
) -> UnsealResult:
    """Grade one block's final frozen candidate against the sealed test pool."""
    plan = record["plan"]
    split_seed = int(plan["split_seed"])
    final = record.get("final_candidate")
    if not final:
        raise ValueError(f"block {record.get('run_id')} has no frozen candidate to unseal")

    pool = MofPool.final_test(prepared_root, split_seed=split_seed)
    labels = pool.load_labels()
    evaluator = MofScreeningEvaluator(
        pool=pool, labels=labels, runner_script=runner_script,
        workdir_root=workdir_root / f"{record['run_id']}-unseal", k=K,
    )
    candidate_dir = Path(final["source_path"]).parent
    verdict = evaluator.evaluate(candidate_dir, invocation_id=f"{record['run_id']}-final")

    agreement: list[bool] = []
    base_ids: list[str] | None = None
    if verdict.status.value == "SUCCESS" or verdict.score > 0.0:
        base_ids = list(verdict.detail.get("ranking") or [])
        agreement = _permutation_agreement(
            pool=pool, labels=labels, runner_script=runner_script,
            workdir_root=workdir_root, candidate_dir=candidate_dir,
            run_id=str(record["run_id"]), expected=base_ids, permutations=permutations,
        )

    ranking = list(verdict.detail.get("ranking") or [])
    recommended = [float(v) for v in verdict.detail.get("recommended_uptakes") or []]
    top = [float(v) for v in verdict.detail.get("top_k_uptakes") or []]
    cutoff = _high_performer_cutoff(len(labels))
    threshold = sorted(labels.values(), reverse=True)[cutoff - 1] if labels else 0.0
    hits = sum(1 for value in recommended if value >= threshold)

    return UnsealResult(
        run_id=str(record["run_id"]), method=str(plan["method"]),
        split_seed=split_seed, repeat=int(plan["repeat"]),
        origin=str(final["origin"]), candidate_sha256=str(final["candidate_sha256"]),
        development_S=float(final["development_S"]),
        test_S=float(verdict.score),
        test_Q=float(verdict.detail.get("Q") or 0.0),
        test_Q_star=float(verdict.detail.get("Q_star") or 0.0),
        recommended_uptakes=recommended, top_k_uptakes=top,
        high_performer_hits=hits, high_performer_cutoff=cutoff,
        contract_status=str(verdict.public_detail.get("contract_status", "")),
        error_type=verdict.error_type, duration_s=float(verdict.duration_s),
        permutation_agreement=agreement,
    )


def _permutation_agreement(
    *, pool: MofPool, labels: Mapping[str, float], runner_script: Path,
    workdir_root: Path, candidate_dir: Path, run_id: str,
    expected: Sequence[str], permutations: int,
) -> list[bool]:
    """Re-grade the same program on reshuffled pool rows.

    Diagnostic only. The design fixes this check in advance and forbids picking
    the most flattering permutation; a program whose recommendation set moves
    when the rows move has a contract problem, and the honest report says so
    instead of quoting the best of four numbers.
    """
    with pool.pool_features.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    header, body = rows[0], rows[1:]
    results: list[bool] = []
    for index in range(permutations):
        shuffled = list(body)
        random.Random(BOOTSTRAP_SEED + index).shuffle(shuffled)
        variant_dir = workdir_root / f"{run_id}-perm{index}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_features = variant_dir / "pool_features.csv"
        with variant_features.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(shuffled)
        variant = MofPool(
            label=f"{pool.label}-perm{index}",
            train_features=pool.train_features, train_targets=pool.train_targets,
            pool_features=variant_features, private_targets=pool.private_targets,
            payload_seed=pool.payload_seed,
        )
        evaluator = MofScreeningEvaluator(
            pool=variant, labels=labels, runner_script=runner_script,
            workdir_root=variant_dir, k=K,
        )
        verdict = evaluator.evaluate(candidate_dir, invocation_id=f"perm{index}")
        got = list(verdict.detail.get("ranking") or [])
        results.append(set(got) == set(expected))
    return results


# ---------------------------------------------------------------------- statistics


def cluster_bootstrap_ci(
    per_split: Mapping[int, Sequence[float]], *,
    draws: int = BOOTSTRAP_DRAWS, seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Percentile interval for a paired difference, resampling whole splits.

    ``per_split`` maps a split seed to that split's repeat-level differences.
    A split's repeats stay together in every resample, which is what keeps the
    interval honest about the design: repeats inside a split share a training set
    and a development pool, so treating them as independent draws would shrink
    the interval for no reason.
    """
    import numpy as np

    seeds = sorted(per_split)
    if not seeds:
        return {"available": False, "reason": "no paired differences"}
    split_means = np.array([float(np.mean(per_split[seed])) for seed in seeds], dtype=float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(seeds), size=(draws, len(seeds)))
    resampled = split_means[samples].mean(axis=1)
    return {
        "available": True,
        "splits": seeds,
        "per_split_mean": {int(s): float(m) for s, m in zip(seeds, split_means)},
        "mean": float(split_means.mean()),
        "median": float(np.median(split_means)),
        "min": float(split_means.min()),
        "max": float(split_means.max()),
        "ci95": [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))],
        "draws": draws,
        "consistent_direction": int(sum(1 for m in split_means if m > 0)),
    }


def verdict_from_ci(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the design's pre-registered reading to a paired A-B summary."""
    if not summary.get("available"):
        return {"verdict": "no_evidence", "reason": "no paired runs to compare"}
    low, high = summary["ci95"]
    mean = summary["mean"]
    consistent = int(summary["consistent_direction"])
    total = len(summary["splits"])
    crosses_zero = low <= 0.0 <= high
    if mean >= PRACTICAL_THRESHOLD and consistent >= 4 and not crosses_zero:
        verdict = "worth_further_validation"
    elif crosses_zero:
        verdict = "insufficient_evidence"
    elif mean < 0:
        verdict = "negative_result"
    else:
        verdict = "below_practical_threshold"
    return {
        "verdict": verdict,
        "mean": mean,
        "ci95": [low, high],
        "practical_threshold": PRACTICAL_THRESHOLD,
        "splits_favouring_A": consistent,
        "splits_total": total,
        "crosses_zero": crosses_zero,
        "rules": ("pre-registered: A-B >= 0.05 S units, same direction in >= 4/5 splits, "
                  "and an exploratory interval that does not cross 0"),
    }


def paired_table(
    unsealed: Sequence[UnsealResult],
) -> dict[str, Any]:
    """Group unsealed runs into the block-level paired table the design asks for."""
    by_block: dict[tuple[int, int], dict[str, UnsealResult]] = {}
    for result in unsealed:
        by_block.setdefault((result.split_seed, result.repeat), {})[result.method] = result
    rows: list[dict[str, Any]] = []
    for (split_seed, repeat), arms in sorted(by_block.items()):
        row: dict[str, Any] = {"split_seed": split_seed, "repeat": repeat,
                               "complete": set(arms) == {"A", "B", "C"}}
        for key, result in sorted(arms.items()):
            row[f"{key}_test_S"] = result.test_S
            row[f"{key}_test_Q"] = result.test_Q
            row[f"{key}_dev_S"] = result.development_S
            row[f"{key}_origin"] = result.origin
        if "A" in arms and "B" in arms:
            row["A_minus_B"] = arms["A"].test_S - arms["B"].test_S
            row["A_minus_B_Q"] = arms["A"].test_Q - arms["B"].test_Q
        if "A" in arms and "C" in arms:
            row["A_minus_C"] = arms["A"].test_S - arms["C"].test_S
        rows.append(row)

    per_split_ab: dict[int, list[float]] = {}
    for row in rows:
        if "A_minus_B" in row:
            per_split_ab.setdefault(int(row["split_seed"]), []).append(float(row["A_minus_B"]))
    per_split_ac: dict[int, list[float]] = {}
    for row in rows:
        if "A_minus_C" in row:
            per_split_ac.setdefault(int(row["split_seed"]), []).append(float(row["A_minus_C"]))

    ab = cluster_bootstrap_ci(per_split_ab)
    ac = cluster_bootstrap_ci(per_split_ac)
    return {
        "rows": rows,
        "complete_blocks": sum(1 for row in rows if row["complete"]),
        "blocks": len(rows),
        "A_minus_B": ab,
        "A_minus_C": ac,
        "A_minus_B_verdict": verdict_from_ci(ab),
        "A_minus_C_verdict": verdict_from_ci(ac),
    }


def development_test_gap(unsealed: Sequence[UnsealResult]) -> dict[str, Any]:
    """How much of each arm's development score survives contact with the pool."""
    out: dict[str, Any] = {}
    for method in ("A", "B", "C"):
        rows = [r for r in unsealed if r.method == method]
        if not rows:
            continue
        dev = [r.development_S for r in rows]
        test = [r.test_S for r in rows]
        out[method] = {
            "n": len(rows),
            "mean_development_S": sum(dev) / len(dev),
            "mean_test_S": sum(test) / len(test),
            "mean_gap": sum(d - t for d, t in zip(dev, test)) / len(dev),
            "failures": sum(1 for r in rows if r.contract_status not in ("OK", "")
                            or r.error_type is not None),
        }
    return out


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_candidate(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    return sha256_file(destination)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return rows
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


__all__ = [
    "BOOTSTRAP_DRAWS", "BOOTSTRAP_SEED", "UnsealResult", "cluster_bootstrap_ci",
    "copy_candidate", "development_test_gap", "load_jsonl", "paired_table",
    "sha256_file", "unseal_block", "verdict_from_ci", "write_json",
]
