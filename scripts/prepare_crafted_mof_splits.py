"""Validate CRAFTED 2.0.1 and prepare reproducible group-wise MOF splits.

The archive and all generated data stay under ``.runtime`` and are never
committed. The ``agent`` folders contain only training labels and development
features; development labels and final-test data are stored in sibling private
folders. This filesystem separation is organizational, not an OS security
boundary. A formal Pi run must still restrict the agent to its ``agent`` folder.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_MD5 = "e11e4f84cdd484db7811bc758faaed37"
EXPECTED_SHA256 = "f29f7b3d6b7d6c51fd113138bc4a811c448d4dcce66ed37fddbfa4141a238dde"
PRESSURE_PA = 100_000.0
PRESSURE_TOLERANCE_PA = 1.0
FINAL_SPLIT_SEED = 20261012
SPLIT_SEEDS = (101, 211, 307, 401, 503)
OPAQUE_ID_SEED = "dream-rsi-mof-20261012-v1"
TASK_SUFFIX = "_UFF_CO2_298.csv"
TARGET_FILE_RE = re.compile(r"^DDEC_(.+)_UFF_CO2_298\.csv$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def numeric_features(
    rows: list[dict[str, str]],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    if not rows or "FrameworkName" not in rows[0]:
        raise ValueError("MOF descriptor table is empty or has no FrameworkName column")
    feature_names = [name for name in rows[0] if name != "FrameworkName"]
    if not feature_names:
        raise ValueError("MOF descriptor table contains no features")
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        material_id = row["FrameworkName"].strip()
        if not material_id or material_id in result:
            raise ValueError(f"empty or duplicate MOF FrameworkName: {material_id!r}")
        values: dict[str, float] = {}
        for name in feature_names:
            raw = row[name].strip()
            if not raw:
                raise ValueError(f"empty feature {name!r} for {material_id}")
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError(f"non-finite feature {name!r} for {material_id}")
            values[name] = value
        result[material_id] = values
    return feature_names, result


def read_mof_ids(path: Path) -> set[str]:
    rows = read_csv(path)
    ids = [row.get("FrameworkName", "").strip() for row in rows]
    if not ids or any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"invalid or duplicate FrameworkName values in {path}")
    return set(ids)


def read_isotherm(path: Path) -> tuple[float, float, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        expected = ["# pressure[Pa]", "mean_volume[mol/kg]", "mean_error[mol/kg]"]
        if header != expected:
            raise ValueError(f"unexpected isotherm header in {path.name}: {header!r}")
        rows: list[tuple[float, float, float]] = []
        for line_number, row in enumerate(reader, start=2):
            if len(row) != 3:
                raise ValueError(f"malformed row {line_number} in {path.name}")
            pressure, uptake, uncertainty = (float(value) for value in row)
            if not all(math.isfinite(value) for value in (pressure, uptake, uncertainty)):
                raise ValueError(f"non-finite isotherm value in {path.name}:{line_number}")
            if pressure <= 0 or uptake < 0 or uncertainty < 0:
                raise ValueError(f"invalid pressure, uptake, or uncertainty in {path.name}:{line_number}")
            rows.append((pressure, uptake, uncertainty))
    if not rows:
        raise ValueError(f"empty isotherm: {path.name}")
    matches = [
        (uptake, uncertainty)
        for pressure, uptake, uncertainty in rows
        if abs(pressure - PRESSURE_PA) <= PRESSURE_TOLERANCE_PA
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one point within {PRESSURE_TOLERANCE_PA:g} Pa of "
            f"{PRESSURE_PA:g} Pa in {path.name}, found {len(matches)}"
        )
    return matches[0][0], matches[0][1], len(rows)


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def join_equal_keys(
    union_find: UnionFind, keys: dict[str, str]
) -> None:
    first_by_key: dict[str, str] = {}
    for material_id, key in keys.items():
        first = first_by_key.setdefault(key, material_id)
        union_find.union(first, material_id)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def hash_order(seed: int, group_id: str, purpose: str) -> str:
    material = f"{purpose}:{seed}:{group_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def material_rows(
    ids: set[str],
    aliases: dict[str, str],
    features: dict[str, dict[str, float]],
    targets: dict[str, tuple[float, float]],
    groups: dict[str, str],
) -> list[dict[str, Any]]:
    rows = []
    for source_id in sorted(ids, key=lambda value: aliases[value]):
        rows.append({
            "material_id": aliases[source_id],
            "source_id": source_id,
            "group_id": groups[source_id],
            **features[source_id],
            "target_mol_kg": targets[source_id][0],
            "uncertainty_mol_kg": targets[source_id][1],
        })
    return rows


def write_agent_split(
    split_dir: Path,
    rows_by_id: dict[str, dict[str, Any]],
    feature_names: list[str],
    train_ids: set[str],
    dev_ids: set[str],
) -> None:
    agent_dir = split_dir / "agent"
    private_dir = split_dir / "evaluator_private"
    train_rows = [rows_by_id[value] for value in sorted(train_ids)]
    dev_rows = [rows_by_id[value] for value in sorted(dev_ids)]
    common_fields = ["material_id", *feature_names]
    write_csv(agent_dir / "train_features.csv", common_fields, [
        {key: row[key] for key in common_fields} for row in train_rows
    ])
    write_csv(agent_dir / "train_targets.csv", ["material_id", "target_mol_kg"], [
        {"material_id": row["material_id"], "target_mol_kg": row["target_mol_kg"]}
        for row in train_rows
    ])
    write_csv(agent_dir / "dev_features.csv", common_fields, [
        {key: row[key] for key in common_fields} for row in dev_rows
    ])
    write_csv(
        private_dir / "dev_targets.private.csv",
        ["material_id", "target_mol_kg", "uncertainty_mol_kg"],
        [{key: row[key] for key in ("material_id", "target_mol_kg", "uncertainty_mol_kg")}
         for row in dev_rows],
    )


def prepare(project_root: Path, dataset_root: Path, output_root: Path) -> dict[str, Any]:
    archive = project_root / ".runtime/mof_experiment/raw/CRAFTED-2.0.1.verified.tar.xz"
    if not archive.is_file():
        raise FileNotFoundError(f"verified CRAFTED archive not found: {archive}")
    if md5_file(archive) != EXPECTED_MD5 or sha256_file(archive) != EXPECTED_SHA256:
        raise ValueError("CRAFTED archive checksum differs from the pinned Zenodo 2.0.1 file")

    mof_table = dataset_root / "RAC_DBSCAN/CRAFTED_MOF_full_featurization_frame.csv"
    cof_table = dataset_root / "RAC_DBSCAN/CRAFTED_COF_full_featurization_frame.csv"
    mof_rows = read_csv(mof_table)
    feature_names, features = numeric_features(mof_rows)
    mof_ids = set(features)
    cof_ids = read_mof_ids(cof_table)
    if len(mof_ids) != 690 or len(cof_ids) != 667 or mof_ids & cof_ids:
        raise ValueError("CRAFTED MOF/COF material inventory differs from the pinned version")

    target_dir = dataset_root / "ISOTHERM_FILES"
    target_paths = sorted(target_dir.glob("DDEC_*_UFF_CO2_298.csv"))
    target_ids: dict[str, Path] = {}
    for path in target_paths:
        match = TARGET_FILE_RE.fullmatch(path.name)
        if match is None or match.group(1) in target_ids:
            raise ValueError(f"unexpected or duplicate target filename: {path.name}")
        target_ids[match.group(1)] = path
    if len(target_ids) != len(mof_ids) + len(cof_ids):
        raise ValueError("target-condition isotherm count differs from the MOF+COF inventory")
    if set(target_ids) != (mof_ids | cof_ids):
        raise ValueError("target-condition isotherm IDs do not exactly match MOF+COF tables")

    targets: dict[str, tuple[float, float]] = {}
    point_counts: set[int] = set()
    for material_id in sorted(mof_ids):
        uptake, uncertainty, point_count = read_isotherm(target_ids[material_id])
        targets[material_id] = (uptake, uncertainty)
        point_counts.add(point_count)
    if point_counts != {13}:
        raise ValueError(f"unexpected pressure-grid sizes among target MOF isotherms: {point_counts}")

    if len(mof_ids) != len(targets):
        raise ValueError("not every MOF has a target label")
    cif_dir = dataset_root / "CIF_FILES/DDEC"
    cif_hashes: dict[str, str] = {}
    for material_id in mof_ids:
        cif_path = cif_dir / f"{material_id}.cif"
        if not cif_path.is_file():
            raise FileNotFoundError(f"DDEC CIF missing for MOF {material_id}")
        cif_hashes[material_id] = sha256_file(cif_path)

    union_find = UnionFind(sorted(mof_ids))
    feature_hashes: dict[str, str] = {}
    for material_id, values in features.items():
        canonical = "\x1f".join(
            f"{values[name]:.17g}" for name in feature_names
        ).encode("utf-8")
        feature_hashes[material_id] = hashlib.sha256(canonical).hexdigest()
    join_equal_keys(union_find, feature_hashes)
    join_equal_keys(union_find, cif_hashes)

    aliases = {
        source_id: "mof_" + hashlib.sha256(
            f"{OPAQUE_ID_SEED}:{source_id}".encode("utf-8")
        ).hexdigest()[:12]
        for source_id in sorted(mof_ids)
    }
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("opaque material ID collision")
    members_by_root: dict[str, list[str]] = defaultdict(list)
    for material_id in sorted(mof_ids):
        members_by_root[union_find.find(material_id)].append(material_id)
    group_for_id: dict[str, str] = {}
    for members in members_by_root.values():
        group_id = "grp_" + hashlib.sha256(
            "\x1f".join(sorted(aliases[value] for value in members)).encode("utf-8")
        ).hexdigest()[:12]
        for material_id in members:
            group_for_id[material_id] = group_id
    groups = sorted(set(group_for_id.values()))
    if len(groups) < 500:
        raise ValueError(f"only {len(groups)} exact-duplicate-aware groups remain; need >= 500")

    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is non-empty; refusing to overwrite: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    rows = material_rows(mof_ids, aliases, features, targets, group_for_id)
    rows_by_id = {row["material_id"]: row for row in rows}
    feature_fields = ["material_id", *feature_names]
    all_private = output_root / "private"
    write_csv(all_private / "id_and_group_map.private.csv",
              ["source_id", "material_id", "group_id"], [
                  {"source_id": source_id, "material_id": aliases[source_id],
                   "group_id": group_for_id[source_id]}
                  for source_id in sorted(mof_ids)
              ])
    write_csv(all_private / "all_targets.private.csv",
              ["material_id", "target_mol_kg", "uncertainty_mol_kg"], [
                  {"material_id": aliases[source_id],
                   "target_mol_kg": targets[source_id][0],
                   "uncertainty_mol_kg": targets[source_id][1]}
                  for source_id in sorted(mof_ids, key=lambda value: aliases[value])
              ])
    write_csv(all_private / "cof_exclusions.private.csv",
              ["source_id", "reason"], [
                  {"source_id": source_id, "reason": "not in official MOF descriptor table; COF"}
                  for source_id in sorted(cof_ids)
              ])

    final_test_count = (3 * len(groups) + 5) // 10
    test_groups = set(sorted(
        groups, key=lambda group: hash_order(FINAL_SPLIT_SEED, group, "final-test")
    )[:final_test_count])
    work_groups = set(groups) - test_groups
    test_ids = {mid for mid in mof_ids if group_for_id[mid] in test_groups}
    test_rows = [rows_by_id[aliases[mid]] for mid in sorted(test_ids, key=lambda value: aliases[value])]
    write_csv(all_private / "final_test_features.private.csv", feature_fields, [
        {key: row[key] for key in feature_fields} for row in test_rows
    ])
    write_csv(all_private / "final_test_targets.private.csv",
              ["material_id", "target_mol_kg", "uncertainty_mol_kg"], [
                  {key: row[key] for key in ("material_id", "target_mol_kg", "uncertainty_mol_kg")}
                  for row in test_rows
              ])

    split_summaries = []
    for seed in SPLIT_SEEDS:
        ordered_work = sorted(work_groups, key=lambda group: hash_order(seed, group, "work-split"))
        train_group_count = (5 * len(ordered_work)) // 7
        train_groups = set(ordered_work[:train_group_count])
        dev_groups = set(ordered_work[train_group_count:])
        train_ids = {mid for mid in mof_ids if group_for_id[mid] in train_groups}
        dev_ids = {mid for mid in mof_ids if group_for_id[mid] in dev_groups}
        if train_groups & dev_groups or train_groups & test_groups or dev_groups & test_groups:
            raise AssertionError("group leakage across train, development, and final test")
        if len(dev_ids) < 100 or len(dev_groups) < 50 or len(test_ids) < 100 or len(test_groups) < 50:
            raise ValueError("development/final-test material or group minimum is not met")
        split_dir = output_root / "splits" / str(seed)
        write_agent_split(split_dir, rows_by_id, feature_names,
                          {aliases[value] for value in train_ids},
                          {aliases[value] for value in dev_ids})
        split_summaries.append({
            "seed": seed,
            "agent_dir": f"splits/{seed}/agent",
            "private_development_targets": f"splits/{seed}/evaluator_private/dev_targets.private.csv",
            "train_materials": len(train_ids),
            "train_groups": len(train_groups),
            "development_materials": len(dev_ids),
            "development_groups": len(dev_groups),
            "final_test_materials": len(test_ids),
            "final_test_groups": len(test_groups),
        })

    manifest: dict[str, Any] = {
        "schema": "dream-rsi-crafted-mof-splits-v1",
        "source": {
            "doi": "10.5281/zenodo.10120180",
            "archive_md5": EXPECTED_MD5,
            "archive_sha256": EXPECTED_SHA256,
            "condition": {"gas": "CO2", "temperature_K": 298,
                          "pressure_Pa": PRESSURE_PA, "forcefield": "UFF",
                          "charge_scheme": "DDEC"},
        },
        "population": {
            "mof_materials": len(mof_ids), "cof_materials_excluded": len(cof_ids),
            "target_isotherms": len(target_paths), "feature_columns": len(feature_names),
            "pressure_points_per_mof": sorted(point_counts),
            "groups_after_exact_feature_and_cif_deduplication": len(groups),
        },
        "split_policy": {
            "final_test_seed": FINAL_SPLIT_SEED,
            "final_test_group_fraction": 0.30,
            "development_train_ratio": "5:2",
            "development_split_seeds": list(SPLIT_SEEDS),
            "ordering": "SHA-256 keyed ordering; does not inspect target labels",
            "grouping": "union of exact canonical feature-vector matches and exact DDEC CIF SHA-256 matches",
            "residual_risk": "Near-structure and source-family duplicates are not fully resolved.",
        },
        "splits": split_summaries,
        "private_test_features": "private/final_test_features.private.csv",
        "private_test_targets": "private/final_test_targets.private.csv",
        "agent_label_isolation_note": (
            "The agent folders contain only training labels and development features, "
            "but sibling private files are not OS-permission-isolated. Before a formal "
            "Pi run, restrict the agent to its agent folder. The source dataset is "
            "public and this cannot establish protection against model memory."
        ),
    }
    (output_root / "split-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--dataset-root", type=Path,
        default=root / ".runtime/mof_experiment/data/CRAFTED-2.0.1",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=root / ".runtime/mof_experiment/prepared/CRAFTED-2.0.1-CO2-298-UFF-DDEC-v1",
    )
    args = parser.parse_args()
    manifest = prepare(args.project_root.resolve(), args.dataset_root.resolve(), args.output_root.resolve())
    print(json.dumps({"status": "OK", **manifest["population"], "splits": manifest["splits"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
