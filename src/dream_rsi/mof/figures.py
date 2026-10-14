"""Paper-style figures for the MOF screening experiment.

Five figures, following the v2 result figures' logic rather than their task:

1. **Quality-budget staircase** -- development best-so-far ``S`` against
   cumulative Pi tokens, one trajectory per paired run, with the outer worlds
   marked. Test-pool numbers never enter a trajectory.
2. **Recursive dynamics** -- one panel per split seed: sealed-pool ``S`` and
   ``Q`` at each budget checkpoint, every registered arm on the same checkpoints.
3. **Paired final comparison** -- per-split A/B/C final test ``S`` with paired
   A-B lines, the cluster-bootstrap interval, and the pre-registered 0.05 bar.
4. **Recommendation quality** -- the CO2 uptake distribution of each arm's top
   ten, against the random, ridge and random-forest references.
5. **Internal mechanism** -- Eq(1) ``V`` and tree best-so-far ``S`` by outer
   round. Captioned as an internal policy signal, never as screening evidence.

Every figure is written as PNG for reading and SVG for reuse, and every plotted
series is also written as CSV so a reader can re-derive the picture.
"""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Fixed colour per arm so a reader can carry one legend across all five
#: figures. Chosen to stay distinguishable in grayscale as well as colour.
ARM_STYLE: Mapping[str, dict[str, str]] = {
    "A": {"color": "#1b6ca8", "label": "A  Dream-RSI (learned policy)"},
    "B": {"color": "#c1440e", "label": "B  frozen initial policy (control)"},
    "C": {"color": "#3f7d20", "label": "C  frozen serial policy"},
}
REFERENCE_STYLE = {"color": "#555555", "alpha": "0.9"}


def _matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: Any, out_dir: Path, name: str, data_rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    svg = out_dir / f"{name}.svg"
    csv_path = out_dir / f"{name}.csv"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    _write_csv(csv_path, data_rows)
    return {"png": str(png), "svg": str(svg), "csv": str(csv_path)}


def _caption(fig: Any, text: str) -> None:
    fig.text(0.005, -0.02, text, ha="left", va="top", fontsize=7.2, color="#333333", wrap=True)


# --------------------------------------------------------------- figure 1


def quality_budget_staircase(blocks: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
    """Development best-so-far S against cumulative tokens, per paired run."""
    plt = _matplotlib()
    rows: list[dict[str, Any]] = []
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    from .protocol import PRACTICAL_THRESHOLD  # noqa: F401  (documented in caption)

    for record in blocks:
        plan = record["plan"]
        method = plan["method"]
        style = ARM_STYLE[method]
        baseline = float(record.get("baseline_S") or 0.0)
        points = [(0, baseline)]
        best = baseline
        for finding in record.get("findings", []):
            best = max(best, float(finding["score"]))
            points.append((int(finding["cumulative"]), best))
        if len(points) == 1:
            points.append((0, baseline))
        xs = [p[0] / 1e6 for p in points]
        ys = [p[1] for p in points]
        ax.step(xs, ys, where="post", color=style["color"], alpha=0.75, linewidth=1.4,
                label=style["label"] if _first_of_method(blocks, record) else None)
        for finding in record.get("findings", []):
            rows.append({
                "block_id": plan.get("block_id", ""), "method": method,
                "split_seed": plan["split_seed"], "repeat": plan["repeat"],
                "cumulative_tokens": finding["cumulative"],
                "node_score": finding["score"], "world_outer_t": finding["outer_t"],
                "round_index": finding["round_index"],
                "ledger_seq": finding["ledger_seq"], "charged": finding["charged"],
            })

    ax.set_xlabel("cumulative Pi tokens for the arm (millions)")
    ax.set_ylabel("development pool  best-so-far $S$")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25, linewidth=0.6)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc="lower right", frameon=False)
    ax.set_title("Screening quality against spent budget (development pool)", fontsize=10)
    _caption(fig, "Each line is one paired search run; steps are candidates in the order they "
                  "settled. Only the development score is shown -- the sealed pool is not part "
                  "of this figure and never fed the search.")
    return _save(fig, out_dir, "fig1_quality_budget_staircase", rows)


def _first_of_method(blocks: Sequence[Mapping[str, Any]], record: Mapping[str, Any]) -> bool:
    method = record["plan"]["method"]
    for other in blocks:
        if other["plan"]["method"] == method:
            return other is record
    return True


# --------------------------------------------------------------- figure 2


def recursive_dynamics(
    checkpoints_by_block: Sequence[Mapping[str, Any]], out_dir: Path,
) -> dict[str, str]:
    """One panel per split seed: sealed-pool S and Q at each budget checkpoint."""
    plt = _matplotlib()
    seeds = sorted({int(entry["split_seed"]) for entry in checkpoints_by_block})
    if not seeds:
        seeds = [0]
    fig, axes = plt.subplots(1, len(seeds), figsize=(3.4 * len(seeds), 3.4), squeeze=False)
    rows: list[dict[str, Any]] = []
    for axis, seed in zip(axes[0], seeds):
        for entry in checkpoints_by_block:
            if int(entry["split_seed"]) != seed:
                continue
            style = ARM_STYLE[entry["method"]]
            axis.plot([c["checkpoint"] / 1e6 for c in entry["points"]],
                      [c["test_S"] for c in entry["points"]],
                      marker="o", markersize=3.2, linewidth=1.3, color=style["color"],
                      label=entry["method"])
            for point in entry["points"]:
                rows.append({"split_seed": seed, "method": entry["method"],
                             "repeat": entry["repeat"],
                             "checkpoint_tokens": point["checkpoint"],
                             "test_S": point["test_S"], "test_Q": point["test_Q"],
                             "development_S": point["development_S"]})
        axis.set_title(f"split seed {seed}", fontsize=9)
        axis.set_xlabel("budget checkpoint (M tokens)", fontsize=8)
        axis.set_ylim(0.0, 1.0)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.tick_params(labelsize=7.5)
    axes[0][0].set_ylabel("sealed-pool $S$ at that budget", fontsize=8)
    axes[0][-1].legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Where each arm stands on the sealed pool at each budget checkpoint",
                 fontsize=10)
    _caption(fig, "Panels are split seeds; each line is one arm, averaged over its repeats "
                  "within the seed. The sealed pool is opened once, after every candidate is "
                  "frozen, so a point at a small checkpoint is genuinely what the arm had "
                  "produced by then.")
    return _save(fig, out_dir, "fig2_recursive_dynamics", rows)


# --------------------------------------------------------------- figure 3


def paired_final_comparison(
    table: Mapping[str, Any], out_dir: Path,
) -> dict[str, str]:
    """Per-split final S with paired A-B lines, the interval and the 0.05 bar."""
    plt = _matplotlib()
    rows = list(table["rows"])
    seeds = sorted({int(row["split_seed"]) for row in rows})
    # Legend entries for the arms this phase actually ran, in a fixed order, so a
    # two-arm phase does not advertise a third arm that never produced a point.
    present = [m for m in ("A", "B", "C") if any(f"{m}_test_S" in row for row in rows)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 4.2),
                                      gridspec_kw={"width_ratios": [2.1, 1.0]})
    positions = {seed: index for index, seed in enumerate(seeds)}
    for row in rows:
        x = positions[int(row["split_seed"])] + (int(row["repeat"]) - 1.5) * 0.16
        if "A_test_S" in row and "B_test_S" in row:
            left.plot([x - 0.05, x + 0.05], [row["B_test_S"], row["A_test_S"]],
                      color="#999999", linewidth=0.9, zorder=1)
        for method in ("A", "B", "C"):
            key = f"{method}_test_S"
            if key in row:
                left.scatter([x], [row[key]], s=22, color=ARM_STYLE[method]["color"],
                             zorder=2, label=method if (row is rows[0]) else None)
    left.set_xticks(range(len(seeds)))
    left.set_xticklabels([str(seed) for seed in seeds], fontsize=8)
    left.set_xlabel("material split seed")
    left.set_ylabel("final sealed-pool  $S$")
    left.set_ylim(0.0, 1.0)
    left.grid(alpha=0.25, linewidth=0.6)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=ARM_STYLE[m]["color"],
                          label=ARM_STYLE[m]["label"]) for m in present]
    left.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")
    left.set_title("Final screening score by split", fontsize=10)

    summary = table["A_minus_B"]
    if summary.get("available"):
        low, high = summary["ci95"]
        mean = summary["mean"]
        right.errorbar([0], [mean], yerr=[[mean - low], [high - mean]], fmt="o",
                       color=ARM_STYLE["A"]["color"], capsize=4, markersize=6)
        right.axhline(0.0, color="#333333", linewidth=0.9)
        right.axhline(table["A_minus_B_verdict"]["practical_threshold"], color="#c1440e",
                      linewidth=0.9, linestyle="--")
        right.text(0.06, table["A_minus_B_verdict"]["practical_threshold"], " 0.05 bar",
                   fontsize=7.5, color="#c1440e", va="bottom")
        for seed in seeds:
            if str(seed) in {str(k) for k in summary["per_split_mean"]}:
                pass
        for seed, value in summary["per_split_mean"].items():
            right.scatter([0.32], [value], s=18, color="#999999", zorder=2)
        right.set_xlim(-0.5, 0.6)
        right.set_xticks([0, 0.32])
        right.set_xticklabels(["mean", "per split"], fontsize=8)
    right.set_ylabel("$A - B$  (sealed-pool $S$)", fontsize=8)
    right.grid(alpha=0.25, linewidth=0.6)
    right.tick_params(labelsize=8)
    right.set_title("Paired difference, exploratory 95%", fontsize=10)
    _caption(fig, "Left: one point per paired run, grey lines pair A with B inside a block. "
                  f"Right: the cluster bootstrap resamples whole splits, keeping a split's "
                  f"repeats together; the interval is exploratory, over the {len(seeds)} "
                  "split(s) this phase ran, which share one sealed pool. It is not a claim "
                  "about MOFs in general.")
    return _save(fig, out_dir, "fig3_paired_final_comparison", rows)


# --------------------------------------------------------------- figure 4


def recommendation_explanation(
    unsealed: Sequence[Mapping[str, Any]], references: Mapping[str, Any], out_dir: Path,
) -> dict[str, str]:
    """Top-ten uptake distributions per arm against the screening references."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    rows: list[dict[str, Any]] = []
    groups: list[tuple[str, list[list[float]], str]] = []
    for method in ("A", "B", "C"):
        series = [list(item["recommended_uptakes"]) for item in unsealed
                  if item["method"] == method and item["recommended_uptakes"]]
        if series:
            groups.append((f"{method}", series, ARM_STYLE[method]["color"]))
    for start, (name, series, color) in enumerate(groups):
        flat = [value for run in series for value in run]
        jitter = [start + ((index % 10) - 4.5) * 0.018 for index in range(len(flat))]
        ax.scatter(jitter, flat, s=9, color=color, alpha=0.55, linewidths=0)
        for run in series:
            rows.append({"group": name, "kind": "arm_recommendation",
                         "mean_uptake": sum(run) / len(run), "n": len(run)})
    offset = len(groups)
    reference_rows: list[tuple[str, float]] = []
    for key, label in (("random", "random\norder"), ("ridge", "fixed\nridge"),
                       ("random_forest", "fixed random\nforest")):
        block = references.get(key) or {}
        if "Q" in block:
            reference_rows.append((label, float(block["Q"])))
    for index, (label, value) in enumerate(reference_rows):
        ax.scatter([offset + index], [value], marker="_", s=520,
                   color=REFERENCE_STYLE["color"], linewidths=2.4, zorder=3)
        rows.append({"group": " ".join(label.split()), "kind": "reference",
                     "mean_uptake": value, "n": 10})
    ax.set_xticks(list(range(len(groups))) + [offset + i for i in range(len(reference_rows))])
    ax.set_xticklabels([name for name, _s, _c in groups]
                       + [name for name, _v in reference_rows], fontsize=8)
    ax.set_ylabel("CO$_2$ uptake of the recommended materials (mol/kg)")
    ax.grid(alpha=0.25, linewidth=0.6, axis="y")
    ax.set_title("What each arm actually recommended, against the references", fontsize=10)
    _caption(fig, "Each dot is one material in one frozen candidate's top ten; the bars are the "
                  "fixed references. Raw uptake in mol/kg -- the normalised S is a different "
                  "axis and is reported in the tables, not mixed in here.")
    return _save(fig, out_dir, "fig4_recommendation_explanation", rows)


# --------------------------------------------------------------- figure 5


def internal_mechanism(blocks: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
    """Eq(1) V and tree best-so-far S by outer round -- an internal signal only."""
    plt = _matplotlib()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.2, 3.8))
    rows: list[dict[str, Any]] = []
    for record in blocks:
        plan = record["plan"]
        style = ARM_STYLE[plan["method"]]
        xs, ys, best = [], [], []
        running = float(record.get("baseline_S") or 0.0)
        for world in record.get("worlds", []):
            xs.append(world["outer_t"])
            ys.append(world["s_best"])
            running = max(running, float(world["s_best"]))
            best.append(running)
            rows.append({"run_id": record["run_id"], "method": plan["method"],
                         "split_seed": plan["split_seed"], "repeat": plan["repeat"],
                         "outer_t": world["outer_t"], "world_s_best": world["s_best"],
                         "world_b_count": world["b_count"],
                         "best_so_far": running})
        if xs:
            left.plot(xs, ys, marker="o", markersize=3, linewidth=1.2,
                      color=style["color"], alpha=0.8,
                      label=style["label"] if _first_of_method(blocks, record) else None)
            right.plot(xs, best, marker="o", markersize=3, linewidth=1.2,
                       color=style["color"], alpha=0.8)
    for axis, title in ((left, "sealed world $S$ by outer round"),
                        (right, "tree best-so-far $S$ by outer round")):
        axis.set_xlabel("outer world $t$", fontsize=8)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.set_title(title, fontsize=9.5)
        axis.set_ylim(0.0, 1.0)
    left.set_ylabel("$S$", fontsize=8)
    left.legend(fontsize=7.5, frameon=False, loc="lower right")
    _caption(fig, "Internal signals only. The sealed world's S is what Eq(1) consumes through the "
                  "replay sweep, and V is the arms' own basis for choosing a policy. A rising V "
                  "is not evidence that the materials recommended to a user got better; that "
                  "claim comes from the sealed-pool figures.")
    return _save(fig, out_dir, "fig5_internal_mechanism", rows)


def policy_value_curve(score_rows: Sequence[Mapping[str, Any]], out_dir: Path) -> dict[str, str]:
    """Eq(1) V per (policy, world) if score records were exported."""
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    rows: list[dict[str, Any]] = []
    by_policy: dict[str, list[tuple[int, float]]] = {}
    for index, row in enumerate(score_rows):
        by_policy.setdefault(str(row.get("policy_version_id")), []).append(
            (index, float(row.get("value", 0.0))))
        rows.append(dict(row))
    for policy, points in sorted(by_policy.items()):
        ax.plot([p[0] for p in points], [p[1] for p in points], marker=".", linewidth=1.0,
                label=policy[-24:])
    ax.set_xlabel("recorded Eq(1) evaluation, in order", fontsize=8)
    ax.set_ylabel("$V$", fontsize=8)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_title("Internal policy objective $V$ over recorded evaluations", fontsize=9.5)
    ax.legend(fontsize=6, frameon=False, ncol=2)
    _caption(fig, "V is the selector's internal objective, computed from replayed trajectories. "
                  "It is reported for mechanism only; it never substitutes for the sealed-pool "
                  "screening result.")
    return _save(fig, out_dir, "fig5b_policy_value", rows)


def write_references(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


__all__ = [
    "ARM_STYLE", "internal_mechanism", "paired_final_comparison", "policy_value_curve",
    "quality_budget_staircase", "recommendation_explanation", "recursive_dynamics",
    "write_references",
]
