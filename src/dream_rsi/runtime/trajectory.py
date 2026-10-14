"""Compact, lossless encoding for a replay episode's decision trace.

The problem this solves is measured, not theoretical. A format-2 trace stores,
for every decision step, the *entire* observation the policy saw -- every
revealed node's full public record, plus the newly revealed nodes a second time.
Because the prefix grows with each step, the same node record is written once per
later step: in the policy-development prompt that killed the first formal A arm,
300 steps restated 110 distinct nodes, 6.16 MB of payload where the same
information needs about 0.3 MB.

The encoding here keeps **all** of it. Nothing is dropped on the grounds that
some other component does not read it: the node records are stored once, in a
table, and every step refers to them by id. :func:`decode_decisions` rebuilds the
rich steps byte-for-byte equal to the ones ``replay_world`` produced, so a
consumer that wants the full observation of any step still gets it.

Layout, as a ``tuple[str, ...]`` so ``ReplayEpisode.decision_trace`` keeps its
declared type:

* element 0 -- header: ``{"format": 3, "nodes": [[id, record], ...]}``
* elements 1..n -- one compact step each:
  ``{"batch": [...], "revealed": [id, ...], "observation": {...}}``

The header carries no ``batch`` key, so anything iterating steps must skip index
0 explicitly rather than assuming every element is a step.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import Any

#: The encoding that stores a full observation per step.
FORMAT_FULL = 2

#: The encoding below: node records once, steps by reference.
FORMAT_SHARED_NODES = 3

CURRENT_FORMAT = FORMAT_SHARED_NODES

_HEADER = "format"
_NODES = "nodes"


def encode_decisions(steps: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Encode rich steps (as ``replay_world`` builds them) into the compact form."""
    table: dict[str, Mapping[str, Any]] = {}
    compact: list[dict[str, Any]] = []
    for step in steps:
        observation = step["observation"]
        for node in observation["nodes"]:
            table[str(node["node_id"])] = node
        for node in step["revealed"]:
            table[str(node["node_id"])] = node
        compact.append({
            "batch": list(step["batch"]),
            "revealed": [str(node["node_id"]) for node in step["revealed"]],
            "observation": {
                # node_ids is stored rather than derived: it is what the policy
                # actually saw, in the order it saw it, and reconstructing order
                # from a sorted table would be a guess.
                "node_ids": list(observation["node_ids"]),
                "legal_actions": list(observation["legal_actions"]),
                "completed_rounds": observation["completed_rounds"],
                "max_parallelism": observation["max_parallelism"],
            },
        })
    header = {_HEADER: FORMAT_SHARED_NODES,
              _NODES: [[node_id, table[node_id]] for node_id in sorted(table)]}
    return (json.dumps(header, sort_keys=True),
            *(json.dumps(step, sort_keys=True) for step in compact))


def decode_decisions(encoded: Sequence[str]) -> list[dict[str, Any]]:
    """Rebuild the rich steps exactly, including every step's full observation."""
    if not encoded:
        return []
    header = json.loads(encoded[0])
    table = {str(node_id): record for node_id, record in header[_NODES]}
    steps: list[dict[str, Any]] = []
    for raw in encoded[1:]:
        step = json.loads(raw)
        observation = step["observation"]
        steps.append({
            "observation": {
                "nodes": [table[node_id] for node_id in observation["node_ids"]],
                "node_ids": list(observation["node_ids"]),
                "legal_actions": list(observation["legal_actions"]),
                "completed_rounds": observation["completed_rounds"],
                "max_parallelism": observation["max_parallelism"],
            },
            "batch": list(step["batch"]),
            "revealed": [table[node_id] for node_id in step["revealed"]],
        })
    return steps


def node_table(encoded: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    """The shared node records, once, for a caller that wants to render them."""
    if not encoded:
        return {}
    header = json.loads(encoded[0])
    if header.get(_HEADER) != FORMAT_SHARED_NODES:
        return {}
    return {str(node_id): record for node_id, record in header[_NODES]}


def iter_batches(episode: Any) -> Iterator[list[str]]:
    """The recorded action of every step, in order, for both trace formats."""
    if episode.trajectory_format == FORMAT_FULL:
        for raw in episode.decision_trace:
            yield json.loads(raw)["batch"]
        return
    for raw in list(episode.decision_trace)[1:]:
        yield json.loads(raw)["batch"]


def decode_any(episode: Any) -> list[dict[str, Any]]:
    """Rich steps for either format. Format 2 is already rich."""
    if episode.trajectory_format == FORMAT_FULL:
        return [json.loads(raw) for raw in episode.decision_trace]
    return decode_decisions(episode.decision_trace)


def header_of(encoded: Sequence[str]) -> Mapping[str, Any]:
    return json.loads(encoded[0]) if encoded else {}


def is_supported_format(value: int) -> bool:
    return value in (FORMAT_FULL, FORMAT_SHARED_NODES)


def encode_stream(steps: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    return encode_decisions(list(steps))


__all__ = [
    "CURRENT_FORMAT", "FORMAT_FULL", "FORMAT_SHARED_NODES", "decode_any",
    "decode_decisions", "encode_decisions", "encode_stream", "header_of",
    "is_supported_format", "iter_batches", "node_table",
]


def render_node_table(episodes: Iterable[Any]) -> list[str]:
    """One node table for every episode given, emitted a single time.

    A world's node records do not depend on which policy was replayed or which
    rho was used, so a feedback block that reports ten policies on the same five
    worlds was writing the same hundred and ten records fifty times over. The
    union of what the episodes reference, written once, is the whole table.
    """
    union: dict[str, Any] = {}
    legacy: list[Any] = []
    for episode in episodes:
        if episode.trajectory_format == FORMAT_FULL:
            legacy.append(episode)
            continue
        for node_id, record in header_of(episode.decision_trace).get(_NODES, []):
            union[str(node_id)] = record
    lines: list[str] = []
    if union:
        lines.append("  node-table (written once for this feedback block; the steps below "
                     "refer to these ids):")
        for node_id in sorted(union, key=lambda k: (union[k].get("creation_index", 0), k)):
            lines.append("    " + json.dumps([node_id, union[node_id]], sort_keys=True))
    for episode in legacy:
        # Format 2 already carries a full observation per step; it needs no table.
        lines.append(f"  (legacy trace, rho={episode.rho}, world={episode.world_id})")
    return lines


def render_episode_steps(episode: Any) -> list[str]:
    """The episode's steps, referring to the ids of an already-emitted table."""
    if episode.trajectory_format == FORMAT_FULL:
        return [f"    decision={raw}" for raw in episode.decision_trace]
    return [f"    step {index}={raw}"
            for index, raw in enumerate(list(episode.decision_trace)[1:], start=1)]


def render_episode(episode: Any) -> list[str]:
    """Table plus steps, for one episode standing on its own."""
    """The episode's steps as prompt lines, with the node table written once.

    This is the function whose absence killed the first formal A arm. Format 2
    rendered one line per step, each embedding the whole observation, so the
    developer prompt grew with the square of the tree. The node records are
    emitted once here and each step refers to them by id -- the same information,
    laid out so that it does not have to be repeated 300 times.

    The step lines are the *encoded* steps, which decode back to the full
    observations (see :func:`decode_decisions`). Nothing is withheld: a reader
    who wants a step's complete observation can ask for it by id.
    """
    if episode.trajectory_format == FORMAT_FULL:
        return [f"    decision={raw}" for raw in episode.decision_trace]

    header = header_of(episode.decision_trace)
    lines = [
        "    node-table (each id once; steps below refer to these ids):"
    ]
    # Emitted oldest-first by creation_index so the table reads like the tree.
    records = sorted(header.get(_NODES, []),
                     key=lambda pair: (pair[1].get("creation_index", 0), pair[0]))
    for node_id, record in records:
        lines.append("      " + json.dumps([node_id, record], sort_keys=True))
    for index, raw in enumerate(list(episode.decision_trace)[1:], start=1):
        lines.append(f"    step {index}={raw}")
    return lines
