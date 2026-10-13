"""The shared-node decision trace must lose nothing.

The encoding exists to stop the policy-development prompt from growing with the
square of the tree -- the first formal A arm was rejected by the provider at
1,143,191 tokens against a 1,048,576 window. The whole point is that it buys that
space by *deduplicating* rather than by discarding, so these tests are mostly
about the inverse: every step's complete observation must come back, byte for
byte, including the fields nothing currently reads.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dream_rsi.domain.models import ReplayEpisode
from dream_rsi.runtime.trajectory import (
    CURRENT_FORMAT,
    FORMAT_FULL,
    decode_any,
    decode_decisions,
    encode_decisions,
    header_of,
    iter_batches,
    node_table,
    render_episode_steps,
    render_node_table,
)


def _node(index: int, *, world: str = "w0") -> dict:
    return {
        "node_id": f"node-{world}-{index:032x}",
        "parent_id": None if index == 0 else f"node-{world}-{index - 1:032x}",
        "creation_index": index,
        "depth": index,
        "score": index / 10.0,
        "evaluator_status": "SUCCESS",
        "error_type": None,
        "summary": f"candidate {index} " + "x" * 40,
        "observation_json": json.dumps({"diagnostics": {"k": 10}, "index": index}),
    }


def _rich_steps(count: int) -> list[dict]:
    """Steps shaped exactly like replay_world builds them."""
    steps = []
    revealed: list[dict] = []
    for index in range(count):
        visible = [_node(0)] + revealed
        fresh = _node(index + 1)
        steps.append({
            "observation": {
                "nodes": list(visible),
                "node_ids": [n["node_id"] for n in visible],
                "legal_actions": [n["node_id"] for n in visible[-2:]],
                "completed_rounds": index,
                "max_parallelism": 4,
            },
            "batch": [visible[-1]["node_id"]],
            "revealed": [fresh],
        })
        revealed.append(fresh)
    # the stopping decision: an empty batch, still carrying its observation
    visible = [_node(0)] + revealed
    steps.append({
        "observation": {"nodes": list(visible), "node_ids": [n["node_id"] for n in visible],
                        "legal_actions": [visible[-1]["node_id"]],
                        "completed_rounds": count, "max_parallelism": 4},
        "batch": [], "revealed": [],
    })
    return steps


def test_every_step_decodes_back_to_the_same_observation() -> None:
    """Lossless, including the empty-batch stopping step."""
    rich = _rich_steps(12)
    encoded = encode_decisions(rich)
    assert decode_decisions(encoded) == rich


def test_nothing_is_dropped_on_the_grounds_that_replay_ignores_it() -> None:
    """Replay only reads `batch`; the other fields must survive anyway."""
    rich = _rich_steps(6)
    decoded = decode_decisions(encode_decisions(rich))
    for original, restored in zip(rich, decoded, strict=True):
        assert restored["observation"]["nodes"] == original["observation"]["nodes"]
        assert restored["revealed"] == original["revealed"]
        assert restored["observation"]["legal_actions"] == original["observation"]["legal_actions"]
        # a field nothing consumes today, kept deliberately
        assert restored["observation"]["max_parallelism"] == original["observation"]["max_parallelism"]
        assert restored["observation"]["completed_rounds"] == original["observation"]["completed_rounds"]


def test_the_node_records_are_written_once() -> None:
    encoded = encode_decisions(_rich_steps(20))
    table = node_table(encoded)
    assert len(table) == 21  # root plus twenty reveals
    size = sum(len(json.dumps(entry)) for entry in table.values())
    repeated = sum(len(json.dumps(node))
                   for step in _rich_steps(20)
                   for node in step["observation"]["nodes"])
    assert repeated > 6 * size, "the fixture must actually exhibit the duplication"


def test_batches_iterate_for_both_formats() -> None:
    rich = _rich_steps(5)
    encoded = encode_decisions(rich)
    episode = ReplayEpisode(
        episode_id="e", world_id="w", policy_version_id="p", rho=0.5, n=5, kstar=5,
        best_score=1.0, attained_quality=1.0, selected_batches=(), revealed_order=(),
        stop_reason=__import__("dream_rsi.domain.models", fromlist=["StopReason"]).StopReason.MAX_ROUNDS,
        replay_digest="d", trajectory_format=CURRENT_FORMAT, decision_trace=encoded,
    )
    assert list(iter_batches(episode)) == [step["batch"] for step in rich]

    legacy = replace(episode, trajectory_format=FORMAT_FULL,
                     decision_trace=tuple(json.dumps(s, sort_keys=True) for s in rich))
    assert list(iter_batches(legacy)) == [step["batch"] for step in rich]
    # a legacy trace is already rich, and decodes to itself
    assert decode_any(legacy) == rich


def test_the_rendered_block_is_far_smaller_than_the_steps_it_replaces() -> None:
    rich = _rich_steps(200)
    encoded = encode_decisions(rich)
    episode = ReplayEpisode(
        episode_id="e", world_id="w", policy_version_id="p", rho=0.5, n=200, kstar=200,
        best_score=1.0, attained_quality=1.0, selected_batches=(), revealed_order=(),
        stop_reason=__import__("dream_rsi.domain.models", fromlist=["StopReason"]).StopReason.MAX_ROUNDS,
        replay_digest="d", trajectory_format=CURRENT_FORMAT, decision_trace=encoded,
    )
    # The comparison that matters is against what the prompt used to carry: one
    # line per step, each embedding that step's whole observation.
    old = sum(len(f"    decision={json.dumps(step, sort_keys=True)}") for step in rich)
    new = sum(len(line) for line in render_node_table([episode]))
    new += sum(len(line) for line in render_episode_steps(episode))
    assert new * 4 < old, f"only {old / new:.1f}x smaller; expected a large reduction"


# ------------------------------------------------------- measured against real data


FORMAL_TRACES = Path(".runtime/runs")
WINDOW_TOKENS = 1_048_576
#: Calibrated on the request the provider rejected: 3,812,746 characters that
#: were 1,143,191 tokens. Using bytes/3 would overstate the prompt by ~15%.
BYTES_PER_TOKEN = 3.335


@pytest.mark.skipif(not FORMAL_TRACES.is_dir(),
                    reason="no stored run traces to measure against")
def test_the_aborted_formal_feedback_fits_after_re_encoding() -> None:
    """The shape that failed, measured on the traces it actually produced.

    This is the regression the encoding exists to prevent, so it is asserted in
    the units that failed: estimated tokens against the window, using the ratio
    calibrated on the rejected request itself.
    """
    import collections

    from dream_rsi.storage.sqlite_registry import Registry

    registry = Registry(".runtime/registry.sqlite3")
    try:
        episodes = [e for e in registry.list_replays()
                    if e.world_id.startswith("mof-formal-s101-r1-A")]
    finally:
        registry.close()
    if not episodes:
        pytest.skip("the aborted formal A run has no stored traces on this machine")

    class ReEncoded:
        def __init__(self, episode):  # noqa: ANN001
            self.decision_trace = encode_decisions(
                [json.loads(raw) for raw in episode.decision_trace])
            self.trajectory_format = CURRENT_FORMAT
            self.rho = episode.rho
            self.world_id = episode.world_id
            self.policy_version_id = episode.policy_version_id

    reencoded = [ReEncoded(e) for e in episodes]
    by_policy: dict[str, list] = collections.OrderedDict()
    for episode in reencoded:
        by_policy.setdefault(episode.policy_version_id, []).append(episode)

    emitted: set[str] = set()
    total = 0
    for group in by_policy.values():
        by_world: dict[str, list] = collections.OrderedDict()
        for episode in group:
            by_world.setdefault(episode.world_id, []).append(episode)
        for world_id, world_episodes in by_world.items():
            ordered = sorted(world_episodes, key=lambda e: e.rho)
            if world_id not in emitted:
                total += sum(len(line) for line in render_node_table(ordered))
                emitted.add(world_id)
            total += sum(sum(len(line) for line in render_episode_steps(e)) + 90
                         for e in ordered)

    estimated = total / BYTES_PER_TOKEN
    assert estimated < WINDOW_TOKENS, (
        f"re-encoded feedback is ~{estimated:,.0f} tokens against a "
        f"{WINDOW_TOKENS:,} window"
    )
    # And it must stay comfortably inside, not merely under: the real prompt also
    # carries the contract, the incumbent source and the output reservation.
    assert estimated < 0.5 * WINDOW_TOKENS
