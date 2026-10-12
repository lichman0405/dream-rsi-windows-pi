"""Method C's frozen policy: one rewrite of the best-scoring legal leaf at a time.

This file is the auxiliary control in the MOF screening experiment. It is
*method C*, not the Dream-RSI arm and not the incumbent:

* **A (Dream-RSI)** learns `pi_t` from the sealed history with the policy
  developer in the loop.
* **B (frozen initial policy)** freezes `policies/incumbent.py` for the whole
  run; it is the primary control.
* **C (frozen serial policy)** is this file: a fixed greedy rule that never
  learns anything, run at `W = 1` so that it measures *search order* rather than
  search order plus batching.

The rule, fixed before any result was seen:

1. The best legal leaf is the one with the highest observed score. Ties break on
   the older `creation_index` (smaller first) and then on node id, so the choice
   is a function of the observed prefix alone and rebuilds identically on replay.
2. With no legal leaf -- which is the first round of every world, when the root
   is the whole tree -- expand the root instead. That is the only way the tree
   ever gains a second branch.
3. Return at most one parent. `W` is 1 for method C, so a longer batch would be
   rejected outright rather than silently truncated.

Contract (the only interface a policy may rely on)
--------------------------------------------------
``select(observation, rho) -> list[str]``

``observation`` carries ``nodes`` (already-revealed ``PublicNode``s),
``legal_actions`` (root plus current leaves), ``completed_rounds``,
``max_parallelism`` and ``node_ids``. ``rho`` is accepted and ignored: this
policy is deliberately budget-indifferent, and saying so is more honest than
pretending to sweep a knob it does not consult.

Return only ids from ``observation.legal_actions``, never a duplicate, never
more than ``max_parallelism``. Return ``[]`` to stop. Nothing hidden is passed
in, and nothing hidden may be inferred from here.
"""
from __future__ import annotations

from typing import Any


def select(observation: Any, rho: float) -> list[str]:
    del rho  # frozen serial policy: the budget knob does not change its rule
    legal = list(observation.legal_actions)
    if not legal:
        return []

    by_id = {node.node_id: node for node in observation.nodes}
    root_id = observation.node_ids[0] if observation.node_ids else None
    leaves = [node_id for node_id in legal if node_id != root_id]

    def rank(node_id: str) -> tuple[float, int, str]:
        node = by_id.get(node_id)
        if node is None:
            return (0.0, 0, node_id)
        # Highest score first; then the older node; then the id. Fixed, total,
        # and independent of the order the ids arrived in.
        return (float(node.score), -int(node.creation_index), node_id)

    if leaves:
        return [max(leaves, key=rank)]
    if root_id is not None and root_id in legal:
        return [root_id]
    return []
