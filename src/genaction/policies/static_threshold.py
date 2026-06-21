"""Ablation baseline: a myopic cost-aware threshold.

This is the natural ablation of :class:`DoublyOptimisticPolicy`: keep the
cost-aware comparison, but remove the two ingredients that make the main method
"doubly optimistic" -- the confidence bounds (no LCB/UCB exploration) and the
online learning (no feedback is ever used).

For the current query it reuses the nearest action when that action's *prior*
estimated loss is at most the creation cost, and creates otherwise:

    create  iff  d(x, a_nearest) ** prior_power  >  creation_cost

It is a strong, sensible baseline -- and the right thing to compare against to
answer "does the confidence-bound machinery actually earn its keep, or would a
plain cost-aware threshold do?". The experiments show it is competitive when the
distance->loss prior is well specified, but degrades when the prior is
mis-specified, exactly where the learned method stays robust.
"""

from __future__ import annotations

from typing import Sequence

from ..environment import Action, Decision, Query
from ..loss import semantic_distance
from .base import Policy, nearest_index


class StaticThresholdPolicy(Policy):
    def __init__(self, creation_cost: float = 0.2, prior_power: float = 2.0) -> None:
        self.creation_cost = float(creation_cost)
        self.prior_power = float(prior_power)
        self.name = "StaticThreshold"

    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        if not library:
            return Decision.create()
        i = nearest_index(query, library)
        prior_loss = semantic_distance(query.embedding, library[i].embedding) ** self.prior_power
        if prior_loss > self.creation_cost:
            return Decision.create()
        return Decision.reuse(i)
