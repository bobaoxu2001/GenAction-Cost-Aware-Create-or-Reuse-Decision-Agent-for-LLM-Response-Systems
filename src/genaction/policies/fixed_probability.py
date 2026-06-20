"""Baseline: create with a fixed probability ``p``, otherwise reuse nearest.

A cost-*unaware* stochastic baseline. It spends the same creation budget
regardless of whether a query is well covered, so it is the key foil for the
adaptive policy: matching its average create rate but not *where* it creates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..environment import Action, Decision, Query
from .base import Policy, nearest_index


class FixedProbabilityPolicy(Policy):
    def __init__(self, p: float = 0.3, seed: int = 0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1]")
        self.p = p
        self.seed = seed
        self.name = f"FixedProb(p={p:g})"
        self.reset()

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        if not library or self._rng.random() < self.p:
            return Decision.create()
        return Decision.reuse(nearest_index(query, library))
