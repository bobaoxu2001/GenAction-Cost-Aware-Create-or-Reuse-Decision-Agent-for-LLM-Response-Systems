"""Base class and shared helpers for create-or-reuse policies.

A policy implements two methods:

* :meth:`act` -- given the current query and the (pre-step) action library,
  return a :class:`~genaction.environment.Decision`.
* :meth:`update` -- given the realised :class:`~genaction.environment.StepResult`,
  update any internal statistics. The default is a no-op (memoryless policies).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..environment import Action, Decision, Query, StepResult
from ..loss import semantic_distance


def nearest_index(query: Query, library: Sequence[Action]) -> int:
    """Index of the library action closest to ``query`` in embedding space."""
    dists = [semantic_distance(query.embedding, a.embedding) for a in library]
    return int(np.argmin(dists))


class Policy:
    """Abstract base policy."""

    #: human-readable name used in result tables and plots
    name: str = "policy"

    def reset(self) -> None:
        """Clear any internal state between episodes."""

    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        raise NotImplementedError

    def update(
        self,
        query: Query,
        decision: Decision,
        result: StepResult,
        library: Sequence[Action],
    ) -> None:
        """Observe the outcome of a round (default: stateless)."""
        return None
