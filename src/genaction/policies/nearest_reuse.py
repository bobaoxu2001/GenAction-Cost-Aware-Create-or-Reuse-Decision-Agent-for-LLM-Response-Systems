"""Baseline: always reuse the nearest existing action.

Never pays a creation cost, so the library never grows. It is the natural
"cheap" baseline whose quality is limited by how well the initial FAQ library
covers the incoming stream.
"""

from __future__ import annotations

from typing import Sequence

from ..environment import Action, Decision, Query
from .base import Policy, nearest_index


class NearestReusePolicy(Policy):
    name = "NearestReuse"

    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        if not library:
            return Decision.create()
        return Decision.reuse(nearest_index(query, library))
