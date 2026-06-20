"""Baseline: always create a fresh action.

Upper bound on quality (mismatch loss is always zero because the gold response
is stored verbatim) and upper bound on creation spend (pays ``c`` every round).
"""

from __future__ import annotations

from typing import Sequence

from ..environment import Action, Decision, Query
from .base import Policy


class AlwaysCreatePolicy(Policy):
    name = "AlwaysCreate"

    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        return Decision.create()
