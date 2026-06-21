"""The online create-or-reuse environment.

At each round a query arrives. A policy returns a :class:`Decision` to either
REUSE an existing action or CREATE a new one. The environment scores the
decision, mutates the action library on CREATE, and returns a :class:`StepResult`.

Per-round cost
--------------
* REUSE action ``a``: ``total_loss = mismatch_loss(x, a)`` in ``[0, 1]``.
* CREATE: ``total_loss = creation_cost`` and the gold response for ``x`` is added
  to the library as a new reusable action (canonical query = the query text),
  available from the next round onwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .embeddings import Embedder
from .loss import LossModel


class DecisionKind(str, Enum):
    REUSE = "REUSE"
    CREATE = "CREATE"


@dataclass
class Action:
    """A reusable stored response."""

    action_id: str
    category: str
    canonical_query: str
    response_text: str
    embedding: np.ndarray
    created_round: int = -1  # -1 for seed FAQ actions
    quality: float = 0.0  # latent quality offset (hidden from the distance prior)

    @property
    def is_seed(self) -> bool:
        return self.created_round < 0


@dataclass
class Query:
    """An incoming user query."""

    query_id: str
    category: str
    text: str
    gold_response: str
    embedding: np.ndarray


@dataclass
class Decision:
    """A policy's choice for a round."""

    kind: DecisionKind
    action_index: Optional[int] = None  # required when kind == REUSE

    @classmethod
    def create(cls) -> "Decision":
        return cls(DecisionKind.CREATE, None)

    @classmethod
    def reuse(cls, action_index: int) -> "Decision":
        return cls(DecisionKind.REUSE, action_index)

    @property
    def is_create(self) -> bool:
        return self.kind == DecisionKind.CREATE


@dataclass
class StepResult:
    """Outcome of a single round."""

    round: int
    query_id: str
    category: str
    kind: DecisionKind
    chosen_action_id: str
    mismatch_loss: float
    creation_cost: float
    total_loss: float
    library_size: int
    n_created: int
    # cumulative trackers (filled in by the environment)
    cum_total_loss: float = 0.0
    cum_mismatch_loss: float = 0.0
    cum_creation_cost: float = 0.0


@dataclass
class CreateOrReuseEnv:
    """Stateful online environment for the create-or-reuse problem."""

    faq_df: pd.DataFrame
    stream_df: pd.DataFrame
    embedder: Embedder
    loss_model: LossModel = field(default_factory=LossModel)
    creation_cost: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        self._fit_embedder()
        self._seed_actions = self._build_seed_actions()
        self._queries = self._build_queries()
        self.reset()

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _fit_embedder(self) -> None:
        if not self.embedder.is_fitted:
            corpus = (
                list(self.faq_df["canonical_query"])
                + list(self.faq_df["response_text"])
                + list(self.stream_df["query_text"])
                + list(self.stream_df["gold_response"])
            )
            self.embedder.fit(corpus)

    def _build_seed_actions(self) -> list[Action]:
        embs = self.embedder.encode(list(self.faq_df["canonical_query"]))
        # latent per-action quality for the seed library (deterministic per seed).
        # Created actions are authored on demand and stay clean (quality 0).
        q_rng = np.random.default_rng(self.seed + 7777)
        qstd = self.loss_model.quality_std
        actions = []
        for row, emb in zip(self.faq_df.itertuples(index=False), embs):
            quality = float(q_rng.normal(0.0, qstd)) if qstd > 0 else 0.0
            actions.append(
                Action(
                    action_id=str(row.action_id),
                    category=str(row.category),
                    canonical_query=str(row.canonical_query),
                    response_text=str(row.response_text),
                    embedding=emb,
                    created_round=-1,
                    quality=quality,
                )
            )
        return actions

    def _build_queries(self) -> list[Query]:
        embs = self.embedder.encode(list(self.stream_df["query_text"]))
        queries = []
        for row, emb in zip(self.stream_df.itertuples(index=False), embs):
            queries.append(
                Query(
                    query_id=str(row.query_id),
                    category=str(row.category),
                    text=str(row.query_text),
                    gold_response=str(row.gold_response),
                    embedding=emb,
                )
            )
        return queries

    # ------------------------------------------------------------------ #
    # episode control
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        # fresh copy of the seed library so episodes are independent
        self.library: list[Action] = [
            Action(a.action_id, a.category, a.canonical_query, a.response_text,
                   a.embedding, a.created_round, a.quality)
            for a in self._seed_actions
        ]
        self.t = 0
        self.n_created = 0
        self.cum_total_loss = 0.0
        self.cum_mismatch_loss = 0.0
        self.cum_creation_cost = 0.0
        self._rng = np.random.default_rng(self.seed)

    @property
    def n_queries(self) -> int:
        return len(self._queries)

    @property
    def n_seed_actions(self) -> int:
        return len(self._seed_actions)

    def has_next(self) -> bool:
        return self.t < self.n_queries

    def current_query(self) -> Optional[Query]:
        return self._queries[self.t] if self.has_next() else None

    # ------------------------------------------------------------------ #
    # the step function
    # ------------------------------------------------------------------ #
    def step(self, decision: Decision) -> StepResult:
        if not self.has_next():
            raise RuntimeError("No more queries in the stream.")
        query = self._queries[self.t]

        # an empty library can only be served by creating
        if decision.kind == DecisionKind.REUSE and not self.library:
            decision = Decision.create()

        if decision.kind == DecisionKind.CREATE:
            new_action = Action(
                action_id=f"NEW-{self.t:04d}",
                category=query.category,
                canonical_query=query.text,
                response_text=query.gold_response,
                embedding=query.embedding,
                created_round=self.t,
            )
            self.library.append(new_action)
            self.n_created += 1
            mismatch_loss = 0.0
            creation_cost = float(self.creation_cost)
            chosen_action_id = new_action.action_id
        else:
            if decision.action_index is None or not (
                0 <= decision.action_index < len(self.library)
            ):
                raise ValueError(
                    f"Invalid REUSE index {decision.action_index} for library of "
                    f"size {len(self.library)}."
                )
            action = self.library[decision.action_index]
            mismatch_loss, _ = self.loss_model.realized_loss(
                query.embedding, query.category,
                action.embedding, action.category,
                self._rng,
                action_quality=action.quality,
            )
            creation_cost = 0.0
            chosen_action_id = action.action_id

        total_loss = mismatch_loss + creation_cost
        self.cum_total_loss += total_loss
        self.cum_mismatch_loss += mismatch_loss
        self.cum_creation_cost += creation_cost

        result = StepResult(
            round=self.t,
            query_id=query.query_id,
            category=query.category,
            kind=decision.kind,
            chosen_action_id=chosen_action_id,
            mismatch_loss=mismatch_loss,
            creation_cost=creation_cost,
            total_loss=total_loss,
            library_size=len(self.library),
            n_created=self.n_created,
            cum_total_loss=self.cum_total_loss,
            cum_mismatch_loss=self.cum_mismatch_loss,
            cum_creation_cost=self.cum_creation_cost,
        )
        self.t += 1
        return result
