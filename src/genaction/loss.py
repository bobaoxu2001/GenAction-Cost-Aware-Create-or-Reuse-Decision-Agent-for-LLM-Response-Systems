"""Mismatch-loss model for reusing an action on a query.

The realised cost of *reusing* a stored action ``a`` for a query ``x`` combines

1. a **semantic** term -- how far the query is from the action's canonical
   query in embedding space, passed through a convex ``distance_power`` so that
   small mismatches barely hurt while large ones are penalised heavily, and
2. a **category** term -- a flat penalty when the action's category differs from
   the query's category (a stored answer from the wrong intent is usually wrong
   regardless of surface similarity),

plus a little observation noise. The result is clipped to ``[0, 1]``.

The *creation* cost is a separate fixed constant ``c`` handled by the
environment; this module only scores reuse.

Information asymmetry (important for the learning policies)
----------------------------------------------------------
A decision policy can always measure the *semantic distance* (it has the
embeddings). It does **not** observe the category penalty or the noise before
acting -- those are only revealed through the realised loss after a reuse. This
is what makes the online estimation in :class:`DoublyOptimisticPolicy`
meaningful rather than a closed-form lookup.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two (assumed L2-normalised) vectors."""
    return float(np.dot(a, b))


def semantic_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Semantic distance in ``[0, 1]`` derived from cosine similarity."""
    return float(np.clip(1.0 - np.dot(a, b), 0.0, 1.0))


@dataclass
class LossModel:
    """Parameters of the reuse mismatch loss.

    Parameters
    ----------
    semantic_weight:
        Multiplier on the (powered) semantic distance term.
    category_penalty:
        Flat penalty added when categories differ.
    distance_power:
        Convexity of the distance -> loss curve. ``2.0`` means a query at
        distance ``0.3`` costs only ``0.09`` while one at ``0.9`` costs ``0.81``.
    noise_std:
        Standard deviation of zero-mean Gaussian observation noise.
    """

    semantic_weight: float = 1.0
    category_penalty: float = 0.4
    distance_power: float = 2.0
    noise_std: float = 0.03

    # ------------------------------------------------------------------ #
    def semantic_term(self, query_emb: np.ndarray, action_emb: np.ndarray) -> float:
        d = semantic_distance(query_emb, action_emb)
        return self.semantic_weight * (d ** self.distance_power)

    def base_loss(
        self,
        query_emb: np.ndarray,
        query_category: str,
        action_emb: np.ndarray,
        action_category: str,
    ) -> float:
        """Deterministic (noise-free) reuse loss in ``[0, 1]``."""
        loss = self.semantic_term(query_emb, action_emb)
        if query_category != action_category:
            loss += self.category_penalty
        return float(np.clip(loss, 0.0, 1.0))

    def realized_loss(
        self,
        query_emb: np.ndarray,
        query_category: str,
        action_emb: np.ndarray,
        action_category: str,
        rng: np.random.Generator,
    ) -> tuple[float, dict]:
        """Noisy reuse loss actually incurred, plus a breakdown of components."""
        sem = self.semantic_term(query_emb, action_emb)
        cat = self.category_penalty if query_category != action_category else 0.0
        noise = float(rng.normal(0.0, self.noise_std)) if self.noise_std > 0 else 0.0
        loss = float(np.clip(sem + cat + noise, 0.0, 1.0))
        components = {
            "semantic_distance": semantic_distance(query_emb, action_emb),
            "semantic_term": sem,
            "category_penalty": cat,
            "noise": noise,
            "loss": loss,
        }
        return loss, components
