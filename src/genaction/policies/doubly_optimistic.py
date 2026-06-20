"""The Doubly Optimistic create-or-reuse policy (the main method).

Inspired by Jianyu Xu's work on online decision making with generative action
sets, in particular the create-or-reuse setting. We do **not** reproduce the
paper's theorems; we implement a clean, explainable LCB/UCB rule with the same
flavour.

Per-action estimate
-------------------
For the current query ``x`` and each library action ``a_i`` we maintain an
estimate of the reuse mismatch loss as a small Bayesian-style update:

    mu_i  = (k0 * prior_i + sum_j w_ij * loss_ij) / n_i
    n_i   = k0 + sum_j w_ij
    sigma_i = sigma0 / sqrt(n_i)

* ``prior_i = d(x, a_i) ** prior_power`` is the policy's *belief before any
  feedback*, computed from the always-available semantic distance. (The policy
  does **not** know the hidden category penalty or noise in the true loss.)
* ``w_ij = exp(-(d(x, x_ij) / bandwidth)^2)`` weights past reuses of ``a_i`` by
  how similar their query ``x_ij`` was to the current query ``x``. Nearby
  feedback counts more; uncertainty shrinks where we have relevant experience.
* ``k0 = min_observations`` is a prior pseudo-count anchoring the estimate to
  ``prior_i`` until real feedback accumulates.

Two confidence bounds (the "doubly optimistic" part)
----------------------------------------------------
    LCB_i = mu_i - alpha_lcb * sigma_i
    UCB_i = mu_i + beta_ucb  * sigma_i

1. **Optimistic selection (LCB).** Pick ``i* = argmin_i LCB_i`` -- give actions
   the benefit of the doubt so genuinely promising (but under-observed) ones get
   a chance to be tried and learned about.
2. **Confidence-aware creation (UCB).** Reuse only if we are confident it beats
   creating: if ``UCB_{i*} > creation_cost`` the best candidate could plausibly
   cost more than a guaranteed-good new action, so CREATE; otherwise REUSE ``i*``.

Intuitively: create when reuse is either *expected* to be poor (high ``mu``) or
*too uncertain to trust* (high ``sigma``) relative to the price of a new
reusable action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..environment import Action, Decision, Query, StepResult
from ..loss import semantic_distance
from .base import Policy


@dataclass
class ActionEstimate:
    index: int
    mu: float
    sigma: float
    lcb: float
    ucb: float
    prior: float
    eff_n: float


class DoublyOptimisticPolicy(Policy):
    def __init__(
        self,
        creation_cost: float = 0.2,
        alpha_lcb: float = 1.0,
        beta_ucb: float = 1.0,
        min_observations: float = 4.0,
        distance_bandwidth: float = 0.4,
        sigma0: float = 0.2,
        prior_power: float = 2.0,
    ) -> None:
        self.creation_cost = float(creation_cost)
        self.alpha_lcb = float(alpha_lcb)
        self.beta_ucb = float(beta_ucb)
        self.min_observations = float(min_observations)
        self.distance_bandwidth = float(distance_bandwidth)
        self.sigma0 = float(sigma0)
        self.prior_power = float(prior_power)
        self.name = "DoublyOptimistic"
        self.reset()

    def reset(self) -> None:
        # action_id -> list of (query_embedding, observed_loss)
        self._obs: dict[str, list[tuple[np.ndarray, float]]] = {}
        # populated each round for inspection / the demo
        self.last_estimates: list[ActionEstimate] = []
        self.last_choice: int | None = None

    # ------------------------------------------------------------------ #
    def _estimate(self, query: Query, index: int, action: Action) -> ActionEstimate:
        prior = semantic_distance(query.embedding, action.embedding) ** self.prior_power

        obs = self._obs.get(action.action_id, [])
        sum_w = 0.0
        sum_wl = 0.0
        for past_emb, past_loss in obs:
            d = semantic_distance(query.embedding, past_emb)
            w = float(np.exp(-((d / self.distance_bandwidth) ** 2)))
            sum_w += w
            sum_wl += w * past_loss

        eff_n = self.min_observations + sum_w
        mu = (self.min_observations * prior + sum_wl) / eff_n
        sigma = self.sigma0 / np.sqrt(eff_n)
        return ActionEstimate(
            index=index,
            mu=mu,
            sigma=sigma,
            lcb=mu - self.alpha_lcb * sigma,
            ucb=mu + self.beta_ucb * sigma,
            prior=prior,
            eff_n=eff_n,
        )

    def estimate_all(self, query: Query, library: Sequence[Action]) -> list[ActionEstimate]:
        return [self._estimate(query, i, a) for i, a in enumerate(library)]

    # ------------------------------------------------------------------ #
    def act(self, query: Query, library: Sequence[Action]) -> Decision:
        if not library:
            self.last_estimates = []
            self.last_choice = None
            return Decision.create()

        estimates = self.estimate_all(query, library)
        self.last_estimates = estimates

        # 1) optimistic selection via LCB
        best = min(estimates, key=lambda e: e.lcb)
        self.last_choice = best.index

        # 2) confidence-aware creation via UCB
        if best.ucb > self.creation_cost:
            return Decision.create()
        return Decision.reuse(best.index)

    def update(
        self,
        query: Query,
        decision: Decision,
        result: StepResult,
        library: Sequence[Action],
    ) -> None:
        # we only learn about an action by actually reusing it
        if decision.kind.value == "REUSE" and decision.action_index is not None:
            action = library[decision.action_index]
            self._obs.setdefault(action.action_id, []).append(
                (query.embedding, float(result.mismatch_loss))
            )
