"""Run a policy on the environment and summarise the trajectory."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .environment import CreateOrReuseEnv
from .policies.base import Policy


@dataclass
class EpisodeResult:
    policy_name: str
    creation_cost: float
    steps: pd.DataFrame  # one row per round
    summary: dict        # scalar metrics


def run_episode(env: CreateOrReuseEnv, policy: Policy) -> EpisodeResult:
    """Play one full pass over the stream with ``policy`` on ``env``."""
    env.reset()
    policy.reset()

    rows: list[dict] = []
    while env.has_next():
        query = env.current_query()
        library = env.library  # pre-step view
        decision = policy.act(query, library)
        result = env.step(decision)
        policy.update(query, decision, result, env.library)
        rows.append(
            {
                "round": result.round,
                "query_id": result.query_id,
                "category": result.category,
                "kind": result.kind.value,
                "chosen_action_id": result.chosen_action_id,
                "mismatch_loss": result.mismatch_loss,
                "creation_cost": result.creation_cost,
                "total_loss": result.total_loss,
                "library_size": result.library_size,
                "n_created": result.n_created,
                "cum_total_loss": result.cum_total_loss,
                "cum_mismatch_loss": result.cum_mismatch_loss,
                "cum_creation_cost": result.cum_creation_cost,
            }
        )

    steps = pd.DataFrame(rows)
    summary = summarize(steps, policy.name, env.creation_cost, env.n_seed_actions)
    return EpisodeResult(policy.name, env.creation_cost, steps, summary)


def summarize(
    steps: pd.DataFrame,
    policy_name: str,
    creation_cost: float,
    n_seed_actions: int,
) -> dict:
    """Aggregate a per-round trajectory into scalar metrics."""
    n_rounds = len(steps)
    n_created = int((steps["kind"] == "CREATE").sum())
    cum_total = float(steps["total_loss"].sum())
    cum_mismatch = float(steps["mismatch_loss"].sum())
    cum_creation = float(steps["creation_cost"].sum())
    return {
        "policy": policy_name,
        "creation_cost": float(creation_cost),
        "n_rounds": n_rounds,
        "cum_total_loss": cum_total,
        "cum_mismatch_loss": cum_mismatch,
        "cum_creation_cost": cum_creation,
        "n_created": n_created,
        "final_library_size": int(n_seed_actions + n_created),
        "avg_loss": cum_total / n_rounds if n_rounds else 0.0,
        "avg_mismatch_loss": cum_mismatch / n_rounds if n_rounds else 0.0,
        "create_rate": n_created / n_rounds if n_rounds else 0.0,
    }
