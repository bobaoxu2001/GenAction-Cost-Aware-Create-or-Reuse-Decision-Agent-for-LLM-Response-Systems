"""Tests for the create-or-reuse policies."""

from __future__ import annotations

from genaction import (
    AlwaysCreatePolicy,
    DoublyOptimisticPolicy,
    FixedProbabilityPolicy,
    NearestReusePolicy,
    StaticThresholdPolicy,
    run_episode,
)
from genaction.policies.base import nearest_index


def test_nearest_index_picks_exact_match(env):
    # a query equal to a seed action's canonical query is its own nearest action
    seed = env.library[3]
    q = env.current_query()
    q.embedding = seed.embedding  # pretend the query is identical to the action
    assert nearest_index(q, env.library) == 3


def test_always_create_always_creates(env):
    policy = AlwaysCreatePolicy()
    result = run_episode(env, policy)
    assert (result.steps["kind"] == "CREATE").all()
    assert result.summary["n_created"] == env.n_queries
    assert result.summary["create_rate"] == 1.0
    assert result.summary["cum_mismatch_loss"] == 0.0


def test_nearest_reuse_never_creates(env):
    policy = NearestReusePolicy()
    result = run_episode(env, policy)
    assert (result.steps["kind"] == "REUSE").all()
    assert result.summary["n_created"] == 0
    assert result.summary["final_library_size"] == env.n_seed_actions
    # every chosen action is a real, pre-existing library id
    assert result.steps["chosen_action_id"].str.startswith("FAQ-").all()


def test_fixed_probability_extremes(env):
    never = run_episode(env, FixedProbabilityPolicy(p=0.0, seed=0))
    assert never.summary["n_created"] == 0

    always = run_episode(env, FixedProbabilityPolicy(p=1.0, seed=0))
    assert always.summary["n_created"] == env.n_queries


def test_fixed_probability_rejects_bad_p():
    import pytest

    with pytest.raises(ValueError):
        FixedProbabilityPolicy(p=1.5)


def test_doubly_optimistic_high_cost_never_creates(env):
    # creation_cost above the maximum possible UCB => always reuse
    env.creation_cost = 10.0
    policy = DoublyOptimisticPolicy(creation_cost=10.0)
    result = run_episode(env, policy)
    assert result.summary["n_created"] == 0
    assert (result.steps["kind"] == "REUSE").all()


def test_doubly_optimistic_zero_cost_always_creates(env):
    # creation is free => UCB (strictly positive) always exceeds it
    env.creation_cost = 0.0
    policy = DoublyOptimisticPolicy(creation_cost=0.0)
    result = run_episode(env, policy)
    assert result.summary["n_created"] == env.n_queries


def test_doubly_optimistic_beats_baselines_at_mid_cost(env):
    """At a moderate cost the adaptive policy should not be worse than the
    naive baselines (it is in fact strictly better on the sample data)."""
    env.creation_cost = 0.2
    do = run_episode(env, DoublyOptimisticPolicy(creation_cost=0.2))

    env.creation_cost = 0.2
    ac = run_episode(env, AlwaysCreatePolicy())
    env.creation_cost = 0.2
    nr = run_episode(env, NearestReusePolicy())

    assert do.summary["avg_loss"] <= ac.summary["avg_loss"]
    assert do.summary["avg_loss"] <= nr.summary["avg_loss"]


def test_doubly_optimistic_create_rate_decreases_with_cost(env):
    """The signature adaptive behaviour: create less when creation is pricier."""
    rates = []
    for c in [0.05, 0.2, 0.5]:
        env.creation_cost = c
        res = run_episode(env, DoublyOptimisticPolicy(creation_cost=c))
        rates.append(res.summary["create_rate"])
    assert rates[0] >= rates[1] >= rates[2]


def test_static_threshold_high_cost_never_creates(env):
    # prior loss is at most 1, so a cost above 1 means reuse always
    env.creation_cost = 2.0
    res = run_episode(env, StaticThresholdPolicy(creation_cost=2.0))
    assert res.summary["n_created"] == 0
    assert (res.steps["kind"] == "REUSE").all()


def test_static_threshold_create_rate_decreases_with_cost(env):
    rates = []
    for c in [0.05, 0.2, 0.5]:
        env.creation_cost = c
        res = run_episode(env, StaticThresholdPolicy(creation_cost=c))
        rates.append(res.summary["create_rate"])
    assert rates[0] >= rates[1] >= rates[2]


def test_static_threshold_is_a_strong_cost_aware_baseline(env):
    """It should beat the cost-unaware baselines at a moderate cost."""
    env.creation_cost = 0.2
    st = run_episode(env, StaticThresholdPolicy(creation_cost=0.2))
    env.creation_cost = 0.2
    nr = run_episode(env, NearestReusePolicy())
    assert st.summary["avg_loss"] < nr.summary["avg_loss"]
