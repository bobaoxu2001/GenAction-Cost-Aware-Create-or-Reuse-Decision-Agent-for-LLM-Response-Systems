"""Tests for the create-or-reuse environment."""

from __future__ import annotations

from genaction import Decision


def test_seed_library_loaded(env):
    assert env.n_seed_actions == 12
    assert len(env.library) == 12
    assert env.n_queries == 66


def test_create_adds_reusable_action(env):
    query = env.current_query()
    size_before = len(env.library)

    result = env.step(Decision.create())

    assert result.kind.value == "CREATE"
    assert len(env.library) == size_before + 1
    assert result.mismatch_loss == 0.0
    assert result.creation_cost == env.creation_cost
    assert result.total_loss == env.creation_cost

    # the new action stores the gold response and is keyed on the query text
    new_action = env.library[-1]
    assert new_action.canonical_query == query.text
    assert new_action.response_text == query.gold_response
    assert new_action.category == query.category
    assert not new_action.is_seed


def test_reuse_does_not_grow_library(env):
    size_before = len(env.library)
    result = env.step(Decision.reuse(0))

    assert result.kind.value == "REUSE"
    assert len(env.library) == size_before
    assert result.creation_cost == 0.0
    assert 0.0 <= result.mismatch_loss <= 1.0
    assert result.total_loss == result.mismatch_loss


def test_cumulative_accounting_matches_step_sum(env):
    totals, mismatches, creations = [], [], []
    for i in range(10):
        decision = Decision.create() if i % 2 == 0 else Decision.reuse(0)
        r = env.step(decision)
        totals.append(r.total_loss)
        mismatches.append(r.mismatch_loss)
        creations.append(r.creation_cost)

    assert abs(env.cum_total_loss - sum(totals)) < 1e-9
    assert abs(env.cum_mismatch_loss - sum(mismatches)) < 1e-9
    assert abs(env.cum_creation_cost - sum(creations)) < 1e-9
    assert abs(env.cum_total_loss
               - (env.cum_mismatch_loss + env.cum_creation_cost)) < 1e-9


def test_reset_restores_seed_library(env):
    for _ in range(5):
        env.step(Decision.create())
    assert len(env.library) > env.n_seed_actions

    env.reset()
    assert len(env.library) == env.n_seed_actions
    assert env.t == 0
    assert env.n_created == 0
    assert env.cum_total_loss == 0.0


def test_reuse_on_empty_library_coerced_to_create(env):
    env.library = []  # force the empty-library edge case
    result = env.step(Decision.reuse(0))
    assert result.kind.value == "CREATE"
    assert len(env.library) == 1


def test_self_reuse_has_low_loss(env):
    """Reusing an action on its own canonical query should be (near) free."""
    seed = env.library[0]
    loss, comps = env.loss_model.realized_loss(
        seed.embedding, seed.category, seed.embedding, seed.category, env._rng
    )
    assert comps["semantic_distance"] < 1e-6
    assert loss < 0.15  # only observation noise remains
