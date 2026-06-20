"""Smoke test: the experiment pipeline runs end-to-end on the sample data."""

from __future__ import annotations

import importlib.util
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_run_experiments():
    spec = importlib.util.spec_from_file_location(
        "run_experiments", ROOT / "scripts" / "run_experiments.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_experiment_pipeline(faq_df, stream_df):
    mod = _load_run_experiments()
    summary, traj = mod.run_experiment(
        faq_df,
        stream_df,
        creation_costs=[0.1, 0.3],
        fixed_probs=[0.3],
        backend="tfidf",
        seed=0,
    )

    # one row per (policy, cost): AlwaysCreate, NearestReuse, 1 FixedProb, DO
    assert len(summary) == 2 * 4
    assert not traj.empty

    expected_cols = {
        "policy", "creation_cost", "avg_loss", "cum_mismatch_loss",
        "cum_creation_cost", "n_created", "final_library_size", "create_rate",
    }
    assert expected_cols.issubset(summary.columns)

    # metrics are finite and within sane ranges
    n_rounds = len(stream_df)
    assert summary["avg_loss"].between(0.0, 1.0).all()
    assert summary["create_rate"].between(0.0, 1.0).all()
    assert summary["n_created"].between(0, n_rounds).all()

    # adaptive policy is no worse than naive baselines at c = 0.3
    at_cost = summary[summary["creation_cost"] == 0.3]
    do = at_cost[at_cost["policy"] == "DoublyOptimistic"]["avg_loss"].iloc[0]
    baselines = at_cost[at_cost["policy"].isin(["AlwaysCreate", "NearestReuse"])]
    assert do <= baselines["avg_loss"].min()
