"""Run every policy across a sweep of creation costs and save the results.

Usage
-----
    python scripts/run_experiments.py
    python scripts/run_experiments.py --backend tfidf --seed 0

Outputs (under ``results/``)
    experiment_results.csv       one row per (policy, creation_cost) with summary
                                 metrics.
    experiment_trajectories.csv  per-round trajectories (used by the cumulative
                                 plots).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

# make ``genaction`` importable without installing the package
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genaction import (  # noqa: E402
    AlwaysCreatePolicy,
    CreateOrReuseEnv,
    DoublyOptimisticPolicy,
    Embedder,
    FixedProbabilityPolicy,
    LossModel,
    NearestReusePolicy,
    run_episode,
)

DEFAULT_COSTS = [0.05, 0.10, 0.20, 0.35, 0.50]
DEFAULT_FIXED_PROBS = [0.25, 0.50]


def build_policies(creation_cost: float, seed: int, fixed_probs):
    """Construct a fresh set of policies for one creation cost."""
    policies = [
        AlwaysCreatePolicy(),
        NearestReusePolicy(),
    ]
    for p in fixed_probs:
        policies.append(FixedProbabilityPolicy(p=p, seed=seed + 1))
    policies.append(DoublyOptimisticPolicy(creation_cost=creation_cost))
    return policies


def _policy_p(policy) -> float | None:
    return getattr(policy, "p", None)


def run_experiment(
    faq_df: pd.DataFrame,
    stream_df: pd.DataFrame,
    creation_costs=DEFAULT_COSTS,
    fixed_probs=DEFAULT_FIXED_PROBS,
    backend: str = "tfidf",
    seed: int = 0,
):
    """Run the full sweep. Returns ``(summary_df, trajectory_df)``."""
    # one embedder, fitted once, shared across every run for comparability
    embedder = Embedder(backend=backend, seed=seed)

    summary_rows: list[dict] = []
    traj_frames: list[pd.DataFrame] = []

    for cost in creation_costs:
        env = CreateOrReuseEnv(
            faq_df=faq_df,
            stream_df=stream_df,
            embedder=embedder,
            loss_model=LossModel(),
            creation_cost=cost,
            seed=seed,
        )
        for policy in build_policies(cost, seed, fixed_probs):
            result = run_episode(env, policy)
            row = dict(result.summary)
            row["p"] = _policy_p(policy)
            row["backend"] = env.embedder.backend
            row["seed"] = seed
            summary_rows.append(row)

            traj = result.steps.copy()
            traj.insert(0, "policy", policy.name)
            traj.insert(1, "cost_setting", cost)
            traj_frames.append(traj)

    summary_df = pd.DataFrame(summary_rows)
    trajectory_df = pd.concat(traj_frames, ignore_index=True)
    return summary_df, trajectory_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="tfidf",
                        choices=["auto", "tfidf", "sentence-transformers"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    faq_df = pd.read_csv(data_dir / "sample_faq_library.csv")
    stream_df = pd.read_csv(data_dir / "sample_stream.csv")

    summary_df, trajectory_df = run_experiment(
        faq_df, stream_df, backend=args.backend, seed=args.seed
    )

    summary_path = results_dir / "experiment_results.csv"
    traj_path = results_dir / "experiment_trajectories.csv"
    summary_df.to_csv(summary_path, index=False)
    trajectory_df.to_csv(traj_path, index=False)

    backend_used = summary_df["backend"].iloc[0]
    print(f"Embedding backend: {backend_used}")
    print(f"Saved summary       -> {summary_path}  ({len(summary_df)} rows)")
    print(f"Saved trajectories  -> {traj_path}  ({len(trajectory_df)} rows)")
    print()

    cols = ["policy", "creation_cost", "avg_loss", "cum_mismatch_loss",
            "cum_creation_cost", "n_created", "create_rate"]
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
