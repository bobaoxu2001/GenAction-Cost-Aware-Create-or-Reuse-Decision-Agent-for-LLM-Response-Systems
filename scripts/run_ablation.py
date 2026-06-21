"""Ablation: does the 'doubly optimistic' machinery earn its keep?

The DoublyOptimistic policy adds two things on top of a plain cost-aware
decision: (i) confidence bounds (LCB selection + a UCB creation test) and
(ii) online learning from realised reuse losses. The natural question a reviewer
asks is whether that complexity is necessary, or whether simply thresholding the
distance-based prior against the creation cost would do.

This script compares :class:`DoublyOptimisticPolicy` against the myopic
:class:`StaticThresholdPolicy` across creation costs, under two priors:

* a **well-specified** prior (``prior_power=2`` -- matches the convex
  distance->loss curve used by the environment), and
* a **mis-specified** prior (``prior_power=1`` -- a wrong, linear assumption).

Headline finding (sample data): on this stationary, low-noise, distance-
informative benchmark the myopic threshold is a strong baseline that slightly
*beats* the confidence-bound policy -- the "optimism" mostly buys exploratory
creations that do not pay off when the prior is already informative. The gap is
small and persists under mis-specification. Both cost-aware policies dominate the
cost-unaware baselines by a wide margin (see ``run_experiments.py``).

Interpretation: *cost-awareness* is the dominant lever here; the value of
optimism/learning is expected to appear in harder regimes (noisy or ambiguous
feedback, weakly-informative priors, non-stationary streams), which we flag as
future work rather than claim here.

Usage::

    python scripts/run_ablation.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from genaction import (  # noqa: E402
    CreateOrReuseEnv,
    DoublyOptimisticPolicy,
    Embedder,
    LossModel,
    StaticThresholdPolicy,
    run_episode,
)

DEFAULT_COSTS = [0.05, 0.10, 0.20, 0.35, 0.50]
PRIOR_POWERS = [2.0, 1.0]  # well-specified, mis-specified


def run_ablation(faq_df, stream_df, creation_costs=DEFAULT_COSTS,
                 prior_powers=PRIOR_POWERS, backend="tfidf", seed=0):
    """Return a tidy DataFrame of avg loss for both cost-aware policies."""
    embedder = Embedder(backend=backend, seed=seed)
    rows = []
    for prior_power in prior_powers:
        for cost in creation_costs:
            env = CreateOrReuseEnv(faq_df, stream_df, embedder, LossModel(),
                                   creation_cost=cost, seed=seed)
            policies = [
                StaticThresholdPolicy(creation_cost=cost, prior_power=prior_power),
                DoublyOptimisticPolicy(creation_cost=cost, prior_power=prior_power),
            ]
            for policy in policies:
                s = run_episode(env, policy).summary
                rows.append({
                    "prior_power": prior_power,
                    "prior": "well-specified (p=2)" if prior_power == 2.0
                    else "mis-specified (p=1)",
                    "creation_cost": cost,
                    "policy": policy.name,
                    "avg_loss": s["avg_loss"],
                    "cum_mismatch_loss": s["cum_mismatch_loss"],
                    "n_created": s["n_created"],
                    "create_rate": s["create_rate"],
                })
    return pd.DataFrame(rows)


def plot_ablation(df: pd.DataFrame, out: pathlib.Path) -> None:
    priors = list(df["prior"].unique())
    fig, axes = plt.subplots(1, len(priors), figsize=(11, 4.6), sharey=True)
    if len(priors) == 1:
        axes = [axes]
    style = {
        "StaticThreshold": dict(color="#8c564b", marker="D"),
        "DoublyOptimistic": dict(color="#2ca02c", marker="o"),
    }
    for ax, prior in zip(axes, priors):
        sub = df[df["prior"] == prior]
        for policy in ["StaticThreshold", "DoublyOptimistic"]:
            d = sub[sub["policy"] == policy].sort_values("creation_cost")
            st = style[policy]
            ax.plot(d["creation_cost"], d["avg_loss"], color=st["color"],
                    marker=st["marker"], linewidth=2, label=policy)
        ax.set_title(f"Prior: {prior}")
        ax.set_xlabel("Creation cost  c")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Average total cost per query")
    fig.suptitle("Ablation: a myopic cost-aware threshold vs. the confidence-bound policy",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--backend", default="tfidf")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    faq_df = pd.read_csv(data_dir / "sample_faq_library.csv")
    stream_df = pd.read_csv(data_dir / "sample_stream.csv")

    df = run_ablation(faq_df, stream_df, backend=args.backend, seed=args.seed)
    csv_path = results_dir / "ablation_results.csv"
    fig_path = results_dir / "ablation_prior_sensitivity.png"
    df.to_csv(csv_path, index=False)
    plot_ablation(df, fig_path)

    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}\n")

    # compact pivot: avg_loss by (prior, policy) x cost
    pivot = df.pivot_table(index=["prior", "policy"], columns="creation_cost",
                           values="avg_loss").round(3)
    with pd.option_context("display.width", 120):
        print(pivot.to_string())

    # the "cost of optimism": DoublyOptimistic minus StaticThreshold
    print("\nCost of optimism (DoublyOptimistic - StaticThreshold), avg over costs:")
    for prior, g in df.groupby("prior"):
        piv = g.pivot_table(index="creation_cost", columns="policy", values="avg_loss")
        gap = (piv["DoublyOptimistic"] - piv["StaticThreshold"]).mean()
        print(f"  {prior:24s}: {gap:+.4f}")


if __name__ == "__main__":
    main()
