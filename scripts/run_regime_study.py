"""Regime study: *when* does the confidence-bound policy beat the threshold?

The ablation (`scripts/run_ablation.py`) shows that on the easy, single-pass
benchmark a myopic cost-aware threshold slightly beats the DoublyOptimistic
policy. This script characterises the regime where the extra machinery pays off,
by turning on two things the easy benchmark lacks:

1. **Hidden structure** -- a latent per-action quality (``LossModel.quality_std``)
   that shifts an action's true reuse loss in a way the distance prior cannot
   predict but feedback can reveal. (Applied to seed actions; created actions are
   authored on demand and stay clean.)
2. **Sustained traffic** -- the stream is replayed ``reps`` times to simulate a
   higher-volume deployment, giving online learning enough feedback to act on.

Two sweeps (both at a fixed creation cost), averaged over seeds:

* **traffic**   -- fix ``quality_std`` and vary the number of stream passes.
* **structure** -- fix the number of passes and vary ``quality_std``.

Headline finding (sample data): with sustained traffic the DoublyOptimistic
policy *learns to route around the secretly-bad actions* -- it stays roughly flat
as hidden structure grows, while the myopic threshold (blind to feedback)
degrades. With only a single pass there is too little feedback and the threshold
wins. In other words, **optimism/learning pays off when there is hidden structure
to learn *and* enough feedback to learn it** -- exactly the regime the easy
benchmark excludes.

Usage::

    python scripts/run_regime_study.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
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

CREATION_COST = 0.20
QUALITY_FIXED = 0.4         # hidden-structure level for the traffic sweep
REPS_FIXED = 6             # traffic level for the structure sweep
TRAFFIC_REPS = [1, 2, 3, 4, 5, 6]
QUALITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = range(5)


def _replay_stream(stream0: pd.DataFrame, reps: int) -> pd.DataFrame:
    parts = []
    for r in range(reps):
        s = stream0.copy()
        s["query_id"] = s["query_id"].astype(str) + f"_r{r}"
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def _avg_loss(faq_df, stream_df, policy_factory, quality_std, seed, backend):
    env = CreateOrReuseEnv(
        faq_df, stream_df,
        Embedder(backend=backend, seed=seed),
        LossModel(quality_std=quality_std),
        creation_cost=CREATION_COST, seed=seed,
    )
    return run_episode(env, policy_factory()).summary["avg_loss"]


POLICIES = {
    "StaticThreshold": lambda: StaticThresholdPolicy(creation_cost=CREATION_COST),
    "DoublyOptimistic": lambda: DoublyOptimisticPolicy(creation_cost=CREATION_COST),
}


def run_regime_study(faq_df, stream0, seeds=SEEDS, backend="tfidf",
                     traffic_reps=TRAFFIC_REPS, quality_levels=QUALITY_LEVELS,
                     structure_reps=REPS_FIXED):
    rows = []

    # 1) traffic sweep (hidden structure fixed)
    for reps in traffic_reps:
        stream = _replay_stream(stream0, reps)
        for name, factory in POLICIES.items():
            vals = [_avg_loss(faq_df, stream, factory, QUALITY_FIXED, s, backend)
                    for s in seeds]
            rows.append({"study": "traffic", "x": reps, "policy": name,
                         "avg_loss": float(np.mean(vals)),
                         "avg_loss_std": float(np.std(vals))})

    # 2) structure sweep (traffic fixed)
    stream = _replay_stream(stream0, structure_reps)
    for qstd in quality_levels:
        for name, factory in POLICIES.items():
            vals = [_avg_loss(faq_df, stream, factory, qstd, s, backend)
                    for s in seeds]
            rows.append({"study": "structure", "x": qstd, "policy": name,
                         "avg_loss": float(np.mean(vals)),
                         "avg_loss_std": float(np.std(vals))})

    return pd.DataFrame(rows)


def plot_regime(df: pd.DataFrame, out: pathlib.Path) -> None:
    style = {
        "StaticThreshold": dict(color="#8c564b", marker="D"),
        "DoublyOptimistic": dict(color="#2ca02c", marker="o"),
    }
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    a = df[df["study"] == "traffic"]
    for name, st in style.items():
        d = a[a["policy"] == name].sort_values("x")
        axA.errorbar(d["x"], d["avg_loss"], yerr=d["avg_loss_std"], color=st["color"],
                     marker=st["marker"], linewidth=2, capsize=3, label=name)
    axA.set_title(f"More traffic → learning pays off\n(hidden structure fixed, quality_std={QUALITY_FIXED})")
    axA.set_xlabel("Stream passes (traffic volume)")
    axA.set_ylabel("Average total cost per query")
    axA.grid(True, alpha=0.3)
    axA.legend(fontsize=9)

    b = df[df["study"] == "structure"]
    for name, st in style.items():
        d = b[b["policy"] == name].sort_values("x")
        axB.errorbar(d["x"], d["avg_loss"], yerr=d["avg_loss_std"], color=st["color"],
                     marker=st["marker"], linewidth=2, capsize=3, label=name)
    axB.set_title(f"More hidden structure → threshold degrades\n(traffic fixed, {REPS_FIXED} passes)")
    axB.set_xlabel("Latent action-quality spread  (quality_std)")
    axB.set_ylabel("Average total cost per query")
    axB.grid(True, alpha=0.3)
    axB.legend(fontsize=9)

    fig.suptitle("When does optimism pay off? (creation cost = %.2f)" % CREATION_COST,
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(ROOT / "data"))
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--backend", default="tfidf")
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    results_dir = pathlib.Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    faq_df = pd.read_csv(data_dir / "sample_faq_library.csv")
    stream0 = pd.read_csv(data_dir / "sample_stream.csv")

    df = run_regime_study(faq_df, stream0, backend=args.backend)
    csv_path = results_dir / "regime_results.csv"
    fig_path = results_dir / "regime_when_optimism_helps.png"
    df.to_csv(csv_path, index=False)
    plot_regime(df, fig_path)
    print(f"Saved {csv_path}")
    print(f"Saved {fig_path}\n")

    for study, g in df.groupby("study"):
        piv = g.pivot_table(index="x", columns="policy", values="avg_loss").round(3)
        piv["winner"] = np.where(piv["DoublyOptimistic"] < piv["StaticThreshold"],
                                 "DoublyOptimistic", "StaticThreshold")
        print(f"=== {study} sweep ===")
        print(piv.to_string())
        print()


if __name__ == "__main__":
    main()
