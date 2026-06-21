"""Generate the figures for the project from the saved experiment results.

Run ``scripts/run_experiments.py`` first, then::

    python scripts/make_plots.py
    python scripts/make_plots.py --reference-cost 0.20

Figures written to ``results/``:
    cost_quality_tradeoff.png   the headline figure: quality vs creation effort
    cumulative_loss.png         cumulative total cost over the stream
    create_rate_by_cost.png     how each policy's create rate responds to cost
    action_library_growth.png   how the action library grows over the stream

Matplotlib only (no seaborn).
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")  # headless / reproducible
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]

# stable per-policy styling
STYLE = {
    "AlwaysCreate": dict(color="#d62728", marker="s"),
    "NearestReuse": dict(color="#1f77b4", marker="^"),
    "DoublyOptimistic": dict(color="#2ca02c", marker="o"),
}
FIXED_COLORS = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]

# category coverage groups (must match the dataset design in make_dataset.py)
WELL_COVERED = ["billing", "refund", "account_login", "troubleshooting",
                "pricing", "onboarding"]
POORLY_COVERED = ["subscription_cancel", "sales_followup", "crm_update",
                  "meeting_summary"]


def _style_for(policy: str, fixed_idx: dict) -> dict:
    if policy in STYLE:
        return STYLE[policy]
    idx = fixed_idx.setdefault(policy, len(fixed_idx))
    return dict(color=FIXED_COLORS[idx % len(FIXED_COLORS)], marker="D")


def plot_cost_quality_tradeoff(summary: pd.DataFrame, out: pathlib.Path,
                               reference_cost: float) -> None:
    """Headline figure: creation effort (x) vs mismatch loss (y).

    Lower-left is better. The adaptive policy traces a frontier by varying the
    creation cost; the baselines sit above/right of it.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fixed_idx: dict = {}

    # adaptive policy: a real frontier as the creation cost varies
    do = summary[summary["policy"] == "DoublyOptimistic"].sort_values("n_created")
    st = _style_for("DoublyOptimistic", fixed_idx)
    ax.plot(do["n_created"], do["cum_mismatch_loss"], "-", color=st["color"],
            marker=st["marker"], markersize=8, linewidth=2,
            label="DoublyOptimistic (sweep over cost)", zorder=5)
    for _, r in do.iterrows():
        ax.annotate(f"c={r['creation_cost']:g}",
                    (r["n_created"], r["cum_mismatch_loss"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8,
                    color=st["color"])

    # baselines: (n_created, mismatch) is cost-independent -> one point each
    for policy in summary["policy"].unique():
        if policy == "DoublyOptimistic":
            continue
        sub = summary[summary["policy"] == policy]
        row = sub[sub["creation_cost"] == reference_cost]
        row = row.iloc[0] if len(row) else sub.iloc[0]
        st = _style_for(policy, fixed_idx)
        ax.scatter([row["n_created"]], [row["cum_mismatch_loss"]],
                   color=st["color"], marker=st["marker"], s=90,
                   edgecolor="black", linewidth=0.5, zorder=4, label=policy)

    ax.set_xlabel("Number of created actions  (creation effort)")
    ax.set_ylabel("Cumulative mismatch loss  (quality cost)")
    ax.set_title("Cost–quality trade-off: adaptive creation dominates the baselines")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_cumulative_loss(traj: pd.DataFrame, out: pathlib.Path,
                         reference_cost: float) -> None:
    sub = traj[traj["cost_setting"] == reference_cost]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fixed_idx: dict = {}
    for policy in sub["policy"].unique():
        d = sub[sub["policy"] == policy]
        st = _style_for(policy, fixed_idx)
        ax.plot(d["round"], d["cum_total_loss"], color=st["color"],
                linewidth=2, label=policy)
    ax.set_xlabel("Round (queries processed)")
    ax.set_ylabel("Cumulative total cost  (mismatch + creation)")
    ax.set_title(f"Cumulative total cost over the stream  (creation cost = {reference_cost:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_create_rate_by_cost(summary: pd.DataFrame, out: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fixed_idx: dict = {}
    for policy in summary["policy"].unique():
        d = summary[summary["policy"] == policy].sort_values("creation_cost")
        st = _style_for(policy, fixed_idx)
        ax.plot(d["creation_cost"], d["create_rate"], color=st["color"],
                marker=st["marker"], linewidth=2, label=policy)
    ax.set_xlabel("Creation cost  c")
    ax.set_ylabel("Create rate  (fraction of rounds that create)")
    ax.set_title("Only the adaptive policy lowers its create rate as creation gets pricier")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_action_library_growth(traj: pd.DataFrame, out: pathlib.Path,
                               reference_cost: float) -> None:
    sub = traj[traj["cost_setting"] == reference_cost]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    fixed_idx: dict = {}
    for policy in sub["policy"].unique():
        d = sub[sub["policy"] == policy]
        st = _style_for(policy, fixed_idx)
        ax.plot(d["round"], d["library_size"], color=st["color"],
                linewidth=2, label=policy)
    ax.set_xlabel("Round (queries processed)")
    ax.set_ylabel("Action library size")
    ax.set_title(f"Action library growth over the stream  (creation cost = {reference_cost:g})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_create_rate_by_category(traj: pd.DataFrame, out: pathlib.Path,
                                 cost: float) -> None:
    """Mechanism diagnostic: *where* each policy spends its creation budget.

    The adaptive policy should concentrate creation on the poorly-covered
    categories (which have no seed FAQ) and mostly reuse for the well-covered
    ones, whereas the fixed-probability policy creates uniformly regardless of
    coverage. This is direct evidence of targeting, not just better aggregates.
    """
    sub = traj[traj["cost_setting"] == cost]
    order = WELL_COVERED + POORLY_COVERED
    series = {
        "FixedProb(p=0.5)": "#9467bd",
        "DoublyOptimistic": "#2ca02c",
    }
    rates = {}
    for pol in series:
        d = sub[sub["policy"] == pol]
        cr = d.assign(c=(d["kind"] == "CREATE")).groupby("category")["c"].mean()
        rates[pol] = [float(cr.get(cat, 0.0)) for cat in order]

    y = np.arange(len(order))
    h = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    ax.barh(y - h / 2, rates["DoublyOptimistic"], height=h,
            color=series["DoublyOptimistic"], label="DoublyOptimistic")
    ax.barh(y + h / 2, rates["FixedProb(p=0.5)"], height=h,
            color=series["FixedProb(p=0.5)"], label="FixedProb(p=0.5)")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.invert_yaxis()  # first category at the top
    # shade + label the poorly-covered band
    ax.axhspan(len(WELL_COVERED) - 0.5, len(order) - 0.5,
               color="#ffe8a3", alpha=0.45, zorder=0)
    ax.text(0.99, (len(WELL_COVERED) + len(order) - 1) / 2,
            "poorly-covered\n(no seed FAQ)", transform=ax.get_yaxis_transform(),
            ha="right", va="center", fontsize=9, color="#9a7d0a")
    ax.text(0.99, (len(WELL_COVERED) - 1) / 2, "well-covered\n(FAQ exists)",
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=9, color="gray")
    ax.set_xlabel("Create rate  (fraction of that category's queries that create)")
    ax.set_xlim(0, 1.0)
    ax.set_title(f"Where each policy spends its creation budget  (creation cost = {cost:g})")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(ROOT / "results"))
    parser.add_argument("--reference-cost", type=float, default=0.20,
                        help="creation cost used for the per-round figures")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results_dir)
    summary = pd.read_csv(results_dir / "experiment_results.csv")
    traj = pd.read_csv(results_dir / "experiment_trajectories.csv")

    ref = args.reference_cost
    if ref not in set(summary["creation_cost"]):
        ref = float(summary["creation_cost"].median())
    # the targeting mechanism is clearest at the highest cost (most selective)
    high = float(summary["creation_cost"].max())

    plots = {
        "cost_quality_tradeoff.png":
            lambda p: plot_cost_quality_tradeoff(summary, p, ref),
        "cumulative_loss.png":
            lambda p: plot_cumulative_loss(traj, p, ref),
        "create_rate_by_cost.png":
            lambda p: plot_create_rate_by_cost(summary, p),
        "action_library_growth.png":
            lambda p: plot_action_library_growth(traj, p, ref),
        "create_rate_by_category.png":
            lambda p: plot_create_rate_by_category(traj, p, high),
    }
    for name, fn in plots.items():
        path = results_dir / name
        fn(path)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
