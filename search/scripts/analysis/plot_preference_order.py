#!/usr/bin/env python3
"""Plot preference order from user study as a stacked bar chart.

Usage:
    python scripts/analysis/plot_preference_order.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

RESULT_DIR = Path("result")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt
import numpy as np


def load_preference_data(csv_path: Path) -> dict[str, list[int]]:
    """Load preference data from CSV.

    Returns:
        Dict mapping feature name to list of ranks (one per participant)
    """
    feature_ranks = defaultdict(list)

    with csv_path.open() as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.strip().split(",")
        if len(parts) < 6:
            continue
        # parts[0] is participant id, parts[1-5] are features in order 1-5
        for rank, feature in enumerate(parts[1:6], start=1):
            feature = feature.strip()
            # Fix typo in data
            if feature == "discontiuous":
                feature = "discontinuous"
            feature_ranks[feature].append(rank)

    return dict(feature_ranks)


def plot_preference_stacked_bar(feature_ranks: dict[str, list[int]], output_path: Path):
    """Create a stacked bar chart of preference rankings.

    X-axis: features
    Y-axis: count
    Stacked colors: rank 1 (best) to rank 5 (worst)
    """
    features = ["sentence", "keyword", "keyword2", "word", "discontinuous"]
    ranks = [1, 2, 3, 4, 5]

    # Count how many times each feature got each rank
    rank_counts = {feature: {r: 0 for r in ranks} for feature in features}
    for feature in features:
        for rank in feature_ranks.get(feature, []):
            rank_counts[feature][rank] += 1

    # Prepare data for stacking
    x = np.arange(len(features))
    width = 0.6

    # Colors: green (good/rank 1) to red (bad/rank 5)
    rank_colors = {1: "#2ecc71", 2: "#82e0aa", 3: "#f9e79f", 4: "#f5b7b1", 5: "#e74c3c"}

    fig, ax = plt.subplots(figsize=(10, 6))

    # Stack from rank 5 (bottom) to rank 1 (top)
    bottom = np.zeros(len(features))
    for rank in [5, 4, 3, 2, 1]:
        counts = [rank_counts[f][rank] for f in features]
        ax.bar(x, counts, width, label=f"Rank {rank}", bottom=bottom, color=rank_colors[rank])
        bottom += counts

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Preference Order by Feature\n(1 = Most Preferred, 5 = Least Preferred)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=45, ha="right")
    # Reverse legend order to show Rank 1 at top
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title="Rank", loc="upper right")
    ax.set_ylim(0, len(list(feature_ranks.values())[0]) + 0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    csv_path = RESULT_DIR / "preference_order.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    feature_ranks = load_preference_data(csv_path)
    print(f"Loaded preferences for {len(list(feature_ranks.values())[0])} participants")
    print(f"Features: {list(feature_ranks.keys())}")

    # Print summary
    print("\nMean rank per feature (lower is better):")
    for feature in ["sentence", "keyword", "keyword2", "word", "discontinuous"]:
        ranks = feature_ranks.get(feature, [])
        if ranks:
            mean_rank = sum(ranks) / len(ranks)
            print(f"  {feature}: {mean_rank:.2f}")

    output_path = RESULT_DIR / "preference_order_stacked.png"
    plot_preference_stacked_bar(feature_ranks, output_path)


if __name__ == "__main__":
    main()
