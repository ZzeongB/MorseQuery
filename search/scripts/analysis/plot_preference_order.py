#!/usr/bin/env python3
"""Plot preference order from user study as a stacked bar chart.

Usage:
    python scripts/analysis/plot_preference_order.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path

RESULT_DIR = Path("result")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
    features = ["discontinuous", "keyword", "keyword2", "word", "sentence"]
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
        ax.bar(
            x,
            counts,
            width,
            label=f"Rank {rank}",
            bottom=bottom,
            color=rank_colors[rank],
        )
        bottom += counts

    ax.set_xlabel("Feature", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Preference Order by Feature\n(1 = Most Preferred, 5 = Least Preferred)",
        fontsize=14,
        fontweight="bold",
    )
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


def compute_kendall_w(rank_matrix: np.ndarray) -> tuple[float, float]:
    """Compute Kendall's W (coefficient of concordance).

    Args:
        rank_matrix: (n_participants, n_conditions) matrix of ranks

    Returns:
        (W, chi2) where W is Kendall's W and chi2 is the test statistic
    """
    n, k = rank_matrix.shape  # n participants, k conditions
    # Sum of ranks for each condition
    R = rank_matrix.sum(axis=0)
    R_mean = R.mean()
    # Sum of squared deviations
    S = np.sum((R - R_mean) ** 2)
    # Kendall's W
    W = (12 * S) / (n**2 * (k**3 - k))
    # Chi-square approximation
    chi2 = n * (k - 1) * W
    return W, chi2


def run_statistical_analysis(feature_ranks: dict[str, list[int]], features: list[str]):
    """Run statistical analysis on preference rankings."""
    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    n_participants = len(list(feature_ranks.values())[0])
    n_conditions = len(features)

    # Build rank matrix (participants x conditions)
    rank_matrix = np.array([feature_ranks[f] for f in features]).T

    # 1. Mean Rank Analysis
    print("\n--- Mean Rank per Feature (lower = more preferred) ---")
    mean_ranks = {}
    for feature in features:
        ranks = feature_ranks[feature]
        mean_rank = np.mean(ranks)
        std_rank = np.std(ranks, ddof=1)
        mean_ranks[feature] = mean_rank
        print(f"  {feature:15s}: M = {mean_rank:.2f}, SD = {std_rank:.2f}")

    # Sort by mean rank
    sorted_features = sorted(mean_ranks.items(), key=lambda x: x[1])
    print("\n  Preference order (most to least preferred):")
    print("  " + " > ".join([f"{f}({m:.2f})" for f, m in sorted_features]))

    # 2. Kendall's W (Coefficient of Concordance)
    print("\n--- Kendall's W (Agreement among participants) ---")
    W, chi2_w = compute_kendall_w(rank_matrix)
    df_w = n_conditions - 1
    p_w = 1 - stats.chi2.cdf(chi2_w, df_w)
    print(f"  W = {W:.3f}")
    print(f"  χ²({df_w}) = {chi2_w:.3f}, p = {p_w:.4f}")
    if W < 0.3:
        agreement = "weak"
    elif W < 0.5:
        agreement = "fair"
    elif W < 0.7:
        agreement = "moderate"
    else:
        agreement = "strong"
    sig_text = "significant" if p_w < 0.05 else "not significant"
    print(f"  Interpretation: {agreement} agreement ({sig_text})")

    # 3. Friedman Test
    print("\n--- Friedman Test (Differences among conditions) ---")
    friedman_stat, friedman_p = stats.friedmanchisquare(*[feature_ranks[f] for f in features])
    print(f"  χ²({n_conditions - 1}) = {friedman_stat:.3f}, p = {friedman_p:.4f}")
    if friedman_p < 0.05:
        print("  Result: SIGNIFICANT difference among conditions (p < 0.05)")
    else:
        print("  Result: No significant difference among conditions (p >= 0.05)")

    # 4. Post-hoc pairwise comparisons (Wilcoxon signed-rank tests) with Bonferroni correction
    if friedman_p < 0.05:
        print("\n--- Post-hoc Pairwise Comparisons (Wilcoxon signed-rank) ---")
        n_comparisons = n_conditions * (n_conditions - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        print(f"  Bonferroni corrected α = {alpha_corrected:.4f} ({n_comparisons} comparisons)")
        print()

        results = []
        for f1, f2 in combinations(features, 2):
            ranks1 = np.array(feature_ranks[f1])
            ranks2 = np.array(feature_ranks[f2])
            # Wilcoxon signed-rank test
            stat, p = stats.wilcoxon(ranks1, ranks2, alternative="two-sided")
            mean_diff = np.mean(ranks1) - np.mean(ranks2)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < alpha_corrected else ""
            results.append((f1, f2, mean_diff, stat, p, sig))

        # Sort by p-value
        results.sort(key=lambda x: x[4])
        for f1, f2, mean_diff, stat, p, sig in results:
            direction = "←" if mean_diff < 0 else "→" if mean_diff > 0 else "="
            print(f"  {f1:15s} vs {f2:15s}: W = {stat:5.1f}, p = {p:.4f} {sig}")
            if sig:
                better = f1 if mean_diff < 0 else f2
                print(f"    {direction} {better} is significantly more preferred")

        # Print legend
        print()
        print(f"  * p < {alpha_corrected:.4f} (Bonferroni corrected)")
        print("  ** p < 0.01")
        print("  *** p < 0.001")

    print("\n" + "=" * 60)


def main():
    csv_path = RESULT_DIR / "preference_order.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} not found")
        return

    feature_ranks = load_preference_data(csv_path)
    print(f"Loaded preferences for {len(list(feature_ranks.values())[0])} participants")
    print(f"Features: {list(feature_ranks.keys())}")

    features = ["keyword", "word", "sentence", "keyword2", "discontinuous"]

    # Run statistical analysis
    run_statistical_analysis(feature_ranks, features)

    # Plot
    output_path = RESULT_DIR / "preference_order_stacked.png"
    plot_preference_stacked_bar(feature_ranks, output_path)


if __name__ == "__main__":
    main()
