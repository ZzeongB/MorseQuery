#!/usr/bin/env python3
"""
Keyboard interaction analysis for within-subject study.

Analyzes left/right/up key presses by condition.
Generates stacked bar chart and statistical analysis.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

# Condition mapping (old -> new)
CONDITION_MAP = {
    "temporal": "Temporal",
    "keyword": "Keyword (Target)",
    "keyword2": "Keyword (No Target)",
    "word": "Word",
    "sentence": "Sentence",
}

# New condition order
CONDITION_ORDER = [
    "Temporal",
    "Sentence",
    "Word",
    "Keyword (Target)",
    "Keyword (No Target)",
]

# Display labels for x-axis
CONDITION_DISPLAY_LABELS = [
    "Temporal",
    "Sentence",
    "Word",
    "Keyword\n(Target)",
    "Keyword\n(No Target)",
]

# Condition colors (matching other figures)
CONDITION_COLORS = {
    "Temporal": "#808080",
    "Sentence": "#4C78A8",
    "Word": "#E15759",
    "Keyword (Target)": "#59A14F",
    "Keyword (No Target)": "#8CD17D",
}

# Key colors for stacked bars (distinct from condition colors)
KEY_COLORS = {
    "left": "#f28e2b",  # orange
    "right": "#76b7b2",  # teal
    "up": "#808080",  # gray
}


def get_script_dir():
    return Path(__file__).parent


def get_output_dir():
    output_dir = get_script_dir().parent.parent / "result" / "figures_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_keyboard_data():
    """Load keyboard interaction data from study logs (per-trial)."""
    logs_dir = get_script_dir().parent.parent / "logs" / "study"

    # List of per-trial records: {participant, condition, trial, left, right, up}
    trials = []

    for jsonl_file in logs_dir.glob("*.jsonl"):
        name = jsonl_file.stem
        if "tutorial" in name.lower():
            continue

        parts = name.split("_")
        if len(parts) < 4:
            continue

        participant = parts[0]
        condition_raw = parts[1]
        condition = CONDITION_MAP.get(condition_raw, condition_raw)

        # Parse events and group by trial (taskIndex)
        events = []
        with open(jsonl_file, "r") as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    events.append(event)
                except json.JSONDecodeError:
                    continue

        # Find interruptions to identify trial boundaries and delay types
        current_trial = 0
        trial_actions = defaultdict(lambda: {"left": 0, "right": 0, "up": 0})
        trial_delay_type = {}  # trial_idx -> "short" or "long"

        for event in events:
            if event.get("event") == "interruption":
                current_trial = event.get("taskIndex", current_trial)
                delay_type = event.get("delayType", "unknown")
                trial_delay_type[current_trial] = delay_type
            elif event.get("event") == "user_action":
                action = event.get("action", "")
                if action in ["left", "right", "up"]:
                    trial_actions[current_trial][action] += 1

        # Add each trial as a separate record
        for trial_idx, actions in trial_actions.items():
            if trial_idx == 0:
                continue  # Skip pre-first-interruption actions
            trials.append(
                {
                    "participant": participant,
                    "condition": condition,
                    "trial": trial_idx,
                    "left": actions["left"],
                    "right": actions["right"],
                    "up": actions["up"],
                    "delay_type": trial_delay_type.get(trial_idx, "unknown"),
                }
            )

    return trials


def build_dataframe(trials):
    """Build pandas DataFrame from per-trial records."""
    df = pd.DataFrame(trials)
    df["total"] = df["left"] + df["right"] + df["up"]
    return df


def add_significance_bracket(ax, x1, x2, y, h, text, fontsize=9):
    """Add significance bracket between two bars."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def assign_bracket_rows(brackets):
    """Assign brackets to rows so non-overlapping brackets share the same row."""
    if not brackets:
        return []

    rows = []
    row_assignments = []

    for x1, x2, _ in brackets:
        left, right = min(x1, x2), max(x1, x2)

        placed = False
        for row_idx, row_ranges in enumerate(rows):
            overlaps = False
            for r_left, r_right in row_ranges:
                if not (right <= r_left or left >= r_right):
                    overlaps = True
                    break

            if not overlaps:
                row_ranges.append((left, right))
                row_assignments.append(row_idx)
                placed = True
                break

        if not placed:
            rows.append([(left, right)])
            row_assignments.append(len(rows) - 1)

    return row_assignments


def adjust_shared_endpoints(brackets, offset=0.05):
    """Adjust bracket endpoints that are shared to separate them slightly."""
    if not brackets:
        return []

    endpoint_usage = {}
    for i, (c1, c2, _) in enumerate(brackets):
        left, right = min(c1, c2), max(c1, c2)
        if left not in endpoint_usage:
            endpoint_usage[left] = []
        if right not in endpoint_usage:
            endpoint_usage[right] = []
        endpoint_usage[left].append((i, "left"))
        endpoint_usage[right].append((i, "right"))

    adjusted = []
    for c1, c2, _ in brackets:
        adjusted.append([min(c1, c2), max(c1, c2)])

    for endpoint, usages in endpoint_usage.items():
        if len(usages) > 1:
            usages.sort(key=lambda x: x[0])
            for idx, (bracket_idx, side) in enumerate(usages):
                if side == "left":
                    adjusted[bracket_idx][0] = endpoint + offset
                else:
                    adjusted[bracket_idx][1] = endpoint - offset

    return adjusted


def run_statistical_analysis(df, output_file):
    """Run Friedman test and post-hoc Wilcoxon tests on per-trial data."""
    results = []
    n_participants = df["participant"].nunique()
    n_trials = len(df)
    results.append("=" * 70)
    results.append(
        f"KEYBOARD INTERACTION ANALYSIS (Per-Trial, N={n_participants} participants, {n_trials} trials)"
    )
    results.append("=" * 70)

    # Descriptive statistics (per-trial)
    results.append("\n1. DESCRIPTIVE STATISTICS (Per-Trial Interactions)")
    results.append("-" * 70)

    desc_data = []
    for cond in CONDITION_ORDER:
        cond_data = df[df["condition"] == cond]["total"]
        desc_data.append(
            {
                "Condition": cond,
                "N_trials": len(cond_data),
                "Mean": cond_data.mean(),
                "SD": cond_data.std(),
                "Median": cond_data.median(),
                "Min": cond_data.min(),
                "Max": cond_data.max(),
            }
        )
    desc_df = pd.DataFrame(desc_data)
    results.append(desc_df.to_string(index=False))

    # Breakdown by action type (per-trial)
    results.append("\n2. BREAKDOWN BY ACTION TYPE (Per-Trial Mean ± SD)")
    results.append("-" * 70)
    for cond in CONDITION_ORDER:
        cond_df = df[df["condition"] == cond]
        left_m, left_s = cond_df["left"].mean(), cond_df["left"].std()
        right_m, right_s = cond_df["right"].mean(), cond_df["right"].std()
        up_m, up_s = cond_df["up"].mean(), cond_df["up"].std()
        results.append(
            f"{cond:25} | left: {left_m:5.2f}±{left_s:5.2f} | right: {right_m:4.2f}±{right_s:4.2f} | up: {up_m:3.2f}±{up_s:3.2f}"
        )

    # Aggregate per-trial to per-participant mean for paired tests
    participant_means = (
        df.groupby(["participant", "condition"])["total"].mean().reset_index()
    )

    # Normality test (on participant means)
    results.append("\n3. NORMALITY TEST (Shapiro-Wilk, on participant means)")
    results.append("-" * 70)
    for cond in CONDITION_ORDER:
        cond_data = participant_means[participant_means["condition"] == cond]["total"]
        if len(cond_data) >= 3:
            stat, p = stats.shapiro(cond_data)
            normal = "Yes" if p > 0.05 else "No"
            results.append(f"{cond:25}: W={stat:.3f}, p={p:.4f} → Normal: {normal}")
        else:
            results.append(f"{cond:25}: Not enough data")

    # Friedman test (on participant means)
    results.append("\n4. OMNIBUS TEST (Friedman, on participant means)")
    results.append("-" * 70)

    # Prepare data for Friedman test - need same participants in all conditions
    participants_with_all = set(participant_means["participant"].unique())
    for cond in CONDITION_ORDER:
        cond_participants = set(
            participant_means[participant_means["condition"] == cond]["participant"]
        )
        participants_with_all &= cond_participants

    if len(participants_with_all) >= 3:
        totals_by_cond = []
        for cond in CONDITION_ORDER:
            cond_data = (
                participant_means[
                    (participant_means["condition"] == cond)
                    & (participant_means["participant"].isin(participants_with_all))
                ]
                .sort_values("participant")["total"]
                .values
            )
            totals_by_cond.append(cond_data)

        friedman_stat, friedman_p = stats.friedmanchisquare(*totals_by_cond)
        results.append(
            f"Friedman Test: χ²({len(CONDITION_ORDER)-1})={friedman_stat:.3f}, p={friedman_p:.6f}"
        )
        results.append(
            f"(Using {len(participants_with_all)} participants with data in all conditions)"
        )

        if friedman_p < 0.05:
            results.append("→ Significant difference between conditions (p < 0.05)")
        else:
            results.append("→ No significant difference between conditions")
    else:
        results.append(
            "Not enough participants with data in all conditions for Friedman test"
        )
        friedman_p = 1.0

    # Post-hoc Wilcoxon with Holm correction (on participant means)
    results.append("\n5. PAIRWISE COMPARISONS (Wilcoxon signed-rank, Holm corrected)")
    results.append("-" * 70)

    pairwise_results = []
    for i, cond1 in enumerate(CONDITION_ORDER):
        for j, cond2 in enumerate(CONDITION_ORDER):
            if i < j:
                # Get participants with data in both conditions
                p1 = set(
                    participant_means[participant_means["condition"] == cond1][
                        "participant"
                    ]
                )
                p2 = set(
                    participant_means[participant_means["condition"] == cond2][
                        "participant"
                    ]
                )
                common_p = p1 & p2

                if len(common_p) >= 3:
                    data1 = (
                        participant_means[
                            (participant_means["condition"] == cond1)
                            & (participant_means["participant"].isin(common_p))
                        ]
                        .sort_values("participant")["total"]
                        .values
                    )
                    data2 = (
                        participant_means[
                            (participant_means["condition"] == cond2)
                            & (participant_means["participant"].isin(common_p))
                        ]
                        .sort_values("participant")["total"]
                        .values
                    )
                    try:
                        stat, p = stats.wilcoxon(data1, data2)
                        diff = data1.mean() - data2.mean()
                        pairwise_results.append(
                            {
                                "A": cond1,
                                "B": cond2,
                                "i": i,
                                "j": j,
                                "Diff": diff,
                                "W": stat,
                                "p_unc": p,
                                "n": len(common_p),
                            }
                        )
                    except Exception as e:
                        results.append(
                            f"Warning: Wilcoxon failed for {cond1} vs {cond2}: {e}"
                        )
                else:
                    results.append(
                        f"Warning: Not enough paired data for {cond1} vs {cond2}"
                    )

    # Holm correction
    if pairwise_results:
        p_values = [r["p_unc"] for r in pairwise_results]
        n_tests = len(p_values)
        sorted_indices = np.argsort(p_values)

        p_corrected = np.zeros(n_tests)
        for rank, idx in enumerate(sorted_indices):
            p_corrected[idx] = min(1.0, p_values[idx] * (n_tests - rank))

        # Ensure monotonicity
        for rank in range(1, n_tests):
            idx = sorted_indices[rank]
            prev_idx = sorted_indices[rank - 1]
            p_corrected[idx] = max(p_corrected[idx], p_corrected[prev_idx])

        for i, r in enumerate(pairwise_results):
            r["p_corr"] = p_corrected[i]
            p = r["p_corr"]
            if p < 0.001:
                r["Sig"] = "***"
            elif p < 0.01:
                r["Sig"] = "**"
            elif p < 0.05:
                r["Sig"] = "*"
            else:
                r["Sig"] = ""

    # Sort by p_corr
    pairwise_results = sorted(pairwise_results, key=lambda x: x["p_corr"])

    results.append(
        f"{'Comparison':<50} {'Diff':>8} {'W':>8} {'p_unc':>10} {'p_corr':>10} {'Sig':>5}"
    )
    for r in pairwise_results:
        comparison = f"{r['A']} vs {r['B']}"
        results.append(
            f"{comparison:<50} {r['Diff']:>8.1f} {r['W']:>8.0f} {r['p_unc']:>10.4f} {r['p_corr']:>10.4f} {r['Sig']:>5}"
        )

    results.append("\n" + "=" * 70)
    results.append("* p < .05, ** p < .01, *** p < .001 (Holm corrected)")
    results.append("=" * 70)

    # Write to file
    # with open(output_file, "w") as f:
    #     f.write("\n".join(results))

    print("\n".join(results))

    # Return significant pairs for plotting
    significant_pairs = []
    for r in pairwise_results:
        if r["p_corr"] < 0.05:
            significant_pairs.append((r["i"], r["j"], r["Sig"]))

    return significant_pairs


def plot_keyboard_interactions(df, significant_pairs, use_brackets=True):
    """Create stacked bar chart of keyboard interactions (per-trial means)."""
    output_dir = get_output_dir()

    # Calculate per-trial means for each action type
    means_data = {"left": [], "right": [], "up": [], "se": []}
    for cond in CONDITION_ORDER:
        cond_df = df[df["condition"] == cond]
        means_data["left"].append(cond_df["left"].mean())
        means_data["right"].append(cond_df["right"].mean())
        means_data["up"].append(cond_df["up"].mean())
        # SE based on per-trial total
        means_data["se"].append(cond_df["total"].std() / np.sqrt(len(cond_df)))

    left_means = np.array(means_data["left"])
    right_means = np.array(means_data["right"])
    up_means = np.array(means_data["up"])
    total_se = np.array(means_data["se"])

    # Create figure (same size as SUS score)
    fig, ax = plt.subplots(figsize=(4, 3.2))

    x = np.arange(len(CONDITION_ORDER))
    width = 0.65

    # Stacked bars with distinct key colors (solid, no hatch)
    bars_left = ax.bar(
        x,
        left_means,
        width,
        label="Left",
        color=KEY_COLORS["left"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    bars_right = ax.bar(
        x,
        right_means,
        width,
        bottom=left_means,
        label="Right",
        color=KEY_COLORS["right"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    bars_up = ax.bar(
        x,
        up_means,
        width,
        bottom=left_means + right_means,
        label="Up",
        color=KEY_COLORS["up"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.9,
    )

    # Error bars for total
    totals = left_means + right_means + up_means
    ax.errorbar(
        x,
        totals,
        yerr=total_se,
        fmt="none",
        color="black",
        capsize=4,
        capthick=1.0,
        linewidth=1.0,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Navigation Action Counts", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_DISPLAY_LABELS, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(np.arange(0, 21, 5))  # Show ticks only up to 20
    ax.tick_params(axis="y", labelsize=9)

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance brackets
    if significant_pairs and use_brackets:
        row_assignments = assign_bracket_rows(significant_pairs)
        adjusted_endpoints = adjust_shared_endpoints(significant_pairs, offset=0.08)
        y_base = max(totals + total_se) + 2  # Start above data
        row_height = 2.5  # Spacing between bracket rows

        for i, (c1, c2, marker) in enumerate(significant_pairs):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            y = y_base + row_assignments[i] * row_height
            add_significance_bracket(ax, adj_c1, adj_c2, y, 0.8, marker, fontsize=8)

        max_bracket_y = y_base + (max(row_assignments) + 1) * row_height
        ax.set_ylim(0, max_bracket_y + 2)
    else:
        ax.set_ylim(0, max(totals + total_se) + 5)

    # Legend for key types
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=KEY_COLORS["left"], edgecolor="white", alpha=0.9, label="Left"),
        Patch(
            facecolor=KEY_COLORS["right"], edgecolor="white", alpha=0.9, label="Right"
        ),
        Patch(facecolor=KEY_COLORS["up"], edgecolor="white", alpha=0.9, label="Up"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        fontsize=8,
        frameon=False,
        handlelength=2,
        handleheight=1.2,
    )

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"keyboard_interactions{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nKeyboard interactions figure saved to: {output_path}")
    return output_path


def plot_keyboard_interactions_by_lag(df):
    """Create grouped bar chart showing Overall, Short-Lag, Long-Lag."""
    output_dir = get_output_dir()

    # Filter out unknown delay types for lag-specific analysis
    df_with_lag = df[df["delay_type"].isin(["short", "long"])]

    # Lag categories
    lag_categories = ["Overall", "Short-Lag", "Long-Lag"]

    # Calculate means for each condition and lag type
    data = {
        lag: {"left": [], "right": [], "up": [], "se": []} for lag in lag_categories
    }

    for cond in CONDITION_ORDER:
        # Overall
        cond_df = df[df["condition"] == cond]
        data["Overall"]["left"].append(cond_df["left"].mean())
        data["Overall"]["right"].append(cond_df["right"].mean())
        data["Overall"]["up"].append(cond_df["up"].mean())
        data["Overall"]["se"].append(cond_df["total"].std() / np.sqrt(len(cond_df)))

        # Short-Lag
        short_df = df_with_lag[
            (df_with_lag["condition"] == cond) & (df_with_lag["delay_type"] == "short")
        ]
        if len(short_df) > 0:
            data["Short-Lag"]["left"].append(short_df["left"].mean())
            data["Short-Lag"]["right"].append(short_df["right"].mean())
            data["Short-Lag"]["up"].append(short_df["up"].mean())
            data["Short-Lag"]["se"].append(
                short_df["total"].std() / np.sqrt(len(short_df))
            )
        else:
            data["Short-Lag"]["left"].append(0)
            data["Short-Lag"]["right"].append(0)
            data["Short-Lag"]["up"].append(0)
            data["Short-Lag"]["se"].append(0)

        # Long-Lag
        long_df = df_with_lag[
            (df_with_lag["condition"] == cond) & (df_with_lag["delay_type"] == "long")
        ]
        if len(long_df) > 0:
            data["Long-Lag"]["left"].append(long_df["left"].mean())
            data["Long-Lag"]["right"].append(long_df["right"].mean())
            data["Long-Lag"]["up"].append(long_df["up"].mean())
            data["Long-Lag"]["se"].append(
                long_df["total"].std() / np.sqrt(len(long_df))
            )
        else:
            data["Long-Lag"]["left"].append(0)
            data["Long-Lag"]["right"].append(0)
            data["Long-Lag"]["up"].append(0)
            data["Long-Lag"]["se"].append(0)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 3.5))

    x = np.arange(len(CONDITION_ORDER))
    n_groups = 3  # Overall, Short-Lag, Long-Lag
    width = 0.25
    offsets = [-width, 0, width]

    # Colors for lag categories (lighter shades for distinction)
    lag_alphas = [0.9, 0.7, 0.5]

    for i, (lag, offset, alpha) in enumerate(zip(lag_categories, offsets, lag_alphas)):
        left_means = np.array(data[lag]["left"])
        right_means = np.array(data[lag]["right"])
        up_means = np.array(data[lag]["up"])
        total_se = np.array(data[lag]["se"])
        totals = left_means + right_means + up_means

        # Stacked bars
        ax.bar(
            x + offset,
            left_means,
            width,
            color=KEY_COLORS["left"],
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )
        ax.bar(
            x + offset,
            right_means,
            width,
            bottom=left_means,
            color=KEY_COLORS["right"],
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )
        ax.bar(
            x + offset,
            up_means,
            width,
            bottom=left_means + right_means,
            color=KEY_COLORS["up"],
            edgecolor="white",
            linewidth=0.5,
            alpha=alpha,
        )

        # Error bars
        ax.errorbar(
            x + offset,
            totals,
            yerr=total_se,
            fmt="none",
            color="black",
            capsize=2,
            capthick=0.8,
            linewidth=0.8,
        )

    ax.set_xlabel("")
    ax.set_ylabel("Navigation Action Counts", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_DISPLAY_LABELS, rotation=0, ha="center", fontsize=9)
    ax.set_yticks(np.arange(0, 26, 5))
    ax.tick_params(axis="y", labelsize=9)

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_ylim(0, 25)

    # Legend for lag categories
    from matplotlib.patches import Patch

    lag_legend = [
        Patch(facecolor="gray", alpha=0.9, label="Overall"),
        Patch(facecolor="gray", alpha=0.7, label="Short-Lag"),
        Patch(facecolor="gray", alpha=0.5, label="Long-Lag"),
    ]
    key_legend = [
        Patch(facecolor=KEY_COLORS["left"], edgecolor="white", alpha=0.9, label="Left"),
        Patch(
            facecolor=KEY_COLORS["right"], edgecolor="white", alpha=0.9, label="Right"
        ),
        Patch(facecolor=KEY_COLORS["up"], edgecolor="white", alpha=0.9, label="Up"),
    ]

    # Two legends
    leg1 = ax.legend(
        handles=lag_legend,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=8,
        frameon=False,
        title="Lag Type",
        title_fontsize=8,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=key_legend,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        fontsize=8,
        frameon=False,
        title="Key",
        title_fontsize=8,
    )

    plt.tight_layout()

    output_path = output_dir / "keyboard_interactions_by_lag.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"Keyboard interactions by lag figure saved to: {output_path}")
    return output_path


def save_csv(df):
    """Save per-trial keyboard interaction data to CSV."""
    output_dir = get_script_dir().parent.parent / "result"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by participant and condition
    df_sorted = df.sort_values(
        ["participant", "condition", "trial"],
        key=lambda x: x.map(
            lambda v: int(v[1:]) if isinstance(v, str) and v[1:].isdigit() else v
        )
        if x.name == "participant"
        else x,
    )

    # Save per-trial data (long format)
    csv_path = output_dir / "keyboard_interactions.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path} ({len(df_sorted)} trials)")


def main():
    print("=" * 70)
    print("KEYBOARD INTERACTION ANALYSIS")
    print("=" * 70)
    print()

    # Load data (per-trial)
    print("Loading keyboard interaction data (per-trial)...")
    trials = load_keyboard_data()
    df = build_dataframe(trials)
    print(f"  Loaded {len(df)} trials from {df['participant'].nunique()} participants")
    print()

    # Save CSV
    save_csv(df)
    print()

    # Run statistical analysis
    output_dir = get_output_dir()
    stats_file = output_dir / "keyboard_interactions_stats.txt"
    print("Running statistical analysis...")
    significant_pairs = run_statistical_analysis(df, stats_file)
    print(f"\nStatistical analysis saved to: {stats_file}")
    print()

    # Generate figures
    print("Generating figures...")
    plot_keyboard_interactions(df, significant_pairs, use_brackets=True)
    plot_keyboard_interactions_by_lag(df)

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
