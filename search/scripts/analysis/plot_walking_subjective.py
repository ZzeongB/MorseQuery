#!/usr/bin/env python3
"""
Generate walking study subjective figures (NASA-TLX and SUS) with consistent styling.

Generates:
1. walking_nasa_tlx_bracket.svg - NASA-TLX 6 subscales grouped bar chart
2. walking_sus_bracket.svg - SUS score bar chart
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns


# Condition order for walking study
CONDITION_ORDER = ["Meta Neuralband", "Headphone"]

# Display labels for figures
CONDITION_DISPLAY_LABELS = ["Microgesture", "Headphone"]

# NASA-TLX column mapping (full question text -> short name)
NASA_TLX_COLUMNS = {
    "How mentally demanding was the task?": "Mental\nDemand",
    "How physically demanding was the task?": "Physical\nDemand",
    "How hurried or rushed was the pace of the task?": "Temporal\nDemand",
    "How successful were you in accomplishing what you were asked to do?": "Performance",
    "How hard did you have to work to accomplish your level of performance?": "Effort",
    "How insecure, discouraged, irritated, stressed, and annoyed were you?": "Frustration",
}

# Condition colors
CONDITION_COLORS = {
    "Meta Neuralband": "#4C78A8",  # Blue
    "Headphone": "#E15759",  # Red/coral
}


def get_output_dir():
    """Get output directory for figures."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "result" / "figures_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_walking_survey_data():
    """Load and preprocess walking survey data."""
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "result" / "walking_survey.csv"

    df = pd.read_csv(data_path, header=1)
    # Rename NASA-TLX columns from full question text to short names
    df = df.rename(columns=NASA_TLX_COLUMNS)
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


def add_significance_marker(ax, x, y, text, fontsize=9):
    """Add significance marker (asterisk only, no bracket) above a bar."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def assign_bracket_rows(brackets):
    """Assign brackets to rows so that non-overlapping brackets share the same row."""
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


def run_nasa_tlx_tests(df):
    """Run paired t-tests for NASA-TLX dimensions."""
    dimension_order = [
        "Mental\nDemand",
        "Physical\nDemand",
        "Temporal\nDemand",
        "Performance",
        "Effort",
        "Frustration",
    ]

    significance_by_dim = {}

    for dim in dimension_order:
        df_dim = df[["Participant ID", "Condition", dim]].copy()
        df_dim.columns = ["subject", "condition", "score"]

        # Pivot for paired t-test
        df_wide = df_dim.pivot(index="subject", columns="condition", values="score")

        print(f"\n{dim} Paired T-Test:")
        try:
            # Paired t-test between the two conditions
            result = pg.ttest(
                df_wide["Meta Neuralband"],
                df_wide["Headphone"],
                paired=True,
            )
            print(result.to_string(index=False))

            p_col = "p_val" if "p_val" in result.columns else "p-val"
            p_val = result[p_col].values[0]
            if p_val < 0.05:
                marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                significance_by_dim[dim] = [(0, 1, marker)]
        except Exception as e:
            print(f"Error running NASA-TLX t-test for {dim}: {e}")
            import traceback

            traceback.print_exc()

    return significance_by_dim


def plot_nasa_tlx(df, significance_by_dim=None):
    """Create NASA-TLX grouped barplot with all 6 subscales in one row."""
    output_dir = get_output_dir()

    if significance_by_dim is None:
        significance_by_dim = {}

    dimension_order = [
        "Mental\nDemand",
        "Physical\nDemand",
        "Temporal\nDemand",
        "Performance",
        "Effort",
        "Frustration",
    ]

    # Melt data for grouped barplot
    df_melted = df.melt(
        id_vars=["Participant ID", "Condition"],
        value_vars=dimension_order,
        var_name="Dimension",
        value_name="Score",
    )

    df_melted["Condition"] = pd.Categorical(
        df_melted["Condition"], categories=CONDITION_ORDER, ordered=True
    )
    df_melted["Dimension"] = pd.Categorical(
        df_melted["Dimension"], categories=dimension_order, ordered=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Create grouped barplot
    sns.barplot(
        data=df_melted,
        x="Dimension",
        y="Score",
        hue="Condition",
        ax=ax,
        palette=CONDITION_COLORS,
        order=dimension_order,
        hue_order=CONDITION_ORDER,
        errorbar="se",
        capsize=0.05,
        err_kws={"linewidth": 1.0},
        gap=0.15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Score (1-7)", fontsize=10)
    ax.set_ylim(0, 7)
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Calculate bar positions for grouped barplot
    n_conds = len(CONDITION_ORDER)
    bar_width = 0.8 / n_conds

    for dim_idx, dim in enumerate(dimension_order):
        if dim not in significance_by_dim:
            continue

        brackets = significance_by_dim[dim]
        y_base = 7.3
        row_height = 0.9

        for i, (c1, c2, marker) in enumerate(brackets):
            x1 = dim_idx + (c1 - (n_conds - 1) / 2) * bar_width
            x2 = dim_idx + (c2 - (n_conds - 1) / 2) * bar_width
            y = y_base
            add_significance_bracket(ax, x1, x2, y, 0.18, marker, fontsize=8)

    # Legend at bottom, horizontal
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        CONDITION_DISPLAY_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_multialignment("center")

    plt.tight_layout()

    output_path = output_dir / "walking_nasa_tlx_bracket.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nWalking NASA-TLX figure saved to: {output_path}")
    return output_path


def run_sus_test(df):
    """Run paired t-test for SUS score."""
    df_sus = df[["Participant ID", "Condition", "SUS score"]].copy()
    df_sus.columns = ["subject", "condition", "score"]

    # Pivot for paired t-test
    df_wide = df_sus.pivot(index="subject", columns="condition", values="score")

    print("\nSUS Score Paired T-Test:")
    try:
        result = pg.ttest(
            df_wide["Meta Neuralband"],
            df_wide["Headphone"],
            paired=True,
        )
        print(result.to_string(index=False))

        p_col = "p_val" if "p_val" in result.columns else "p-val"
        p_val = result[p_col].values[0]
        if p_val < 0.05:
            marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
            return [(0, 1, marker)]
        return []
    except Exception as e:
        print(f"Error running SUS t-test: {e}")
        import traceback

        traceback.print_exc()
        return []


def plot_sus_score(df, significant_pairs=None):
    """Create SUS score barplot with condition colors."""
    output_dir = get_output_dir()

    if significant_pairs is None:
        significant_pairs = []

    df = df.copy()
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=CONDITION_ORDER, ordered=True
    )

    # Calculate means and SEs by condition
    stats_df = (
        df.groupby("Condition", observed=True)["SUS score"]
        .agg(["mean", "sem"])
        .reindex(CONDITION_ORDER)
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(1.8, 2.5))

    # Add bars with condition colors
    bar_width = 0.8
    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(
        x,
        stats_df["mean"],
        width=bar_width,
        yerr=stats_df["sem"],
        capsize=4,
        color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("")
    ax.set_ylabel("SUS Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_DISPLAY_LABELS, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Legend at bottom, horizontal
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        CONDITION_DISPLAY_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_multialignment("center")

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance brackets
    if significant_pairs:
        y_base = 98
        row_height = 12

        for i, (c1, c2, marker) in enumerate(significant_pairs):
            y = y_base
            add_significance_bracket(ax, c1, c2, y, 2, marker)

    plt.tight_layout()

    output_path = output_dir / "walking_sus_bracket.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nWalking SUS figure saved to: {output_path}")
    return output_path


def main():
    """Generate walking study subjective figures."""
    print("=" * 60)
    print("GENERATING WALKING STUDY SUBJECTIVE FIGURES")
    print("=" * 60)
    print()

    # Load data
    print("Loading walking survey data...")
    df = load_walking_survey_data()
    print(f"  Loaded {len(df)} responses")
    print(f"  Participants: {df['Participant ID'].unique()}")
    print(f"  Conditions: {df['Condition'].unique()}")
    print()

    # Run statistical tests
    print("Running paired t-tests for NASA-TLX...")
    nasa_tlx_significant = run_nasa_tlx_tests(df)
    print()

    print("Running paired t-test for SUS...")
    sus_significant_pairs = run_sus_test(df)
    print()

    # Generate figures
    print("\nGenerating figures...")
    plot_nasa_tlx(df, significance_by_dim=nasa_tlx_significant)
    plot_sus_score(df, significant_pairs=sus_significant_pairs)

    print()
    print("=" * 60)
    print("DONE - Figures saved to result/figures_final/")
    print("=" * 60)


if __name__ == "__main__":
    main()
