#!/usr/bin/env python3
"""
Survey data analysis for NASA-TLX and SUS scores.

Generates:
1. NASA-TLX grouped boxplot by condition
2. SUS score boxplot by condition
3. Statistical analysis (RM-ANOVA, post-hoc tests)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# Statistical packages
try:
    import pingouin as pg

    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not installed. Run: pip install pingouin")

try:
    from scipy import stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# Column mapping for NASA-TLX
NASA_TLX_COLUMNS = {
    "How mentally demanding was the task?": "Mental Demand",
    "How physically demanding was the task?": "Physical Demand",
    "How hurried or rushed was the pace of the task?": "Temporal Demand",
    "How successful were you in accomplishing what you were asked to do?": "Performance",
    "How hard did you have to work to accomplish your level of performance?": "Effort",
    "How insecure, discouraged, irritated, stressed, and annoyed were you?": "Frustration",
}

# Condition order for plotting
CONDITION_ORDER = ["Discontinuous", "Keyword", "Keyword2", "Word", "Sentence"]


def load_data():
    """Load survey data."""
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "result" / "survey.csv"

    # Skip first row (secondary header row)
    df = pd.read_csv(data_path, header=1)

    # Rename NASA-TLX columns
    df = df.rename(columns=NASA_TLX_COLUMNS)

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total responses: {len(df)}")
    print(f"Participants: {df['Participant ID'].nunique()}")
    print(f"Conditions: {df['Condition'].unique().tolist()}")
    print()

    return df


def add_significance_bracket(ax, x1, x2, y, h, text):
    """Add significance bracket between two bars."""
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.2, c="black")
    ax.text(
        (x1 + x2) / 2,
        y + h,
        text,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )


def assign_bracket_rows(brackets):
    """
    Assign brackets to rows so that non-overlapping brackets share the same row.

    Args:
        brackets: List of (x1, x2, marker) tuples

    Returns:
        List of row indices for each bracket
    """
    if not brackets:
        return []

    # Each row contains list of (x1, x2) ranges that are already placed
    rows = []
    row_assignments = []

    for x1, x2, _ in brackets:
        # Ensure x1 < x2
        left, right = min(x1, x2), max(x1, x2)

        # Find a row where this bracket doesn't overlap with existing ones
        placed = False
        for row_idx, row_ranges in enumerate(rows):
            overlaps = False
            for (r_left, r_right) in row_ranges:
                # Check if ranges overlap (with small margin for visual clarity)
                if not (right < r_left or left > r_right):
                    overlaps = True
                    break

            if not overlaps:
                # Place in this row
                row_ranges.append((left, right))
                row_assignments.append(row_idx)
                placed = True
                break

        if not placed:
            # Create a new row
            rows.append([(left, right)])
            row_assignments.append(len(rows) - 1)

    return row_assignments


def plot_nasa_tlx_barplot(df):
    """Create NASA-TLX subplots with standard error and significance markers."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "result" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    tlx_cols = list(NASA_TLX_COLUMNS.values())
    dimension_order = [
        "Mental Demand",
        "Physical Demand",
        "Temporal Demand",
        "Performance",
        "Effort",
        "Frustration",
    ]

    # Significance markers from post-hoc analysis
    # Format: {dimension: [(condition1_idx, condition2_idx, marker), ...]}
    # Condition order: Discontinuous(0), Keyword(1), Keyword2(2), Word(3), Sentence(4)
    significance = {
        "Mental Demand": [(0, 1, "*")],  # Discontinuous vs Keyword p=.041
        "Performance": [(1, 2, "*")],  # Keyword vs Keyword2 p=.031
        "Effort": [
            (0, 3, "*"),  # Discontinuous vs Word p=.039
            (2, 3, "**"),  # Keyword2 vs Word p=.001
            (0, 1, "*"),  # Discontinuous vs Keyword p=.022
            (1, 2, "**"),  # Keyword vs Keyword2 p=.002
        ],
    }

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    palette = sns.color_palette("Set2", n_colors=len(CONDITION_ORDER))

    for idx, dim in enumerate(dimension_order):
        ax = axes[idx]
        df_dim = df[["Condition", dim]].copy()
        df_dim["Condition"] = pd.Categorical(
            df_dim["Condition"], categories=CONDITION_ORDER, ordered=True
        )

        sns.barplot(
            data=df_dim,
            x="Condition",
            y=dim,
            ax=ax,
            palette=palette,
            order=CONDITION_ORDER,
            errorbar="se",
            capsize=0.15,
            err_kws={"linewidth": 1.5},
        )

        ax.set_xlabel("")
        ax.set_ylabel("Score (1-7)", fontsize=10)
        ax.set_title(dim, fontsize=12, fontweight="bold")
        ax.set_ylim(0, 9)
        ax.tick_params(axis="x", rotation=30)

        # Add significance brackets
        if dim in significance:
            brackets = significance[dim]
            row_assignments = assign_bracket_rows(brackets)
            y_base = 7.0
            row_height = 0.7
            for i, (x1, x2, marker) in enumerate(brackets):
                y = y_base + row_assignments[i] * row_height
                add_significance_bracket(ax, x1, x2, y, 0.15, marker)

    plt.tight_layout()

    output_path = output_dir / "nasa_tlx_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"NASA-TLX barplot saved to: {output_path}")
    return output_path


def plot_sus_barplot(df):
    """Create SUS score barplot with standard error and significance markers."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "result" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=CONDITION_ORDER, ordered=True
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Color palette
    palette = sns.color_palette("Set2", n_colors=len(CONDITION_ORDER))

    # Create barplot with standard error
    sns.barplot(
        data=df,
        x="Condition",
        y="SUS score",
        ax=ax,
        palette=palette,
        order=CONDITION_ORDER,
        errorbar="se",
        capsize=0.1,
        err_kws={"linewidth": 1.5},
    )

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("SUS Score", fontsize=12)
    ax.set_title(
        "System Usability Scale (SUS) Scores by Condition",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(0, 115)

    # Add significance brackets (from post-hoc analysis)
    # Word (3) vs Keyword2 (2): p=.028 *
    # Keyword (1) vs Keyword2 (2): p=.038 *
    brackets = [(2, 3, "*"), (1, 2, "*")]
    row_assignments = assign_bracket_rows(brackets)
    y_base = 95
    row_height = 8
    for i, (x1, x2, marker) in enumerate(brackets):
        y = y_base + row_assignments[i] * row_height
        add_significance_bracket(ax, x1, x2, y, 2, marker)

    plt.tight_layout()

    output_path = output_dir / "sus_score_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"SUS barplot saved to: {output_path}")
    return output_path


def descriptive_stats(df):
    """Print descriptive statistics."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    # NASA-TLX by condition
    print("\n--- NASA-TLX by Condition ---")
    tlx_cols = list(NASA_TLX_COLUMNS.values())

    for col in tlx_cols:
        print(f"\n{col}:")
        stats_df = df.groupby("Condition")[col].agg(["mean", "std", "count"])
        stats_df.columns = ["Mean", "SD", "N"]
        stats_df = stats_df.reindex(CONDITION_ORDER)
        print(stats_df.round(2).to_string())

    # SUS by condition
    print("\n\n--- SUS Score by Condition ---")
    sus_stats = df.groupby("Condition")["SUS score"].agg(["mean", "std", "count"])
    sus_stats.columns = ["Mean", "SD", "N"]
    sus_stats = sus_stats.reindex(CONDITION_ORDER)
    print(sus_stats.round(2).to_string())
    print()


def run_statistical_analysis(df):
    """Run RM-ANOVA and post-hoc tests."""
    if not HAS_PINGOUIN:
        print("Skipping statistical analysis (pingouin not installed)")
        return

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    tlx_cols = list(NASA_TLX_COLUMNS.values())

    # NASA-TLX RM-ANOVA for each dimension
    print("\n--- NASA-TLX: RM-ANOVA by Condition ---")

    for col in tlx_cols:
        print(f"\n{'='*40}")
        print(f"{col}")
        print("=" * 40)

        df_analysis = df[["Participant ID", "Condition", col]].copy()
        df_analysis.columns = ["subject", "condition", "score"]

        try:
            # RM-ANOVA
            aov = pg.rm_anova(
                data=df_analysis,
                dv="score",
                within="condition",
                subject="subject",
                detailed=True,
            )
            print("\nRM-ANOVA:")
            # Handle different pingouin versions
            if "DF" in aov.columns:
                print(aov[["Source", "DF", "F", "p_unc", "ng2"]].to_string(index=False))
                p_val = aov["p_unc"].values[0]
                eta2 = aov["ng2"].values[0]
            else:
                print(
                    aov[["Source", "ddof1", "ddof2", "F", "p-unc", "np2"]].to_string(
                        index=False
                    )
                )
                p_val = aov["p-unc"].values[0]
                eta2 = aov["np2"].values[0]

            # Effect size interpretation (generalized eta-squared)
            if eta2 < 0.01:
                effect = "negligible"
            elif eta2 < 0.06:
                effect = "small"
            elif eta2 < 0.14:
                effect = "medium"
            else:
                effect = "large"
            print(f"Effect size (eta-squared): {eta2:.4f} ({effect})")

            # Post-hoc if significant
            if p_val < 0.05:
                print("\nPost-hoc pairwise comparisons (Bonferroni corrected):")
                posthoc = pg.pairwise_tests(
                    data=df_analysis,
                    dv="score",
                    within="condition",
                    subject="subject",
                    padjust="bonferroni",
                )
                # Handle different pingouin versions
                if "p_unc" in posthoc.columns:
                    posthoc_display = posthoc[
                        ["A", "B", "T", "p_unc", "p_corr", "hedges"]
                    ].copy()
                    posthoc_display["sig"] = posthoc_display["p_corr"].apply(
                        lambda x: "***"
                        if x < 0.001
                        else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
                    )
                else:
                    posthoc_display = posthoc[
                        ["A", "B", "T", "p-unc", "p-corr", "hedges"]
                    ].copy()
                    posthoc_display["sig"] = posthoc_display["p-corr"].apply(
                        lambda x: "***"
                        if x < 0.001
                        else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
                    )
                print(posthoc_display.to_string(index=False))

        except Exception as e:
            print(f"Error: {e}")

    # SUS Score RM-ANOVA
    print("\n" + "=" * 60)
    print("SUS Score: RM-ANOVA")
    print("=" * 60)

    df_sus = df[["Participant ID", "Condition", "SUS score"]].copy()
    df_sus.columns = ["subject", "condition", "score"]

    try:
        aov_sus = pg.rm_anova(
            data=df_sus,
            dv="score",
            within="condition",
            subject="subject",
            detailed=True,
        )
        print("\nRM-ANOVA:")
        # Handle different pingouin versions
        if "DF" in aov_sus.columns:
            print(aov_sus[["Source", "DF", "F", "p_unc", "ng2"]].to_string(index=False))
            p_val = aov_sus["p_unc"].values[0]
            eta2 = aov_sus["ng2"].values[0]
        else:
            print(
                aov_sus[["Source", "ddof1", "ddof2", "F", "p-unc", "np2"]].to_string(
                    index=False
                )
            )
            p_val = aov_sus["p-unc"].values[0]
            eta2 = aov_sus["np2"].values[0]

        if eta2 < 0.01:
            effect = "negligible"
        elif eta2 < 0.06:
            effect = "small"
        elif eta2 < 0.14:
            effect = "medium"
        else:
            effect = "large"
        print(f"Effect size (eta-squared): {eta2:.4f} ({effect})")

        if p_val < 0.05:
            print("\nPost-hoc pairwise comparisons (Bonferroni corrected):")
            posthoc_sus = pg.pairwise_tests(
                data=df_sus,
                dv="score",
                within="condition",
                subject="subject",
                padjust="bonferroni",
            )
            # Handle different pingouin versions
            if "p_unc" in posthoc_sus.columns:
                posthoc_display = posthoc_sus[
                    ["A", "B", "T", "p_unc", "p_corr", "hedges"]
                ].copy()
                posthoc_display["sig"] = posthoc_display["p_corr"].apply(
                    lambda x: "***"
                    if x < 0.001
                    else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
                )
            else:
                posthoc_display = posthoc_sus[
                    ["A", "B", "T", "p-unc", "p-corr", "hedges"]
                ].copy()
                posthoc_display["sig"] = posthoc_display["p-corr"].apply(
                    lambda x: "***"
                    if x < 0.001
                    else ("**" if x < 0.01 else ("*" if x < 0.05 else ""))
                )
            print(posthoc_display.to_string(index=False))

    except Exception as e:
        print(f"Error: {e}")

    # Sphericity test
    print("\n" + "=" * 60)
    print("SPHERICITY TESTS (Mauchly's Test)")
    print("=" * 60)

    # SUS sphericity
    print("\n--- SUS Score ---")
    try:
        df_sus_wide = df_sus.pivot(index="subject", columns="condition", values="score")
        _, W, chi2, dof, pval = pg.sphericity(df_sus_wide)
        print(
            f"Mauchly's W = {W:.4f}, Chi-square = {chi2:.4f}, df = {dof}, p = {pval:.4f}"
        )
        if pval < 0.05:
            print(
                "Sphericity assumption VIOLATED - consider Greenhouse-Geisser correction"
            )
        else:
            print("Sphericity assumption MET")
    except Exception as e:
        print(f"Error: {e}")

    print()


def save_summary_csv(df):
    """Save summary statistics to CSV."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "result"

    tlx_cols = list(NASA_TLX_COLUMNS.values())

    # Create summary dataframe
    summary_rows = []

    for condition in CONDITION_ORDER:
        df_cond = df[df["Condition"] == condition]
        row = {"Condition": condition}

        # NASA-TLX
        for col in tlx_cols:
            row[f"{col}_mean"] = df_cond[col].mean()
            row[f"{col}_sd"] = df_cond[col].std()

        # SUS
        row["SUS_mean"] = df_cond["SUS score"].mean()
        row["SUS_sd"] = df_cond["SUS score"].std()
        row["N"] = len(df_cond)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    output_path = output_dir / "survey_summary.csv"
    summary_df.to_csv(output_path, index=False)
    print(f"Summary saved to: {output_path}")


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 60)
    print("SURVEY DATA ANALYSIS")
    print("NASA-TLX & SUS Scores")
    print("=" * 60 + "\n")

    # Load data
    df = load_data()

    # Descriptive statistics
    descriptive_stats(df)

    # Generate plots
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    plot_nasa_tlx_barplot(df)
    plot_sus_barplot(df)

    # Statistical analysis
    run_statistical_analysis(df)

    # Save summary
    save_summary_csv(df)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
