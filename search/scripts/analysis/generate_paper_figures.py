#!/usr/bin/env python3
"""
Generate paper figures with consistent styling.

Changes from original:
1. Condition renaming:
   - temporal -> Temporal
   - keyword -> Keyword (Target Present)
   - keyword2 -> Keyword (Target Absent)
   - word -> Word
   - sentence -> Sentence

2. Condition order: Temporal, Sentence, Word, Keyword (Target Present), Keyword (Target Absent)

3. NASA-TLX: 6 subscales in one row, conditions distinguished by color

4. Consistent color palette across all figures

5. Smaller figure sizes for paper
"""

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch

# Condition mapping (old -> new)
CONDITION_MAP = {
    "temporal": "Temporal",
    "temporal": "Temporal",
    "Keyword": "Keyword (Target Present)",
    "keyword": "Keyword (Target Present)",
    "Keyword2": "Keyword (Target Absent)",
    "keyword2": "Keyword (Target Absent)",
    "Word": "Word",
    "word": "Word",
    "Sentence": "Sentence",
    "sentence": "Sentence",
}

# New condition order
CONDITION_ORDER = [
    "Temporal",
    "Sentence",
    "Word",
    "Keyword (Target Present)",
    "Keyword (Target Absent)",
]

# Display labels for all figures
CONDITION_DISPLAY_LABELS = [
    "Temporal",
    "Sentence",
    "Word",
    "Keyword\n(Target)",
    "Keyword\n(No Target)",
]

# Condition index mapping for significance brackets
# Order: Temporal(0), Sentence(1), Word(2), Keyword (Target Present)(3), Keyword (Target Absent)(4)
COND_IDX = {c: i for i, c in enumerate(CONDITION_ORDER)}

# NASA-TLX column mapping
NASA_TLX_COLUMNS = {
    "How mentally demanding was the task?": "Mental Demand",
    "How physically demanding was the task?": "Physical Demand",
    "How hurried or rushed was the pace of the task?": "Temporal Demand",
    "How successful were you in accomplishing what you were asked to do?": "Performance",
    "How hard did you have to work to accomplish your level of performance?": "Effort",
    "How insecure, discouraged, irritated, stressed, and annoyed were you?": "Frustration",
}

# Consistent color palette - single color for all bars
BAR_COLOR = "#4C78A8"  # Uniform blue color for all bars

# Condition colors
# Order: Temporal, Sentence, Word, Keyword (Target Present), Keyword (Target Absent)
CONDITION_COLORS = {
    "Temporal": "#808080",  # Gray
    "Sentence": "#4C78A8",  # Blue
    "Keyword (Target Present)": "#59A14F",  # Green
    "Keyword (Target Absent)": "#8CD17D",  # Light green (same family)
    "Word": "#E15759",  # Red/coral accent
}


def get_output_dir():
    """Get output directory for figures."""
    script_dir = Path(__file__).parent
    output_dir = script_dir.parent.parent / "result" / "figures_final"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_survey_data():
    """Load and preprocess survey data."""
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "result" / "survey.csv"

    df = pd.read_csv(data_path, header=1)
    df = df.rename(columns=NASA_TLX_COLUMNS)

    # Map condition names
    df["Condition"] = df["Condition"].map(CONDITION_MAP)

    return df


def load_word_search_data():
    """Load and preprocess word search trial data."""
    script_dir = Path(__file__).parent
    logs_dir = script_dir.parent.parent / "logs" / "study"

    # Load interruption metadata
    meta = {}
    for path in sorted(logs_dir.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                event = json.loads(line)
                audio = event.get("audio", "")
                if audio == "0.mp3" or audio.startswith("tutorial"):
                    continue
                if event.get("event") != "interruption":
                    continue
                log_base = event["logBaseName"]
                pid = log_base.split("_")[0]
                meta[(log_base, event["taskIndex"])] = {
                    "participant_id": pid,
                    "condition": event["feature"],
                    "audio": event["audio"],
                    "word_search_type": event["delayType"],
                }

    # Build trial rows
    rows = []
    for path in sorted(logs_dir.glob("*.json")):
        with path.open() as f:
            session = json.load(f)
        audio = session.get("audio", "")
        if audio == "0.mp3" or audio.startswith("tutorial"):
            continue

        log_base = session["logBaseName"]
        pid = log_base.split("_")[0]

        for task in session.get("tasks", []):
            key = (log_base, task["taskIndex"])
            if key not in meta:
                continue
            info = meta[key]
            distance = task.get("distanceSeconds")
            threshold = 2.0
            success = (
                task.get("outcome") == "spacebar"
                and distance is not None
                and abs(distance) <= threshold
            )
            failure = task.get("outcome") == "timeout" or (
                distance is not None and abs(distance) > threshold
            )

            if task.get("responseTimeMs") is not None:
                time_sec = task["responseTimeMs"] / 1000
            else:
                time_sec = None

            rows.append(
                {
                    "participant_id": info["participant_id"],
                    "condition": CONDITION_MAP.get(
                        info["condition"], info["condition"]
                    ),
                    "word_search_type": info["word_search_type"],
                    "word_search_success": success,
                    "word_search_failure": failure,
                    "word_search_time_sec": time_sec,
                }
            )

    return rows


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


def adjust_shared_endpoints(brackets, offset=0.03):
    """
    Adjust bracket endpoints that are shared to separate them slightly.
    Returns list of (x1_adj, x2_adj) for each bracket.
    """
    if not brackets:
        return []

    # Collect all endpoints and which brackets use them
    endpoint_usage = {}  # endpoint -> list of (bracket_idx, 'left' or 'right')
    for i, (c1, c2, _) in enumerate(brackets):
        left, right = min(c1, c2), max(c1, c2)
        if left not in endpoint_usage:
            endpoint_usage[left] = []
        if right not in endpoint_usage:
            endpoint_usage[right] = []
        endpoint_usage[left].append((i, "left"))
        endpoint_usage[right].append((i, "right"))

    # Initialize adjusted coordinates
    adjusted = []
    for c1, c2, _ in brackets:
        adjusted.append([min(c1, c2), max(c1, c2)])

    # For shared endpoints, apply offset
    for endpoint, usages in endpoint_usage.items():
        if len(usages) > 1:
            # Sort by bracket index for consistent ordering
            usages.sort(key=lambda x: x[0])
            for idx, (bracket_idx, side) in enumerate(usages):
                if side == "left":
                    # Left endpoint of this bracket - move right
                    adjusted[bracket_idx][0] = endpoint + offset
                else:
                    # Right endpoint of this bracket - move left
                    adjusted[bracket_idx][1] = endpoint - offset

    return adjusted


def assign_bracket_rows(brackets):
    """
    Assign brackets to rows so that non-overlapping brackets share the same row.
    """
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
                # Allow adjacent brackets (sharing endpoint) on same row
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


def add_significance_column(posthoc_df, p_col="p_corr"):
    """Add significance level column and sort by p_corr ascending."""
    df = posthoc_df.copy()

    def get_sig_level(p):
        if p < 0.001:
            return "p < .001"
        elif p < 0.01:
            return "p < .01"
        elif p < 0.05:
            return "p < .05"
        else:
            return "n.s."

    df["Sig."] = df[p_col].apply(get_sig_level)
    df = df.sort_values(by=p_col, ascending=True)
    return df


def run_nasa_tlx_anova(df):
    """Run RM-ANOVA and post-hoc tests for NASA-TLX dimensions with Holm correction."""
    dimension_order = [
        "Mental Demand",
        "Physical Demand",
        "Temporal Demand",
        "Performance",
        "Effort",
        "Frustration",
    ]

    significance_by_dim = {}

    for dim in dimension_order:
        df_dim = df[["Participant ID", "Condition", dim]].copy()
        df_dim.columns = ["subject", "condition", "score"]

        print(f"\n{dim} RM-ANOVA:")
        try:
            aov = pg.rm_anova(
                data=df_dim,
                dv="score",
                within="condition",
                subject="subject",
            )
            print(aov.to_string(index=False))

            # Check if significant
            p_col_aov = "p-unc" if "p-unc" in aov.columns else "p_unc"
            if aov[p_col_aov].values[0] < 0.05:
                posthoc = pg.pairwise_tests(
                    data=df_dim,
                    dv="score",
                    within="condition",
                    subject="subject",
                    padjust="holm",
                )
                p_col = "p_corr" if "p_corr" in posthoc.columns else "p-corr"
                posthoc = add_significance_column(posthoc, p_col)
                print(f"\n{dim} Post-hoc (Holm corrected):")
                print(posthoc.to_string(index=False))

                # Collect significant pairs
                significant_pairs = []
                p_col = "p_corr" if "p_corr" in posthoc.columns else "p-corr"
                for _, row in posthoc.iterrows():
                    if row[p_col] < 0.05:
                        c1_idx = CONDITION_ORDER.index(row["A"])
                        c2_idx = CONDITION_ORDER.index(row["B"])
                        marker = (
                            "***"
                            if row[p_col] < 0.001
                            else "**"
                            if row[p_col] < 0.01
                            else "*"
                        )
                        significant_pairs.append((c1_idx, c2_idx, marker))

                if significant_pairs:
                    significance_by_dim[dim] = significant_pairs
        except Exception as e:
            print(f"Error running NASA-TLX ANOVA for {dim}: {e}")

    return significance_by_dim


def plot_nasa_tlx(df, use_brackets=False, significance_by_dim=None):
    """Create NASA-TLX grouped barplot with all 6 subscales in one row."""
    output_dir = get_output_dir()

    if significance_by_dim is None:
        significance_by_dim = {}

    dimension_order = [
        "Mental Demand",
        "Physical Demand",
        "Temporal Demand",
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

    # Create figure - wider for bottom legend
    fig, ax = plt.subplots(figsize=(9, 3))

    # Create grouped barplot with gap between groups
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
        gap=0.15,  # Add gap between bar groups
    )

    ax.set_xlabel("")
    ax.set_ylabel("Score (1-7)", fontsize=10)
    ax.set_ylim(0, 9.5)  # Space for brackets
    ax.set_yticks([1, 2, 3, 4, 5, 6, 7])
    ax.tick_params(axis="x", rotation=0, labelsize=9)  # No rotation
    ax.tick_params(axis="y", labelsize=9)

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Calculate bar positions for grouped barplot
    n_dims = len(dimension_order)
    n_conds = len(CONDITION_ORDER)
    bar_width = 0.8 / n_conds

    for dim_idx, dim in enumerate(dimension_order):
        if dim not in significance_by_dim:
            continue

        brackets = significance_by_dim[dim]
        row_assignments = assign_bracket_rows(brackets)
        adjusted_endpoints = adjust_shared_endpoints(brackets, offset=0.06)
        y_base = 7.3
        row_height = 0.9 if use_brackets else 0.4

        for i, (c1, c2, marker) in enumerate(brackets):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            x1 = dim_idx + (adj_c1 - (n_conds - 1) / 2) * bar_width
            x2 = dim_idx + (adj_c2 - (n_conds - 1) / 2) * bar_width
            y = y_base + row_assignments[i] * row_height
            if use_brackets:
                add_significance_bracket(ax, x1, x2, y, 0.18, marker, fontsize=8)
            else:
                x_mid = (x1 + x2) / 2
                add_significance_marker(ax, x_mid, y, marker, fontsize=8)

    # Legend at bottom, horizontal
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        CONDITION_DISPLAY_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,  # All in one row
        fontsize=9,
        frameon=False,
    )
    # Center-align multi-line legend labels
    for text in legend.get_texts():
        text.set_multialignment("center")

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"nasa_tlx{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"NASA-TLX figure saved to: {output_path}")
    return output_path


def run_word_search_anova():
    """Run RM-ANOVA and post-hoc tests for word search time."""
    script_dir = Path(__file__).parent
    result_dir = script_dir.parent.parent / "result"

    # Condition name mapping for CSV columns
    col_map = {
        "temporal": "Temporal",
        "keyword": "Keyword (Target Present)",
        "keyword2": "Keyword (Target Absent)",
        "word": "Word",
        "sentence": "Sentence",
    }

    significant_pairs = {"Overall": [], "Short-lag": [], "Long-lag": []}

    # Process each lag type
    for lag_type, suffix in [
        ("Overall", ""),
        ("Short-lag", "_short"),
        ("Long-lag", "_long"),
    ]:
        if suffix == "":
            # Overall: combine short and long
            df_short = pd.read_csv(result_dir / "word_search_summary_wide_short.csv")
            df_long = pd.read_csv(result_dir / "word_search_summary_wide_long.csv")

            # Average short and long for each condition
            data = {"participant_id": df_short["participant_id"]}
            for old_name, new_name in col_map.items():
                short_col = f"{old_name}_short"
                long_col = f"{old_name}_long"
                if short_col in df_short.columns and long_col in df_long.columns:
                    data[new_name] = (df_short[short_col] + df_long[long_col]) / 2
            df_wide = pd.DataFrame(data)
        else:
            csv_path = result_dir / f"word_search_summary_wide{suffix}.csv"
            df_wide = pd.read_csv(csv_path)
            # Rename columns
            rename_map = {"participant_id": "participant_id"}
            for old_name, new_name in col_map.items():
                old_col = f"{old_name}{suffix}"
                if old_col in df_wide.columns:
                    rename_map[old_col] = new_name
            df_wide = df_wide.rename(columns=rename_map)

        # Melt to long format
        df_long_fmt = df_wide.melt(
            id_vars=["participant_id"],
            value_vars=CONDITION_ORDER,
            var_name="Condition",
            value_name="Time",
        )

        # Run RM-ANOVA
        try:
            aov = pg.rm_anova(
                data=df_long_fmt,
                dv="Time",
                within="Condition",
                subject="participant_id",
            )
            print(f"\n{lag_type} RM-ANOVA:")
            print(aov.to_string(index=False))

            # Post-hoc with Holm correction
            posthoc = pg.pairwise_tests(
                data=df_long_fmt,
                dv="Time",
                within="Condition",
                subject="participant_id",
                padjust="holm",
            )
            # Collect significant pairs - find the p-value column
            p_col = None
            for col in ["p-corr", "p_corr", "p-unc", "p_unc", "pval"]:
                if col in posthoc.columns:
                    p_col = col
                    break

            if p_col:
                posthoc = add_significance_column(posthoc, p_col)

            print(f"\n{lag_type} Post-hoc (Holm corrected):")
            print(posthoc.to_string(index=False))

            if p_col:
                for _, row in posthoc.iterrows():
                    if row[p_col] < 0.05:
                        c1_idx = CONDITION_ORDER.index(row["A"])
                        c2_idx = CONDITION_ORDER.index(row["B"])
                        marker = (
                            "***"
                            if row[p_col] < 0.001
                            else "**"
                            if row[p_col] < 0.01
                            else "*"
                        )
                        significant_pairs[lag_type].append((c1_idx, c2_idx, marker))
            else:
                print(f"Warning: No p-value column found in post-hoc results")
        except Exception as e:
            print(f"Error running ANOVA for {lag_type}: {e}")
            import traceback

            traceback.print_exc()

    return significant_pairs


def plot_word_search_time(rows, use_brackets=False, significant_pairs=None):
    """Create word search time grouped barplot: X=lag type, color=condition."""
    output_dir = get_output_dir()

    if significant_pairs is None:
        significant_pairs = {}

    # Group data by condition and word_search_type
    data_records = []

    for row in rows:
        condition = row["condition"]
        word_type = row["word_search_type"]

        if row["word_search_success"]:
            if row["word_search_time_sec"] is not None:
                time_sec = row["word_search_time_sec"]
            else:
                continue
        elif row["word_search_failure"]:
            time_sec = 60.0
        else:
            continue

        # Add to Overall
        data_records.append(
            {
                "Condition": condition,
                "Lag Type": "Overall",
                "Time": time_sec,
            }
        )
        # Add to specific lag type
        if word_type == "short":
            data_records.append(
                {
                    "Condition": condition,
                    "Lag Type": "Short-lag",
                    "Time": time_sec,
                }
            )
        elif word_type == "long":
            data_records.append(
                {
                    "Condition": condition,
                    "Lag Type": "Long-lag",
                    "Time": time_sec,
                }
            )

    df = pd.DataFrame(data_records)
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=CONDITION_ORDER, ordered=True
    )
    lag_order = ["Overall", "Short-lag", "Long-lag"]
    df["Lag Type"] = pd.Categorical(df["Lag Type"], categories=lag_order, ordered=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(7, 3.5))

    sns.barplot(
        data=df,
        x="Lag Type",
        y="Time",
        hue="Condition",
        ax=ax,
        palette=CONDITION_COLORS,
        order=lag_order,
        hue_order=CONDITION_ORDER,
        errorbar="se",
        capsize=0.05,
        err_kws={"linewidth": 1.0},
        gap=0.15,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Time (sec)", fontsize=10)
    ax.set_ylim(0, 80)  # Space for brackets
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance brackets for each lag type
    n_conds = len(CONDITION_ORDER)
    bar_width = 0.8 / n_conds

    for lag_idx, lag_type in enumerate(lag_order):
        brackets = significant_pairs.get(lag_type, [])
        if not brackets:
            continue

        row_assignments = assign_bracket_rows(brackets)
        adjusted_endpoints = adjust_shared_endpoints(brackets, offset=0.06)
        y_base = 48
        row_height = 6 if use_brackets else 4

        for i, (c1, c2, marker) in enumerate(brackets):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            x1 = lag_idx + (adj_c1 - (n_conds - 1) / 2) * bar_width
            x2 = lag_idx + (adj_c2 - (n_conds - 1) / 2) * bar_width
            y = y_base + row_assignments[i] * row_height
            if use_brackets:
                add_significance_bracket(ax, x1, x2, y, 1.8, marker, fontsize=8)
            else:
                x_mid = (x1 + x2) / 2
                add_significance_marker(ax, x_mid, y, marker, fontsize=8)

    # Legend at bottom, horizontal
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        CONDITION_DISPLAY_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    # Center-align multi-line legend labels
    for text in legend.get_texts():
        text.set_multialignment("center")

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"word_search_time{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nWord search time figure saved to: {output_path}")
    return output_path


def plot_word_search_time_overall(rows, use_brackets=False, significant_pairs=None):
    """Create overall-only word search time bar chart: X=condition."""
    output_dir = get_output_dir()

    if significant_pairs is None:
        significant_pairs = {}

    data_records = []
    for row in rows:
        condition = row["condition"]

        if row["word_search_success"]:
            if row["word_search_time_sec"] is None:
                continue
            time_sec = row["word_search_time_sec"]
        elif row["word_search_failure"]:
            time_sec = 60.0
        else:
            continue

        data_records.append({"Condition": condition, "Time": time_sec})

    df = pd.DataFrame(data_records)
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=CONDITION_ORDER, ordered=True
    )

    fig, ax = plt.subplots(figsize=(4, 3.4))

    sns.barplot(
        data=df,
        x="Condition",
        y="Time",
        ax=ax,
        order=CONDITION_ORDER,
        palette=[CONDITION_COLORS[c] for c in CONDITION_ORDER],
        errorbar="se",
        capsize=0.05,
        err_kws={"linewidth": 1.0},
    )

    ax.set_xlabel("")
    ax.set_ylabel("Time (sec)", fontsize=10)
    ax.set_ylim(0, 80)
    ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
    ax.set_xticklabels(CONDITION_DISPLAY_LABELS)
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", labelsize=9)

    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    overall_brackets = significant_pairs.get("Overall", [])
    if overall_brackets and use_brackets:
        row_assignments = assign_bracket_rows(overall_brackets)
        adjusted_endpoints = adjust_shared_endpoints(overall_brackets, offset=0.08)
        y_base = 48
        row_height = 6

        for i, (_, _, marker) in enumerate(overall_brackets):
            x1, x2 = adjusted_endpoints[i]
            y = y_base + row_assignments[i] * row_height
            add_significance_bracket(ax, x1, x2, y, 1.8, marker, fontsize=8)

    rank_palette = sns.blend_palette(["#1a9850", "#f0f0f0", "#d73027"], n_colors=5)
    legend_handles = [
        Patch(facecolor=rank_palette[i], edgecolor="none", label=str(i + 1))
        for i in range(5)
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"word_search_time_overall{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nOverall word search time figure saved to: {output_path}")
    return output_path


def build_search_accuracy_df(rows):
    """Build participant-level search accuracy table for plotting and RM-ANOVA."""
    data_records = []
    lag_order = ["Overall", "Short-lag", "Long-lag"]

    for row in rows:
        participant_id = row["participant_id"]
        condition = row["condition"]
        word_type = row["word_search_type"]

        if row["word_search_success"]:
            accuracy = 1.0
        elif row["word_search_failure"]:
            accuracy = 0.0
        else:
            continue

        data_records.append(
            {
                "participant_id": participant_id,
                "Condition": condition,
                "Lag Type": "Overall",
                "Accuracy": accuracy,
            }
        )

        if word_type == "short":
            data_records.append(
                {
                    "participant_id": participant_id,
                    "Condition": condition,
                    "Lag Type": "Short-lag",
                    "Accuracy": accuracy,
                }
            )
        elif word_type == "long":
            data_records.append(
                {
                    "participant_id": participant_id,
                    "Condition": condition,
                    "Lag Type": "Long-lag",
                    "Accuracy": accuracy,
                }
            )

    df = pd.DataFrame(data_records)
    df = (
        df.groupby(["participant_id", "Condition", "Lag Type"], observed=True)[
            "Accuracy"
        ]
        .mean()
        .reset_index()
    )
    df["Condition"] = pd.Categorical(
        df["Condition"], categories=CONDITION_ORDER, ordered=True
    )
    df["Lag Type"] = pd.Categorical(df["Lag Type"], categories=lag_order, ordered=True)
    return df


def run_search_accuracy_anova(rows):
    """Run RM-ANOVA and post-hoc tests for participant-level search accuracy."""
    df = build_search_accuracy_df(rows)
    significant_pairs = {"Overall": [], "Short-lag": [], "Long-lag": []}

    for lag_type in ["Overall", "Short-lag", "Long-lag"]:
        df_lag = df[df["Lag Type"] == lag_type].copy()

        try:
            aov = pg.rm_anova(
                data=df_lag,
                dv="Accuracy",
                within="Condition",
                subject="participant_id",
            )
            print(f"\n{lag_type} Search Accuracy RM-ANOVA:")
            print(aov.to_string(index=False))

            posthoc = pg.pairwise_tests(
                data=df_lag,
                dv="Accuracy",
                within="Condition",
                subject="participant_id",
                padjust="holm",
            )
            p_col = None
            for col in ["p-corr", "p_corr", "p-unc", "p_unc", "pval"]:
                if col in posthoc.columns:
                    p_col = col
                    break

            if p_col:
                posthoc = add_significance_column(posthoc, p_col)

            print(f"\n{lag_type} Search Accuracy Post-hoc (Holm corrected):")
            print(posthoc.to_string(index=False))

            if p_col:
                for _, row in posthoc.iterrows():
                    if row[p_col] < 0.05:
                        c1_idx = CONDITION_ORDER.index(row["A"])
                        c2_idx = CONDITION_ORDER.index(row["B"])
                        marker = (
                            "***"
                            if row[p_col] < 0.001
                            else "**"
                            if row[p_col] < 0.01
                            else "*"
                        )
                        significant_pairs[lag_type].append((c1_idx, c2_idx, marker))
            else:
                print(
                    "Warning: No p-value column found in search accuracy post-hoc results"
                )
        except Exception as e:
            print(f"Error running search accuracy ANOVA for {lag_type}: {e}")
            import traceback

            traceback.print_exc()

    return significant_pairs


def plot_search_accuracy(rows, use_brackets=False, significant_pairs=None):
    """Create participant-level search accuracy grouped barplot."""
    output_dir = get_output_dir()
    lag_order = ["Overall", "Short-lag", "Long-lag"]

    if significant_pairs is None:
        significant_pairs = {}

    df = build_search_accuracy_df(rows)

    fig, ax = plt.subplots(figsize=(7, 3.0))

    sns.barplot(
        data=df,
        x="Lag Type",
        y="Accuracy",
        hue="Condition",
        ax=ax,
        palette=CONDITION_COLORS,
        order=lag_order,
        hue_order=CONDITION_ORDER,
        errorbar="se",
        capsize=0.05,
        err_kws={"linewidth": 1.0},
        gap=0.15,
    )

    accuracy_summary = (
        df.groupby(["Lag Type", "Condition"], observed=True)["Accuracy"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    ax.set_xlabel("")
    ax.set_ylabel("Success Rate (%)", fontsize=10)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{int(v * 100)}" for v in np.arange(0, 1.01, 0.2)])
    ax.tick_params(axis="x", rotation=0, labelsize=10)
    ax.tick_params(axis="y", labelsize=9)

    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    n_conds = len(CONDITION_ORDER)
    bar_width = 0.8 / n_conds
    max_bracket_y = 1.0

    for lag_idx, lag_type in enumerate(lag_order):
        brackets = significant_pairs.get(lag_type, [])
        if not brackets:
            continue

        row_assignments = assign_bracket_rows(brackets)
        adjusted_endpoints = adjust_shared_endpoints(brackets, offset=0.06)
        lag_max = accuracy_summary.loc[
            accuracy_summary["Lag Type"] == lag_type, ["mean", "sem"]
        ].fillna(0)
        y_base = (lag_max["mean"] + lag_max["sem"]).max() + 0.05
        row_height = 0.12 if use_brackets else 0.05

        for i, (c1, c2, marker) in enumerate(brackets):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            x1 = lag_idx + (adj_c1 - (n_conds - 1) / 2) * bar_width
            x2 = lag_idx + (adj_c2 - (n_conds - 1) / 2) * bar_width
            y = y_base + row_assignments[i] * row_height
            max_bracket_y = max(max_bracket_y, y + (0.02 if use_brackets else 0.03))
            if use_brackets:
                add_significance_bracket(ax, x1, x2, y, 0.02, marker, fontsize=8)
            else:
                x_mid = (x1 + x2) / 2
                add_significance_marker(ax, x_mid, y, marker, fontsize=8)

    ax.set_ylim(0, min(1.25, max_bracket_y + 0.08))

    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles,
        CONDITION_DISPLAY_LABELS,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        fontsize=9,
        frameon=False,
    )
    for text in legend.get_texts():
        text.set_multialignment("center")

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"search_accuracy{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"\nSearch accuracy figure saved to: {output_path}")
    return output_path


def load_preference_data():
    """Load preference data from CSV.

    Returns:
        Dict mapping condition name to list of ranks (one per participant)
    """
    from collections import defaultdict

    script_dir = Path(__file__).parent
    csv_path = script_dir.parent.parent / "result" / "preference_order.csv"

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
                feature = "temporal"
            # Map to new condition names
            condition = CONDITION_MAP.get(feature, feature)
            feature_ranks[condition].append(rank)

    return dict(feature_ranks)


def run_preference_friedman():
    """Run Friedman test and post-hoc Wilcoxon signed-rank tests for preference rankings."""
    feature_ranks = load_preference_data()

    # Build matrix: rows = participants, cols = conditions (in CONDITION_ORDER)
    rank_matrix = np.array([feature_ranks[cond] for cond in CONDITION_ORDER]).T

    # Friedman test
    stat, p_value = stats.friedmanchisquare(
        *[rank_matrix[:, i] for i in range(len(CONDITION_ORDER))]
    )
    print(f"\nPreference Order Friedman Test:")
    print(f"  Chi-square = {stat:.3f}, p = {p_value:.4f}")

    # Post-hoc pairwise Wilcoxon signed-rank tests with Holm correction
    n_conds = len(CONDITION_ORDER)
    pairwise_results = []

    for i in range(n_conds):
        for j in range(i + 1, n_conds):
            cond_i = CONDITION_ORDER[i]
            cond_j = CONDITION_ORDER[j]
            ranks_i = rank_matrix[:, i]
            ranks_j = rank_matrix[:, j]

            # Wilcoxon signed-rank test
            try:
                stat_w, p_w = stats.wilcoxon(ranks_i, ranks_j)
                pairwise_results.append(
                    {
                        "A": cond_i,
                        "B": cond_j,
                        "i": i,
                        "j": j,
                        "statistic": stat_w,
                        "p_unc": p_w,
                    }
                )
            except Exception as e:
                print(f"  Warning: Wilcoxon test failed for {cond_i} vs {cond_j}: {e}")

    # Apply Holm correction
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

    # Add significance level and sort by p_corr
    for r in pairwise_results:
        p = r["p_corr"]
        if p < 0.001:
            r["Sig."] = "p < .001"
        elif p < 0.01:
            r["Sig."] = "p < .01"
        elif p < 0.05:
            r["Sig."] = "p < .05"
        else:
            r["Sig."] = "n.s."
    pairwise_results = sorted(pairwise_results, key=lambda x: x["p_corr"])

    print(f"\nPreference Order Post-hoc (Wilcoxon with Holm correction):")
    print(f"  {'A':<30} {'B':<30} {'p_unc':<10} {'p_corr':<10} {'Sig.':<10}")
    for r in pairwise_results:
        print(
            f"  {r['A']:<30} {r['B']:<30} {r['p_unc']:<10.4f} {r['p_corr']:<10.4f} {r['Sig.']:<10}"
        )

    # Collect significant pairs
    significant_pairs = []
    for r in pairwise_results:
        if r["p_corr"] < 0.05:
            marker = (
                "***" if r["p_corr"] < 0.001 else "**" if r["p_corr"] < 0.01 else "*"
            )
            significant_pairs.append((r["i"], r["j"], marker))

    return significant_pairs


def plot_preference_order(use_brackets=False, significant_pairs=None):
    """Create preference order stacked bar chart with paper styling."""
    output_dir = get_output_dir()

    if significant_pairs is None:
        significant_pairs = []

    feature_ranks = load_preference_data()
    n_participants = len(list(feature_ranks.values())[0])

    # Count how many times each condition got each rank
    ranks = [1, 2, 3, 4, 5]
    rank_counts = {cond: {r: 0 for r in ranks} for cond in CONDITION_ORDER}
    for cond in CONDITION_ORDER:
        for rank in feature_ranks.get(cond, []):
            rank_counts[cond][rank] += 1

    # Prepare data for stacking
    x = np.arange(len(CONDITION_ORDER))
    width = 0.65

    # Use a green-light gray-red diverging palette for rank preference.
    rank_palette = sns.blend_palette(["#1a9850", "#f0f0f0", "#d73027"], n_colors=5)
    rank_colors = {rank: rank_palette[rank - 1] for rank in ranks}

    fig, ax = plt.subplots(figsize=(4, 3.4))

    # Stack from rank 5 (bottom) to rank 1 (top)
    bottom = np.zeros(len(CONDITION_ORDER))
    for rank in [5, 4, 3, 2, 1]:
        counts = [rank_counts[c][rank] for c in CONDITION_ORDER]
        ax.bar(
            x,
            counts,
            width,
            label=f"{rank}",
            bottom=bottom,
            color=rank_colors[rank],
        )
        bottom += counts

    ax.set_xlabel("")
    ax.set_ylabel("Count", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITION_DISPLAY_LABELS, rotation=0, ha="center", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, n_participants + 8)  # Extra space for brackets
    ax.set_yticks(np.arange(0, n_participants + 1, 3))

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
        y_base = n_participants + 1.5
        row_height = 2.0

        for i, (_, _, marker) in enumerate(significant_pairs):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            y = y_base + row_assignments[i] * row_height
            add_significance_bracket(ax, adj_c1, adj_c2, y, 0.5, marker, fontsize=8)

    # Reverse legend order to show Rank 1 at top
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=5,
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"preference_order{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"Preference order figure saved to: {output_path}")
    return output_path


def run_sus_anova(df):
    """Run RM-ANOVA and post-hoc tests for SUS score with Holm correction."""
    df_sus = df[["Participant ID", "Condition", "SUS score"]].copy()
    df_sus.columns = ["subject", "condition", "score"]

    print("\nSUS Score RM-ANOVA:")
    try:
        aov = pg.rm_anova(
            data=df_sus,
            dv="score",
            within="condition",
            subject="subject",
        )
        print(aov.to_string(index=False))

        posthoc = pg.pairwise_tests(
            data=df_sus,
            dv="score",
            within="condition",
            subject="subject",
            padjust="holm",
        )
        # Collect significant pairs
        p_col = "p_corr" if "p_corr" in posthoc.columns else "p-corr"
        posthoc = add_significance_column(posthoc, p_col)
        print("\nSUS Post-hoc (Holm corrected):")
        print(posthoc.to_string(index=False))

        significant_pairs = []
        for _, row in posthoc.iterrows():
            if row[p_col] < 0.05:
                c1_idx = CONDITION_ORDER.index(row["A"])
                c2_idx = CONDITION_ORDER.index(row["B"])
                marker = (
                    "***" if row[p_col] < 0.001 else "**" if row[p_col] < 0.01 else "*"
                )
                significant_pairs.append((c1_idx, c2_idx, marker))

        return significant_pairs
    except Exception as e:
        print(f"Error running SUS ANOVA: {e}")
        return []


def plot_sus_score(df, use_brackets=False, significant_pairs=None):
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

    # Create figure - smaller size
    fig, ax = plt.subplots(figsize=(4, 3.2))

    # Add spacing between bars with condition colors
    bar_width = 0.75
    x = np.arange(len(CONDITION_ORDER))
    bars = ax.bar(
        x,
        stats_df["mean"],
        width=bar_width,
        yerr=stats_df["sem"],
        capsize=4,
        color=[CONDITION_COLORS[c] for c in CONDITION_ORDER],  # Condition colors
        edgecolor="white",
        linewidth=0.5,
    )

    ax.set_xlabel("")
    ax.set_ylabel("SUS Score", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(
        CONDITION_DISPLAY_LABELS, rotation=0, ha="center", fontsize=9
    )  # No rotation
    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylim(0, 125)  # Space for brackets
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    # Light horizontal grid lines
    ax.yaxis.grid(True, linestyle="-", alpha=0.2, color="gray")
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add significance brackets from Holm-corrected post-hoc analysis
    if significant_pairs:
        row_assignments = assign_bracket_rows(significant_pairs)
        adjusted_endpoints = adjust_shared_endpoints(significant_pairs, offset=0.05)
        y_base = 98
        row_height = 12 if use_brackets else 6

        for i, (c1, c2, marker) in enumerate(significant_pairs):
            adj_c1, adj_c2 = adjusted_endpoints[i]
            y = y_base + row_assignments[i] * row_height
            if use_brackets:
                add_significance_bracket(ax, adj_c1, adj_c2, y, 2, marker)
            else:
                x_mid = (c1 + c2) / 2
                add_significance_marker(ax, x_mid, y, marker)

    plt.tight_layout()

    suffix = "_bracket" if use_brackets else ""
    output_path = output_dir / f"sus_score{suffix}.svg"
    plt.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close()

    print(f"SUS score figure saved to: {output_path}")
    return output_path


def main():
    """Generate all paper figures."""
    print("=" * 60)
    print("GENERATING PAPER FIGURES")
    print("=" * 60)
    print()

    # Load data
    print("Loading survey data...")
    survey_df = load_survey_data()
    print(f"  Loaded {len(survey_df)} survey responses")

    print("Loading word search data...")
    word_search_rows = load_word_search_data()
    print(f"  Loaded {len(word_search_rows)} word search trials")
    print()

    # Run RM-ANOVA for word search time
    print("Running RM-ANOVA for word search time...")
    significant_pairs = run_word_search_anova()
    print()

    print("Running RM-ANOVA for search accuracy...")
    accuracy_significant_pairs = run_search_accuracy_anova(word_search_rows)
    print()

    print("Running Friedman test for preference order...")
    preference_significant_pairs = run_preference_friedman()
    print()

    print("Running RM-ANOVA for NASA-TLX...")
    nasa_tlx_significant = run_nasa_tlx_anova(survey_df)
    print()

    print("Running RM-ANOVA for SUS...")
    sus_significant_pairs = run_sus_anova(survey_df)
    print()

    # Generate figures - both with and without brackets
    print("Generating figures...")
    print()

    # With brackets
    print("\n--- With brackets ---")
    plot_nasa_tlx(
        survey_df, use_brackets=True, significance_by_dim=nasa_tlx_significant
    )
    plot_word_search_time(
        word_search_rows, use_brackets=True, significant_pairs=significant_pairs
    )
    plot_search_accuracy(
        word_search_rows,
        use_brackets=True,
        significant_pairs=accuracy_significant_pairs,
    )
    plot_sus_score(
        survey_df, use_brackets=True, significant_pairs=sus_significant_pairs
    )
    plot_preference_order(
        use_brackets=True,
        significant_pairs=preference_significant_pairs,
    )

    print()
    print("=" * 60)
    print("DONE - Figures saved to result/figures_final/")
    print("=" * 60)


if __name__ == "__main__":
    main()
