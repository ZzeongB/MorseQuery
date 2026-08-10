#!/usr/bin/env python3
"""
RM-ANOVA analysis for word_search_trials.csv

Analyzes:
1. Main effect of condition (temporal, keyword, keyword2, sentence, word)
2. Main effect of word_search_type (short, long)
3. Interaction effect between condition and word_search_type
4. Per-participant breakdown
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Try to import statistical packages
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


def load_data():
    """Load and preprocess the data."""
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / "result" / "word_search_trials.csv"

    df = pd.read_csv(data_path)

    # For failed trials, set word_search_time_sec to 60 seconds
    df["word_search_time_sec"] = df.apply(
        lambda row: 60.0 if row["word_search_failure"] else row["word_search_time_sec"],
        axis=1,
    )
    df_valid = df.copy()

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"Total trials: {len(df)}")
    print(f"Failed trials (set to 60s): {df['word_search_failure'].sum()}")
    print(f"Participants: {df_valid['participant_id'].nunique()}")
    print(f"Conditions: {df_valid['condition'].unique().tolist()}")
    print(f"Word search types: {df_valid['word_search_type'].unique().tolist()}")
    print()

    return df_valid


def descriptive_stats(df_valid):
    """Print descriptive statistics."""
    print("=" * 60)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 60)

    # By condition
    print("\n--- By Condition ---")
    cond_stats = df_valid.groupby("condition")["word_search_time_sec"].agg(
        ["mean", "std", "count"]
    )
    cond_stats.columns = ["Mean", "SD", "N"]
    print(cond_stats.round(3))

    # By word_search_type
    print("\n--- By Word Search Type ---")
    type_stats = df_valid.groupby("word_search_type")["word_search_time_sec"].agg(
        ["mean", "std", "count"]
    )
    type_stats.columns = ["Mean", "SD", "N"]
    print(type_stats.round(3))

    # By condition x word_search_type
    print("\n--- By Condition x Word Search Type ---")
    cross_stats = df_valid.groupby(["condition", "word_search_type"])[
        "word_search_time_sec"
    ].agg(["mean", "std", "count"])
    cross_stats.columns = ["Mean", "SD", "N"]
    print(cross_stats.round(3))

    # By participant
    print("\n--- By Participant ---")
    part_stats = df_valid.groupby("participant_id")["word_search_time_sec"].agg(
        ["mean", "std", "count"]
    )
    part_stats.columns = ["Mean", "SD", "N"]
    print(part_stats.round(3))
    print()


def run_rm_anova_pingouin(df_valid):
    """Run RM-ANOVA using pingouin."""
    if not HAS_PINGOUIN:
        print("Skipping pingouin analysis (not installed)")
        return

    print("=" * 60)
    print("REPEATED MEASURES ANOVA (pingouin)")
    print("=" * 60)

    # For RM-ANOVA, we need aggregated data per participant x condition x type
    # Aggregate to get mean per participant per condition per type
    df_agg = (
        df_valid.groupby(["participant_id", "condition", "word_search_type"])[
            "word_search_time_sec"
        ]
        .mean()
        .reset_index()
    )

    # Check for missing cells
    print("\n--- Data Balance Check ---")
    pivot_check = df_agg.pivot_table(
        index="participant_id",
        columns=["condition", "word_search_type"],
        values="word_search_time_sec",
        aggfunc="count",
    )
    print("Cells per participant (should all be 1 for balanced design):")
    print(pivot_check)
    print()

    # RM-ANOVA: Main effect of condition
    print("\n--- RM-ANOVA: Main Effect of Condition ---")
    try:
        aov_condition = pg.rm_anova(
            data=df_agg.groupby(["participant_id", "condition"])["word_search_time_sec"]
            .mean()
            .reset_index(),
            dv="word_search_time_sec",
            within="condition",
            subject="participant_id",
            detailed=True,
        )
        print(aov_condition.to_string())

        # Effect size interpretation
        if "np2" in aov_condition.columns:
            eta2 = aov_condition["np2"].values[0]
            print(f"\nPartial eta-squared: {eta2:.4f}")
            if eta2 < 0.01:
                print("Effect size: negligible")
            elif eta2 < 0.06:
                print("Effect size: small")
            elif eta2 < 0.14:
                print("Effect size: medium")
            else:
                print("Effect size: large")
    except Exception as e:
        print(f"Error: {e}")

    # RM-ANOVA: Main effect of word_search_type
    print("\n--- RM-ANOVA: Main Effect of Word Search Type ---")
    try:
        aov_type = pg.rm_anova(
            data=df_agg.groupby(["participant_id", "word_search_type"])[
                "word_search_time_sec"
            ]
            .mean()
            .reset_index(),
            dv="word_search_time_sec",
            within="word_search_type",
            subject="participant_id",
            detailed=True,
        )
        print(aov_type.to_string())

        if "np2" in aov_type.columns:
            eta2 = aov_type["np2"].values[0]
            print(f"\nPartial eta-squared: {eta2:.4f}")
            if eta2 < 0.01:
                print("Effect size: negligible")
            elif eta2 < 0.06:
                print("Effect size: small")
            elif eta2 < 0.14:
                print("Effect size: medium")
            else:
                print("Effect size: large")
    except Exception as e:
        print(f"Error: {e}")

    # Two-way RM-ANOVA: condition x word_search_type
    print("\n--- Two-Way RM-ANOVA: Condition x Word Search Type ---")
    try:
        aov_2way = pg.rm_anova(
            data=df_agg,
            dv="word_search_time_sec",
            within=["condition", "word_search_type"],
            subject="participant_id",
            detailed=True,
        )
        print(aov_2way.to_string())
        print()
    except Exception as e:
        print(f"Error: {e}")

    # Post-hoc tests for condition
    print("\n--- Post-hoc Pairwise Comparisons: Condition ---")
    try:
        posthoc_cond = pg.pairwise_tests(
            data=df_agg.groupby(["participant_id", "condition"])["word_search_time_sec"]
            .mean()
            .reset_index(),
            dv="word_search_time_sec",
            within="condition",
            subject="participant_id",
            padjust="bonferroni",
        )
        print(posthoc_cond.to_string())
    except Exception as e:
        print(f"Error: {e}")

    # Post-hoc tests for word_search_type
    print("\n--- Post-hoc Pairwise Comparisons: Word Search Type ---")
    try:
        posthoc_type = pg.pairwise_tests(
            data=df_agg.groupby(["participant_id", "word_search_type"])[
                "word_search_time_sec"
            ]
            .mean()
            .reset_index(),
            dv="word_search_time_sec",
            within="word_search_type",
            subject="participant_id",
            padjust="bonferroni",
        )
        print(posthoc_type.to_string())
    except Exception as e:
        print(f"Error: {e}")
    print()


def run_participant_analysis(df_valid):
    """Analyze per-participant effects."""
    if not HAS_SCIPY:
        print("Skipping participant analysis (scipy not installed)")
        return

    print("=" * 60)
    print("PER-PARTICIPANT ANALYSIS")
    print("=" * 60)

    participants = df_valid["participant_id"].unique()

    for pid in sorted(participants):
        print(f"\n--- Participant: {pid} ---")
        df_p = df_valid[df_valid["participant_id"] == pid]

        # Condition effect (one-way ANOVA for this participant)
        conditions = df_p["condition"].unique()
        groups = [
            df_p[df_p["condition"] == c]["word_search_time_sec"].values
            for c in conditions
        ]

        if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
            f_stat, p_val = stats.f_oneway(*groups)
            print(f"  Condition effect: F = {f_stat:.3f}, p = {p_val:.4f}", end="")
            print(" *" if p_val < 0.05 else "")

        # Word search type effect (t-test for this participant)
        short_times = df_p[df_p["word_search_type"] == "short"][
            "word_search_time_sec"
        ].values
        long_times = df_p[df_p["word_search_type"] == "long"][
            "word_search_time_sec"
        ].values

        if len(short_times) >= 2 and len(long_times) >= 2:
            t_stat, p_val = stats.ttest_ind(short_times, long_times)
            print(
                f"  Word search type effect: t = {t_stat:.3f}, p = {p_val:.4f}", end=""
            )
            print(" *" if p_val < 0.05 else "")
            print(
                f"    Short: M = {np.mean(short_times):.2f}, SD = {np.std(short_times):.2f}"
            )
            print(
                f"    Long:  M = {np.mean(long_times):.2f}, SD = {np.std(long_times):.2f}"
            )
    print()


def run_mixed_effects_analysis(df_valid):
    """Run mixed effects analysis if statsmodels is available."""
    try:
        import statsmodels.formula.api as smf
        from statsmodels.stats.anova import AnovaRM
    except ImportError:
        print("Skipping mixed effects analysis (statsmodels not installed)")
        return

    print("=" * 60)
    print("MIXED EFFECTS / ALTERNATIVE ANALYSIS (statsmodels)")
    print("=" * 60)

    # Aggregate data
    df_agg = (
        df_valid.groupby(["participant_id", "condition", "word_search_type"])[
            "word_search_time_sec"
        ]
        .mean()
        .reset_index()
    )

    # Linear Mixed Effects Model
    print("\n--- Linear Mixed Effects Model ---")
    print("DV: word_search_time_sec")
    print("Fixed effects: condition, word_search_type")
    print("Random effects: participant_id")
    print()

    try:
        # Create dummy variables for formula
        model = smf.mixedlm(
            "word_search_time_sec ~ C(condition) + C(word_search_type)",
            df_agg,
            groups=df_agg["participant_id"],
        )
        result = model.fit()
        print(result.summary())
    except Exception as e:
        print(f"Mixed model error: {e}")

    # statsmodels RM-ANOVA (alternative)
    print("\n--- statsmodels AnovaRM ---")
    try:
        # Need complete cases - aggregate to have one observation per cell
        df_agg2 = (
            df_valid.groupby(["participant_id", "condition"])["word_search_time_sec"]
            .mean()
            .reset_index()
        )

        aovrm = AnovaRM(
            df_agg2, "word_search_time_sec", "participant_id", within=["condition"]
        )
        result = aovrm.fit()
        print(result)
    except Exception as e:
        print(f"AnovaRM error: {e}")
    print()


def sphericity_test(df_valid):
    """Test sphericity assumption for RM-ANOVA."""
    if not HAS_PINGOUIN:
        return

    print("=" * 60)
    print("SPHERICITY TEST (Mauchly's Test)")
    print("=" * 60)

    df_agg = (
        df_valid.groupby(["participant_id", "condition"])["word_search_time_sec"]
        .mean()
        .reset_index()
    )

    # Pivot for sphericity test
    df_wide = df_agg.pivot(
        index="participant_id", columns="condition", values="word_search_time_sec"
    )

    try:
        _, W, chi2, dof, pval = pg.sphericity(df_wide)
        print(f"Mauchly's W = {W:.4f}")
        print(f"Chi-square = {chi2:.4f}")
        print(f"df = {dof}")
        print(f"p-value = {pval:.4f}")
        if pval < 0.05:
            print("\nSphericity assumption VIOLATED (p < 0.05)")
            print("Consider using Greenhouse-Geisser or Huynh-Feldt correction")
        else:
            print("\nSphericity assumption MET (p >= 0.05)")
    except Exception as e:
        print(f"Sphericity test error: {e}")
    print()


def main():
    """Main analysis pipeline."""
    print("\n" + "=" * 60)
    print("WORD SEARCH TRIALS - RM-ANOVA ANALYSIS")
    print("=" * 60 + "\n")

    # Load data
    df_valid = load_data()

    # Descriptive statistics
    descriptive_stats(df_valid)

    # Sphericity test
    sphericity_test(df_valid)

    # Main RM-ANOVA analyses
    run_rm_anova_pingouin(df_valid)

    # Per-participant analysis
    run_participant_analysis(df_valid)

    # Mixed effects (alternative)
    run_mixed_effects_analysis(df_valid)

    print("=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
