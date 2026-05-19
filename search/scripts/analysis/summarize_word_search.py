#!/usr/bin/env python3
"""Summarize word search outcomes from study logs.

Usage:
    python scripts/analysis/summarize_word_search.py           # All participants aggregated
    python scripts/analysis/summarize_word_search.py --id p0   # Only participant p0
    python scripts/analysis/summarize_word_search.py --id all  # Per-participant + aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path


LOGS_DIR = Path("logs/study")
RESULT_DIR = Path("result")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt
import pandas as pd

try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False


def extract_participant_id(log_base_name: str) -> str:
    """Extract participant ID from log base name (e.g., 'p0_keyword_7_20260505' -> 'p0')."""
    return log_base_name.split("_")[0]


def load_interruption_meta(participant_id: str | None = None):
    """Load interruption metadata, optionally filtered by participant ID."""
    meta = {}
    for path in sorted(LOGS_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                event = json.loads(line)
                audio = event.get("audio", "")
                if audio == "0.mp3" or audio.startswith("tutorial"):
                    continue
                if event.get("event") != "interruption":
                    continue
                log_base = event["logBaseName"]
                pid = extract_participant_id(log_base)
                if participant_id is not None and pid != participant_id:
                    continue
                meta[(log_base, event["taskIndex"])] = {
                    "participant_id": pid,
                    "condition": event["feature"],
                    "audio": event["audio"],
                    "word_search_type": event["delayType"],
                }
    return meta


def build_trial_rows(meta, participant_id: str | None = None):
    """Build trial rows, optionally filtered by participant ID."""
    rows = []
    for path in sorted(LOGS_DIR.glob("*.json")):
        with path.open() as f:
            session = json.load(f)
        audio = session.get("audio", "")
        if audio == "0.mp3" or audio.startswith("tutorial"):
            continue

        log_base = session["logBaseName"]
        pid = extract_participant_id(log_base)
        if participant_id is not None and pid != participant_id:
            continue

        for task in session.get("tasks", []):
            key = (log_base, task["taskIndex"])
            if key not in meta:
                continue
            info = meta[key]
            distance = task.get("distanceSeconds")
            success = (
                task.get("outcome") == "spacebar"
                and distance is not None
                and abs(distance) <= 2
            )
            failure = task.get("outcome") == "timeout" or (
                distance is not None and abs(distance) > 2
            )
            rows.append(
                {
                    "participant_id": info["participant_id"],
                    "condition": info["condition"],
                    "audio": info["audio"],
                    "task_index": task["taskIndex"],
                    "word_search_type": info["word_search_type"],
                    "outcome": task.get("outcome"),
                    "distance_seconds": distance,
                    "word_search_success": success,
                    "word_search_failure": failure,
                    "word_search_time_sec": (
                        "" if task.get("responseTimeMs") is None else task["responseTimeMs"] / 1000
                    ),
                }
            )
    return rows


def write_trial_csv(rows, suffix: str = ""):
    """Write trial-level CSV with optional suffix for filtering."""
    csv_path = RESULT_DIR / f"word_search_trials{suffix}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "participant_id",
                "condition",
                "audio",
                "task_index",
                "word_search_type",
                "outcome",
                "distance_seconds",
                "word_search_success",
                "word_search_failure",
                "word_search_time_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def compute_summary_stats(group: list[dict]) -> dict:
    """Compute summary statistics for a group of trial rows."""
    n_trials = len(group)
    n_success = sum(1 for row in group if row["word_search_success"])
    n_failure = sum(1 for row in group if row["word_search_failure"])
    times = [
        row["word_search_time_sec"]
        for row in group
        if row["word_search_time_sec"] != ""
    ]
    mean_time = "" if not times else sum(times) / len(times)
    return {
        "n_trials": n_trials,
        "n_success": n_success,
        "n_failure": n_failure,
        "success_rate": n_success / n_trials if n_trials else "",
        "mean_word_search_time_sec": mean_time,
    }


def write_summary_csv(rows, suffix: str = "", by_participant: bool = False):
    """Write summary CSV, optionally grouped by participant."""
    csv_path = RESULT_DIR / f"word_search_summary{suffix}.csv"

    if by_participant:
        # Group by (participant_id, condition, word_search_type)
        grouped = defaultdict(list)
        for row in rows:
            pid = row["participant_id"]
            grouped[(pid, row["condition"], row["word_search_type"])].append(row)
            grouped[(pid, row["condition"], "all")].append(row)

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "participant_id",
                "condition",
                "word_search_type",
                "n_trials",
                "n_success",
                "n_failure",
                "success_rate",
                "mean_word_search_time_sec",
            ])
            for (pid, condition, word_search_type) in sorted(grouped):
                group = grouped[(pid, condition, word_search_type)]
                stats = compute_summary_stats(group)
                writer.writerow([
                    pid,
                    condition,
                    word_search_type,
                    stats["n_trials"],
                    stats["n_success"],
                    stats["n_failure"],
                    stats["success_rate"],
                    stats["mean_word_search_time_sec"],
                ])
    else:
        # Group by (condition, word_search_type) only
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["condition"], row["word_search_type"])].append(row)
            grouped[(row["condition"], "all")].append(row)

        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "condition",
                "word_search_type",
                "n_trials",
                "n_success",
                "n_failure",
                "success_rate",
                "mean_word_search_time_sec",
            ])
            for (condition, word_search_type) in sorted(grouped):
                group = grouped[(condition, word_search_type)]
                stats = compute_summary_stats(group)
                writer.writerow([
                    condition,
                    word_search_type,
                    stats["n_trials"],
                    stats["n_success"],
                    stats["n_failure"],
                    stats["success_rate"],
                    stats["mean_word_search_time_sec"],
                ])
    return csv_path


def run_repeated_measures_anova(rows, include_failures: bool = True) -> dict:
    """Run repeated measures ANOVA on word_search_time_sec by condition.

    Args:
        rows: Trial rows
        include_failures: If True, treat failed cases as 60 seconds

    Returns:
        Dictionary with ANOVA results for overall and by word_search_type
    """
    # Build DataFrame with participant-level means
    participant_data = defaultdict(lambda: defaultdict(list))

    for row in rows:
        pid = row["participant_id"]
        condition = row["condition"]
        word_type = row["word_search_type"]

        if row["word_search_success"]:
            if row["word_search_time_sec"] == "":
                continue
            time_sec = row["word_search_time_sec"]
        elif include_failures and row["word_search_failure"]:
            time_sec = 60.0
        else:
            continue

        participant_data[pid][(condition, word_type)].append(time_sec)
        participant_data[pid][(condition, "all")].append(time_sec)

    results = {}

    # Helper to build RM-ANOVA DataFrame for a given word_type filter
    def build_rm_df(word_type_filter: str | None):
        """Build DataFrame for repeated measures ANOVA."""
        records = []
        for pid, cond_data in participant_data.items():
            for (condition, wtype), times in cond_data.items():
                if word_type_filter is not None and wtype != word_type_filter:
                    continue
                mean_time = sum(times) / len(times)
                records.append({
                    "participant_id": pid,
                    "condition": condition,
                    "word_search_type": wtype,
                    "mean_time": mean_time,
                })
        return pd.DataFrame(records)

    def extract_anova_results(aov_df):
        """Extract results from pingouin ANOVA DataFrame."""
        # pingouin uses 'p_unc' for uncorrected p-value, 'ng2' for generalized eta-squared
        # DF is in a single column, with df1 in row 0 and df2 in row 1
        return {
            "F": aov_df.loc[0, "F"],
            "p": aov_df.loc[0, "p_unc"] if "p_unc" in aov_df.columns else aov_df.loc[0, "p-unc"],
            "df1": int(aov_df.loc[0, "DF"]),
            "df2": int(aov_df.loc[1, "DF"]) if len(aov_df) > 1 else 0,
            "eta_sq": aov_df.loc[0, "ng2"] if "ng2" in aov_df.columns else (
                aov_df.loc[0, "np2"] if "np2" in aov_df.columns else None
            ),
            "table": aov_df,
        }

    # 1) Overall ANOVA (ignoring long/short)
    df_all = build_rm_df("all")
    if HAS_PINGOUIN and len(df_all) > 0:
        try:
            aov_all = pg.rm_anova(
                data=df_all,
                dv="mean_time",
                within="condition",
                subject="participant_id",
                detailed=True,
            )
            results["overall"] = extract_anova_results(aov_all)
        except Exception as e:
            results["overall"] = {"error": str(e)}
    else:
        results["overall"] = {"error": "pingouin not available or no data"}

    # 2) ANOVA by word_search_type (long, short)
    for wtype in ["long", "short"]:
        df_type = build_rm_df(wtype)
        if HAS_PINGOUIN and len(df_type) > 0:
            try:
                aov_type = pg.rm_anova(
                    data=df_type,
                    dv="mean_time",
                    within="condition",
                    subject="participant_id",
                    detailed=True,
                )
                results[wtype] = extract_anova_results(aov_type)
            except Exception as e:
                results[wtype] = {"error": str(e)}
        else:
            results[wtype] = {"error": "pingouin not available or no data"}

    return results


def format_anova_results(anova_results: dict) -> str:
    """Format ANOVA results as a string for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("REPEATED MEASURES ANOVA RESULTS")
    lines.append("(failed trials treated as 60 sec)")
    lines.append("=" * 60)

    def format_single_result(r):
        """Format a single ANOVA result."""
        eta_str = f", η²g = {r['eta_sq']:.3f}" if r.get("eta_sq") is not None else ""
        sig = "***" if r["p"] < 0.001 else "**" if r["p"] < 0.01 else "*" if r["p"] < 0.05 else " (n.s.)"
        return f"F({r['df1']}, {r['df2']}) = {r['F']:.3f}, p = {r['p']:.4f}{sig}{eta_str}"

    # Overall
    lines.append("\n1) OVERALL (ignoring long/short):")
    lines.append("-" * 40)
    if "error" in anova_results.get("overall", {}):
        lines.append(f"   Error: {anova_results['overall']['error']}")
    else:
        lines.append(f"   {format_single_result(anova_results['overall'])}")

    # By word search type
    lines.append("\n2) BY WORD SEARCH TYPE (long/short):")
    lines.append("-" * 40)
    for wtype in ["long", "short"]:
        if "error" in anova_results.get(wtype, {}):
            lines.append(f"   [{wtype}] Error: {anova_results[wtype]['error']}")
        else:
            lines.append(f"   [{wtype}] {format_single_result(anova_results[wtype])}")

    lines.append("=" * 60)
    return "\n".join(lines)


def plot_search_times(rows, suffix: str = "", include_failures: bool = True, title: str = None):
    """Plot search times by condition and word search type.

    Args:
        rows: Trial rows to plot
        suffix: Filename suffix
        include_failures: If True, treat failed cases as 60 seconds
        title: Figure title (e.g., "(P1) Search Time")
    """
    plot_path = RESULT_DIR / f"word_search_time_by_condition{suffix}.png"
    condition_order = ["discontinuous", "keyword", "keyword2", "sentence", "word"]
    type_order = ["all", "long", "short"]

    grouped = defaultdict(list)
    for row in rows:
        if row["word_search_success"]:
            # Successful case: use actual time
            if row["word_search_time_sec"] == "":
                continue
            time_sec = row["word_search_time_sec"]
        elif include_failures and row["word_search_failure"]:
            # Failed case: use 60 seconds
            time_sec = 60.0
        else:
            continue
        grouped[(row["condition"], row["word_search_type"])].append(time_sec)
        grouped[(row["condition"], "all")].append(time_sec)

    # Run ANOVA and print results
    anova_results = run_repeated_measures_anova(rows, include_failures=include_failures)
    print(format_anova_results(anova_results))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, word_type in zip(axes, type_order):
        xs = []
        ys = []
        cis = []
        for condition in condition_order:
            times = grouped.get((condition, word_type), [])
            if not times:
                continue
            xs.append(condition)
            mean_val = sum(times) / len(times)
            ys.append(mean_val)
            # Calculate standard error of the mean
            if len(times) > 1:
                sem = pd.Series(times).sem()
            else:
                sem = 0
            cis.append(sem)

        ax.bar(xs, ys, color="#4C78A8", yerr=cis, capsize=4, error_kw={"elinewidth": 1.5, "capthick": 1.5})
        ax.set_title(word_type)
        ax.set_xlabel("condition")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("time (sec)")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def get_all_participant_ids() -> list[str]:
    """Get all unique participant IDs from log files."""
    pids = set()
    for path in LOGS_DIR.glob("*.json"):
        pid = extract_participant_id(path.stem)
        pids.add(pid)
    return sorted(pids)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize word search outcomes from study logs."
    )
    parser.add_argument(
        "--id",
        type=str,
        default=None,
        help="Participant ID to filter (e.g., 'p0'), or 'all' for per-participant breakdown + aggregate",
    )
    args = parser.parse_args()

    if args.id == "all":
        # Per-participant summaries + aggregate
        all_rows = []
        meta = load_interruption_meta()
        all_rows = build_trial_rows(meta)

        # Write per-participant breakdown
        trial_path = write_trial_csv(all_rows, suffix="_by_id")
        summary_path = write_summary_csv(all_rows, suffix="_by_id", by_participant=True)
        plot_path = plot_search_times(all_rows, suffix="_aggregate", title="(All) Search Time")

        # Also write aggregate summary (without participant grouping)
        agg_summary_path = write_summary_csv(all_rows, suffix="_aggregate", by_participant=False)

        print(f"Per-participant trials: {trial_path}")
        print(f"Per-participant summary: {summary_path}")
        print(f"Aggregate summary: {agg_summary_path}")
        print(f"Plot: {plot_path}")

    elif args.id is not None:
        # Single participant
        suffix = f"_{args.id}"
        meta = load_interruption_meta(participant_id=args.id)
        rows = build_trial_rows(meta, participant_id=args.id)

        if not rows:
            print(f"No data found for participant '{args.id}'")
            print(f"Available IDs: {', '.join(get_all_participant_ids())}")
            return

        trial_path = write_trial_csv(rows, suffix=suffix)
        summary_path = write_summary_csv(rows, suffix=suffix, by_participant=False)
        plot_path = plot_search_times(rows, suffix=suffix, title=f"({args.id.upper()}) Search Time")

        print(f"Participant: {args.id}")
        print(f"Trials: {trial_path}")
        print(f"Summary: {summary_path}")
        print(f"Plot: {plot_path}")

    else:
        # Default: aggregate all participants (no per-participant breakdown)
        meta = load_interruption_meta()
        rows = build_trial_rows(meta)

        trial_path = write_trial_csv(rows)
        summary_path = write_summary_csv(rows)
        plot_path = plot_search_times(rows, title="(All) Search Time")

        print(f"Trials: {trial_path}")
        print(f"Summary: {summary_path}")
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
