#!/usr/bin/env python3
"""Plot study target-word uniqueness summaries for videos 6-10."""

import json
import math
import os
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = Path("/tmp") / "morsequery-matplotlib"
XDG_CACHE_HOME = Path("/tmp") / "morsequery-cache"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

VIDEO_IDS = ["6", "7", "8", "9", "10"]
TARGET_DIR = ROOT_DIR / "data" / "study" / "target_words"
TRANSCRIPT_DIR = ROOT_DIR / "data" / "transcripts"
WORD_UNIQUENESS_ANALYSIS_PATH = ROOT_DIR / "data" / "analysis" / "word_uniqueness_analysis.json"
OUTPUT_DIR = ROOT_DIR / "result" / "data"
PLOT_PATH = OUTPUT_DIR / "target_word_overall_relative_uniqueness_6_10.png"
SUMMARY_PATH = OUTPUT_DIR / "target_word_overall_relative_uniqueness_6_10.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def mean_sd(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def compute_one_way_anova(groups: list[list[float]]) -> dict[str, float | int]:
    clean_groups = [np.asarray(group, dtype=float) for group in groups if len(group) > 0]
    group_count = len(clean_groups)
    total_count = sum(len(group) for group in clean_groups)
    grand_mean = sum(group.sum() for group in clean_groups) / total_count

    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in clean_groups)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in clean_groups)
    df_between = group_count - 1
    df_within = total_count - group_count
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within if ms_within else float("inf")

    scipy_f, scipy_p = stats.f_oneway(*clean_groups)

    return {
        "f_stat": float(f_stat),
        "p_value": float(scipy_p),
        "f_stat_scipy": float(scipy_f),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
    }


def normalize_word(raw_word: str) -> str | None:
    token = re.sub(r"[^a-z]", "", raw_word.lower())
    if len(token) <= 1:
        return None
    return token


def load_uniqueness_lookup() -> dict[str, dict[str, float]]:
    analysis_data = load_json(WORD_UNIQUENESS_ANALYSIS_PATH)
    lookup: dict[str, dict[str, float]] = {}
    for result in analysis_data.get("analysis_results", []):
        if not isinstance(result, dict):
            continue
        srt_file = result.get("srt_file")
        word_data = result.get("word_data")
        if not isinstance(srt_file, str) or not isinstance(word_data, list):
            continue
        stem_lookup: dict[str, float] = {}
        for item in word_data:
            if not isinstance(item, dict):
                continue
            word = item.get("word")
            uniqueness = item.get("uniqueness")
            if isinstance(word, str) and isinstance(uniqueness, (int, float)):
                stem_lookup[word.lower()] = float(uniqueness)
        lookup[Path(srt_file).stem] = stem_lookup
    return lookup


def collect_transcript_uniqueness_values(
    transcript_data: dict,
    *,
    audio_start: float,
    audio_end: float,
    uniqueness_lookup: dict[str, float],
) -> list[float]:
    values: list[float] = []
    for segment in transcript_data.get("segments", []):
        if not isinstance(segment, dict):
            continue
        for word_info in segment.get("words", []):
            if not isinstance(word_info, dict):
                continue
            start_time = word_info.get("start")
            raw_word = word_info.get("word")
            if not isinstance(start_time, (int, float)) or not isinstance(raw_word, str):
                continue
            if not (audio_start <= float(start_time) < audio_end):
                continue
            token = normalize_word(raw_word)
            if token is None:
                continue
            values.append(float(uniqueness_lookup.get(token, 7.0)))
    return values


def collect_video_stats(video_id: str, analysis_lookup: dict[str, dict[str, float]]) -> dict:
    target_data = load_json(TARGET_DIR / f"{video_id}.json")
    transcript_data = load_json(TRANSCRIPT_DIR / f"{video_id}.json")

    audio_start = float(target_data["audio_start_time"])
    audio_end = audio_start + float(target_data["target_window_seconds"])

    target_values = [
        float(item["target_word_uniqueness"])
        for item in target_data.get("interruptions", [])
        if isinstance(item.get("target_word_uniqueness"), (int, float))
    ]
    overall_values = collect_transcript_uniqueness_values(
        transcript_data,
        audio_start=audio_start,
        audio_end=audio_end,
        uniqueness_lookup=analysis_lookup.get(video_id, {}),
    )

    overall_mean, overall_sd = mean_sd(overall_values)
    if overall_sd == 0:
        relative_values = [0.0 for _ in target_values]
    else:
        relative_values = [(value - overall_mean) / overall_sd for value in target_values]

    target_mean, target_sd = mean_sd(target_values)
    relative_mean, relative_sd = mean_sd(relative_values)

    return {
        "video_id": video_id,
        "audio_window": {"start": audio_start, "end": audio_end},
        "target_uniqueness": {
            "values": target_values,
            "mean": target_mean,
            "sd": target_sd,
            "count": len(target_values),
        },
        "overall_uniqueness": {
            "values": overall_values,
            "mean": overall_mean,
            "sd": overall_sd,
            "count": len(overall_values),
        },
        "relative_uniqueness": {
            "definition": "(target_uniqueness - overall_mean) / overall_sd",
            "values": relative_values,
            "mean": relative_mean,
            "sd": relative_sd,
            "count": len(relative_values),
        },
    }


def plot_panel(ax, stats_by_video: list[dict], key: str, color: str, ylabel: str, title: str):
    x = np.arange(len(stats_by_video))
    means = [item[key]["mean"] for item in stats_by_video]
    counts = [item[key]["count"] for item in stats_by_video]
    labels = [item["video_id"] for item in stats_by_video]
    datasets = [item[key]["values"] for item in stats_by_video]

    boxplot = ax.boxplot(
        datasets,
        positions=x,
        widths=0.62,
        patch_artist=True,
        showfliers=False,
    )
    for box in boxplot["boxes"]:
        box.set_facecolor(color)
        box.set_edgecolor("black")
        box.set_alpha(0.5)
    for median in boxplot["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)
    for whisker in boxplot["whiskers"]:
        whisker.set_color("black")
    for cap in boxplot["caps"]:
        cap.set_color("black")

    for idx, item in enumerate(stats_by_video):
        values = item[key]["values"]
        jitter = np.linspace(-0.14, 0.14, num=len(values)) if values else np.array([])
        ax.scatter(np.full(len(values), x[idx]) + jitter, values, color="black", s=18, alpha=0.75, zorder=3)
        ax.scatter(x[idx], means[idx], color=color, edgecolor="black", s=52, zorder=4)
        y_max = max(values) if values else means[idx]
        ax.text(x[idx], y_max + 0.06, f"n={counts[idx]}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.22)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    analysis_lookup = load_uniqueness_lookup()
    stats_by_video = [collect_video_stats(video_id, analysis_lookup) for video_id in VIDEO_IDS]
    anova_results = {}
    for key in ("target_uniqueness", "overall_uniqueness", "relative_uniqueness"):
        groups = [item[key]["values"] for item in stats_by_video]
        anova_results[key] = compute_one_way_anova(groups)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    plot_panel(
        axes[0],
        stats_by_video,
        key="target_uniqueness",
        color="#D32F2F",
        ylabel="Uniqueness",
        title="Target Word Uniqueness",
    )
    axes[0].text(
        0.02,
        0.98,
        f"ANOVA F({anova_results['target_uniqueness']['df_between']}, {anova_results['target_uniqueness']['df_within']})="
        f"{anova_results['target_uniqueness']['f_stat']:.3f}\n"
        f"p={anova_results['target_uniqueness']['p_value']:.4f}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes[0].set_ylim(bottom=0)
    plot_panel(
        axes[1],
        stats_by_video,
        key="overall_uniqueness",
        color="#1E88E5",
        ylabel="Uniqueness",
        title="6-Min Transcript Overall Uniqueness",
    )
    axes[1].text(
        0.02,
        0.98,
        f"ANOVA F({anova_results['overall_uniqueness']['df_between']}, {anova_results['overall_uniqueness']['df_within']})="
        f"{anova_results['overall_uniqueness']['f_stat']:.3f}\n"
        f"p={anova_results['overall_uniqueness']['p_value']:.4f}",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes[1].set_ylim(bottom=0)
    plot_panel(
        axes[2],
        stats_by_video,
        key="relative_uniqueness",
        color="#6D4C41",
        ylabel="Relative uniqueness (z-score)",
        title="Relative Uniqueness",
    )
    axes[2].axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    axes[2].text(
        0.02,
        0.98,
        f"ANOVA F({anova_results['relative_uniqueness']['df_between']}, {anova_results['relative_uniqueness']['df_within']})="
        f"{anova_results['relative_uniqueness']['f_stat']:.3f}\n"
        f"p={anova_results['relative_uniqueness']['p_value']:.4f}",
        transform=axes[2].transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    axes[2].set_ylim(bottom=0)

    fig.suptitle("Study Target vs 6-Min Transcript Uniqueness (Videos 6-10)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(PLOT_PATH, dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "videos": stats_by_video,
        "anova": anova_results,
        "plot_path": str(PLOT_PATH),
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved plot: {PLOT_PATH}")
    print(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
