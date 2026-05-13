#!/usr/bin/env python3
"""Compare semantic-word time-gap distributions for videos 6-10."""

from __future__ import annotations

import json
import os
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
SEMANTIC_DIR = ROOT_DIR / "data" / "semantic_words"
OUTPUT_DIR = ROOT_DIR / "result" / "data"
PLOT_PATH = OUTPUT_DIR / "semantic_word_gap_comparison_6_10.png"
SUMMARY_PATH = OUTPUT_DIR / "semantic_word_gap_comparison_6_10.json"


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_time_gaps(path: Path) -> list[float]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON: {path}")

    times = sorted(
        float(item["time"])
        for item in data
        if isinstance(item, dict) and isinstance(item.get("time"), (int, float))
    )
    return [curr - prev for prev, curr in zip(times, times[1:]) if curr >= prev]


def compute_anova(groups: list[list[float]]) -> dict[str, float | int]:
    arrays = [np.asarray(group, dtype=float) for group in groups]
    scipy_f, scipy_p = stats.f_oneway(*arrays)

    total_count = sum(len(group) for group in arrays)
    group_count = len(arrays)
    grand_mean = sum(group.sum() for group in arrays) / total_count
    ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in arrays)
    ss_within = sum(((group - group.mean()) ** 2).sum() for group in arrays)
    df_between = group_count - 1
    df_within = total_count - group_count

    return {
        "f_stat": float(scipy_f),
        "p_value": float(scipy_p),
        "df_between": int(df_between),
        "df_within": int(df_within),
        "ss_between": float(ss_between),
        "ss_within": float(ss_within),
    }


def summarize(values: list[float]) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
    }


def plot_groups(groups: dict[str, list[float]], *, anova: dict[str, float | int]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    labels = list(groups.keys())
    series = [groups[label] for label in labels]
    positions = np.arange(1, len(labels) + 1)

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), height_ratios=[2.3, 1.2])

    ax = axes[0]
    violin = ax.violinplot(
        series,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        widths=0.85,
    )
    for body in violin["bodies"]:
        body.set_facecolor("#7db7c9")
        body.set_edgecolor("#2b5d6e")
        body.set_alpha(0.55)

    box = ax.boxplot(
        series,
        positions=positions,
        widths=0.28,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#f6e7b0")
        patch.set_edgecolor("#6f5b2a")
    for key in ("whiskers", "caps", "medians"):
        for artist in box[key]:
            artist.set_color("#6f5b2a")

    means = [np.mean(values) for values in series]
    ax.scatter(positions, means, color="#c9492c", s=45, zorder=3, label="mean")

    ax.set_title("Semantic Word Gap Comparison (IDs 6-10)")
    ax.set_ylabel("Gap between adjacent semantic words (seconds)")
    ax.set_xticks(positions, labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.text(
        0.015,
        0.98,
        (
            f"ANOVA F({anova['df_between']}, {anova['df_within']})="
            f"{anova['f_stat']:.3f}, p={anova['p_value']:.4g}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    hist_ax = axes[1]
    bins = np.linspace(2.0, max(max(values) for values in series), 20)
    for label, values in groups.items():
        hist_ax.hist(
            values,
            bins=bins,
            histtype="step",
            linewidth=2,
            label=label,
            density=True,
        )
    hist_ax.set_title("Gap Density Overlay")
    hist_ax.set_xlabel("Gap (seconds)")
    hist_ax.set_ylabel("Density")
    hist_ax.grid(alpha=0.25)
    hist_ax.legend(ncol=len(labels), loc="upper right")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    groups: dict[str, list[float]] = {}
    summaries: dict[str, dict[str, float | int]] = {}
    for video_id in VIDEO_IDS:
        path = SEMANTIC_DIR / f"{video_id}.json"
        gaps = load_time_gaps(path)
        groups[video_id] = gaps
        summaries[video_id] = summarize(gaps)

    group_values = [groups[video_id] for video_id in VIDEO_IDS]
    anova = compute_anova(group_values)
    levene_stat, levene_p = stats.levene(*group_values)
    kruskal_stat, kruskal_p = stats.kruskal(*group_values)

    plot_groups(groups, anova=anova)

    payload = {
        "video_ids": VIDEO_IDS,
        "summaries": summaries,
        "anova": anova,
        "levene": {
            "stat": float(levene_stat),
            "p_value": float(levene_p),
        },
        "kruskal": {
            "stat": float(kruskal_stat),
            "p_value": float(kruskal_p),
        },
        "plot_path": str(PLOT_PATH),
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote plot: {PLOT_PATH}")
    print(f"Wrote summary: {SUMMARY_PATH}")
    print(
        f"ANOVA F({anova['df_between']}, {anova['df_within']})="
        f"{anova['f_stat']:.4f}, p={anova['p_value']:.6f}"
    )


if __name__ == "__main__":
    main()
