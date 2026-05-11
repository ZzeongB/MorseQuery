#!/usr/bin/env python3
"""Plot semantic, jargon, and target words across time for selected videos."""

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

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


VIDEO_CONFIGS = [
    {
        "video_id": "0",
        "semantic_path": Path("data/semantic_words/0_ts_edit.json"),
        "jargon_path": None,
        "target_path": Path("data/study/target_words/0_tutorial_full.json"),
    },
    {
        "video_id": "6",
        "semantic_path": Path("data/semantic_words/6.json"),
        "jargon_path": Path("data/semantic_words/6.zero.jargon.json"),
        "target_path": Path("data/study/target_words/6.json"),
    },
    {
        "video_id": "7",
        "semantic_path": Path("data/semantic_words/7.json"),
        "jargon_path": Path("data/semantic_words/7.zero.jargon.json"),
        "target_path": Path("data/study/target_words/7.json"),
    },
    {
        "video_id": "8",
        "semantic_path": Path("data/semantic_words/8.json"),
        "jargon_path": Path("data/semantic_words/8.zero.jargon.json"),
        "target_path": Path("data/study/target_words/8.json"),
    },
    {
        "video_id": "9",
        "semantic_path": Path("data/semantic_words/9.json"),
        "jargon_path": Path("data/semantic_words/9.zero.jargon.json"),
        "target_path": Path("data/study/target_words/9.json"),
    },
    {
        "video_id": "10",
        "semantic_path": Path("data/semantic_words/10.json"),
        "jargon_path": Path("data/semantic_words/10.zero.jargon.json"),
        "target_path": Path("data/study/target_words/10.json"),
    },
]


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_word_times(path: Path | None):
    if path is None or not path.exists():
        return []
    data = load_json(path)
    items = []
    for item in data:
        word = item.get("word")
        time = item.get("time")
        if isinstance(word, str) and isinstance(time, (int, float)):
            items.append((word, float(time)))
    return items


def load_target_words(path: Path):
    if not path.exists():
        return []
    data = load_json(path)
    items = []
    for item in data.get("interruptions", []):
        word = item.get("target_word")
        time = item.get("target_word_time")
        if isinstance(word, str) and isinstance(time, (int, float)):
            items.append((word, float(time)))
    return items


def draw_word_rectangles(ax, items, y, color, height, width, alpha, zorder):
    if not items:
        return
    ax.broken_barh(
        [(time - width / 2, width) for _, time in items],
        (y - height / 2, height),
        facecolors=color,
        edgecolors="none",
        alpha=alpha,
        zorder=zorder,
    )


def main():
    output_path = ROOT_DIR / "result" / "data" / "jargon_word_distribution.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    semantic_color = "#111111"
    jargon_color = "#1E88E5"
    target_color = "#D32F2F"
    long_color = "#4CAF50"
    short_color = "#FF9800"
    search_color = "#B71C1C"

    rect_width = 0.5
    rect_height = 0.12

    fig, ax = plt.subplots(figsize=(14, 8.5))

    y_positions = {
        config["video_id"]: len(VIDEO_CONFIGS) - index
        for index, config in enumerate(VIDEO_CONFIGS)
    }

    for config in VIDEO_CONFIGS:
        video_id = config["video_id"]
        y = y_positions[video_id]

        semantic_items = load_word_times(config["semantic_path"])
        jargon_items = load_word_times(config["jargon_path"])
        target_items = load_target_words(config["target_path"])

        draw_word_rectangles(
            ax,
            semantic_items,
            y + 0.12,
            semantic_color,
            rect_height,
            rect_width,
            0.9,
            3,
        )
        draw_word_rectangles(
            ax,
            jargon_items,
            y - 0.02,
            jargon_color,
            rect_height,
            rect_width,
            0.9,
            4,
        )
        draw_word_rectangles(
            ax,
            target_items,
            y - 0.16,
            target_color,
            rect_height,
            rect_width * 1.15,
            0.95,
            5,
        )

        for word, time in target_items:
            ax.annotate(
                word,
                (time, y - 0.28),
                fontsize=6,
                ha="center",
                va="top",
                rotation=45,
                color=target_color,
            )

        for minute in range(1, 7):
            search_time = minute * 60
            long_start = search_time - 60
            long_end = search_time - 40
            short_start = search_time - 20
            short_end = search_time

            ax.barh(
                y + 0.27,
                long_end - long_start,
                left=long_start,
                height=0.08,
                color=long_color,
                alpha=0.28,
                zorder=1,
            )
            ax.barh(
                y + 0.37,
                short_end - short_start,
                left=short_start,
                height=0.08,
                color=short_color,
                alpha=0.28,
                zorder=1,
            )
            ax.axvline(
                x=search_time,
                color=search_color,
                linestyle="--",
                alpha=0.25,
                linewidth=0.6,
                zorder=0,
            )

    ax.set_xlim(-5, 370)
    ax.set_ylim(0.3, len(VIDEO_CONFIGS) + 0.75)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Video {config['video_id']}" for config in VIDEO_CONFIGS])
    ax.set_xlabel("Time (seconds)")
    ax.set_title(
        "Semantic / Jargon / Target Word Distribution\n"
        "(Long delay: 40-60s, Short delay: 0-20s)"
    )

    for minute in range(7):
        ax.axvline(x=minute * 60, color="gray", linestyle="-", alpha=0.18, linewidth=1)
        if minute > 0:
            ax.text(
                minute * 60,
                len(VIDEO_CONFIGS) + 0.58,
                f"{minute}min",
                ha="center",
                fontsize=8,
                color="gray",
            )

    legend_elements = [
        mpatches.Patch(color=semantic_color, alpha=0.9, label="Semantic words"),
        mpatches.Patch(color=jargon_color, alpha=0.9, label="Jargon words"),
        mpatches.Patch(color=target_color, alpha=0.95, label="Target words"),
        mpatches.Patch(color=long_color, alpha=0.28, label="Long delay range (40-60s)"),
        mpatches.Patch(
            color=short_color, alpha=0.28, label="Short delay range (0-20s)"
        ),
        plt.Line2D(
            [0], [0], color=search_color, linestyle="--", alpha=0.5, label="Search time"
        ),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
