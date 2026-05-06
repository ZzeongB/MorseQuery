#!/usr/bin/env python3
"""Plot jargon word distribution across time for each video."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def main():
    # Configuration
    video_ids = ["6", "7", "8", "9", "10"]
    jargon_dir = Path("data/semantic_words")
    output_path = Path("result/jargon_word_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delay parameters
    short_mean = 10
    long_mean = 50
    tolerance = 10
    short_min, short_max = short_mean - tolerance, short_mean + tolerance  # 0-20
    long_min, long_max = long_mean - tolerance, long_mean + tolerance  # 40-60

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Colors
    word_color = "#2196F3"
    long_color = "#4CAF50"
    short_color = "#FF9800"
    search_color = "#F44336"

    y_positions = {}
    for i, video_id in enumerate(video_ids):
        y_positions[video_id] = len(video_ids) - i

    # Plot each video
    for video_id in video_ids:
        y = y_positions[video_id]
        jargon_path = jargon_dir / f"{video_id}.zero.jargon.json"

        if not jargon_path.exists():
            print(f"Warning: {jargon_path} not found")
            continue

        jargon_data = load_json(jargon_path)
        times = [item["time"] for item in jargon_data]
        words = [item["word"] for item in jargon_data]

        # Plot word positions
        ax.scatter(times, [y] * len(times), c=word_color, s=50, zorder=5, alpha=0.8)

        # Add word labels for some words (to avoid clutter, show every 3rd word)
        for j, (t, w) in enumerate(zip(times, words)):
            if j % 3 == 0:
                ax.annotate(w, (t, y + 0.15), fontsize=6, ha="center", rotation=45)

        # Draw delay ranges for each minute (search times at 60, 120, 180, 240, 300, 360)
        for minute in range(1, 7):
            search_time = minute * 60

            # Long delay range: target_time in [search_time - long_max, search_time - long_min]
            long_start = search_time - long_max  # search_time - 60
            long_end = search_time - long_min    # search_time - 40

            # Short delay range: target_time in [search_time - short_max, search_time - short_min]
            short_start = search_time - short_max  # search_time - 20
            short_end = search_time - short_min    # search_time - 0

            # Draw ranges as horizontal bars
            bar_height = 0.15
            ax.barh(y - 0.2, long_end - long_start, left=long_start, height=bar_height,
                    color=long_color, alpha=0.3, zorder=1)
            ax.barh(y - 0.35, short_end - short_start, left=short_start, height=bar_height,
                    color=short_color, alpha=0.3, zorder=1)

            # Draw search time vertical line
            ax.axvline(x=search_time, color=search_color, linestyle="--", alpha=0.3, linewidth=0.5)

    # Formatting
    ax.set_xlim(-10, 370)
    ax.set_ylim(0.3, len(video_ids) + 0.7)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels([f"Video {v}" for v in video_ids])
    ax.set_xlabel("Time (seconds)")
    ax.set_title("Jargon Word Distribution with Long/Short Delay Ranges\n(Long: 40-60s delay, Short: 0-20s delay)")

    # Add minute markers
    for minute in range(7):
        ax.axvline(x=minute * 60, color="gray", linestyle="-", alpha=0.2, linewidth=1)
        if minute > 0:
            ax.text(minute * 60, len(video_ids) + 0.5, f"{minute}min", ha="center", fontsize=8, color="gray")

    # Legend
    legend_elements = [
        mpatches.Patch(color=word_color, alpha=0.8, label="Jargon words"),
        mpatches.Patch(color=long_color, alpha=0.3, label="Long delay range (40-60s)"),
        mpatches.Patch(color=short_color, alpha=0.3, label="Short delay range (0-20s)"),
        plt.Line2D([0], [0], color=search_color, linestyle="--", alpha=0.5, label="Search time"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
