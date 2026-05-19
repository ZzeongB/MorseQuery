#!/usr/bin/env python3
"""Plot listening heatmaps per participant.

Creates heatmaps showing which audio segments each participant listened to most.
X-axis: time (seconds)
Y-axis: conditions
Color intensity: listening frequency (more intense = listened more times)

Usage:
    python scripts/analysis/plot_listening_heatmap.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

LOGS_DIR = Path("logs/study")
RESULT_DIR = Path("result")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Time resolution for heatmap (seconds)
TIME_RESOLUTION = 0.5
# Max audio duration to consider (seconds)
MAX_DURATION = 400


def extract_participant_id(log_base_name: str) -> str:
    """Extract participant ID from log base name."""
    return log_base_name.split("_")[0]


def extract_condition(log_base_name: str) -> str:
    """Extract condition from log base name."""
    return log_base_name.split("_")[1]


def load_listening_data():
    """Load listening events and interruption data from jsonl files.

    Returns:
        dict: {participant_id: {condition: {"listening": [...], "interruptions": [...]}}}
    """
    data = defaultdict(lambda: defaultdict(lambda: {"listening": [], "interruptions": []}))

    for path in sorted(LOGS_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                event = json.loads(line)
                audio = event.get("audio", "")
                # Skip tutorial and audio 0
                if audio == "0.mp3" or audio.startswith("tutorial"):
                    continue

                log_base = event["logBaseName"]
                pid = extract_participant_id(log_base)
                condition = event.get("feature", extract_condition(log_base))

                if event.get("event") == "listening":
                    start = event.get("startAudioTime", 0)
                    end = event.get("endAudioTime", 0)
                    progress = event.get("audioProgressSeconds", 0)
                    duration_ms = event.get("listeningDurationMs", 0)

                    if start is not None and end is not None and duration_ms > 50:
                        if progress and progress > 0:
                            # Forward listening: count the whole range
                            data[pid][condition]["listening"].append({
                                "start": start,
                                "end": end,
                            })
                        else:
                            # Backward jump or pause: count just the end position (brief listen)
                            data[pid][condition]["listening"].append({
                                "start": end,
                                "end": end + 0.5,
                            })

                elif event.get("event") == "interruption":
                    target_time = event.get("targetTime")
                    word = event.get("word", "")
                    if target_time is not None:
                        data[pid][condition]["interruptions"].append({
                            "time": target_time,
                            "word": word,
                        })

    return data


def build_heatmap_array(listening_events: list, max_time: float) -> np.ndarray:
    """Build a 1D array representing listening frequency over time.

    Args:
        listening_events: List of {"start": float, "end": float}
        max_time: Maximum time to consider

    Returns:
        1D numpy array where each bin represents TIME_RESOLUTION seconds
    """
    n_bins = int(max_time / TIME_RESOLUTION) + 1
    heatmap = np.zeros(n_bins)

    for event in listening_events:
        start_bin = max(0, int(event["start"] / TIME_RESOLUTION))
        end_bin = min(n_bins - 1, int(event["end"] / TIME_RESOLUTION))
        # Add 1 to all bins in this range
        for i in range(start_bin, end_bin + 1):
            heatmap[i] += 1

    return heatmap


def plot_participant_heatmap(pid: str, conditions_data: dict, output_path: Path):
    """Plot heatmap for a single participant.

    Args:
        pid: Participant ID
        conditions_data: {condition: {"listening": [...], "interruptions": [...]}}
        output_path: Path to save the figure
    """
    condition_order = ["word", "sentence", "keyword2", "keyword", "discontinuous"]
    conditions = [c for c in condition_order if c in conditions_data]

    if not conditions:
        print(f"No data for {pid}")
        return

    # Find max time across all conditions
    max_time = 0
    for cond in conditions:
        for event in conditions_data[cond]["listening"]:
            max_time = max(max_time, event["end"])
    max_time = min(max_time + 10, MAX_DURATION)

    # Build heatmap matrix
    n_bins = int(max_time / TIME_RESOLUTION) + 1
    heatmap_matrix = np.zeros((len(conditions), n_bins))

    for i, cond in enumerate(conditions):
        heatmap_matrix[i] = build_heatmap_array(
            conditions_data[cond]["listening"], max_time
        )

    # Collect all interruptions with positions
    all_interruptions = []
    for i, cond in enumerate(conditions):
        for intr in conditions_data[cond]["interruptions"]:
            all_interruptions.append({
                "row": i,
                "time": intr["time"],
                "word": intr["word"],
            })

    # Subtract 1 from all values (everyone listens at least once)
    heatmap_matrix = np.maximum(heatmap_matrix - 1, 0)

    # Create figure with more vertical space
    row_height = 0.7
    fig, ax = plt.subplots(figsize=(16, len(conditions) * row_height + 1.5))

    # Custom red colormap (white to red)
    colors = ["#FFFFFF", "#FFCCCC", "#FF9999", "#FF6666", "#FF3333", "#FF0000", "#CC0000"]
    cmap = LinearSegmentedColormap.from_list("white_red", colors)

    # Clip values to 95th percentile for better visualization
    nonzero_vals = heatmap_matrix[heatmap_matrix > 0]
    if len(nonzero_vals) > 0:
        vmax = np.percentile(nonzero_vals, 95)
        vmax = max(vmax, 1)  # At least 1
    else:
        vmax = 1
    heatmap_clipped = np.clip(heatmap_matrix, 0, vmax)

    # Plot each condition as separate row with gaps
    row_positions = []
    gap = 0.3  # Gap between rows
    bar_height = 0.8
    for i, cond in enumerate(conditions):
        y_pos = i * (bar_height + gap)
        row_positions.append(y_pos + bar_height / 2)

        # Plot heatmap row
        im = ax.imshow(
            heatmap_clipped[i:i+1],
            aspect="auto",
            cmap=cmap,
            extent=[0, max_time, y_pos, y_pos + bar_height],
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )

    # Add colorbar with integer ticks only
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Re-listen count")
    cbar.locator = plt.MaxNLocator(integer=True)
    cbar.update_ticks()

    # Plot target word markers (no text)
    for intr in all_interruptions:
        y_pos = intr["row"] * (bar_height + gap)
        # Draw vertical line marker
        ax.plot(
            [intr["time"], intr["time"]],
            [y_pos + 0.05, y_pos + bar_height - 0.05],
            color="blue",
            linewidth=2,
            alpha=0.9,
        )
        # Draw triangle marker at the top
        ax.scatter(
            intr["time"], y_pos + bar_height,
            marker="v", color="blue", s=25, zorder=5
        )

    # Labels
    ax.set_yticks(row_positions)
    ax.set_yticklabels(conditions)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Condition")
    ax.set_title(f"Listening Heatmap - {pid.upper()}")

    # Set y limits
    total_height = len(conditions) * (bar_height + gap) - gap
    ax.set_ylim(-0.1, total_height + 0.3)

    # X-axis ticks every 30 seconds
    ax.set_xticks(np.arange(0, max_time + 1, 30))

    # Draw vertical lines every 60 seconds
    for t in np.arange(60, max_time, 60):
        ax.axvline(x=t, color="gray", linewidth=1, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    data = load_listening_data()

    for pid in sorted(data.keys()):
        output_path = RESULT_DIR / f"listening_heatmap_{pid}.png"
        plot_participant_heatmap(pid, data[pid], output_path)

    print(f"\nGenerated {len(data)} heatmaps in {RESULT_DIR}/")


if __name__ == "__main__":
    main()
