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

DEFAULT_LOGS_DIR = Path("logs/study")
DEFAULT_RESULT_DIR = Path("result/figures")

# Will be set by main() based on command-line arguments
LOGS_DIR = DEFAULT_LOGS_DIR
RESULT_DIR = DEFAULT_RESULT_DIR

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
        dict: {participant_id: {condition: {"listening": [...], "interruptions": [...], "jumps": [...]}}}
    """
    data = defaultdict(
        lambda: defaultdict(lambda: {"listening": [], "interruptions": [], "jumps": []})
    )

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
                        # Record jump (start -> end transition)
                        data[pid][condition]["jumps"].append(
                            {
                                "start": start,
                                "end": end,
                            }
                        )

                        if progress and progress > 0:
                            # Forward listening: count the whole range
                            data[pid][condition]["listening"].append(
                                {
                                    "start": start,
                                    "end": end,
                                }
                            )
                        else:
                            # Backward jump or pause: count just the end position (brief listen)
                            data[pid][condition]["listening"].append(
                                {
                                    "start": end,
                                    "end": end + 0.5,
                                }
                            )

                elif event.get("event") == "interruption":
                    target_time = event.get("targetTime")
                    word = event.get("word", "")
                    delay_type = event.get("delayType", "unknown")
                    if target_time is not None:
                        data[pid][condition]["interruptions"].append(
                            {
                                "time": target_time,
                                "word": word,
                                "delay_type": delay_type,
                            }
                        )

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


def plot_participant_heatmap(
    pid: str, conditions_data: dict, output_path: Path, show_jumps: bool = False
):
    """Plot heatmap for a single participant (time-based x-axis).

    Args:
        pid: Participant ID
        conditions_data: {condition: {"listening": [...], "interruptions": [...], "jumps": [...]}}
        output_path: Path to save the figure
        show_jumps: If True, overlay jump arrows on the heatmap
    """
    condition_order = ["temporal", "sentence", "word", "keyword2", "keyword"]
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
            all_interruptions.append(
                {
                    "row": i,
                    "time": intr["time"],
                    "word": intr["word"],
                }
            )

    # Subtract 1 from all values (everyone listens at least once)
    heatmap_matrix = np.maximum(heatmap_matrix - 1, 0)

    # Create figure with more vertical space
    row_height = 0.7
    fig, ax = plt.subplots(figsize=(16, len(conditions) * row_height + 1.5))

    # Custom red colormap (white to red)
    colors = [
        "#FFFFFF",
        "#FFCCCC",
        "#FF9999",
        "#FF6666",
        "#FF3333",
        "#FF0000",
        "#CC0000",
    ]
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
            heatmap_clipped[i : i + 1],
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
            intr["time"], y_pos + bar_height, marker="v", color="blue", s=25, zorder=5
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


def plot_participant_heatmap_by_task(
    pid: str, conditions_data: dict, output_path: Path
):
    """Plot heatmap with 6 side-by-side 1-minute windows (short 3 + long 3).

    Fixed 60-second segments: 0-60, 60-120, 120-180, 180-240, 240-300, 300-360
    Each segment is classified as short/long based on the interruption in that segment.
    Arranged as: [Short1][Short2][Short3] | [Long1][Long2][Long3]

    Args:
        pid: Participant ID
        conditions_data: {condition: {"listening": [...], "interruptions": [...], "jumps": [...]}}
        output_path: Path to save the figure
    """
    condition_order = ["temporal", "sentence", "word", "keyword2", "keyword"]
    conditions = [c for c in condition_order if c in conditions_data]

    if not conditions:
        print(f"No data for {pid}")
        return

    # Window parameters - fixed 60-second segments
    window_duration = 60
    n_bins = int(window_duration / TIME_RESOLUTION)
    n_segments = 6  # 6 segments total (0-60, 60-120, ..., 300-360)

    # Build heatmap for each condition: 6 windows side by side
    # [S1 60s][S2 60s][S3 60s][L1 60s][L2 60s][L3 60s]
    total_bins = n_bins * 6
    heatmap_matrix = np.zeros((len(conditions), total_bins))

    # Track interruption positions for markers
    interruption_markers = []  # List of (row, x_position)

    for i, cond in enumerate(conditions):
        interruptions = conditions_data[cond]["interruptions"]
        listening = conditions_data[cond]["listening"]

        if not interruptions:
            continue

        # Build segment info: which segment has which delay_type
        # Segments: 0-60, 60-120, 120-180, 180-240, 240-300, 300-360
        segment_info = []  # List of {"segment_idx": 0-5, "delay_type": "short"/"long", "intr_time": float}
        for intr in interruptions:
            intr_time = intr["time"]
            segment_idx = int(intr_time // window_duration)
            if segment_idx < n_segments:
                segment_info.append(
                    {
                        "segment_idx": segment_idx,
                        "delay_type": intr.get("delay_type", "unknown"),
                        "intr_time": intr_time,
                    }
                )

        # Separate segments into short and long
        short_segments = sorted(
            [s for s in segment_info if s["delay_type"] == "short"],
            key=lambda x: x["segment_idx"],
        )
        long_segments = sorted(
            [s for s in segment_info if s["delay_type"] == "long"],
            key=lambda x: x["segment_idx"],
        )

        # Process short segments (positions 0, 1, 2 in output)
        for out_idx, seg in enumerate(short_segments[:3]):
            seg_idx = seg["segment_idx"]
            window_start = seg_idx * window_duration
            window_end = (seg_idx + 1) * window_duration

            # Build heatmap for this window
            for evt in listening:
                evt_start = max(evt["start"], window_start)
                evt_end = min(evt["end"], window_end)
                if evt_start < evt_end:
                    rel_start = evt_start - window_start
                    rel_end = evt_end - window_start
                    start_bin = max(0, int(rel_start / TIME_RESOLUTION))
                    end_bin = min(n_bins - 1, int(rel_end / TIME_RESOLUTION))
                    offset = out_idx * n_bins
                    for b in range(start_bin, end_bin + 1):
                        heatmap_matrix[i, offset + b] += 1

            # Mark interruption position within the window
            rel_intr_time = seg["intr_time"] - window_start
            marker_x = out_idx * n_bins + int(rel_intr_time / TIME_RESOLUTION)
            interruption_markers.append((i, marker_x))

        # Process long segments (positions 3, 4, 5 in output)
        for out_idx, seg in enumerate(long_segments[:3]):
            seg_idx = seg["segment_idx"]
            window_start = seg_idx * window_duration
            window_end = (seg_idx + 1) * window_duration

            for evt in listening:
                evt_start = max(evt["start"], window_start)
                evt_end = min(evt["end"], window_end)
                if evt_start < evt_end:
                    rel_start = evt_start - window_start
                    rel_end = evt_end - window_start
                    start_bin = max(0, int(rel_start / TIME_RESOLUTION))
                    end_bin = min(n_bins - 1, int(rel_end / TIME_RESOLUTION))
                    offset = (out_idx + 3) * n_bins
                    for b in range(start_bin, end_bin + 1):
                        heatmap_matrix[i, offset + b] += 1

            rel_intr_time = seg["intr_time"] - window_start
            marker_x = (out_idx + 3) * n_bins + int(rel_intr_time / TIME_RESOLUTION)
            interruption_markers.append((i, marker_x))

    # Subtract 1 (initial listen)
    heatmap_matrix = np.maximum(heatmap_matrix - 1, 0)

    # Create figure
    row_height = 0.7
    fig, ax = plt.subplots(figsize=(16, len(conditions) * row_height + 1.5))

    # Custom red colormap
    colors = [
        "#FFFFFF",
        "#FFCCCC",
        "#FF9999",
        "#FF6666",
        "#FF3333",
        "#FF0000",
        "#CC0000",
    ]
    cmap = LinearSegmentedColormap.from_list("white_red", colors)

    # Get vmax
    nonzero_vals = heatmap_matrix[heatmap_matrix > 0]
    if len(nonzero_vals) > 0:
        vmax = np.percentile(nonzero_vals, 95)
        vmax = max(vmax, 1)
    else:
        vmax = 1
    heatmap_clipped = np.clip(heatmap_matrix, 0, vmax)

    # Plot each condition as separate row with gaps
    row_positions = []
    gap = 0.3
    bar_height = 0.8
    for i, cond in enumerate(conditions):
        y_pos = i * (bar_height + gap)
        row_positions.append(y_pos + bar_height / 2)

        im = ax.imshow(
            heatmap_clipped[i : i + 1],
            aspect="auto",
            cmap=cmap,
            extent=[0, total_bins, y_pos, y_pos + bar_height],
            interpolation="nearest",
            vmin=0,
            vmax=vmax,
        )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Re-listen count")
    cbar.locator = plt.MaxNLocator(integer=True)
    cbar.update_ticks()

    # Plot interruption markers
    for row, marker_x in interruption_markers:
        y_pos = row * (bar_height + gap)
        ax.plot(
            [marker_x, marker_x],
            [y_pos + 0.05, y_pos + bar_height - 0.05],
            color="blue",
            linewidth=2,
            alpha=0.9,
        )
        ax.scatter(
            marker_x, y_pos + bar_height, marker="v", color="blue", s=25, zorder=5
        )

    # Draw vertical lines between tasks
    for task_boundary in range(1, 6):
        x = task_boundary * n_bins
        ax.axvline(x=x, color="gray", linewidth=1, linestyle="-", alpha=0.5)

    # Draw thicker line between short and long
    ax.axvline(x=3 * n_bins, color="black", linewidth=2)

    # Labels
    ax.set_yticks(row_positions)
    ax.set_yticklabels(conditions)
    ax.set_ylabel("Condition")
    ax.set_title(f"Listening Heatmap - {pid.upper()}")

    # X-axis: show time within each 1-min window
    # Ticks at 0, 30, 60 for each window
    tick_positions = []
    tick_labels = []
    for task_idx in range(6):
        offset = task_idx * n_bins
        tick_positions.extend([offset, offset + n_bins // 2, offset + n_bins])
        tick_labels.extend(["0", "30", "60"])
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, fontsize=8)
    ax.set_xlabel("Time within window (seconds)")

    # Add Short/Long labels at top
    total_height = len(conditions) * (bar_height + gap) - gap
    ax.text(
        1.5 * n_bins,
        total_height + 0.4,
        "Short",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax.text(
        4.5 * n_bins,
        total_height + 0.4,
        "Long",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )

    # Add task numbers
    for task_idx in range(3):
        ax.text(
            (task_idx + 0.5) * n_bins,
            total_height + 0.15,
            f"S{task_idx+1}",
            ha="center",
            va="center",
            fontsize=9,
        )
        ax.text(
            (task_idx + 3.5) * n_bins,
            total_height + 0.15,
            f"L{task_idx+1}",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_ylim(-0.1, total_height + 0.6)
    ax.set_xlim(0, total_bins)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    global LOGS_DIR, RESULT_DIR

    import argparse

    parser = argparse.ArgumentParser(description="Plot listening heatmaps")
    parser.add_argument(
        "--jumps",
        action="store_true",
        help="Show jump arrows on heatmap (only for time-based)",
    )
    parser.add_argument(
        "--by-time",
        action="store_true",
        help="Use time-based x-axis (default is by-task)",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=None,
        help="Directory containing log files (default: logs/study)",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save results (default: result/figures)",
    )
    args = parser.parse_args()

    if args.logs_dir:
        LOGS_DIR = Path(args.logs_dir)
    if args.result_dir:
        RESULT_DIR = Path(args.result_dir)

    os.makedirs(RESULT_DIR, exist_ok=True)

    data = load_listening_data()

    for pid in sorted(data.keys()):
        if args.by_time:
            output_path = RESULT_DIR / f"listening_heatmap_{pid}.png"
            plot_participant_heatmap(pid, data[pid], output_path, show_jumps=args.jumps)
        else:
            output_path = RESULT_DIR / f"listening_heatmap_task_{pid}.png"
            plot_participant_heatmap_by_task(pid, data[pid], output_path)

    print(f"\nGenerated {len(data)} heatmaps in {RESULT_DIR}/")


if __name__ == "__main__":
    main()
