#!/usr/bin/env python3
"""Plot search trajectory for each trial.

X-axis: elapsed time since interruption start
Y-axis: audio position

Usage:
    python scripts/analysis/plot_search_trajectory.py
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs/study")
RESULT_DIR = Path("result/figures/trajectories")

os.makedirs(RESULT_DIR, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt
import numpy as np


def parse_timestamp(ts_str: str) -> datetime:
    """Parse timestamp string to datetime."""
    return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))


def load_trial_data():
    """Load trial data from jsonl files.

    Returns:
        list of trial dicts with listening events
    """
    trials = []

    for path in sorted(LOGS_DIR.glob("*.jsonl")):
        with path.open() as f:
            events = [json.loads(line) for line in f]

        current_trial = None

        for e in events:
            audio = e.get("audio", "")
            if audio == "0.mp3" or audio.startswith("tutorial"):
                continue

            if e.get("event") == "interruption":
                # Save previous trial if exists
                if current_trial and (current_trial["navigations"] or current_trial["user_actions"]):
                    trials.append(current_trial)

                # Start new trial
                current_trial = {
                    "participant": e.get("participant"),
                    "condition": e.get("feature"),
                    "audio": audio,
                    "task_index": e.get("taskIndex"),
                    "target_word": e.get("word"),
                    "target_time": e.get("targetTime"),
                    "trigger_time": e.get("triggerTime"),
                    "interruption_ts": e.get("ts"),
                    "navigations": [],  # Jump events
                    "user_actions": [],
                    "spacebar_ts": None,
                }

            elif e.get("event") == "navigation" and current_trial:
                # Skip if spacebar already pressed
                if current_trial["spacebar_ts"] is not None:
                    continue

                from_time = e.get("fromTime")
                to_time = e.get("toTime")
                ts = e.get("ts")
                action = e.get("action", "")

                if from_time is not None and to_time is not None and abs(from_time - to_time) > 0.5:
                    current_trial["navigations"].append({
                        "from": from_time,
                        "to": to_time,
                        "ts": ts,
                        "action": action,
                    })

            elif e.get("event") == "user_action" and current_trial:
                action = e.get("action")
                ts = e.get("ts")
                audio_time = e.get("audioTime")

                # Skip if spacebar already pressed
                if current_trial["spacebar_ts"] is not None:
                    continue

                # Record user action
                if audio_time is not None:
                    current_trial["user_actions"].append({
                        "action": action,
                        "ts": ts,
                        "audio_time": audio_time,
                    })

                # Record spacebar timestamp
                if action == "space":
                    current_trial["spacebar_ts"] = ts

            elif e.get("event") == "task_complete" and current_trial:
                current_trial["outcome"] = e.get("outcome")
                current_trial["distance"] = e.get("distanceSeconds")
                trials.append(current_trial)
                current_trial = None

        # Save last trial if not completed
        if current_trial and (current_trial["navigations"] or current_trial["user_actions"]):
            trials.append(current_trial)

    return trials


def plot_trial_trajectory(trial: dict, output_path: Path):
    """Plot trajectory for a single trial."""
    navigations = trial.get("navigations", [])
    if not navigations:
        return

    # Parse interruption timestamp
    int_ts = parse_timestamp(trial["interruption_ts"])
    trigger_time = trial.get("trigger_time", 60)

    # Build segments: start from trigger_time, follow navigations
    segments = []
    current_pos = trigger_time
    prev_elapsed = 0

    for nav in navigations:
        ts = parse_timestamp(nav["ts"])
        elapsed = (ts - int_ts).total_seconds()

        # Skip if beyond 60 seconds
        if elapsed > 60:
            break

        # Play segment: from previous position to current (before jump)
        # Time elapsed = audio progress (assuming 1x playback)
        time_diff = elapsed - prev_elapsed
        play_end_pos = current_pos + time_diff  # Audio progresses with time

        if time_diff > 0.1:  # Only draw if significant time passed
            segments.append({
                "type": "play",
                "x1": prev_elapsed, "y1": current_pos,
                "x2": elapsed, "y2": nav["from"],  # Use nav's from as the actual position before jump
            })

        # Jump segment: instant position change
        segments.append({
            "type": "jump",
            "x1": elapsed, "y1": nav["from"],
            "x2": elapsed, "y2": nav["to"],
        })

        current_pos = nav["to"]
        prev_elapsed = elapsed

    # Add final play segment from last navigation to spacebar (or end)
    spacebar_ts = trial.get("spacebar_ts")
    if spacebar_ts and prev_elapsed < 60:
        spacebar_elapsed = (parse_timestamp(spacebar_ts) - int_ts).total_seconds()
        spacebar_elapsed = min(spacebar_elapsed, 60)

        if spacebar_elapsed > prev_elapsed + 0.1:
            # Find spacebar audio position from user_actions
            spacebar_audio = None
            for ua in trial.get("user_actions", []):
                if ua["action"] == "space":
                    spacebar_audio = ua["audio_time"]
                    break

            if spacebar_audio is not None:
                segments.append({
                    "type": "play",
                    "x1": prev_elapsed, "y1": current_pos,
                    "x2": spacebar_elapsed, "y2": spacebar_audio,
                })

    if not segments:
        return

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot segments
    for seg in segments:
        if seg["type"] == "jump":
            # Vertical line for jump (purple, dashed)
            ax.plot([seg["x1"], seg["x2"]], [seg["y1"], seg["y2"]],
                    color="purple", linewidth=1.5, linestyle="--", alpha=0.7)
        else:
            # Diagonal line for playback (red, solid)
            ax.plot([seg["x1"], seg["x2"]], [seg["y1"], seg["y2"]],
                    color="#CC0000", linewidth=1.5, alpha=0.8)

    # Plot user actions as markers
    action_colors = {
        "left": "green",
        "right": "orange",
        "up": "blue",
        "space": "red",
        "seek": "gray",
    }
    action_markers = {
        "left": "<",
        "right": ">",
        "up": "^",
        "space": "s",
        "seek": "o",
    }

    # Track which actions we've seen for legend
    seen_actions = set()
    for ua in trial.get("user_actions", []):
        ts = parse_timestamp(ua["ts"])
        elapsed = (ts - int_ts).total_seconds()
        if elapsed > 60:
            continue
        audio_time = ua["audio_time"]
        action = ua["action"]
        color = action_colors.get(action, "black")
        marker = action_markers.get(action, "o")
        label = action if action not in seen_actions else None
        ax.scatter(elapsed, audio_time, color=color, marker=marker, s=40, zorder=10, alpha=0.8, label=label)
        seen_actions.add(action)

    # Mark target word time
    target_time = trial.get("target_time")
    if target_time:
        ax.axhline(y=target_time, color="blue", linewidth=2, linestyle="--",
                   label=f"Target: {trial.get('target_word')} @ {target_time:.1f}s")

    # Mark trigger time (where interruption happened)
    trigger_time = trial.get("trigger_time")
    if trigger_time:
        ax.axhline(y=trigger_time, color="gray", linewidth=1, linestyle=":",
                   label=f"Trigger @ {trigger_time:.1f}s")

    # Labels
    ax.set_xlabel("Elapsed time since interruption (seconds)")
    ax.set_ylabel("Audio position (seconds)")
    ax.set_title(
        f"{trial['participant'].upper()} - {trial['condition']} - Task {trial['task_index']}\n"
        f"Target: \"{trial.get('target_word')}\" @ {target_time:.1f}s"
    )
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Set x limit to 60 seconds max
    ax.set_xlim(0, 60)

    # Set y limits based on task_index (task1: 0-60, task2: 60-120, ...)
    task_idx = trial.get("task_index", 1)
    y_min = (task_idx - 1) * 60
    y_max = task_idx * 60
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    trials = load_trial_data()

    # Group by participant
    by_participant = defaultdict(list)
    for t in trials:
        by_participant[t["participant"]].append(t)

    count = 0
    for pid, participant_trials in sorted(by_participant.items()):
        # Create participant directory
        pid_dir = RESULT_DIR / pid
        os.makedirs(pid_dir, exist_ok=True)

        for trial in participant_trials:
            filename = f"{trial['condition']}_task{trial['task_index']}.png"
            output_path = pid_dir / filename
            plot_trial_trajectory(trial, output_path)
            count += 1

    print(f"Generated {count} trajectory plots in {RESULT_DIR}/")


if __name__ == "__main__":
    main()
