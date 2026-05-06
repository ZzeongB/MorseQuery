#!/usr/bin/env python3
"""Summarize word search outcomes from study logs."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path


LOGS_DIR = Path("logs/study")
RESULT_DIR = Path("result")
TRIAL_CSV = RESULT_DIR / "word_search_trials.csv"
SUMMARY_CSV = RESULT_DIR / "word_search_summary.csv"
PLOT_PATH = RESULT_DIR / "word_search_time_by_condition.png"

RESULT_DIR.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str((RESULT_DIR / ".matplotlib").resolve())
os.environ["XDG_CACHE_HOME"] = str((RESULT_DIR / ".cache").resolve())

import matplotlib.pyplot as plt


def load_interruption_meta():
    meta = {}
    for path in sorted(LOGS_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                event = json.loads(line)
                if event.get("audio") == "0.mp3":
                    continue
                if event.get("event") != "interruption":
                    continue
                meta[(event["logBaseName"], event["taskIndex"])] = {
                    "condition": event["feature"],
                    "audio": event["audio"],
                    "word_search_type": event["delayType"],
                }
    return meta


def build_trial_rows(meta):
    rows = []
    for path in sorted(LOGS_DIR.glob("*.json")):
        with path.open() as f:
            session = json.load(f)
        if session.get("audio") == "0.mp3":
            continue

        for task in session.get("tasks", []):
            key = (session["logBaseName"], task["taskIndex"])
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


def write_trial_csv(rows):
    with TRIAL_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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


def write_summary_csv(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["word_search_type"])].append(row)
        grouped[(row["condition"], "all")].append(row)

    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "condition",
                "word_search_type",
                "n_trials",
                "n_success",
                "n_failure",
                "success_rate",
                "mean_word_search_time_sec",
            ]
        )
        for (condition, word_search_type) in sorted(grouped):
            group = grouped[(condition, word_search_type)]
            n_trials = len(group)
            n_success = sum(1 for row in group if row["word_search_success"])
            n_failure = sum(1 for row in group if row["word_search_failure"])
            times = [
                row["word_search_time_sec"]
                for row in group
                if row["word_search_time_sec"] != ""
            ]
            mean_time = "" if not times else sum(times) / len(times)
            writer.writerow(
                [
                    condition,
                    word_search_type,
                    n_trials,
                    n_success,
                    n_failure,
                    n_success / n_trials if n_trials else "",
                    mean_time,
                ]
            )


def plot_search_times(rows):
    condition_order = ["discontinuous", "keyword", "keyword2", "sentence", "word"]
    type_order = ["all", "long", "short"]

    grouped = defaultdict(list)
    for row in rows:
        if row["word_search_time_sec"] == "":
            continue
        grouped[(row["condition"], row["word_search_type"])].append(row["word_search_time_sec"])
        grouped[(row["condition"], "all")].append(row["word_search_time_sec"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    for ax, word_type in zip(axes, type_order):
        xs = []
        ys = []
        for condition in condition_order:
            times = grouped.get((condition, word_type), [])
            if not times:
                continue
            xs.append(condition)
            ys.append(sum(times) / len(times))

        ax.bar(xs, ys, color="#4C78A8")
        ax.set_title(word_type)
        ax.set_xlabel("condition")
        ax.tick_params(axis="x", rotation=45)

    axes[0].set_ylabel("time (sec)")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)


def main():
    meta = load_interruption_meta()
    rows = build_trial_rows(meta)
    write_trial_csv(rows)
    write_summary_csv(rows)
    plot_search_times(rows)


if __name__ == "__main__":
    main()
