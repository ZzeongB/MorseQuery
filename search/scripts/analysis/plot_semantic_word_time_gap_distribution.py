"""Plot semantic-word time-gap distributions for selected transcript ids."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot histogram and boxplot of semantic-word time gaps."
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words",
        help="Directory containing semantic-word JSON files.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        default=["0", "1", "2", "3", "4", "5"],
        help="Transcript ids to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR
        / "data"
        / "semantic_words"
        / "semantic_word_time_gap_distribution_0_1_2_3_4_5.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=12,
        help="Number of histogram bins per file.",
    )
    parser.add_argument(
        "--large-gap-threshold",
        type=float,
        default=30.0,
        help="Print gaps strictly larger than this threshold in seconds.",
    )
    parser.add_argument(
        "--audio-windows",
        type=Path,
        default=ROOT_DIR / "data" / "study" / "audio_windows.json",
        help="Study audio window config.",
    )
    parser.add_argument(
        "--prefer-ts-edit",
        action="store_true",
        help="Prefer {id}_ts_edit.json over {id}.json when both exist.",
    )
    parser.add_argument(
        "--large-gap-output",
        type=Path,
        default=ROOT_DIR
        / "data"
        / "semantic_words"
        / "semantic_word_large_gaps_0_1_2_3_4_5.txt",
        help="Output txt path for printed large-gap records.",
    )
    return parser


def resolve_input_path(semantic_dir: Path, item_id: str, *, prefer_ts_edit: bool) -> Path:
    preferred = semantic_dir / f"{item_id}_ts_edit.json"
    fallback = semantic_dir / f"{item_id}.json"
    if prefer_ts_edit and preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    if preferred.exists():
        return preferred
    raise FileNotFoundError(
        f"Missing semantic words for id={item_id}: {preferred.name} or {fallback.name}"
    )


def load_time_gaps(path: Path) -> list[float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    times = []
    for item in data:
        if not isinstance(item, dict):
            continue
        time = item.get("time")
        if isinstance(time, (int, float)):
            times.append(float(time))

    times.sort()
    return [
        round(curr - prev, 6)
        for prev, curr in zip(times, times[1:])
        if curr >= prev
    ]


def load_gap_records(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        time = item.get("time")
        word = item.get("word")
        if isinstance(time, (int, float)) and isinstance(word, str):
            items.append({"word": word, "time": float(time)})

    items.sort(key=lambda item: item["time"])

    records: list[dict[str, Any]] = []
    for prev, curr in zip(items, items[1:]):
        gap = curr["time"] - prev["time"]
        if gap < 0:
            continue
        records.append(
            {
                "gap": round(gap, 6),
                "prev_word": prev["word"],
                "prev_time": prev["time"],
                "next_word": curr["word"],
                "next_time": curr["time"],
            }
        )
    return records


def print_large_gaps(
    item_id: str, path: Path, gap_records: list[dict[str, Any]], threshold: float
) -> list[str]:
    large_gaps = [record for record in gap_records if record["gap"] > threshold]
    if not large_gaps:
        return []
    lines = [f"{item_id}: {path.name} gaps > {threshold:g}s"]
    print(lines[0])
    for record in large_gaps:
        line = (
            "  "
            f"{record['prev_word']} ({record['prev_time']:.6f})"
            f" -> {record['next_word']} ({record['next_time']:.6f})"
            f" | gap={record['gap']:.6f}"
        )
        print(line)
        lines.append(line)
    return lines


def load_semantic_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}")

    items: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        time = item.get("time")
        word = item.get("word")
        if isinstance(time, (int, float)) and isinstance(word, str):
            items.append({"word": word, "time": float(time)})
    items.sort(key=lambda item: item["time"])
    return items


def load_audio_windows(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected an object in {path}")
    return data


def canonical_video_id(item_id: str) -> str:
    return item_id[:-8] if item_id.endswith("_ts_edit") else item_id


def get_audio_window(audio_windows: dict[str, Any], item_id: str) -> tuple[float, float]:
    default_target_window_seconds = float(
        audio_windows.get("default_target_window_seconds", 300)
    )
    video_config = audio_windows.get("videos", {}).get(canonical_video_id(item_id), {})
    start_time = float(video_config.get("audio_start_time", 0.0))
    target_window_seconds = float(
        video_config.get("target_window_seconds", default_target_window_seconds)
    )
    return start_time, start_time + target_window_seconds


def semantic_words_in_window(
    items: list[dict[str, Any]], start_time: float, end_time: float
) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if start_time <= float(item["time"]) <= end_time
    ]


def print_window_words(
    item_id: str, path: Path, window_start: float, window_end: float, items: list[dict[str, Any]]
) -> None:
    print(
        f"{item_id}: {path.name} audio window "
        f"[{window_start:.2f}, {window_end:.2f}] semantic words ({len(items)})"
    )
    for item in items:
        print(f"  {item['word']} ({item['time']:.6f})")


def plot_distributions(
    *,
    semantic_dir: Path,
    ids: list[str],
    output: Path,
    bins: int,
    large_gap_threshold: float,
    audio_windows_path: Path,
    prefer_ts_edit: bool,
    large_gap_output: Path,
) -> list[tuple[str, Path, int]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audio_windows = load_audio_windows(audio_windows_path)
    rows = 4
    cols = len(ids)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 4.6), squeeze=False)

    summary: list[tuple[str, Path, int]] = []
    large_gap_lines: list[str] = []
    for col, item_id in enumerate(ids):
        path = resolve_input_path(semantic_dir, item_id, prefer_ts_edit=prefer_ts_edit)
        semantic_items = load_semantic_items(path)
        gap_records = load_gap_records(path)
        gaps = [record["gap"] for record in gap_records]
        large_gap_lines.extend(
            print_large_gaps(item_id, path, gap_records, large_gap_threshold)
        )
        window_start, window_end = get_audio_window(audio_windows, item_id)
        window_items = semantic_words_in_window(semantic_items, window_start, window_end)
        window_gap_records = [
            record
            for record in gap_records
            if window_start <= record["prev_time"] <= window_end
            and window_start <= record["next_time"] <= window_end
        ]
        window_gaps = [record["gap"] for record in window_gap_records]
        print_window_words(item_id, path, window_start, window_end, window_items)
        summary.append((item_id, path, len(gaps)))

        hist_ax = axes[0][col]
        box_ax = axes[1][col]
        window_hist_ax = axes[2][col]
        window_ax = axes[3][col]

        if gaps:
            hist_ax.hist(gaps, bins=bins, color="#4c78a8", alpha=0.9, edgecolor="white")
            box_ax.boxplot(
                gaps,
                vert=False,
                patch_artist=True,
                boxprops={"facecolor": "#f58518", "alpha": 0.7},
                medianprops={"color": "#222222"},
            )
        else:
            hist_ax.text(0.5, 0.5, "No gaps", ha="center", va="center")
            box_ax.text(0.5, 0.5, "No gaps", ha="center", va="center")

        hist_ax.set_title(f"{path.name} gap distribution")
        hist_ax.set_xlabel("Gap between semantic words (sec)")
        hist_ax.set_ylabel("Count")
        hist_ax.grid(True, alpha=0.2)

        box_ax.set_title(f"{path.name} gap boxplot")
        box_ax.set_xlabel("Gap between semantic words (sec)")
        box_ax.grid(True, alpha=0.2)

        if window_gaps:
            window_hist_ax.hist(
                window_gaps,
                bins=bins,
                color="#72b7b2",
                alpha=0.9,
                edgecolor="white",
            )
        else:
            window_hist_ax.text(0.5, 0.5, "No window gaps", ha="center", va="center")
        window_hist_ax.set_title(f"{path.name} audio-window gap distribution")
        window_hist_ax.set_xlabel("Gap between semantic words (sec)")
        window_hist_ax.set_ylabel("Count")
        window_hist_ax.grid(True, alpha=0.2)

        all_times = [item["time"] for item in semantic_items]
        window_times = [item["time"] for item in window_items]
        window_ax.axvspan(window_start, window_end, color="#54a24b", alpha=0.18)
        if all_times:
            window_ax.vlines(all_times, 0.05, 0.35, color="#bab0ac", linewidth=1)
        if window_times:
            window_ax.vlines(window_times, 0.45, 0.9, color="#e45756", linewidth=1.2)
        window_ax.set_ylim(0, 1)
        window_ax.set_yticks([])
        window_ax.set_title(f"{path.name} audio window")
        window_ax.set_xlabel("Semantic word time (sec)")
        window_ax.grid(True, axis="x", alpha=0.2)

        if semantic_items:
            max_time = max(item["time"] for item in semantic_items)
            window_ax.set_xlim(0, max(max_time, window_end) * 1.02)

        if window_items:
            words_text = ", ".join(item["word"] for item in window_items)
        else:
            words_text = "None"
        summary_text = "\n".join(
            [
                f"window: {window_start:.1f}-{window_end:.1f}s",
                f"count: {len(window_items)}",
                textwrap.fill(f"words: {words_text}", width=28),
            ]
        )
        window_ax.text(
            0.01,
            0.98,
            summary_text,
            transform=window_ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    large_gap_output.parent.mkdir(parents=True, exist_ok=True)
    large_gap_output.write_text("\n".join(large_gap_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = build_parser().parse_args()
    summary = plot_distributions(
        semantic_dir=args.semantic_dir,
        ids=args.ids,
        output=args.output,
        bins=args.bins,
        large_gap_threshold=args.large_gap_threshold,
        audio_windows_path=args.audio_windows,
        prefer_ts_edit=args.prefer_ts_edit,
        large_gap_output=args.large_gap_output,
    )
    print(f"Wrote plot: {args.output}")
    print(f"Wrote large gaps: {args.large_gap_output}")
    for item_id, path, gap_count in summary:
        print(f"{item_id}: {path.name} ({gap_count} gaps)")


if __name__ == "__main__":
    main()
