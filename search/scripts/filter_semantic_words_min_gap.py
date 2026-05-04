"""Ensure semantic-word entries are at least a minimum time gap apart."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TERM_RE = re.compile(r"[^a-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter semantic-word entries so adjacent kept items are at least a minimum time gap apart."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Semantic-word JSON files to filter in place.",
    )
    parser.add_argument(
        "--min-gap-seconds",
        type=float,
        default=1.0,
        help="Minimum allowed gap between kept semantic-word timestamps.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=None,
        help="Optional directory containing target_words JSON files to preserve.",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional directory used to restore missing target entries before filtering.",
    )
    parser.add_argument(
        "--source-suffix",
        default=".zero.jargon.json",
        help="Suffix for source files when --source-dir is set.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.05,
        help="Tolerance for matching target_word_time to semantic word time.",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_word(word: str) -> str:
    return TERM_RE.sub("", word.lower())


def load_target_entries(path: Path) -> list[tuple[str, float]]:
    data = read_json(path)
    interruptions = data.get("interruptions", []) if isinstance(data, dict) else []
    entries: list[tuple[str, float]] = []
    for item in interruptions:
        word = item.get("target_word")
        time = item.get("target_word_time")
        if isinstance(word, str) and isinstance(time, (int, float)):
            entries.append((normalize_word(word), float(time)))
    return entries


def is_target_item(
    item: dict[str, Any],
    target_entries: list[tuple[str, float]],
    match_tolerance: float,
) -> bool:
    word = item.get("word")
    time = item.get("time")
    if not isinstance(word, str) or not isinstance(time, (int, float)):
        return False
    normalized = normalize_word(word)
    item_time = float(time)
    return any(
        normalized == target_word and abs(item_time - target_time) <= match_tolerance
        for target_word, target_time in target_entries
    )


def restore_missing_target_entries(
    entries: list[dict[str, Any]],
    source_entries: list[dict[str, Any]],
    target_entries: list[tuple[str, float]],
    match_tolerance: float,
) -> list[dict[str, Any]]:
    output = list(entries)
    for target_word, target_time in target_entries:
        exists = any(
            isinstance(item, dict)
            and isinstance(item.get("word"), str)
            and isinstance(item.get("time"), (int, float))
            and normalize_word(str(item["word"])) == target_word
            and abs(float(item["time"]) - target_time) <= match_tolerance
            for item in output
        )
        if exists:
            continue
        for item in source_entries:
            if (
                isinstance(item, dict)
                and isinstance(item.get("word"), str)
                and isinstance(item.get("time"), (int, float))
                and normalize_word(str(item["word"])) == target_word
                and abs(float(item["time"]) - target_time) <= match_tolerance
            ):
                output.append(item)
                break
    return output


def filter_min_gap(
    entries: list[dict[str, Any]],
    min_gap_seconds: float,
    *,
    target_entries: list[tuple[str, float]] | None = None,
    match_tolerance: float = 0.05,
) -> tuple[list[dict[str, Any]], int]:
    sorted_entries = sorted(
        entries,
        key=lambda item: float(item.get("time", 0.0))
        if isinstance(item.get("time"), (int, float))
        else float("inf"),
    )
    target_entries = target_entries or []
    preserved: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for item in sorted_entries:
        if is_target_item(item, target_entries, match_tolerance):
            preserved.append(item)
        else:
            candidates.append(item)

    kept: list[dict[str, Any]] = list(preserved)
    removed = 0
    kept_times = [
        float(item["time"])
        for item in kept
        if isinstance(item.get("time"), (int, float))
    ]

    for item in candidates:
        time = item.get("time")
        if not isinstance(time, (int, float)):
            kept.append(item)
            continue

        item_time = float(time)
        if any(abs(item_time - kept_time) < min_gap_seconds for kept_time in kept_times):
            removed += 1
            continue

        kept.append(item)
        kept_times.append(item_time)

    kept = sorted(
        kept,
        key=lambda item: float(item.get("time", 0.0))
        if isinstance(item.get("time"), (int, float))
        else float("inf"),
    )
    return kept, removed


def main() -> None:
    args = build_parser().parse_args()

    for path in args.paths:
        data = read_json(path)
        if not isinstance(data, list):
            raise SystemExit(f"JSON must contain a list: {path}")

        entries = [item for item in data if isinstance(item, dict)]
        target_entries: list[tuple[str, float]] = []
        if args.target_dir is not None:
            target_path = args.target_dir / f"{path.stem}.json"
            if target_path.exists():
                target_entries = load_target_entries(target_path)

        if args.source_dir is not None and target_entries:
            source_path = args.source_dir / f"{path.stem}{args.source_suffix}"
            if source_path.exists():
                source_data = read_json(source_path)
                if isinstance(source_data, list):
                    entries = restore_missing_target_entries(
                        entries,
                        [item for item in source_data if isinstance(item, dict)],
                        target_entries,
                        args.match_tolerance,
                    )

        filtered, removed = filter_min_gap(
            entries,
            args.min_gap_seconds,
            target_entries=target_entries,
            match_tolerance=args.match_tolerance,
        )
        write_json(path, filtered)
        print(f"{path}: kept {len(filtered)}, removed {removed}")


if __name__ == "__main__":
    main()
