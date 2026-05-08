"""Keep target word/time anchors and remove nearby semantic words."""

from __future__ import annotations
import random
import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
TERM_RE = re.compile(r"[^a-z0-9]+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Keep target (word, time) entries and remove other semantic words within "
            "a time window around each target."
        )
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words",
        help="Directory containing semantic word JSON files.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT_DIR / "data" / "study" / "target_words",
        help="Directory containing target word JSON files.",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=1.0,
        help="Remove non-target semantic words within +/- this many seconds.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.05,
        help="Tolerance used to match a semantic entry to a target (word, time).",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional file stems to process.",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def normalize_word(word: str) -> str:
    return TERM_RE.sub("", word.lower())


def load_target_entries(path: Path) -> list[tuple[str, float]]:
    data = read_json(path)
    interruptions = data.get("interruptions", []) if isinstance(data, dict) else []
    entries: list[tuple[str, float]] = []
    for item in interruptions:
        word = item.get("target_word")
        time = item.get("target_word_time")
        if not isinstance(word, str) or not isinstance(time, (int, float)):
            continue
        entries.append((normalize_word(word), float(time)))
    return entries


def is_exact_target(
    item: dict[str, Any],
    target_entries: list[tuple[str, float]],
    match_tolerance: float,
) -> bool:
    word = item.get("word")
    time = item.get("time")
    if not isinstance(word, str) or not isinstance(time, (int, float)):
        return False
    normalized_word = normalize_word(word)
    item_time = float(time)
    return any(
        normalized_word == target_word
        and abs(item_time - target_time) <= match_tolerance
        for target_word, target_time in target_entries
    )


def should_remove(
    item: dict[str, Any],
    target_entries: list[tuple[str, float]],
    *,
    window_seconds: float,
    match_tolerance: float,
) -> bool:
    if is_exact_target(item, target_entries, match_tolerance):
        return False
    time = item.get("time")
    if not isinstance(time, (int, float)):
        return False
    item_time = float(time)
    return any(
        abs(item_time - target_time) <= window_seconds
        for _, target_time in target_entries
    )


def filter_semantic_words(
    semantic_words: list[dict[str, Any]],
    target_entries: list[tuple[str, float]],
    *,
    window_seconds: float,
    match_tolerance: float,
) -> tuple[list[dict[str, Any]], int]:
    target_items: list[dict[str, Any]] = []
    candidate_items: list[dict[str, Any]] = []
    passthrough_items: list[dict[str, Any]] = []

    for item in semantic_words:
        if not isinstance(item, dict):
            passthrough_items.append(item)
            continue

        time = item.get("time")
        if not isinstance(time, (int, float)):
            passthrough_items.append(item)
            continue

        if is_exact_target(item, target_entries, match_tolerance):
            target_items.append(item)
        else:
            candidate_items.append(item)

    kept: list[dict[str, Any]] = list(target_items)

    # target 주변 +- window_seconds 안의 non-target 제거
    def too_close_to_kept(item: dict[str, Any]) -> bool:
        item_time = float(item["time"])
        return any(
            abs(item_time - float(kept_item["time"])) <= window_seconds
            for kept_item in kept
            if isinstance(kept_item.get("time"), (int, float))
        )

    # non-target은 random 순서로 보면서, 기존 kept와 1초 이상 떨어진 것만 keep
    random.shuffle(candidate_items)

    for item in candidate_items:
        if not too_close_to_kept(item):
            kept.append(item)

    # 원래 시간순으로 정렬
    kept = sorted(
        kept,
        key=lambda x: float(x["time"])
        if isinstance(x.get("time"), (int, float))
        else float("inf"),
    )

    output = passthrough_items + kept
    removed = len(semantic_words) - len(output)

    return output, removed


def main() -> None:
    args = build_parser().parse_args()
    target_paths = sorted(args.target_dir.glob("*.json"))
    if args.stems is not None:
        allowed = set(args.stems)
        target_paths = [path for path in target_paths if path.stem in allowed]

    for target_path in target_paths:
        semantic_path = args.semantic_dir / f"{target_path.stem}.json"
        if not semantic_path.exists():
            print(f"skip {target_path.stem}: missing semantic file {semantic_path}")
            continue

        target_entries = load_target_entries(target_path)
        semantic_words = read_json(semantic_path)
        if not isinstance(semantic_words, list):
            raise ValueError(f"Semantic JSON must contain a list: {semantic_path}")

        filtered, removed = filter_semantic_words(
            semantic_words,
            target_entries,
            window_seconds=args.window_seconds,
            match_tolerance=args.match_tolerance,
        )
        write_json(semantic_path, filtered)
        print(f"{semantic_path}: kept {len(filtered)}, removed {removed}")


if __name__ == "__main__":
    main()
