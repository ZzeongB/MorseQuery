"""Filter semantic-word entries by transcript-word distance and time gap."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
TERM_RE = re.compile(r"[a-z0-9']+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter semantic-word JSON files so kept entries are separated by a "
            "minimum transcript-word distance and a minimum time gap."
        )
    )
    parser.add_argument(
        "--semantic-json",
        type=Path,
        required=True,
        help="Input semantic_words JSON file to filter in place unless --output-json is set.",
    )
    parser.add_argument(
        "--transcript-json",
        type=Path,
        required=True,
        help="Transcript JSON used to map semantic words back to transcript positions.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output path. Defaults to overwriting --semantic-json.",
    )
    parser.add_argument(
        "--min-word-gap",
        type=int,
        default=3,
        help="Minimum difference in transcript word index between kept entries.",
    )
    parser.add_argument(
        "--min-time-gap-seconds",
        type=float,
        default=2.0,
        help="Minimum difference in start time between kept entries.",
    )
    parser.add_argument(
        "--match-tolerance",
        type=float,
        default=0.05,
        help="Allowed absolute difference when matching semantic word time to transcript word start.",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_tokens(text: str) -> list[str]:
    return TERM_RE.findall(text.lower())


def load_transcript_words(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            raw_word = word.get("word", "")
            tokens = normalize_tokens(str(raw_word))
            if not tokens:
                continue
            start = word.get("start")
            if not isinstance(start, (int, float)):
                continue
            words.append(
                {
                    "token": tokens[0],
                    "start": float(start),
                }
            )
    return words


def normalize_word(word: Any) -> str | None:
    if not isinstance(word, str):
        return None
    tokens = normalize_tokens(word)
    if not tokens:
        return None
    return tokens[0]


def attach_transcript_index(
    entries: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
    *,
    match_tolerance: float,
) -> list[dict[str, Any]]:
    next_search_start = 0
    output: list[dict[str, Any]] = []

    for entry in sorted(
        entries,
        key=lambda item: float(item.get("time", 0.0))
        if isinstance(item.get("time"), (int, float))
        else float("inf"),
    ):
        item = dict(entry)
        token = normalize_word(item.get("word"))
        time = item.get("time")
        transcript_index = None

        if token is not None and isinstance(time, (int, float)):
            item_time = float(time)
            for idx in range(next_search_start, len(transcript_words)):
                transcript_word = transcript_words[idx]
                if transcript_word["token"] != token:
                    continue
                if abs(transcript_word["start"] - item_time) > match_tolerance:
                    continue
                transcript_index = idx
                next_search_start = idx + 1
                break

        item["_transcript_index"] = transcript_index
        output.append(item)

    return output


def filter_entries(
    entries: list[dict[str, Any]],
    *,
    min_word_gap: int,
    min_time_gap_seconds: float,
) -> tuple[list[dict[str, Any]], int]:
    kept: list[dict[str, Any]] = []
    removed = 0

    for entry in entries:
        if not kept:
            kept.append(entry)
            continue

        prev = kept[-1]
        current_time = entry.get("time")
        prev_time = prev.get("time")
        current_index = entry.get("_transcript_index")
        prev_index = prev.get("_transcript_index")

        violates_word_gap = (
            isinstance(current_index, int)
            and isinstance(prev_index, int)
            and current_index - prev_index < min_word_gap
        )
        violates_time_gap = (
            isinstance(current_time, (int, float))
            and isinstance(prev_time, (int, float))
            and float(current_time) - float(prev_time) < min_time_gap_seconds
        )

        if violates_word_gap or violates_time_gap:
            removed += 1
            continue

        kept.append(entry)

    cleaned = []
    for entry in kept:
        item = dict(entry)
        item.pop("_transcript_index", None)
        cleaned.append(item)
    return cleaned, removed


def main() -> None:
    args = build_parser().parse_args()

    semantic_data = read_json(args.semantic_json)
    if not isinstance(semantic_data, list):
        raise SystemExit(f"semantic JSON must contain a list: {args.semantic_json}")

    transcript_data = read_json(args.transcript_json)
    if not isinstance(transcript_data, dict):
        raise SystemExit(f"transcript JSON must contain an object: {args.transcript_json}")

    semantic_entries = [item for item in semantic_data if isinstance(item, dict)]
    transcript_words = load_transcript_words(transcript_data)
    indexed_entries = attach_transcript_index(
        semantic_entries,
        transcript_words,
        match_tolerance=args.match_tolerance,
    )
    filtered, removed = filter_entries(
        indexed_entries,
        min_word_gap=args.min_word_gap,
        min_time_gap_seconds=args.min_time_gap_seconds,
    )

    output_path = args.output_json or args.semantic_json
    write_json(output_path, filtered)
    print(
        f"{output_path}: kept {len(filtered)}, removed {removed} "
        f"(min_word_gap={args.min_word_gap}, min_time_gap_seconds={args.min_time_gap_seconds})"
    )


if __name__ == "__main__":
    main()
