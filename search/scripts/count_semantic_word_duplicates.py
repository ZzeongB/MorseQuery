"""Count repeated semantic-word occurrences around each anchor timestamp."""

from __future__ import annotations

import argparse
import json
import re
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

TERM_RE = re.compile(r"[a-z0-9']+")
IRREGULAR_TOKEN_MAP = {
    "nuclei": "nucleus",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count duplicate semantic-word occurrences in local transcript windows."
    )
    parser.add_argument(
        "--semantic-json",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words" / "4.json",
        help="Semantic words JSON with word/time pairs.",
    )
    parser.add_argument(
        "--transcript-json",
        type=Path,
        default=ROOT_DIR / "data" / "transcripts" / "4.json",
        help="Transcript JSON with word timestamps.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words" / "4.counts.json",
        help="Output path for semantic words augmented with duplicate counts.",
    )
    return parser


def normalize_tokens(text: str) -> list[str]:
    return TERM_RE.findall(text.lower())


def canonicalize_token(token: str) -> str:
    token = IRREGULAR_TOKEN_MAP.get(token, token)
    if token.endswith("'s") and len(token) > 2:
        token = token[:-2]
    elif token.endswith("s'") and len(token) > 2:
        token = token[:-1]

    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return f"{token[:-3]}y"
    if token.endswith("sses") or token.endswith("ss"):
        return token
    if token.endswith(("us", "is")):
        return token
    if token.endswith("s"):
        return token[:-1]
    return token


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def transcript_occurrences(transcript: dict[str, Any]) -> dict[str, list[float]]:
    occurrences: dict[str, list[float]] = {}
    for segment in transcript.get("segments", []):
        for word in segment.get("words", []):
            raw_word = str(word.get("word", ""))
            start = word.get("start")
            if start is None:
                continue
            tokens = normalize_tokens(raw_word)
            if len(tokens) != 1:
                continue
            token = canonicalize_token(tokens[0])
            occurrences.setdefault(token, []).append(float(start))
    return occurrences


def count_occurrences_in_window(
    times: list[float],
    center: float,
    radius_seconds: float,
) -> int:
    start = center - radius_seconds
    end = center + radius_seconds
    left = bisect_left(times, start)
    right = bisect_right(times, end)

    count = right - left
    exact_left = bisect_left(times, center)
    exact_right = bisect_right(times, center)
    if exact_right > exact_left:
        count -= 1
    return max(count, 0)


def augment_semantic_words(
    semantic_words: list[dict[str, Any]],
    occurrences: dict[str, list[float]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for item in semantic_words:
        word = str(item.get("word", ""))
        time = item.get("time")
        normalized = [canonicalize_token(token) for token in normalize_tokens(word)]
        entry = dict(item)
        if time is None or len(normalized) != 1:
            entry["duplicate_count_1m"] = None
            entry["duplicate_count_2m"] = None
            entry["duplicate_count_3m"] = None
            augmented.append(entry)
            continue

        times = occurrences.get(normalized[0], [])
        center = float(time)
        entry["duplicate_count_1m"] = count_occurrences_in_window(
            times,
            center,
            30.0,
        )
        entry["duplicate_count_2m"] = count_occurrences_in_window(
            times,
            center,
            60.0,
        )
        entry["duplicate_count_3m"] = count_occurrences_in_window(
            times,
            center,
            90.0,
        )
        augmented.append(entry)
    return augmented


def main() -> None:
    args = build_parser().parse_args()
    if not args.semantic_json.exists():
        raise SystemExit(f"Semantic JSON not found: {args.semantic_json}")
    if not args.transcript_json.exists():
        raise SystemExit(f"Transcript JSON not found: {args.transcript_json}")

    semantic_words = read_json(args.semantic_json)
    if not isinstance(semantic_words, list):
        raise SystemExit("Semantic JSON must contain a list of word/time entries.")

    transcript = read_json(args.transcript_json)
    if not isinstance(transcript, dict):
        raise SystemExit("Transcript JSON must contain an object.")

    output = augment_semantic_words(
        semantic_words,
        transcript_occurrences(transcript),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote semantic-word duplicate counts: {args.output_json}")
    zero_count_1m = sum(1 for item in output if item.get("duplicate_count_1m") == 0)
    zero_count_2m = sum(1 for item in output if item.get("duplicate_count_2m") == 0)
    zero_count_3m = sum(1 for item in output if item.get("duplicate_count_3m") == 0)
    print(f"Unique within 1m window: {zero_count_1m}")
    print(f"Unique within 2m window: {zero_count_2m}")
    print(f"Unique within 3m window: {zero_count_3m}")


if __name__ == "__main__":
    main()
