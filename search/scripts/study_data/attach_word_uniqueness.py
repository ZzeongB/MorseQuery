"""Attach uniqueness scores to word/time JSON entries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_UNIQUENESS_SCORE = 7.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attach uniqueness scores to JSON entries with a word field."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--word-uniqueness-analysis",
        type=Path,
        default=ROOT_DIR / "data" / "analysis" / "word_uniqueness_analysis.json",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_uniqueness_lookup(path: Path) -> dict[str, float]:
    data = read_json(path)
    if not isinstance(data, dict):
        return {}

    analysis_results = data.get("analysis_results", [])
    if not isinstance(analysis_results, list):
        return {}

    lookup: dict[str, float] = {}
    for result in analysis_results:
        if not isinstance(result, dict):
            continue
        word_data = result.get("word_data")
        if not isinstance(word_data, list):
            continue
        for item in word_data:
            if not isinstance(item, dict):
                continue
            word = item.get("word")
            uniqueness = item.get("uniqueness")
            if not isinstance(word, str) or not isinstance(uniqueness, (int, float)):
                continue
            lookup.setdefault(word.lower(), float(uniqueness))
    return lookup


def main() -> None:
    args = build_parser().parse_args()
    data = read_json(args.input_json)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must contain a list.")

    lookup = load_uniqueness_lookup(args.word_uniqueness_analysis)
    output = []
    for item in data:
        if not isinstance(item, dict):
            continue
        entry = dict(item)
        word = entry.get("word")
        if isinstance(word, str):
            entry["uniqueness"] = lookup.get(word.lower(), DEFAULT_UNIQUENESS_SCORE)
        else:
            entry["uniqueness"] = DEFAULT_UNIQUENESS_SCORE
        output.append(entry)

    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote uniqueness scores to {args.output_json}")


if __name__ == "__main__":
    main()
