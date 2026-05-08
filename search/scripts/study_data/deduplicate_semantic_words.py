"""Remove exact duplicate semantic-word entries by (word, time)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT_DIR / "data" / "semantic_words"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Remove exact duplicate semantic-word entries by (word, time)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="JSON files or directories to clean. Defaults to data/semantic_words.",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def should_process(path: Path) -> bool:
    if path.suffix != ".json":
        return False
    if path.name.endswith(".counts.json") or path.name.endswith(".jargon.json"):
        return False
    return True


def iter_target_files(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [DEFAULT_DIR]

    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(item for item in path.glob("*.json") if should_process(item)))
        elif should_process(path):
            files.append(path)
    return files


def dedupe_entries(data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    removed = 0
    for item in data:
        if not isinstance(item, dict):
            output.append(item)
            continue
        key = (item.get("word"), item.get("time"))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        output.append(item)
    return output, removed


def main() -> None:
    args = build_parser().parse_args()
    total_removed = 0
    for path in iter_target_files(args.paths):
        data = read_json(path)
        if not isinstance(data, list):
            continue
        deduped, removed = dedupe_entries(data)
        if removed == 0:
            continue
        path.write_text(
            json.dumps(deduped, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        total_removed += removed
        print(f"{path}: removed {removed} duplicates")

    print(f"Total duplicates removed: {total_removed}")


if __name__ == "__main__":
    main()
