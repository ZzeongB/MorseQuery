"""Filter semantic words down to entries whose duplicate-count field is zero."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keep only semantic-word entries whose duplicate count is zero."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        required=True,
        help="Input semantic-word counts JSON.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON path for zero-count entries.",
    )
    parser.add_argument(
        "--count-field",
        default="duplicate_count_2m",
        help="Count field that must equal zero.",
    )
    return parser


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    data = read_json(args.input_json)
    if not isinstance(data, list):
        raise SystemExit("Input JSON must contain a list.")

    output = [item for item in data if item.get(args.count_field) == 0]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} zero-count entries to {args.output_json}")


if __name__ == "__main__":
    main()
