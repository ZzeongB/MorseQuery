"""Transcribe an MP3 with openai-whisper and optionally save JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from whisper_utils import transcribe_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transcribe an audio file with openai-whisper.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the input audio file.")
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the raw Whisper response as JSON.",
    )
    parser.add_argument("--model", default="tiny", help="Whisper model name. Default: tiny")
    parser.add_argument("--language", default="en", help="Language code. Use empty string for auto.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.input.exists():
        raise SystemExit(f"Audio file not found: {args.input}")

    language = args.language or None
    result = transcribe_file(args.input, model_name=args.model, language=language)

    print(result.get("text", "").strip())

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote Whisper JSON: {args.output_json}")


if __name__ == "__main__":
    main()
