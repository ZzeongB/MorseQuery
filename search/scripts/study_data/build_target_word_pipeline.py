"""Build study target words from semantic words via counts -> zero -> jargon."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run counts, zero filtering, jargon extraction, and target-word generation."
    )
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words",
        help="Directory containing {id}.json semantic word files.",
    )
    parser.add_argument(
        "--transcript-dir",
        type=Path,
        default=ROOT_DIR / "data" / "transcripts",
        help="Directory containing {id}.json transcript files.",
    )
    parser.add_argument(
        "--audio-windows",
        type=Path,
        default=ROOT_DIR / "data" / "study" / "audio_windows.json",
        help="Study audio window config.",
    )
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=ROOT_DIR / "data" / "study" / "target_words",
        help="Output directory for target words.",
    )
    parser.add_argument(
        "--semantic-words2-dir",
        type=Path,
        default=ROOT_DIR / "data" / "semantic_words2",
        help="Output directory for semantic_words2.",
    )
    parser.add_argument(
        "--count-field",
        default="duplicate_count_2m",
        help="Duplicate-count field that must equal zero.",
    )
    parser.add_argument(
        "--stems",
        nargs="*",
        default=None,
        help="Optional file stems to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for target-word generation.",
    )
    parser.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model for jargon extraction.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Batch size for jargon extraction.",
    )
    parser.add_argument(
        "--context-seconds",
        type=float,
        default=12.0,
        help="Context window for jargon extraction.",
    )
    parser.add_argument(
        "--short-count",
        type=int,
        default=3,
        help="Number of short-delay interruptions per file.",
    )
    parser.add_argument(
        "--long-count",
        type=int,
        default=3,
        help="Number of long-delay interruptions per file.",
    )
    parser.add_argument(
        "--short-mean",
        type=float,
        default=15.0,
        help="Short delay Gaussian mean.",
    )
    parser.add_argument(
        "--long-mean",
        type=float,
        default=50.0,
        help="Long delay Gaussian mean.",
    )
    parser.add_argument(
        "--delay-sigma",
        type=float,
        default=5.0,
        help="Delay Gaussian sigma.",
    )
    parser.add_argument(
        "--min-search-offset-seconds",
        type=float,
        default=120.0,
        help="Minimum gap between audio_start_time and search_start_time.",
    )
    parser.add_argument(
        "--preferred-max-offset-seconds",
        type=float,
        default=300.0,
        help="Soft preference for latest search_start_time relative to audio_start_time.",
    )
    parser.add_argument(
        "--hard-max-offset-seconds",
        type=float,
        default=420.0,
        help="Hard cap for latest search_start_time relative to audio_start_time.",
    )
    parser.add_argument(
        "--semantic-words2-count-per-type",
        type=int,
        default=2,
        help="How many target words to remove per delay type when building semantic_words2.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=("gaussian_delay", "minute_slots"),
        default="gaussian_delay",
        help="How to select target words from jargon candidates.",
    )
    parser.add_argument(
        "--slot-seconds",
        default="10,50",
        help="Comma-separated second marks within each minute for minute_slots mode.",
    )
    parser.add_argument(
        "--slot-tolerance-seconds",
        type=float,
        default=5.0,
        help="Allowed +/- window around each slot second for minute_slots mode.",
    )
    parser.add_argument(
        "--search-window-seconds",
        type=float,
        default=60.0,
        help="Search/navigation window size exposed to the study UI.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write as many valid interruptions as possible instead of failing.",
    )
    return parser


def run_cmd(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run(args, check=True, cwd=ROOT_DIR)


def stem_paths(semantic_dir: Path, stems: list[str] | None) -> list[str]:
    all_stems = sorted(
        path.stem
        for path in semantic_dir.glob("*.json")
        if not path.name.endswith(".counts.json")
        and not path.name.endswith(".zero.json")
        and not path.name.endswith(".jargon.json")
    )
    if stems is None:
        return all_stems
    allowed = set(stems)
    return [stem for stem in all_stems if stem in allowed]


def main() -> None:
    args = build_parser().parse_args()
    scripts_dir = ROOT_DIR / "scripts" / "study_data"
    zero_dir = args.semantic_dir
    stems = stem_paths(args.semantic_dir, args.stems)

    for stem in stems:
        semantic_json = args.semantic_dir / f"{stem}.json"
        transcript_json = args.transcript_dir / f"{stem}.json"
        counts_json = args.semantic_dir / f"{stem}.counts.json"
        zero_json = zero_dir / f"{stem}.zero.json"
        jargon_json = args.semantic_dir / f"{stem}.zero.jargon.json"

        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "count_semantic_word_duplicates.py"),
                "--semantic-json",
                str(semantic_json),
                "--transcript-json",
                str(transcript_json),
                "--output-json",
                str(counts_json),
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "filter_zero_count_semantic_words.py"),
                "--input-json",
                str(counts_json),
                "--output-json",
                str(zero_json),
                "--count-field",
                args.count_field,
            ]
        )
        run_cmd(
            [
                sys.executable,
                str(scripts_dir / "extract_jargon_words.py"),
                "--semantic-json",
                str(zero_json),
                "--transcript-json",
                str(transcript_json),
                "--output-json",
                str(jargon_json),
                "--model",
                args.model,
                "--batch-size",
                str(args.batch_size),
                "--context-seconds",
                str(args.context_seconds),
            ]
        )

    generate_cmd = [
        sys.executable,
        str(scripts_dir / "generate_target_words.py"),
        "--keywords-dir",
        str(args.semantic_dir),
        "--keywords-suffix",
        ".zero.jargon.json",
        "--audio-windows",
        str(args.audio_windows),
        "--output-dir",
        str(args.target_dir),
        "--short-count",
        str(args.short_count),
        "--long-count",
        str(args.long_count),
        "--short-mean",
        str(args.short_mean),
        "--long-mean",
        str(args.long_mean),
        "--delay-sigma",
        str(args.delay_sigma),
        "--min-search-offset-seconds",
        str(args.min_search_offset_seconds),
        "--preferred-max-offset-seconds",
        str(args.preferred_max_offset_seconds),
        "--hard-max-offset-seconds",
        str(args.hard_max_offset_seconds),
        "--selection-mode",
        args.selection_mode,
        "--slot-seconds",
        args.slot_seconds,
        "--slot-tolerance-seconds",
        str(args.slot_tolerance_seconds),
        "--search-window-seconds",
        str(args.search_window_seconds),
    ]
    if args.seed is not None:
        generate_cmd.extend(["--seed", str(args.seed)])
    if args.allow_partial:
        generate_cmd.append("--allow-partial")
    if stems:
        generate_cmd.extend(["--stems", *stems])
    run_cmd(generate_cmd)

    semantic_words2_cmd = [
        sys.executable,
        str(scripts_dir / "build_semantic_words2.py"),
        "--semantic-dir",
        str(args.semantic_dir),
        "--target-dir",
        str(args.target_dir),
        "--output-dir",
        str(args.semantic_words2_dir),
        "--count-per-type",
        str(args.semantic_words2_count_per_type),
    ]
    if args.seed is not None:
        semantic_words2_cmd.extend(["--seed", str(args.seed)])
    if stems:
        semantic_words2_cmd.extend(["--stems", *stems])
    run_cmd(semantic_words2_cmd)


if __name__ == "__main__":
    main()
