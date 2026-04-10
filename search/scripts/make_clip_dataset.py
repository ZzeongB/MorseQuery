"""Cut an MP3 clip and matching SRT segment into clip_id-based outputs.

Example:
    python3 search/scripts/make_clip_dataset.py \
        --input-mp3 /path/to/source.mp3 \
        --input-srt /path/to/source.srt \
        --start 12:30 \
        --end 18:00 \
        --clip-id lecture01_clip_0750_1080 \
        --output-mp3-dir search/data/mp3 \
        --output-srt-dir search/data/srt
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2},\d{3})"
)


@dataclass
class SrtEntry:
    index: int
    start_ms: int
    end_ms: int
    text: str


def parse_cli_time(value: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS(.mmm) into seconds."""
    value = value.strip()

    if re.fullmatch(r"\d+(\.\d+)?", value):
        return float(value)

    parts = value.split(":")
    if len(parts) not in {2, 3}:
        raise argparse.ArgumentTypeError(
            f"Invalid time '{value}'. Use seconds, MM:SS, or HH:MM:SS(.mmm)."
        )

    try:
        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds

        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid time '{value}'.") from exc


def parse_srt_timestamp(value: str) -> int:
    hours, minutes, rest = value.split(":")
    seconds, millis = rest.split(",")
    return (
        int(hours) * 3600 * 1000
        + int(minutes) * 60 * 1000
        + int(seconds) * 1000
        + int(millis)
    )


def format_srt_timestamp(ms: int) -> str:
    ms = max(0, ms)
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1000
    millis = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def parse_srt(content: str) -> list[SrtEntry]:
    entries: list[SrtEntry] = []
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        if lines[0].isdigit():
            index_line = lines[0]
            timing_line = lines[1]
            text_lines = lines[2:]
        else:
            index_line = str(len(entries) + 1)
            timing_line = lines[0]
            text_lines = lines[1:]

        match = TIME_RANGE_RE.fullmatch(timing_line)
        if not match:
            continue

        entries.append(
            SrtEntry(
                index=int(index_line),
                start_ms=parse_srt_timestamp(match.group("start")),
                end_ms=parse_srt_timestamp(match.group("end")),
                text="\n".join(text_lines),
            )
        )

    return entries


def clip_srt_entries(
    entries: list[SrtEntry],
    start_ms: int,
    end_ms: int,
    shift_to_zero: bool,
) -> list[SrtEntry]:
    clipped: list[SrtEntry] = []

    for entry in entries:
        overlap_start = max(entry.start_ms, start_ms)
        overlap_end = min(entry.end_ms, end_ms)
        if overlap_end <= overlap_start:
            continue

        if shift_to_zero:
            new_start = overlap_start - start_ms
            new_end = overlap_end - start_ms
        else:
            new_start = overlap_start
            new_end = overlap_end

        clipped.append(
            SrtEntry(
                index=len(clipped) + 1,
                start_ms=new_start,
                end_ms=new_end,
                text=entry.text,
            )
        )

    return clipped


def write_srt(entries: list[SrtEntry], output_path: Path) -> None:
    blocks = []
    for entry in entries:
        blocks.append(
            "\n".join(
                [
                    str(entry.index),
                    f"{format_srt_timestamp(entry.start_ms)} --> {format_srt_timestamp(entry.end_ms)}",
                    entry.text,
                ]
            )
        )

    output_path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")


def infer_clip_id(input_mp3: Path, start_sec: float, end_sec: float) -> str:
    start_int = int(start_sec)
    end_int = int(end_sec)
    return f"{input_mp3.stem}_clip_{start_int}_{end_int}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cut an MP3 clip and matching SRT segment."
    )
    parser.add_argument("--input-mp3", type=Path, required=True)
    parser.add_argument("--input-srt", type=Path, required=True)
    parser.add_argument("--start", type=parse_cli_time, required=True)
    parser.add_argument("--end", type=parse_cli_time, required=True)
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--output-mp3-dir", type=Path, required=True)
    parser.add_argument("--output-srt-dir", type=Path, required=True)
    parser.add_argument(
        "--keep-original-srt-time",
        action="store_true",
        help="Do not shift kept subtitle timestamps to start at 00:00:00,000.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.end <= args.start:
        raise SystemExit("--end must be greater than --start.")
    if not args.input_mp3.exists():
        raise SystemExit(f"MP3 not found: {args.input_mp3}")
    if not args.input_srt.exists():
        raise SystemExit(f"SRT not found: {args.input_srt}")

    clip_id = args.clip_id or infer_clip_id(args.input_mp3, args.start, args.end)
    output_mp3_path = args.output_mp3_dir / f"{clip_id}.mp3"
    output_srt_path = args.output_srt_dir / f"{clip_id}.srt"

    args.output_mp3_dir.mkdir(parents=True, exist_ok=True)
    args.output_srt_dir.mkdir(parents=True, exist_ok=True)

    start_ms = int(args.start * 1000)
    end_ms = int(args.end * 1000)

    cut_audio(args.input_mp3, output_mp3_path, args.start, args.end)

    srt_entries = parse_srt(args.input_srt.read_text(encoding="utf-8-sig"))
    clipped_entries = clip_srt_entries(
        srt_entries,
        start_ms=start_ms,
        end_ms=end_ms,
        shift_to_zero=not args.keep_original_srt_time,
    )
    write_srt(clipped_entries, output_srt_path)

    print(f"Wrote MP3: {output_mp3_path}")
    print(f"Wrote SRT: {output_srt_path}")
    print(f"Clip ID: {clip_id}")
    print(f"Subtitle entries kept: {len(clipped_entries)}")


def cut_audio(input_mp3: Path, output_mp3: Path, start_sec: float, end_sec: float) -> None:
    duration_sec = end_sec - start_sec
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(input_mp3),
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        "-acodec",
        "copy",
        str(output_mp3),
    ]

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode == 0:
        return

    fallback_command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(input_mp3),
        "-t",
        f"{duration_sec:.3f}",
        "-vn",
        str(output_mp3),
    ]
    fallback = subprocess.run(fallback_command, capture_output=True, text=True)
    if fallback.returncode != 0:
        raise SystemExit(f"ffmpeg failed:\n{fallback.stderr or result.stderr}")


if __name__ == "__main__":
    main()
