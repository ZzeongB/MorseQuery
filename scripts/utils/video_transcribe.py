#!/usr/bin/env python3
"""
Video Transcription with Timestamps using Whisper

Usage:
    python video_transcribe.py <video_file_path> [--output output.json] [--model base]

Example:
    python video_transcribe.py video.mp4
    python video_transcribe.py video.mp4 --output result.json --model medium
"""

import argparse
import json
import sys
from pathlib import Path

import whisper


def format_timestamp(seconds):
    """Convert seconds to HH:MM:SS.mmm format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def transcribe_video(video_path, model_name="base", output_file=None):
    """
    Transcribe video file with timestamps

    Args:
        video_path: Path to video file
        model_name: Whisper model size (tiny, base, small, medium, large)
        output_file: Optional path to save JSON output

    Returns:
        Dictionary with transcription results
    """
    # Check if file exists
    if not Path(video_path).exists():
        print(f"Error: File not found: {video_path}")
        sys.exit(1)

    print(f"Loading Whisper model: {model_name}...")
    model = whisper.load_model(model_name)

    print(f"Transcribing video: {video_path}")
    print("This may take a few minutes depending on video length...\n")

    # Transcribe with word-level timestamps
    result = model.transcribe(
        video_path,
        word_timestamps=True,
        verbose=False
    )

    # Print results
    print("=" * 80)
    print("TRANSCRIPTION WITH TIMESTAMPS")
    print("=" * 80)
    print()

    # Print full text
    print("Full Transcription:")
    print("-" * 80)
    print(result["text"])
    print()

    # Print segment-level timestamps
    print("\nSegment-level Timestamps:")
    print("-" * 80)
    for i, segment in enumerate(result["segments"], 1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        print(f"[{start_time} --> {end_time}] {text}")

    # Print word-level timestamps if available
    print("\n\nWord-level Timestamps:")
    print("-" * 80)
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                start = format_timestamp(word_info["start"])
                end = format_timestamp(word_info["end"])
                word = word_info["word"]
                print(f"[{start} --> {end}] {word}")

    # Prepare output data
    output_data = {
        "video_file": str(video_path),
        "model": model_name,
        "language": result.get("language", "unknown"),
        "full_text": result["text"],
        "segments": [
            {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "start_formatted": format_timestamp(segment["start"]),
                "end_formatted": format_timestamp(segment["end"]),
                "text": segment["text"].strip(),
                "words": [
                    {
                        "word": w["word"],
                        "start": w["start"],
                        "end": w["end"],
                        "start_formatted": format_timestamp(w["start"]),
                        "end_formatted": format_timestamp(w["end"])
                    }
                    for w in segment.get("words", [])
                ] if "words" in segment else []
            }
            for segment in result["segments"]
        ]
    }

    # Save to JSON if output file specified
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n\nResults saved to: {output_file}")

    print("\n" + "=" * 80)
    print(f"Total duration: {format_timestamp(result['segments'][-1]['end'])}")
    print(f"Total segments: {len(result['segments'])}")
    print("=" * 80)

    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe video with timestamps using Whisper"
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to video file (mp4, avi, mov, mkv, etc.)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file path (optional)"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)"
    )

    args = parser.parse_args()

    # Default output filename if not specified
    output_file = args.output
    if output_file is None:
        video_name = Path(args.video_path).stem
        output_file = f"{video_name}_transcript.json"

    transcribe_video(args.video_path, args.model, output_file)


if __name__ == "__main__":
    main()
