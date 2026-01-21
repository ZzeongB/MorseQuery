#!/usr/bin/env python3
"""
Script to cut mp3 files based on start_time and end_time from transcript JSON files.
"""

import json
import subprocess
from pathlib import Path


def cut_mp3_file(input_mp3, output_mp3, start_time, end_time):
    """
    Cut media file (mp3/mp4) using ffmpeg.

    Args:
        input_mp3: Path to input media file
        output_mp3: Path to output media file
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    duration = end_time - start_time

    cmd = [
        "ffmpeg",
        "-i",
        str(input_mp3),
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-vcodec",
        "copy",  # Copy video codec (if present)
        "-acodec",
        "copy",  # Copy audio codec
        "-y",  # Overwrite output file if exists
        str(output_mp3),
    ]

    print(
        f"Cutting {input_mp3.name}: {start_time}s to {end_time}s (duration: {duration}s)"
    )

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"✓ Created: {output_mp3}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error cutting {input_mp3.name}: {e.stderr.decode()}")
        return False


def main():
    # Directories
    mp3_dir = Path("mp4")
    transcripts_dir = Path("data/transcripts")
    output_dir = Path("mp4/mp4_clips")

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Process each transcript file
    transcript_files = list(transcripts_dir.glob("*.json"))

    if not transcript_files:
        print("No transcript files found in data/transcripts/")
        return

    print(f"Found {len(transcript_files)} transcript files")
    print()

    success_count = 0
    failed_count = 0

    for transcript_path in transcript_files:
        # Read transcript
        with open(transcript_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_id = data.get("video_id")
        start_time = data.get("start_time")
        end_time = data.get("end_time")

        if not all([video_id, start_time is not None, end_time is not None]):
            print(
                f"✗ Skipping {transcript_path.name}: missing video_id, start_time, or end_time"
            )
            failed_count += 1
            continue

        # Check if corresponding mp3 exists
        input_mp3 = mp3_dir / f"{video_id}.mp4"
        if not input_mp3.exists():
            print(
                f"✗ Skipping {transcript_path.name}: mp3 file not found at {input_mp3}"
            )
            failed_count += 1
            continue

        # Output file name
        output_mp3 = (
            output_dir / f"{video_id}_clip_{int(start_time)}_{int(end_time)}.mp4"
        )

        # Cut the mp3
        if cut_mp3_file(input_mp3, output_mp3, start_time, end_time):
            success_count += 1
        else:
            failed_count += 1

        print()

    print("=" * 60)
    print(f"Summary: {success_count} successful, {failed_count} failed")
    print(f"Output directory: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
