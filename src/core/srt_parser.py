"""SRT file parsing utilities for MorseQuery."""

import re
from typing import List, Tuple


def parse_srt(srt_content: str) -> List[Tuple[int, int, str]]:
    """Parse SRT file content and return list of (start_time_ms, end_time_ms, text) tuples.

    Args:
        srt_content: Raw SRT file content as string

    Returns:
        List of tuples containing (start_ms, end_ms, text) for each subtitle entry
    """
    entries = []
    blocks = re.split(r"\n\n+", srt_content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) >= 2:
            # Find timestamp line (contains -->)
            timestamp_line = None
            text_start_idx = 0
            for i, line in enumerate(lines):
                if "-->" in line:
                    timestamp_line = line
                    text_start_idx = i + 1
                    break

            if timestamp_line:
                # Parse timestamp: 00:00:11,040 --> 00:00:14,337
                match = re.match(
                    r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)",
                    timestamp_line,
                )
                if match:
                    h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, match.groups())
                    start_ms = h1 * 3600000 + m1 * 60000 + s1 * 1000 + ms1
                    end_ms = h2 * 3600000 + m2 * 60000 + s2 * 1000 + ms2
                    text = " ".join(lines[text_start_idx:])
                    entries.append((start_ms, end_ms, text))

    return entries


def format_timestamp(ms: int) -> str:
    """Convert milliseconds to SRT timestamp format (HH:MM:SS,mmm).

    Args:
        ms: Time in milliseconds

    Returns:
        Formatted timestamp string
    """
    hours = ms // 3600000
    minutes = (ms % 3600000) // 60000
    seconds = (ms % 60000) // 1000
    milliseconds = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def find_subtitle_at_time(
    entries: List[Tuple[int, int, str]], time_ms: int
) -> Tuple[int, int, str] | None:
    """Find the subtitle entry that should be displayed at a given time.

    Args:
        entries: List of SRT entries from parse_srt()
        time_ms: Current playback time in milliseconds

    Returns:
        The matching entry tuple or None if no subtitle at this time
    """
    for start_ms, end_ms, text in entries:
        if start_ms <= time_ms <= end_ms:
            return (start_ms, end_ms, text)
    return None
