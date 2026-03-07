"""Thread-safe store for multi-speaker dialogue entries."""

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class DialogueEntry:
    """A single dialogue entry with timestamp and speaker."""

    timestamp: float
    speaker_id: str  # "A" or "B"
    text: str
    source_id: Optional[str] = None  # e.g. "sum0", "sum1"


class DialogueStore:
    """Thread-safe store for multi-speaker dialogue.

    Stores timestamped dialogue entries from multiple speakers (A, B)
    and provides methods to retrieve them in chronological order.
    """

    def __init__(self) -> None:
        self._entries: list[DialogueEntry] = []
        self._lock = threading.Lock()

    def add_entry(
        self,
        speaker_id: str,
        text: str,
        timestamp: Optional[float] = None,
        source_id: Optional[str] = None,
    ) -> DialogueEntry:
        """Add a dialogue entry.

        Args:
            speaker_id: Speaker identifier ("A" or "B")
            text: The spoken text
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            The created DialogueEntry
        """
        if timestamp is None:
            timestamp = time.time()

        entry = DialogueEntry(
            timestamp=timestamp,
            speaker_id=speaker_id.upper(),
            text=text.strip(),
            source_id=source_id,
        )

        with self._lock:
            self._entries.append(entry)

        return entry

    def get_dialogue_chronological(self) -> list[DialogueEntry]:
        """Get all dialogue entries sorted by timestamp.

        Returns:
            List of DialogueEntry objects sorted chronologically
        """
        with self._lock:
            return sorted(self._entries, key=lambda e: e.timestamp)

    def get_formatted_dialogue(self) -> str:
        """Get dialogue formatted as "A: ...\nB: ..." string.

        Returns:
            Formatted dialogue string with speaker labels
        """
        entries = self.get_dialogue_chronological()
        if not entries:
            return ""

        lines = [f"{e.speaker_id}: {e.text}" for e in entries]
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored entries."""
        with self._lock:
            self._entries.clear()

    def get_entry_count(self) -> int:
        """Get the number of stored entries.

        Returns:
            Number of dialogue entries
        """
        with self._lock:
            return len(self._entries)

    def get_entries_since(self, timestamp: float) -> list[DialogueEntry]:
        """Get entries since a given timestamp.

        Args:
            timestamp: Unix timestamp threshold

        Returns:
            List of entries after the given timestamp
        """
        with self._lock:
            return sorted(
                [e for e in self._entries if e.timestamp >= timestamp],
                key=lambda e: e.timestamp,
            )

    def get_entries_between(self, start_ts: float, end_ts: float) -> list[DialogueEntry]:
        """Get entries in a closed time window [start_ts, end_ts]."""
        lo = min(start_ts, end_ts)
        hi = max(start_ts, end_ts)
        with self._lock:
            return sorted(
                [e for e in self._entries if lo <= e.timestamp <= hi],
                key=lambda e: e.timestamp,
            )

    def prune_before(self, timestamp: float) -> int:
        """Drop entries older than timestamp. Returns number of removed entries."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.timestamp >= timestamp]
            return max(0, before - len(self._entries))

    def get_formatted_dialogue_since(self, timestamp: float) -> str:
        """Get formatted dialogue since a given timestamp.

        Args:
            timestamp: Unix timestamp threshold

        Returns:
            Formatted dialogue string
        """
        entries = self.get_entries_since(timestamp)
        if not entries:
            return ""

        lines = [f"{e.speaker_id}: {e.text}" for e in entries]
        return "\n".join(lines)
