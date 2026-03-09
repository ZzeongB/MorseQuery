"""Filter SummaryClient transcripts based on RealtimeClient similarity.

Compares transcripts from multiple SummaryClients against RealtimeClient
and keeps only those that match closely. Zero-latency immediate matching.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Optional

from logger import log_print


@dataclass
class TranscriptEntry:
    """A transcript entry with metadata."""

    timestamp: float
    text: str
    source_id: str  # "realtime", "sum0", "sum1", etc.
    speaker_id: str = ""  # "A", "B" for summary clients
    matched: bool = False  # Whether this entry has been matched


@dataclass
class FilterConfig:
    """Configuration for transcript filtering."""

    # Time window to look for matching transcripts (seconds)
    match_window_sec: float = 3.0
    # Minimum similarity ratio to consider a match (0.0 - 1.0)
    min_similarity: float = 0.6
    # How long to keep unmatched transcripts before dropping (seconds)
    ttl_sec: float = 5.0
    # Minimum word overlap ratio
    min_word_overlap: float = 0.3


class TranscriptFilter:
    """Filter SummaryClient transcripts based on RealtimeClient similarity.

    Zero-latency immediate matching:
    - When realtime arrives: immediately check pending summaries for match
    - When summary arrives: immediately check realtime buffer for match
    - Unmatched summaries are dropped after TTL
    """

    def __init__(
        self,
        session_id: str,
        config: Optional[FilterConfig] = None,
        on_filtered_transcript: Optional[Callable[[str, str, str, float], None]] = None,
    ):
        """Initialize the transcript filter.

        Args:
            session_id: Session ID for logging
            config: Filter configuration
            on_filtered_transcript: Callback for filtered transcripts
                (speaker_id, source_id, text, timestamp)
        """
        self.session_id = session_id
        self.config = config or FilterConfig()
        self.on_filtered_transcript = on_filtered_transcript

        # Buffer for realtime transcripts (reference)
        self._realtime_buffer: deque[TranscriptEntry] = deque(maxlen=100)

        # Pending summary transcripts waiting for realtime match
        self._pending_summary: list[TranscriptEntry] = []

        self._lock = threading.Lock()

        # Background TTL cleanup
        self._cleanup_running = True
        self._cleanup_thread = threading.Thread(
            target=self._ttl_cleanup_loop, daemon=True
        )
        self._cleanup_thread.start()

        log_print(
            "INFO",
            "TranscriptFilter created (zero-latency mode)",
            session_id=session_id,
            config=vars(self.config),
        )

    def add_realtime_transcript(self, text: str, timestamp: Optional[float] = None):
        """Add a transcript from RealtimeClient.

        Immediately checks pending summaries for a match.

        Args:
            text: The transcript text
            timestamp: Optional timestamp (defaults to current time)
        """
        if not text or not text.strip():
            return

        ts = timestamp or time.time()
        entry = TranscriptEntry(
            timestamp=ts,
            text=text.strip(),
            source_id="realtime",
        )

        with self._lock:
            self._realtime_buffer.append(entry)
            self._prune_old_realtime()

            # Case A: Realtime arrived - check pending summaries for match
            self._match_realtime_to_pending(entry)

    def add_summary_transcript(
        self,
        text: str,
        source_id: str,
        speaker_id: str,
        timestamp: Optional[float] = None,
    ):
        """Add a transcript from SummaryClient.

        Immediately checks realtime buffer for a match.
        If no match, stores in pending for later matching.

        Args:
            text: The transcript text
            source_id: Source identifier (e.g., "sum0", "sum1")
            speaker_id: Speaker identifier (e.g., "A", "B")
            timestamp: Optional timestamp (defaults to current time)
        """
        if not text or not text.strip():
            return

        ts = timestamp or time.time()
        entry = TranscriptEntry(
            timestamp=ts,
            text=text.strip(),
            source_id=source_id,
            speaker_id=speaker_id,
        )

        with self._lock:
            # Case B: Summary arrived - check realtime buffer for match
            matched_realtime = self._find_best_realtime_match(entry)

            if matched_realtime and not matched_realtime.matched:
                # Found a match - emit immediately
                self._emit_match(entry, matched_realtime)
                matched_realtime.matched = True
                entry.matched = True
            else:
                # No match yet - add to pending
                self._pending_summary.append(entry)
                log_print(
                    "DEBUG",
                    f"[{source_id}] No realtime match yet, pending: {text[:40]}...",
                    session_id=self.session_id,
                )

    def _match_realtime_to_pending(self, realtime_entry: TranscriptEntry):
        """Try to match a new realtime entry to pending summaries.

        Called when realtime arrives. Finds best matching pending summary.
        """
        if not self._pending_summary:
            return

        best_summary: Optional[TranscriptEntry] = None
        best_score = 0.0

        # Find candidates within time window
        candidates = []
        for summary in self._pending_summary:
            if summary.matched:
                continue
            time_diff = abs(summary.timestamp - realtime_entry.timestamp)
            if time_diff <= self.config.match_window_sec:
                candidates.append(summary)

        # Find best match among candidates
        for summary in candidates:
            score = self._calculate_similarity(summary.text, realtime_entry.text)
            time_diff = abs(summary.timestamp - realtime_entry.timestamp)
            # Time proximity bonus
            time_factor = 1.0 - (time_diff / self.config.match_window_sec) * 0.2
            score *= time_factor

            if score > best_score:
                best_score = score
                best_summary = summary

        # If best match exceeds threshold, emit
        if best_summary and best_score >= self.config.min_similarity:
            self._emit_match(best_summary, realtime_entry)
            best_summary.matched = True
            realtime_entry.matched = True

            # Remove matched summary from pending
            self._pending_summary = [
                s for s in self._pending_summary if s is not best_summary
            ]

    def _find_best_realtime_match(
        self, summary_entry: TranscriptEntry
    ) -> Optional[TranscriptEntry]:
        """Find the best matching realtime transcript for a summary.

        Returns the best match if score >= threshold, else None.
        """
        best_match: Optional[TranscriptEntry] = None
        best_score = 0.0

        for rt_entry in self._realtime_buffer:
            time_diff = abs(summary_entry.timestamp - rt_entry.timestamp)
            if time_diff > self.config.match_window_sec:
                continue

            score = self._calculate_similarity(summary_entry.text, rt_entry.text)
            # Time proximity bonus
            time_factor = 1.0 - (time_diff / self.config.match_window_sec) * 0.2
            score *= time_factor

            if score > best_score:
                best_score = score
                best_match = rt_entry

        if best_score >= self.config.min_similarity:
            return best_match
        return None

    def _emit_match(
        self, summary_entry: TranscriptEntry, realtime_entry: TranscriptEntry
    ):
        """Emit a matched transcript."""
        score = self._calculate_similarity(summary_entry.text, realtime_entry.text)
        log_print(
            "INFO",
            f"[FILTER PASS] {summary_entry.source_id} matched realtime "
            f"(score={score:.2f}): {summary_entry.text[:40]}...",
            session_id=self.session_id,
        )

        if self.on_filtered_transcript:
            # Use realtime timestamp for window slicing stability.
            # Summary transcripts can arrive late; using arrival ts causes
            # segment-window misses during parallel compression.
            self.on_filtered_transcript(
                summary_entry.speaker_id,
                summary_entry.source_id,
                summary_entry.text,
                realtime_entry.timestamp,
            )

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.

        Uses a combination of:
        1. SequenceMatcher ratio (character-level)
        2. Word overlap ratio

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not text1 or not text2:
            return 0.0

        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Character-level similarity
        char_sim = SequenceMatcher(None, t1, t2).ratio()

        # Word-level overlap
        words1 = set(t1.split())
        words2 = set(t2.split())

        if not words1 or not words2:
            return char_sim

        intersection = words1 & words2
        union = words1 | words2
        word_overlap = len(intersection) / len(union) if union else 0.0

        # Combined score (weighted average)
        combined = char_sim * 0.6 + word_overlap * 0.4

        return combined

    def _prune_old_realtime(self):
        """Remove realtime entries older than TTL + match window."""
        cutoff = time.time() - (self.config.ttl_sec + self.config.match_window_sec)
        while self._realtime_buffer and self._realtime_buffer[0].timestamp < cutoff:
            self._realtime_buffer.popleft()

    def _ttl_cleanup_loop(self):
        """Background loop to drop expired pending summaries."""
        while self._cleanup_running:
            time.sleep(1.0)  # Check every second

            with self._lock:
                now = time.time()
                expired = []
                remaining = []

                for entry in self._pending_summary:
                    if entry.matched:
                        continue
                    age = now - entry.timestamp
                    if age > self.config.ttl_sec:
                        expired.append(entry)
                    else:
                        remaining.append(entry)

                self._pending_summary = remaining

                for entry in expired:
                    log_print(
                        "INFO",
                        f"[FILTER DROP] {entry.source_id} TTL expired "
                        f"({self.config.ttl_sec}s): {entry.text[:40]}...",
                        session_id=self.session_id,
                    )

    def clear(self):
        """Clear all buffers and stop cleanup thread."""
        self._cleanup_running = False

        with self._lock:
            self._realtime_buffer.clear()
            self._pending_summary.clear()

        log_print("INFO", "TranscriptFilter cleared", session_id=self.session_id)
