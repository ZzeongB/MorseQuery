"""Transcription correction using Gemini Live as ground truth."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


@dataclass
class TranscriptionSegment:
    """A segment of transcription with metadata."""

    text: str
    timestamp: datetime
    source: str  # 'whisper' or 'gemini'
    is_corrected: bool = False
    original_text: Optional[str] = None
    similarity: float = 1.0


class CorrectionBuffer:
    """Manages buffering and alignment of Whisper and Gemini transcriptions."""

    def __init__(self, window_seconds: float = 5.0, similarity_threshold: float = 0.6):
        """Initialize correction buffer.

        Args:
            window_seconds: How far back to look for matching segments
            similarity_threshold: Minimum similarity (0.0-1.0) to consider a match
        """
        self.gemini_segments: List[TranscriptionSegment] = []
        self.correction_log: List[Dict] = []
        self.window_seconds = window_seconds
        self.similarity_threshold = similarity_threshold

    def add_gemini(self, text: str, timestamp: datetime) -> None:
        """Add Gemini transcription to buffer.

        Args:
            text: Transcription text from Gemini Live
            timestamp: When the transcription was received
        """
        if not text or not text.strip():
            return

        segment = TranscriptionSegment(
            text=text.strip(),
            timestamp=timestamp,
            source="gemini",
        )
        self.gemini_segments.append(segment)
        print(f"[Correction] Added Gemini segment: '{text[:50]}...' at {timestamp.isoformat()}")

        # Prune old segments
        self._prune_old_segments()

    def add_whisper(self, text: str, timestamp: datetime) -> TranscriptionSegment:
        """Add Whisper transcription and attempt correction.

        Args:
            text: Transcription text from Whisper
            timestamp: When the transcription was received

        Returns:
            TranscriptionSegment with correction applied if needed
        """
        if not text or not text.strip():
            return TranscriptionSegment(
                text="",
                timestamp=timestamp,
                source="whisper",
            )

        text = text.strip()

        # Find matching Gemini text
        gemini_match, similarity = self._find_best_alignment(text)

        if gemini_match and similarity < 0.9:
            # Correction needed
            segment = TranscriptionSegment(
                text=gemini_match,
                timestamp=timestamp,
                source="whisper",
                is_corrected=True,
                original_text=text,
                similarity=similarity,
            )
            print(
                f"[Correction] Corrected: '{text[:30]}...' -> '{gemini_match[:30]}...' (sim={similarity:.2f})"
            )
        else:
            # No correction needed
            segment = TranscriptionSegment(
                text=text,
                timestamp=timestamp,
                source="whisper",
                is_corrected=False,
                similarity=similarity if gemini_match else 1.0,
            )
            if gemini_match:
                print(f"[Correction] Kept original (sim={similarity:.2f}): '{text[:50]}...'")
            else:
                print(f"[Correction] No Gemini match found for: '{text[:50]}...'")

        # Log correction
        self.correction_log.append(
            {
                "timestamp": timestamp.isoformat(),
                "whisper_text": text,
                "gemini_match": gemini_match,
                "similarity": similarity if gemini_match else None,
                "was_corrected": segment.is_corrected,
                "final_text": segment.text,
            }
        )

        return segment

    def _find_best_alignment(self, whisper_text: str) -> Tuple[Optional[str], float]:
        """Find the best matching Gemini segment for a Whisper transcription.

        Strategy:
        1. Try single segment matching
        2. Try sliding window (combine 2-3 adjacent segments)

        Returns:
            (best_match_text, similarity_score)
        """
        if not self.gemini_segments:
            return None, 0.0

        # Get recent segments within window
        recent_segments = self._get_recent_segments()
        if not recent_segments:
            return None, 0.0

        segment_texts = [s.text for s in recent_segments]

        best_match = None
        best_score = 0.0

        # Strategy 1: Single segment matching
        for gemini_text in segment_texts:
            score = self._word_level_similarity(whisper_text, gemini_text)
            if score > best_score:
                best_score = score
                best_match = gemini_text

        # Strategy 2: Sliding window (combine 2-3 adjacent segments)
        for window_size in [2, 3]:
            if len(segment_texts) < window_size:
                continue
            for i in range(len(segment_texts) - window_size + 1):
                combined = " ".join(segment_texts[i : i + window_size])
                score = self._word_level_similarity(whisper_text, combined)
                if score > best_score:
                    best_score = score
                    best_match = combined

        if best_score >= self.similarity_threshold:
            return best_match, best_score
        return None, best_score

    def _word_level_similarity(self, text1: str, text2: str) -> float:
        """Calculate word-level similarity using SequenceMatcher.

        Args:
            text1: First text to compare
            text2: Second text to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        words1 = self._normalize_text(text1).split()
        words2 = self._normalize_text(text2).split()

        if not words1 or not words2:
            return 0.0

        matcher = SequenceMatcher(None, words1, words2)
        return matcher.ratio()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison.

        Args:
            text: Text to normalize

        Returns:
            Normalized text (lowercase, no punctuation, collapsed whitespace)
        """
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
        text = re.sub(r"\s+", " ", text).strip()  # Collapse whitespace
        return text

    def _get_recent_segments(self) -> List[TranscriptionSegment]:
        """Get Gemini segments within the alignment window.

        Returns:
            List of recent TranscriptionSegment objects
        """
        if not self.gemini_segments:
            return []

        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_seconds)
        return [s for s in self.gemini_segments if s.timestamp >= cutoff_time]

    def _prune_old_segments(self) -> None:
        """Remove segments older than the alignment window."""
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.window_seconds * 2)
        self.gemini_segments = [s for s in self.gemini_segments if s.timestamp >= cutoff_time]

    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections made.

        Returns:
            Dictionary with correction statistics
        """
        total = len(self.correction_log)
        corrected = sum(1 for log in self.correction_log if log["was_corrected"])
        avg_similarity = (
            sum(log["similarity"] for log in self.correction_log if log["similarity"] is not None)
            / max(1, sum(1 for log in self.correction_log if log["similarity"] is not None))
        )

        return {
            "total_segments": total,
            "corrected_segments": corrected,
            "correction_rate": corrected / max(1, total),
            "average_similarity": avg_similarity,
        }

    def clear(self) -> None:
        """Clear all buffers and logs."""
        self.gemini_segments.clear()
        self.correction_log.clear()
