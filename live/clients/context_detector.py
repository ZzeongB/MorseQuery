"""Realtime Context Change Detector.

Uses text-embedding-3-small to compute sentence embeddings and detect context changes
by comparing aggregated previous N sentences vs current M sentences in real-time.
"""

import os
import threading
from typing import Optional

import numpy as np
from logger import log_print
from openai import OpenAI

# Configuration
LOW_THRESHOLD = 0.4  # Below this = context change (need info)
HIGH_THRESHOLD = 0.85  # Above this = high similarity (ok to interrupt)
PREV_WINDOW = 5  # Number of previous sentences to aggregate
CURR_WINDOW = 3  # Number of current sentences to aggregate
EMBEDDING_MODEL = "text-embedding-3-small"


class RealtimeContextDetector:
    """Detects context changes in real-time using sentence embeddings.

    Tracks sentences as they arrive, computes embeddings, and calculates
    similarity between previous context window and current sentence(s).
    """

    def __init__(
        self,
        low_threshold: float = LOW_THRESHOLD,
        high_threshold: float = HIGH_THRESHOLD,
        prev_window: int = PREV_WINDOW,
        curr_window: int = CURR_WINDOW,
        session_id: str = "default",
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.prev_window = prev_window
        self.curr_window = curr_window
        self.session_id = session_id

        self.sentences: list[str] = []
        self.embeddings: list[np.ndarray] = []
        self.current_similarity: float = 1.0  # Start with high similarity
        self._lock = threading.Lock()

        # OpenAI client for embeddings
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key else None

        log_print(
            "INFO",
            "RealtimeContextDetector created",
            session_id=session_id,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for a single text using OpenAI API."""
        if not self.client:
            log_print(
                "ERROR",
                "OpenAI client not initialized (missing API key)",
                session_id=self.session_id,
            )
            return None

        try:
            response = self.client.embeddings.create(
                input=[text],
                model=EMBEDDING_MODEL,
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to get embedding: {e}",
                session_id=self.session_id,
            )
            return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _aggregate_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Aggregate multiple embeddings by averaging."""
        if not embeddings:
            return np.zeros(1536)  # text-embedding-3-small dimension
        return np.mean(embeddings, axis=0)

    def add_sentence(self, sentence: str) -> dict:
        """Add a new sentence and calculate similarity with previous context.

        Args:
            sentence: The new sentence to add

        Returns:
            dict with keys:
                - similarity: float, cosine similarity value
                - is_context_change: bool, True if similarity < low_threshold
                - is_high_similarity: bool, True if similarity > high_threshold
                - sentence_count: int, total sentences so far
        """
        sentence = sentence.strip()
        if not sentence:
            return {
                "similarity": self.current_similarity,
                "is_context_change": False,
                "is_high_similarity": False,
                "sentence_count": len(self.sentences),
            }

        # Get embedding for new sentence
        log_print(
            "DEBUG",
            f"Getting embedding for sentence: {sentence[:50]}...",
            session_id=self.session_id,
        )
        embedding = self._get_embedding(sentence)
        if embedding is None:
            log_print(
                "WARN",
                "Failed to get embedding, returning current similarity",
                session_id=self.session_id,
            )
            return {
                "similarity": self.current_similarity,
                "is_context_change": False,
                "is_high_similarity": False,
                "sentence_count": len(self.sentences),
            }

        with self._lock:
            # Add sentence and embedding
            self.sentences.append(sentence)
            self.embeddings.append(embedding)

            # Need at least prev_window + curr_window sentences to compare
            if len(self.sentences) < self.prev_window + self.curr_window:
                log_print(
                    "DEBUG",
                    f"Not enough sentences for comparison: {len(self.sentences)}/{self.prev_window + self.curr_window}",
                    session_id=self.session_id,
                )
                return {
                    "similarity": self.current_similarity,
                    "is_context_change": False,
                    "is_high_similarity": False,
                    "sentence_count": len(self.sentences),
                }

            # Get previous window embeddings
            prev_start = len(self.embeddings) - self.prev_window - self.curr_window
            prev_end = len(self.embeddings) - self.curr_window
            prev_embeddings = self.embeddings[prev_start:prev_end]

            # Get current window embeddings
            curr_embeddings = self.embeddings[-self.curr_window :]

            # Aggregate and calculate similarity
            prev_agg = self._aggregate_embeddings(prev_embeddings)
            curr_agg = self._aggregate_embeddings(curr_embeddings)
            similarity = self._cosine_similarity(prev_agg, curr_agg)

            self.current_similarity = similarity

            is_context_change = similarity < self.low_threshold
            is_high_similarity = similarity > self.high_threshold

            log_print(
                "INFO",
                f"Similarity: {similarity:.4f} | prev[{prev_start}:{prev_end}] vs curr[{len(self.embeddings)-self.curr_window}:{len(self.embeddings)}] | total={len(self.sentences)}",
                session_id=self.session_id,
                is_context_change=is_context_change,
                is_high_similarity=is_high_similarity,
            )

            return {
                "similarity": similarity,
                "is_context_change": is_context_change,
                "is_high_similarity": is_high_similarity,
                "sentence_count": len(self.sentences),
            }

    def get_current_similarity(self) -> float:
        """Get the current similarity value."""
        with self._lock:
            return self.current_similarity

    def should_play_tts(self) -> bool:
        """Check if TTS should play based on current similarity.

        Returns True if:
        - similarity < low_threshold (context change, need info)
        - similarity > high_threshold (very similar, ok to interrupt)
        """
        with self._lock:
            return (
                self.current_similarity < self.low_threshold
                or self.current_similarity > self.high_threshold
            )

    def get_play_reason(self) -> Optional[str]:
        """Get the reason why TTS should play.

        Returns:
            "context_change" if similarity < low_threshold
            "high_similarity" if similarity > high_threshold
            None otherwise
        """
        with self._lock:
            if self.current_similarity < self.low_threshold:
                return "context_change"
            elif self.current_similarity > self.high_threshold:
                return "high_similarity"
            return None

    def reset(self) -> None:
        """Reset the detector state."""
        with self._lock:
            self.sentences = []
            self.embeddings = []
            self.current_similarity = 1.0

        log_print(
            "INFO",
            "RealtimeContextDetector reset",
            session_id=self.session_id,
        )

    def get_stats(self) -> dict:
        """Get current statistics."""
        with self._lock:
            return {
                "sentence_count": len(self.sentences),
                "current_similarity": self.current_similarity,
                "low_threshold": self.low_threshold,
                "high_threshold": self.high_threshold,
                "should_play": self.should_play_tts(),
            }
