"""Transcription session management for MorseQuery Simple."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

from openai import OpenAI

from src.core.config import GOOGLE_API_KEY, LOGS_DIR, OPENAI_API_KEY
from src.core.prompt_config import get_config

# Initialize OpenAI client for GPT-based keyword extraction
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Gemini client for Gemini-based keyword extraction
gemini_client = None
if GOOGLE_API_KEY:
    try:
        from google import genai

        gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
    except ImportError:
        print("[Gemini] google-genai package not installed")

GEMINI_MODEL = "gemini-2.0-flash-lite"


class TranscriptionSession:
    """Manages transcription state for a single user session."""

    # ============================================================
    # AUTO-INFERENCE CONFIGURATION OPTIONS
    # ============================================================
    # Auto-inference mode: "off", "time", "sentence"
    AUTO_INFERENCE_DEFAULT_MODE = "off"

    # Time-based auto-inference interval (seconds)
    AUTO_INFERENCE_TIME_INTERVAL = 3.0

    # Minimum words required before auto-inference triggers
    AUTO_INFERENCE_MIN_WORDS = 3

    # Cooldown between auto-inferences (seconds) to avoid spam
    AUTO_INFERENCE_COOLDOWN = 2.0
    # ============================================================

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.words: List[str] = []
        self.word_timestamps: List[datetime] = []
        self.full_text = ""
        self.sentences: List[str] = []

        # GPT keyword extraction state
        self.gpt_keyword_pairs: List[Dict] = []  # Current batch of keywords
        self.current_keyword_index = 0

        # Keyword history - track all extracted keywords to avoid duplicates
        self.keyword_history: List[Dict] = []  # All keywords ever extracted

        # OpenAI transcription state
        self.openai_active = False
        self.openai_ws = None
        self.openai_audio_queue = None
        self.last_audio_timestamp = None

        # Pending search state
        self.pending_search: Dict = None

        # Auto-inference state
        self.auto_inference_mode = (
            self.AUTO_INFERENCE_DEFAULT_MODE
        )  # "off", "time", "sentence"
        self.auto_inference_interval = self.AUTO_INFERENCE_TIME_INTERVAL
        self.last_auto_inference_time: datetime = None
        self.last_auto_inference_word_count = 0
        self.auto_inference_timer_active = False

        # Prompt configuration
        self.config_id = 1  # Default configuration

        # Context manager state
        self.context_summary = ""  # Latest context summary (1-2 sentences)
        self.context_last_updated = None  # When context was last updated
        self.context_manager_active = False  # Whether context manager is running
        self.context_update_interval = 5.0  # Seconds between context updates
        self.context_last_word_count = 0  # Words when context was last generated

        # Logging system
        self.session_start_time = datetime.utcnow()
        self.event_log: List[Dict] = []

        self._log_event(
            "session_start",
            {
                "session_id": session_id,
                "start_time": self.session_start_time.isoformat(),
            },
            save_immediately=False,
        )

    def _log_event(self, event_type: str, data: Dict, save_immediately: bool = True):
        """Internal method to log events."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self.event_log.append(event)
        print(f"[LOG] {event_type}: {data}")

        if save_immediately:
            self.save_logs_to_file()

    def add_text(self, text: str):
        """Add transcribed text to the session."""
        if text.strip():
            self.full_text += " " + text
            self.sentences.append(text)

            new_words = text.split()
            current_time = datetime.utcnow()
            for word in new_words:
                self.words.append(word)
                self.word_timestamps.append(current_time)

    def add_keywords_to_history(self, keywords: List[Dict]):
        """Add extracted keywords to history."""
        for kw in keywords:
            # Check if keyword already exists in history
            exists = any(
                h["keyword"].lower() == kw["keyword"].lower()
                for h in self.keyword_history
            )
            if not exists:
                self.keyword_history.append(
                    {
                        "keyword": kw["keyword"],
                        "description": kw.get("description", ""),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )

    def get_history_keywords_list(self) -> List[str]:
        """Get list of all keywords from history."""
        return [h["keyword"] for h in self.keyword_history]

    def get_top_keyword_gpt(
        self, time_threshold: int = 15, auto_mode: bool = False
    ) -> List[Dict]:
        """Use GPT to predict keywords. Returns list of keyword-description pairs.

        Args:
            time_threshold: How many seconds of recent words to consider
            auto_mode: If True, GPT can return 0-3 keywords (flexible). If False, exactly 3.

        Excludes keywords that were already extracted in previous calls.
        """
        if not openai_client:
            print("[GPT] OpenAI client not initialized")
            return []

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return []

        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            print(f"[GPT] No words found in last {time_threshold}s")
            return []

        context_text = " ".join(recent_words)
        print(
            f"\n[GPT] Found {len(recent_words)} words in last {time_threshold}s (auto_mode={auto_mode})"
        )
        print(f"[GPT] Context sent to GPT: '{context_text}'")

        # Build exclusion list from history
        history_keywords = self.get_history_keywords_list()
        exclusion_text = ""
        if history_keywords:
            exclusion_text = f"\n\nIMPORTANT: Do NOT suggest these keywords that were already extracted: {', '.join(history_keywords)}"
            print(f"[GPT] Excluding keywords: {history_keywords}")

        # Different prompt for auto mode vs manual mode
        if auto_mode:
            prompt = f"""You are analyzing transcripts in real-time. Extract important keywords that users might want to look up.

Given the transcript context, identify words or phrases worth looking up. The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Return 0 to 3 keywords. If there are no important keywords in this context, respond with "None".
If there are keywords, use this format for each:
Keyword: <word or phrase>
Description: <a brief 1-sentence description>{exclusion_text}

Context: {context_text}"""
        else:
            prompt = f"""You are analyzing transcripts. Users listen to content and select specific words or phrases they want to look up.

Given the transcript context, predict the top three words or phrases the user would most likely want to look up. The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Respond with EXACTLY 3 keyword-description pairs in this format:
Keyword: <word or phrase 1 - most important>
Description: <a brief 1-sentence description>
Keyword: <word or phrase 2 - second most important>
Description: <a brief 1-sentence description>
Keyword: <word or phrase 3 - third most important>
Description: <a brief 1-sentence description>{exclusion_text}

Context: {context_text}"""

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
            )

            raw_response = response.choices[0].message.content.strip()
            # print(f"[GPT] Raw response: {raw_response}")

            keyword_description_pairs = []
            current_keyword = None

            lines = raw_response.split("\n")

            for line in lines:
                line = line.strip()
                if line.lower().startswith("keyword:"):
                    current_keyword = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:") and current_keyword:
                    description = line.split(":", 1)[1].strip()
                    # Skip if keyword is in history
                    if current_keyword.lower() not in [
                        h.lower() for h in history_keywords
                    ]:
                        keyword_description_pairs.append(
                            {
                                "keyword": current_keyword,
                                "description": description,
                            }
                        )
                    current_keyword = None

            if current_keyword and current_keyword.lower() not in [
                h.lower() for h in history_keywords
            ]:
                keyword_description_pairs.append(
                    {
                        "keyword": current_keyword,
                        "description": None,
                    }
                )

            # Store current batch and add to history
            self.gpt_keyword_pairs = keyword_description_pairs
            self.current_keyword_index = 0
            self.add_keywords_to_history(keyword_description_pairs)

            self._log_event(
                "keyword_extraction_gpt",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "raw_response": raw_response,
                    "keyword_description_pairs": keyword_description_pairs,
                    "excluded_keywords": history_keywords,
                    "model": "gpt-4o-mini",
                },
            )

            return keyword_description_pairs

        except Exception as e:
            print(f"[GPT] Error calling GPT API: {e}")
            self._log_event(
                "keyword_extraction_gpt",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "error": str(e),
                },
            )
            return []

    def get_top_keyword_gemini(
        self, time_threshold: int = 15, auto_mode: bool = False
    ) -> List[Dict]:
        """Use Gemini to predict keywords. Returns list of keyword-description pairs.

        Args:
            time_threshold: How many seconds of recent words to consider
            auto_mode: If True, Gemini can return 0-3 keywords (flexible). If False, exactly 3.

        Excludes keywords that were already extracted in previous calls.
        """
        if not gemini_client:
            print("[Gemini] Gemini client not initialized")
            return []

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return []

        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            print(f"[Gemini] No words found in last {time_threshold}s")
            return []

        context_text = " ".join(recent_words)
        print(
            f"\n[Gemini] Found {len(recent_words)} words in last {time_threshold}s (auto_mode={auto_mode})"
        )
        print(f"[Gemini] Context sent to Gemini: '{context_text}'")

        # Build exclusion list from history
        history_keywords = self.get_history_keywords_list()
        exclusion_text = ""
        if history_keywords:
            exclusion_text = f"\n\nIMPORTANT: Do NOT suggest these keywords that were already extracted: {', '.join(history_keywords)}"
            print(f"[Gemini] Excluding keywords: {history_keywords}")

        # Build conversation context summary if available
        context_summary_text = ""
        if self.context_summary:
            context_summary_text = f"\n\nConversation topic: {self.context_summary}"
            print(f"[Gemini] Including context summary: {self.context_summary}")

        # Different prompt for auto mode vs manual mode
        if auto_mode:
            prompt = f"""You are analyzing transcripts in real-time. Extract important keywords that users might want to look up.{context_summary_text}

Given the transcript context, identify words or phrases worth looking up. The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Return 0 to 3 keywords. If there are no important keywords in this context, respond with "None".
If there are keywords, use this format for each:
Keyword: <word or phrase>
Description: <what user needs to know - e.g. acronym expansion, brief definition, or key fact>{exclusion_text}

Context: {context_text}"""
        else:
            prompt = f"""You are analyzing transcripts. Users listen to content and select specific words or phrases they want to look up.{context_summary_text}

Given the transcript context, predict the top three words or phrases the user would most likely want to look up. The selected words or phrases should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

Respond with EXACTLY 3 keyword-description pairs in this format:
Keyword: <word or phrase 1 - most important>
Description: <what user needs to know - e.g. acronym expansion, brief definition, or key fact>
Keyword: <word or phrase 2 - second most important>
Description: <what user needs to know - e.g. acronym expansion, brief definition, or key fact>
Keyword: <word or phrase 3 - third most important>
Description: <what user needs to know - e.g. acronym expansion, brief definition, or key fact>{exclusion_text}

Context: {context_text}"""

        try:
            from google.genai import types

            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200,
                ),
            )

            raw_response = response.text.strip()
            print(f"[Gemini] Raw response: {raw_response}")

            keyword_description_pairs = []
            current_keyword = None

            lines = raw_response.split("\n")

            for line in lines:
                line = line.strip()
                if line.lower().startswith("keyword:"):
                    current_keyword = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:") and current_keyword:
                    description = line.split(":", 1)[1].strip()
                    # Skip if keyword is in history
                    if current_keyword.lower() not in [
                        h.lower() for h in history_keywords
                    ]:
                        keyword_description_pairs.append(
                            {
                                "keyword": current_keyword,
                                "description": description,
                            }
                        )
                    current_keyword = None

            if current_keyword and current_keyword.lower() not in [
                h.lower() for h in history_keywords
            ]:
                keyword_description_pairs.append(
                    {
                        "keyword": current_keyword,
                        "description": None,
                    }
                )

            # Store current batch and add to history
            self.gpt_keyword_pairs = keyword_description_pairs
            self.current_keyword_index = 0
            self.add_keywords_to_history(keyword_description_pairs)

            self._log_event(
                "keyword_extraction_gemini",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "context_summary": self.context_summary,
                    "raw_response": raw_response,
                    "keyword_description_pairs": keyword_description_pairs,
                    "excluded_keywords": history_keywords,
                    "model": GEMINI_MODEL,
                },
            )

            return keyword_description_pairs

        except Exception as e:
            print(f"[Gemini] Error calling Gemini API: {e}")
            self._log_event(
                "keyword_extraction_gemini",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "error": str(e),
                },
            )
            return []

    def log_search_action(
        self,
        search_mode: str,
        search_type: str,
        keyword: str = None,
        num_results: int = 0,
    ):
        """Log when user performs a search (spacebar action)."""
        self._log_event(
            "search_action",
            {
                "search_mode": search_mode,
                "search_type": search_type,
                "keyword": keyword,
                "num_results": num_results,
                "total_words_at_search": len(self.words),
                "transcription_length": len(self.full_text),
            },
        )

    def should_auto_inference(self) -> bool:
        """Check if auto-inference should trigger based on current mode and state."""
        if self.auto_inference_mode == "off":
            return False

        # Check minimum word count since last inference
        new_words = len(self.words) - self.last_auto_inference_word_count
        if new_words < self.AUTO_INFERENCE_MIN_WORDS:
            return False

        # Check cooldown
        if self.last_auto_inference_time:
            elapsed = (
                datetime.utcnow() - self.last_auto_inference_time
            ).total_seconds()
            if elapsed < self.AUTO_INFERENCE_COOLDOWN:
                return False

        return True

    def mark_auto_inference_done(self):
        """Mark that auto-inference was just performed."""
        self.last_auto_inference_time = datetime.utcnow()
        self.last_auto_inference_word_count = len(self.words)

    def set_auto_inference_mode(self, mode: str, interval: float = None):
        """Set auto-inference mode and optionally the interval."""
        if mode in ("off", "time", "sentence"):
            self.auto_inference_mode = mode
            if interval is not None and interval > 0:
                self.auto_inference_interval = interval
            self._log_event(
                "auto_inference_mode_changed",
                {
                    "mode": mode,
                    "interval": self.auto_inference_interval,
                },
            )

    def set_config(self, config_id: int):
        """Set the prompt configuration."""
        if config_id in range(1, 7):
            self.config_id = config_id
            self._log_event(
                "config_changed",
                {
                    "config_id": config_id,
                    "config": get_config(config_id),
                },
            )

    def get_current_config(self) -> dict:
        """Get the current prompt configuration."""
        return get_config(self.config_id)

    def save_logs_to_file(self) -> str | None:
        """Save session logs to a JSON file."""
        try:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
            filepath = os.path.join(LOGS_DIR, filename)

            log_data = {
                "session_id": self.session_id,
                "session_start": self.session_start_time.isoformat(),
                "session_end": datetime.utcnow().isoformat(),
                "total_words": len(self.words),
                "total_events": len(self.event_log),
                "full_transcription": self.full_text,
                "keyword_history": self.keyword_history,
                "events": self.event_log,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            print(f"[LOG] Session logs saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"[LOG] Error saving logs: {e}")
            return None
