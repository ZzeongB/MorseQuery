"""Transcription session management for MorseQuery Simple."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

from openai import OpenAI

from src.core.config import LOGS_DIR, OPENAI_API_KEY

# Initialize OpenAI client for GPT-based keyword extraction
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class TranscriptionSession:
    """Manages transcription state for a single user session."""

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
                self.keyword_history.append({
                    "keyword": kw["keyword"],
                    "description": kw.get("description", ""),
                    "timestamp": datetime.utcnow().isoformat(),
                })

    def get_history_keywords_list(self) -> List[str]:
        """Get list of all keywords from history."""
        return [h["keyword"] for h in self.keyword_history]

    def get_top_keyword_gpt(self, time_threshold: int = 15) -> List[Dict]:
        """Use GPT to predict keywords. Returns list of keyword-description pairs.

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
        print(f"\n[GPT] Found {len(recent_words)} words in last {time_threshold}s")
        print(f"[GPT] Context sent to GPT: '{context_text}'")

        # Build exclusion list from history
        history_keywords = self.get_history_keywords_list()
        exclusion_text = ""
        if history_keywords:
            exclusion_text = f"\n\nIMPORTANT: Do NOT suggest these keywords that were already extracted: {', '.join(history_keywords)}"
            print(f"[GPT] Excluding keywords: {history_keywords}")

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
            print(f"[GPT] Raw response: {raw_response}")

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
                    if current_keyword.lower() not in [h.lower() for h in history_keywords]:
                        keyword_description_pairs.append({
                            "keyword": current_keyword,
                            "description": description,
                        })
                    current_keyword = None

            if current_keyword and current_keyword.lower() not in [h.lower() for h in history_keywords]:
                keyword_description_pairs.append({
                    "keyword": current_keyword,
                    "description": None,
                })

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
