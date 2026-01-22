"""Transcription session management for MorseQuery."""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from openai import OpenAI

from src.core.config import LOGS_DIR, OPENAI_API_KEY
from src.core.gemini_parser import parse_gemini_output
from src.core.lexicon import lexicon_dict, preprocess_word

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

        # Real-time transcription state (like transcribe_demo.py)
        self.phrase_bytes = b""
        self.phrase_time = None
        self.transcription_lines = [""]

        # GPT keyword extraction state (for double-spacebar navigation)
        self.gpt_keyword_pairs: List[Dict] = []
        self.current_keyword_index = 0

        # Gemini Live API state
        self.gemini_active = False
        self.gemini_session = None
        self.gemini_audio_queue = None
        self.gemini_raw_output = ""
        self.gemini_captions = ""
        self.gemini_summary = {"overall_context": "", "current_segment": ""}
        self.gemini_terms: List[Dict] = []
        self.gemini_terms_history: List[Dict] = []

        # Logging system
        self.session_start_time = datetime.utcnow()
        self.event_log: List[Dict] = []

        # Log session start (don't save immediately during init)
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

    def update_gemini_data(self, raw_text: str) -> Dict:
        """Update Gemini data from raw model output."""
        self.gemini_raw_output += raw_text

        print(
            f"[Gemini DEBUG] Raw text chunk ({len(raw_text)} chars): {raw_text[:200]}..."
        )
        print(f"[Gemini DEBUG] Total accumulated ({len(self.gemini_raw_output)} chars)")

        parsed = parse_gemini_output(self.gemini_raw_output)

        print(
            f"[Gemini DEBUG] Parsed captions: {parsed['captions'][:100] if parsed['captions'] else 'EMPTY'}..."
        )
        print(f"[Gemini DEBUG] Parsed summary: {parsed['summary']}")
        print(
            f"[Gemini DEBUG] Parsed terms ({len(parsed['terms'])}): {parsed['terms']}"
        )

        if parsed["captions"]:
            self.gemini_captions = parsed["captions"]

        if parsed["summary"]["overall_context"]:
            self.gemini_summary["overall_context"] = parsed["summary"][
                "overall_context"
            ]
        if parsed["summary"]["current_segment"]:
            self.gemini_summary["current_segment"] = parsed["summary"][
                "current_segment"
            ]

        if parsed["terms"]:
            self.gemini_terms = parsed["terms"]
            existing_term_names = {t["term"].lower() for t in self.gemini_terms_history}
            for term in parsed["terms"]:
                if term["term"].lower() not in existing_term_names:
                    self.gemini_terms_history.append(term)
                    existing_term_names.add(term["term"].lower())
                    print(f"[Gemini DEBUG] New term added: {term['term']}")

        return parsed

    def get_gemini_terms_for_search(self) -> List[Dict]:
        """Get Gemini terms formatted for search (like GPT keyword pairs)."""
        return [
            {"keyword": t["term"], "description": t["definition"]}
            for t in self.gemini_terms
        ]

    def get_top_keyword(self, context_window: int = 20) -> str:
        """Extract most relevant keyword using OpenLexicon frequency filtering.

        Args:
            context_window: Number of recent words to consider (default: 20)

        Returns:
            The most important keyword (lowest frequency or not in lexicon)
        """
        if len(self.words) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        recent_words = (
            self.words[-context_window:]
            if len(self.words) > context_window
            else self.words
        )

        candidate_words = []

        for word in recent_words:
            cleaned = preprocess_word(word)
            if not cleaned:
                continue

            freq_value = lexicon_dict.get(cleaned)
            print(f"[Lexicon] '{word}' -> preprocessed: '{cleaned}'")

            if freq_value is None:
                candidate_words.append((cleaned, -1, word))
                print(f"[Lexicon] '{cleaned}' not in lexicon -> candidate")
            elif pd.isna(freq_value):
                candidate_words.append((cleaned, 0, word))
                print(f"[Lexicon] '{cleaned}' has NaN frequency -> candidate")
            elif freq_value < 3.0:
                candidate_words.append((cleaned, freq_value, word))
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} < 3.0 -> candidate")
            else:
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} >= 3.0 -> skipped")

        if not candidate_words:
            fallback_word = self.words[-1] if self.words else ""
            print(
                f"[Lexicon] No important words found (all freq >= 3.0), using fallback: '{fallback_word}'"
            )
            return fallback_word

        candidate_words.sort(key=lambda x: x[1])
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        print(f"\n[Lexicon] Selected keyword: '{selected_word}' (freq={selected_freq})")
        print(
            f"[Lexicon] All candidates: {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        return selected_word

    def get_top_keyword_with_time_threshold(self, time_threshold: int = 5) -> str:
        """Extract most important keyword from words within the last N seconds.

        Args:
            time_threshold: Number of seconds to look back (default: 5)

        Returns:
            The most important keyword (lowest frequency) within the time window
        """
        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            print(
                f"[Recent] No words found in last {time_threshold}s, using last word as fallback"
            )
            return self.words[-1] if self.words else ""

        print(
            f"\n[Recent] Found {len(recent_words)} words in last {time_threshold}s: {recent_words}"
        )

        candidate_words = []

        for word in recent_words:
            cleaned = preprocess_word(word)
            if not cleaned:
                continue

            freq_value = lexicon_dict.get(cleaned)
            print(f"[Recent] '{word}' -> preprocessed: '{cleaned}'")

            if freq_value is None:
                candidate_words.append((cleaned, -1, word))
                print(
                    f"[Recent] '{cleaned}' not in lexicon (highest importance) -> candidate"
                )
            elif pd.isna(freq_value):
                candidate_words.append((cleaned, 0, word))
                print(
                    f"[Recent] '{cleaned}' has NaN frequency (very high importance) -> candidate"
                )
            elif freq_value < 3.0:
                candidate_words.append((cleaned, freq_value, word))
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} < 3.0 (high importance) -> candidate"
                )
            else:
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} >= 3.0 (low importance) -> skipped"
                )

        if not candidate_words:
            fallback_word = recent_words[-1] if recent_words else ""
            print(
                f"[Recent] No important words found (all freq >= 3.0) in last {time_threshold}s"
            )
            print(f"[Recent] Using fallback: '{fallback_word}'")

            self._log_event(
                "keyword_extraction_recent",
                {
                    "time_threshold_seconds": time_threshold,
                    "recent_words": recent_words,
                    "candidates": [],
                    "selected_keyword": fallback_word,
                    "selected_frequency": None,
                    "selection_reason": "fallback_no_important_words",
                    "is_fallback": True,
                },
            )

            return fallback_word

        candidate_words.sort(key=lambda x: x[1])
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        freq_label = (
            "not in lexicon" if selected_freq == -1 else f"freq={selected_freq}"
        )

        print(
            f"\n[Recent] Selected HIGHEST importance keyword: '{selected_word}' ({freq_label}) from last {time_threshold}s"
        )
        print(
            f"[Recent] All candidates (sorted by importance): {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        self._log_event(
            "keyword_extraction_recent",
            {
                "time_threshold_seconds": time_threshold,
                "recent_words": recent_words,
                "candidates": [
                    {
                        "word": word,
                        "cleaned": cleaned,
                        "frequency": freq,
                        "importance_rank": idx + 1,
                    }
                    for idx, (cleaned, freq, word) in enumerate(candidate_words)
                ],
                "selected_keyword": selected_word,
                "selected_frequency": selected_freq,
                "selection_reason": "highest_importance",
                "is_fallback": False,
            },
        )

        return selected_word

    def get_top_keyword_gpt(self, time_threshold: int = 10) -> str:
        """Use GPT to predict which word the user would want to look up.

        Args:
            time_threshold: Number of seconds to look back for context (default: 10)

        Returns:
            GPT-predicted keyword that user would want to look up
        """
        if not openai_client:
            print("[GPT] OpenAI client not initialized, falling back to lexicon method")
            return self.get_top_keyword_with_time_threshold(time_threshold)

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        print(f"\n[GPT DEBUG] Current time: {current_time.isoformat()}")
        print(
            f"[GPT DEBUG] Threshold time: {threshold_time.isoformat()} ({time_threshold}s ago)"
        )
        print(f"[GPT DEBUG] Total words in session: {len(self.words)}")

        print("[GPT DEBUG] Last 10 words with timestamps:")
        for i in range(max(0, len(self.words) - 10), len(self.words)):
            word = self.words[i]
            timestamp = self.word_timestamps[i]
            time_diff = (current_time - timestamp).total_seconds()
            in_range = "V" if timestamp >= threshold_time else "X"
            print(
                f"  {in_range} [{i}] '{word}' at {timestamp.isoformat()} ({time_diff:.2f}s ago)"
            )

        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            print(
                f"[GPT] No words found in last {time_threshold}s, using last word as fallback"
            )
            return self.words[-1] if self.words else ""

        context_text = " ".join(recent_words)
        print(f"\n[GPT] Found {len(recent_words)} words in last {time_threshold}s")
        print(f"[GPT] Context sent to GPT: '{context_text}'")

        prompt = (
            """You are analyzing transcripts. Users listen to content and select specific words or phrases they want to look up.

Given the transcript context, predict the TOP 3 words/phrases the user would want to look up, ranked by importance (most important first). The selected words should be:
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
Description: <a brief 1-sentence description>

Few-shot examples:

Example 1:
Context: really are related to black holes. Like, I get paid for this,
Keyword: black holes
Description: Regions in space where gravity is so strong that nothing, not even light, can escape.
Keyword: gravity
Description: The force that attracts objects toward each other, especially the Earth's pull on objects.
Keyword: paid
Description: Receiving money in exchange for work or services.

Example 2:
Context: goes by way of the thalamus on route to the cortex.
Keyword: cortex
Description: The outer layer of the brain responsible for higher cognitive functions like thinking and memory.
Keyword: thalamus
Description: A brain region that relays sensory and motor signals to the cerebral cortex.
Keyword: route
Description: A path or way taken to reach a destination.

Example 3:
Context: it senses this big glucose spike, it calls your pancreas and it's like,
Keyword: pancreas
Description: An organ that produces insulin and digestive enzymes, regulating blood sugar levels.
Keyword: glucose spike
Description: A rapid increase in blood sugar levels, often after eating carbohydrates.
Keyword: insulin
Description: A hormone that regulates blood sugar by helping cells absorb glucose.

Now predict for this new context:

Context: """
            + context_text
        )

        try:
            gpt_start_time = datetime.utcnow()

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )

            gpt_end_time = datetime.utcnow()
            gpt_duration = (gpt_end_time - gpt_start_time).total_seconds()

            raw_response = response.choices[0].message.content.strip()

            keyword_description_pairs = []
            current_keyword = None

            lines = raw_response.split("\n")

            for line in lines:
                line = line.strip()
                if line.lower().startswith("keyword:"):
                    current_keyword = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:") and current_keyword:
                    description = line.split(":", 1)[1].strip()
                    keyword_description_pairs.append(
                        {
                            "keyword": current_keyword,
                            "description": description,
                        }
                    )
                    current_keyword = None

            if current_keyword:
                keyword_description_pairs.append(
                    {
                        "keyword": current_keyword,
                        "description": None,
                    }
                )

            if keyword_description_pairs:
                predicted_keyword = keyword_description_pairs[0]["keyword"]
            else:
                predicted_keyword = raw_response

            self.gpt_keyword_pairs = keyword_description_pairs
            self.current_keyword_index = 0

            self._log_event(
                "keyword_extraction_gpt",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "raw_response": raw_response,
                    "predicted_keyword": predicted_keyword,
                    "keyword_description_pairs": keyword_description_pairs,
                    "model": "gpt-4o-mini",
                    "temperature": 0.3,
                },
            )

            return predicted_keyword

        except Exception as e:
            print(f"[GPT] Error calling GPT API: {e}")
            fallback = recent_words[-1] if recent_words else ""
            print(f"[GPT] Using fallback: {fallback}")

            self._log_event(
                "keyword_extraction_gpt",
                {
                    "time_threshold_seconds": time_threshold,
                    "context": context_text,
                    "error": str(e),
                    "fallback_keyword": fallback,
                },
            )

            return fallback

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
                "events": self.event_log,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            print(f"[LOG] Session logs saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"[LOG] Error saving logs: {e}")
            return None
