"""Transcription session management for MorseQuery."""

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from google import genai
from openai import OpenAI

from src.core.config import GOOGLE_API_KEY, LOGS_DIR, OPENAI_API_KEY, SECTION_RE
from src.core.gemini_parser import _parse_terms_block
from src.core.lexicon import lexicon_dict, preprocess_word

# Initialize OpenAI client for GPT-based keyword extraction
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Gemini client for on-demand keyword extraction
gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None


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
        self.gemini_mode = "transcription"  # "transcription" or "inference"
        self.gemini_audio_queue = None
        self.gemini_text_queue = None  # For inference requests
        self.gemini_raw_output = ""
        self.gemini_parse_buffer = ""  # Buffer for incomplete sections
        self.gemini_captions = ""
        self.gemini_summary = {"overall_context": "", "current_segment": ""}
        self.gemini_terms: List[Dict] = []  # All accumulated terms
        self.gemini_terms_history: List[Dict] = []
        self.gemini_recent_terms: List[Dict] = []  # Most recently parsed terms batch
        self.gemini_recent_term_index = 0  # Navigation index for recent terms

        # Pending search state (wait for transcription before GPT call)
        self.pending_search: Dict = None  # {timestamp, search_type, client_timestamp, ...}
        self.pending_search_text_before: str = ""  # Text at spacebar moment
        self.pending_search_timeout: float = 1.0  # Seconds to wait for new transcription

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
        """Update Gemini data using buffer-based parsing.

        Buffer approach:
        1. Add new text to buffer
        2. Find section markers ([Captions], [Summary], [Terms])
        3. If 2+ markers exist, parse complete sections (between markers)
        4. Keep incomplete section (after last marker) in buffer for next time
        """
        # Keep raw_output for logging/debugging
        self.gemini_raw_output += raw_text
        self.gemini_parse_buffer += raw_text

        print(f"[Gemini Buffer] New chunk ({len(raw_text)} chars), buffer now {len(self.gemini_parse_buffer)} chars")

        # Find all section markers in buffer
        markers = list(SECTION_RE.finditer(self.gemini_parse_buffer))

        # Result for this update (only newly parsed content)
        parsed = {
            "captions": "",
            "summary": {"overall_context": "", "current_segment": ""},
            "terms": [],
        }

        if len(markers) < 2:
            # Not enough markers - keep buffering
            print(f"[Gemini Buffer] Only {len(markers)} marker(s), waiting for more...")
            return parsed

        # Parse complete sections (all except the last one which may be incomplete)
        last_parsed_pos = 0

        for i in range(len(markers) - 1):
            section_name = markers[i].group(1).lower()
            content_start = markers[i].end()
            content_end = markers[i + 1].start()
            content = self.gemini_parse_buffer[content_start:content_end].strip()
            last_parsed_pos = markers[i + 1].start()

            # Clean content (remove leading colons, etc.)
            content = re.sub(r"^[\s:]+", "", content)

            if not content:
                continue

            print(f"[Gemini Buffer] Parsing complete section [{section_name}]: '{content[:50]}...'")

            if section_name in ("caption", "captions"):
                # Append captions to session (like Whisper)
                self.add_text(content)
                if self.gemini_captions:
                    self.gemini_captions += " " + content
                else:
                    self.gemini_captions = content
                parsed["captions"] = content

            elif section_name in ("summary", "summ"):
                # Parse summary structure
                self._parse_summary_content(content, parsed["summary"])

            elif section_name in ("term", "terms"):
                # Parse terms using existing parser
                new_terms = _parse_terms_block(content)
                if new_terms:
                    parsed["terms"].extend(new_terms)
                    # Append unique terms to session
                    existing_names = {t["term"].lower() for t in self.gemini_terms}
                    for term in new_terms:
                        if term["term"].lower() not in existing_names:
                            self.gemini_terms.append(term)
                            existing_names.add(term["term"].lower())
                            print(f"[Gemini Buffer] New term: {term['term']}")

                    # Also update history
                    existing_history = {t["term"].lower() for t in self.gemini_terms_history}
                    for term in new_terms:
                        if term["term"].lower() not in existing_history:
                            self.gemini_terms_history.append(term)

                    # Update recent terms (for spacebar navigation)
                    self.gemini_recent_terms = new_terms
                    self.gemini_recent_term_index = 0
                    print(f"[Gemini Buffer] Recent terms updated: {[t['term'] for t in new_terms]}")

        # Keep only the unparsed part (from last marker onwards)
        self.gemini_parse_buffer = self.gemini_parse_buffer[last_parsed_pos:]
        print(f"[Gemini Buffer] Keeping {len(self.gemini_parse_buffer)} chars in buffer for next time")

        # Update session summary with latest parsed values
        if parsed["summary"]["overall_context"]:
            self.gemini_summary["overall_context"] = parsed["summary"]["overall_context"]
        if parsed["summary"]["current_segment"]:
            self.gemini_summary["current_segment"] = parsed["summary"]["current_segment"]

        return parsed

    def _parse_summary_content(self, content: str, summary_dict: Dict):
        """Parse summary content and extract overall_context and current_segment."""
        # Pattern for "- **Overall context**: text" or "Overall context: text"
        overall_patterns = [
            r"-?\s*\*\*\s*Overall\s*(?:context)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|$)",
            r"Overall\s*(?:context)?\s*:\s*(.+?)(?=Current|Segment|$)",
        ]
        for pattern in overall_patterns:
            m = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if m:
                summary_dict["overall_context"] = re.sub(r"\s+", " ", m.group(1)).strip()
                break

        # Pattern for "- **Current segment**: text" or "Current segment: text"
        current_patterns = [
            r"-?\s*\*\*\s*Current\s*(?:segment)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|\[|$)",
            r"Current\s*(?:segment)?\s*:\s*(.+?)(?=\[|$)",
        ]
        for pattern in current_patterns:
            m = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
            if m:
                summary_dict["current_segment"] = re.sub(r"\s+", " ", m.group(1)).strip()
                break

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

    def get_top_keyword_gemini_ondemand(self, time_threshold: int = 10) -> str:
        """Use Gemini to extract keywords on-demand when user presses spacebar.

        Args:
            time_threshold: Number of seconds of recent captions to use (default: 10)

        Returns:
            Gemini-predicted keyword that user would want to look up
        """
        if not gemini_client:
            print("[Gemini OnDemand] Gemini client not initialized, falling back")
            return self.get_top_keyword()

        # Build context from recent captions and summary
        context_parts = []

        # Get recent portion of gemini_captions (approximate last N seconds)
        if self.gemini_captions:
            # Use last ~100 words as approximation for 10 seconds
            words = self.gemini_captions.split()
            recent_words_count = min(len(words), 100)
            recent_captions = " ".join(words[-recent_words_count:])
            context_parts.append(f"Recent transcript: {recent_captions}")

        # Add summary context
        if self.gemini_summary["overall_context"]:
            context_parts.append(f"Overall context: {self.gemini_summary['overall_context']}")
        if self.gemini_summary["current_segment"]:
            context_parts.append(f"Current segment: {self.gemini_summary['current_segment']}")

        if not context_parts:
            print("[Gemini OnDemand] No context available")
            return ""

        context_text = "\n".join(context_parts)
        print(f"[Gemini OnDemand] Context:\n{context_text[:300]}...")

        prompt = f"""You are analyzing a transcript. The user wants to look up important terms.

Given the context below, identify the TOP 3 most important terms or concepts that a user would want to search for. These should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids

{context_text}

Respond with EXACTLY 3 terms in this format:
Term: <word or phrase 1 - most important>
Definition: <a brief 1-sentence explanation>
Term: <word or phrase 2>
Definition: <a brief 1-sentence explanation>
Term: <word or phrase 3>
Definition: <a brief 1-sentence explanation>"""

        try:
            response = gemini_client.models.generate_content(
                model="models/gemini-2.0-flash-exp",
                contents=prompt,
            )

            raw_response = response.text.strip()
            print(f"[Gemini OnDemand] Raw response:\n{raw_response}")

            # Parse response into term-definition pairs
            keyword_pairs = []
            current_term = None

            lines = raw_response.split("\n")
            for line in lines:
                line = line.strip()
                if line.lower().startswith("term:"):
                    current_term = line.split(":", 1)[1].strip()
                elif line.lower().startswith("definition:") and current_term:
                    definition = line.split(":", 1)[1].strip()
                    keyword_pairs.append({
                        "term": current_term,
                        "definition": definition,
                    })
                    current_term = None

            if current_term:
                keyword_pairs.append({"term": current_term, "definition": ""})

            if keyword_pairs:
                # Update recent terms for navigation
                self.gemini_recent_terms = keyword_pairs
                self.gemini_recent_term_index = 0
                predicted_keyword = keyword_pairs[0]["term"]
                print(f"[Gemini OnDemand] Extracted terms: {[t['term'] for t in keyword_pairs]}")
            else:
                predicted_keyword = raw_response.split("\n")[0] if raw_response else ""

            self._log_event(
                "keyword_extraction_gemini_ondemand",
                {
                    "context": context_text[:500],
                    "raw_response": raw_response,
                    "predicted_keyword": predicted_keyword,
                    "keyword_pairs": keyword_pairs,
                },
            )

            return predicted_keyword

        except Exception as e:
            print(f"[Gemini OnDemand] Error: {e}")
            self._log_event(
                "keyword_extraction_gemini_ondemand",
                {"context": context_text[:500], "error": str(e)},
            )
            return ""

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
                "full_gemini_response": self.gemini_raw_output,
                "events": self.event_log,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            print(f"[LOG] Session logs saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"[LOG] Error saving logs: {e}")
            return None
