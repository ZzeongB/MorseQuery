import asyncio
import base64
import json
import os
import re
import tempfile
import threading
import warnings
from datetime import datetime
from typing import Dict, List

import nltk
import numpy as np
import pandas as pd
import requests
import whisper
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from google import genai
from nltk.stem import WordNetLemmatizer
from openai import OpenAI
from pydub import AudioSegment

# Ignore FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize NLTK lemmatizer
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("[NLTK] Downloading WordNet data...")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Common English contractions
CONTRACTIONS = {
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "who's": "who is",
    "where's": "where is",
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "morsequery-secret-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,
    ping_interval=25,
    max_http_buffer_size=100000000,
)

# Initialize Whisper model (base model for speed)
whisper_model = None

# Google Custom Search API config
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

# Initialize OpenAI client for GPT-based keyword extraction
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Gemini client for Live API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = None
if GOOGLE_API_KEY:
    gemini_client = genai.Client(
        http_options={"api_version": "v1beta"}, api_key=GOOGLE_API_KEY
    )
    print("[Gemini] Client initialized")
else:
    print("[Gemini] Warning: GOOGLE_API_KEY not set, Gemini Live will not work")

# Gemini Live API configuration
GEMINI_MODEL = "models/gemini-2.0-flash-exp"
GEMINI_SEND_SAMPLE_RATE = 16000
GEMINI_CHUNK_SIZE = 1024

GEMINI_STUDY_CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": """You are a real-time learning assistant. Listen to the audio and perform the following tasks:

1. **[Captions]**: Transcribe what you hear in real time (verbatim, in the original language).
2. **[Summary]**: Continuously update the summary with TWO parts:
   - **Overall context**: a brief one-line summary of everything understood so far.
   - **Current segment**: a one-line summary of what is being discussed right now.
   Update both as new information comes in.

3. **[Terms]**: When technical terms, difficult words, or important concepts appear, briefly explain them.
   Always provide at least THREE terms when possible.

Output format:
[Captions] Transcribed content...
[Summary] Key point summary...
[Terms] Term: explanation...

Be concise and respond in real time.""",
}

# Gemini output parsing helpers - robust version for streaming
# Multiple patterns to match section headers (Gemini output varies)
SECTION_PATTERNS = [
    # Standard format: [Captions], [Summary], [Terms]
    re.compile(r"\[\s*(Captions?|Caption)\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*(Summary|Summ)\s*\]", re.IGNORECASE),
    re.compile(r"\[\s*(Terms?|Term)\s*\]", re.IGNORECASE),
    # Alternative format: **Captions**, **Summary**, **Terms**
    re.compile(r"\*\*\s*(Captions?|Caption)\s*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\s*(Summary|Summ)\s*\*\*", re.IGNORECASE),
    re.compile(r"\*\*\s*(Terms?|Term)\s*\*\*", re.IGNORECASE),
    # Plain format: Captions:, Summary:, Terms:
    re.compile(r"^(Captions?|Caption)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(Summary|Summ)\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(Terms?|Term)\s*:", re.IGNORECASE | re.MULTILINE),
]

# Combined pattern to find any section header
SECTION_RE = re.compile(
    r"(?:\[|\*\*|^)\s*(Captions?|Caption|Summary|Summ|Terms?|Term)\s*(?:\]|\*\*|:)",
    re.IGNORECASE | re.MULTILINE,
)


def _norm_ws(s: str) -> str:
    """Collapse weird newlines/spaces into readable single spaces."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return s


def _squash_inline(s: str) -> str:
    """Turn arbitrary newlines into spaces, collapse repeated whitespace."""
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def _normalize_section_name(name: str) -> str:
    """Normalize section name to standard form."""
    name = name.lower().strip()
    if name in ("caption", "captions"):
        return "captions"
    elif name in ("summary", "summ"):
        return "summary"
    elif name in ("term", "terms"):
        return "terms"
    return name


def _parse_terms_block(text: str) -> List[Dict[str, str]]:
    """Parse a terms block with multiple format support.

    Supports formats:
    - Term: word - definition
    - Term: word: definition
    - **word**: definition
    - • word - definition
    - word: definition (fallback)
    """
    if not text:
        return []

    t = _norm_ws(text)
    terms: List[Dict[str, str]] = []
    seen_terms = set()

    # Split by lines first to handle line-based formats
    lines = t.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        term = None
        definition = None

        # Pattern 1: "- **TermName**: definition" (Gemini's primary format)
        m1 = re.match(r"^-?\s*\*\*\s*(.+?)\s*\*\*\s*:\s*(.+)$", line)
        if m1:
            term = m1.group(1).strip()
            definition = m1.group(2).strip()

        # Pattern 2: "Term: word - definition"
        if not term:
            m2 = re.match(r"^Term\s*:\s*(.+?)\s*[-–—:]\s*(.+)$", line, re.IGNORECASE)
            if m2:
                term = m2.group(1).strip()
                definition = m2.group(2).strip()

        # Pattern 3: "**word** - definition" (no colon after **)
        if not term:
            m3 = re.match(r"^\*\*\s*(.+?)\s*\*\*\s*[-–—]\s*(.+)$", line)
            if m3:
                term = m3.group(1).strip()
                definition = m3.group(2).strip()

        # Pattern 4: "• word - definition" or "- word: definition" (no markdown)
        if not term:
            m4 = re.match(r"^[•\-]\s*([^*]+?)\s*[-–—:]\s*(.+)$", line)
            if m4:
                term = m4.group(1).strip()
                definition = m4.group(2).strip()

        # Pattern 5: "Word: definition" (capitalized word)
        if not term:
            m5 = re.match(r"^([A-Z][a-zA-Z0-9\s\(\)]{1,50})\s*:\s*(.+)$", line)
            if m5:
                term = m5.group(1).strip()
                definition = m5.group(2).strip()

        # Pattern 6: Numbered list "1. word - definition"
        if not term:
            m6 = re.match(r"^\d+\.\s*(.+?)\s*[-–—:]\s*(.+)$", line)
            if m6:
                term = m6.group(1).strip()
                definition = m6.group(2).strip()

        # Validate and add term
        if term and definition:
            # Clean up term
            term = re.sub(r"^[\s\-•\*:]+|[\s\-•\*:]+$", "", term)
            # Skip if term is a section header or common word
            skip_terms = {
                "term",
                "terms",
                "caption",
                "captions",
                "summary",
                "overall",
                "current",
                "context",
                "segment",
            }
            if (
                len(term) >= 2
                and len(definition) >= 5
                and term.lower() not in skip_terms
                and term.lower() not in seen_terms
            ):
                terms.append({"term": term, "definition": definition})
                seen_terms.add(term.lower())

    # Fallback: try multi-line patterns if no terms found
    if not terms:
        # Pattern for "Term: word - definition" spanning multiple lines
        term_pat = re.compile(
            r"Term\s*:\s*([^-:\n]+?)\s*[-–—:]\s*([^\n]+)", re.IGNORECASE
        )
        for m in term_pat.finditer(t):
            term = _squash_inline(m.group(1))
            definition = _squash_inline(m.group(2))
            if term and definition and len(term) >= 2 and len(definition) >= 5:
                term_lower = term.lower()
                if term_lower not in seen_terms and term_lower not in {"term", "terms"}:
                    terms.append({"term": term, "definition": definition})
                    seen_terms.add(term_lower)

    return terms


def parse_gemini_output(raw: str) -> Dict[str, object]:
    """Parse Gemini model output containing [Captions], [Summary], [Terms] sections.

    More robust version that handles:
    - Various section header formats ([Section], **Section**, Section:)
    - Streaming/partial content
    - Missing sections
    - Various term formats
    """
    if not raw or not raw.strip():
        return {
            "captions": "",
            "summary": {"overall_context": "", "current_segment": ""},
            "terms": [],
        }

    raw = _norm_ws(raw)

    # Find all section markers and their positions
    matches = list(SECTION_RE.finditer(raw))
    sections: Dict[str, str] = {"captions": "", "summary": "", "terms": ""}

    # If no section markers found, try to extract content anyway
    if not matches:
        # Try to find terms even without section markers
        terms = _parse_terms_block(raw)

        # Use raw text as captions if it looks like transcription
        captions = ""
        if len(raw) > 10 and not any(kw in raw.lower() for kw in ["term:", "**", "•"]):
            captions = _squash_inline(raw)

        return {
            "captions": captions,
            "summary": {"overall_context": "", "current_segment": ""},
            "terms": terms,
        }

    # Extract content for each section
    preamble = raw[: matches[0].start()].strip() if matches else ""

    for i, m in enumerate(matches):
        section_name = _normalize_section_name(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        content = raw[start:end].strip()

        # Clean content (remove leading colons, dashes, etc.)
        content = re.sub(r"^[\s:]+", "", content)

        if section_name in sections:
            if sections[section_name]:
                sections[section_name] += "\n" + content
            else:
                sections[section_name] = content

    # Parse captions
    captions = _squash_inline(sections["captions"])

    # If no captions section but we have preamble, use it
    if not captions and preamble:
        # Only use preamble as captions if it doesn't look like terms
        if not any(kw in preamble.lower() for kw in ["term:", "**"]):
            captions = _squash_inline(preamble)

    # Parse summary - flexible format matching
    # Gemini uses format like: "- **Overall context**: text" and "- **Current segment**: text"
    summary_text = sections["summary"]
    overall = ""
    current = ""

    if summary_text:
        # Try multiple patterns for Overall context
        # Pattern 1: "- **Overall context**: text" (markdown bullet)
        # Pattern 2: "Overall context: text"
        overall_patterns = [
            r"-\s*\*\*\s*Overall\s*(?:context)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|$)",
            r"\*\*\s*Overall\s*(?:context)?\s*\*\*\s*:\s*(.+?)(?=\*\*|$)",
            r"Overall\s*(?:context)?\s*:\s*(.+?)(?=Current|Segment|-\s*\*\*|$)",
            r"Context\s*:\s*(.+?)(?=Current|Segment|$)",
        ]

        for pattern in overall_patterns:
            m_overall = re.search(
                pattern, summary_text, flags=re.IGNORECASE | re.DOTALL
            )
            if m_overall:
                overall = _squash_inline(m_overall.group(1))
                if overall:
                    break

        # Try multiple patterns for Current segment
        current_patterns = [
            r"-\s*\*\*\s*Current\s*(?:segment)?\s*\*\*\s*:\s*(.+?)(?=-\s*\*\*|\[|$)",
            r"\*\*\s*Current\s*(?:segment)?\s*\*\*\s*:\s*(.+?)(?=\*\*|\[|$)",
            r"Current\s*(?:segment)?\s*:\s*(.+?)(?=-\s*\*\*|\[|$)",
            r"Segment\s*:\s*(.+)$",
        ]

        for pattern in current_patterns:
            m_current = re.search(
                pattern, summary_text, flags=re.IGNORECASE | re.DOTALL
            )
            if m_current:
                current = _squash_inline(m_current.group(1))
                if current:
                    break

        # If no structured summary found, use the whole text as overall
        if not overall and not current:
            summary_inline = _squash_inline(summary_text)
            if summary_inline:
                overall = summary_inline

    # Parse terms from multiple sources
    terms = []

    # Terms from preamble (sometimes Gemini puts terms before markers)
    terms.extend(_parse_terms_block(preamble))

    # Terms from [Terms] section
    terms.extend(_parse_terms_block(sections["terms"]))

    # Also check if terms are embedded in other sections
    if not terms:
        terms.extend(_parse_terms_block(sections["summary"]))
        terms.extend(_parse_terms_block(sections["captions"]))

    return {
        "captions": captions,
        "summary": {
            "overall_context": overall,
            "current_segment": current,
        },
        "terms": terms,
    }


def test_parse_gemini_output():
    """Test function to verify parsing logic works correctly."""
    # Test case 1: Standard format
    test1 = """[Captions] Hello, this is a test transcription about machine learning.
[Summary] Overall context: Discussion about AI. Current segment: Machine learning basics.
[Terms] Term: Machine Learning - A subset of AI that enables systems to learn from data.
Term: Neural Network - Computing systems inspired by biological neural networks."""

    result1 = parse_gemini_output(test1)
    print("Test 1 (Standard format):")
    print(f"  Captions: {result1['captions']}")
    print(f"  Summary: {result1['summary']}")
    print(f"  Terms: {result1['terms']}")
    print()

    # Test case 2: Markdown format with **
    test2 = """**Captions** The speaker is discussing deep learning applications.
**Summary** This is about neural networks and their uses.
**Terms**
**Deep Learning**: A subset of machine learning using neural networks.
**Backpropagation**: Algorithm for training neural networks."""

    result2 = parse_gemini_output(test2)
    print("Test 2 (Markdown format):")
    print(f"  Captions: {result2['captions']}")
    print(f"  Summary: {result2['summary']}")
    print(f"  Terms: {result2['terms']}")
    print()

    # Test case 3: Plain text without section markers
    test3 = """The lecture covers important topics in computer science.
Term: Algorithm - A step-by-step procedure for solving problems.
Term: Data Structure - A way of organizing data for efficient use."""

    result3 = parse_gemini_output(test3)
    print("Test 3 (No section markers):")
    print(f"  Captions: {result3['captions']}")
    print(f"  Summary: {result3['summary']}")
    print(f"  Terms: {result3['terms']}")
    print()

    # Test case 4: Bullet point format
    test4 = """[Captions] Testing bullet format
[Terms]
• API - Application Programming Interface
• REST - Representational State Transfer
- HTTP - Hypertext Transfer Protocol"""

    result4 = parse_gemini_output(test4)
    print("Test 4 (Bullet format):")
    print(f"  Captions: {result4['captions']}")
    print(f"  Terms: {result4['terms']}")
    print()

    return True


# Uncomment to run test on startup:
# test_parse_gemini_output()


# Load OpenLexicon for keyword extraction
lexicon_df = None
lexicon_dict = {}


def load_lexicon():
    """Load OpenLexicon.xlsx and create lookup dictionary"""
    global lexicon_df, lexicon_dict
    try:
        lexicon_path = os.path.join(
            os.path.dirname(__file__), "data/lexicon/OpenLexicon.xlsx"
        )
        print(f"Loading OpenLexicon from {lexicon_path}...")
        lexicon_df = pd.read_excel(lexicon_path)

        # Create dictionary for faster lookup: {word: LgSUBTLWF_value}
        lexicon_dict = {}
        for _, row in lexicon_df.iterrows():
            word = str(row["ortho"]).lower()
            freq_value = row["English_Lexicon_Project__LgSUBTLWF"]
            lexicon_dict[word] = freq_value

        print(f"Lexicon loaded: {len(lexicon_dict)} words")
    except Exception as e:
        print(f"Warning: Could not load OpenLexicon.xlsx: {e}")
        print("Keyword extraction will use fallback method")


# Load lexicon on startup
load_lexicon()

# Store transcription history per session
transcription_sessions = {}


def preprocess_word(word):
    """Preprocess word for lexicon lookup

    1. Expand contractions (you're -> you are)
    2. Lemmatize to base form (declares -> declare, remains -> remain)
    3. Return the most searchable form

    Args:
        word: Original word from transcription

    Returns:
        Preprocessed word in base form, or None if should be skipped
    """
    # Remove punctuation and lowercase
    cleaned = "".join(c for c in word if c.isalnum()).lower()

    if not cleaned or len(cleaned) < 3:
        return None

    # Expand contractions first
    if cleaned in CONTRACTIONS:
        # Return the first word of the expansion
        # e.g., "you're" -> "you are" -> "you"
        expanded = CONTRACTIONS[cleaned].split()[0]
        cleaned = expanded

    # Lemmatize: try as verb, then noun, then adjective
    # Verbs: declares -> declare, remains -> remain
    lemma_v = lemmatizer.lemmatize(cleaned, pos="v")
    if lemma_v != cleaned:
        return lemma_v

    # Nouns: cats -> cat
    lemma_n = lemmatizer.lemmatize(cleaned, pos="n")
    if lemma_n != cleaned:
        return lemma_n

    # Adjectives: better -> good
    lemma_a = lemmatizer.lemmatize(cleaned, pos="a")
    if lemma_a != cleaned:
        return lemma_a

    # No lemmatization needed, return as-is
    return cleaned


class TranscriptionSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.words = []
        self.word_timestamps = []  # Track when each word was added
        self.full_text = ""
        self.sentences = []
        # Real-time transcription state (like transcribe_demo.py)
        self.phrase_bytes = b""
        self.phrase_time = None
        self.transcription_lines = [""]

        # GPT keyword extraction state (for double-spacebar navigation)
        self.gpt_keyword_pairs = []  # List of {keyword, description} dicts
        self.current_keyword_index = 0  # Current keyword being shown

        # Gemini Live API state
        self.gemini_active = False
        self.gemini_session = None
        self.gemini_audio_queue = None
        self.gemini_raw_output = ""  # Accumulated raw output from Gemini
        self.gemini_captions = ""  # Latest parsed captions
        self.gemini_summary = {"overall_context": "", "current_segment": ""}
        self.gemini_terms = []  # List of {term, definition} dicts
        self.gemini_terms_history = []  # All terms ever extracted (for spacebar)

        # Logging system
        self.session_start_time = datetime.utcnow()
        self.event_log = []  # List of all events (words added, searches performed)

        # Log session start (don't save immediately during init)
        self._log_event(
            "session_start",
            {
                "session_id": session_id,
                "start_time": self.session_start_time.isoformat(),
            },
            save_immediately=False,
        )

    def _log_event(self, event_type, data, save_immediately=True):
        """Internal method to log events"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self.event_log.append(event)
        print(f"[LOG] {event_type}: {data}")

        # Save logs immediately for real-time persistence
        if save_immediately:
            self.save_logs_to_file()

    def add_text(self, text):
        if text.strip():
            from datetime import datetime

            self.full_text += " " + text
            self.sentences.append(text)

            # Add words with timestamps
            new_words = text.split()
            current_time = datetime.utcnow()
            for word in new_words:
                self.words.append(word)
                self.word_timestamps.append(current_time)

                # Log each word addition
                # self._log_event(
                #     "word_added",
                #     {
                #         "word": word,
                #         "timestamp": current_time.isoformat(),
                #         "word_index": len(self.words) - 1,
                #     },
                # )

    def update_gemini_data(self, raw_text):
        """Update Gemini data from raw model output"""
        self.gemini_raw_output += raw_text

        # Debug logging - show raw text received
        print(
            f"[Gemini DEBUG] Raw text chunk ({len(raw_text)} chars): {raw_text[:200]}..."
        )
        print(f"[Gemini DEBUG] Total accumulated ({len(self.gemini_raw_output)} chars)")

        # Parse the accumulated output
        parsed = parse_gemini_output(self.gemini_raw_output)

        # Debug logging - show parsed results
        print(
            f"[Gemini DEBUG] Parsed captions: {parsed['captions'][:100] if parsed['captions'] else 'EMPTY'}..."
        )
        print(f"[Gemini DEBUG] Parsed summary: {parsed['summary']}")
        print(
            f"[Gemini DEBUG] Parsed terms ({len(parsed['terms'])}): {parsed['terms']}"
        )

        # Update captions
        if parsed["captions"]:
            self.gemini_captions = parsed["captions"]
            # Also add captions to regular transcription for consistency
            # (only add new text, not duplicates)

        # Update summary
        if parsed["summary"]["overall_context"]:
            self.gemini_summary["overall_context"] = parsed["summary"][
                "overall_context"
            ]
        if parsed["summary"]["current_segment"]:
            self.gemini_summary["current_segment"] = parsed["summary"][
                "current_segment"
            ]

        # Update terms
        if parsed["terms"]:
            self.gemini_terms = parsed["terms"]
            # Add new terms to history (avoid duplicates)
            existing_term_names = {t["term"].lower() for t in self.gemini_terms_history}
            for term in parsed["terms"]:
                if term["term"].lower() not in existing_term_names:
                    self.gemini_terms_history.append(term)
                    existing_term_names.add(term["term"].lower())
                    print(f"[Gemini DEBUG] New term added: {term['term']}")

        return parsed

    def get_gemini_terms_for_search(self):
        """Get Gemini terms formatted for search (like GPT keyword pairs)"""
        # Return current terms as keyword-description pairs
        return [
            {"keyword": t["term"], "description": t["definition"]}
            for t in self.gemini_terms
        ]

    def get_top_keyword(self, context_window=20):
        """Extract most relevant keyword using OpenLexicon frequency filtering

        Args:
            context_window: Number of recent words to consider (default: 20)

        Returns:
            The most important keyword (lowest frequency or not in lexicon)
        """
        if len(self.words) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        # Get recent words (spacebar 근처 단어들)
        recent_words = (
            self.words[-context_window:]
            if len(self.words) > context_window
            else self.words
        )

        # Clean words and check against lexicon
        candidate_words = []

        for word in recent_words:
            # Preprocess word (expand contractions, lemmatize)
            cleaned = preprocess_word(word)

            # Skip if preprocessing returned None
            if not cleaned:
                continue

            # Check lexicon frequency (using preprocessed form)
            freq_value = lexicon_dict.get(cleaned)
            print(f"[Lexicon] '{word}' -> preprocessed: '{cleaned}'")

            # Keep words that are:
            # 1. Not in lexicon (freq_value is None)
            # 2. Have NaN frequency (pd.isna)
            # 3. Have frequency < 3.0 (rare words)
            if freq_value is None:
                # Not in lexicon - likely important/rare
                candidate_words.append(
                    (cleaned, -1, word)
                )  # -1 for sorting (highest priority)
                print(f"[Lexicon] '{cleaned}' not in lexicon -> candidate")
            elif pd.isna(freq_value):
                # NaN frequency - keep as candidate
                candidate_words.append((cleaned, 0, word))  # 0 for sorting
                print(f"[Lexicon] '{cleaned}' has NaN frequency -> candidate")
            elif freq_value < 3.0:
                # Low frequency (rare word) - keep as candidate
                candidate_words.append((cleaned, freq_value, word))
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} < 3.0 -> candidate")
            else:
                # High frequency (common word) - skip
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} >= 3.0 -> skipped")

        if not candidate_words:
            # No candidates found - fallback to last word
            fallback_word = self.words[-1] if self.words else ""
            print(
                f"[Lexicon] ⚠️ No important words found (all freq >= 3.0), using fallback: '{fallback_word}'"
            )
            return fallback_word

        # Sort by frequency (lowest first, with not-in-lexicon/-1 first)
        candidate_words.sort(key=lambda x: x[1])

        # Return the rarest word (or most recent if tied)
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        print(f"\n[Lexicon] Selected keyword: '{selected_word}' (freq={selected_freq})")
        print(
            f"[Lexicon] All candidates: {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        return selected_word

    def get_top_keyword_with_time_threshold(self, time_threshold=5):
        """Extract most important keyword from words within the last N seconds

        Finds the word with HIGHEST importance (LOWEST frequency) among words
        from the last N seconds that have OpenLexicon frequency < 3.0

        Args:
            time_threshold: Number of seconds to look back (default: 5)

        Returns:
            The most important keyword (lowest frequency) within the time window
        """
        from datetime import datetime, timedelta

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        # Get current time and calculate threshold
        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        # Filter words within the time threshold
        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            # No recent words - fallback to last word
            print(
                f"[Recent] No words found in last {time_threshold}s, using last word as fallback"
            )
            return self.words[-1] if self.words else ""

        print(
            f"\n[Recent] Found {len(recent_words)} words in last {time_threshold}s: {recent_words}"
        )

        # Only keep words with frequency < 3.0 (high importance threshold)
        candidate_words = []

        for word in recent_words:
            # Preprocess word (expand contractions, lemmatize)
            cleaned = preprocess_word(word)

            # Skip if preprocessing returned None
            if not cleaned:
                continue

            # Check lexicon frequency (using preprocessed form)
            freq_value = lexicon_dict.get(cleaned)
            print(f"[Recent] '{word}' -> preprocessed: '{cleaned}'")

            # Only keep words with importance threshold below 3.0:
            # 1. Not in lexicon (freq_value is None) -> highest importance
            # 2. Have NaN frequency (pd.isna) -> high importance
            # 3. Have frequency < 3.0 (rare words) -> high importance
            if freq_value is None:
                # Not in lexicon - highest importance
                candidate_words.append((cleaned, -1, word))
                print(
                    f"[Recent] '{cleaned}' not in lexicon (highest importance) -> candidate"
                )
            elif pd.isna(freq_value):
                # NaN frequency - very high importance
                candidate_words.append((cleaned, 0, word))
                print(
                    f"[Recent] '{cleaned}' has NaN frequency (very high importance) -> candidate"
                )
            elif freq_value < 3.0:
                # Low frequency = high importance
                candidate_words.append((cleaned, freq_value, word))
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} < 3.0 (high importance) -> candidate"
                )
            else:
                # High frequency (common word) - skip
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} >= 3.0 (low importance) -> skipped"
                )

        if not candidate_words:
            # No high-importance words found - fallback to last recent word
            fallback_word = recent_words[-1] if recent_words else ""
            print(
                f"[Recent] ⚠️ No important words found (all freq >= 3.0) in last {time_threshold}s"
            )
            print(f"[Recent] Using fallback: '{fallback_word}'")

            # Log the analysis even when using fallback
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

        # Sort by frequency: LOWEST first = HIGHEST importance first
        # -1 (not in lexicon) < 0 (NaN) < 0.x (rare) < 3.0 (common, already filtered out)
        candidate_words.sort(key=lambda x: x[1])

        # Return the word with HIGHEST importance (LOWEST frequency)
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        freq_label = (
            "not in lexicon" if selected_freq == -1 else f"freq={selected_freq}"
        )

        print(
            f"\n[Recent] ✓ Selected HIGHEST importance keyword: '{selected_word}' ({freq_label}) from last {time_threshold}s"
        )
        print(
            f"[Recent] All candidates (sorted by importance): {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        # Log the keyword extraction analysis
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

    def get_top_keyword_gpt(self, time_threshold=10):
        """Use GPT to predict which word the user would want to look up

        Uses a few-shot prompt based on video lecture transcription patterns
        to intelligently predict technical terms, unfamiliar vocabulary, and
        concepts that need clarification.

        Args:
            time_threshold: Number of seconds to look back for context (default: 3)

        Returns:
            GPT-predicted keyword that user would want to look up
        """
        from datetime import datetime, timedelta

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        # Get recent context words (last N seconds)
        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        print(f"\n[GPT DEBUG] Current time: {current_time.isoformat()}")
        print(
            f"[GPT DEBUG] Threshold time: {threshold_time.isoformat()} ({time_threshold}s ago)"
        )
        print(f"[GPT DEBUG] Total words in session: {len(self.words)}")

        # Show last 10 words with their timestamps
        print("[GPT DEBUG] Last 10 words with timestamps:")
        for i in range(max(0, len(self.words) - 10), len(self.words)):
            word = self.words[i]
            timestamp = self.word_timestamps[i]
            time_diff = (current_time - timestamp).total_seconds()
            in_range = "✓" if timestamp >= threshold_time else "✗"
            print(
                f"  {in_range} [{i}] '{word}' at {timestamp.isoformat()} ({time_diff:.2f}s ago)"
            )

        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            # Fallback to last word
            print(
                f"[GPT] ⚠️ No words found in last {time_threshold}s, using last word as fallback"
            )
            return self.words[-1] if self.words else ""

        context_text = " ".join(recent_words)
        print(f"\n[GPT] ✓ Found {len(recent_words)} words in last {time_threshold}s")
        print(f"[GPT] Context sent to GPT: '{context_text}'")

        # Create GPT prompt (based on test_gpt_prediction.py)
        # Request multiple keywords ranked by importance
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
            # Call GPT API (using gpt-4o-mini like in test script)
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

            # Parse the response to extract multiple "Keyword:" and "Description:" pairs
            # Expected format:
            # Keyword: keyword1
            # Description: description1
            # Keyword: keyword2
            # Description: description2
            keyword_description_pairs = []
            current_keyword = None

            # Split by newline to separate keyword and description
            lines = raw_response.split("\n")

            for line in lines:
                line = line.strip()
                if line.lower().startswith("keyword:"):
                    # Start a new keyword-description pair
                    current_keyword = line.split(":", 1)[1].strip()
                elif line.lower().startswith("description:") and current_keyword:
                    # Complete the pair with description
                    description = line.split(":", 1)[1].strip()
                    keyword_description_pairs.append(
                        {
                            "keyword": current_keyword,
                            "description": description,
                        }
                    )
                    current_keyword = None

            # Handle case where keyword has no description
            if current_keyword:
                keyword_description_pairs.append(
                    {
                        "keyword": current_keyword,
                        "description": None,
                    }
                )

            # Use first keyword as the main predicted keyword
            if keyword_description_pairs:
                predicted_keyword = keyword_description_pairs[0]["keyword"]
            else:
                predicted_keyword = raw_response

            # Store all keyword-description pairs in session for double-spacebar navigation
            self.gpt_keyword_pairs = keyword_description_pairs
            self.current_keyword_index = 0  # Reset to first keyword

            # Log the GPT prediction with all keyword-description pairs
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
            # Fallback to last word on error
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

    def log_search_action(self, search_mode, search_type, keyword=None, num_results=0):
        """Log when user performs a search (spacebar action)"""
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

    def save_logs_to_file(self):
        """Save session logs to a JSON file"""
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


def parse_srt(srt_content):
    """Parse SRT file content and return list of (start_time_ms, end_time_ms, text) tuples"""
    import re

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


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    transcription_sessions[session_id] = TranscriptionSession(session_id)
    emit("connected", {"status": "Connected to MorseQuery"})
    print(f"[Session] New session connected: {session_id}")


@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in transcription_sessions:
        # Save logs before deleting session
        session = transcription_sessions[session_id]
        session._log_event(
            "session_end",
            {
                "total_words": len(session.words),
                "total_searches": sum(
                    1 for e in session.event_log if e["event_type"] == "search_action"
                ),
            },
        )
        log_file = session.save_logs_to_file()
        print(
            f"[Session] Session disconnected: {session_id}, logs saved to: {log_file}"
        )
        del transcription_sessions[session_id]


@socketio.on("start_whisper")
def handle_start_whisper():
    """Initialize Whisper model"""
    global whisper_model
    if whisper_model is None:
        emit("status", {"message": "Loading Whisper model..."})
        # Use 'tiny' model for faster processing with lower latency
        whisper_model = whisper.load_model("tiny")
        emit("status", {"message": "Whisper model loaded (tiny)"})
    else:
        emit("status", {"message": "Whisper already loaded"})


# Gemini Live session management
gemini_live_sessions = {}  # session_id -> async event loop + tasks


def run_gemini_live_loop(session_id):
    """Background thread that runs the Gemini Live async event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def gemini_session_handler():
        """Main async handler for Gemini Live session"""
        if not gemini_client:
            socketio.emit(
                "error",
                {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."},
                room=session_id,
            )
            return

        session = transcription_sessions.get(session_id)
        if not session:
            return

        try:
            socketio.emit(
                "status", {"message": "Connecting to Gemini Live..."}, room=session_id
            )

            async with gemini_client.aio.live.connect(
                model=GEMINI_MODEL, config=GEMINI_STUDY_CONFIG
            ) as gemini_session:
                session.gemini_session = gemini_session
                session.gemini_active = True
                session.gemini_audio_queue = asyncio.Queue(maxsize=10)

                socketio.emit(
                    "status",
                    {"message": "Gemini Live connected. Start speaking or play audio."},
                    room=session_id,
                )
                socketio.emit(
                    "gemini_connected", {"status": "connected"}, room=session_id
                )

                # Create tasks for sending audio and receiving responses
                async def send_audio():
                    """Send audio chunks from queue to Gemini"""
                    while session.gemini_active:
                        try:
                            audio_data = await asyncio.wait_for(
                                session.gemini_audio_queue.get(), timeout=1.0
                            )
                            await gemini_session.send(input=audio_data)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"[Gemini] Send error: {e}")
                            break

                async def receive_responses():
                    """Receive and process responses from Gemini"""
                    while session.gemini_active:
                        try:
                            turn = gemini_session.receive()
                            async for response in turn:
                                # Debug: print full response structure
                                # print(f"[Gemini DEBUG] Response type: {type(response)}")
                                # print(f"[Gemini DEBUG] Response attrs: {dir(response)}")

                                # Try multiple ways to get text
                                text = None

                                # Method 1: direct .text attribute
                                text = getattr(response, "text", None)

                                # Method 2: check for parts
                                if not text:
                                    parts = getattr(response, "parts", None)
                                    if parts:
                                        # print(f"[Gemini DEBUG] Found parts: {parts}")
                                        for part in parts:
                                            part_text = getattr(part, "text", None)
                                            if part_text:
                                                text = (text or "") + part_text

                                # Method 3: check for content
                                if not text:
                                    content = getattr(response, "content", None)
                                    if content:
                                        # print(
                                        #     f"[Gemini DEBUG] Found content: {content}"
                                        # )
                                        text = str(content)

                                # Method 4: check server_content (Live API specific)
                                if not text:
                                    server_content = getattr(
                                        response, "server_content", None
                                    )
                                    if server_content:
                                        # print(
                                        #     f"[Gemini DEBUG] Found server_content: {server_content}"
                                        # )
                                        model_turn = getattr(
                                            server_content, "model_turn", None
                                        )
                                        if model_turn:
                                            parts = getattr(model_turn, "parts", None)
                                            if parts:
                                                for part in parts:
                                                    part_text = getattr(
                                                        part, "text", None
                                                    )
                                                    if part_text:
                                                        text = (text or "") + part_text

                                if text:
                                    print(
                                        f"[Gemini] Got text ({len(text)} chars): {text[:200]}..."
                                    )

                                    # Update session with Gemini output
                                    parsed = session.update_gemini_data(text)

                                    # Emit Gemini data to client
                                    socketio.emit(
                                        "gemini_transcription",
                                        {
                                            "raw": text,
                                            "captions": session.gemini_captions,
                                            "summary": session.gemini_summary,
                                            "terms": session.gemini_terms,
                                        },
                                        room=session_id,
                                    )

                                    # Log Gemini output (less frequently to avoid spam)
                                    if len(session.gemini_raw_output) % 500 < len(text):
                                        session._log_event(
                                            "gemini_output",
                                            {
                                                "raw_text_length": len(
                                                    session.gemini_raw_output
                                                ),
                                                "captions": session.gemini_captions[
                                                    :200
                                                ]
                                                if session.gemini_captions
                                                else "",
                                                "summary": session.gemini_summary,
                                                "terms": session.gemini_terms,
                                            },
                                        )
                                else:
                                    # Check for audio data
                                    data = getattr(response, "data", None)
                                    if data:
                                        print(
                                            f"[Gemini DEBUG] Got audio data: {len(data)} bytes"
                                        )

                        except Exception as e:
                            if session.gemini_active:
                                print(f"[Gemini] Receive error: {e}")
                                import traceback

                                traceback.print_exc()
                            break

                # Run send and receive concurrently
                send_task = asyncio.create_task(send_audio())
                receive_task = asyncio.create_task(receive_responses())

                # Wait for session to be stopped
                while session.gemini_active:
                    await asyncio.sleep(0.5)

                # Cancel tasks
                send_task.cancel()
                receive_task.cancel()

        except Exception as e:
            print(f"[Gemini] Session error: {e}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Gemini error: {str(e)}"}, room=session_id
            )
        finally:
            session.gemini_active = False
            session.gemini_session = None
            socketio.emit(
                "gemini_disconnected", {"status": "disconnected"}, room=session_id
            )

    try:
        loop.run_until_complete(gemini_session_handler())
    finally:
        loop.close()


@socketio.on("start_gemini_live")
def handle_start_gemini_live():
    """Start Gemini Live session for real-time transcription with captions/summary/terms"""
    session_id = request.sid

    if not gemini_client:
        emit("error", {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."})
        return

    if session_id not in transcription_sessions:
        transcription_sessions[session_id] = TranscriptionSession(session_id)

    session = transcription_sessions[session_id]

    if session.gemini_active:
        emit("status", {"message": "Gemini Live already active"})
        return

    # Reset Gemini state
    session.gemini_raw_output = ""
    session.gemini_captions = ""
    session.gemini_summary = {"overall_context": "", "current_segment": ""}
    session.gemini_terms = []

    # Log session start
    session._log_event("gemini_live_start", {"model": GEMINI_MODEL})

    emit("status", {"message": "Starting Gemini Live session..."})

    # Start background thread for Gemini Live
    thread = threading.Thread(target=run_gemini_live_loop, args=(session_id,))
    thread.daemon = True
    thread.start()
    gemini_live_sessions[session_id] = thread


@socketio.on("stop_gemini_live")
def handle_stop_gemini_live():
    """Stop Gemini Live session"""
    session_id = request.sid

    if session_id in transcription_sessions:
        session = transcription_sessions[session_id]
        session.gemini_active = False

        # Log session stop
        session._log_event(
            "gemini_live_stop",
            {
                "total_terms": len(session.gemini_terms_history),
                "final_captions_length": len(session.gemini_captions),
            },
        )

    emit("status", {"message": "Gemini Live session stopped"})
    emit("gemini_disconnected", {"status": "disconnected"})


@socketio.on("audio_chunk_gemini_live")
def handle_audio_chunk_gemini_live(data):
    """Process audio chunk with Gemini Live"""
    session_id = request.sid

    if session_id not in transcription_sessions:
        emit("error", {"message": "No active session"})
        return

    session = transcription_sessions[session_id]

    if not session.gemini_active or not session.gemini_audio_queue:
        emit("error", {"message": "Gemini Live not active. Start Gemini first."})
        return

    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])
        file_format = data.get("format", "webm")

        # Convert audio to PCM format for Gemini
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=f".{file_format}", delete=False
            ) as temp_audio:
                temp_audio.write(audio_data)
                temp_path = temp_audio.name

            # Convert to 16kHz mono PCM
            audio = AudioSegment.from_file(temp_path, format=file_format)
            audio = (
                audio.set_channels(1)
                .set_frame_rate(GEMINI_SEND_SAMPLE_RATE)
                .set_sample_width(2)
            )
            raw_data = audio.raw_data

            # Put audio data in queue for async sender
            # Create a proper Gemini audio input format
            audio_input = {"data": raw_data, "mime_type": "audio/pcm"}

            # Use a thread-safe way to add to async queue
            def add_to_queue():
                try:
                    session.gemini_audio_queue.put_nowait(audio_input)
                except asyncio.QueueFull:
                    print("[Gemini] Audio queue full, dropping chunk")

            # Schedule the queue addition
            if session.gemini_audio_queue:
                add_to_queue()

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"[Gemini] Audio chunk error: {e}")
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"Gemini audio error: {str(e)}"})


def process_whisper_background(audio_data, file_format, session_id):
    """Background thread for Whisper processing"""
    from datetime import datetime

    global whisper_model
    temp_path = None

    audio_received_time = datetime.utcnow()
    try:
        # Check if audio data is valid
        if not audio_data or len(audio_data) < 100:
            print(
                f"[Whisper] Audio data too small: {len(audio_data)} bytes - skipping silently"
            )
            # Silently skip without showing error to user
            return

        # Map common formats to extensions
        extension_map = {
            "webm": ".webm",
            "wav": ".wav",
            "mp3": ".mp3",
            "mp4": ".mp4",
            "m4a": ".m4a",
            "ogg": ".ogg",
            "flac": ".flac",
        }

        suffix = extension_map.get(file_format, ".webm")

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name

        # print(f"Processing audio file: {temp_path}, size: {len(audio_data)} bytes")

        # Transcribe with Whisper
        if whisper_model is None:
            socketio.emit(
                "status", {"message": "Loading Whisper model..."}, room=session_id
            )
            whisper_model = whisper.load_model("tiny")

        socketio.emit(
            "status", {"message": "Transcribing with Whisper..."}, room=session_id
        )

        transcription_start_time = datetime.utcnow()

        result = whisper_model.transcribe(temp_path)
        text = result["text"].strip()

        transcription_end_time = datetime.utcnow()
        transcription_duration = (
            transcription_end_time - transcription_start_time
        ).total_seconds()
        total_duration = (transcription_end_time - audio_received_time).total_seconds()

        if text:
            word_add_time = datetime.utcnow()
            session = transcription_sessions[session_id]
            session.add_text(text)

            # Log Whisper transcription to JSON
            session._log_event(
                "whisper_transcription",
                {
                    "text": text,
                    "word_count": len(text.split()),
                    "transcription_duration_seconds": transcription_duration,
                    "total_processing_seconds": total_duration,
                    "audio_size_bytes": len(audio_data),
                    "format": file_format,
                },
            )

            socketio.emit(
                "transcription", {"text": text, "source": "whisper"}, room=session_id
            )
        else:
            socketio.emit(
                "status", {"message": "No speech detected in audio"}, room=session_id
            )

    except RuntimeError as e:
        # Handle specific tensor reshape errors (empty/invalid audio)
        error_msg = str(e)
        if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
            print("[Whisper] Empty or invalid audio data - skipping silently")
            # Silently skip without showing error to user
            # This is normal in real-time recording scenarios
        elif "Linear(in_features=" in error_msg or "out_features=" in error_msg:
            print(f"[Whisper] Model or audio format error: {error_msg}")
            socketio.emit(
                "error",
                {
                    "message": "Audio format incompatible. Please try recording again or use a different audio source."
                },
                room=session_id,
            )
        else:
            print(f"Whisper RuntimeError: {error_msg}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Whisper error: {error_msg}"}, room=session_id
            )

    except ValueError as e:
        # Handle audio format/conversion errors
        error_msg = str(e)
        print(f"[Whisper] Value error (likely audio format issue): {error_msg}")
        socketio.emit(
            "error",
            {"message": "Audio format error. Please try a different recording format."},
            room=session_id,
        )

    except Exception as e:
        print(f"Whisper error: {str(e)}")
        import traceback

        traceback.print_exc()
        socketio.emit("error", {"message": f"Whisper error: {str(e)}"}, room=session_id)

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@socketio.on("audio_chunk_whisper")
def handle_audio_chunk_whisper(data):
    """Process audio chunk with Whisper"""
    session_id = request.sid

    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])

        # Detect file type from data or use provided format
        file_format = data.get("format", "webm")
        # print(f"Format: {file_format}, Data size: {len(audio_data)} bytes")

        emit(
            "status",
            {"message": f"Received audio ({len(audio_data)} bytes). Processing..."},
        )

        # Process in background thread to avoid blocking WebSocket
        thread = threading.Thread(
            target=process_whisper_background,
            args=(audio_data, file_format, session_id),
        )
        thread.daemon = True
        thread.start()

    except Exception as e:
        print(f"Whisper handler error: {str(e)}")
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"Whisper error: {str(e)}"})


@socketio.on("audio_chunk_realtime")
def handle_audio_chunk_realtime(data):
    """Process audio chunk with real-time accumulation (like transcribe_demo.py)"""
    session_id = request.sid

    try:
        from datetime import datetime, timedelta

        # Log audio chunk arrival time
        audio_received_time = datetime.utcnow()

        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])
        file_format = data.get("format", "webm")
        phrase_timeout = data.get("phrase_timeout", 3)  # seconds
        is_final = data.get("is_final", False)

        # print(
        #     f"Format: {file_format}, Data size: {len(audio_data)} bytes, is_final: {is_final}"
        # )

        session = transcription_sessions[session_id]
        now = datetime.utcnow()

        # Check if this is a new phrase (phrase_timeout logic from demo)
        phrase_complete = False
        if session.phrase_time and now - session.phrase_time > timedelta(
            seconds=phrase_timeout
        ):
            session.phrase_bytes = b""
            phrase_complete = True

        session.phrase_time = now

        # Convert audio to raw PCM format
        temp_path = None
        try:
            # Save to temporary file for conversion
            with tempfile.NamedTemporaryFile(
                suffix=f".{file_format}", delete=False
            ) as temp_audio:
                temp_audio.write(audio_data)
                temp_path = temp_audio.name

            # Load audio and convert to raw PCM
            audio = AudioSegment.from_file(temp_path, format=file_format)
            # Convert to mono, 16kHz, 16-bit PCM (like demo)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            raw_data = audio.raw_data

            # Accumulate audio data
            session.phrase_bytes += raw_data

            # print(f"Accumulated audio size: {len(session.phrase_bytes)} bytes")

            # Check if accumulated audio is valid
            if len(session.phrase_bytes) < 100:
                print(
                    f"[Real-time] Accumulated audio too small: {len(session.phrase_bytes)} bytes, skipping transcription"
                )
                return

            # Convert to numpy array for Whisper
            audio_np = (
                np.frombuffer(session.phrase_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            # Check if audio array is valid
            if len(audio_np) == 0:
                print("[Real-time] Audio array is empty, skipping transcription")
                return

            # Transcribe with Whisper
            global whisper_model
            if whisper_model is None:
                socketio.emit(
                    "status", {"message": "Loading Whisper model..."}, room=session_id
                )
                whisper_model = whisper.load_model("tiny")

            socketio.emit("status", {"message": "Transcribing..."}, room=session_id)

            transcription_start_time = datetime.utcnow()

            try:
                result = whisper_model.transcribe(audio_np, fp16=False)
                text = result["text"].strip()

                transcription_end_time = datetime.utcnow()
                transcription_duration = (
                    transcription_end_time - transcription_start_time
                ).total_seconds()
                total_duration = (
                    transcription_end_time - audio_received_time
                ).total_seconds()

            except RuntimeError as whisper_error:
                error_msg = str(whisper_error)
                if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
                    print(f"[Real-time] Empty or invalid audio: {error_msg}")
                    socketio.emit(
                        "status",
                        {"message": "Audio too short, waiting for more..."},
                        room=session_id,
                    )
                    return
                elif "Linear(in_features=" in error_msg or "out_features=" in error_msg:
                    print(f"[Real-time] Model or audio format error: {error_msg}")
                    socketio.emit(
                        "error",
                        {
                            "message": "Audio format incompatible. Please try a different recording method."
                        },
                        room=session_id,
                    )
                    # Reset phrase bytes to recover
                    session.phrase_bytes = b""
                    return
                else:
                    raise  # Re-raise if it's a different RuntimeError
            except ValueError as whisper_error:
                error_msg = str(whisper_error)
                print(f"[Real-time] Value error: {error_msg}")
                socketio.emit(
                    "error",
                    {"message": "Audio format error. Please try recording again."},
                    room=session_id,
                )
                # Reset phrase bytes to recover
                session.phrase_bytes = b""
                return

            if text:
                # Update or append transcription (like demo)
                if phrase_complete or is_final:
                    session.transcription_lines.append(text)
                    session.add_text(text)

                    # Log real-time Whisper transcription to JSON
                    session._log_event(
                        "whisper_transcription",
                        {
                            "text": text,
                            "word_count": len(text.split()),
                            "transcription_duration_seconds": transcription_duration,
                            "total_processing_seconds": total_duration,
                            "mode": "realtime",
                            "is_complete": True,
                            "phrase_complete": phrase_complete,
                            "is_final": is_final,
                        },
                    )

                    socketio.emit(
                        "transcription",
                        {
                            "text": text,
                            "source": "whisper-realtime",
                            "is_complete": True,
                        },
                        room=session_id,
                    )
                else:
                    session.transcription_lines[-1] = text
                    print(
                        f"[Real-time] ⏳ INCOMPLETE phrase (not added to session): '{text}'"
                    )
                    socketio.emit(
                        "transcription",
                        {
                            "text": text,
                            "source": "whisper-realtime",
                            "is_complete": False,
                        },
                        room=session_id,
                    )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"Real-time handler error: {str(e)}")
        import traceback

        traceback.print_exc()
        socketio.emit(
            "error", {"message": f"Real-time error: {str(e)}"}, room=session_id
        )


@socketio.on("search_request")
def handle_search_request(data):
    """Handle search request triggered by spacebar"""
    from datetime import datetime

    session_id = request.sid
    search_request_time = datetime.utcnow()

    search_mode = data.get(
        "mode", "gpt"
    )  # 'instant', 'recent', 'important', 'gpt', or 'gemini'
    search_type = data.get("type", "text")  # 'text' or 'image'
    client_timestamp = data.get(
        "client_timestamp"
    )  # Client-side timestamp when spacebar was pressed
    skip_search = data.get("skip_search", False)  # Skip Google search if true
    show_all_keywords = data.get(
        "show_all_keywords", False
    )  # Show all GPT/Gemini keywords at once

    # log action
    transcription_sessions[session_id].log_search_action(
        search_mode=search_mode,
        search_type=search_type,
        # keyword="not yet",
        # num_results="not yet",
    )

    if client_timestamp:
        # Calculate client-server latency
        try:
            client_time = datetime.fromisoformat(
                client_timestamp.replace("Z", "+00:00")
            )
            latency = (search_request_time - client_time).total_seconds()
            print(f"[TIMING] Client pressed spacebar at: {client_timestamp}")
            print(f"[TIMING] Client-server latency: {latency:.3f}s")
        except Exception as e:
            print(f"[TIMING] Could not parse client timestamp: {e}")

    if session_id not in transcription_sessions:
        emit("error", {"message": "No active session"})
        return

    # Get keyword based on search mode
    keyword_extraction_start = datetime.utcnow()

    session = transcription_sessions[session_id]

    if search_mode == "instant":
        # Use the keyword provided by client (last word)
        keyword = data.get("keyword", "")
    elif search_mode == "recent":
        # Recent mode: important word from last 5 seconds
        time_threshold = data.get("time_threshold", 5)  # default 5 seconds
        keyword = session.get_top_keyword_with_time_threshold(time_threshold)
    elif search_mode == "gpt":
        # GPT mode: use GPT to predict what user wants to look up
        time_threshold = data.get("time_threshold", 5)  # default 3 seconds
        keyword = session.get_top_keyword_gpt(time_threshold)
    elif search_mode == "gemini":
        # Gemini mode: use terms extracted by Gemini Live
        if not session.gemini_terms:
            emit(
                "error",
                {
                    "message": "No Gemini terms available. Start Gemini Live first and wait for terms to be extracted."
                },
            )
            return

        # Get the first term as keyword, or show all terms
        gemini_terms = session.get_gemini_terms_for_search()

        if show_all_keywords:
            emit("all_keywords", {"keywords": gemini_terms, "mode": "gemini"})
            if skip_search:
                return
            # Use first term for search
            keyword = gemini_terms[0]["keyword"] if gemini_terms else ""
        else:
            # Store Gemini terms like GPT keyword pairs for navigation
            session.gpt_keyword_pairs = gemini_terms
            session.current_keyword_index = 0
            keyword = gemini_terms[0]["keyword"] if gemini_terms else ""
    else:
        # TF-IDF/Important mode: calculate important keyword
        keyword = session.get_top_keyword()

    if not keyword:
        emit("error", {"message": "No keywords available for search"})
        return

    # Clean the keyword (remove punctuation)
    keyword_clean = "".join(c for c in keyword if c.isalnum() or c.isspace()).strip()

    # Get description and keyword count for GPT/Gemini mode
    description = None
    total_keywords = 1
    current_index = 0

    if search_mode in ("gpt", "gemini") and session.gpt_keyword_pairs:
        total_keywords = len(session.gpt_keyword_pairs)
        current_index = session.current_keyword_index
        if current_index < len(session.gpt_keyword_pairs):
            description = session.gpt_keyword_pairs[current_index].get("description")

    # If show_all_keywords is enabled and in GPT mode, emit all keywords at once
    # (Gemini mode handles this earlier in the code)
    if show_all_keywords and search_mode == "gpt" and session.gpt_keyword_pairs:
        emit(
            "all_keywords", {"keywords": session.gpt_keyword_pairs, "mode": search_mode}
        )
        if skip_search:
            return
    elif search_mode != "gemini" or not show_all_keywords:
        # Don't emit search_keyword if Gemini mode already emitted all_keywords
        emit(
            "search_keyword",
            {
                "keyword": keyword_clean,
                "mode": search_mode,
                "description": description,
                "total_keywords": total_keywords,
                "current_index": current_index,
            },
        )

    # Skip Google search if requested
    if skip_search:
        print("[Search] Skipping Google search (skip_search=True)")
        return

    # Perform Google Custom Search
    try:
        search_results = google_custom_search(keyword_clean, search_type)

        # Calculate number of results
        if search_type == "both" and isinstance(search_results, dict):
            num_results = len(search_results.get("text", [])) + len(
                search_results.get("image", [])
            )
        else:
            num_results = len(search_results) if isinstance(search_results, list) else 0

        # Log the search action
        session.log_search_action(
            search_mode=search_mode,
            search_type=search_type,
            keyword=keyword_clean,
            num_results=num_results,
        )

        emit(
            "search_results",
            {
                "keyword": keyword_clean,
                "mode": search_mode,
                "type": search_type,
                "results": search_results,
            },
        )
    except Exception as e:
        print(f"Search error: {str(e)}")
        emit("error", {"message": f"Search error: {str(e)}"})


@socketio.on("next_keyword")
def handle_next_keyword(data):
    """Handle double-spacebar request to show next GPT keyword"""
    session_id = request.sid
    search_type = data.get("type", "text")

    if session_id not in transcription_sessions:
        emit("error", {"message": "No active session"})
        return

    session = transcription_sessions[session_id]

    # Check if we have GPT keyword pairs
    if not session.gpt_keyword_pairs:
        emit("error", {"message": "No GPT keywords available. Press spacebar first."})
        return

    # Move to next keyword (cycle back to first if at end)
    session.current_keyword_index += 1
    if session.current_keyword_index >= len(session.gpt_keyword_pairs):
        session.current_keyword_index = 0  # Cycle back to first

    current_pair = session.gpt_keyword_pairs[session.current_keyword_index]
    keyword = current_pair.get("keyword", "")
    description = current_pair.get("description")

    # Emit the new keyword info
    emit(
        "search_keyword",
        {
            "keyword": keyword,
            "mode": "gpt",
            "description": description,
            "total_keywords": len(session.gpt_keyword_pairs),
            "current_index": session.current_keyword_index,
        },
    )

    # Perform search for the new keyword
    try:
        search_results = google_custom_search(keyword, search_type)

        # Calculate number of results
        if search_type == "both" and isinstance(search_results, dict):
            num_results = len(search_results.get("text", [])) + len(
                search_results.get("image", [])
            )
        else:
            num_results = len(search_results) if isinstance(search_results, list) else 0

        # Log the search action
        session.log_search_action(
            search_mode="gpt_next",
            search_type=search_type,
            keyword=keyword,
            num_results=num_results,
        )

        emit(
            "search_results",
            {
                "keyword": keyword,
                "mode": "gpt",
                "type": search_type,
                "results": search_results,
            },
        )
    except Exception as e:
        print(f"Search error: {str(e)}")
        emit("error", {"message": f"Search error: {str(e)}"})


@socketio.on("search_single_keyword")
def handle_search_single_keyword(data):
    """Handle click on a keyword from all_keywords list"""
    session_id = request.sid
    keyword = data.get("keyword", "")
    search_type = data.get("type", "text")

    if not keyword:
        emit("error", {"message": "No keyword provided"})
        return

    # Emit the keyword info
    emit(
        "search_keyword",
        {
            "keyword": keyword,
            "mode": "gpt",
            "description": None,
            "total_keywords": 1,
            "current_index": 0,
        },
    )

    # Perform Google Custom Search
    try:
        search_results = google_custom_search(keyword, search_type)

        # Calculate number of results
        if search_type == "both" and isinstance(search_results, dict):
            num_results = len(search_results.get("text", [])) + len(
                search_results.get("image", [])
            )
        else:
            num_results = len(search_results) if isinstance(search_results, list) else 0

        if session_id in transcription_sessions:
            transcription_sessions[session_id].log_search_action(
                search_mode="gpt_single",
                search_type=search_type,
                keyword=keyword,
                num_results=num_results,
            )

        emit(
            "search_results",
            {
                "keyword": keyword,
                "mode": "gpt",
                "type": search_type,
                "results": search_results,
            },
        )
    except Exception as e:
        print(f"Search error: {str(e)}")
        emit("error", {"message": f"Search error: {str(e)}"})


def google_custom_search(query, search_type="text"):
    """Perform Google Custom Search"""
    url = "https://www.googleapis.com/customsearch/v1"

    # Handle 'both' search type by making two separate API calls
    if search_type == "both":
        text_results = google_custom_search(query, "text")
        image_results = google_custom_search(query, "image")
        return {"text": text_results, "image": image_results}

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    if search_type == "image":
        params["searchType"] = "image"

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    results = []

    if "items" in data:
        for item in data["items"]:
            if search_type == "image":
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context": item.get("snippet", ""),
                    }
                )
            else:
                # Extract image from pagemap if available
                pagemap = item.get("pagemap", {})
                image_url = None

                # Try cse_image first, then cse_thumbnail, then metatags og:image
                if "cse_image" in pagemap and len(pagemap["cse_image"]) > 0:
                    image_url = pagemap["cse_image"][0].get("src", "")
                elif "cse_thumbnail" in pagemap and len(pagemap["cse_thumbnail"]) > 0:
                    image_url = pagemap["cse_thumbnail"][0].get("src", "")
                elif "metatags" in pagemap and len(pagemap["metatags"]) > 0:
                    image_url = pagemap["metatags"][0].get("og:image", "")

                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "image": image_url,  # Add image from pagemap
                    }
                )

    return results


@socketio.on("clear_session")
def handle_clear_session():
    """Clear transcription session"""
    session_id = request.sid

    # Save logs from old session before clearing
    if session_id in transcription_sessions:
        old_session = transcription_sessions[session_id]
        old_session._log_event(
            "session_cleared",
            {
                "total_words": len(old_session.words),
                "total_searches": sum(
                    1
                    for e in old_session.event_log
                    if e["event_type"] == "search_action"
                ),
            },
        )
        log_file = old_session.save_logs_to_file()
        print(f"[Session] Session cleared: {session_id}, logs saved to: {log_file}")

    # Create new session
    transcription_sessions[session_id] = TranscriptionSession(session_id)
    emit("session_cleared", {"status": "Session cleared"})


@socketio.on("clear_srt")
def handle_clear_srt():
    """Clear SRT data from session"""
    session_id = request.sid

    if session_id in transcription_sessions:
        session = transcription_sessions[session_id]
        if hasattr(session, "srt_entries"):
            del session.srt_entries
        if hasattr(session, "last_srt_text"):
            del session.last_srt_text
        session._log_event("srt_cleared", {})
        print(f"[SRT] Cleared SRT data for session {session_id}")


@socketio.on("check_srt_for_media")
def handle_check_srt_for_media(data):
    """Check if matching SRT file exists for uploaded media file"""
    session_id = request.sid
    media_filename = data.get("filename", "")

    if not media_filename:
        return

    # Get base name without extension
    base_name = os.path.splitext(media_filename)[0]

    # Search for matching SRT in srt/ directory (including subdirectories)
    srt_dir = os.path.join(os.path.dirname(__file__), "srt")
    srt_path = None

    # Check direct match first
    for root, dirs, files in os.walk(srt_dir):
        for file in files:
            if file.endswith(".srt"):
                file_base = os.path.splitext(file)[0]
                # Check if SRT filename matches media filename
                if file_base == base_name or file_base.startswith(base_name):
                    srt_path = os.path.join(root, file)
                    break
        if srt_path:
            break

    if srt_path and os.path.exists(srt_path):
        try:
            with open(srt_path, "r", encoding="utf-8") as f:
                srt_content = f.read()

            entries = parse_srt(srt_content)
            print(
                f"[SRT] Auto-loaded matching SRT: {srt_path} ({len(entries)} entries)"
            )

            if session_id not in transcription_sessions:
                transcription_sessions[session_id] = TranscriptionSession(session_id)

            session = transcription_sessions[session_id]
            session.srt_entries = entries
            session.srt_index = 0

            session._log_event(
                "srt_auto_loaded",
                {
                    "media_file": media_filename,
                    "srt_file": os.path.basename(srt_path),
                    "entry_count": len(entries),
                },
            )

            emit(
                "srt_loaded",
                {
                    "count": len(entries),
                    "status": f"Auto-loaded SRT: {os.path.basename(srt_path)}",
                    "auto": True,
                },
            )
        except Exception as e:
            print(f"[SRT] Error auto-loading SRT: {e}")
    else:
        print(f"[SRT] No matching SRT found for: {media_filename}")
        emit("srt_not_found", {"filename": media_filename})


@socketio.on("load_srt")
def handle_load_srt(data):
    """Load transcription from SRT file content"""
    session_id = request.sid
    srt_content = data.get("content", "")

    if session_id not in transcription_sessions:
        transcription_sessions[session_id] = TranscriptionSession(session_id)

    session = transcription_sessions[session_id]

    try:
        entries = parse_srt(srt_content)
        print(f"[SRT] Parsed {len(entries)} subtitle entries")

        # Store SRT entries for timed playback
        session.srt_entries = entries
        session.srt_index = 0

        # Log SRT load event
        session._log_event(
            "srt_loaded",
            {
                "entry_count": len(entries),
                "total_duration_ms": entries[-1][1] if entries else 0,
            },
        )

        emit("srt_loaded", {"count": len(entries), "status": "SRT file loaded"})
    except Exception as e:
        print(f"[SRT] Error parsing SRT: {e}")
        emit("error", {"message": f"SRT parse error: {str(e)}"})


@socketio.on("srt_time_update")
def handle_srt_time_update(data):
    """Send transcription based on current video time"""
    session_id = request.sid
    current_time_ms = data.get("time_ms", 0)

    if session_id not in transcription_sessions:
        return

    session = transcription_sessions[session_id]

    if not hasattr(session, "srt_entries") or not session.srt_entries:
        return

    # Find entries that should be displayed at current time
    for start_ms, end_ms, text in session.srt_entries:
        if start_ms <= current_time_ms <= end_ms:
            # Check if this text was already added recently
            if not hasattr(session, "last_srt_text") or session.last_srt_text != text:
                session.last_srt_text = text
                session.add_text(text)

                # Log SRT transcription event
                session._log_event(
                    "srt_transcription",
                    {
                        "video_time_ms": current_time_ms,
                        "subtitle_start_ms": start_ms,
                        "subtitle_end_ms": end_ms,
                        "text": text,
                        "word_count": len(text.split()),
                    },
                )

                emit(
                    "transcription",
                    {"text": text, "source": "srt", "is_complete": True},
                )
            break


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
