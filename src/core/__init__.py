"""Core modules for MorseQuery."""

from src.core.config import (
    CONTRACTIONS,
    GEMINI_CHUNK_SIZE,
    GEMINI_MODEL,
    GEMINI_SEND_SAMPLE_RATE,
    GEMINI_STUDY_CONFIG,
    GOOGLE_API_KEY,
    GOOGLE_SEARCH_API_KEY,
    GOOGLE_SEARCH_ENGINE_ID,
    LOGS_DIR,
    OPENAI_API_KEY,
    SECTION_PATTERNS,
    SECTION_RE,
)
from src.core.lexicon import (
    get_word_frequency,
    is_rare_word,
    lexicon_dict,
    load_lexicon,
    preprocess_word,
)
from src.core.session import TranscriptionSession
