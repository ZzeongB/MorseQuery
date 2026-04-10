"""Centralized path configuration for the search app."""

from pathlib import Path

SEARCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = SEARCH_DIR.parent

SEARCH_DATA_DIR = SEARCH_DIR / "data"
LEGACY_DATA_DIR = REPO_ROOT / "data"


def _prefer_search_dir(relative_path: str) -> Path:
    """Use search-local data when present, otherwise fall back to the legacy root path."""
    preferred = SEARCH_DATA_DIR / relative_path
    if preferred.exists():
        return preferred
    return LEGACY_DATA_DIR / relative_path


MP3_DIR = SEARCH_DATA_DIR / "mp3" if (SEARCH_DATA_DIR / "mp3").exists() else REPO_ROOT / "mp3"
TRANSCRIPT_DIR = _prefer_search_dir("transcripts")
KEYWORDS_DIR = _prefer_search_dir("keywords")
KEYWORDS2_DIR = _prefer_search_dir("keywords2")
LEXICON_PATH = _prefer_search_dir("lexicon/OpenLexicon.xlsx")
STUDY_DIR = _prefer_search_dir("study")
INTERRUPTIONS_DIR = _prefer_search_dir("interruptions")
LOGS_DIR = SEARCH_DIR / "logs" / "study"
WORD_CLIPS_DIR = _prefer_search_dir("word_clips")
