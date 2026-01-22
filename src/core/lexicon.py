"""Lexicon loading and text preprocessing for MorseQuery."""

import os

import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

from src.core.config import CONTRACTIONS

# Initialize NLTK lemmatizer
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("[NLTK] Downloading WordNet data...")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Global lexicon data
lexicon_df = None
lexicon_dict = {}


def load_lexicon():
    """Load OpenLexicon.xlsx and create lookup dictionary."""
    global lexicon_df, lexicon_dict
    try:
        lexicon_path = os.path.join(
            os.path.dirname(__file__), "../../data/lexicon/OpenLexicon.xlsx"
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


def preprocess_word(word):
    """Preprocess word for lexicon lookup.

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


def get_word_frequency(word: str) -> float | None:
    """Get the frequency value for a word from the lexicon.

    Args:
        word: The word to look up (will be preprocessed)

    Returns:
        Frequency value or None if not found
    """
    cleaned = preprocess_word(word)
    if not cleaned:
        return None
    return lexicon_dict.get(cleaned)


def is_rare_word(word: str, threshold: float = 3.0) -> bool:
    """Check if a word is rare (low frequency).

    Args:
        word: The word to check
        threshold: Frequency threshold (words below this are considered rare)

    Returns:
        True if the word is rare (not in lexicon, NaN, or freq < threshold)
    """
    freq = get_word_frequency(word)
    if freq is None:
        return True
    if pd.isna(freq):
        return True
    return freq < threshold


# Load lexicon on module import
load_lexicon()
