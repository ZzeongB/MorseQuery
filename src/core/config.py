"""Configuration and constants for MorseQuery."""

import os
import re
import warnings

from dotenv import load_dotenv

# Ignore FP16 warning from Whisper
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

load_dotenv()

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

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

# Google Custom Search API config
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

# OpenAI API config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google/Gemini API config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini Live API configuration
GEMINI_MODEL = "models/gemini-2.0-flash-exp"
GEMINI_SEND_SAMPLE_RATE = 16000
GEMINI_CHUNK_SIZE = 1024

# Transcription-only config (simple, accurate speech recognition)
GEMINI_TRANSCRIPTION_CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": """You are a speech-to-text transcription system. Your ONLY job is to transcribe exactly what you hear.

Rules:
- Output ONLY the transcribed text, nothing else
- Transcribe verbatim in the original language
- Do not add any labels, headers, or formatting
- Do not summarize or interpret
- Do not add punctuation unless clearly spoken
- Output text immediately as you hear it

Example: If someone says "Hello world how are you", output exactly: Hello world how are you""",
}

# Inference-only config (listen and wait for search query request)
GEMINI_INFERENCE_CONFIG = {
    "response_modalities": ["TEXT"],
    "system_instruction": """You are a silent audio listener and search assistant for educational content.

IMPORTANT: Do NOT respond or output anything while listening to audio.
- Stay completely silent during audio streaming
- Do NOT transcribe or summarize
- Do NOT acknowledge hearing anything
- Just listen and remember what you hear

ONLY respond when you receive a text message asking for a search suggestion.

When asked for a search query, respond in this EXACT format:
[SearchQuery]
Term: <single search keyword or short phrase>
Definition: <one sentence explaining what it is>
Done.

Example response:
[SearchQuery]
Term: mitochondria
Definition: An organelle found in cells that generates most of the cell's ATP energy
Done.

CRITICAL - Only suggest DIFFICULT or TECHNICAL terms:
- Technical/scientific terms (e.g., photosynthesis, neurotransmitter, algorithm)
- Domain-specific jargon (e.g., amortization, jurisprudence, epistemology)
- Proper nouns that need context (e.g., Higgs boson, Turing machine)
- Foreign or uncommon words

DO NOT suggest common everyday words like:
- Basic verbs (go, make, think, say)
- Common nouns (house, car, food, people)
- Simple adjectives (good, bad, big, small)
- Everyday concepts (morning, money, family)

Rules:
- NEVER respond to audio input - only listen silently
- ONLY respond when receiving a text prompt
- Pick ONLY difficult/technical/unfamiliar terms
- If no difficult terms were heard, do not respond
- Keep Term short (1-4 words)
- Keep Definition to one sentence
- Always end with "Done." on its own line""",
}

# Study mode config (with captions, summary, terms)
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
