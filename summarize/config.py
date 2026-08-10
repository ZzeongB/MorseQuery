"""Configuration constants and environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Always prefer project-root .env over summarize/.env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")

# OpenAI Realtime API
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-realtime"

# Audio settings
AUDIO_RATE = 24000
AUDIO_CHUNK = 2400  # 100ms chunks

# OpenAI Realtime Session Config (GA transcription session)
OPENAI_SESSION_CONFIG = {
    "type": "transcription",
    "audio": {
        "input": {
            "format": {
                "type": "audio/pcm",
                "rate": AUDIO_RATE,
            },
            "transcription": {
                "model": "gpt-realtime-whisper",
                "language": "en",
                "delay": "low",
            },
            "turn_detection": None,
        }
    },
}

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
QUIZ_DIR = BASE_DIR / "quiz"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
QUIZ_DIR.mkdir(exist_ok=True)
