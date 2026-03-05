"""Configuration constants and environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Always prefer project-root .env over live/.env
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
CARTESIA_API_KEY = os.environ.get("CARTESIA_API_KEY")

# OpenAI Realtime API
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

# Audio settings
AUDIO_RATE = 24000
AUDIO_CHUNK = 4800
DEFAULT_AUDIO_FILE = "../mp3/tariffs_clips/clip_002.mp3"

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
TTS_LOG_DIR = LOG_DIR / "tts"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
TTS_LOG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
