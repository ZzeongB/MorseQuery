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
# 100ms chunks (was 200ms). This reduces "just-before-tap" miss window.
AUDIO_CHUNK = 2400
DEFAULT_AUDIO_FILE = "../mp3/tariffs_clips/clip_002.mp3"

# OpenAI Realtime Session Config (shared between RealtimeClient and SummaryClient)
# Note: "instructions" should be added separately per client
OPENAI_SESSION_CONFIG = {
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "turn_detection": {
        "type": "server_vad",
        "threshold": 0.5,
        "prefix_padding_ms": 300,
        "silence_duration_ms": 400,
        "create_response": False,
    },
    "input_audio_transcription": {
        "model": "gpt-4o-transcribe",
        "language": "en",
    },
}

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
TTS_LOG_DIR = LOG_DIR / "tts"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
TTS_LOG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
