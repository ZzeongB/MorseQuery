"""Configuration constants and environment variables."""

import os
from pathlib import Path

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# OpenAI Realtime API
OPENAI_REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"

# Audio settings
AUDIO_RATE = 24000
AUDIO_CHUNK = 4800
DEFAULT_AUDIO_FILE = "../mp3/narration.mp3"
# Auto mode interval (seconds)
AUTO_INTERVAL = 5

# Paths
BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
TEMPLATES_DIR = BASE_DIR / "templates"

# Ensure directories exist
LOG_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)
