"""Configuration and constants for MorseQuery Simple."""

import os
import warnings

from dotenv import load_dotenv

# Ignore FP16 warning
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

load_dotenv()

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# OpenAI API config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Google/Gemini API config (for grounding)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
