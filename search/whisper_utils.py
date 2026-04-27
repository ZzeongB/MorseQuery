"""Reusable helpers for local transcription with openai-whisper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import whisper


def load_audio_file(audio_path: str | Path) -> tuple[list[float], int]:
    """Load an audio file using Whisper's preprocessing pipeline."""
    audio = whisper.load_audio(str(audio_path))
    return audio, whisper.audio.SAMPLE_RATE


def transcribe_file(
    audio_path: str | Path,
    *,
    model_name: str = "tiny",
    language: str | None = "en",
    word_timestamps: bool = True,
) -> dict[str, Any]:
    """Transcribe an audio file and return the raw Whisper result."""
    model = whisper.load_model(model_name)
    return model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
        condition_on_previous_text=False,
        temperature=0.0,
    )
