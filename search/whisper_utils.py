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
    model_name: str = "small",
    language: str | None = "en",
    word_timestamps: bool = True,
    beam_size: int = 5,
) -> dict[str, Any]:
    """Transcribe an audio file and return the raw Whisper result."""
    model = whisper.load_model(model_name)
    return model.transcribe(
        str(audio_path),
        language=language,
        word_timestamps=word_timestamps,
        verbose=False,
        beam_size=beam_size,
        condition_on_previous_text=False,
        temperature=0.0,
    )
