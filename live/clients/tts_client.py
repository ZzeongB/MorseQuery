"""Cartesia TTS client for converting text to speech."""

import base64
import threading
from typing import Optional

import requests
from config import CARTESIA_API_KEY
from flask_socketio import SocketIO
from logger import log_print

# Cartesia API settings
CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2025-04-16"

# Default voice settings
DEFAULT_MODEL_ID = "sonic-3"
DEFAULT_VOICE_ID = "67192626-0be9-4f45-b660-580d318c994d"  # Barbershop Man


class TTSClient:
    """Client for Cartesia TTS API."""

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.voice_id = voice_id
        self.model_id = model_id

        log_print(
            "INFO",
            "TTSClient created",
            session_id=session_id,
            voice_id=voice_id,
        )

    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """Synthesize text to speech and return audio bytes.

        Args:
            text: Text to convert to speech
            language: Language code (default: "en")

        Returns:
            Audio bytes in WAV format, or None if failed
        """
        if not text or not text.strip():
            log_print("WARN", "Empty text for TTS", session_id=self.session_id)
            return None

        if not CARTESIA_API_KEY:
            log_print("ERROR", "CARTESIA_API_KEY not set", session_id=self.session_id)
            return None

        payload = {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": 24000,
            },
            "language": language,
            "generation_config": {"volume": 1, "speed": 1.4, "emotion": "neutral"},
        }

        headers = {
            "Cartesia-Version": CARTESIA_VERSION,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(
                CARTESIA_API_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )

            if response.status_code == 200:
                log_print(
                    "INFO",
                    f"TTS synthesis successful, {len(response.content)} bytes",
                    session_id=self.session_id,
                )
                return response.content
            else:
                log_print(
                    "ERROR",
                    f"TTS API error: {response.status_code} - {response.text}",
                    session_id=self.session_id,
                )
                return None

        except requests.RequestException as e:
            log_print(
                "ERROR",
                f"TTS request failed: {e}",
                session_id=self.session_id,
            )
            return None

    def synthesize_and_emit(
        self,
        text: str,
        event_name: str = "tts_audio",
        language: str = "en",
    ) -> None:
        """Synthesize text and emit audio via Socket.IO.

        Args:
            text: Text to convert to speech
            event_name: Socket.IO event name to emit
            language: Language code
        """
        audio_bytes = self.synthesize(text, language)

        if audio_bytes:
            # Encode as base64 for JSON transport
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            self.sio.emit(
                event_name,
                {
                    "audio": audio_b64,
                    "format": "wav",
                    "sample_rate": 24000,
                    "text": text,
                },
            )
            log_print(
                "INFO",
                f"TTS audio emitted: {event_name}",
                session_id=self.session_id,
            )
        else:
            self.sio.emit(
                f"{event_name}_error",
                {"error": "TTS synthesis failed", "text": text},
            )

    def synthesize_async(
        self,
        text: str,
        event_name: str = "tts_audio",
        language: str = "en",
    ) -> None:
        """Synthesize text asynchronously in a background thread.

        Args:
            text: Text to convert to speech
            event_name: Socket.IO event name to emit
            language: Language code
        """
        thread = threading.Thread(
            target=self.synthesize_and_emit,
            args=(text, event_name, language),
            daemon=True,
        )
        thread.start()
