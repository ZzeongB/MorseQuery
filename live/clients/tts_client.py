"""Cartesia TTS client for converting text to speech."""

import base64
import io
import threading
import wave
from collections import OrderedDict
from datetime import datetime
from time import perf_counter
from typing import Callable, Optional

import pyaudio
import requests
from config import CARTESIA_API_KEY
from flask_socketio import SocketIO
from logger import get_logger, get_session_subdir, log_print

# Cartesia API settings
CARTESIA_API_URL = "https://api.cartesia.ai/tts/bytes"
CARTESIA_VERSION = "2025-04-16"

# Default voice settings
DEFAULT_MODEL_ID = "sonic-turbo-2025-06-04"  # "sonic-3"
DEFAULT_VOICE_ID = "67192626-0be9-4f45-b660-580d318c994d"  # Barbershop Man

# Audio playback settings
TTS_SAMPLE_RATE = 24000
TTS_CHANNELS = 1
TTS_SAMPLE_WIDTH = 2  # 16-bit
TTS_CACHE_MAX_ITEMS = 128


class TTSClient:
    """Client for Cartesia TTS API with queue support for deferred playback."""

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        output_device_index: Optional[int] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_device_index = output_device_index

        # Audio queue for deferred playback
        self.audio_queue: list[tuple[bytes, str]] = []  # (audio_bytes, text)
        self._queue_lock = threading.Lock()
        self.is_playing = False
        self._stop_playback_event = threading.Event()
        self._play_thread: Optional[threading.Thread] = None

        # TTS logging
        self.logger = get_logger(session_id)
        self._tts_counter = 0
        self._tts_counter_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._synth_cache: OrderedDict[tuple[str, str, str, str], bytes] = OrderedDict()

        log_print(
            "INFO",
            "TTSClient created",
            session_id=session_id,
            voice_id=voice_id,
            output_device_index=output_device_index,
        )

    def synthesize(self, text: str, language: str = "en") -> Optional[bytes]:
        """Synthesize text to speech and return audio bytes.

        Args:
            text: Text to convert to speech
            language: Language code (default: "en")

        Returns:
            Audio bytes in WAV format, or None if failed
        """
        normalized_text = (text or "").strip()
        if not normalized_text:
            log_print("WARN", "Empty text for TTS", session_id=self.session_id)
            return None

        if not CARTESIA_API_KEY:
            log_print("ERROR", "CARTESIA_API_KEY not set", session_id=self.session_id)
            return None

        if not self.voice_id:
            log_print("ERROR", "Voice ID not set", session_id=self.session_id)
            return None

        cache_key = (
            self.voice_id,
            self.model_id,
            language or "en",
            normalized_text,
        )
        with self._cache_lock:
            cached = self._synth_cache.get(cache_key)
            if cached is not None:
                self._synth_cache.move_to_end(cache_key)
                log_print(
                    "DEBUG",
                    "TTS cache hit",
                    session_id=self.session_id,
                    chars=len(normalized_text),
                )
                return cached

        payload = {
            "model_id": self.model_id,
            "transcript": normalized_text,
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
            "generation_config": {"volume": 1, "speed": 1, "emotion": "neutral"},
        }

        headers = {
            "Cartesia-Version": CARTESIA_VERSION,
            "Authorization": f"Bearer {CARTESIA_API_KEY}",
            "Content-Type": "application/json",
        }

        try:
            request_started = perf_counter()
            response = requests.post(
                CARTESIA_API_URL,
                json=payload,
                headers=headers,
                timeout=30,
            )
            generation_latency_ms = (perf_counter() - request_started) * 1000

            if response.status_code == 200:
                audio_bytes = response.content
                # Save TTS result to file
                self._save_tts_audio(
                    audio_bytes,
                    normalized_text,
                    generation_latency_ms=generation_latency_ms,
                )
                with self._cache_lock:
                    self._synth_cache[cache_key] = audio_bytes
                    self._synth_cache.move_to_end(cache_key)
                    while len(self._synth_cache) > TTS_CACHE_MAX_ITEMS:
                        self._synth_cache.popitem(last=False)
                log_print(
                    "INFO",
                    f"TTS synthesis successful, {len(audio_bytes)} bytes",
                    session_id=self.session_id,
                    generation_latency_ms=round(generation_latency_ms, 2),
                )
                return audio_bytes
            else:
                log_print(
                    "ERROR",
                    f"TTS API error: {response.status_code} - {response.text}",
                    session_id=self.session_id,
                    generation_latency_ms=round(generation_latency_ms, 2),
                )
                return None

        except requests.RequestException as e:
            log_print(
                "ERROR",
                f"TTS request failed: {e}",
                session_id=self.session_id,
            )
            return None

    def _save_tts_audio(
        self,
        audio_bytes: bytes,
        text: str,
        generation_latency_ms: Optional[float] = None,
    ) -> Optional[str]:
        """Save TTS audio to session-scoped tts directory."""
        try:
            with self._tts_counter_lock:
                self._tts_counter += 1
                tts_id = self._tts_counter

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tts_dir = get_session_subdir(self.session_id, "tts")
            filename = f"{timestamp}_{self.session_id}_{tts_id:04d}.wav"
            filepath = tts_dir / filename

            with open(filepath, "wb") as f:
                f.write(audio_bytes)

            meta_filepath = tts_dir / f"{timestamp}_{self.session_id}_{tts_id:04d}.txt"
            with open(meta_filepath, "w", encoding="utf-8") as f:
                f.write(f"Text: {text}\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"TTS ID: {tts_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Size: {len(audio_bytes)} bytes\n")
                if generation_latency_ms is not None:
                    f.write(f"Generation Latency (ms): {generation_latency_ms:.2f}\n")
                f.write(f"Audio Path: {filepath.resolve()}\n")

            log_print(
                "INFO",
                f"TTS saved: {filename}",
                session_id=self.session_id,
                tts_id=tts_id,
            )
            self.logger.log(
                "tts_saved",
                tts_id=tts_id,
                tts_text=text,
                tts_audio_path=str(filepath.resolve()),
                tts_meta_path=str(meta_filepath.resolve()),
                tts_size_bytes=len(audio_bytes),
                tts_generation_latency_ms=generation_latency_ms,
            )
            return str(filepath)

        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to save TTS audio: {e}",
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

    # -------------------------
    # Queue-based TTS methods
    # -------------------------

    def queue_audio(self, text: str, language: str = "en") -> bool:
        """Synthesize TTS and add to queue (don't play immediately).

        Args:
            text: Text to convert to speech
            language: Language code

        Returns:
            True if audio was queued successfully, False otherwise
        """
        audio_bytes = self.synthesize(text, language)
        if audio_bytes:
            with self._queue_lock:
                self.audio_queue.append((audio_bytes, text))
            log_print(
                "INFO",
                f"TTS queued: {text[:50]}...",
                session_id=self.session_id,
                queue_size=len(self.audio_queue),
            )
            return True
        return False

    def queue_audio_bytes(self, audio_bytes: bytes, text: str = "") -> bool:
        """Queue already synthesized audio bytes."""
        if not audio_bytes:
            return False
        with self._queue_lock:
            self.audio_queue.append((audio_bytes, text))
            queue_size = len(self.audio_queue)
        log_print(
            "INFO",
            f"Prebuilt TTS queued: {text[:50]}...",
            session_id=self.session_id,
            queue_size=queue_size,
        )
        return True

    def queue_audio_async(self, text: str, language: str = "en") -> None:
        """Synthesize TTS asynchronously and add to queue.

        Args:
            text: Text to convert to speech
            language: Language code
        """
        thread = threading.Thread(
            target=self.queue_audio,
            args=(text, language),
            daemon=True,
        )
        thread.start()

    def queue_audio_with_callback(
        self,
        text: str,
        callback: Callable[[bool], None],
        language: str = "en",
    ) -> None:
        """Synthesize TTS asynchronously, add to queue, then call callback.

        This is used for parallel TTS/judgment execution. The callback is called
        when TTS synthesis completes (success or failure).

        Args:
            text: Text to convert to speech
            callback: Function called with True if queued successfully, False otherwise
            language: Language code
        """

        def _synthesize_and_callback():
            success = self.queue_audio(text, language)
            try:
                callback(success)
            except Exception as e:
                log_print(
                    "ERROR",
                    f"TTS callback error: {e}",
                    session_id=self.session_id,
                )

        thread = threading.Thread(
            target=_synthesize_and_callback,
            daemon=True,
        )
        thread.start()

    def play_queued(self, reason: str = "", emit_done: bool = True) -> bool:
        """Play all queued audio using PyAudio (server-side playback).

        Args:
            reason: Optional reason for playing (for logging)
            emit_done: Whether to emit tts_done when playback finishes

        Returns:
            True if audio was played, False if queue was empty or already playing
        """
        with self._queue_lock:
            if self.is_playing:
                log_print(
                    "WARN",
                    "Already playing TTS",
                    session_id=self.session_id,
                )
                return False

            if not self.audio_queue:
                log_print(
                    "DEBUG",
                    "No audio in queue to play",
                    session_id=self.session_id,
                )
                return False

            # Take all queued audio
            to_play = self.audio_queue.copy()
            self.audio_queue.clear()
            self.is_playing = True
            self._stop_playback_event.clear()

        log_print(
            "INFO",
            f"Playing {len(to_play)} queued TTS items",
            session_id=self.session_id,
            reason=reason,
        )

        # Emit event to notify client
        self.sio.emit("tts_playing", {"reason": reason, "count": len(to_play)})

        # Play in background thread
        thread = threading.Thread(
            target=self._play_audio_list,
            args=(to_play, emit_done),
            daemon=True,
        )
        self._play_thread = thread
        thread.start()
        return True

    def _play_audio_list(
        self, audio_list: list[tuple[bytes, str]], emit_done: bool = True
    ) -> None:
        """Play a list of audio bytes using PyAudio.

        Args:
            audio_list: List of (audio_bytes, text) tuples
            emit_done: Whether to emit tts_done when playback finishes
        """
        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            open_kwargs = {
                "format": pyaudio.paInt16,
                "channels": TTS_CHANNELS,
                "rate": TTS_SAMPLE_RATE,
                "output": True,
            }
            if self.output_device_index is not None:
                open_kwargs["output_device_index"] = self.output_device_index
            stream = pa.open(**open_kwargs)

            for audio_bytes, text in audio_list:
                if self._stop_playback_event.is_set():
                    break
                try:
                    # Parse WAV and extract PCM data
                    pcm_data = self._extract_pcm_from_wav(audio_bytes)
                    if pcm_data:
                        # Write in chunks so stop requests can interrupt promptly
                        chunk_size = 4096
                        for i in range(0, len(pcm_data), chunk_size):
                            if self._stop_playback_event.is_set():
                                break
                            stream.write(pcm_data[i : i + chunk_size])
                        log_print(
                            "DEBUG",
                            f"Played TTS: {text[:30]}...",
                            session_id=self.session_id,
                        )
                except Exception as e:
                    log_print(
                        "ERROR",
                        f"Error playing audio: {e}",
                        session_id=self.session_id,
                    )

        except Exception as e:
            log_print(
                "ERROR",
                f"PyAudio playback error: {e}",
                session_id=self.session_id,
            )
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass

            self.is_playing = False
            self._play_thread = None
            if emit_done:
                self.sio.emit("tts_done")

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> Optional[bytes]:
        """Extract raw PCM data from WAV bytes.

        Args:
            wav_bytes: WAV format audio bytes

        Returns:
            Raw PCM bytes, or None if parsing failed
        """
        try:
            with io.BytesIO(wav_bytes) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to extract PCM from WAV: {e}",
                session_id=self.session_id,
            )
            return None

    def clear_queue(self) -> int:
        """Clear all queued audio.

        Returns:
            Number of items cleared
        """
        with self._queue_lock:
            count = len(self.audio_queue)
            self.audio_queue.clear()

        if count > 0:
            log_print(
                "INFO",
                f"Cleared {count} TTS items from queue",
                session_id=self.session_id,
            )
        return count

    def stop_playback(self, wait: bool = False, timeout_sec: float = 0.8) -> None:
        """Stop current playback and clear queued audio.

        Args:
            wait: If True, wait briefly for active playback thread to exit
            timeout_sec: Max wait time when wait=True
        """
        self._stop_playback_event.set()
        cleared = self.clear_queue()
        stopped = None
        thread = self._play_thread
        if (
            wait
            and thread is not None
            and thread.is_alive()
            and threading.current_thread() is not thread
        ):
            thread.join(timeout=max(0.0, timeout_sec))
            stopped = not thread.is_alive()
        log_print(
            "INFO",
            "TTS stop requested",
            session_id=self.session_id,
            cleared=cleared,
            is_playing=self.is_playing,
            wait=wait,
            stopped=stopped,
        )

    def get_queue_size(self) -> int:
        """Get the number of items in the queue."""
        with self._queue_lock:
            return len(self.audio_queue)

    def has_queued_audio(self) -> bool:
        """Check if there is audio in the queue."""
        with self._queue_lock:
            return len(self.audio_queue) > 0
