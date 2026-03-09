"""Streaming Cartesia TTS client using WebSocket for real-time text-to-speech."""

import base64
import threading
import time
from collections import deque
from datetime import datetime
from typing import Callable, Optional

from cartesia import Cartesia
from config import CARTESIA_API_KEY
from flask_socketio import SocketIO
from logger import get_logger, get_session_subdir, log_print

# Default voice settings
DEFAULT_MODEL_ID = "sonic-3"
DEFAULT_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"  # Barbershop Man

# Audio output settings
STREAMING_SAMPLE_RATE = 24000
STREAMING_ENCODING = "pcm_s16le"
STREAMING_CONTAINER = "raw"


class StreamingTTSClient:
    """WebSocket-based streaming TTS client for Cartesia.

    Supports:
    - Streaming text input: push text chunks as they arrive
    - Streaming audio output: emit audio chunks to client as they're generated
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        language: str = "en",
        chunk_event: str = "tts_audio_chunk",
        done_event: str = "tts_stream_done",
        emit_to: Optional[str] = None,
        event_extra: Optional[dict] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.voice_id = voice_id
        self.model_id = model_id
        self.language = language
        self.chunk_event = chunk_event
        self.done_event = done_event
        self.emit_to = emit_to
        self.event_extra = event_extra or {}

        # Cartesia client
        self.client: Optional[Cartesia] = None
        self._connection_manager = None  # TTSResourceConnectionManager (for cleanup)
        self._connection = None  # Actual connection (from __enter__)
        self._context = None
        self._context_lock = threading.Lock()

        # State tracking
        self._is_streaming = False
        self._receive_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Text input queue for streaming
        self._text_queue: deque[str] = deque()
        self._text_queue_lock = threading.Lock()

        # Audio chunk buffer for logging
        self._audio_chunks: list[bytes] = []
        self._current_text: str = ""
        self._stream_start_time: float = 0.0

        # Logging
        self.logger = get_logger(session_id)
        self._tts_counter = 0
        self._tts_counter_lock = threading.Lock()

        log_print(
            "INFO",
            "StreamingTTSClient created",
            session_id=session_id,
            voice_id=voice_id,
            model_id=model_id,
        )

    def _init_client(self) -> bool:
        """Initialize Cartesia client if not already initialized."""
        if self.client is not None:
            return True

        if not CARTESIA_API_KEY:
            log_print("ERROR", "CARTESIA_API_KEY not set", session_id=self.session_id)
            return False

        try:
            self.client = Cartesia(api_key=CARTESIA_API_KEY)
            log_print("INFO", "Cartesia client initialized", session_id=self.session_id)
            return True
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to initialize Cartesia client: {e}",
                session_id=self.session_id,
            )
            return False

    def start_stream(self, initial_text: str = "") -> bool:
        """Start a new streaming TTS session.

        Args:
            initial_text: Optional initial text to start with

        Returns:
            True if stream started successfully
        """
        if not self._init_client():
            return False

        if self._is_streaming:
            log_print(
                "WARN",
                "Stream already active, stopping previous stream",
                session_id=self.session_id,
            )
            self.stop_stream()

        try:
            self._stop_event.clear()
            self._audio_chunks = []
            self._current_text = initial_text
            self._stream_start_time = time.time()

            # Connect via WebSocket
            self._connection_manager = self.client.tts.websocket_connect()
            self._connection = self._connection_manager.__enter__()

            # Create context for this stream
            self._context = self._connection.context(
                model_id=self.model_id,
                voice={"mode": "id", "id": self.voice_id},
                output_format={
                    "container": STREAMING_CONTAINER,
                    "encoding": STREAMING_ENCODING,
                    "sample_rate": STREAMING_SAMPLE_RATE,
                },
                language=self.language,
            )

            self._is_streaming = True

            # Start receive thread
            self._receive_thread = threading.Thread(
                target=self._receive_audio_loop,
                daemon=True,
            )
            self._receive_thread.start()

            log_print(
                "INFO",
                "TTS stream started",
                session_id=self.session_id,
            )
            self.logger.log("tts_stream_started")

            # Push initial text if provided
            if initial_text:
                self.push_text(initial_text)

            # Emit status to client
            self.sio.emit(
                "tts_stream_started",
                {"session_id": self.session_id},
                to=self.emit_to,
            )
            return True

        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to start TTS stream: {e}",
                session_id=self.session_id,
            )
            self._cleanup()
            return False

    def push_text(self, text: str) -> bool:
        """Push text chunk to the TTS stream.

        Args:
            text: Text chunk to synthesize

        Returns:
            True if text was pushed successfully
        """
        if not text or not text.strip():
            return False

        if not self._is_streaming:
            log_print(
                "WARN",
                "Cannot push text: stream not active",
                session_id=self.session_id,
            )
            return False

        with self._context_lock:
            if self._context is None:
                return False
            try:
                self._context.push(text)
                self._current_text += text

                log_print(
                    "DEBUG",
                    f"Pushed text chunk: {text[:50]}...",
                    session_id=self.session_id,
                    chunk_len=len(text),
                )
                return True
            except Exception as e:
                log_print(
                    "ERROR",
                    f"Failed to push text: {e}",
                    session_id=self.session_id,
                )
                return False

    def finish_input(self) -> bool:
        """Signal that no more text input will be sent.

        Call this when all text chunks have been pushed.

        Returns:
            True if signal was sent successfully
        """
        with self._context_lock:
            if self._context is None:
                return False
            try:
                self._context.no_more_inputs()
                log_print(
                    "INFO",
                    "TTS input finished",
                    session_id=self.session_id,
                )
                return True
            except Exception as e:
                log_print(
                    "ERROR",
                    f"Failed to signal input finish: {e}",
                    session_id=self.session_id,
                )
                return False

    def _receive_audio_loop(self) -> None:
        """Background thread to receive and emit audio chunks."""
        total_bytes = 0
        chunk_count = 0

        try:
            with self._context_lock:
                ctx = self._context

            if ctx is None:
                return

            for response in ctx.receive():
                if self._stop_event.is_set():
                    break

                if response.type == "chunk" and response.audio:
                    audio_bytes = response.audio
                    total_bytes += len(audio_bytes)
                    chunk_count += 1

                    # Store for logging
                    self._audio_chunks.append(audio_bytes)

                    # Emit to client as base64
                    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                    self.sio.emit(
                        self.chunk_event,
                        {
                            "audio": audio_b64,
                            "format": "pcm_s16le",
                            "sample_rate": STREAMING_SAMPLE_RATE,
                            "chunk_index": chunk_count,
                            **self.event_extra,
                        },
                        to=self.emit_to,
                    )

                    if chunk_count % 10 == 0:
                        log_print(
                            "DEBUG",
                            f"TTS chunks emitted: {chunk_count}, bytes: {total_bytes}",
                            session_id=self.session_id,
                        )

                elif response.type == "done":
                    log_print(
                        "INFO",
                        "TTS stream done",
                        session_id=self.session_id,
                        total_chunks=chunk_count,
                        total_bytes=total_bytes,
                    )
                    break

        except Exception as e:
            if not self._stop_event.is_set():
                log_print(
                    "ERROR",
                    f"TTS receive error: {e}",
                    session_id=self.session_id,
                )
        finally:
            # Save audio and cleanup
            self._save_audio()
            self._emit_stream_done(chunk_count, total_bytes)
            self._cleanup()

    def _emit_stream_done(self, chunk_count: int, total_bytes: int) -> None:
        """Emit stream completion event to client."""
        duration_ms = (time.time() - self._stream_start_time) * 1000 if self._stream_start_time else 0
        self.sio.emit(
            self.done_event,
            {
                "session_id": self.session_id,
                "chunk_count": chunk_count,
                "total_bytes": total_bytes,
                "duration_ms": round(duration_ms, 2),
                "text": self._current_text,
                "sample_rate": STREAMING_SAMPLE_RATE,
                "stopped": self._stop_event.is_set(),
                **self.event_extra,
            },
            to=self.emit_to,
        )

    def _save_audio(self) -> Optional[str]:
        """Save streamed audio to file for logging."""
        if not self._audio_chunks:
            return None

        try:
            with self._tts_counter_lock:
                self._tts_counter += 1
                tts_id = self._tts_counter

            # Combine all chunks
            raw_audio = b"".join(self._audio_chunks)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tts_dir = get_session_subdir(self.session_id, "tts")
            filename = f"{timestamp}_{self.session_id}_stream_{tts_id:04d}.raw"
            filepath = tts_dir / filename

            # Save raw PCM
            with open(filepath, "wb") as f:
                f.write(raw_audio)

            # Save metadata
            meta_filepath = tts_dir / f"{timestamp}_{self.session_id}_stream_{tts_id:04d}.txt"
            duration_sec = len(raw_audio) / (STREAMING_SAMPLE_RATE * 2)  # 16-bit = 2 bytes
            with open(meta_filepath, "w", encoding="utf-8") as f:
                f.write(f"Text: {self._current_text}\n")
                f.write(f"Session: {self.session_id}\n")
                f.write(f"TTS ID: {tts_id}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Size: {len(raw_audio)} bytes\n")
                f.write(f"Duration: {duration_sec:.2f} sec\n")
                f.write(f"Format: PCM S16LE {STREAMING_SAMPLE_RATE}Hz mono\n")

            log_print(
                "INFO",
                f"Streaming TTS saved: {filename}",
                session_id=self.session_id,
                tts_id=tts_id,
                duration_sec=duration_sec,
            )
            self.logger.log(
                "streaming_tts_saved",
                tts_id=tts_id,
                tts_text=self._current_text[:200],
                tts_audio_path=str(filepath.resolve()),
                tts_size_bytes=len(raw_audio),
                tts_duration_sec=duration_sec,
            )
            return str(filepath)

        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to save streaming TTS: {e}",
                session_id=self.session_id,
            )
            return None

    def stop_stream(self) -> None:
        """Stop the current TTS stream."""
        log_print("INFO", "Stopping TTS stream", session_id=self.session_id)
        self._stop_event.set()
        self._cleanup()

    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_streaming = False

        with self._context_lock:
            self._context = None

        self._connection = None
        if self._connection_manager:
            try:
                self._connection_manager.__exit__(None, None, None)
            except Exception:
                pass
            self._connection_manager = None

        self._audio_chunks = []
        self._current_text = ""

    def synthesize_streaming(
        self,
        text: str,
        on_chunk: Optional[Callable[[bytes, int], None]] = None,
    ) -> bool:
        """Convenience method: synthesize text with streaming output.

        Args:
            text: Full text to synthesize
            on_chunk: Optional callback called for each audio chunk (bytes, chunk_index)

        Returns:
            True if synthesis completed successfully
        """
        if not self.start_stream():
            return False

        # Push all text at once
        if not self.push_text(text):
            self.stop_stream()
            return False

        # Signal end of input
        self.finish_input()

        # Wait for completion (receive thread handles the rest)
        if self._receive_thread:
            self._receive_thread.join(timeout=60)

        return True

    def synthesize_from_generator(
        self,
        text_generator,
        emit_event: str = "tts_audio_chunk",
    ) -> bool:
        """Synthesize from a text generator (streaming input + output).

        Args:
            text_generator: Iterator/generator yielding text chunks
            emit_event: Socket.IO event name for audio chunks

        Returns:
            True if synthesis completed successfully
        """
        if not self.start_stream():
            return False

        try:
            # Push text chunks as they arrive
            for text_chunk in text_generator:
                if self._stop_event.is_set():
                    break
                if text_chunk:
                    self.push_text(text_chunk)

            # Signal end of input
            self.finish_input()

            # Wait for completion
            if self._receive_thread:
                self._receive_thread.join(timeout=60)

            return True

        except Exception as e:
            log_print(
                "ERROR",
                f"Generator synthesis failed: {e}",
                session_id=self.session_id,
            )
            self.stop_stream()
            return False

    @property
    def is_streaming(self) -> bool:
        """Check if a stream is currently active."""
        return self._is_streaming

    def close(self) -> None:
        """Close the client and release all resources."""
        self.stop_stream()
        self.client = None
        log_print("INFO", "StreamingTTSClient closed", session_id=self.session_id)
