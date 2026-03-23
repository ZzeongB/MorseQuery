"""Streaming TTS client for browser playback."""

import base64
import threading
import time
from datetime import datetime
from typing import Optional

try:
    from cartesia import Cartesia
except ImportError:  # pragma: no cover - optional dependency
    Cartesia = None

from config import CARTESIA_API_KEY
from flask_socketio import SocketIO
from logger import get_logger, get_session_subdir, log_print

DEFAULT_MODEL_ID = "sonic-3"
DEFAULT_VOICE_ID = "f786b574-daa5-4673-aa0c-cbe3e8534c02"
STREAMING_SAMPLE_RATE = 24000
STREAMING_ENCODING = "pcm_s16le"
STREAMING_CONTAINER = "raw"


class StreamingTTSClient:
    """WebSocket-based Cartesia streaming TTS for Socket.IO clients."""

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str,
        voice_id: str = DEFAULT_VOICE_ID,
        model_id: str = DEFAULT_MODEL_ID,
        chunk_event: str = "streaming_tts_chunk",
        done_event: str = "streaming_tts_done",
        emit_to: Optional[str] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.voice_id = voice_id
        self.model_id = model_id
        self.chunk_event = chunk_event
        self.done_event = done_event
        self.emit_to = emit_to

        self.client = None
        self._connection_manager = None
        self._connection = None
        self._connection_lock = threading.Lock()
        self._context = None
        self._context_lock = threading.Lock()
        self._receive_thread: Optional[threading.Thread] = None
        self._audio_chunks: list[bytes] = []
        self._current_text = ""
        self._stream_start_time = 0.0
        self._stream_token = 0
        self._active_stream_token = 0
        self._stopped_stream_tokens: set[int] = set()
        self._is_streaming = False
        self.logger = get_logger(session_id)

    def _init_client(self) -> bool:
        if self.client is not None:
            return True
        if Cartesia is None:
            log_print("ERROR", "cartesia package is not installed", session_id=self.session_id)
            return False
        if not CARTESIA_API_KEY:
            log_print("ERROR", "CARTESIA_API_KEY not set", session_id=self.session_id)
            return False
        try:
            self.client = Cartesia(api_key=CARTESIA_API_KEY)
            return True
        except Exception as e:
            log_print("ERROR", f"Failed to init Cartesia client: {e}", session_id=self.session_id)
            return False

    def _ensure_connection(self) -> bool:
        with self._connection_lock:
            if self._connection is not None:
                return True
            try:
                self._connection_manager = self.client.tts.websocket_connect(
                    websocket_connection_options={"ping_interval": 10, "ping_timeout": 60}
                )
                self._connection = self._connection_manager.__enter__()
                return True
            except Exception as e:
                log_print("ERROR", f"Failed to open TTS websocket: {e}", session_id=self.session_id)
                self._connection_manager = None
                self._connection = None
                return False

    def _close_connection_unsafe(self) -> None:
        if self._connection_manager:
            try:
                self._connection_manager.__exit__(None, None, None)
            except Exception:
                pass
        self._connection_manager = None
        self._connection = None

    def _cleanup_stream(self, stream_token: Optional[int] = None) -> None:
        with self._context_lock:
            if stream_token is not None and stream_token != self._active_stream_token:
                return
            self._context = None
        self._is_streaming = False
        self._audio_chunks = []
        self._current_text = ""

    def _emit_done(self, stream_id: str, chunk_count: int, total_bytes: int, stopped: bool) -> None:
        duration_ms = (time.time() - self._stream_start_time) * 1000 if self._stream_start_time else 0
        self.sio.emit(
            self.done_event,
            {
                "stream_id": stream_id,
                "chunk_count": chunk_count,
                "total_bytes": total_bytes,
                "duration_ms": round(duration_ms, 2),
                "sample_rate": STREAMING_SAMPLE_RATE,
                "stopped": stopped,
            },
            to=self.emit_to,
        )

    def _save_audio(self) -> None:
        if not self._audio_chunks:
            return
        try:
            raw_audio = b"".join(self._audio_chunks)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tts_dir = get_session_subdir(self.session_id, "tts")
            filepath = tts_dir / f"{timestamp}_{self.session_id}_stream.raw"
            with open(filepath, "wb") as f:
                f.write(raw_audio)
            self.logger.log("streaming_tts_saved", tts_audio_path=str(filepath.resolve()), tts_size_bytes=len(raw_audio))
        except Exception as e:
            log_print("ERROR", f"Failed to save streaming TTS: {e}", session_id=self.session_id)

    def _receive_audio_loop(self, stream_token: int, stream_id: str) -> None:
        total_bytes = 0
        chunk_count = 0
        connection_error = False
        try:
            with self._context_lock:
                ctx = self._context
            if ctx is None:
                return
            for response in ctx.receive():
                with self._context_lock:
                    if stream_token in self._stopped_stream_tokens:
                        break
                if response.type == "chunk" and response.audio:
                    audio_bytes = response.audio
                    self._audio_chunks.append(audio_bytes)
                    total_bytes += len(audio_bytes)
                    chunk_count += 1
                    self.sio.emit(
                        self.chunk_event,
                        {
                            "stream_id": stream_id,
                            "audio": base64.b64encode(audio_bytes).decode("utf-8"),
                            "format": "pcm_s16le",
                            "sample_rate": STREAMING_SAMPLE_RATE,
                            "chunk_index": chunk_count,
                            "type": "summary",
                        },
                        to=self.emit_to,
                    )
                elif response.type == "done":
                    break
        except Exception as e:
            log_print("ERROR", f"TTS receive error: {e}", session_id=self.session_id)
            connection_error = True
        finally:
            self._save_audio()
            stopped = stream_token in self._stopped_stream_tokens
            self._emit_done(stream_id, chunk_count, total_bytes, stopped)
            self._cleanup_stream(stream_token)
            with self._context_lock:
                self._stopped_stream_tokens.discard(stream_token)
            if connection_error:
                with self._connection_lock:
                    self._close_connection_unsafe()

    def synthesize_streaming(self, text: str, stream_id: str) -> bool:
        normalized = (text or "").strip()
        if not normalized:
            return False
        if not self._init_client():
            return False
        if not self._ensure_connection():
            return False
        try:
            with self._context_lock:
                self._stream_token += 1
                stream_token = self._stream_token
                self._active_stream_token = stream_token
                self._stopped_stream_tokens.discard(stream_token)
            self._audio_chunks = []
            self._current_text = normalized
            self._stream_start_time = time.time()
            with self._connection_lock:
                self._context = self._connection.context(
                    model_id=self.model_id,
                    voice={"mode": "id", "id": self.voice_id},
                    output_format={
                        "container": STREAMING_CONTAINER,
                        "encoding": STREAMING_ENCODING,
                        "sample_rate": STREAMING_SAMPLE_RATE,
                    },
                    language="en",
                    generation_config={"volume": 1, "speed": 1, "emotion": "neutral"},
                )
            self._is_streaming = True
            self._receive_thread = threading.Thread(
                target=self._receive_audio_loop,
                args=(stream_token, stream_id),
                daemon=True,
            )
            self._receive_thread.start()
            with self._context_lock:
                self._context.push(normalized)
                self._context.no_more_inputs()
            return True
        except Exception as e:
            log_print("ERROR", f"Failed to start streaming TTS: {e}", session_id=self.session_id)
            return False

    def stop_stream(self) -> None:
        with self._context_lock:
            active = self._active_stream_token
            ctx = self._context
            if active > 0:
                self._stopped_stream_tokens.add(active)
        if ctx is not None:
            try:
                ctx.no_more_inputs()
            except Exception:
                pass
        self._cleanup_stream(active)

    def close(self) -> None:
        self.stop_stream()
        with self._connection_lock:
            self._close_connection_unsafe()
        self.client = None
