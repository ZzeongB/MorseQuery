"""OpenAI Realtime API client for keyword extraction."""

import base64
import json
import threading
from typing import TYPE_CHECKING, Optional

import pyaudio
import websocket
from config import (
    AUDIO_CHUNK,
    AUDIO_RATE,
    AUTO_INTERVAL,
    DEFAULT_AUDIO_FILE,
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
)
from flask_socketio import SocketIO
from logger import get_logger, log_print
from pydub import AudioSegment

if TYPE_CHECKING:
    from clients.summary_client import SummaryClient

from .prompt import KEYWORD_EXTRACTION_PROMPT, KEYWORD_SESSION_INSTRUCTIONS


class RealtimeClient:
    """Client for OpenAI Realtime API with keyword extraction."""

    def __init__(
        self,
        socketio: SocketIO,
        mode: str = "manual",
        source: str = "mp3",
        session_id: str = "default",
    ):
        self.sio = socketio
        self.mode = mode
        self.source = source
        self.session_id = session_id
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.chunks_sent = 0
        self.response_buffer = ""
        self.logger = get_logger(session_id)
        self.summary_client: Optional["SummaryClient"] = None

        log_print(
            "INFO",
            "RealtimeClient created",
            session_id=session_id,
            mode=mode,
            source=source,
        )
        self.logger.log("client_created", mode=mode, source=source)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opened."""
        log_print("INFO", "WebSocket connected to OpenAI", session_id=self.session_id)
        self.logger.log("websocket_connected")
        self.sio.emit("status", "Connected")

        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": KEYWORD_SESSION_INSTRUCTIONS,
                    },
                }
            )
        )
        log_print("DEBUG", "Session update sent", session_id=self.session_id)
        self.logger.log("session_update_sent")
        threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        event = json.loads(message)
        etype = event.get("type", "")

        # Log major events only (skip frequent delta events)
        if etype not in [
            "response.text.delta",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        ]:
            log_print("DEBUG", f"OpenAI event: {etype}", session_id=self.session_id)

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log("openai_session_created", session_id=session_info.get("id"))
        elif etype == "session.updated":
            self.logger.log("openai_session_updated")
        elif etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
        elif etype == "response.done":
            self._handle_response_done()
        elif etype == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            log_print("ERROR", f"OpenAI error: {error_msg}", session_id=self.session_id)
            self.logger.log("openai_error", error=error_msg)

    def _handle_response_done(self) -> None:
        """Parse and emit completed response."""
        keywords_text = self.response_buffer
        context_text = ""

        # Try CONTEXT:: first, then \n\n as fallback
        if "CONTEXT::" in self.response_buffer:
            parts = self.response_buffer.split("CONTEXT::", 1)
            keywords_text = parts[0].strip()
            context_text = parts[1].strip() if len(parts) > 1 else ""
        elif "\n\n" in self.response_buffer:
            parts = self.response_buffer.split("\n\n", 1)
            keywords_text = parts[0].strip()
            context_text = parts[1].strip() if len(parts) > 1 else ""

        # Parse keywords into list
        keywords = []
        for line in keywords_text.split("\n"):
            if ":" in line and not line.upper().startswith("CONTEXT"):
                word, desc = line.split(":", 1)
                keywords.append({"word": word.strip(), "desc": desc.strip()})

        log_print(
            "INFO",
            "Response complete",
            session_id=self.session_id,
            keywords=keywords,
            context=context_text,
        )
        self.logger.log(
            "response_done",
            response=self.response_buffer,
            keywords=keywords,
            context=context_text,
        )

        # Emit to frontend
        self.sio.emit("keywords", keywords)
        if context_text:
            self.sio.emit("context", context_text)
            if self.summary_client:
                self.summary_client.set_context(context_text)

        self.response_buffer = ""

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket error."""
        log_print("ERROR", f"WebSocket error: {error}", session_id=self.session_id)
        self.logger.log("websocket_error", error=str(error))
        self.sio.emit("status", f"Error: {error}")

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        """Handle WebSocket connection closed."""
        log_print(
            "INFO",
            "WebSocket closed",
            session_id=self.session_id,
            status=status,
            msg=msg,
        )
        self.logger.log("websocket_closed", status=status, message=msg)
        self.sio.emit("status", "Disconnected")
        self.running = False

    def _stream_audio(self) -> None:
        """Stream audio from configured source."""
        log_print(
            "INFO",
            "Starting audio stream",
            session_id=self.session_id,
            source=self.source,
        )
        self.logger.log("stream_start", source=self.source)

        if self.source == "mic":
            self._stream_from_mic()
        else:
            self._stream_from_mp3()

    def _stream_from_mic(self) -> None:
        """Stream audio from microphone."""
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=AUDIO_RATE,
            input=True,
            frames_per_buffer=AUDIO_CHUNK,
        )
        log_print(
            "INFO", "Mic recording started", session_id=self.session_id, mode=self.mode
        )
        self.logger.log("mic_recording_start", mode=self.mode)
        self.sio.emit("status", f"🎤 Mic recording... ({self.mode} mode)")

        chunks_per_interval = int(AUTO_INTERVAL / 0.2)

        while self.running:
            chunk = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
            self._send_audio_chunk(chunk)

            if self.chunks_sent % 100 == 0:
                log_print(
                    "DEBUG",
                    f"Audio chunks sent: {self.chunks_sent}",
                    session_id=self.session_id,
                )

            if self.mode == "auto" and self.chunks_sent % chunks_per_interval == 0:
                self.request()

        stream.stop_stream()
        stream.close()
        pa.terminate()
        log_print(
            "INFO",
            "Mic recording stopped",
            session_id=self.session_id,
            total_chunks=self.chunks_sent,
        )
        self.logger.log("mic_recording_stop", total_chunks=self.chunks_sent)
        self.sio.emit("status", "Stopped")

    def _stream_from_mp3(self) -> None:
        """Stream audio from MP3 file."""
        log_print(
            "INFO",
            f"Loading MP3 file: {DEFAULT_AUDIO_FILE}",
            session_id=self.session_id,
        )
        audio = AudioSegment.from_file(DEFAULT_AUDIO_FILE)
        audio = audio.set_frame_rate(AUDIO_RATE).set_channels(1).set_sample_width(2)
        raw = audio.raw_data
        duration_sec = len(audio) / 1000.0

        log_print(
            "INFO",
            "MP3 loaded",
            session_id=self.session_id,
            duration_sec=duration_sec,
            bytes=len(raw),
        )
        self.logger.log(
            "mp3_loaded",
            file=DEFAULT_AUDIO_FILE,
            duration_sec=duration_sec,
            bytes=len(raw),
        )

        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE, output=True
        )
        self.sio.emit("status", f"🎵 Playing MP3... ({self.mode} mode)")

        chunk_bytes = AUDIO_CHUNK * 2
        chunks_per_interval = int(AUTO_INTERVAL / 0.2)
        total_chunks = len(raw) // chunk_bytes

        log_print(
            "INFO",
            "MP3 playback started",
            session_id=self.session_id,
            mode=self.mode,
            total_chunks=total_chunks,
        )
        self.logger.log("mp3_playback_start", mode=self.mode, total_chunks=total_chunks)

        for i in range(0, len(raw), chunk_bytes):
            if not self.running:
                break
            chunk = raw[i : i + chunk_bytes]
            stream.write(chunk)
            self._send_audio_chunk(chunk)

            if self.chunks_sent % 100 == 0:
                progress = (
                    (self.chunks_sent / total_chunks) * 100 if total_chunks > 0 else 0
                )
                log_print(
                    "DEBUG",
                    f"Playback progress: {progress:.1f}%",
                    session_id=self.session_id,
                    chunks=self.chunks_sent,
                )

            if self.mode == "auto" and self.chunks_sent % chunks_per_interval == 0:
                self.request()

        if self.running:
            self.request()

        stream.stop_stream()
        stream.close()
        pa.terminate()
        log_print(
            "INFO",
            "MP3 playback complete",
            session_id=self.session_id,
            total_chunks=self.chunks_sent,
        )
        self.logger.log("mp3_playback_complete", total_chunks=self.chunks_sent)
        self.sio.emit("status", "Done")

    def _send_audio_chunk(self, chunk: bytes) -> None:
        """Send audio chunk to OpenAI and forward to summary client."""
        audio_b64 = base64.b64encode(chunk).decode()
        self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )
        if self.summary_client:
            self.summary_client.send_audio(audio_b64)
        self.chunks_sent += 1

    def request(self) -> None:
        """Request keyword extraction from accumulated audio."""
        log_print(
            "INFO",
            "Requesting keyword extraction",
            session_id=self.session_id,
            chunks_so_far=self.chunks_sent,
        )
        self.logger.log("keyword_request", chunks_so_far=self.chunks_sent)
        self.sio.emit("clear")
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self.ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": KEYWORD_EXTRACTION_PROMPT,
                    },
                }
            )
        )

    def start(self) -> None:
        """Start the realtime client."""
        log_print("INFO", "Starting RealtimeClient", session_id=self.session_id)
        self.logger.log("client_start")
        self.running = True
        self.chunks_sent = 0
        self.ws = websocket.WebSocketApp(
            OPENAI_REALTIME_URL,
            header=[
                f"Authorization: Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta: realtime=v1",
            ],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop the realtime client."""
        log_print("INFO", "Stopping RealtimeClient", session_id=self.session_id)
        self.logger.log("client_stop", total_chunks=self.chunks_sent)
        self.running = False
        if self.ws:
            self.ws.close()
