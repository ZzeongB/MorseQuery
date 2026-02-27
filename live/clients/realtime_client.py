"""OpenAI Realtime API client for keyword extraction."""

import base64
import io
import json
import re
import threading
import time
import wave
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import pyaudio
import websocket
from config import (
    AUDIO_CHUNK,
    AUDIO_RATE,
    AUTO_INTERVAL,
    DEFAULT_AUDIO_FILE,
    LOG_DIR,
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
)
from flask_socketio import SocketIO
from logger import get_logger, log_print
from openai import OpenAI
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
        self.extracted_keywords: list[str] = []  # Track previously extracted keywords

        # Session recording
        self.recording_buffer: list[bytes] = []
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.user_actions: list[dict] = []  # Track user keyword requests with timing
        self.stream_start_time: float = 0.0  # Audio stream start timestamp

        # Live transcription
        self.transcript_buffer: str = ""  # Accumulated transcript text
        self.last_transcription_commit: float = 0.0  # Last time we committed for transcription
        self.transcription_interval: float = 3.0  # Seconds between transcription commits

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

        # All modes use the same session config (turn_detection=None, manual control)
        session_config = {
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "turn_detection": None,
            "instructions": KEYWORD_SESSION_INSTRUCTIONS,
            "input_audio_transcription": {"model": "whisper-1"},
        }

        ws.send(json.dumps({"type": "session.update", "session": session_config}))
        log_print("DEBUG", "Session update sent", session_id=self.session_id, mode=self.mode)
        self.logger.log("session_update_sent", mode=self.mode)
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
        elif etype == "conversation.item.input_audio_transcription.completed":
            # Real-time transcription from audio input
            transcript = event.get("transcript", "")
            if transcript:
                self._handle_transcription(transcript)
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

    def _parse_json_format(self, text: str) -> list[dict]:
        """Parse JSON-like format: {"key": value, "key2": value2}"""
        keywords = []
        # Remove outer braces
        inner = text[1:-1].strip()
        if not inner:
            return keywords

        # Split by comma, but be careful with commas inside values
        # Pattern: "key": value (value may or may not be quoted)
        pattern = r'"([^"]+)":\s*"?([^",}]+)"?'
        matches = re.findall(pattern, inner)

        for word, desc in matches:
            word = word.strip()
            desc = desc.strip().rstrip('"')
            if word and desc:
                keywords.append({"word": word, "desc": desc})

        return keywords

    def _parse_line_format(self, text: str) -> list[dict]:
        """Parse line-by-line format: keyword: description"""
        keywords = []
        for line in text.split("\n"):
            if ":" in line:
                word, desc = line.split(":", 1)
                word = word.strip()
                desc = desc.strip()

                # Skip invalid entries
                if not word or not desc:
                    continue
                # Skip lines with braces (malformed JSON-like output)
                if "{" in word or "}" in word or "{" in desc or "}" in desc:
                    continue
                # Skip meta labels like "keyword:", "keywords:", "term:", etc.
                word_lower = word.lower()
                if word_lower in ("keyword", "keywords", "term", "terms", "word", "words"):
                    continue
                # Strip leading numbers/bullets (e.g., "1.", "1)", "-", "*")
                word = re.sub(r"^[\d\.\)\-\*\s]+", "", word).strip()
                if not word:
                    continue

                keywords.append({"word": word, "desc": desc})

        return keywords

    def _handle_response_done(self) -> None:
        """Parse and emit completed response."""
        keywords_text = self.response_buffer.strip()

        # Parse keywords into list
        # Try JSON-like format first: {"key": value, "key2": value2}
        if keywords_text.startswith("{") and keywords_text.endswith("}"):
            keywords = self._parse_json_format(keywords_text)
        else:
            keywords = self._parse_line_format(keywords_text)

        log_print(
            "INFO",
            "Response complete",
            session_id=self.session_id,
            keywords=keywords,
        )
        self.logger.log(
            "response_done",
            response=self.response_buffer,
            keywords=keywords,
        )

        # Track extracted keywords to avoid repetition
        for kw in keywords:
            word = kw["word"].lower()
            if word not in self.extracted_keywords:
                self.extracted_keywords.append(word)

        # Update last user action with extracted keywords
        if self.user_actions:
            self.user_actions[-1]["keywords"] = keywords

        # Emit to frontend
        self.sio.emit("keywords", keywords)

        self.response_buffer = ""

    def _handle_transcription(self, transcript: str) -> None:
        """Handle incoming transcription and emit to frontend."""
        # Append to buffer
        if self.transcript_buffer:
            self.transcript_buffer += " " + transcript.strip()
        else:
            self.transcript_buffer = transcript.strip()

        log_print(
            "DEBUG",
            "Transcription received",
            session_id=self.session_id,
            transcript=transcript[:50] + "..." if len(transcript) > 50 else transcript,
        )

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
        self.stream_start_time = time.time()
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

            if self.mode in ("auto", "automatic") and self.chunks_sent % chunks_per_interval == 0:
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
        self.sio.emit("session_ended")

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

            if self.mode in ("auto", "automatic") and self.chunks_sent % chunks_per_interval == 0:
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
        self.sio.emit("session_ended")

    def _send_audio_chunk(self, chunk: bytes) -> None:
        """Send audio chunk to OpenAI and forward to summary client."""
        # Record audio for full session transcription
        self.recording_buffer.append(chunk)

        audio_b64 = base64.b64encode(chunk).decode()
        self.ws.send(
            json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
        )
        if self.summary_client:
            self.summary_client.send_audio(audio_b64)
        self.chunks_sent += 1

        # Periodic commit for continuous transcription
        now = time.time()
        if now - self.last_transcription_commit >= self.transcription_interval:
            self.last_transcription_commit = now
            self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

    def request(self) -> None:
        """Request keyword extraction from accumulated audio."""
        elapsed_sec = time.time() - self.stream_start_time if self.stream_start_time else 0.0

        log_print(
            "INFO",
            "Requesting keyword extraction",
            session_id=self.session_id,
            chunks_so_far=self.chunks_sent,
            elapsed_sec=elapsed_sec,
        )
        self.logger.log("keyword_request", chunks_so_far=self.chunks_sent, elapsed_sec=elapsed_sec)

        # Track user action with timing
        self.user_actions.append({
            "elapsed_sec": elapsed_sec,
            "action": "keyword_request",
            "keywords": [],  # Will be filled in _handle_response_done
        })

        self.sio.emit("clear")

        # Commit current audio buffer (also resets transcription timer)
        self.last_transcription_commit = time.time()
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        # Build prompt with previously extracted keywords to avoid repetition
        prompt = KEYWORD_EXTRACTION_PROMPT
        if self.extracted_keywords:
            already_extracted = ", ".join(self.extracted_keywords[-20:])  # Last 20
            prompt = f"{KEYWORD_EXTRACTION_PROMPT}\n\nALREADY EXTRACTED (do NOT repeat): {already_extracted}"

        self.ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": prompt,
                    },
                }
            )
        )

        # Clear buffer after commit to analyze only new audio next time
        self.ws.send(json.dumps({"type": "input_audio_buffer.clear"}))

    def start(self) -> None:
        """Start the realtime client."""
        log_print("INFO", "Starting RealtimeClient", session_id=self.session_id)
        self.logger.log("client_start")
        self.running = True
        self.chunks_sent = 0
        self.extracted_keywords = []  # Reset for new session
        self.recording_buffer = []  # Reset recording buffer
        self.user_actions = []  # Reset user actions
        self.stream_start_time = 0.0  # Reset stream start time
        self.transcript_buffer = ""  # Reset transcript buffer
        self.last_transcription_commit = 0.0  # Reset transcription timer
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
        was_running = self.running
        self.running = False
        if self.ws:
            self.ws.close()

        # Emit session_ended if we were running (streaming thread will also emit, but this ensures it)
        if was_running:
            self.sio.emit("session_ended")

        # Transcribe recorded audio in background
        if self.recording_buffer:
            threading.Thread(target=self._transcribe_session, daemon=True).start()

    def _transcribe_session(self) -> None:
        """Transcribe the full recorded session using Whisper API."""
        if not self.recording_buffer:
            return

        log_print(
            "INFO",
            "Starting session transcription",
            session_id=self.session_id,
            chunks=len(self.recording_buffer),
        )
        self.sio.emit("status", "Transcribing session...")

        try:
            # Combine all audio chunks
            raw_audio = b"".join(self.recording_buffer)
            duration_sec = len(raw_audio) / (AUDIO_RATE * 2)  # 16-bit = 2 bytes

            log_print(
                "INFO",
                f"Session audio: {duration_sec:.1f}s, {len(raw_audio)} bytes",
                session_id=self.session_id,
            )

            # Convert to WAV in memory
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(AUDIO_RATE)
                wav_file.writeframes(raw_audio)

            # Save audio file
            datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            audio_dir = LOG_DIR / "audio"
            audio_dir.mkdir(exist_ok=True)
            audio_file = audio_dir / f"{datetime_str}_{self.session_id}.wav"
            with open(audio_file, "wb") as f:
                wav_buffer.seek(0)
                f.write(wav_buffer.read())

            log_print(
                "INFO",
                f"Audio saved: {audio_file}",
                session_id=self.session_id,
            )

            wav_buffer.seek(0)
            wav_buffer.name = "session.wav"  # Required for OpenAI API

            # Call Whisper API with word timestamps
            transcript_result = self.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=wav_buffer,
                response_format="verbose_json",
                timestamp_granularities=["word"],
            )

            log_print(
                "INFO",
                "Transcription complete",
                session_id=self.session_id,
                length=len(transcript_result.text),
            )

            # Prepare transcript data with timestamps
            transcript_data = {
                "session_id": self.session_id,
                "duration_sec": duration_sec,
                "audio_file": str(audio_file.name),
                "text": transcript_result.text,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end}
                    for w in (transcript_result.words or [])
                ],
                "user_actions": self.user_actions,
            }

            self.logger.log(
                "session_transcription_done",
                duration_sec=duration_sec,
                transcript_length=len(transcript_result.text),
                transcript=transcript_result.text,
            )

            # Save transcript with timestamps to JSON
            transcript_dir = LOG_DIR / "transcript"
            transcript_dir.mkdir(exist_ok=True)
            transcript_file = transcript_dir / f"{datetime_str}_{self.session_id}.json"
            with open(transcript_file, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, ensure_ascii=False, indent=2)

            # Emit to frontend
            self.sio.emit(
                "session_transcript",
                {
                    "transcript": transcript_result.text,
                    "duration_sec": duration_sec,
                },
            )

        except Exception as e:
            log_print(
                "ERROR",
                f"Transcription failed: {e}",
                session_id=self.session_id,
            )
            self.logger.log("session_transcription_error", error=str(e))
            self.sio.emit("error", {"message": f"Transcription failed: {e}"})
