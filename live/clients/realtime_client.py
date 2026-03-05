"""OpenAI Realtime API client for keyword extraction."""

import base64
import json
import re
import threading
import time
from typing import Optional

import pyaudio
import websocket
from config import (
    AUDIO_CHUNK,
    AUDIO_RATE,
    DEFAULT_AUDIO_FILE,
    LOG_DIR,
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
)
from flask_socketio import SocketIO
from logger import get_logger, log_print
from pydub import AudioSegment

from .audio_filter import AdaptiveNoiseGate, NoiseGateConfig
from .prompt import KEYWORD_EXTRACTION_PROMPT, KEYWORD_SESSION_INSTRUCTIONS


class RealtimeClient:
    """Client for OpenAI Realtime API with keyword extraction."""

    def __init__(
        self,
        socketio: SocketIO,
        source: str = "mp3",
        session_id: str = "default",
        device_index: int | None = None,
        enable_noise_gate: bool = False,
        noise_gate_config: NoiseGateConfig | None = None,
    ):
        self.sio = socketio
        self.source = source
        self.session_id = session_id
        self.device_index = device_index  # Specific mic device index
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.chunks_sent = 0
        self.chunks_filtered = 0  # Chunks filtered by noise gate
        self.chunks_since_last_request = 0  # Track audio since last request
        self.response_buffer = ""
        self.logger = get_logger(session_id)
        self.extracted_keywords: list[str] = []  # Track previously extracted keywords

        # Thread synchronization
        self._ws_lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # Session recording
        self.user_actions: list[dict] = []  # Track user keyword requests with timing
        self.stream_start_time: float = 0.0  # Audio stream start timestamp
        self.recording_buffer: list[bytes] = []  # Audio recording

        # Reference to summary clients for forwarding transcripts
        self.summary_clients: list = []
        self._vad_transcript_callbacks: list = []

        # Noise gate for filtering ambient audio
        self.enable_noise_gate = enable_noise_gate
        self.noise_gate: AdaptiveNoiseGate | None = None
        if enable_noise_gate:
            config = noise_gate_config or NoiseGateConfig(
                sample_rate=AUDIO_RATE,
                chunk_size=AUDIO_CHUNK,
            )
            self.noise_gate = AdaptiveNoiseGate(config=config, session_id=session_id)

        log_print(
            "INFO",
            "RealtimeClient created",
            session_id=session_id,
            source=source,
            device=device_index,
            noise_gate=enable_noise_gate,
        )
        self.logger.log(
            "client_created",
            source=source,
            device=device_index,
            noise_gate=enable_noise_gate,
        )

    def set_summary_clients(self, clients: list) -> None:
        """Set reference to SummaryClients for transcript forwarding.

        Args:
            clients: List of SummaryClient instances
        """
        self.summary_clients = clients
        log_print(
            "INFO",
            f"Set {len(clients)} summary clients for transcript forwarding",
            session_id=self.session_id,
        )

    def add_vad_transcript_callback(self, callback) -> None:
        """Register callback called with each VAD transcript string."""
        if callback is None:
            return
        self._vad_transcript_callbacks.append(callback)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opened."""
        log_print("INFO", "WebSocket connected to OpenAI", session_id=self.session_id)
        self.logger.log("websocket_connected")
        self.sio.emit("status", "Connected")

        # Configure with server-side VAD transcription so we can capture
        # context_before / next_sentence more frequently.
        session_config = {
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "turn_detection": {
                "type": "server_vad",
                "threshold": 0.45,
                "prefix_padding_ms": 180,
                "silence_duration_ms": 420,
                "create_response": False,
            },
            "input_audio_transcription": {
                "model": "whisper-1",
            },
            "instructions": KEYWORD_SESSION_INSTRUCTIONS,
        }

        ws.send(json.dumps({"type": "session.update", "session": session_config}))
        log_print(
            "DEBUG",
            "Session update sent (server_vad transcription enabled)",
            session_id=self.session_id,
        )
        self.logger.log("session_update_sent")
        threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        event = json.loads(message)
        etype = event.get("type", "")

        # # Log major events only (skip frequent delta events)
        # if etype not in [
        #     "response.text.delta",
        #     "input_audio_buffer.speech_started",
        #     "input_audio_buffer.speech_stopped",
        #     "conversation.item.input_audio_transcription.completed",
        #     "response.audio.delta",
        #     "response.audio_transcript.delta",
        #     "conversation.item.created",
        #     "response.output_item.added",
        #     "response.content_part.done",
        # ]:
        #     log_print("DEBUG", f"OpenAI event: {etype}", session_id=self.session_id)

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log("openai_session_created", session_id=session_info.get("id"))
        elif etype == "session.updated":
            self.logger.log("openai_session_updated")
        elif etype == "conversation.item.input_audio_transcription.completed":
            # Handle VAD transcription - forward to summary clients
            transcript = event.get("transcript", "")
            if transcript and transcript.strip():
                self._handle_vad_transcript(transcript.strip())
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

    def _handle_vad_transcript(self, transcript: str) -> None:
        """Handle VAD transcription.

        Args:
            transcript: The transcribed text from VAD
        """
        log_print(
            "INFO",
            f"VAD transcript: {transcript[:60]}...",
            session_id=self.session_id,
        )
        self.logger.log("vad_transcript", transcript=transcript)

        # Don't emit to frontend - VAD transcripts are for internal context only
        for callback in self._vad_transcript_callbacks:
            try:
                callback(transcript)
            except Exception as e:
                log_print(
                    "WARN",
                    f"VAD transcript callback failed: {e}",
                    session_id=self.session_id,
                )

    def _parse_json_format(self, text: str) -> list[dict]:
        """Parse JSON-like format: {"key": value, "key2": value2}

        Also handles: {"keyword": "term: description"} where key is a meta label.
        """
        keywords = []
        meta_keys = {"keyword", "keywords", "term", "terms", "word", "words"}

        # Try proper JSON parse first
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    key = str(key).strip()
                    value = str(value).strip()

                    # If key is a meta label like "keyword", extract from value
                    if key.lower() in meta_keys and ":" in value:
                        word, desc = value.split(":", 1)
                        word = word.strip()
                        desc = desc.strip()
                        if word and desc:
                            keywords.append({"word": word, "desc": desc})
                    elif value:
                        keywords.append({"word": key, "desc": value})
                return keywords
        except json.JSONDecodeError:
            pass

        # Fallback: regex parsing
        inner = text[1:-1].strip()
        if not inner:
            return keywords

        pattern = r'"([^"]+)":\s*"?([^",}]+)"?'
        matches = re.findall(pattern, inner)

        for word, desc in matches:
            word = word.strip()
            desc = desc.strip().rstrip('"')

            # If key is meta label, extract from value
            if word.lower() in meta_keys and ":" in desc:
                actual_word, actual_desc = desc.split(":", 1)
                word = actual_word.strip()
                desc = actual_desc.strip()

            if word and desc:
                keywords.append({"word": word, "desc": desc})

        return keywords

    def _sanitize_keywords(self, keywords: list[dict]) -> list[dict]:
        """Remove malformed/meta keyword entries and deduplicate."""
        blocked_words = {
            "keyword",
            "keywords",
            "term",
            "terms",
            "word",
            "words",
            "definition",
            "definitions",
            "desc",
            "description",
            "explanation",
            "keypoint",
            "topic",
        }

        cleaned: list[dict] = []
        seen: set[str] = set()

        for kw in keywords:
            word = str(kw.get("word", "")).strip()
            desc = str(kw.get("desc", "")).strip()
            if not word or not desc:
                continue

            # Strip bullet/number prefixes
            word = re.sub(r"^[\d\.\)\-\*\s]+", "", word).strip()
            if not word:
                continue

            word_lower = word.lower()
            if word_lower in blocked_words:
                continue
            # Skip generic "x argument/definition" style placeholders
            if word_lower in ("argument", "concept", "idea"):
                continue
            # Avoid obvious malformed structured fragments
            if any(ch in word for ch in "{}[]"):
                continue
            if len(word.split()) > 6:
                continue
            if len(desc.split()) < 3:
                continue

            if word_lower in seen:
                continue
            seen.add(word_lower)
            cleaned.append({"word": word, "desc": desc})

        return cleaned

    def _parse_line_format(self, text: str) -> list[dict]:
        """Parse line-by-line format: keyword only OR keyword: description"""
        keywords = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Strip leading numbers/bullets (e.g., "1.", "1)", "-", "*")
            line = re.sub(r"^[\d\.\)\-\*\s]+", "", line).strip()
            if not line:
                continue

            # Skip lines with braces (malformed JSON-like output)
            if "{" in line or "}" in line:
                continue

            if ":" in line:
                word, desc = line.split(":", 1)
                word = word.strip()
                desc = desc.strip()

                # Skip invalid entries
                if not word:
                    continue
                # Skip meta labels like "keyword:", "keywords:", "term:", etc.
                word_lower = word.lower()
                if word_lower in (
                    "keyword",
                    "keywords",
                    "term",
                    "terms",
                    "word",
                    "words",
                ):
                    continue

                keywords.append({"word": word, "desc": desc})
            else:
                # No colon - treat entire line as keyword with empty description
                word = line.strip()
                # Skip meta labels
                word_lower = word.lower()
                if word_lower in (
                    "keyword",
                    "keywords",
                    "term",
                    "terms",
                    "word",
                    "words",
                ):
                    continue
                if word:
                    keywords.append({"word": word, "desc": ""})

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
        keywords = self._sanitize_keywords(keywords)

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

        # Emit only when we have valid keywords. Empty output should not change UI state.
        if keywords:
            self.sio.emit("keywords", keywords)
        else:
            self.logger.log("keywords_empty_ignored")

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
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()

            # Get device info for logging
            device_name = "Default"
            if self.device_index is not None:
                try:
                    info = pa.get_device_info_by_index(self.device_index)
                    device_name = info["name"]
                except Exception:
                    pass

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=self.device_index,  # None = default device
                frames_per_buffer=AUDIO_CHUNK,
            )
            log_print(
                "INFO",
                "Mic recording started",
                session_id=self.session_id,
                device=device_name,
            )
            self.logger.log("mic_recording_start", device=device_name)
            self.sio.emit("status", f"🎤 {device_name}")

            while self.running and not self._shutdown_event.is_set():
                try:
                    chunk = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                except Exception as e:
                    if self._shutdown_event.is_set():
                        break
                    log_print(
                        "WARN", f"Mic read error: {e}", session_id=self.session_id
                    )
                    continue

                # Track noise gate status (but don't filter - send all audio to OpenAI)
                if self.noise_gate:
                    is_speech = self.noise_gate.process(chunk)
                    if not is_speech:
                        self.chunks_filtered += 1
                        # Emit gate status periodically
                        if self.chunks_filtered % 50 == 0:
                            self.sio.emit(
                                "noise_gate_status", self.noise_gate.get_status()
                            )
                    # Note: We still send ALL audio to OpenAI regardless of noise gate

                if not self._send_audio_chunk(chunk):
                    if self._shutdown_event.is_set():
                        break

                if self.chunks_sent % 100 == 0:
                    log_print(
                        "DEBUG",
                        f"Audio chunks sent: {self.chunks_sent}",
                        session_id=self.session_id,
                    )
                    # Emit gate status with chunk stats
                    if self.noise_gate:
                        self.sio.emit("noise_gate_status", self.noise_gate.get_status())

        except Exception as e:
            log_print("ERROR", f"Mic stream error: {e}", session_id=self.session_id)
        finally:
            # Clean up PyAudio resources safely
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

            log_print(
                "INFO",
                "Mic recording stopped",
                session_id=self.session_id,
                total_chunks=self.chunks_sent,
            )
            self.logger.log("mic_recording_stop", total_chunks=self.chunks_sent)
            if not self._shutdown_event.is_set():
                self.sio.emit("status", "Stopped")
                self.sio.emit("session_ended")

    def _stream_from_mp3(self) -> None:
        """Stream audio from MP3 file."""
        pa = None
        stream = None
        try:
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
            self.sio.emit("status", "🎵 Playing MP3...")

            chunk_bytes = AUDIO_CHUNK * 2
            total_chunks = len(raw) // chunk_bytes

            log_print(
                "INFO",
                "MP3 playback started",
                session_id=self.session_id,
                total_chunks=total_chunks,
            )
            self.logger.log("mp3_playback_start", total_chunks=total_chunks)

            for i in range(0, len(raw), chunk_bytes):
                if not self.running or self._shutdown_event.is_set():
                    break
                chunk = raw[i : i + chunk_bytes]
                try:
                    stream.write(chunk)
                except Exception:
                    if self._shutdown_event.is_set():
                        break
                self._send_audio_chunk(chunk)

                if self.chunks_sent % 100 == 0:
                    progress = (
                        (self.chunks_sent / total_chunks) * 100
                        if total_chunks > 0
                        else 0
                    )
                    log_print(
                        "DEBUG",
                        f"Playback progress: {progress:.1f}%",
                        session_id=self.session_id,
                        chunks=self.chunks_sent,
                    )

        except Exception as e:
            log_print("ERROR", f"MP3 stream error: {e}", session_id=self.session_id)
        finally:
            # Clean up PyAudio resources safely
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

            log_print(
                "INFO",
                "MP3 playback complete",
                session_id=self.session_id,
                total_chunks=self.chunks_sent,
            )
            self.logger.log("mp3_playback_complete", total_chunks=self.chunks_sent)
            if not self._shutdown_event.is_set():
                self.sio.emit("status", "Done")
                self.sio.emit("session_ended")

    def _send_audio_chunk(self, chunk: bytes) -> bool:
        """Send audio chunk to OpenAI. Returns False if send failed."""
        # Record audio
        self.recording_buffer.append(chunk)

        audio_b64 = base64.b64encode(chunk).decode()
        with self._ws_lock:
            ws = self.ws
        if ws:
            try:
                ws.send(
                    json.dumps(
                        {"type": "input_audio_buffer.append", "audio": audio_b64}
                    )
                )
                self.chunks_sent += 1
                self.chunks_since_last_request += 1
                return True
            except Exception:
                return False
        return False

    def request(self) -> None:
        """Request keyword extraction from accumulated audio."""
        elapsed_sec = (
            time.time() - self.stream_start_time if self.stream_start_time else 0.0
        )

        has_new_audio = self.chunks_since_last_request >= 1

        log_print(
            "INFO",
            "Requesting keyword extraction",
            session_id=self.session_id,
            chunks_so_far=self.chunks_sent,
            chunks_since_last=self.chunks_since_last_request,
            has_new_audio=has_new_audio,
            elapsed_sec=elapsed_sec,
        )
        self.logger.log(
            "keyword_request",
            chunks_so_far=self.chunks_sent,
            chunks_since_last=self.chunks_since_last_request,
            has_new_audio=has_new_audio,
            elapsed_sec=elapsed_sec,
        )

        # Track user action with timing
        self.user_actions.append(
            {
                "elapsed_sec": elapsed_sec,
                "action": "keyword_request",
                "keywords": [],  # Will be filled in _handle_response_done
            }
        )

        self.sio.emit("clear")

        with self._ws_lock:
            ws = self.ws

        if not ws:
            log_print(
                "WARN", "request() called but no websocket", session_id=self.session_id
            )
            return

        try:
            # Only commit if there's new audio (avoids "buffer too small" error)
            if has_new_audio:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

            # Build prompt with previously extracted keywords to avoid repetition
            prompt = KEYWORD_EXTRACTION_PROMPT
            if self.extracted_keywords:
                already_extracted = ", ".join(self.extracted_keywords[-20:])  # Last 20
                prompt = f"{KEYWORD_EXTRACTION_PROMPT}\n\nALREADY EXTRACTED (do NOT repeat): {already_extracted}"

            ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": prompt,
                            "max_output_tokens": 500,
                        },
                    }
                )
            )

            # Clear buffer and reset counter only if we committed new audio
            if has_new_audio:
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                self.chunks_since_last_request = 0
        except Exception as e:
            log_print(
                "ERROR", f"request() send failed: {e}", session_id=self.session_id
            )

    # --------------------------
    # Noise Gate Control Methods
    # --------------------------

    def start_noise_calibration(self) -> bool:
        """Start noise gate calibration.

        Returns:
            True if calibration started, False if noise gate not enabled
        """
        if not self.noise_gate:
            return False
        self.noise_gate.start_calibration()
        self.sio.emit("noise_gate_calibrating", {"status": "started"})
        return True

    def stop_noise_calibration(self) -> dict | None:
        """Stop noise gate calibration and get results.

        Returns:
            Calibration results dict, or None if noise gate not enabled
        """
        if not self.noise_gate:
            return None
        results = self.noise_gate.stop_calibration()
        self.sio.emit(
            "noise_gate_calibrating", {"status": "stopped", "results": results}
        )
        return results

    def set_noise_threshold(self, threshold: float | None) -> bool:
        """Set noise gate threshold manually.

        Args:
            threshold: Fixed threshold value, or None to use adaptive

        Returns:
            True if set, False if noise gate not enabled
        """
        if not self.noise_gate:
            return False
        self.noise_gate.set_threshold_override(threshold)
        return True

    def update_noise_gate_config(self, **kwargs) -> bool:
        """Update noise gate configuration.

        Args:
            **kwargs: Config parameters to update

        Returns:
            True if updated, False if noise gate not enabled
        """
        if not self.noise_gate:
            return False
        self.noise_gate.update_config(**kwargs)
        return True

    def get_noise_gate_status(self) -> dict | None:
        """Get current noise gate status.

        Returns:
            Status dict, or None if noise gate not enabled
        """
        if not self.noise_gate:
            return None
        return self.noise_gate.get_status()

    def start(self) -> None:
        """Start the realtime client."""
        log_print("INFO", "Starting RealtimeClient", session_id=self.session_id)
        self.logger.log("client_start")
        self._shutdown_event.clear()
        self.running = True
        self.chunks_sent = 0
        self.chunks_filtered = 0  # Reset filtered count
        self.chunks_since_last_request = 0  # Reset for new session
        self.extracted_keywords = []  # Reset for new session
        self.user_actions = []  # Reset user actions
        self.stream_start_time = 0.0  # Reset stream start time
        self.recording_buffer = []  # Reset recording

        # Reset noise gate if enabled
        if self.noise_gate:
            self.noise_gate.reset()

        with self._ws_lock:
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
            ws = self.ws
        threading.Thread(target=ws.run_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop the realtime client."""
        # Log noise gate stats if enabled
        gate_stats = None
        if self.noise_gate:
            gate_stats = self.noise_gate.get_status()

        log_print(
            "INFO",
            "Stopping RealtimeClient",
            session_id=self.session_id,
            chunks_sent=self.chunks_sent,
            chunks_filtered=self.chunks_filtered,
        )
        self.logger.log(
            "client_stop",
            total_chunks=self.chunks_sent,
            filtered_chunks=self.chunks_filtered,
            noise_gate_stats=gate_stats,
        )
        was_running = self.running
        self.running = False
        self._shutdown_event.set()

        with self._ws_lock:
            ws = self.ws
            self.ws = None

        if ws:
            try:
                ws.close()
            except Exception:
                pass

        # Wait briefly for audio thread to exit cleanly
        time.sleep(0.15)

        # Save recorded audio
        self._save_audio()

        # Emit session_ended if we were running (streaming thread will also emit, but this ensures it)
        if was_running:
            self.sio.emit("session_ended")

    def _save_audio(self) -> None:
        """Save recorded audio as WAV."""
        if not self.recording_buffer:
            return

        try:
            from datetime import datetime

            # Combine all chunks
            raw_audio = b"".join(self.recording_buffer)
            self.recording_buffer = []

            # Create audio directory
            audio_dir = LOG_DIR / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

            # Convert raw PCM to AudioSegment
            audio = AudioSegment(
                data=raw_audio,
                sample_width=2,  # 16-bit
                frame_rate=AUDIO_RATE,
                channels=1,
            )

            # Save as WAV with date_time_sessionid format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = audio_dir / f"{timestamp}_{self.session_id}.wav"
            audio.export(output_path, format="wav")

            log_print(
                "INFO",
                f"Audio saved: {output_path}",
                session_id=self.session_id,
                duration_sec=len(audio) / 1000,
            )
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to save audio: {e}",
                session_id=self.session_id,
            )
