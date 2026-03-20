import base64
import json
import re
import threading
import time
from collections import deque
from typing import Optional

import pyaudio
import websocket
from config import (
    AUDIO_CHUNK,
    AUDIO_RATE,
    DEFAULT_AUDIO_FILE,
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
    OPENAI_SESSION_CONFIG,
)
from flask_socketio import SocketIO
from logger import get_logger, get_session_subdir, log_print
from pydub import AudioSegment

from .audio_filter import AdaptiveNoiseGate, NoiseGateConfig
from .prompt import (
    KEYWORD_EXTRACTION_PROMPT,
    KEYWORD_EXTRACTION_PROMPT_SINGLE,
    KEYWORD_SESSION_INSTRUCTIONS,
)


class RealtimeClient:
    def __init__(
        self,
        socketio: SocketIO,
        source: str = "mp3",
        session_id: str = "default",
        device_index: int | None = None,
        mp3_file: str | None = None,
        enable_noise_gate: bool = False,
        noise_gate_config: NoiseGateConfig | None = None,
    ):
        self.sio = socketio
        self.source = source
        self.session_id = session_id
        self.device_index = device_index  # Specific mic device index
        self.mp3_file = mp3_file
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.chunks_sent = 0
        self.chunks_filtered = 0  # Chunks filtered by noise gate
        self.chunks_since_last_request = 0  # Track audio since last request
        self.response_buffer = ""
        self.logger = get_logger(session_id)
        self.extracted_keywords: list[str] = []  # Track previously extracted keywords
        self.completed_tts_keywords: list[str] = []  # Track keyword-TTS completed words
        self._completed_tts_keyword_set: set[str] = set()
        self._ws_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._audio_ring_lock = threading.Lock()
        self.user_actions: list[dict] = []  # Track user keyword requests with timing
        self.latest_keywords: list[
            dict
        ] = []  # Last parsed keyword list with descriptions
        self.stream_start_time: float = 0.0  # Audio stream start timestamp
        self.recording_buffer: list[bytes] = []  # Audio recording
        self._audio_ring: deque[tuple[float, float, bytes]] = deque()
        self._audio_ring_retention_sec = 20 * 60
        self.summary_clients: list = []
        self._vad_transcript_callbacks: list = []
        self._vad_boundary_callbacks: list = []
        self._vad_transcript_history: deque[str] = deque(maxlen=120)
        self._keyword_retry_attempt = 0
        self._max_keyword_retry_attempts = 2  # Retry up to 2 times with realtime API
        self.single_keyword_mode = False  # Extract only 1 keyword when True
        self._current_request_id = 0  # Request ID for keyword request tracking
        self.enable_noise_gate = enable_noise_gate
        self.noise_gate: AdaptiveNoiseGate | None = None
        if enable_noise_gate:
            config = noise_gate_config or NoiseGateConfig(
                sample_rate=AUDIO_RATE,
                chunk_size=AUDIO_CHUNK,
            )
            self.noise_gate = AdaptiveNoiseGate(config=config, session_id=session_id)
        # Audio stream instance variables for explicit cleanup on stop()
        self._audio_stream_lock = threading.Lock()
        self._audio_stream: Optional[pyaudio.Stream] = None
        self._audio_pa: Optional[pyaudio.PyAudio] = None
        log_print(
            "INFO",
            "RealtimeClient created",
            session_id=session_id,
            source=source,
            device=device_index,
            mp3_file=mp3_file,
            noise_gate=enable_noise_gate,
        )
        self.logger.log(
            "client_created",
            source=source,
            device=device_index,
            mp3_file=mp3_file,
            noise_gate=enable_noise_gate,
        )

    @staticmethod
    def _normalize_keyword_text(keyword: str) -> str:
        return re.sub(r"\s+", " ", str(keyword or "").strip().lower())

    def set_summary_clients(self, clients: list) -> None:
        self.summary_clients = clients
        log_print(
            "INFO",
            f"Set {len(clients)} summary clients for transcript forwarding",
            session_id=self.session_id,
        )

    def add_vad_transcript_callback(self, callback) -> None:
        if callback is None:
            return
        self._vad_transcript_callbacks.append(callback)

    def add_vad_boundary_callback(self, callback) -> None:
        if callback is None:
            return
        self._vad_boundary_callbacks.append(callback)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        log_print(
            "INFO",
            f"WebSocket connected to OpenAI with config: {OPENAI_SESSION_CONFIG}",
            session_id=self.session_id,
        )
        self.logger.log("websocket_connected")
        self.sio.emit("status", "Connected")
        session_config = {
            **OPENAI_SESSION_CONFIG,
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
        event = json.loads(message)
        etype = event.get("type", "")
        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log("openai_session_created", session_id=session_info.get("id"))
        elif etype == "session.updated":
            self.logger.log("openai_session_updated")
        elif etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript and transcript.strip():
                self._handle_vad_transcript(transcript.strip())
        elif etype == "input_audio_buffer.speech_started":
            self._handle_vad_boundary("speech_started", event)
        elif etype == "input_audio_buffer.speech_stopped":
            self._handle_vad_boundary("speech_stopped", event)
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
        log_print(
            "INFO",
            f"VAD transcript: {transcript[:60]}...",
            session_id=self.session_id,
        )
        self.logger.log("vad_transcript", transcript=transcript)
        self._vad_transcript_history.append(transcript)
        for callback in self._vad_transcript_callbacks:
            try:
                callback(transcript)
            except Exception as e:
                log_print(
                    "WARN",
                    f"VAD transcript callback failed: {e}",
                    session_id=self.session_id,
                )

    def _handle_vad_boundary(self, boundary_type: str, event: dict) -> None:
        payload = {
            "boundary_type": boundary_type,
            "received_at_ts": time.time(),
            "audio_start_ms": event.get("audio_start_ms"),
            "audio_end_ms": event.get("audio_end_ms"),
            "item_id": event.get("item_id"),
        }
        self.logger.log("vad_boundary", **payload)
        for callback in self._vad_boundary_callbacks:
            try:
                callback(boundary_type, payload)
            except Exception as e:
                log_print(
                    "WARN",
                    f"VAD boundary callback failed: {e}",
                    session_id=self.session_id,
                )

    def _parse_json_format(self, text: str) -> list[dict]:
        keywords = []
        meta_keys = {"keyword", "keywords", "term", "terms", "word", "words"}
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    word = str(item.get("word", "")).strip()
                    desc = str(item.get("desc", "")).strip()
                    if word and desc:
                        keywords.append({"word": word, "desc": desc})
                return keywords
            if isinstance(parsed, dict):
                for key, value in parsed.items():
                    key = str(key).strip()
                    value = str(value).strip()
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
        inner = text[1:-1].strip()
        if not inner:
            return keywords
        pattern = r'"([^"]+)":\s*"?([^",}]+)"?'
        matches = re.findall(pattern, inner)
        for word, desc in matches:
            word = word.strip()
            desc = desc.strip().rstrip('"')
            if word.lower() in meta_keys and ":" in desc:
                actual_word, actual_desc = desc.split(":", 1)
                word = actual_word.strip()
                desc = actual_desc.strip()
            if word and desc:
                keywords.append({"word": word, "desc": desc})
        return keywords

    def _sanitize_keywords(self, keywords: list[dict]) -> list[dict]:
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
            word = re.sub(r"^[\d\.\)\-\*\s]+", "", word).strip()
            if not word:
                continue
            word_lower = word.lower()
            if word_lower in blocked_words:
                continue
            if word_lower in ("argument", "concept", "idea"):
                continue
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

    def _filter_completed_tts_keywords(self, keywords: list[dict]) -> list[dict]:
        with self._state_lock:
            blocked = set(self._completed_tts_keyword_set)
        if not blocked:
            return keywords
        filtered: list[dict] = []
        for kw in keywords:
            normalized = self._normalize_keyword_text(kw.get("word", ""))
            if normalized and normalized in blocked:
                continue
            filtered.append(kw)
        return filtered

    def _parse_line_format(self, text: str) -> list[dict]:
        keywords = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\d\.\)\-\*\s]+", "", line).strip()
            if not line:
                continue
            if "{" in line or "}" in line:
                continue
            if ":" in line:
                word, desc = line.split(":", 1)
                word = word.strip()
                desc = desc.strip()
                if not word:
                    continue
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
                word = line.strip()
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
        keywords_text = self.response_buffer.strip()
        if keywords_text.startswith("{") and keywords_text.endswith("}"):
            keywords = self._parse_json_format(keywords_text)
        else:
            keywords = self._parse_line_format(keywords_text)
        keywords = self._sanitize_keywords(keywords)
        keywords = self._filter_completed_tts_keywords(keywords)
        log_print(
            "DEBUG",
            f"Parsed keywords: {keywords}",
            session_id=self.session_id,
        )
        if keywords == []:
            log_print(
                "WARN",
                "No valid keywords parsed from response",
                session_id=self.session_id,
            )
            if self._request_keyword_retry_in_realtime():
                self.response_buffer = ""
                return
            # Empty follow-up response - don't log response_done or update latest_keywords
            log_print(
                "DEBUG",
                "Ignoring empty follow-up response (not updating latest_keywords)",
                session_id=self.session_id,
            )
            self.response_buffer = ""
            return
        # Only log and update state when we have actual keywords
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
        for kw in keywords:
            word = self._normalize_keyword_text(kw["word"])
            if word not in self.extracted_keywords:
                self.extracted_keywords.append(word)
        with self._state_lock:
            self.latest_keywords = [dict(kw) for kw in keywords]
            if self.user_actions:
                self.user_actions[-1]["keywords"] = keywords
        # Include requestId for client-side staleness check
        self.sio.emit("keywords", {"keywords": keywords, "requestId": self._current_request_id})
        self.response_buffer = ""

    def _request_keyword_retry_in_realtime(self) -> bool:
        if self._keyword_retry_attempt >= self._max_keyword_retry_attempts:
            self.logger.log("keywords_retry_exhausted")
            return False
        with self._ws_lock:
            ws = self.ws
        if not ws:
            self.logger.log("keywords_retry_skipped_no_ws")
            return False
        base_prompt = (
            KEYWORD_EXTRACTION_PROMPT_SINGLE
            if self.single_keyword_mode
            else KEYWORD_EXTRACTION_PROMPT
        )
        retry_prompt = (
            f"{base_prompt}\n\n"
            "Your previous output was invalid or empty. Refer to previous context and try again. "
            "You MUST give at least one `<keyword>: <description>` format."
        )
        try:
            ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": retry_prompt,
                            "max_output_tokens": 500,
                        },
                    }
                )
            )
            self._keyword_retry_attempt += 1
            self.logger.log(
                "keywords_retry_requested",
                attempt=self._keyword_retry_attempt,
            )
            return True
        except Exception as e:
            log_print(
                "WARN",
                f"keywords retry request failed: {e}",
                session_id=self.session_id,
            )
            self.logger.log("keywords_retry_request_failed", error=str(e))
            return False

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        log_print("ERROR", f"WebSocket error: {error}", session_id=self.session_id)
        self.logger.log("websocket_error", error=str(error))
        self.sio.emit("status", f"Error: {error}")

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
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
        try:
            with self._audio_stream_lock:
                self._audio_pa = pyaudio.PyAudio()
                pa = self._audio_pa
            device_name = "Default"
            if self.device_index is not None:
                try:
                    info = pa.get_device_info_by_index(self.device_index)
                    device_name = info["name"]
                except Exception:
                    pass
            with self._audio_stream_lock:
                self._audio_stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=AUDIO_RATE,
                    input=True,
                    input_device_index=self.device_index,  # None = default device
                    frames_per_buffer=AUDIO_CHUNK,
                )
                stream = self._audio_stream
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
                if self.noise_gate:
                    is_speech = self.noise_gate.process(chunk)
                    if not is_speech:
                        self.chunks_filtered += 1
                        if self.chunks_filtered % 50 == 0:
                            self.sio.emit(
                                "noise_gate_status", self.noise_gate.get_status()
                            )
                if not self._send_audio_chunk(chunk):
                    if self._shutdown_event.is_set():
                        break
                if self.chunks_sent % 100 == 0:
                    log_print(
                        "DEBUG",
                        f"Audio chunks sent: {self.chunks_sent}",
                        session_id=self.session_id,
                    )
                    if self.noise_gate:
                        self.sio.emit("noise_gate_status", self.noise_gate.get_status())
        except Exception as e:
            log_print("ERROR", f"Mic stream error: {e}", session_id=self.session_id)
        finally:
            with self._audio_stream_lock:
                if self._audio_stream:
                    try:
                        self._audio_stream.stop_stream()
                        self._audio_stream.close()
                    except Exception:
                        pass
                    self._audio_stream = None
                if self._audio_pa:
                    try:
                        self._audio_pa.terminate()
                    except Exception:
                        pass
                    self._audio_pa = None
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
        try:
            target_file = self.mp3_file or DEFAULT_AUDIO_FILE
            log_print(
                "INFO",
                f"Loading MP3 file: {target_file}",
                session_id=self.session_id,
            )
            audio = AudioSegment.from_file(target_file)
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
                file=target_file,
                duration_sec=duration_sec,
                bytes=len(raw),
            )
            with self._audio_stream_lock:
                self._audio_pa = pyaudio.PyAudio()
                self._audio_stream = self._audio_pa.open(
                    format=pyaudio.paInt16, channels=1, rate=AUDIO_RATE, output=True
                )
                stream = self._audio_stream
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
            with self._audio_stream_lock:
                if self._audio_stream:
                    try:
                        self._audio_stream.stop_stream()
                        self._audio_stream.close()
                    except Exception:
                        pass
                    self._audio_stream = None
                if self._audio_pa:
                    try:
                        self._audio_pa.terminate()
                    except Exception:
                        pass
                    self._audio_pa = None
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
        self.recording_buffer.append(chunk)
        now = time.time()
        chunk_duration = (len(chunk) / 2.0) / float(AUDIO_RATE)
        chunk_start = now - chunk_duration
        with self._audio_ring_lock:
            self._audio_ring.append((chunk_start, now, bytes(chunk)))
            cutoff = now - float(self._audio_ring_retention_sec)
            while self._audio_ring and self._audio_ring[0][1] < cutoff:
                self._audio_ring.popleft()
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

    def get_audio_window_pcm(self, start_ts: float, end_ts: float) -> bytes:
        start_ts = float(start_ts or 0.0)
        end_ts = float(end_ts or 0.0)
        if start_ts <= 0 or end_ts <= 0:
            return b""
        if end_ts < start_ts:
            start_ts, end_ts = end_ts, start_ts
        with self._audio_ring_lock:
            chunks = list(self._audio_ring)
        if not chunks:
            return b""
        out_parts: list[bytes] = []
        bytes_per_sample = 2
        for chunk_start, chunk_end, chunk in chunks:
            if chunk_end <= start_ts or chunk_start >= end_ts:
                continue
            ov_start = max(start_ts, chunk_start)
            ov_end = min(end_ts, chunk_end)
            if ov_end <= ov_start:
                continue
            duration = max(1e-6, chunk_end - chunk_start)
            start_ratio = (ov_start - chunk_start) / duration
            end_ratio = (ov_end - chunk_start) / duration
            chunk_samples = len(chunk) // bytes_per_sample
            i0 = int(max(0, min(chunk_samples, round(start_ratio * chunk_samples))))
            i1 = int(max(0, min(chunk_samples, round(end_ratio * chunk_samples))))
            if i1 <= i0:
                continue
            b0 = i0 * bytes_per_sample
            b1 = i1 * bytes_per_sample
            out_parts.append(chunk[b0:b1])
        if not out_parts:
            return b""
        return b"".join(out_parts)

    def request(self, request_id: int = 0) -> None:
        self._keyword_retry_attempt = 0
        self._current_request_id = request_id
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
        with self._state_lock:
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
            if has_new_audio:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            base_prompt = (
                KEYWORD_EXTRACTION_PROMPT_SINGLE
                if self.single_keyword_mode
                else KEYWORD_EXTRACTION_PROMPT
            )
            prompt = base_prompt
            excluded_keywords: list[str] = []
            excluded_keywords.extend(self.extracted_keywords[-20:])
            excluded_keywords.extend(self.completed_tts_keywords[-40:])
            if excluded_keywords:
                deduped_excluded = list(dict.fromkeys(excluded_keywords))
                blocked = ", ".join(deduped_excluded)
                prompt = (
                    f"{base_prompt}\n\n"
                    "Do NOT include any keyword from this history "
                    "(already extracted + keyword-tts playback completed): "
                    f"{blocked}"
                )
            ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": prompt,
                            "max_output_tokens": 700,
                        },
                    }
                )
            )
            if has_new_audio:
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                self.chunks_since_last_request = 0
        except Exception as e:
            log_print(
                "ERROR", f"request() send failed: {e}", session_id=self.session_id
            )

    def start_noise_calibration(self) -> bool:
        if not self.noise_gate:
            return False
        self.noise_gate.start_calibration()
        self.sio.emit("noise_gate_calibrating", {"status": "started"})
        return True

    def stop_noise_calibration(self) -> dict | None:
        if not self.noise_gate:
            return None
        results = self.noise_gate.stop_calibration()
        self.sio.emit(
            "noise_gate_calibrating", {"status": "stopped", "results": results}
        )
        return results

    def set_noise_threshold(self, threshold: float | None) -> bool:
        if not self.noise_gate:
            return False
        self.noise_gate.set_threshold_override(threshold)
        return True

    def update_noise_gate_config(self, **kwargs) -> bool:
        if not self.noise_gate:
            return False
        self.noise_gate.update_config(**kwargs)
        return True

    def get_noise_gate_status(self) -> dict | None:
        if not self.noise_gate:
            return None
        return self.noise_gate.get_status()

    def start(self) -> None:
        log_print("INFO", "Starting RealtimeClient", session_id=self.session_id)
        self.logger.log("client_start")
        self._shutdown_event.clear()
        self.running = True
        self.chunks_sent = 0
        self.chunks_filtered = 0  # Reset filtered count
        self.chunks_since_last_request = 0  # Reset for new session
        self.extracted_keywords = []  # Reset for new session
        self.completed_tts_keywords = []  # Reset for new session
        self._completed_tts_keyword_set.clear()
        self._vad_transcript_history.clear()
        with self._state_lock:
            self.user_actions = []  # Reset user actions
            self.latest_keywords = []  # Reset keyword cache
        self.stream_start_time = 0.0  # Reset stream start time
        self.recording_buffer = []  # Reset recording
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
        # Explicitly close audio stream to stop playback immediately
        with self._audio_stream_lock:
            if self._audio_stream:
                try:
                    self._audio_stream.stop_stream()
                    self._audio_stream.close()
                except Exception:
                    pass
                self._audio_stream = None
            if self._audio_pa:
                try:
                    self._audio_pa.terminate()
                except Exception:
                    pass
                self._audio_pa = None
        time.sleep(0.15)
        self._save_audio()
        if was_running:
            self.sio.emit("session_ended")

    def get_recent_keywords(self, limit: int = 3) -> list[dict]:
        limit = max(0, int(limit))
        with self._state_lock:
            items = [dict(kw) for kw in (self.latest_keywords or [])]
        if not items or limit == 0:
            return []
        return items[:limit]

    def mark_keyword_tts_completed(self, keyword: str) -> None:
        normalized = self._normalize_keyword_text(keyword)
        if not normalized:
            return
        with self._state_lock:
            if normalized in self._completed_tts_keyword_set:
                return
            self._completed_tts_keyword_set.add(normalized)
            self.completed_tts_keywords.append(normalized)

    def set_single_keyword_mode(self, enabled: bool) -> None:
        self.single_keyword_mode = bool(enabled)
        log_print(
            "INFO",
            f"Single keyword mode: {self.single_keyword_mode}",
            session_id=self.session_id,
        )

    def _save_audio(self) -> None:
        if not self.recording_buffer:
            return
        try:
            from datetime import datetime

            raw_audio = b"".join(self.recording_buffer)
            self.recording_buffer = []
            audio_dir = get_session_subdir(self.session_id, "audio")
            audio = AudioSegment(
                data=raw_audio,
                sample_width=2,  # 16-bit
                frame_rate=AUDIO_RATE,
                channels=1,
            )
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
