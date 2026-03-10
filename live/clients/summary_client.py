"""OpenAI Realtime API client for missed-segment recovery.

Audio is continuously streamed from a microphone to the Realtime API.
start_listening() / end_listening() mark segment boundaries for summarization.
"""

import base64
import json
import re
import threading
import time
from typing import Callable, Optional

import noisereduce as nr
import numpy as np
import pyaudio
import websocket
from config import (
    AUDIO_CHUNK,
    AUDIO_RATE,
    OPENAI_API_KEY,
    OPENAI_REALTIME_URL,
    OPENAI_SESSION_CONFIG,
)
from flask_socketio import SocketIO
from logger import get_logger, get_session_subdir, log_print
from pydub import AudioSegment

from .prompt import SUMMARY_SESSION_INSTRUCTIONS
from .tts_client import TTSClient

_VAD_TRANSCRIPT_BATCH_TARGET = 2
_VAD_TRANSCRIPT_DEBOUNCE_SEC = 0.55


class SummaryClient:
    """Realtime client that continuously listens to audio from a microphone.

    start_listening() / end_listening() mark segment boundaries.
    On end_listening(), summarizes what was said in that segment.
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        source: str = "mic",
        device_indices: list[int] | None = None,
        audio_file: str | None = None,
        noise_cut_threshold: int = 0,
        enable_tts: bool = True,
        prepare_tts_on_callback: bool = True,
        mic_id: str = "summary",
        voice_id: str | None = None,
        output_device_index: int | None = None,
        enable_noise_reduction: bool = False,
        noise_profile_duration: float = 2.0,
        noise_reduction_buffer_duration: float = 0.5,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.source = source
        self.device_indices = device_indices or []
        self.audio_file = audio_file
        self.noise_cut_threshold = max(0, int(noise_cut_threshold or 0))
        self.mic_id = mic_id
        self.voice_id = voice_id
        self.output_device_index = output_device_index

        # Noise reduction settings
        self.enable_noise_reduction = enable_noise_reduction
        self.noise_profile_duration = noise_profile_duration
        self.noise_reduction_buffer_duration = noise_reduction_buffer_duration
        self.noise_profile: Optional[np.ndarray] = None
        self.noise_profile_collected = False
        self.chunks_for_profile = int(noise_profile_duration * AUDIO_RATE / AUDIO_CHUNK)
        self.chunks_for_nr_buffer = int(
            noise_reduction_buffer_duration * AUDIO_RATE / AUDIO_CHUNK
        )
        self.nr_buffer: list[bytes] = []

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Audio streaming & recording
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.recording_buffer: list[bytes] = []

        # Rolling buffer for last 3 seconds (used when start_listening)
        self.recent_audio_buffer: list[bytes] = []
        self.chunks_for_3_seconds = int(3 * AUDIO_RATE / AUDIO_CHUNK)  # 15 chunks

        # Context from before the current listening segment
        self.pre_context = ""

        # Segment state
        self.listening = False
        self.segment_id = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # TTS client
        self.enable_tts = enable_tts
        self.prepare_tts_on_callback = prepare_tts_on_callback
        if enable_tts and voice_id:
            self.tts_client = TTSClient(
                socketio,
                session_id,
                voice_id=voice_id,
                output_device_index=output_device_index,
            )
        else:
            self.tts_client = None

        # Callback for summary judgment (called when summary is generated)
        self.on_summary_callback: Optional[Callable[[str, int], None]] = None
        self.on_tts_ready_callback: Optional[Callable[[bool], None]] = None

        # VAD transcript callbacks (called when input audio transcription completes)
        # Callback signature: (transcript: str, start_ts: Optional[float]) -> None
        self._vad_transcript_callbacks: list[
            Callable[[str, Optional[float]], None]
        ] = []
        self._vad_transcript_lock = threading.Lock()
        # Store tuples of (transcript, start_ts)
        self._vad_pending_transcripts: list[tuple[str, Optional[float]]] = []
        self._vad_flush_timer: Optional[threading.Timer] = None

        # VAD state tracking for diarization
        self._vad_state_lock = threading.Lock()
        self._is_speaking = False
        self._speech_start_ts: Optional[float] = None
        self._last_speech_start_ts: Optional[float] = (
            None  # Preserved until transcript arrives
        )
        self._current_rms: float = 0.0
        self._rms_history: list[float] = []  # Rolling window for smoothed RMS
        self._rms_history_max_len = 10  # ~200ms at 50 chunks/sec

        log_print(
            "INFO",
            "SummaryClient created",
            session_id=session_id,
            source=source,
            devices=device_indices,
            audio_file=audio_file,
            noise_cut_threshold=self.noise_cut_threshold,
            tts_enabled=enable_tts,
            prepare_tts_on_callback=prepare_tts_on_callback,
            voice_id=voice_id,
            output_device_index=output_device_index,
            enable_noise_reduction=enable_noise_reduction,
            noise_profile_duration=noise_profile_duration,
            noise_reduction_buffer_duration=noise_reduction_buffer_duration,
        )
        self.logger.log(
            "summary_client_created",
            source=source,
            devices=device_indices,
            audio_file=audio_file,
            noise_cut_threshold=self.noise_cut_threshold,
            tts_enabled=enable_tts,
            prepare_tts_on_callback=prepare_tts_on_callback,
            voice_id=voice_id,
            output_device_index=output_device_index,
            enable_noise_reduction=enable_noise_reduction,
            noise_profile_duration=noise_profile_duration,
            noise_reduction_buffer_duration=noise_reduction_buffer_duration,
        )

    # -------------------------
    # Public APIs
    # -------------------------

    def set_summary_callback(self, callback: Callable[[str, int], None]) -> None:
        """Set callback to be called when a summary is generated.

        The callback receives (summary_text, segment_id) and is used by
        ContextJudgeClient to determine whether to play the TTS.

        Args:
            callback: Function taking (summary: str, segment_id: int)
        """
        self.on_summary_callback = callback
        log_print(
            "INFO",
            "Summary callback set",
            session_id=self.session_id,
        )

    def set_tts_ready_callback(self, callback: Callable[[bool], None]) -> None:
        """Set callback called when async TTS queueing completes."""
        self.on_tts_ready_callback = callback
        log_print(
            "INFO",
            "Summary TTS ready callback set",
            session_id=self.session_id,
        )

    def add_vad_transcript_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback to be called when VAD transcript is completed.

        Args:
            callback: Function taking (transcript: str)
        """
        self._vad_transcript_callbacks.append(callback)
        log_print(
            "INFO",
            "VAD transcript callback added",
            session_id=self.session_id,
        )

    def get_vad_state(self) -> dict:
        """Get current VAD state for diarization.

        Returns:
            dict with keys:
                - is_speaking: bool
                - speech_start_ts: float or None
                - current_rms: float
                - mic_id: str
        """
        with self._vad_state_lock:
            return {
                "is_speaking": self._is_speaking,
                "speech_start_ts": self._speech_start_ts,
                "current_rms": self._current_rms,
                "mic_id": self.mic_id,
            }

    def _handle_vad_transcript(
        self, transcript: str, start_ts: Optional[float] = None
    ) -> None:
        """Handle completed VAD transcript by calling all registered callbacks."""
        for callback in self._vad_transcript_callbacks:
            try:
                callback(transcript, start_ts)
            except Exception as e:
                log_print(
                    "ERROR",
                    f"VAD transcript callback failed: {e}",
                    session_id=self.session_id,
                )
                self.logger.log("vad_transcript_callback_failed", error=str(e))

    def _flush_pending_vad_transcripts(self, force: bool = False) -> None:
        """Flush queued VAD transcripts to callbacks."""
        with self._vad_transcript_lock:
            if not self._vad_pending_transcripts:
                return
            if (
                not force
                and len(self._vad_pending_transcripts) < _VAD_TRANSCRIPT_BATCH_TARGET
            ):
                return
            pending = self._vad_pending_transcripts[:]
            self._vad_pending_transcripts.clear()
            timer = self._vad_flush_timer
            self._vad_flush_timer = None

        if timer:
            try:
                timer.cancel()
            except Exception:
                pass

        for text, start_ts in pending:
            self._handle_vad_transcript(text, start_ts)

        self.logger.log(
            "vad_transcript_flush",
            count=len(pending),
            force=force,
            segment_id=self.segment_id,
        )

    def _schedule_vad_transcript_flush(self) -> None:
        with self._vad_transcript_lock:
            if self._vad_flush_timer:
                try:
                    self._vad_flush_timer.cancel()
                except Exception:
                    pass
            timer = threading.Timer(
                _VAD_TRANSCRIPT_DEBOUNCE_SEC,
                self._flush_pending_vad_transcripts,
                kwargs={"force": True},
            )
            timer.daemon = True
            self._vad_flush_timer = timer
            timer.start()

    def _queue_vad_transcript(self, transcript: str) -> None:
        flush_now = False
        # Capture start_ts before it gets overwritten by next speech
        with self._vad_state_lock:
            start_ts = self._last_speech_start_ts
        with self._vad_transcript_lock:
            self._vad_pending_transcripts.append((transcript, start_ts))
            queued_count = len(self._vad_pending_transcripts)
            if queued_count >= _VAD_TRANSCRIPT_BATCH_TARGET:
                flush_now = True

        self.logger.log(
            "vad_transcript_queued",
            queued_count=queued_count,
            segment_id=self.segment_id,
            start_ts=start_ts,
        )

        if flush_now:
            self._flush_pending_vad_transcripts(force=True)
        else:
            self._schedule_vad_transcript_flush()

    def _run_summary_callback(
        self, callback: Callable[[str, int], None], summary: str, segment_id: int
    ) -> None:
        """Run summary callback safely in background."""
        try:
            callback(summary, segment_id)
        except Exception as e:
            log_print(
                "ERROR",
                f"Summary callback failed: {e}",
                session_id=self.session_id,
            )
            self.logger.log("summary_callback_failed", error=str(e))

    def _resolve_tts_ready_callback(self) -> Optional[Callable[[bool], None]]:
        """Resolve TTS completion callback from the summary callback target.

        When summary callback is a bound method on ContextJudgeClient, we can
        use its on_tts_ready(success) method to synchronize playback.
        """
        if self.on_tts_ready_callback:
            return self.on_tts_ready_callback

        callback = self.on_summary_callback
        if not callback:
            return None

        callback_owner = getattr(callback, "__self__", None)
        if callback_owner is None:
            return None

        tts_ready_callback = getattr(callback_owner, "on_tts_ready", None)
        if callable(tts_ready_callback):
            return tts_ready_callback
        return None

    def speak_summary(self, text: str, language: str = "en") -> None:
        """Convert text to speech and emit via Socket.IO.

        Args:
            text: Text to convert to speech
            language: Language code (default: "en")
        """
        if not self.tts_client:
            log_print(
                "WARN",
                "TTS not enabled, cannot speak summary",
                session_id=self.session_id,
            )
            return

        self.tts_client.synthesize_async(
            text, event_name="summary_tts", language=language
        )

    def start_listening(self) -> None:
        """Mark the start of a segment to summarize later.

        Audio continues streaming. This just marks a boundary.
        """
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "start_listening ignored (not connected)",
                session_id=self.session_id,
            )
            return

        self.segment_id += 1
        self.listening = True
        self.response_buffer = ""

        self.logger.log("start_listening", segment_id=self.segment_id)
        self.sio.emit("listening_start", {"segment_id": self.segment_id})

    def end_listening(self) -> None:
        """End the segment (VAD transcript mode only; no summary generation)."""
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "end_listening ignored (not connected)",
                session_id=self.session_id,
            )
            return

        if not self.listening:
            log_print(
                "WARN",
                "end_listening ignored (not listening)",
                session_id=self.session_id,
            )
            return

        self.listening = False
        self.response_buffer = ""

        self.logger.log("end_listening", segment_id=self.segment_id)
        self.sio.emit("listening_end", {"segment_id": self.segment_id})
        self.logger.log(
            "summary_request_skipped_vad_only",
            segment_id=self.segment_id,
        )

    # -------------------------
    # Websocket handlers
    # -------------------------

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        log_print(
            "INFO", "SummaryClient WebSocket connected", session_id=self.session_id
        )
        self.logger.log("summary_ws_connected")

        session_config = {
            **OPENAI_SESSION_CONFIG,
            "instructions": SUMMARY_SESSION_INSTRUCTIONS,
        }
        ws.send(json.dumps({"type": "session.update", "session": session_config}))

        # Start audio streaming if source is configured
        if (self.source == "mic" and self.device_indices) or (
            self.source == "mp3" and self.audio_file
        ):
            threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "summary_openai_session_created", session_id=session_info.get("id")
            )
            return

        if etype == "session.updated":
            self.logger.log("summary_openai_session_updated")
            return

        if etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            if transcript and transcript.strip():
                self._queue_vad_transcript(transcript.strip())
                self.logger.log(
                    "vad_transcript_completed",
                    transcript=transcript.strip(),
                    segment_id=self.segment_id,
                )
            return

        if etype == "input_audio_buffer.speech_started":
            received_at_ts = time.time()
            with self._vad_state_lock:
                self._is_speaking = True
                self._speech_start_ts = received_at_ts
            log_print(
                "INFO",
                "VAD speech_started ({})".format(self.mic_id or "unknown"),
                session_id=self.session_id,
                received_at_ts=received_at_ts,
            )
            self.logger.log(
                "summary_vad_boundary",
                boundary_type="speech_started",
                received_at_ts=received_at_ts,
                audio_start_ms=event.get("audio_start_ms"),
                item_id=event.get("item_id"),
                mic_id=self.mic_id,
            )
            return

        if etype == "input_audio_buffer.speech_stopped":
            received_at_ts = time.time()
            with self._vad_state_lock:
                self._is_speaking = False
                # Preserve start_ts for transcript callback
                self._last_speech_start_ts = self._speech_start_ts
                self._speech_start_ts = None
            log_print(
                "INFO",
                "VAD speech_stopped ({})".format(self.mic_id or "unknown"),
                session_id=self.session_id,
                received_at_ts=received_at_ts,
            )
            self.logger.log(
                "summary_vad_boundary",
                boundary_type="speech_stopped",
                received_at_ts=received_at_ts,
                audio_end_ms=event.get("audio_end_ms"),
                item_id=event.get("item_id"),
                mic_id=self.mic_id,
            )
            return

        if etype in {"response.text.delta", "response.done"}:
            # Summary generation path is disabled in VAD-only mode.
            return

        if etype == "error":
            err = event.get("error", {}).get("message", "Unknown error")
            log_print(
                "ERROR",
                f"SummaryClient OpenAI error: {err}",
                session_id=self.session_id,
            )
            self.logger.log("summary_openai_error", error=err)
            self.sio.emit("summary_error", {"error": err})
            return

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        log_print(
            "ERROR",
            f"SummaryClient websocket error: {error}",
            session_id=self.session_id,
        )
        self.logger.log("summary_ws_error", error=str(error))
        self.sio.emit("summary_error", {"error": str(error)})

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        self.connected = False
        self.running = False
        log_print("INFO", "SummaryClient closed", session_id=self.session_id)
        self.logger.log("summary_ws_closed", status=status, message=msg)

        with self._lock:
            self.ws = None

        self.sio.emit("summary_closed")

    def _sanitize_summary_output(self, raw: str) -> str:
        """Normalize and validate summary output.

        Reject structured/metadata-like outputs to prevent JSON/timestamp leakage.
        Returns empty string when output is not a valid plain summary sentence.
        """
        summary = (raw or "").strip()
        if not summary or summary == "...":
            return ""

        # Remove a single pair of wrapping quotes if model returns quoted sentence.
        if len(summary) >= 2 and summary[0] == '"' and summary[-1] == '"':
            summary = summary[1:-1].strip()
        # Also handle single quotes
        if len(summary) >= 2 and summary[0] == "'" and summary[-1] == "'":
            summary = summary[1:-1].strip()

        # Re-check emptiness after stripping quotes
        if not summary:
            return ""

        lower = summary.lower()

        # Reject obvious structured/meta outputs.
        if lower.startswith("{") or lower.startswith("["):
            return ""
        if any(key in lower for key in ("start_time", "end_time", "timestamp")):
            return ""
        if re.search(r"\b\d{1,2}:\d{2}\b", summary):
            return ""
        if re.search(r"^\s*\w+\s*:\s*\w+", summary):
            # e.g. "summary: ...", "speaker: ..."
            return ""

        # Try JSON parse guard for object/array shaped strings.
        try:
            parsed = json.loads(summary)
            if isinstance(parsed, (dict, list)):
                return ""
        except Exception:
            pass

        return summary

    # -------------------------
    # Audio streaming
    # -------------------------

    def _collect_noise_profile(self, audio_data: np.ndarray) -> None:
        """Collect noise profile from initial audio samples."""
        self.noise_profile = audio_data.astype(np.float32)
        self.noise_profile_collected = True
        log_print(
            "INFO",
            f"Noise profile collected ({len(audio_data)} samples)",
            session_id=self.session_id,
        )
        self.logger.log(
            "noise_profile_collected",
            samples=len(audio_data),
            duration_sec=len(audio_data) / AUDIO_RATE,
        )

    def _apply_noise_reduction(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply noise reduction to audio data using the collected profile."""
        if self.noise_profile is None:
            return audio_data

        try:
            reduced = nr.reduce_noise(
                y=audio_data.astype(np.float32),
                sr=AUDIO_RATE,
                y_noise=self.noise_profile,
                stationary=True,
                prop_decrease=0.8,
            )
            return reduced.astype(np.int16)
        except Exception as e:
            log_print(
                "WARN",
                f"Noise reduction failed: {e}",
                session_id=self.session_id,
            )
            return audio_data

    def _send_audio_chunk(self, data: bytes) -> bool:
        """Send a single audio chunk to WebSocket and update buffers.

        Returns True if sent successfully, False otherwise.
        """
        # Record audio
        self.recording_buffer.append(data)

        # Maintain rolling buffer for last 3 seconds
        self.recent_audio_buffer.append(data)
        if len(self.recent_audio_buffer) > self.chunks_for_3_seconds:
            self.recent_audio_buffer.pop(0)

        # Calculate RMS for diarization
        try:
            arr = np.frombuffer(data, dtype=np.int16)
            if len(arr) > 0:
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                with self._vad_state_lock:
                    self._rms_history.append(rms)
                    if len(self._rms_history) > self._rms_history_max_len:
                        self._rms_history.pop(0)
                    self._current_rms = sum(self._rms_history) / len(self._rms_history)
        except Exception:
            pass

        audio_b64 = base64.b64encode(data).decode()

        with self._lock:
            ws = self.ws
            if ws and not self._shutdown_event.is_set():
                try:
                    ws.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64,
                            }
                        )
                    )
                    return True
                except Exception:
                    return False
        return False

    def _stream_audio(self) -> None:
        """Stream audio from configured source."""
        if self.source == "mp3":
            self._stream_audio_from_mp3()
        else:
            self._stream_audio_from_mic()

    def _stream_audio_from_mic(self) -> None:
        """Stream audio from configured microphone."""
        if not self.device_indices:
            log_print(
                "WARN",
                "SummaryClient: no device configured",
                session_id=self.session_id,
            )
            return

        if not self.running:
            log_print(
                "WARN",
                "SummaryClient: not running, skipping audio stream",
                session_id=self.session_id,
            )
            return

        device_idx = self.device_indices[0]
        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(device_idx)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=AUDIO_CHUNK,
            )
            device_name = info["name"]
            log_print(
                "INFO",
                f"SummaryClient opened mic {device_idx}: {device_name}",
                session_id=self.session_id,
            )
            self.logger.log("summary_audio_start", device=device_name)

            # Store references for cleanup
            self.pa = pa
            self.stream = stream

            # Noise reduction state for this stream
            profile_buffer: list[bytes] = []
            nr_chunk_buffer: list[bytes] = []

            while self.running and self.connected and not self._shutdown_event.is_set():
                try:
                    data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                except Exception as e:
                    if self._shutdown_event.is_set():
                        break
                    log_print(
                        "WARN",
                        f"SummaryClient error reading device {device_idx}: {e}",
                        session_id=self.session_id,
                    )
                    continue

                # Apply fixed threshold gate for mic input when configured.
                if self.noise_cut_threshold > 0:
                    arr = np.frombuffer(data, dtype=np.int16).copy()
                    arr[np.abs(arr) < self.noise_cut_threshold] = 0
                    data = arr.tobytes()

                # Noise reduction: collect profile first, then buffer and process
                if self.enable_noise_reduction:
                    if not self.noise_profile_collected:
                        # Phase 1: Collecting noise profile
                        profile_buffer.append(data)
                        if len(profile_buffer) >= self.chunks_for_profile:
                            profile_audio = np.frombuffer(
                                b"".join(profile_buffer), dtype=np.int16
                            )
                            self._collect_noise_profile(profile_audio)
                            profile_buffer.clear()
                        continue  # Don't stream during profile collection

                    # Phase 2: Buffer chunks and apply noise reduction
                    nr_chunk_buffer.append(data)
                    if len(nr_chunk_buffer) >= self.chunks_for_nr_buffer:
                        # Combine buffer, apply NR, then split back
                        combined = np.frombuffer(
                            b"".join(nr_chunk_buffer), dtype=np.int16
                        )
                        reduced = self._apply_noise_reduction(combined)
                        reduced_bytes = reduced.tobytes()

                        # Split back into original chunk sizes
                        chunk_bytes = AUDIO_CHUNK * 2  # 16-bit = 2 bytes per sample
                        for i in range(0, len(reduced_bytes), chunk_bytes):
                            chunk_data = reduced_bytes[i : i + chunk_bytes]
                            if len(chunk_data) < chunk_bytes:
                                break
                            self._send_audio_chunk(chunk_data)
                        nr_chunk_buffer.clear()
                else:
                    # No noise reduction - send directly
                    self._send_audio_chunk(data)

        except Exception as e:
            log_print(
                "ERROR",
                f"SummaryClient failed to open device {device_idx}: {e}",
                session_id=self.session_id,
            )
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
            self.stream = None
            self.pa = None

            log_print(
                "INFO",
                "SummaryClient audio streaming stopped",
                session_id=self.session_id,
            )

    def _stream_audio_from_mp3(self) -> None:
        """Stream audio from configured MP3 file."""
        if not self.audio_file:
            log_print(
                "WARN",
                "SummaryClient: no mp3 file configured",
                session_id=self.session_id,
            )
            return

        if not self.running:
            log_print(
                "WARN",
                "SummaryClient: not running, skipping MP3 stream",
                session_id=self.session_id,
            )
            return

        try:
            audio = AudioSegment.from_file(self.audio_file)
            audio = audio.set_frame_rate(AUDIO_RATE).set_channels(1).set_sample_width(2)
            raw = audio.raw_data

            log_print(
                "INFO",
                "SummaryClient MP3 stream started",
                session_id=self.session_id,
                file=self.audio_file,
                duration_ms=len(audio),
            )
            self.logger.log(
                "summary_mp3_stream_start",
                file=self.audio_file,
                duration_ms=len(audio),
            )

            chunk_bytes = AUDIO_CHUNK * 2

            # Noise reduction state for this stream
            profile_buffer: list[bytes] = []
            nr_chunk_buffer: list[bytes] = []

            for i in range(0, len(raw), chunk_bytes):
                if (
                    not self.running
                    or not self.connected
                    or self._shutdown_event.is_set()
                ):
                    break

                data = raw[i : i + chunk_bytes]
                if not data:
                    break

                # Pad tail chunk to keep frame alignment.
                if len(data) < chunk_bytes:
                    data = data + b"\x00" * (chunk_bytes - len(data))

                if self.noise_cut_threshold > 0:
                    arr = np.frombuffer(data, dtype=np.int16).copy()
                    arr[np.abs(arr) < self.noise_cut_threshold] = 0
                    data = arr.tobytes()

                # Noise reduction: collect profile first, then buffer and process
                if self.enable_noise_reduction:
                    if not self.noise_profile_collected:
                        # Phase 1: Collecting noise profile
                        profile_buffer.append(data)
                        if len(profile_buffer) >= self.chunks_for_profile:
                            profile_audio = np.frombuffer(
                                b"".join(profile_buffer), dtype=np.int16
                            )
                            self._collect_noise_profile(profile_audio)
                            profile_buffer.clear()
                        time.sleep(AUDIO_CHUNK / AUDIO_RATE)
                        continue  # Don't stream during profile collection

                    # Phase 2: Buffer chunks and apply noise reduction
                    nr_chunk_buffer.append(data)
                    if len(nr_chunk_buffer) >= self.chunks_for_nr_buffer:
                        # Combine buffer, apply NR, then split back
                        combined = np.frombuffer(
                            b"".join(nr_chunk_buffer), dtype=np.int16
                        )
                        reduced = self._apply_noise_reduction(combined)
                        reduced_bytes = reduced.tobytes()

                        # Split back into original chunk sizes and send
                        for j in range(0, len(reduced_bytes), chunk_bytes):
                            chunk_data = reduced_bytes[j : j + chunk_bytes]
                            if len(chunk_data) < chunk_bytes:
                                break
                            self._send_audio_chunk(chunk_data)
                            time.sleep(AUDIO_CHUNK / AUDIO_RATE)
                        nr_chunk_buffer.clear()
                    continue  # Skip final sleep when using NR buffering

                # No noise reduction - send directly
                self._send_audio_chunk(data)

                # Keep real-time pacing
                time.sleep(AUDIO_CHUNK / AUDIO_RATE)

        except Exception as e:
            log_print(
                "ERROR",
                f"SummaryClient failed to stream mp3: {e}",
                session_id=self.session_id,
            )
            self.logger.log(
                "summary_mp3_stream_error", error=str(e), file=self.audio_file
            )
        finally:
            log_print(
                "INFO",
                "SummaryClient MP3 streaming stopped",
                session_id=self.session_id,
                file=self.audio_file,
            )

    def _close_stream(self) -> None:
        """Close audio stream."""
        if hasattr(self, "stream") and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        if self.pa:
            try:
                self.pa.terminate()
            except Exception:
                pass
            self.pa = None

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start the client; use start_listening/end_listening for segment summaries."""
        self._shutdown_event.clear()
        self.running = True
        self.logger.log("summary_client_start")

        with self._lock:
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
        """Stop the client and close the websocket."""
        self.running = False
        self.connected = False
        self.listening = False
        self._shutdown_event.set()
        self._flush_pending_vad_transcripts(force=True)

        # Reset noise reduction state for potential reuse
        self.noise_profile = None
        self.noise_profile_collected = False
        self.nr_buffer.clear()

        self.logger.log("summary_client_stop")

        # Wait briefly for audio thread to exit cleanly
        time.sleep(0.15)

        # Save recorded audio
        self._save_audio()

        with self._lock:
            ws = self.ws
            self.ws = None

        if ws:
            try:
                ws.close()
            except Exception:
                pass

        self.sio.emit("summary_closed")

    def _save_audio(self) -> None:
        """Save recorded audio as WAV (empty file if no audio above threshold)."""
        try:
            from datetime import datetime

            # Combine all chunks (may be empty)
            raw_audio = (
                b"".join(self.recording_buffer) if self.recording_buffer else b""
            )
            self.recording_buffer = []

            # Create session-scoped audio directory
            audio_dir = get_session_subdir(self.session_id, "audio")

            # Convert raw PCM to AudioSegment (or create empty)
            if raw_audio:
                audio = AudioSegment(
                    data=raw_audio,
                    sample_width=2,  # 16-bit
                    frame_rate=AUDIO_RATE,
                    channels=1,
                )
            else:
                audio = AudioSegment.silent(duration=0, frame_rate=AUDIO_RATE)

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
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to save audio: {e}",
                session_id=self.session_id,
            )
