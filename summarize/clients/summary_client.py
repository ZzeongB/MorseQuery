"""OpenAI Realtime API client for audio transcription.

Audio is continuously streamed from a microphone to the Realtime API.
start_listening() / end_listening() mark segment boundaries for summarization.
"""

import base64
import json
import re
import threading
import time
from enum import Enum
from typing import Callable, Optional

import numpy as np
import pyaudio
import websocket

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


class TranscriptSyncMode(Enum):
    """Transcription synchronization mode for summarization."""
    VAD = "vad"
    COMMIT = "commit"
    SPEECH_WAIT = "speech_wait"
    VAD_THEN_COMMIT = "vad_then_commit"


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
        mic_id: str = "summary",
    ):
        self.sio = socketio
        self.session_id = session_id
        self.source = source
        self.device_indices = device_indices or []
        self.audio_file = audio_file
        self.noise_cut_threshold = max(0, int(noise_cut_threshold or 0))
        self.mic_id = mic_id

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Audio streaming & recording
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream = None
        self.recording_buffer: list[bytes] = []

        # Rolling buffer for last 3 seconds
        self.recent_audio_buffer: list[bytes] = []
        self.chunks_for_3_seconds = int(3 * AUDIO_RATE / AUDIO_CHUNK)

        # Segment state
        self.listening = False
        self.segment_id = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        # VAD transcript callbacks
        self._vad_transcript_callbacks: list[
            Callable[[str, Optional[float], Optional[float]], None]
        ] = []
        self._vad_transcript_lock = threading.Lock()
        self._vad_pending_transcripts: list[
            tuple[str, Optional[float], Optional[float]]
        ] = []
        self._vad_flush_timer: Optional[threading.Timer] = None

        # VAD state tracking
        self._vad_state_lock = threading.Lock()
        self._is_speaking = False
        self._speech_start_ts: Optional[float] = None
        self._speech_stop_ts: Optional[float] = None
        self._last_speech_start_ts: Optional[float] = None
        self._last_speech_stop_ts: Optional[float] = None
        self._current_rms: float = 0.0
        self._rms_history: list[float] = []
        self._rms_history_max_len = 10
        self._speech_segment_rms_samples: list[float] = []
        self._last_speech_segment_rms: float = 0.0

        # Transcript synchronization state
        self._sync_lock = threading.Lock()
        self._sync_mode: TranscriptSyncMode = TranscriptSyncMode.VAD
        self._waiting_for_commit_transcript = False
        self._commit_transcript_event = threading.Event()
        self._commit_transcript_result: Optional[str] = None
        self._waiting_for_speech_transcript = False
        self._speech_transcript_event = threading.Event()
        self._speech_transcript_result: Optional[str] = None
        self._pending_speech_count = 0
        self._pending_commit_no_wait = False
        self._pending_commit_no_wait_ts: Optional[float] = None

        log_print(
            "INFO",
            "SummaryClient created",
            session_id=session_id,
            source=source,
            devices=device_indices,
            audio_file=audio_file,
            noise_cut_threshold=self.noise_cut_threshold,
        )
        self.logger.log(
            "summary_client_created",
            source=source,
            devices=device_indices,
            audio_file=audio_file,
            noise_cut_threshold=self.noise_cut_threshold,
        )

    def add_vad_transcript_callback(
        self, callback: Callable[[str, Optional[float], Optional[float]], None]
    ) -> None:
        """Add callback to be called when VAD transcript is completed."""
        self._vad_transcript_callbacks.append(callback)
        log_print(
            "INFO",
            "VAD transcript callback added",
            session_id=self.session_id,
        )

    def get_vad_state(self) -> dict:
        """Get current VAD state for diarization."""
        with self._vad_state_lock:
            return {
                "is_speaking": self._is_speaking,
                "speech_start_ts": self._speech_start_ts,
                "speech_stop_ts": self._speech_stop_ts,
                "current_rms": self._current_rms,
                "last_speech_segment_rms": self._last_speech_segment_rms,
                "mic_id": self.mic_id,
            }

    def _handle_vad_transcript(
        self,
        transcript: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> None:
        """Handle completed VAD transcript by calling all registered callbacks."""
        for callback in self._vad_transcript_callbacks:
            try:
                callback(transcript, start_ts, end_ts)
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

        for text, start_ts, end_ts in pending:
            self._handle_vad_transcript(text, start_ts, end_ts)

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
        with self._vad_state_lock:
            start_ts = self._last_speech_start_ts
            end_ts = self._last_speech_stop_ts
        with self._vad_transcript_lock:
            self._vad_pending_transcripts.append((transcript, start_ts, end_ts))
            queued_count = len(self._vad_pending_transcripts)
            if queued_count >= _VAD_TRANSCRIPT_BATCH_TARGET:
                flush_now = True

        self.logger.log(
            "vad_transcript_queued",
            queued_count=queued_count,
            segment_id=self.segment_id,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        if flush_now:
            self._flush_pending_vad_transcripts(force=True)
        else:
            self._schedule_vad_transcript_flush()

    def start_listening(self) -> None:
        """Mark the start of a segment to summarize later."""
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

    def end_listening(
        self,
        mode: TranscriptSyncMode = TranscriptSyncMode.VAD,
        timeout: float = 5.0,
    ) -> Optional[str]:
        """End the segment and optionally wait for transcription based on mode."""
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "end_listening ignored (not connected)",
                session_id=self.session_id,
            )
            return None

        if not self.listening:
            log_print(
                "WARN",
                "end_listening ignored (not listening)",
                session_id=self.session_id,
            )
            return None

        self.listening = False
        self.response_buffer = ""

        self.logger.log(
            "end_listening",
            segment_id=self.segment_id,
            mode=mode.value,
        )
        self.sio.emit("listening_end", {"segment_id": self.segment_id, "mode": mode.value})

        return None

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
            transcript_text = transcript.strip() if transcript else ""

            with self._sync_lock:
                if self._pending_speech_count > 0:
                    self._pending_speech_count -= 1

                if self._waiting_for_commit_transcript:
                    self._commit_transcript_result = transcript_text
                    self._commit_transcript_event.set()

                if self._waiting_for_speech_transcript and self._pending_speech_count == 0:
                    self._speech_transcript_result = transcript_text
                    self._speech_transcript_event.set()

                if self._pending_commit_no_wait:
                    self._pending_commit_no_wait = False
                    self._pending_commit_no_wait_ts = None

            if transcript_text:
                self._queue_vad_transcript(transcript_text)
                self.logger.log(
                    "vad_transcript_completed",
                    transcript=transcript_text,
                    segment_id=self.segment_id,
                )
            return

        if etype == "input_audio_buffer.speech_started":
            received_at_ts = time.time()
            with self._vad_state_lock:
                self._is_speaking = True
                self._speech_start_ts = received_at_ts
                self._speech_segment_rms_samples.clear()
            with self._sync_lock:
                self._pending_speech_count += 1
            log_print(
                "INFO",
                f"VAD speech_started ({self.mic_id})",
                session_id=self.session_id,
                received_at_ts=received_at_ts,
            )
            self.logger.log(
                "summary_vad_boundary",
                boundary_type="speech_started",
                received_at_ts=received_at_ts,
                mic_id=self.mic_id,
            )
            return

        if etype == "input_audio_buffer.speech_stopped":
            received_at_ts = time.time()
            with self._vad_state_lock:
                self._is_speaking = False
                self._speech_stop_ts = received_at_ts
                self._last_speech_start_ts = self._speech_start_ts
                self._last_speech_stop_ts = received_at_ts
                self._speech_start_ts = None
                if self._speech_segment_rms_samples:
                    self._last_speech_segment_rms = sum(
                        self._speech_segment_rms_samples
                    ) / len(self._speech_segment_rms_samples)
                else:
                    self._last_speech_segment_rms = 0.0
                segment_rms = self._last_speech_segment_rms
            log_print(
                "INFO",
                f"VAD speech_stopped ({self.mic_id})",
                session_id=self.session_id,
                received_at_ts=received_at_ts,
                segment_rms=segment_rms,
            )
            self.logger.log(
                "summary_vad_boundary",
                boundary_type="speech_stopped",
                received_at_ts=received_at_ts,
                mic_id=self.mic_id,
                segment_rms=segment_rms,
            )
            return

        if etype in {"response.text.delta", "response.done"}:
            return

        if etype == "input_audio_buffer.committed":
            self.logger.log(
                "input_audio_buffer_committed",
                item_id=event.get("item_id"),
            )
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

    # -------------------------
    # Audio streaming
    # -------------------------

    def _send_audio_chunk(self, data: bytes) -> bool:
        """Send a single audio chunk to WebSocket and update buffers."""
        self.recording_buffer.append(data)

        self.recent_audio_buffer.append(data)
        if len(self.recent_audio_buffer) > self.chunks_for_3_seconds:
            self.recent_audio_buffer.pop(0)

        # Calculate RMS
        try:
            arr = np.frombuffer(data, dtype=np.int16)
            if len(arr) > 0:
                rms = float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))
                with self._vad_state_lock:
                    self._rms_history.append(rms)
                    if len(self._rms_history) > self._rms_history_max_len:
                        self._rms_history.pop(0)
                    self._current_rms = sum(self._rms_history) / len(self._rms_history)
                    if self._is_speaking:
                        self._speech_segment_rms_samples.append(rms)
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

            self.pa = pa
            self.stream = stream

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

                if self.noise_cut_threshold > 0:
                    arr = np.frombuffer(data, dtype=np.int16).copy()
                    arr[np.abs(arr) < self.noise_cut_threshold] = 0
                    data = arr.tobytes()

                self._send_audio_chunk(data)

        except Exception as e:
            log_print(
                "ERROR",
                f"SummaryClient failed to open device {device_idx}: {e}",
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

                if len(data) < chunk_bytes:
                    data = data + b"\x00" * (chunk_bytes - len(data))

                if self.noise_cut_threshold > 0:
                    arr = np.frombuffer(data, dtype=np.int16).copy()
                    arr[np.abs(arr) < self.noise_cut_threshold] = 0
                    data = arr.tobytes()

                self._send_audio_chunk(data)
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

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start the client."""
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

        with self._sync_lock:
            self._waiting_for_commit_transcript = False
            self._commit_transcript_event.set()
            self._commit_transcript_result = None
            self._waiting_for_speech_transcript = False
            self._speech_transcript_event.set()
            self._speech_transcript_result = None
            self._pending_speech_count = 0

        self.logger.log("summary_client_stop")

        with self._lock:
            if self.stream:
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

        time.sleep(0.15)

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
        """Save recorded audio as WAV."""
        try:
            from datetime import datetime

            raw_audio = (
                b"".join(self.recording_buffer) if self.recording_buffer else b""
            )
            self.recording_buffer = []

            audio_dir = get_session_subdir(self.session_id, "audio")

            if raw_audio:
                audio = AudioSegment(
                    data=raw_audio,
                    sample_width=2,
                    frame_rate=AUDIO_RATE,
                    channels=1,
                )
            else:
                audio = AudioSegment.silent(duration=0, frame_rate=AUDIO_RATE)

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
