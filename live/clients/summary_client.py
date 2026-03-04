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

import pyaudio
import websocket
from config import AUDIO_CHUNK, AUDIO_RATE, LOG_DIR, OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print
from pydub import AudioSegment

from .prompt import SUMMARY_SESSION_INSTRUCTIONS, build_summary_prompt
from .tts_client import TTSClient


class SummaryClient:
    """Realtime client that continuously listens to audio from a microphone.

    start_listening() / end_listening() mark segment boundaries.
    On end_listening(), summarizes what was said in that segment.
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        device_indices: list[int] | None = None,
        enable_tts: bool = True,
        mic_id: str = "summary",
        voice_id: str | None = None,
        output_device_index: int | None = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.device_indices = device_indices or []
        self.mic_id = mic_id
        self.voice_id = voice_id
        self.output_device_index = output_device_index

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

        # White noise playback
        self.noise_stream = None
        self.noise_playing = False

        # TTS client
        self.enable_tts = enable_tts
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

        log_print(
            "INFO",
            "SummaryClient created",
            session_id=session_id,
            devices=device_indices,
            tts_enabled=enable_tts,
            voice_id=voice_id,
            output_device_index=output_device_index,
        )
        self.logger.log(
            "summary_client_created",
            devices=device_indices,
            tts_enabled=enable_tts,
            voice_id=voice_id,
            output_device_index=output_device_index,
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

        # Clear buffer and re-append last 3 seconds as the start of the new segment
        with self._lock:
            ws = self.ws
            if ws:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                    # Re-append last 3 seconds of audio
                    for chunk in self.recent_audio_buffer:
                        audio_b64 = base64.b64encode(chunk).decode()
                        ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }
                            )
                        )
                except Exception:
                    pass

        self.logger.log("start_listening", segment_id=self.segment_id)
        self.sio.emit("listening_start", {"segment_id": self.segment_id})

        # Start white noise playback
        # self._start_white_noise()

    def end_listening(self) -> None:
        """End the segment and request a summary of what was said."""
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

        # Stop white noise playback
        # self._stop_white_noise()

        self.logger.log("end_listening", segment_id=self.segment_id)
        self.sio.emit("listening_end", {"segment_id": self.segment_id})

        prompt = build_summary_prompt(self.pre_context)

        self.logger.log(
            "summary_request",
            segment_id=self.segment_id,
            pre_context_chars=len(self.pre_context),
        )

        # Commit the segment audio and request summary (all under lock)
        with self._lock:
            ws = self.ws
            if not ws:
                log_print(
                    "WARN",
                    "end_listening ignored (no websocket)",
                    session_id=self.session_id,
                )
                return

            try:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text"],
                                "instructions": prompt,
                                "temperature": 0.6,
                                "max_output_tokens": 120,
                            },
                        }
                    )
                )
            except Exception as e:
                self.logger.log("end_listening_send_failed", error=str(e))

    # -------------------------
    # Websocket handlers
    # -------------------------

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        log_print(
            "INFO", "SummaryClient WebSocket connected", session_id=self.session_id
        )
        self.logger.log("summary_ws_connected")

        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": SUMMARY_SESSION_INSTRUCTIONS,
                    },
                }
            )
        )

        # Start audio streaming if devices are configured
        if self.device_indices:
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

        if etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
                self.sio.emit("summary_chunk", delta)
            return

        if etype == "response.done":
            raw = self.response_buffer
            self.response_buffer = ""

            self.logger.log("summary_response_done", raw=raw)

            summary = self._sanitize_summary_output(raw)
            is_empty = not summary

            payload = {
                "segment_id": self.segment_id,
                "is_empty": is_empty,
                "summary": "" if is_empty else summary,
            }

            # Update pre_context for next segment
            if not is_empty:
                self.pre_context = (self.pre_context + " " + summary).strip()
                if len(self.pre_context) > 500:
                    self.pre_context = self.pre_context[-500:]

            self.sio.emit("summary_done", payload)

            # If we have a valid summary and callback is set, run TTS + judge in parallel
            if not is_empty and self.on_summary_callback:
                segment_id = self.segment_id
                callback = self.on_summary_callback

                # Queue TTS for potential playback (ContextJudgeClient decides)
                if self.enable_tts and self.tts_client:
                    tts_ready_callback = self._resolve_tts_ready_callback()
                    if tts_ready_callback:
                        self.tts_client.queue_audio_with_callback(
                            summary,
                            callback=tts_ready_callback,
                            language="en",
                        )
                    else:
                        self.tts_client.queue_audio_async(summary, "en")
                    log_print(
                        "INFO",
                        f"TTS queued for parallel judgment: {summary[:50]}...",
                        session_id=self.session_id,
                    )

                # Trigger judgment in background so TTS and judge start in parallel
                threading.Thread(
                    target=self._run_summary_callback,
                    args=(callback, summary, segment_id),
                    daemon=True,
                ).start()

            elif not is_empty and self.enable_tts and self.tts_client:
                # No callback set - play TTS directly (fallback behavior)
                self.tts_client.synthesize_async(
                    summary,
                    event_name="summary_tts",
                    language="en",
                )
            elif is_empty:
                log_print(
                    "INFO",
                    "Skipping summary TTS: empty summary",
                    session_id=self.session_id,
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

    def _generate_white_noise(self, num_samples: int, volume: float = 0.1) -> bytes:
        """Generate white noise audio chunk."""
        import random
        import struct

        # Generate random samples scaled by volume
        max_val = int(32767 * volume)
        samples = [random.randint(-max_val, max_val) for _ in range(num_samples)]
        return struct.pack(f"{num_samples}h", *samples)

    def _start_white_noise(self) -> None:
        """Start playing white noise through speakers."""
        if self.noise_playing:
            return

        self.noise_playing = True
        threading.Thread(target=self._play_white_noise_loop, daemon=True).start()
        log_print("INFO", "White noise started", session_id=self.session_id)

    def _stop_white_noise(self) -> None:
        """Stop playing white noise."""
        self.noise_playing = False
        log_print("INFO", "White noise stopped", session_id=self.session_id)

    def _play_white_noise_loop(self) -> None:
        """Loop that plays white noise until stopped."""
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                output=True,
                frames_per_buffer=AUDIO_CHUNK,
            )

            while (
                self.noise_playing
                and self.running
                and not self._shutdown_event.is_set()
            ):
                noise = self._generate_white_noise(AUDIO_CHUNK, volume=0.05)
                try:
                    stream.write(noise)
                except Exception:
                    if self._shutdown_event.is_set():
                        break
        except Exception as e:
            log_print("ERROR", f"White noise error: {e}", session_id=self.session_id)
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

    def _stream_audio(self) -> None:
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

            commit_interval = 3.0
            last_commit = 0.0

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

                # Record audio
                self.recording_buffer.append(data)

                # Maintain rolling buffer for last 3 seconds
                self.recent_audio_buffer.append(data)
                if len(self.recent_audio_buffer) > self.chunks_for_3_seconds:
                    self.recent_audio_buffer.pop(0)

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

                            # Periodic commit (skip during listening to keep segment intact)
                            now = time.time()
                            if (
                                now - last_commit >= commit_interval
                                and not self.listening
                            ):
                                ws.send(
                                    json.dumps({"type": "input_audio_buffer.commit"})
                                )
                                last_commit = now
                        except Exception:
                            break

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

        # Stop white noise if playing
        self._stop_white_noise()

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

            # Create audio directory
            audio_dir = LOG_DIR / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)

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
