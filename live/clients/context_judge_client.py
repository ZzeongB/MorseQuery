"""OpenAI Realtime API client for context-aware TTS judgment.

This client continuously streams audio to OpenAI Realtime API and judges
whether a summary TTS should be played based on:
1. Catch-up Value: Does the missed content have important information?
2. Context Relevance: Is the summary related to current discussion?
3. Interrupt Timing: Is this a good moment to interrupt?
"""

import base64
import json
import threading
import time
from typing import Callable, Optional

import pyaudio
import websocket
from config import AUDIO_CHUNK, AUDIO_RATE, OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print

from .prompt import JUDGE_SESSION_INSTRUCTIONS, build_judgment_prompt
from .tts_client import TTSClient


class ContextJudgeClient:
    """Realtime client that judges whether to play TTS based on audio context.

    Continuously streams audio to OpenAI Realtime API. When judge_summary() is called,
    commits current audio buffer and requests LLM judgment on whether to play TTS.
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        device_indices: list[int] | None = None,
        tts_client: Optional[TTSClient] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.device_indices = device_indices or []
        self.tts_client = tts_client

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Audio streaming
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

        # Rolling buffer for last 3 seconds
        self.recent_audio_buffer: list[bytes] = []
        self.chunks_for_3_seconds = int(3 * AUDIO_RATE / AUDIO_CHUNK)

        # Listening state (synchronized with SummaryClient)
        self.listening = False
        self.segment_id = 0

        # Pending summary for judgment
        self.pending_summary: Optional[str] = None
        self.pending_segment_id: int = 0

        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        log_print(
            "INFO",
            "ContextJudgeClient created",
            session_id=session_id,
            devices=device_indices,
            has_tts=tts_client is not None,
        )
        self.logger.log(
            "context_judge_created",
            devices=device_indices,
            has_tts=tts_client is not None,
        )

    # -------------------------
    # Public APIs
    # -------------------------

    def start_listening(self) -> None:
        """Mark the start of a listening segment (synchronized with SummaryClient)."""
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

        # Clear buffer and re-append last 3 seconds
        with self._lock:
            ws = self.ws
            if ws:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                    # Re-append last 3 seconds of audio for context
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

        self.logger.log("judge_start_listening", segment_id=self.segment_id)
        log_print(
            "INFO",
            f"ContextJudgeClient start_listening segment={self.segment_id}",
            session_id=self.session_id,
        )

    def end_listening(self) -> None:
        """Mark the end of a listening segment (synchronized with SummaryClient)."""
        if not self.running or not self.connected:
            return

        self.listening = False
        self.logger.log("judge_end_listening", segment_id=self.segment_id)
        log_print(
            "INFO",
            f"ContextJudgeClient end_listening segment={self.segment_id}",
            session_id=self.session_id,
        )

    def judge_summary(self, summary: str, segment_id: int) -> None:
        """Judge whether to play the given summary TTS.

        Called by SummaryClient after generating a summary.
        Commits current audio buffer and requests LLM judgment.

        Args:
            summary: The summary text to judge
            segment_id: The segment ID this summary belongs to
        """
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "judge_summary ignored (not connected)",
                session_id=self.session_id,
            )
            return

        if not summary or not summary.strip():
            log_print(
                "INFO",
                "judge_summary ignored (empty summary)",
                session_id=self.session_id,
            )
            return

        self.pending_summary = summary
        self.pending_segment_id = segment_id
        self.response_buffer = ""

        log_print(
            "INFO",
            f"Judgment request: {summary[:60]}...",
            session_id=self.session_id,
            segment_id=segment_id,
        )
        self.logger.log(
            "judge_request",
            summary=summary,
            segment_id=segment_id,
        )

        # Build judgment prompt with summary text
        prompt = build_judgment_prompt(summary)

        with self._lock:
            ws = self.ws
            if not ws:
                log_print(
                    "WARN",
                    "judge_summary: no websocket",
                    session_id=self.session_id,
                )
                return

            try:
                # Commit current audio buffer
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                # Request judgment from LLM
                ws.send(
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
            except Exception as e:
                self.logger.log("judge_request_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"judge_summary send failed: {e}",
                    session_id=self.session_id,
                )

    # -------------------------
    # WebSocket handlers
    # -------------------------

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opened."""
        self.connected = True
        log_print(
            "INFO",
            "ContextJudgeClient WebSocket connected",
            session_id=self.session_id,
        )
        self.logger.log("judge_ws_connected")

        # Configure session for audio input with no VAD (we control timing)
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,  # No VAD - we control timing
                        "instructions": JUDGE_SESSION_INSTRUCTIONS,
                    },
                }
            )
        )

        # Start audio streaming if devices are configured
        if self.device_indices:
            threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "judge_openai_session_created", session_id=session_info.get("id")
            )
            return

        if etype == "session.updated":
            self.logger.log("judge_openai_session_updated")
            return

        if etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
            return

        if etype == "response.done":
            self._handle_judgment_response()
            return

        if etype == "error":
            err = event.get("error", {}).get("message", "Unknown error")
            log_print(
                "ERROR",
                f"ContextJudgeClient OpenAI error: {err}",
                session_id=self.session_id,
            )
            self.logger.log("judge_openai_error", error=err)
            return

    def _handle_judgment_response(self) -> None:
        """Parse LLM judgment response and decide whether to play TTS."""
        response = self.response_buffer.strip()
        self.response_buffer = ""

        log_print(
            "INFO",
            f"Judgment response: {response}",
            session_id=self.session_id,
        )
        self.logger.log(
            "judge_response",
            response=response,
            pending_summary=self.pending_summary,
            segment_id=self.pending_segment_id,
        )

        # Parse response: "YES: reason" or "NO: reason"
        response_upper = response.upper()
        approved = response_upper.startswith("YES")

        # Extract reason (after colon)
        reason = ""
        if ":" in response:
            reason = response.split(":", 1)[1].strip()

        if approved:
            log_print(
                "INFO",
                f"Judgment APPROVED: {reason}",
                session_id=self.session_id,
            )
            self.logger.log(
                "judge_approved",
                reason=reason,
                segment_id=self.pending_segment_id,
            )

            # Play queued TTS
            if self.tts_client and self.tts_client.has_queued_audio():
                self.tts_client.play_queued(reason=f"judge_approved: {reason}")

            self.sio.emit(
                "judge_approved",
                {
                    "reason": reason,
                    "summary": self.pending_summary,
                    "segment_id": self.pending_segment_id,
                },
            )
        else:
            log_print(
                "INFO",
                f"Judgment REJECTED: {reason}",
                session_id=self.session_id,
            )
            self.logger.log(
                "judge_rejected",
                reason=reason,
                segment_id=self.pending_segment_id,
            )

            # Clear queued TTS since we're not playing it
            if self.tts_client:
                self.tts_client.clear_queue()

            self.sio.emit(
                "judge_rejected",
                {
                    "reason": reason,
                    "summary": self.pending_summary,
                    "segment_id": self.pending_segment_id,
                },
            )

        # Clear pending state
        self.pending_summary = None
        self.pending_segment_id = 0

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket error."""
        log_print(
            "ERROR",
            f"ContextJudgeClient websocket error: {error}",
            session_id=self.session_id,
        )
        self.logger.log("judge_ws_error", error=str(error))

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        """Handle WebSocket connection closed."""
        self.connected = False
        self.running = False
        log_print("INFO", "ContextJudgeClient closed", session_id=self.session_id)
        self.logger.log("judge_ws_closed", status=status, message=msg)

        with self._lock:
            self.ws = None

    # -------------------------
    # Audio streaming
    # -------------------------

    def _stream_audio(self) -> None:
        """Stream audio from configured microphone."""
        if not self.device_indices:
            log_print(
                "WARN",
                "ContextJudgeClient: no device configured",
                session_id=self.session_id,
            )
            return

        if not self.running:
            log_print(
                "WARN",
                "ContextJudgeClient: not running, skipping audio stream",
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
                f"ContextJudgeClient opened mic {device_idx}: {device_name}",
                session_id=self.session_id,
            )
            self.logger.log("judge_audio_start", device=device_name)

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
                        f"ContextJudgeClient error reading device {device_idx}: {e}",
                        session_id=self.session_id,
                    )
                    continue

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
                f"ContextJudgeClient failed to open device {device_idx}: {e}",
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
                "ContextJudgeClient audio streaming stopped",
                session_id=self.session_id,
            )

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start the context judge client."""
        self._shutdown_event.clear()
        self.running = True
        self.logger.log("context_judge_start")

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
        """Stop the context judge client."""
        self.running = False
        self.connected = False
        self.listening = False
        self._shutdown_event.set()

        self.logger.log("context_judge_stop")

        # Wait briefly for audio thread to exit cleanly
        time.sleep(0.15)

        with self._lock:
            ws = self.ws
            self.ws = None

        if ws:
            try:
                ws.close()
            except Exception:
                pass
