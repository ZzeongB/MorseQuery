"""OpenAI Realtime API client for missed-conversation reconstruction."""

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

from .prompt import RECONSTRUCTOR_SESSION_INSTRUCTIONS, build_reconstruction_prompt


class ConversationReconstructorClient:
    """Realtime client that reconstructs missed conversation snippets."""

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        device_indices: list[int] | None = None,
        on_reconstruction_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.device_indices = device_indices or []

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

        # Pending request context
        self.pending_segment_id: int = 0
        self.pending_context_before: str = ""
        self.pending_sum0: str = ""
        self.pending_sum1: str = ""
        self.pending_next_sentence: str = ""
        self.pending_prompt: str = ""
        self.on_reconstruction_callback = on_reconstruction_callback
        self._queued_request: Optional[dict] = None

        # Synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()

        log_print(
            "INFO",
            "ConversationReconstructorClient created",
            session_id=session_id,
            devices=device_indices,
        )
        self.logger.log(
            "conversation_reconstructor_created",
            devices=device_indices,
        )

    def start_listening(self) -> None:
        """Mark the start of a listening segment."""
        if not self.running or not self.connected:
            return

        self.segment_id += 1
        self.listening = True
        self.response_buffer = ""

        with self._lock:
            ws = self.ws
            if ws:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
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

        self.logger.log("reconstructor_start_listening", segment_id=self.segment_id)

    def end_listening(self) -> None:
        """Mark the end of a listening segment."""
        if not self.running or not self.connected:
            return

        self.listening = False
        self.logger.log("reconstructor_end_listening", segment_id=self.segment_id)

    def reconstruct_conversation(
        self,
        context_before: str,
        sum0: str,
        sum1: str,
        next_sentence: str,
        segment_id: int,
    ) -> None:
        """Request reconstructed dialogue from LLM."""
        if not self.running:
            log_print(
                "WARN",
                "reconstruct_conversation ignored (not running)",
                session_id=self.session_id,
            )
            return

        if not (sum0 or sum1):
            log_print(
                "INFO",
                "reconstruct_conversation ignored (empty summaries)",
                session_id=self.session_id,
                segment_id=segment_id,
            )
            return

        self.pending_segment_id = segment_id
        self.pending_context_before = context_before or ""
        self.pending_sum0 = sum0 or ""
        self.pending_sum1 = sum1 or ""
        self.pending_next_sentence = next_sentence or ""
        self.response_buffer = ""

        prompt = build_reconstruction_prompt(
            context_before=self.pending_context_before,
            sum0=self.pending_sum0,
            sum1=self.pending_sum1,
            next_sentence=self.pending_next_sentence,
        )
        self.pending_prompt = prompt

        self.logger.log(
            "reconstruction_request",
            segment_id=segment_id,
            context_before=self.pending_context_before,
            sum0=self.pending_sum0,
            sum1=self.pending_sum1,
            next_sentence=self.pending_next_sentence,
            prompt=prompt,
            has_sum0=bool(sum0),
            has_sum1=bool(sum1),
            has_next_sentence=bool(next_sentence),
        )
        log_print(
            "INFO",
            "Reconstruction request",
            session_id=self.session_id,
            segment_id=segment_id,
            has_sum0=bool(sum0),
            has_sum1=bool(sum1),
            has_next_sentence=bool(next_sentence),
        )

        # If websocket is not ready yet, queue and send on connect.
        if not self.connected:
            self._queued_request = {
                "prompt": prompt,
                "segment_id": segment_id,
            }
            self.logger.log(
                "reconstruction_request_queued_not_connected",
                segment_id=segment_id,
            )
            return

        self._send_reconstruction_request(prompt, segment_id)

    def _send_reconstruction_request(self, prompt: str, segment_id: int) -> None:
        with self._lock:
            ws = self.ws
            if not ws:
                self._queued_request = {"prompt": prompt, "segment_id": segment_id}
                self.logger.log(
                    "reconstruction_request_queued_no_ws",
                    segment_id=segment_id,
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
                                "temperature": 1,
                                "max_output_tokens": 200,
                            },
                        }
                    )
                )
            except Exception as e:
                self.logger.log("reconstruction_request_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"reconstruct_conversation send failed: {e}",
                    session_id=self.session_id,
                )

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        self.logger.log("reconstructor_ws_connected")
        log_print(
            "INFO",
            "ConversationReconstructorClient WebSocket connected",
            session_id=self.session_id,
        )

        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": RECONSTRUCTOR_SESSION_INSTRUCTIONS,
                    },
                }
            )
        )

        queued = self._queued_request
        if queued:
            self._queued_request = None
            self.logger.log(
                "reconstruction_request_flushed_after_connect",
                segment_id=queued.get("segment_id"),
            )
            self._send_reconstruction_request(
                str(queued.get("prompt", "")),
                int(queued.get("segment_id", 0) or 0),
            )

        if self.device_indices:
            threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "reconstructor_openai_session_created",
                session_id=session_info.get("id"),
            )
            return

        if etype == "session.updated":
            self.logger.log("reconstructor_openai_session_updated")
            return

        if etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
            return

        if etype == "response.done":
            self._handle_reconstruction_response()
            return

        if etype == "error":
            err = event.get("error", {}).get("message", "Unknown error")
            self.logger.log("reconstructor_openai_error", error=err)
            log_print(
                "ERROR",
                f"ConversationReconstructorClient OpenAI error: {err}",
                session_id=self.session_id,
            )

    def _handle_reconstruction_response(self) -> None:
        raw = self.response_buffer.strip()
        self.response_buffer = ""
        reconstruction = self._normalize_output(raw)

        self.logger.log(
            "reconstruction_response",
            segment_id=self.pending_segment_id,
            prompt=self.pending_prompt,
            context_before=self.pending_context_before,
            sum0=self.pending_sum0,
            sum1=self.pending_sum1,
            next_sentence=self.pending_next_sentence,
            output_raw=raw,
            normalized=reconstruction,
        )
        log_print(
            "INFO",
            "Reconstruction response received",
            session_id=self.session_id,
            segment_id=self.pending_segment_id,
            chars=len(reconstruction),
        )

        self.sio.emit(
            "conversation_reconstructed",
            {
                "segment_id": self.pending_segment_id,
                "conversation": reconstruction,
                "sum0": self.pending_sum0,
                "sum1": self.pending_sum1,
                "next_sentence": self.pending_next_sentence,
            },
        )
        if self.on_reconstruction_callback:
            try:
                self.on_reconstruction_callback(
                    {
                        "segment_id": self.pending_segment_id,
                        "conversation": reconstruction,
                        "sum0": self.pending_sum0,
                        "sum1": self.pending_sum1,
                        "next_sentence": self.pending_next_sentence,
                    }
                )
            except Exception as e:
                self.logger.log("reconstruction_callback_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"reconstruction callback failed: {e}",
                    session_id=self.session_id,
                )

        self.pending_segment_id = 0
        self.pending_context_before = ""
        self.pending_sum0 = ""
        self.pending_sum1 = ""
        self.pending_next_sentence = ""
        self.pending_prompt = ""

    def _normalize_output(self, text: str) -> str:
        """Keep only A:/B: dialogue lines (max 3 lines)."""
        lines = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if line.startswith("A:") or line.startswith("B:"):
                lines.append(line)
            if len(lines) >= 3:
                break

        if lines:
            return "\n".join(lines)
        return text[:400]

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        self.logger.log("reconstructor_ws_error", error=str(error))
        log_print(
            "ERROR",
            f"ConversationReconstructorClient websocket error: {error}",
            session_id=self.session_id,
        )

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        self.connected = False
        self.running = False
        self.logger.log("reconstructor_ws_closed", status=status, message=msg)
        log_print(
            "INFO",
            "ConversationReconstructorClient closed",
            session_id=self.session_id,
        )

        with self._lock:
            self.ws = None

    def _stream_audio(self) -> None:
        """Stream audio from configured microphone."""
        if not self.device_indices:
            return
        if not self.running:
            return

        device_idx = self.device_indices[0]
        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=AUDIO_CHUNK,
            )
            self.pa = pa
            self.stream = stream

            commit_interval = 3.0
            last_commit = 0.0

            while self.running and self.connected and not self._shutdown_event.is_set():
                try:
                    data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                except Exception:
                    if self._shutdown_event.is_set():
                        break
                    continue

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

    def start(self) -> None:
        """Start the reconstructor client."""
        self._shutdown_event.clear()
        self.running = True
        self.logger.log("conversation_reconstructor_start")

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
        """Stop the reconstructor client."""
        self.running = False
        self.connected = False
        self.listening = False
        self._shutdown_event.set()
        self.logger.log("conversation_reconstructor_stop")

        time.sleep(0.15)
        with self._lock:
            ws = self.ws
            self.ws = None
        if ws:
            try:
                ws.close()
            except Exception:
                pass
