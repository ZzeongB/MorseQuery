"""OpenAI Realtime API client for transcript compression (text-only)."""

import json
import threading
import time
from typing import Callable, Optional

import websocket
from config import OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print

from .prompt import (
    TRANSCRIPT_RECONSTRUCTOR_INSTRUCTIONS,
    build_transcript_reconstruction_prompt,
)


class TranscriptReconstructorClient:
    """Realtime client that compresses dialogue transcripts into brief catch-up summaries.

    This client is text-only: it never streams audio to OpenAI.
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        source: str = "mic",
        device_indices: list[int] | None = None,
        audio_file: str | None = None,
        on_reconstruction_callback: Optional[Callable[[dict], None]] = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.source = source
        # Force text-only mode: do not keep any audio source bindings.
        self.device_indices: list[int] = []
        self.audio_file: str | None = None

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Listening state (synchronized with SummaryClient)
        self.listening = False
        self.segment_id = 0

        # Pending request context
        self.pending_segment_id: int = 0
        self.pending_dialogue: str = ""
        self.pending_before_context: str = ""
        self.pending_prompt: str = ""
        self.on_reconstruction_callback = on_reconstruction_callback
        self._queued_request: Optional[dict] = None

        # Timing
        self._request_start_time: float = 0.0

        # Synchronization
        self._lock = threading.Lock()

        log_print(
            "INFO",
            "TranscriptReconstructorClient created",
            session_id=session_id,
            source=source,
            devices=device_indices,
            audio_file=audio_file,
        )
        self.logger.log(
            "transcript_reconstructor_created",
            source=source,
            devices=device_indices,
            audio_file=audio_file,
        )

    def start_listening(self) -> None:
        """Mark the start of a listening segment."""
        if not self.running or not self.connected:
            return

        self.segment_id += 1
        self.listening = True
        self.response_buffer = ""

        self.logger.log("transcript_reconstructor_start_listening", segment_id=self.segment_id)

    def end_listening(self) -> None:
        """Mark the end of a listening segment."""
        if not self.running or not self.connected:
            return

        self.listening = False
        self.logger.log("transcript_reconstructor_end_listening", segment_id=self.segment_id)

    def reconstruct_transcript(
        self,
        dialogue: str,
        segment_id: int,
        before_context: str = "",
    ) -> None:
        """Request compressed transcript from LLM.

        Args:
            dialogue: The formatted dialogue string (A: ...\nB: ...)
            segment_id: The segment identifier
            before_context: Short transcript context before the missed segment
        """
        if not self.running:
            log_print(
                "WARN",
                "reconstruct_transcript ignored (not running)",
                session_id=self.session_id,
            )
            return

        if not dialogue or not dialogue.strip():
            log_print(
                "INFO",
                "reconstruct_transcript ignored (empty dialogue)",
                session_id=self.session_id,
                segment_id=segment_id,
            )
            return

        self.pending_segment_id = segment_id
        self.pending_dialogue = dialogue.strip()
        self.pending_before_context = (before_context or "").strip()
        self.response_buffer = ""
        self._request_start_time = time.time()

        prompt = build_transcript_reconstruction_prompt(
            self.pending_dialogue,
            before_context=self.pending_before_context,
        )
        self.pending_prompt = prompt

        self.logger.log(
            "transcript_reconstruction_request",
            segment_id=segment_id,
            dialogue=self.pending_dialogue[:200],
            before_context=self.pending_before_context[:200],
            prompt=prompt[:200],
        )
        log_print(
            "INFO",
            "Transcript reconstruction request",
            session_id=self.session_id,
            segment_id=segment_id,
            dialogue_chars=len(self.pending_dialogue),
            before_context_chars=len(self.pending_before_context),
        )

        # If websocket is not ready yet, queue and send on connect.
        if not self.connected:
            self._queued_request = {
                "prompt": prompt,
                "segment_id": segment_id,
            }
            self.logger.log(
                "transcript_reconstruction_request_queued_not_connected",
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
                    "transcript_reconstruction_request_queued_no_ws",
                    segment_id=segment_id,
                )
                return
            try:
                ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text"],
                                "instructions": prompt,
                                "temperature": 0.6,
                                "max_output_tokens": 80,
                            },
                        }
                    )
                )
            except Exception as e:
                self.logger.log("transcript_reconstruction_request_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"reconstruct_transcript send failed: {e}",
                    session_id=self.session_id,
                )

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        self.connected = True
        self.logger.log("transcript_reconstructor_ws_connected")
        log_print(
            "INFO",
            "TranscriptReconstructorClient WebSocket connected",
            session_id=self.session_id,
        )

        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text"],
                        "instructions": TRANSCRIPT_RECONSTRUCTOR_INSTRUCTIONS,
                    },
                }
            )
        )

        queued = self._queued_request
        if queued:
            self._queued_request = None
            self.logger.log(
                "transcript_reconstruction_request_flushed_after_connect",
                segment_id=queued.get("segment_id"),
            )
            self._send_reconstruction_request(
                str(queued.get("prompt", "")),
                int(queued.get("segment_id", 0) or 0),
            )

        # Intentionally no audio streaming: transcript reconstructor is text-only.

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "transcript_reconstructor_openai_session_created",
                session_id=session_info.get("id"),
            )
            return

        if etype == "session.updated":
            self.logger.log("transcript_reconstructor_openai_session_updated")
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
            self.logger.log("transcript_reconstructor_openai_error", error=err)
            log_print(
                "ERROR",
                f"TranscriptReconstructorClient OpenAI error: {err}",
                session_id=self.session_id,
            )

    def _handle_reconstruction_response(self) -> None:
        elapsed_ms = (time.time() - self._request_start_time) * 1000
        raw = self.response_buffer.strip()
        self.response_buffer = ""
        compressed = self._normalize_output(raw)

        self.logger.log(
            "transcript_reconstruction_response",
            segment_id=self.pending_segment_id,
            prompt=self.pending_prompt[:200],
            dialogue=self.pending_dialogue[:200],
            before_context=self.pending_before_context[:200],
            output_raw=raw,
            normalized=compressed,
            elapsed_ms=elapsed_ms,
        )
        log_print(
            "INFO",
            "Transcript reconstruction response received",
            session_id=self.session_id,
            segment_id=self.pending_segment_id,
            chars=len(compressed),
            elapsed_ms=int(elapsed_ms),
        )

        payload = {
            "segment_id": self.pending_segment_id,
            "compressed_text": compressed,
            "original_dialogue": self.pending_dialogue,
            "before_context": self.pending_before_context,
            "elapsed_ms": elapsed_ms,
            "method": "realtime_api",
        }

        self.sio.emit("transcript_compressed", payload)

        if self.on_reconstruction_callback:
            try:
                self.on_reconstruction_callback(payload)
            except Exception as e:
                self.logger.log("transcript_reconstruction_callback_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"transcript reconstruction callback failed: {e}",
                    session_id=self.session_id,
                )

        self.pending_segment_id = 0
        self.pending_dialogue = ""
        self.pending_before_context = ""
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
        self.logger.log("transcript_reconstructor_ws_error", error=str(error))
        log_print(
            "ERROR",
            f"TranscriptReconstructorClient websocket error: {error}",
            session_id=self.session_id,
        )

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        self.connected = False
        self.running = False
        self.logger.log("transcript_reconstructor_ws_closed", status=status, message=msg)
        log_print(
            "INFO",
            "TranscriptReconstructorClient closed",
            session_id=self.session_id,
        )

        with self._lock:
            self.ws = None

    def start(self) -> None:
        """Start the transcript reconstructor client."""
        self.running = True
        self.logger.log("transcript_reconstructor_start")

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
        """Stop the transcript reconstructor client."""
        self.running = False
        self.connected = False
        self.listening = False
        self.logger.log("transcript_reconstructor_stop")

        time.sleep(0.15)
        with self._lock:
            ws = self.ws
            self.ws = None
        if ws:
            try:
                ws.close()
            except Exception:
                pass
