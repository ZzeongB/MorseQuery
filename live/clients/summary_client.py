"""OpenAI Realtime API client for missed-segment recovery."""

import json
import threading
from typing import Optional

import websocket
from config import OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print

from .prompt import (
    SUMMARY_SESSION_INSTRUCTIONS,
    build_summary_prompt,
)

class SummaryClient:
    """Realtime client that captures ONLY a user-marked missed segment (start/end),
    then immediately returns a recovery package at end().
    """

    def __init__(self, socketio: SocketIO, session_id: str = "default"):
        self.sio = socketio
        self.session_id = session_id

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Global conversation context (updated externally; used to anchor recovery)
        self.global_context = ""

        # Segment capture state
        self.capturing = False
        self.segment_id = 0

        self._lock = threading.Lock()

        log_print("INFO", "SummaryClient created", session_id=session_id)
        self.logger.log("summary_client_created")

    # -------------------------
    # Public APIs
    # -------------------------

    def set_global_context(self, context: str) -> None:
        """Update rolling global context (from your main pipeline)."""
        self.global_context = context or ""
        log_print("DEBUG", "Global context updated", session_id=self.session_id)
        self.logger.log("global_context_updated", chars=len(self.global_context))

    def start_miss(self) -> None:
        """Begin capturing a missed segment (explicit start)."""
        if not self.running or not self.connected:
            log_print(
                "WARN", "start_miss ignored (not connected)", session_id=self.session_id
            )
            return

        with self._lock:
            ws = self.ws

        self.segment_id += 1
        self.capturing = True
        self.response_buffer = ""

        # Best-effort clear server buffer so segment is isolated.
        if ws:
            try:
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass

        self.logger.log("miss_segment_start", segment_id=self.segment_id)
        self.sio.emit("miss_segment_start", {"segment_id": self.segment_id})

    def end_miss_and_recover(self) -> None:
        """End capturing and immediately request summary for the captured segment."""
        if not self.running or not self.connected:
            log_print(
                "WARN", "end_miss ignored (not connected)", session_id=self.session_id
            )
            return

        with self._lock:
            ws = self.ws

        if not ws:
            log_print(
                "WARN", "end_miss ignored (no websocket)", session_id=self.session_id
            )
            return

        self.capturing = False
        self.response_buffer = ""

        self.logger.log("miss_segment_end", segment_id=self.segment_id)
        self.sio.emit("miss_segment_end", {"segment_id": self.segment_id})

        # Commit the segment audio and request recovery immediately.
        try:
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            self.logger.log("commit_failed", error=str(e))
            return

        prompt = build_summary_prompt(self.global_context)

        self.logger.log(
            "recovery_request",
            segment_id=self.segment_id,
            context_chars=len(self.global_context),
        )

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
                self.sio.emit("recovery_chunk", delta)
            return

        if etype == "response.done":
            raw = self.response_buffer
            self.response_buffer = ""

            self.logger.log("summary_response_done", raw=raw)

            is_empty = raw.strip() == "..."
            summary = raw.strip() if not is_empty else ""

            payload = {
                "segment_id": self.segment_id,
                "is_empty": is_empty,
                "summary": summary,
            }

            self.sio.emit("recovery_done", payload)

            # Update global context
            if not is_empty and summary:
                self.global_context = (self.global_context + " " + summary).strip()
                if len(self.global_context) > 500:
                    self.global_context = self.global_context[-500:]
                self.logger.log(
                    "global_context_appended", segment_id=self.segment_id
                )

            # Prepare for next segment: best-effort clear buffer.
            with self._lock:
                ws = self.ws
            if ws:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                except Exception:
                    pass

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
    # Audio forwarding (segment-gated)
    # -------------------------

    def send_audio(self, audio_b64: str) -> None:
        """Forward audio only during capture."""
        if not self.running or not self.connected or not self.capturing:
            return

        with self._lock:
            ws = self.ws

        if not ws:
            return

        try:
            ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
            )
        except websocket._exceptions.WebSocketConnectionClosedException:
            self.connected = False
            self.running = False
            with self._lock:
                self.ws = None

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start and keep the session alive; use start_miss/end_miss_and_recover for segments."""
        self.running = True
        self.logger.log("summary_client_start")

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
        """Stop the client and close the websocket."""
        self.running = False
        self.connected = False
        self.capturing = False

        self.logger.log("summary_client_stop")

        with self._lock:
            ws = self.ws
            self.ws = None

        if ws:
            try:
                ws.close()
            except Exception:
                pass

        self.sio.emit("summary_closed")
