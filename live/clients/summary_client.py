"""OpenAI Realtime API client for missed-segment recovery.

Audio is continuously streamed to the Realtime API.
start_listening() / end_listening() mark segment boundaries for summarization.
"""

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
    """Realtime client that continuously listens to audio.

    start_listening() / end_listening() mark segment boundaries.
    On end_listening(), summarizes what was said in that segment.
    """

    def __init__(self, socketio: SocketIO, session_id: str = "default"):
        self.sio = socketio
        self.session_id = session_id

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Context from before the current listening segment
        self.pre_context = ""

        # Segment state
        self.listening = False
        self.segment_id = 0

        self._lock = threading.Lock()

        log_print("INFO", "SummaryClient created", session_id=session_id)
        self.logger.log("summary_client_created")

    # -------------------------
    # Public APIs
    # -------------------------

    def start_listening(self) -> None:
        """Mark the start of a segment to summarize later.

        Audio continues streaming. This just marks a boundary.
        """
        if not self.running or not self.connected:
            log_print(
                "WARN", "start_listening ignored (not connected)", session_id=self.session_id
            )
            return

        with self._lock:
            ws = self.ws

        self.segment_id += 1
        self.listening = True
        self.response_buffer = ""

        # Commit current audio as "pre-context", then clear for new segment
        if ws:
            try:
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass

        self.logger.log("start_listening", segment_id=self.segment_id)
        self.sio.emit("listening_start", {"segment_id": self.segment_id})

    def end_listening(self) -> None:
        """End the segment and request a summary of what was said."""
        if not self.running or not self.connected:
            log_print(
                "WARN", "end_listening ignored (not connected)", session_id=self.session_id
            )
            return

        if not self.listening:
            log_print(
                "WARN", "end_listening ignored (not listening)", session_id=self.session_id
            )
            return

        with self._lock:
            ws = self.ws

        if not ws:
            log_print(
                "WARN", "end_listening ignored (no websocket)", session_id=self.session_id
            )
            return

        self.listening = False
        self.response_buffer = ""

        self.logger.log("end_listening", segment_id=self.segment_id)
        self.sio.emit("listening_end", {"segment_id": self.segment_id})

        # Commit the segment audio and request summary
        try:
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            self.logger.log("commit_failed", error=str(e))
            return

        prompt = build_summary_prompt(self.pre_context)

        self.logger.log(
            "summary_request",
            segment_id=self.segment_id,
            pre_context_chars=len(self.pre_context),
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
                self.sio.emit("summary_chunk", delta)
            return

        if etype == "response.done":
            raw = self.response_buffer
            self.response_buffer = ""

            self.logger.log("summary_response_done", raw=raw)

            is_empty = raw.strip() == "..."

            if is_empty:
                payload = {
                    "segment_id": self.segment_id,
                    "is_empty": True,
                    "delta": "",
                    "topic": "",
                    "exchange": "",
                    "script": "",
                }
            else:
                # Parse structured output
                delta = ""
                topic = ""
                exchange = ""
                script = ""
                for line in raw.strip().splitlines():
                    line = line.strip()
                    if line.startswith("DELTA::"):
                        delta = line[7:].strip()
                    elif line.startswith("TOPIC::"):
                        topic = line[7:].strip()
                    elif line.startswith("EXCHANGE::"):
                        exchange = line[10:].strip()
                    elif line.startswith("SCRIPT::"):
                        script = line[8:].strip()

                payload = {
                    "segment_id": self.segment_id,
                    "is_empty": False,
                    "delta": delta,
                    "topic": topic,
                    "exchange": exchange,
                    "script": script,
                }

                # Update pre_context with topic for next segment
                if topic:
                    self.pre_context = (self.pre_context + " " + topic).strip()
                    if len(self.pre_context) > 500:
                        self.pre_context = self.pre_context[-500:]
                    self.logger.log(
                        "pre_context_updated", segment_id=self.segment_id
                    )

            self.sio.emit("summary_done", payload)
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
        """Forward audio continuously (always streaming)."""
        if not self.running or not self.connected:
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
        """Start the client; use start_listening/end_listening for segment summaries."""
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
        self.listening = False

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
