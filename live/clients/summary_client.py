"""OpenAI Realtime API client for conversation summarization."""

import json
import threading
from typing import Optional

import websocket
from config import OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print


class SummaryClient:
    """Client for OpenAI Realtime API for summarizing conversations."""

    def __init__(self, socketio: SocketIO, session_id: str = "default"):
        self.sio = socketio
        self.session_id = session_id
        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.response_buffer = ""
        self.last_context = ""
        self.logger = get_logger(session_id)

        self.connected = False
        self._lock = threading.Lock()

        log_print("INFO", "SummaryClient created", session_id=session_id)
        self.logger.log("summary_client_created")

    def set_context(self, context: str) -> None:
        """Update context from RealtimeClient."""
        self.last_context = context
        log_print("DEBUG", f"Context updated: {context}", session_id=self.session_id)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opened."""
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
                        "instructions": "You are a silent observer. You MUST NOT engage in conversation, ask questions, or respond to the speaker. Your only job is to output a brief summary when requested. Never ask follow-up questions. Never say 'Got it' or acknowledge the speaker.",
                    },
                }
            )
        )

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "summary_openai_session_created", session_id=session_info.get("id")
            )
        elif etype == "session.updated":
            self.logger.log("summary_openai_session_updated")
        elif etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
                self.sio.emit("summary_chunk", delta)
        elif etype == "response.done":
            log_print("INFO", "SummaryClient response done", session_id=self.session_id)
            self.logger.log("summary_response", summary=self.response_buffer)
            self.response_buffer = ""
            self.sio.emit("summary_done")

            self.stop()

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket error."""
        log_print("ERROR", f"SummaryClient error: {error}", session_id=self.session_id)
        self.logger.log("summary_error", error=str(error))

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        """Handle WebSocket connection closed."""
        self.connected = False
        log_print("INFO", "SummaryClient closed", session_id=self.session_id)
        self.logger.log("summary_ws_closed", status=status, message=msg)
        self.running = False

        with self._lock:
            self.ws = None

    def send_audio(self, audio_b64: str) -> None:
        """Forward audio chunk to this client."""
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
            # 이미 닫혔으면 조용히 무시(레이스 상황)
            self.connected = False
            self.running = False
            with self._lock:
                self.ws = None

    def request_summary(self) -> None:
        """Request a summary of what was heard."""
        if not self.ws or not self.running:
            return

        log_print(
            "INFO",
            "Requesting summary",
            session_id=self.session_id,
            last_context=self.last_context,
        )
        self.logger.log("summary_request", last_context=self.last_context)
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        if self.last_context:
            prompt = f"""Previous context: "{self.last_context}"
Compare what you heard to the previous context.
- If nothing new: output exactly "..."
- If new topic: output 1 sentence (≤7 words) describing what's new
RULES: Output ONLY the result. Do NOT ask questions. Do NOT engage. Do NOT say "Got it"."""
        else:
            prompt = "Summarize what you heard in 1 sentence (≤7 words). Output ONLY the summary. Do NOT ask questions. Do NOT engage in conversation."

        self.ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"modalities": ["text"], "instructions": prompt},
                }
            )
        )

    def start(self) -> None:
        """Start the summary client."""
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
        """Stop the summary client."""
        self.running = False
        self.connected = False
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
