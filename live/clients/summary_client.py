"""OpenAI Realtime API client for missed-segment recovery (summary/keywords/rejoin)."""

import json
import re
import threading
import time
from typing import Any, Dict, List, Optional

import websocket
from config import OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print

from .prompt import (
    SUMMARY_SESSION_INSTRUCTIONS,
    build_recovery_prompt,
    build_summary_prompt,
    build_keywords_prompt,
    build_transcript_prompt,
    build_context_prompt,
)

# Words that indicate topic change (case-insensitive, at sentence start)
TOPIC_CHANGE_WORDS = {
    "now", "so", "anyway", "anyway,", "actually", "but", "however",
    "moving on", "next", "alright", "okay", "ok", "well",
    "let's", "let me", "speaking of", "by the way", "btw",
}


def _parse_recovery(text: str) -> Dict[str, Any]:
    """Parse STRICT FORMAT output. Returns dict with keys:
    - is_empty (bool): True if output is "..."
    - summary (str)
    - keywords (list[str])
    - recovery (str)
    - raw (str)
    """
    raw = (text or "").strip()
    if raw == "...":
        return {
            "is_empty": True,
            "summary": "",
            "keywords": [],
            "recovery": "",
            "raw": raw,
        }

    summary = ""
    keywords: List[str] = []
    recovery = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("SUMMARY::"):
            summary = line[len("SUMMARY::") :].strip()
        elif line.startswith("KEYWORDS::"):
            kw = line[len("KEYWORDS::") :].strip()
            # Split by comma, strip whitespace, drop empties
            keywords = [k.strip() for k in kw.split(",") if k.strip()]
        elif line.startswith("RECOVERY::"):
            recovery = line[len("RECOVERY::") :].strip()

    return {
        "is_empty": False,
        "summary": summary,
        "keywords": keywords,
        "recovery": recovery,
        "raw": raw,
    }


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
        self._pending_kind: Optional[str] = None  # "recovery", "context", etc.

        # Continuous context mode - always listen and periodically emit context
        self.continuous_mode = True
        self.context_interval = 15.0  # seconds between context updates
        self._context_timer: Optional[threading.Timer] = None
        self._audio_received = False  # Track if any audio received since last context
        self._last_context_time: float = 0.0  # Last context update time
        self._min_context_interval: float = 5.0  # Minimum seconds between context updates

        # Topic change detection
        self._recent_transcript: str = ""  # Recent transcription text

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

    def receive_transcription(self, text: str) -> None:
        """Receive transcription from realtime_client and check for topic changes."""
        if not text or not self.continuous_mode:
            return

        # Append to recent transcript buffer
        self._recent_transcript += " " + text.strip()
        # Keep buffer manageable (last ~200 chars)
        if len(self._recent_transcript) > 300:
            self._recent_transcript = self._recent_transcript[-200:]

        # Check for topic change indicators
        if self._detect_topic_change(text):
            self._trigger_context_update()

    def _detect_topic_change(self, text: str) -> bool:
        """Check if text starts with a topic change indicator."""
        text_lower = text.strip().lower()

        # Check if starts with any topic change word
        for word in TOPIC_CHANGE_WORDS:
            if text_lower.startswith(word + " ") or text_lower.startswith(word + ","):
                log_print(
                    "DEBUG",
                    f"Topic change detected: '{word}'",
                    session_id=self.session_id,
                )
                return True
        return False

    def _trigger_context_update(self) -> None:
        """Trigger an immediate context update if enough time has passed."""
        now = time.time()
        if now - self._last_context_time < self._min_context_interval:
            return  # Too soon, skip

        # Cancel scheduled timer and request immediately
        if self._context_timer:
            self._context_timer.cancel()

        self._request_context()

    def _schedule_context_update(self) -> None:
        """Schedule the next context update."""
        if not self.running or not self.continuous_mode:
            return

        if self._context_timer:
            self._context_timer.cancel()

        self._context_timer = threading.Timer(
            self.context_interval, self._request_context
        )
        self._context_timer.daemon = True
        self._context_timer.start()

    def _request_context(self) -> None:
        """Request a context update from accumulated audio."""
        if not self.running or not self.connected:
            return

        # Only request if we've received audio since last context
        if not self._audio_received:
            self._schedule_context_update()
            return

        with self._lock:
            ws = self.ws

        if not ws:
            self._schedule_context_update()
            return

        self._audio_received = False
        self._last_context_time = time.time()
        self._pending_kind = "context"

        # Commit accumulated audio
        try:
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            self.logger.log("context_commit_failed", error=str(e))
            self._schedule_context_update()
            return

        # Request context
        prompt = build_context_prompt(self.global_context)
        self.logger.log("context_request", context_chars=len(self.global_context))

        try:
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
            self.logger.log("context_request_failed", error=str(e))
            self._schedule_context_update()

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
        self._pending_kind = None

        # Best-effort clear server buffer so segment is isolated.
        if ws:
            try:
                ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
            except Exception:
                pass

        self.logger.log("miss_segment_start", segment_id=self.segment_id)
        self.sio.emit("miss_segment_start", {"segment_id": self.segment_id})

    def end_miss_and_recover(self, mode: str = "summary") -> None:
        """End capturing and immediately request recovery for the captured segment.

        Args:
            mode: One of "summary", "keywords", or "transcript"
        """
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
        self._pending_kind = mode  # Store the mode for response handling

        self.logger.log("miss_segment_end", segment_id=self.segment_id, mode=mode)
        self.sio.emit("miss_segment_end", {"segment_id": self.segment_id, "mode": mode})

        # Commit the segment audio and request recovery immediately.
        try:
            ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        except Exception as e:
            self.logger.log("commit_failed", error=str(e))
            return

        # Select prompt based on mode
        if mode == "keywords":
            prompt = build_keywords_prompt(self.global_context)
        elif mode == "transcript":
            prompt = build_transcript_prompt(self.global_context)
        else:  # Default to summary
            prompt = build_summary_prompt(self.global_context)

        self.logger.log(
            "recovery_request",
            segment_id=self.segment_id,
            mode=mode,
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

        # Start continuous context updates if enabled
        if self.continuous_mode:
            self._schedule_context_update()

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
                # Optional: stream to UI
                self.sio.emit("recovery_chunk", delta)
            return

        if etype == "response.done":
            raw = self.response_buffer
            self.response_buffer = ""

            mode = self._pending_kind or "summary"
            self._pending_kind = None

            self.logger.log("summary_response_done", mode=mode, raw=raw)

            # Check if empty response
            is_empty = raw.strip() == "..."

            # Handle context mode separately
            if mode == "context":
                if not is_empty:
                    self.sio.emit("context", {"text": raw.strip()})
                    # Update global context with new info
                    self.global_context = (self.global_context + " " + raw.strip()).strip()
                    # Keep context from growing too large (last 500 chars)
                    if len(self.global_context) > 500:
                        self.global_context = self.global_context[-500:]
                self._schedule_context_update()
                return

            if mode == "keywords":
                # Parse comma-separated keywords
                keywords = []
                if not is_empty:
                    keywords = [k.strip() for k in raw.strip().split(",") if k.strip()]
                payload = {
                    "segment_id": self.segment_id,
                    "mode": mode,
                    "is_empty": is_empty,
                    "keywords": keywords,
                    "raw": raw.strip(),
                }
            elif mode == "transcript":
                # Raw transcript text
                payload = {
                    "segment_id": self.segment_id,
                    "mode": mode,
                    "is_empty": is_empty,
                    "transcript": raw.strip() if not is_empty else "",
                    "raw": raw.strip(),
                }
            else:  # summary mode
                # Summary is just the raw text
                payload = {
                    "segment_id": self.segment_id,
                    "mode": mode,
                    "is_empty": is_empty,
                    "summary": raw.strip() if not is_empty else "",
                    "raw": raw.strip(),
                }

            self.sio.emit("recovery_done", payload)

            # Update global context minimally (optional):
            if not is_empty:
                add = raw.strip()
                if add and mode == "summary":
                    self.global_context = (self.global_context + " " + add).strip()
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
        """Forward audio. In continuous mode, always sends. Otherwise, only during capture."""
        if not self.running or not self.connected:
            return

        # In continuous mode, always accept audio. Otherwise, only when capturing.
        if not self.continuous_mode and not self.capturing:
            return

        with self._lock:
            ws = self.ws

        if not ws:
            return

        try:
            ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
            )
            self._audio_received = True
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
        self._pending_kind = None

        # Cancel context timer
        if self._context_timer:
            self._context_timer.cancel()
            self._context_timer = None

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
