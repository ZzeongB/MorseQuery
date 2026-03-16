"""Logging utilities for the realtime application."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from config import LOG_DIR

SESSIONS_DIR = LOG_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
_session_root_dirs: dict[str, Path] = {}
SESSION_LOG_FILENAME = "session_log.json"
SESSION_FULL_LOG_FILENAME = "session_full_log.json"
ALLOWED_EVENT_TYPES = {
    "keyword_request",
    "response_done",
    "input_audio_buffer_commit_sent",
    "end_listening_commit_result",
    "anc_on",
    "anc_off",
    "tts_play_start",
    "tts_play_done",
    "tts_playing_end",
    "summary_tts_playback_done",
    "summary_tts_anc_off",
    "summarizing_start",
    "api_compression_request",
    "api_compression_result",
}


def _sanitize_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "default"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text)


def get_root_session_id(session_id: str) -> str:
    """Return the shared root session id.

    Component clients append suffixes like `_sum0`, `_keyword_tts`.
    We group them under the same root directory.
    """
    text = str(session_id or "default")
    return text.split("_", 1)[0]


def get_session_dir(session_id: str) -> Path:
    """Get/create base directory for a (root) session."""
    root_id = _sanitize_name(get_root_session_id(session_id))
    existing = _session_root_dirs.get(root_id)
    if existing is not None:
        existing.mkdir(parents=True, exist_ok=True)
        return existing

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = SESSIONS_DIR / f"{ts}_{root_id}"
    path.mkdir(parents=True, exist_ok=True)
    _session_root_dirs[root_id] = path
    return path


def get_session_subdir(session_id: str, kind: str) -> Path:
    """Get/create a subdirectory under a session directory."""
    safe_kind = _sanitize_name(kind or "misc")
    path = get_session_dir(session_id) / safe_kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """Return current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_print(level: str, message: str, **kwargs: Any) -> None:
    """Print formatted log message to console."""
    timestamp = get_timestamp()
    extra = f" | {kwargs}" if kwargs else ""
    text = f"[{timestamp}] [{level.upper():5}] {message}{extra}"
    if str(message).startswith("VAD speech_started (") or str(message).startswith(
        "VAD speech_stopped ("
    ):
        yellow = "\033[33m"
        reset = "\033[0m"
        print(f"{yellow}{text}{reset}")
        return
    print(text)


class JsonLogger:
    """JSON file logger for session events."""

    def __init__(self, session_id: str):
        self.session_id = get_root_session_id(session_id)
        self.start_time = datetime.now()
        session_dir = get_session_dir(self.session_id)
        self.log_file = session_dir / SESSION_LOG_FILENAME
        self.full_log_file = session_dir / SESSION_FULL_LOG_FILENAME
        self.events: list[dict] = []
        self.full_events: list[dict] = []
        log_print(
            "INFO",
            "JsonLogger initialized",
            session_id=self.session_id,
            log_file=str(self.log_file),
            full_log_file=str(self.full_log_file),
        )

    def log(self, event_type: str, **data: Any) -> None:
        """Log an event with timestamp and data."""
        event = {
            "timestamp": get_timestamp(),
            "event_type": event_type,
            "session_id": self.session_id,
            **data,
        }
        self.full_events.append(event)
        self._save(self.full_log_file, self.full_events)

        if event_type not in ALLOWED_EVENT_TYPES:
            return
        if (
            event_type in {"api_compression_request", "api_compression_result"}
            and str(data.get("key", "")) != "api_mini"
        ):
            return
        self.events.append(event)
        self._save(self.log_file, self.events)

    def _save(self, filepath: Path, events: list[dict]) -> None:
        """Save events to a JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "start_time": self.start_time.isoformat(),
                    "events": events,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


# Session loggers storage
_session_loggers: dict[str, JsonLogger] = {}


def get_logger(session_id: str) -> JsonLogger:
    """Get or create a logger for a session."""
    root_session_id = get_root_session_id(session_id)
    if root_session_id not in _session_loggers:
        _session_loggers[root_session_id] = JsonLogger(root_session_id)
    return _session_loggers[root_session_id]


def get_existing_logger(session_id: str) -> JsonLogger | None:
    """Return an existing logger without creating a new one."""
    root_session_id = get_root_session_id(session_id)
    return _session_loggers.get(root_session_id)
