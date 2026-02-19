"""Logging utilities for the realtime application."""

import json
from datetime import datetime
from typing import Any

from config import LOG_DIR


def get_timestamp() -> str:
    """Return current timestamp as formatted string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_print(level: str, message: str, **kwargs: Any) -> None:
    """Print formatted log message to console."""
    timestamp = get_timestamp()
    extra = f" | {kwargs}" if kwargs else ""
    print(f"[{timestamp}] [{level.upper():5}] {message}{extra}")


class JsonLogger:
    """JSON file logger for session events."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.log_file = (
            LOG_DIR
            / f"logs/{self.start_time.strftime('%Y%m%d_%H%M%S')}_{session_id}.json"
        )
        self.events: list[dict] = []
        log_print(
            "INFO",
            "JsonLogger initialized",
            session_id=session_id,
            log_file=str(self.log_file),
        )

    def log(self, event_type: str, **data: Any) -> None:
        """Log an event with timestamp and data."""
        event = {
            "timestamp": get_timestamp(),
            "event_type": event_type,
            "session_id": self.session_id,
            **data,
        }
        self.events.append(event)
        self._save()

    def _save(self) -> None:
        """Save events to JSON file."""
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "start_time": self.start_time.isoformat(),
                    "events": self.events,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )


# Session loggers storage
_session_loggers: dict[str, JsonLogger] = {}


def get_logger(session_id: str) -> JsonLogger:
    """Get or create a logger for a session."""
    if session_id not in _session_loggers:
        _session_loggers[session_id] = JsonLogger(session_id)
    return _session_loggers[session_id]
