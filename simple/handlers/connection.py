"""Connection-related SocketIO event handlers."""

from flask import request
from flask_socketio import emit

from handlers.context_manager import stop_context_manager
from src.core.session import TranscriptionSession


def register_connection_handlers(socketio, transcription_sessions):
    """Register connection-related event handlers."""

    @socketio.on("connect")
    def handle_connect():
        session_id = request.sid
        transcription_sessions[session_id] = TranscriptionSession(session_id)
        emit("connected", {"status": "Connected to MorseQuery Simple"})
        print(f"[Session] New session connected: {session_id}")

    @socketio.on("disconnect")
    def handle_disconnect():
        session_id = request.sid
        # Stop context manager
        stop_context_manager(session_id)

        if session_id in transcription_sessions:
            session = transcription_sessions[session_id]
            session._log_event(
                "session_end",
                {
                    "total_words": len(session.words),
                    "total_searches": sum(
                        1
                        for e in session.event_log
                        if e["event_type"] == "search_action"
                    ),
                },
            )
            log_file = session.save_logs_to_file()
            print(
                f"[Session] Session disconnected: {session_id}, logs saved to: {log_file}"
            )
            del transcription_sessions[session_id]

    @socketio.on("clear_session")
    def handle_clear_session():
        """Clear transcription session."""
        session_id = request.sid
        # Stop context manager
        stop_context_manager(session_id)

        if session_id in transcription_sessions:
            old_session = transcription_sessions[session_id]
            old_session._log_event(
                "session_cleared",
                {
                    "total_words": len(old_session.words),
                    "total_searches": sum(
                        1
                        for e in old_session.event_log
                        if e["event_type"] == "search_action"
                    ),
                },
            )
            log_file = old_session.save_logs_to_file()
            print(f"[Session] Session cleared: {session_id}, logs saved to: {log_file}")

        transcription_sessions[session_id] = TranscriptionSession(session_id)
        emit("session_cleared", {"status": "Session cleared"})

    @socketio.on("set_config")
    def handle_set_config(data):
        """Set prompt configuration."""
        session_id = request.sid
        config_id = data.get("config_id", 1)

        if session_id in transcription_sessions:
            session = transcription_sessions[session_id]
            session.set_config(config_id)
            config = session.get_current_config()
            emit(
                "config_status",
                {
                    "config_id": config_id,
                    "name": config.get("name", ""),
                    "description": config.get("description", ""),
                },
            )
            print(f"[Config] Session {session_id} set to config {config_id}: {config.get('name')}")
