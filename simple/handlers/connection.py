"""Connection-related SocketIO event handlers."""

from flask import request
from flask_socketio import emit

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
