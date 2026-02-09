"""Context manager for maintaining conversation context summary.

Periodically calls Gemini to summarize the latest transcription into 1-2 sentences.
This context is then included when making keyword inference requests.
"""

import os
import threading
from datetime import datetime

from flask import request
from flask_socketio import emit

# Gemini configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.0-flash-lite"

# Context manager settings
CONTEXT_UPDATE_INTERVAL = 5.0  # Seconds between context updates
CONTEXT_MIN_NEW_WORDS = 5  # Minimum new words before updating context

# Store active context manager threads
context_manager_threads = {}


def update_context_summary(session, session_id, socketio):
    """Update the context summary using Gemini.

    Args:
        session: TranscriptionSession instance
        session_id: Socket session ID
        socketio: Flask-SocketIO instance
    """
    from google import genai
    from google.genai import types

    if not GOOGLE_API_KEY:
        print("[Context Manager] GOOGLE_API_KEY not configured")
        return

    # Check if we have enough new words
    new_words = len(session.words) - session.context_last_word_count
    if new_words < CONTEXT_MIN_NEW_WORDS and session.context_summary:
        return

    # Get recent transcription text (last 30 seconds worth)
    if not session.full_text.strip():
        return

    # Use recent text for context (last ~500 characters for efficiency)
    recent_text = session.full_text.strip()[-500:]

    prompt = f"""Summarize the following conversation/speech in exactly 1-2 short sentences.
Focus on the main topic being discussed. Be concise and capture the key subject matter.

Text: {recent_text}

Summary:"""

    try:
        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100,
            ),
        )

        summary = response.text.strip()

        # Update session state
        session.context_summary = summary
        session.context_last_updated = datetime.utcnow()
        session.context_last_word_count = len(session.words)

        print(f"[Context Manager] Updated context: {summary}")

        # Emit context update to client
        socketio.emit(
            "context_updated",
            {
                "context": summary,
                "timestamp": datetime.utcnow().isoformat(),
            },
            room=session_id,
        )

        session._log_event(
            "context_summary_updated",
            {
                "summary": summary,
                "word_count": len(session.words),
                "text_length": len(recent_text),
            },
        )

    except Exception as e:
        print(f"[Context Manager] Error updating context: {e}")


def start_context_manager(session, session_id, socketio, transcription_sessions):
    """Start the context manager background thread.

    Args:
        session: TranscriptionSession instance
        session_id: Socket session ID
        socketio: Flask-SocketIO instance
        transcription_sessions: Dict of all sessions
    """
    if session_id in context_manager_threads:
        print(f"[Context Manager] Already running for session {session_id}")
        return

    session.context_manager_active = True

    def context_loop():
        import time

        while True:
            # Check if session still exists and is active
            if session_id not in transcription_sessions:
                break

            current_session = transcription_sessions.get(session_id)
            if not current_session or not current_session.context_manager_active:
                break

            # Only update if transcription is active
            if current_session.openai_active and len(current_session.words) > 0:
                update_context_summary(current_session, session_id, socketio)

            # Wait for next interval
            time.sleep(current_session.context_update_interval)

        # Cleanup
        if session_id in context_manager_threads:
            del context_manager_threads[session_id]
        print(f"[Context Manager] Stopped for session {session_id}")

    thread = threading.Thread(target=context_loop)
    thread.daemon = True
    thread.start()
    context_manager_threads[session_id] = thread

    print(f"[Context Manager] Started for session {session_id}, interval={session.context_update_interval}s")


def stop_context_manager(session_id):
    """Stop the context manager for a session."""
    if session_id in context_manager_threads:
        del context_manager_threads[session_id]
        print(f"[Context Manager] Stopped for session {session_id}")


def register_context_manager_handlers(socketio, transcription_sessions):
    """Register context manager event handlers."""

    @socketio.on("start_context_manager")
    def handle_start_context_manager(data=None):
        """Start the context manager for this session."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        # Update interval if provided
        if data and "interval" in data:
            session.context_update_interval = float(data["interval"])

        start_context_manager(session, session_id, socketio, transcription_sessions)

        emit(
            "context_manager_status",
            {
                "active": True,
                "interval": session.context_update_interval,
            },
        )

    @socketio.on("stop_context_manager")
    def handle_stop_context_manager():
        """Stop the context manager for this session."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            return

        session = transcription_sessions[session_id]
        session.context_manager_active = False
        stop_context_manager(session_id)

        emit(
            "context_manager_status",
            {
                "active": False,
                "interval": session.context_update_interval,
            },
        )

    @socketio.on("set_context_interval")
    def handle_set_context_interval(data):
        """Set the context update interval."""
        session_id = request.sid
        interval = data.get("interval", CONTEXT_UPDATE_INTERVAL)

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]
        session.context_update_interval = max(1.0, float(interval))

        emit(
            "context_manager_status",
            {
                "active": session.context_manager_active,
                "interval": session.context_update_interval,
            },
        )

    @socketio.on("get_context")
    def handle_get_context():
        """Get the current context summary."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        emit(
            "context_updated",
            {
                "context": session.context_summary,
                "timestamp": session.context_last_updated.isoformat() if session.context_last_updated else None,
            },
        )
