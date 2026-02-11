"""Auto-inference SocketIO event handlers."""

import threading

from flask import request
from flask_socketio import emit

auto_inference_timers = {}


def trigger_auto_inference(session, session_id, socketio):
    """Trigger automatic keyword extraction."""
    if not session.should_auto_inference():
        return

    print(
        f"[Auto-Inference] Triggering for session {session_id}, mode={session.auto_inference_mode}"
    )

    # Call Gemini for keywords (auto_mode=True allows 0-3 keywords)
    time_threshold = 5
    keywords = session.get_top_keyword_gemini(time_threshold, auto_mode=True)

    if not keywords:
        print("[Auto-Inference] No keywords extracted")
        return

    session.mark_auto_inference_done()

    # Emit keywords (same as manual mode)
    socketio.emit(
        "keywords_extracted",
        {
            "keywords": keywords,
            "history": session.keyword_history,
        },
        room=session_id,
    )

    session.log_search_action(
        search_mode="gemini_auto",
        search_type="text",
        keyword=keywords[0]["keyword"] if keywords else None,
        num_results=len(keywords),
    )

    print(f"[Auto-Inference] Extracted keywords: {[k['keyword'] for k in keywords]}")


def start_time_based_auto_inference(
    session, session_id, socketio, transcription_sessions
):
    """Start a background timer for time-based auto-inference."""
    if session_id in auto_inference_timers:
        # Already running
        return

    def timer_loop():
        while True:
            # Check if session still exists and auto mode is on
            if session_id not in transcription_sessions:
                break
            current_session = transcription_sessions.get(session_id)
            if not current_session or current_session.auto_inference_mode != "time":
                break
            if not current_session.openai_active:
                # Wait until transcription is active
                import time

                time.sleep(1)
                continue

            # Wait for interval
            import time

            time.sleep(current_session.auto_inference_interval)

            # Check again after sleep
            if session_id not in transcription_sessions:
                break
            current_session = transcription_sessions.get(session_id)
            if not current_session or current_session.auto_inference_mode != "time":
                break

            # Trigger auto-inference
            trigger_auto_inference(current_session, session_id, socketio)

        # Cleanup
        if session_id in auto_inference_timers:
            del auto_inference_timers[session_id]
        print(f"[Auto-Inference] Timer stopped for session {session_id}")

    thread = threading.Thread(target=timer_loop)
    thread.daemon = True
    thread.start()
    auto_inference_timers[session_id] = thread
    print(
        f"[Auto-Inference] Timer started for session {session_id}, interval={session.auto_inference_interval}s"
    )


def stop_auto_inference_timer(session_id):
    """Stop the auto-inference timer for a session."""
    if session_id in auto_inference_timers:
        del auto_inference_timers[session_id]
        print(f"[Auto-Inference] Timer stopped for session {session_id}")


def register_auto_inference_handlers(socketio, transcription_sessions):
    """Register auto-inference event handlers."""

    @socketio.on("set_auto_inference")
    def handle_set_auto_inference(data):
        """Set auto-inference mode and interval.

        data: {
            mode: "off" | "time" | "sentence",
            interval: float (seconds, for time mode)
        }
        """
        session_id = request.sid
        mode = data.get("mode", "off")
        interval = data.get("interval")

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]
        session.set_auto_inference_mode(mode, interval)

        # Handle timer for time-based mode
        if mode == "time":
            start_time_based_auto_inference(
                session, session_id, socketio, transcription_sessions
            )
        else:
            stop_auto_inference_timer(session_id)

        emit(
            "auto_inference_status",
            {
                "mode": session.auto_inference_mode,
                "interval": session.auto_inference_interval,
            },
        )
        print(f"[Auto-Inference] Mode set to '{mode}' for session {session_id}")

    @socketio.on("get_auto_inference_status")
    def handle_get_auto_inference_status():
        """Get current auto-inference settings."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]
        emit(
            "auto_inference_status",
            {
                "mode": session.auto_inference_mode,
                "interval": session.auto_inference_interval,
            },
        )
