"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import json
from typing import Optional

from clients import RealtimeClient, SummaryClient
from config import LOG_DIR, TEMPLATES_DIR
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO
from handlers.grounding import handle_search_grounding
from logger import get_logger, log_print

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
sio = SocketIO(app, cors_allowed_origins="*")

# Active clients (per session in production, global for simplicity here)
client: Optional[RealtimeClient] = None
summary_client: Optional[SummaryClient] = None


@app.route("/")
def index():
    """Serve the main page."""
    log_print("INFO", "Index page requested")
    return render_template("realtime.html")


@app.route("/review")
def review():
    """Serve the review page."""
    log_print("INFO", "Review page requested")
    return render_template("review.html")


@app.route("/api/sessions")
def api_sessions():
    """Return list of available sessions."""
    transcript_dir = LOG_DIR / "transcript"
    sessions = []

    if transcript_dir.exists():
        for f in sorted(transcript_dir.glob("*.json"), reverse=True):
            # Parse filename: {date}_{session_id}.json
            parts = f.stem.split("_", 1)
            if len(parts) == 2:
                date_str, session_id = parts
                sessions.append({
                    "id": f.stem,
                    "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                    "session_id": session_id[:12] + "..." if len(session_id) > 12 else session_id,
                })

    return jsonify(sessions)


@app.route("/api/session/<session_id>")
def api_session(session_id):
    """Return session data (transcript with timestamps)."""
    transcript_file = LOG_DIR / "transcript" / f"{session_id}.json"

    if not transcript_file.exists():
        return jsonify({"error": "Session not found"}), 404

    with open(transcript_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/api/audio/<filename>")
def api_audio(filename):
    """Serve audio file."""
    audio_file = LOG_DIR / "audio" / filename

    if not audio_file.exists():
        return jsonify({"error": "Audio not found"}), 404

    return send_file(audio_file, mimetype="audio/wav")


@app.route("/api/review/<session_id>", methods=["GET"])
def api_get_review(session_id):
    """Get review data for a session (returns latest review)."""
    review_dir = LOG_DIR / "review"

    if not review_dir.exists():
        return jsonify({"reviews": {}})

    # Find all review files for this session and get the latest
    review_files = sorted(review_dir.glob(f"*_{session_id}.json"), reverse=True)

    if not review_files:
        return jsonify({"reviews": {}})

    with open(review_files[0], "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify(data)


@app.route("/api/review/<session_id>", methods=["POST"])
def api_save_review(session_id):
    """Save review data for a session."""
    from datetime import datetime

    review_dir = LOG_DIR / "review"
    review_dir.mkdir(exist_ok=True)

    datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    review_file = review_dir / f"{datetime_str}_{session_id}.json"
    data = request.get_json()

    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    log_print("INFO", f"Review saved: {review_file.name}")
    return jsonify({"status": "ok"})


@sio.on("connect")
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    global client, summary_client
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    logger = get_logger(session_id)
    logger.log("client_disconnected")

    # Stop running clients when user disconnects (e.g., page refresh)
    if client:
        client.stop()
        client = None
    if summary_client:
        summary_client.stop()
        summary_client = None


@sio.on("start")
def handle_start(data: dict):
    """Start audio streaming and keyword extraction."""
    global client, summary_client
    session_id = request.sid
    log_print("INFO", "Start requested", session_id=session_id, data=data)

    if client:
        log_print("INFO", "Stopping previous client", session_id=session_id)
        client.stop()
    if summary_client:
        summary_client.stop()

    mode = data.get("mode", "manual")
    source = data.get("source", "mp3")

    client = RealtimeClient(sio, mode, source, session_id)
    summary_client = SummaryClient(sio, session_id)
    client.summary_client = summary_client

    summary_client.start()
    client.start()


@sio.on("stop")
def handle_stop():
    """Stop audio streaming."""
    global client
    session_id = request.sid
    log_print("INFO", "Stop requested", session_id=session_id)

    if client:
        client.stop()


@sio.on("request")
def handle_request():
    """Handle manual keyword extraction request."""
    global client, summary_client
    session_id = request.sid
    log_print("INFO", "Manual request triggered", session_id=session_id)

    if client and client.running:
        client.request()
    else:
        log_print("WARN", "Request ignored - no running client", session_id=session_id)

    if summary_client:
        summary_client.start_miss()
        # summary_client.stop()

    # summary_client = SummaryClient(sio, session_id)
    # client.summary_client = summary_client
    # summary_client.start()


@sio.on("request_summary")
def handle_request_summary(data=None):
    global summary_client
    session_id = request.sid
    mode = (data or {}).get("mode", "summary")

    if summary_client:
        summary_client.end_miss_and_recover(mode=mode)
        log_print("INFO", "Summary request triggered", session_id=session_id, mode=mode)
    else:
        log_print(
            "WARN",
            "Summary request ignored - no running summary client",
            session_id=session_id,
        )


@sio.on("search_grounding")
def handle_grounding(data: dict):
    """Handle search grounding request (long-press)."""
    session_id = request.sid
    handle_search_grounding(sio, session_id, data)


if __name__ == "__main__":
    log_print("INFO", "=" * 50)
    log_print("INFO", "Starting web_realtime server")
    log_print("INFO", f"Log directory: {LOG_DIR}")
    log_print("INFO", "=" * 50)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)


@sio.on("summary_closed")
def handle_summary_closed():
    global client, summary_client
    session_id = request.sid
    log_print("INFO", "SummaryClient closed", session_id=session_id)

    if client:
        client.summary_client = None
    summary_client = None
