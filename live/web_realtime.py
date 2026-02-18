"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

from typing import Optional

from clients import RealtimeClient, SummaryClient
from config import LOG_DIR, TEMPLATES_DIR
from flask import Flask, render_template, request
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


@sio.on("connect")
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    logger = get_logger(session_id)
    logger.log("client_disconnected")


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
    # summary_client = SummaryClient(sio, session_id)
    # client.summary_client = summary_client

    # summary_client.start()
    summary_client = None
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
        summary_client.stop()

    summary_client = SummaryClient(sio, session_id)
    client.summary_client = summary_client
    summary_client.start()


@sio.on("request_summary")
def handle_request_summary():
    """Handle summary request."""
    global summary_client
    session_id = request.sid
    log_print("INFO", "Summary request triggered", session_id=session_id)

    if summary_client and summary_client.running:
        summary_client.request_summary()
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
