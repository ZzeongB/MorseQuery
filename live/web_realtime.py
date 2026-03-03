"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import threading
from typing import Optional

import pyaudio

from clients import RealtimeClient, SummaryClient
from config import TEMPLATES_DIR
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from handlers.grounding import handle_search_grounding
from logger import get_logger, log_print

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
sio = SocketIO(app, cors_allowed_origins="*")

# Active clients (per session in production, global for simplicity here)
_clients_lock = threading.Lock()
client: Optional[RealtimeClient] = None
summary_clients: list[SummaryClient] = []  # One per summary mic


@app.route("/")
def index():
    """Serve the main page."""
    log_print("INFO", "Index page requested")
    return render_template("realtime.html")


@app.route("/api/devices")
def api_devices():
    """Return list of available audio input devices."""
    pa = pyaudio.PyAudio()
    devices = []

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            devices.append({
                "index": i,
                "name": info["name"],
                "channels": info["maxInputChannels"],
            })

    pa.terminate()
    log_print("INFO", f"Found {len(devices)} input devices")
    return jsonify(devices)


@sio.on("connect")
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    global client, summary_clients
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    logger = get_logger(session_id)
    logger.log("client_disconnected")

    # Stop running clients when user disconnects (e.g., page refresh)
    with _clients_lock:
        if client:
            client.stop()
            client = None
        for sc in summary_clients:
            sc.stop()
        summary_clients = []


@sio.on("start")
def handle_start(data: dict):
    """Start audio streaming and keyword extraction."""
    global client, summary_clients
    session_id = request.sid
    log_print("INFO", "Start requested", session_id=session_id, data=data)

    with _clients_lock:
        if client:
            log_print("INFO", "Stopping previous client", session_id=session_id)
            client.stop()
        for sc in summary_clients:
            sc.stop()
        summary_clients = []

        source = data.get("source", "mic")

        # Get mic selections: keyword_mic (single) and summary_mics (list of up to 2)
        keyword_mic = data.get("keyword_mic")  # int or None
        summary_mics = data.get("summary_mics", [])  # list of ints
        voice_ids = data.get("voice_ids", [])  # list of voice IDs for each summary mic

        client = RealtimeClient(sio, source, session_id, device_index=keyword_mic)

        # Create one SummaryClient per summary mic
        for i, mic_idx in enumerate(summary_mics):
            voice_id = voice_ids[i] if i < len(voice_ids) else None
            sc = SummaryClient(
                sio,
                session_id=f"{session_id}_sum{i}",
                device_indices=[mic_idx],
                enable_tts=bool(voice_id),
                mic_id=f"summary_{i}",
                voice_id=voice_id,
            )
            summary_clients.append(sc)
            sc.start()

        # Connect RealtimeClient to SummaryClients for transcript forwarding
        client.set_summary_clients(summary_clients)

        client.start()


@sio.on("stop")
def handle_stop():
    """Stop audio streaming."""
    global client
    session_id = request.sid
    log_print("INFO", "Stop requested", session_id=session_id)

    with _clients_lock:
        if client:
            client.stop()


@sio.on("request")
def handle_request():
    """Handle manual keyword extraction request."""
    session_id = request.sid
    log_print("INFO", "Manual request triggered", session_id=session_id)

    with _clients_lock:
        if client and client.running:
            client.request()
        else:
            log_print("WARN", "Request ignored - no running client", session_id=session_id)


@sio.on("start_listening")
def handle_start_listening():
    """Start a listening segment for later summarization."""
    session_id = request.sid

    with _clients_lock:
        if summary_clients:
            for sc in summary_clients:
                sc.start_listening()
            log_print("INFO", "Start listening", session_id=session_id, clients=len(summary_clients))
        else:
            log_print(
                "WARN",
                "start_listening ignored - no summary clients",
                session_id=session_id,
            )


@sio.on("end_listening")
def handle_end_listening():
    """End listening segment and request summary."""
    session_id = request.sid

    with _clients_lock:
        if summary_clients:
            for sc in summary_clients:
                sc.end_listening()
            log_print("INFO", "End listening, requesting summary", session_id=session_id, clients=len(summary_clients))
        else:
            log_print(
                "WARN",
                "end_listening ignored - no summary clients",
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
    log_print("INFO", "=" * 50)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
