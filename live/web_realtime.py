"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import struct
import threading
import time
from typing import Optional

import numpy as np
import pyaudio

from clients import RealtimeClient, SummaryClient
from clients.context_judge_client import ContextJudgeClient
from clients.tts_client import TTSClient

# Mic level monitoring
_mic_monitor_streams: dict[int, tuple[pyaudio.Stream, pyaudio.PyAudio]] = {}
_mic_monitor_lock = threading.Lock()
_mic_monitor_running = False
_mic_monitor_thread: Optional[threading.Thread] = None

# Noise gate calibration monitoring
_noise_gate_monitor_running = False
_noise_gate_monitor_thread: Optional[threading.Thread] = None

# Global PyAudio lock to prevent segfaults from concurrent access
_pyaudio_lock = threading.Lock()
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
context_judge: Optional[ContextJudgeClient] = None  # Context-aware TTS judge
keyword_tts_client: Optional[TTSClient] = None

# Aggregate multiple summary-agent outputs into one judge request per segment
_judge_batch_lock = threading.Lock()
_judge_batch: dict[int, dict] = {}
_judge_completed_segments: set[int] = set()
_JUDGE_BATCH_TIMEOUT_SEC = 0.7


def _flush_judge_batch(segment_id: int, session_id: str) -> None:
    """Flush aggregated summaries for a segment into one judge request."""
    with _judge_batch_lock:
        batch = _judge_batch.get(segment_id)
        judge = context_judge
        if (
            not batch
            or batch.get("sent")
            or not judge
            or segment_id in _judge_completed_segments
        ):
            return

        summaries_by_source: dict[str, str] = batch.get("summaries", {})
        if not summaries_by_source:
            return

        batch["sent"] = True
        _judge_batch.pop(segment_id, None)
        _judge_completed_segments.add(segment_id)

    ordered_sources = sorted(summaries_by_source.keys())
    merged_parts = [
        summaries_by_source[source].strip()
        for source in ordered_sources
        if summaries_by_source[source].strip()
    ]
    merged_summary = " ".join(merged_parts)

    log_print(
        "INFO",
        "Flushing batched summaries to judge",
        session_id=session_id,
        segment_id=segment_id,
        count=len(summaries_by_source),
    )
    judge.judge_summary(merged_summary, segment_id)


def _make_summary_batch_callback(source_id: str, session_id: str):
    """Create callback that batches summary outputs before judge request."""

    def _callback(summary: str, segment_id: int) -> None:
        if not summary or not summary.strip():
            return

        flush_now = False
        with _judge_batch_lock:
            if segment_id in _judge_completed_segments:
                return
            expected = max(1, len(summary_clients))
            batch = _judge_batch.setdefault(
                segment_id,
                {
                    "summaries": {},
                    "expected": expected,
                    "sent": False,
                    "timer": None,
                },
            )
            batch["expected"] = expected
            batch["summaries"][source_id] = summary.strip()

            # First arrival: start timeout to avoid waiting forever.
            if batch.get("timer") is None:
                timer = threading.Timer(
                    _JUDGE_BATCH_TIMEOUT_SEC,
                    _flush_judge_batch,
                    args=(segment_id, session_id),
                )
                timer.daemon = True
                batch["timer"] = timer
                timer.start()

            if len(batch["summaries"]) >= batch["expected"]:
                flush_now = True

        if flush_now:
            _flush_judge_batch(segment_id, session_id)

    return _callback


@app.route("/")
def index():
    """Serve the main page."""
    log_print("INFO", "Index page requested")
    return render_template("realtime.html")


@app.route("/api/devices")
def api_devices():
    """Return list of available audio input devices."""
    with _pyaudio_lock:
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


def _mic_monitor_loop(device_indices: list[int], select_ids: list[str]):
    """Background thread to monitor mic levels using PyAudio."""
    global _mic_monitor_running, _mic_monitor_streams

    CHUNK = 1024
    RATE = 16000
    FORMAT = pyaudio.paInt16

    # Build mapping: device_idx -> list of select_ids
    device_to_selects: dict[int, list[str]] = {}
    for device_idx, select_id in zip(device_indices, select_ids):
        if device_idx not in device_to_selects:
            device_to_selects[device_idx] = []
        device_to_selects[device_idx].append(select_id)

    # Open streams for unique devices only
    streams = []
    for device_idx in device_to_selects.keys():
        try:
            with _pyaudio_lock:
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    input_device_index=device_idx,
                    frames_per_buffer=CHUNK,
                )
            streams.append((device_idx, stream, pa))
            log_print("INFO", f"Mic monitor opened for device {device_idx}")
        except Exception as e:
            log_print("ERROR", f"Failed to open mic monitor for device {device_idx}: {e}")

    while _mic_monitor_running and streams:
        levels = {}
        for device_idx, stream, pa in streams:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Calculate RMS level
                samples = struct.unpack(f"<{CHUNK}h", data)
                rms = (sum(s * s for s in samples) / CHUNK) ** 0.5
                # Normalize to 0-100 (max 32768 for 16-bit audio)
                level = min(100, int((rms / 8000) * 100))
                # Apply level to all select_ids that use this device
                for select_id in device_to_selects.get(device_idx, []):
                    levels[select_id] = level
            except Exception as e:
                log_print("ERROR", f"Mic monitor read error: {e}")

        if levels:
            sio.emit("mic_levels", levels)

        sio.sleep(0.05)  # 50ms interval

    # Cleanup
    for device_idx, stream, pa in streams:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except:
            pass

    log_print("INFO", "Mic monitor stopped")


def start_mic_monitor(device_indices: list[int], select_ids: list[str]):
    """Start monitoring mic levels for given device indices."""
    global _mic_monitor_running, _mic_monitor_thread

    stop_mic_monitor()

    if not device_indices:
        return

    # Small delay to ensure old thread has stopped
    time.sleep(0.1)

    _mic_monitor_running = True
    _mic_monitor_thread = sio.start_background_task(
        _mic_monitor_loop, device_indices, select_ids
    )
    log_print("INFO", f"Starting mic monitor for devices: {device_indices}")


def stop_mic_monitor():
    """Stop mic level monitoring."""
    global _mic_monitor_running, _mic_monitor_thread

    _mic_monitor_running = False
    if _mic_monitor_thread:
        _mic_monitor_thread = None


def _noise_gate_monitor_loop(device_indices: list[int], mic_ids: list[str]):
    """Background thread to monitor mic RMS for noise gate calibration."""
    global _noise_gate_monitor_running

    CHUNK = 1024
    RATE = 24000
    FORMAT = pyaudio.paInt16

    # Build mapping: device_idx -> mic_id
    device_to_mic: dict[int, str] = {}
    for device_idx, mic_id in zip(device_indices, mic_ids):
        device_to_mic[device_idx] = mic_id

    streams = []

    try:
        # Open streams for each unique device
        for device_idx in device_to_mic.keys():
            try:
                with _pyaudio_lock:
                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=FORMAT,
                        channels=1,
                        rate=RATE,
                        input=True,
                        input_device_index=device_idx,
                        frames_per_buffer=CHUNK,
                    )
                streams.append((device_idx, stream, pa))
                log_print("INFO", f"Noise gate monitor opened for device {device_idx}")
            except Exception as e:
                log_print("ERROR", f"Failed to open noise gate monitor for device {device_idx}: {e}")

        while _noise_gate_monitor_running and streams:
            levels = {}
            for device_idx, stream, pa in streams:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # Calculate RMS
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
                    mic_id = device_to_mic.get(device_idx)
                    if mic_id:
                        levels[mic_id] = rms
                except Exception as e:
                    log_print("ERROR", f"Noise gate monitor read error: {e}")

            if levels:
                sio.emit("noise_gate_levels", levels)

            sio.sleep(0.05)  # 50ms interval

    except Exception as e:
        log_print("ERROR", f"Noise gate monitor failed: {e}")
    finally:
        for device_idx, stream, pa in streams:
            try:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except:
                pass
        log_print("INFO", "Noise gate monitor stopped")


def start_noise_gate_monitor(device_indices: list[int], mic_ids: list[str]):
    """Start monitoring mic RMS for noise gate calibration."""
    global _noise_gate_monitor_running, _noise_gate_monitor_thread

    stop_noise_gate_monitor()
    time.sleep(0.1)

    _noise_gate_monitor_running = True
    _noise_gate_monitor_thread = sio.start_background_task(
        _noise_gate_monitor_loop, device_indices, mic_ids
    )
    log_print("INFO", f"Starting noise gate monitor for devices {device_indices}")


def stop_noise_gate_monitor():
    """Stop noise gate calibration monitoring."""
    global _noise_gate_monitor_running, _noise_gate_monitor_thread

    _noise_gate_monitor_running = False
    if _noise_gate_monitor_thread:
        _noise_gate_monitor_thread = None


@sio.on("start_noise_gate_monitor")
def handle_start_noise_gate_monitor(data: dict):
    """Start noise gate calibration monitoring."""
    session_id = request.sid
    device_indices = data.get("device_indices", [])
    mic_ids = data.get("mic_ids", [])
    if device_indices:
        log_print("INFO", f"Start noise gate monitor: devices={device_indices}, mics={mic_ids}", session_id=session_id)
        start_noise_gate_monitor(device_indices, mic_ids)


@sio.on("stop_noise_gate_monitor")
def handle_stop_noise_gate_monitor():
    """Stop noise gate calibration monitoring."""
    session_id = request.sid
    log_print("INFO", "Stop noise gate monitor", session_id=session_id)
    stop_noise_gate_monitor()


@sio.on("start_mic_monitor")
def handle_start_mic_monitor(data: dict):
    """Start mic level monitoring for selected devices."""
    session_id = request.sid
    device_indices = data.get("device_indices", [])
    select_ids = data.get("select_ids", [])
    log_print("INFO", f"Start mic monitor: devices={device_indices}, ids={select_ids}", session_id=session_id)
    start_mic_monitor(device_indices, select_ids)


@sio.on("stop_mic_monitor")
def handle_stop_mic_monitor():
    """Stop mic level monitoring."""
    session_id = request.sid
    log_print("INFO", "Stop mic monitor", session_id=session_id)
    stop_mic_monitor()


@sio.on("connect")
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    global client, summary_clients, context_judge, keyword_tts_client
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    logger = get_logger(session_id)
    logger.log("client_disconnected")

    # Stop mic monitors
    stop_mic_monitor()
    stop_noise_gate_monitor()

    # Stop running clients when user disconnects (e.g., page refresh)
    with _clients_lock:
        if client:
            client.stop()
            client = None
        for sc in summary_clients:
            sc.stop()
        summary_clients = []
        if context_judge:
            context_judge.stop()
            context_judge = None
        keyword_tts_client = None
    with _judge_batch_lock:
        _judge_batch.clear()
        _judge_completed_segments.clear()


@sio.on("start")
def handle_start(data: dict):
    """Start audio streaming and keyword extraction."""
    global client, summary_clients, context_judge, keyword_tts_client
    session_id = request.sid
    log_print("INFO", "Start requested", session_id=session_id, data=data)

    # Stop mic monitor when session starts
    stop_mic_monitor()
    stop_noise_gate_monitor()

    with _clients_lock:
        if client:
            log_print("INFO", "Stopping previous client", session_id=session_id)
            client.stop()
        for sc in summary_clients:
            sc.stop()
        summary_clients = []
        if context_judge:
            context_judge.stop()
            context_judge = None
        keyword_tts_client = TTSClient(sio, session_id=f"{session_id}_keyword_tts")
        with _judge_batch_lock:
            _judge_batch.clear()
            _judge_completed_segments.clear()

        source = data.get("source", "mic")

        # Get mic selections: keyword_mic (single) and summary_mics (list of up to 2)
        keyword_mic = data.get("keyword_mic")  # int or None
        summary_mics = data.get("summary_mics", [])  # list of ints
        voice_ids = data.get("voice_ids", [])  # list of voice IDs for each summary mic

        # Noise gate settings
        noise_gate_data = data.get("noise_gate", {})
        enable_noise_gate = noise_gate_data.get("enabled", False)
        noise_gate_config = None
        if enable_noise_gate:
            from clients.audio_filter import NoiseGateConfig
            threshold = noise_gate_data.get("threshold", 500)
            # Calculate margin_multiplier based on threshold
            # We set noise_floor to threshold/2 and margin to 2.0 so threshold = noise_floor * 2
            noise_gate_config = NoiseGateConfig(
                min_threshold=threshold,  # Use threshold directly as min
                margin_multiplier=1.0,    # Direct threshold mode
            )
            log_print(
                "INFO",
                f"Noise gate enabled with threshold={threshold}",
                session_id=session_id,
            )

        client = RealtimeClient(
            sio,
            source,
            session_id,
            device_index=keyword_mic,
            enable_noise_gate=enable_noise_gate,
            noise_gate_config=noise_gate_config,
        )

        # If noise gate is enabled with fixed threshold, set it
        if enable_noise_gate and client.noise_gate:
            client.set_noise_threshold(noise_gate_data.get("threshold", 500))

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

        # Create ContextJudgeClient if we have summary clients with TTS
        # Uses the first summary mic for audio context
        if summary_mics and summary_clients:
            judge_tts_clients = [
                sc.tts_client for sc in summary_clients if sc.tts_client is not None
            ]
            context_judge = ContextJudgeClient(
                sio,
                session_id=f"{session_id}_judge",
                device_indices=[summary_mics[0]],
                tts_clients=judge_tts_clients,
            )
            context_judge.start()

            # Connect summary callbacks to judge
            for i, sc in enumerate(summary_clients):
                sc.set_summary_callback(
                    _make_summary_batch_callback(source_id=f"sum{i}", session_id=session_id)
                )
                sc.set_tts_ready_callback(context_judge.on_tts_ready)

            log_print(
                "INFO",
                "ContextJudgeClient created and connected",
                session_id=session_id,
            )

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
            # Also notify context judge
            if context_judge:
                context_judge.start_listening()
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
            # Also notify context judge
            if context_judge:
                context_judge.end_listening()
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


@sio.on("keyword_tts")
def handle_keyword_tts(data: dict):
    """Synthesize keyword definition audio and emit to frontend."""
    session_id = request.sid
    text = str((data or {}).get("text", "")).strip()
    if not text:
        return

    with _clients_lock:
        tts = keyword_tts_client
    if not tts:
        return

    tts.synthesize_async(text, event_name="keyword_tts", language="en")
    log_print("INFO", "keyword_tts requested", session_id=session_id, chars=len(text))


@sio.on("cancel_tts")
def handle_cancel_tts():
    """Cancel pending/playing summary TTS."""
    session_id = request.sid
    with _clients_lock:
        if context_judge:
            context_judge.cancel_tts(reason="doubleclick_cancel")
        else:
            # Fallback: clear any summary client queues directly
            for sc in summary_clients:
                if sc.tts_client:
                    sc.tts_client.stop_playback()
    log_print("INFO", "cancel_tts handled", session_id=session_id)


if __name__ == "__main__":
    log_print("INFO", "=" * 50)
    log_print("INFO", "Starting web_realtime server")
    log_print("INFO", "=" * 50)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
