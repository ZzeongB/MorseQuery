#!/usr/bin/env python3
"""Simple audio recording server for 3 microphones.

Usage:
    python record_server.py

Frontend sends:
    start: {mic0: int, mic1: int, mic2: int, sample_rate?: int, threshold?: int}
    stop: {}

Server emits:
    status: {recording: bool, elapsed_sec: float}
    recording_saved: {files: [path, path, path], duration_sec: float}
    error: {message: str}
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pyaudio
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from pydub import AudioSegment

from config import AUDIO_RATE, AUDIO_CHUNK, LOG_DIR

# Audio settings (from config.py, same as web_realtime.py)
DEFAULT_SAMPLE_RATE = AUDIO_RATE  # 24000
DEFAULT_CHUNK_SIZE = AUDIO_CHUNK  # 2400
MAX_DURATION_SEC = 600.0  # 10 minutes max

# Directories
AUDIO_DIR = LOG_DIR / "recordings"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")


class RecordingSession:
    """Manages concurrent recording from multiple microphones."""

    def __init__(
        self,
        device_indices: list[int],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        threshold: int = 0,
    ):
        self.device_indices = device_indices
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.threshold = threshold

        self.running = False
        self.start_time: Optional[float] = None
        self.buffers: dict[int, list[bytes]] = {idx: [] for idx in device_indices}
        self.lock = threading.Lock()
        self.threads: list[threading.Thread] = []

    def _apply_noise_gate(self, chunk: bytes) -> bytes:
        """Apply simple noise gate to audio chunk."""
        if self.threshold <= 0:
            return chunk
        arr = np.frombuffer(chunk, dtype=np.int16).copy()
        arr[np.abs(arr) < self.threshold] = 0
        return arr.tobytes()

    def _record_device(self, device_idx: int) -> None:
        """Record from a single device in a thread."""
        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(device_idx)
            device_name = info.get("name", f"Device {device_idx}")

            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=self.chunk_size,
            )

            print(f"[INFO] Recording started: {device_name} (index={device_idx})")

            while self.running:
                try:
                    chunk = stream.read(self.chunk_size, exception_on_overflow=False)
                    chunk = self._apply_noise_gate(chunk)
                    with self.lock:
                        self.buffers[device_idx].append(chunk)
                except Exception as e:
                    print(f"[WARN] Read error on device {device_idx}: {e}")
                    continue

        except Exception as e:
            print(f"[ERROR] Failed to open device {device_idx}: {e}")
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
            print(f"[INFO] Recording stopped: device {device_idx}")

    def start(self) -> None:
        """Start recording from all configured devices."""
        if self.running:
            return

        self.running = True
        self.start_time = time.time()
        self.buffers = {idx: [] for idx in self.device_indices}
        self.threads = []

        for device_idx in self.device_indices:
            t = threading.Thread(target=self._record_device, args=(device_idx,), daemon=True)
            t.start()
            self.threads.append(t)

    def stop(self) -> dict[int, bytes]:
        """Stop recording and return raw audio data per device."""
        self.running = False

        # Wait for threads to finish
        for t in self.threads:
            t.join(timeout=1.0)

        with self.lock:
            result = {idx: b"".join(chunks) for idx, chunks in self.buffers.items()}
            self.buffers = {idx: [] for idx in self.device_indices}

        return result

    def get_elapsed(self) -> float:
        """Get elapsed recording time in seconds."""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time


# Global session state
_session: Optional[RecordingSession] = None
_session_lock = threading.Lock()
_status_thread: Optional[threading.Thread] = None


def _emit_status_loop(sid: str) -> None:
    """Periodically emit recording status."""
    global _session
    while True:
        with _session_lock:
            session = _session
        if session is None or not session.running:
            break
        elapsed = session.get_elapsed()
        sio.emit("status", {"recording": True, "elapsed_sec": round(elapsed, 1)}, to=sid)
        time.sleep(0.5)


def _save_audio(raw_data: bytes, output_path: Path, sample_rate: int) -> None:
    """Save raw PCM data as WAV file."""
    if not raw_data:
        # Create empty file
        segment = AudioSegment.silent(duration=0, frame_rate=sample_rate)
    else:
        segment = AudioSegment(
            data=raw_data,
            sample_width=2,  # 16-bit
            frame_rate=sample_rate,
            channels=1,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    segment.export(output_path, format="wav")


def list_input_devices() -> list[dict]:
    """List available input devices."""
    pa = pyaudio.PyAudio()
    devices = []
    try:
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                devices.append({
                    "index": i,
                    "name": info.get("name", f"Device {i}"),
                    "channels": int(info.get("maxInputChannels", 0)),
                    "sample_rate": int(info.get("defaultSampleRate", 0)),
                })
    finally:
        pa.terminate()
    return devices


# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>3-Mic Recorder</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .device-select { margin: 10px 0; }
        .device-select label { display: inline-block; width: 100px; }
        .device-select select { width: 300px; padding: 5px; }
        .controls { margin: 20px 0; }
        button { padding: 10px 30px; font-size: 16px; margin-right: 10px; cursor: pointer; }
        #startBtn { background: #4CAF50; color: white; border: none; border-radius: 5px; }
        #stopBtn { background: #f44336; color: white; border: none; border-radius: 5px; }
        #stopBtn:disabled, #startBtn:disabled { background: #ccc; cursor: not-allowed; }
        .status { margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 5px; }
        .status.recording { background: #ffebee; border: 1px solid #f44336; }
        .files { margin: 20px 0; }
        .file-item { padding: 5px 0; color: #666; }
        .settings { margin: 20px 0; padding: 15px; background: #e3f2fd; border-radius: 5px; }
        .settings label { display: inline-block; width: 150px; }
        .settings input { width: 100px; padding: 5px; }
    </style>
</head>
<body>
    <h1>3-Microphone Recorder</h1>

    <div class="device-select">
        <label>Mic 0:</label>
        <select id="mic0"></select>
    </div>
    <div class="device-select">
        <label>Mic 1:</label>
        <select id="mic1"></select>
    </div>
    <div class="device-select">
        <label>Mic 2:</label>
        <select id="mic2"></select>
    </div>

    <div class="settings">
        <div>
            <label>Sample Rate:</label>
            <input type="number" id="sampleRate" value="{{ sample_rate }}"> Hz
        </div>
        <div style="margin-top: 10px;">
            <label>Chunk Size:</label>
            <input type="number" id="chunkSize" value="{{ chunk_size }}"> samples
        </div>
        <div style="margin-top: 10px;">
            <label>Noise Threshold:</label>
            <input type="number" id="threshold" value="0"> (0 = disabled)
        </div>
    </div>

    <div class="controls">
        <button id="startBtn" onclick="startRecording()">Start Recording</button>
        <button id="stopBtn" onclick="stopRecording()" disabled>Stop Recording</button>
    </div>

    <div id="statusBox" class="status">
        Status: Ready
    </div>

    <div id="filesBox" class="files"></div>

    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script>
        const socket = io();
        let recording = false;

        socket.on('connect', () => {
            console.log('Connected');
            socket.emit('list_devices');
        });

        socket.on('devices', (devices) => {
            const selects = ['mic0', 'mic1', 'mic2'];
            selects.forEach((id, i) => {
                const select = document.getElementById(id);
                select.innerHTML = '<option value="-1">-- Select --</option>';
                devices.forEach(d => {
                    const opt = document.createElement('option');
                    opt.value = d.index;
                    opt.textContent = `[${d.index}] ${d.name}`;
                    select.appendChild(opt);
                });
            });

            // Auto-select defaults: mic0=1, mic1=2, mic2=Jabra or MacBook
            const mic0Select = document.getElementById('mic0');
            const mic1Select = document.getElementById('mic1');
            const mic2Select = document.getElementById('mic2');

            // mic0: index 1
            if (devices.some(d => d.index === 1)) {
                mic0Select.value = '1';
            }
            // mic1: index 2
            if (devices.some(d => d.index === 2)) {
                mic1Select.value = '2';
            }
            // mic2: Jabra if exists, else MacBook
            const jabra = devices.find(d => d.name.toLowerCase().includes('jabra'));
            const macbook = devices.find(d => d.name.toLowerCase().includes('macbook'));
            if (jabra) {
                mic2Select.value = jabra.index.toString();
            } else if (macbook) {
                mic2Select.value = macbook.index.toString();
            }
        });

        socket.on('status', (data) => {
            const box = document.getElementById('statusBox');
            if (data.recording) {
                box.className = 'status recording';
                box.textContent = `Recording... ${data.elapsed_sec.toFixed(1)}s`;
            } else {
                box.className = 'status';
                box.textContent = 'Status: Ready';
            }
        });

        socket.on('recording_saved', (data) => {
            const box = document.getElementById('filesBox');
            box.innerHTML = '<strong>Saved files:</strong>';
            data.files.forEach(f => {
                box.innerHTML += `<div class="file-item">${f}</div>`;
            });
            box.innerHTML += `<div class="file-item">Duration: ${data.duration_sec.toFixed(1)}s</div>`;

            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            recording = false;
        });

        socket.on('error', (data) => {
            alert('Error: ' + data.message);
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            recording = false;
        });

        function startRecording() {
            const mic0 = parseInt(document.getElementById('mic0').value);
            const mic1 = parseInt(document.getElementById('mic1').value);
            const mic2 = parseInt(document.getElementById('mic2').value);
            const sampleRate = parseInt(document.getElementById('sampleRate').value);
            const chunkSize = parseInt(document.getElementById('chunkSize').value);
            const threshold = parseInt(document.getElementById('threshold').value);

            if (mic0 < 0 || mic1 < 0 || mic2 < 0) {
                alert('Please select all 3 microphones');
                return;
            }

            socket.emit('start', {
                mic0: mic0,
                mic1: mic1,
                mic2: mic2,
                sample_rate: sampleRate,
                chunk_size: chunkSize,
                threshold: threshold
            });

            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            recording = true;
        }

        function stopRecording() {
            socket.emit('stop');
            document.getElementById('stopBtn').disabled = true;
        }
    </script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        HTML_TEMPLATE,
        sample_rate=DEFAULT_SAMPLE_RATE,
        chunk_size=DEFAULT_CHUNK_SIZE,
    )


@sio.on("connect")
def handle_connect():
    print(f"[INFO] Client connected: {__import__('flask').request.sid}")


@sio.on("disconnect")
def handle_disconnect():
    print(f"[INFO] Client disconnected")


@sio.on("list_devices")
def handle_list_devices():
    devices = list_input_devices()
    sio.emit("devices", devices)


@sio.on("start")
def handle_start(data: dict):
    global _session, _status_thread

    sid = __import__("flask").request.sid

    mic0 = data.get("mic0", -1)
    mic1 = data.get("mic1", -1)
    mic2 = data.get("mic2", -1)
    sample_rate = data.get("sample_rate", DEFAULT_SAMPLE_RATE)
    chunk_size = data.get("chunk_size", DEFAULT_CHUNK_SIZE)
    threshold = data.get("threshold", 0)

    if mic0 < 0 or mic1 < 0 or mic2 < 0:
        sio.emit("error", {"message": "All 3 microphones must be selected"}, to=sid)
        return

    with _session_lock:
        if _session and _session.running:
            sio.emit("error", {"message": "Recording already in progress"}, to=sid)
            return

        _session = RecordingSession(
            device_indices=[mic0, mic1, mic2],
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            threshold=threshold,
        )
        _session.start()

    # Start status update thread
    _status_thread = threading.Thread(target=_emit_status_loop, args=(sid,), daemon=True)
    _status_thread.start()

    print(f"[INFO] Recording started: mics=[{mic0}, {mic1}, {mic2}], rate={sample_rate}, chunk={chunk_size}, threshold={threshold}")


@sio.on("stop")
def handle_stop():
    global _session

    sid = __import__("flask").request.sid

    with _session_lock:
        session = _session
        _session = None

    if session is None:
        sio.emit("error", {"message": "No recording in progress"}, to=sid)
        return

    # Stop and get audio data
    raw_data = session.stop()
    duration_sec = session.get_elapsed()

    # Save files in session folder: 날짜시간_세션/mic0.wav, mic1.wav, mic2.wav
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = AUDIO_DIR / f"{timestamp}_{sid[:8]}"
    session_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for i, (device_idx, data) in enumerate(raw_data.items()):
        filename = f"mic{i}.wav"
        output_path = session_dir / filename
        _save_audio(data, output_path, session.sample_rate)
        saved_files.append(str(output_path))
        print(f"[INFO] Saved: {output_path}")

    sio.emit("status", {"recording": False, "elapsed_sec": 0}, to=sid)
    sio.emit(
        "recording_saved",
        {"files": saved_files, "duration_sec": duration_sec},
        to=sid,
    )


if __name__ == "__main__":
    import signal
    import os

    def force_exit(signum, frame):
        print("\n[INFO] Shutting down...")
        os._exit(0)

    signal.signal(signal.SIGINT, force_exit)
    signal.signal(signal.SIGTERM, force_exit)

    print("=" * 50)
    print("3-Microphone Recording Server")
    print("http://localhost:5003")
    print("=" * 50)

    # List available devices on startup
    devices = list_input_devices()
    print("\nAvailable input devices:")
    for d in devices:
        print(f"  [{d['index']:2d}] {d['name']} (ch={d['channels']}, sr={d['sample_rate']})")
    print()

    sio.run(app, host="0.0.0.0", port=5003, debug=False, allow_unsafe_werkzeug=True)
