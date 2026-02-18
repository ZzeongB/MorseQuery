# pip install flask flask-socketio websocket-client pydub pyaudio
import base64
import json
import os
import threading
from datetime import datetime
from pathlib import Path

import pyaudio
import websocket
from flask import Flask, render_template_string, request
from flask_socketio import SocketIO
from pydub import AudioSegment

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")

# ============================================================
# Logging Setup
# ============================================================
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


def get_timestamp():
    """현재 시간 문자열 반환"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def log_print(level: str, message: str, **kwargs):
    """콘솔에 로그 출력"""
    timestamp = get_timestamp()
    extra = f" | {kwargs}" if kwargs else ""
    print(f"[{timestamp}] [{level.upper():5}] {message}{extra}")


class JsonLogger:
    """JSON 파일 로거"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.start_time = datetime.now()
        self.log_file = (
            LOG_DIR
            / f"realtime_{self.start_time.strftime('%Y%m%d_%H%M%S')}_{session_id}.json"
        )
        self.events = []
        log_print(
            "INFO",
            "JsonLogger initialized",
            session_id=session_id,
            log_file=str(self.log_file),
        )

    def log(self, event_type: str, **data):
        """이벤트 로깅"""
        event = {
            "timestamp": get_timestamp(),
            "event_type": event_type,
            "session_id": self.session_id,
            **data,
        }
        self.events.append(event)
        self._save()

    def _save(self):
        """로그 파일 저장"""
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


# 세션별 로거 저장
session_loggers = {}


def get_logger(session_id: str) -> JsonLogger:
    """세션별 로거 반환 (없으면 생성)"""
    if session_id not in session_loggers:
        session_loggers[session_id] = JsonLogger(session_id)
    return session_loggers[session_id]


# ============================================================

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
AUDIO_FILE = "../mp3/clips/U6fI3brP8V4_clip_885_1240.mp3"

RATE = 24000
CHUNK = 4800
AUTO_INTERVAL = 5

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Realtime Words</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.4/socket.io.min.js"></script>
    <style>
        body { margin: 0; background: #000; color: #fff; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; align-items: center; height: 100vh; }
        #menu { padding: 40px; }
        #menu button { background: #222; color: #fff; border: 1px solid #444; padding: 15px 30px; margin: 10px; cursor: pointer; font-size: 16px; }
        #menu button.selected { background: #4ade80; color: #000; border-color: #4ade80; }
        .source-select { margin-bottom: 30px; }
        .source-select span { margin-right: 15px; font-size: 14px; color: #888; }
        #box { display: none; width: 50%; padding: 20px; }
        .option { border: 1px solid #333; padding: 12px 16px; margin: 6px 0; font-size: 16px; border-radius: 8px; transition: all 0.2s; }
        .option.active { border-color: #4ade80; background: #1a1a1a; }
        .option .word { font-weight: 600; color: #4ade80; }
        .option .desc { color: #aaa; margin-left: 8px; }
        #context { margin-top: 16px; padding: 12px; background: #111; border: 1px solid #333; border-radius: 8px; font-size: 14px; color: #888; }
        #summary { margin-top: 16px; padding: 12px 16px; background: #1a1a2e; border: 1px solid #a78bfa; border-radius: 8px; font-size: 14px; color: #c4b5fd; }
    </style>
</head>
<body>
    <div id="menu">
        <div class="source-select">
            <span>Source:</span>
            <button id="btn-mic" onclick="setSource('mic')">🎤 Mic</button>
            <button id="btn-mp3" class="selected" onclick="setSource('mp3')">🎵 MP3</button>
        </div>
        <button onclick="start('manual', 'single')">Manual (Single)</button>
        <button onclick="start('manual', 'all')">Manual (All)</button>
        <button onclick="start('auto', 'single')">Auto (Single)</button>
        <button onclick="start('auto', 'all')">Auto (All)</button>
    </div>
    <div id="box"></div>
    <div id="context" style="display:none;"></div>
    <div id="summary" style="display:none;"></div>
    <script>
        const socket = io();
        let mode = 'manual';
        let view = 'single';
        let source = 'mp3';
        let options = [];
        let currentIdx = 0;
        let summaryBuffer = '';
        let lastSpace = 0;
        let infoVisible = false;

        function setSource(s) {
            source = s;
            document.getElementById('btn-mic').classList.toggle('selected', s === 'mic');
            document.getElementById('btn-mp3').classList.toggle('selected', s === 'mp3');
        }

        socket.on('keywords', data => {
            options = data;
            currentIdx = 0;
            render();
            infoVisible = true;
        });
        socket.on('context', data => {
            document.getElementById('context').innerText = data;
            document.getElementById('context').style.display = 'block';
        });
        socket.on('summary_chunk', data => {
            summaryBuffer += data;
            document.getElementById('summary').innerText = summaryBuffer;
        });
        socket.on('summary_done', () => {
            document.getElementById('summary').style.display = 'block';
        });
        socket.on('clear', () => {
            options = [];
            currentIdx = 0;
            document.getElementById('box').innerHTML = '';
            document.getElementById('context').style.display = 'none';
        });

        function render() {
            const box = document.getElementById('box');
            if (options.length === 0) {
                box.innerHTML = '';
                return;
            }
            if (view === 'single') {
                const o = options[currentIdx];
                box.innerHTML = `<div class="option active">
                    <span class="word">${o.word}</span><span class="desc">${o.desc}</span>
                </div>`;
            } else {
                box.innerHTML = options.map((o, i) =>
                    `<div class="option ${i === currentIdx ? 'active' : ''}">
                        <span class="word">${o.word}</span><span class="desc">${o.desc}</span>
                    </div>`
                ).join('');
            }
        }

        function hideInfo() {
            document.getElementById('box').innerHTML = '';
            document.getElementById('context').style.display = 'none';
            options = [];
            infoVisible = false;
        }

        function start(m, v) {
            mode = m;
            view = v;
            document.getElementById('menu').style.display = 'none';
            document.getElementById('box').style.display = 'block';
            socket.emit('start', {mode: m, source: source});
        }

        document.addEventListener('keydown', e => {
            if (e.code !== 'Space') return;
            e.preventDefault();

            const now = Date.now();
            if (now - lastSpace < 300) {
                // Double space
                if (infoVisible) {
                    // Hide info and request summary
                    hideInfo();
                    summaryBuffer = '';
                    document.getElementById('summary').innerText = '';
                    socket.emit('request_summary');
                } else {
                    // New keyword request
                    document.getElementById('summary').style.display = 'none';
                    socket.emit('request');
                }
                lastSpace = 0;
            } else {
                // Single space - navigate
                lastSpace = now;
                setTimeout(() => {
                    if (lastSpace !== 0 && Date.now() - lastSpace >= 280) {
                        if (options.length > 0) {
                            currentIdx = (currentIdx + 1) % options.length;
                            render();
                        }
                        lastSpace = 0;
                    }
                }, 300);
            }
        });
    </script>
</body>
</html>
"""


class RealtimeClient:
    def __init__(self, socketio, mode="manual", source="mp3", session_id="default"):
        self.sio = socketio
        self.mode = mode
        self.source = source
        self.session_id = session_id
        self.ws = None
        self.running = False
        self.chunks_sent = 0
        self.response_buffer = ""  # 응답 버퍼
        self.logger = get_logger(session_id)
        self.summary_client = None  # SummaryClient reference

        log_print(
            "INFO",
            "RealtimeClient created",
            session_id=session_id,
            mode=mode,
            source=source,
        )
        self.logger.log("client_created", mode=mode, source=source)

    def on_open(self, ws):
        log_print("INFO", "WebSocket connected to OpenAI", session_id=self.session_id)
        self.logger.log("websocket_connected")
        self.sio.emit("status", "Connected")
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": "Listen to audio. Identify difficult words. Be very concise.",
                    },
                }
            )
        )
        log_print("DEBUG", "Session update sent", session_id=self.session_id)
        self.logger.log("session_update_sent")
        threading.Thread(target=self.stream_audio, daemon=True).start()

    def on_message(self, _ws, message):
        event = json.loads(message)
        etype = event.get("type", "")

        # 주요 이벤트만 로깅 (너무 많은 delta 이벤트 제외)
        if etype not in [
            "response.text.delta",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
        ]:
            log_print("DEBUG", f"OpenAI event: {etype}", session_id=self.session_id)

        # OpenAI session events
        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log("openai_session_created", session_id=session_info.get("id"))
        elif etype == "session.updated":
            self.logger.log("openai_session_updated")
        elif etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
        elif etype == "response.done":
            # Parse keywords and context from full response
            keywords_text = self.response_buffer
            context_text = ""

            # Try CONTEXT:: first, then \n\n as fallback
            if "CONTEXT::" in self.response_buffer:
                parts = self.response_buffer.split("CONTEXT::", 1)
                keywords_text = parts[0].strip()
                context_text = parts[1].strip() if len(parts) > 1 else ""
            elif "\n\n" in self.response_buffer:
                parts = self.response_buffer.split("\n\n", 1)
                keywords_text = parts[0].strip()
                context_text = parts[1].strip() if len(parts) > 1 else ""

            # Parse keywords into list (skip CONTEXT lines)
            keywords = []
            for line in keywords_text.split("\n"):
                if ":" in line and not line.upper().startswith("CONTEXT"):
                    word, desc = line.split(":", 1)
                    keywords.append({"word": word.strip(), "desc": desc.strip()})

            log_print(
                "INFO",
                "Response complete",
                session_id=self.session_id,
                keywords=keywords,
                context=context_text,
            )
            self.logger.log(
                "response_done",
                response=self.response_buffer,
                keywords=keywords,
                context=context_text,
            )

            # Emit to frontend
            self.sio.emit("keywords", keywords)
            if context_text:
                self.sio.emit("context", context_text)
                # Update SummaryClient with context
                if self.summary_client:
                    self.summary_client.set_context(context_text)

            self.response_buffer = ""
        elif etype == "error":
            error_msg = event.get("error", {}).get("message", "Unknown error")
            log_print("ERROR", f"OpenAI error: {error_msg}", session_id=self.session_id)
            self.logger.log("openai_error", error=error_msg)

    def on_error(self, _ws, error):
        log_print("ERROR", f"WebSocket error: {error}", session_id=self.session_id)
        self.logger.log("websocket_error", error=str(error))
        self.sio.emit("status", f"Error: {error}")

    def on_close(self, _ws, status, msg):
        log_print(
            "INFO",
            "WebSocket closed",
            session_id=self.session_id,
            status=status,
            msg=msg,
        )
        self.logger.log("websocket_closed", status=status, message=msg)
        self.sio.emit("status", "Disconnected")
        self.running = False

    def stream_audio(self):
        log_print(
            "INFO",
            "Starting audio stream",
            session_id=self.session_id,
            source=self.source,
        )
        self.logger.log("stream_start", source=self.source)
        if self.source == "mic":
            self.stream_from_mic()
        else:
            self.stream_from_mp3()

    def stream_from_mic(self):
        pa = pyaudio.PyAudio()
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        log_print(
            "INFO", "Mic recording started", session_id=self.session_id, mode=self.mode
        )
        self.logger.log("mic_recording_start", mode=self.mode)
        self.sio.emit("status", f"🎤 Mic recording... ({self.mode} mode)")

        chunks_per_interval = int(AUTO_INTERVAL / 0.2)

        while self.running:
            chunk = stream.read(CHUNK, exception_on_overflow=False)
            audio_b64 = base64.b64encode(chunk).decode()
            self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
            )
            # Forward to SummaryClient
            if self.summary_client:
                self.summary_client.send_audio(audio_b64)
            self.chunks_sent += 1

            # 100 청크마다 진행상황 로깅
            if self.chunks_sent % 100 == 0:
                log_print(
                    "DEBUG",
                    f"Audio chunks sent: {self.chunks_sent}",
                    session_id=self.session_id,
                )

            if self.mode == "auto" and self.chunks_sent % chunks_per_interval == 0:
                self.request()

        stream.stop_stream()
        stream.close()
        pa.terminate()
        log_print(
            "INFO",
            "Mic recording stopped",
            session_id=self.session_id,
            total_chunks=self.chunks_sent,
        )
        self.logger.log("mic_recording_stop", total_chunks=self.chunks_sent)
        self.sio.emit("status", "Stopped")

    def stream_from_mp3(self):
        log_print("INFO", f"Loading MP3 file: {AUDIO_FILE}", session_id=self.session_id)
        audio = AudioSegment.from_file(AUDIO_FILE)
        audio = audio.set_frame_rate(RATE).set_channels(1).set_sample_width(2)
        raw = audio.raw_data
        duration_sec = len(audio) / 1000.0

        log_print(
            "INFO",
            "MP3 loaded",
            session_id=self.session_id,
            duration_sec=duration_sec,
            bytes=len(raw),
        )
        self.logger.log(
            "mp3_loaded", file=AUDIO_FILE, duration_sec=duration_sec, bytes=len(raw)
        )

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)
        self.sio.emit("status", f"🎵 Playing MP3... ({self.mode} mode)")

        chunk_bytes = CHUNK * 2
        chunks_per_interval = int(AUTO_INTERVAL / 0.2)
        total_chunks = len(raw) // chunk_bytes

        log_print(
            "INFO",
            "MP3 playback started",
            session_id=self.session_id,
            mode=self.mode,
            total_chunks=total_chunks,
        )
        self.logger.log("mp3_playback_start", mode=self.mode, total_chunks=total_chunks)

        for i in range(0, len(raw), chunk_bytes):
            if not self.running:
                break
            chunk = raw[i : i + chunk_bytes]
            stream.write(chunk)
            audio_b64 = base64.b64encode(chunk).decode()
            self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": audio_b64,
                    }
                )
            )
            # Forward to SummaryClient
            if self.summary_client:
                self.summary_client.send_audio(audio_b64)
            self.chunks_sent += 1

            # 100 청크마다 진행상황 로깅
            if self.chunks_sent % 100 == 0:
                progress = (
                    (self.chunks_sent / total_chunks) * 100 if total_chunks > 0 else 0
                )
                log_print(
                    "DEBUG",
                    f"Playback progress: {progress:.1f}%",
                    session_id=self.session_id,
                    chunks=self.chunks_sent,
                )

            if self.mode == "auto" and self.chunks_sent % chunks_per_interval == 0:
                self.request()

        if self.running:
            self.request()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        log_print(
            "INFO",
            "MP3 playback complete",
            session_id=self.session_id,
            total_chunks=self.chunks_sent,
        )
        self.logger.log("mp3_playback_complete", total_chunks=self.chunks_sent)
        self.sio.emit("status", "Done")

    def request(self):
        log_print(
            "INFO",
            "Requesting keyword extraction",
            session_id=self.session_id,
            chunks_so_far=self.chunks_sent,
        )
        self.logger.log("keyword_request", chunks_so_far=self.chunks_sent)
        self.sio.emit("clear")
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self.ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": """Pick 1-3 interesting words from the audio. ONLY words you ACTUALLY heard.
STRICT FORMAT (follow exactly):
word: description
word: description
CONTEXT:: summary sentence

Example:
entropy: measure of disorder
quantum: smallest unit
CONTEXT:: Discussion about physics and thermodynamics""",
                    },
                }
            )
        )

    def start(self):
        log_print("INFO", "Starting RealtimeClient", session_id=self.session_id)
        self.logger.log("client_start")
        self.running = True
        self.chunks_sent = 0
        self.ws = websocket.WebSocketApp(
            URL,
            header=[
                f"Authorization: Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta: realtime=v1",
            ],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        log_print("INFO", "Stopping RealtimeClient", session_id=self.session_id)
        self.logger.log("client_stop", total_chunks=self.chunks_sent)
        self.running = False
        if self.ws:
            self.ws.close()


class SummaryClient:
    """Separate realtime client that listens and summarizes the conversation."""

    def __init__(self, socketio, session_id="default"):
        self.sio = socketio
        self.session_id = session_id
        self.ws = None
        self.running = False
        self.response_buffer = ""
        self.last_context = ""  # 마지막으로 받은 context
        self.logger = get_logger(session_id)
        log_print("INFO", "SummaryClient created", session_id=session_id)
        self.logger.log("summary_client_created")

    def set_context(self, context):
        """RealtimeClient로부터 context 업데이트"""
        self.last_context = context
        log_print("DEBUG", f"Context updated: {context}", session_id=self.session_id)

    def on_open(self, ws):
        log_print(
            "INFO", "SummaryClient WebSocket connected", session_id=self.session_id
        )
        self.logger.log("summary_ws_connected")
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": "You are a passive listener. Summarize the conversation in one sentence (≤15 words). Output only the summary.",
                    },
                }
            )
        )

    def on_message(self, _ws, message):
        event = json.loads(message)
        etype = event.get("type", "")
        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "summary_openai_session_created", session_id=session_info.get("id")
            )
        elif etype == "session.updated":
            self.logger.log("summary_openai_session_updated")
        elif etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
                self.sio.emit("summary_chunk", delta)
        elif etype == "response.done":
            log_print("INFO", "SummaryClient response done", session_id=self.session_id)
            self.logger.log("summary_response", summary=self.response_buffer)
            self.response_buffer = ""
            self.sio.emit("summary_done")

    def on_error(self, _ws, error):
        log_print("ERROR", f"SummaryClient error: {error}", session_id=self.session_id)
        self.logger.log("summary_error", error=str(error))

    def on_close(self, _ws, status, msg):
        log_print("INFO", "SummaryClient closed", session_id=self.session_id)
        self.logger.log("summary_ws_closed", status=status, message=msg)
        self.running = False

    def send_audio(self, audio_b64):
        """Forward audio chunk to this client."""
        if self.ws and self.running:
            self.ws.send(
                json.dumps({"type": "input_audio_buffer.append", "audio": audio_b64})
            )

    def request_summary(self):
        """Ask for a summary of what was heard."""
        if self.ws and self.running:
            log_print(
                "INFO",
                "Requesting summary",
                session_id=self.session_id,
                last_context=self.last_context,
            )
            self.logger.log("summary_request", last_context=self.last_context)
            self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
            print("Requesting summary with context:", self.last_context)

            if self.last_context:
                prompt = f"""Previous context: "{self.last_context}"
Based on what you heard, is there anything NEW or different from the previous context?
- If nothing new: output exactly "..."
- If new topic: output 1 sentence (less than 7 words) describing what's new
Output ONLY the result, no other text."""
            else:
                prompt = "Summarize what you heard in 1 sentence (less than 7 words). Output ONLY the summary."

            self.ws.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "modalities": ["text"],
                            "instructions": prompt,
                        },
                    }
                )
            )

    def start(self):
        self.running = True
        self.logger.log("summary_client_start")
        self.ws = websocket.WebSocketApp(
            URL,
            header=[
                f"Authorization: Bearer {OPENAI_API_KEY}",
                "OpenAI-Beta: realtime=v1",
            ],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        self.running = False
        self.logger.log("summary_client_stop")
        if self.ws:
            self.ws.close()


client = None
summary_client = None


@app.route("/")
def index():
    log_print("INFO", "Index page requested")
    return render_template_string(HTML)


@sio.on("connect")
def handle_connect():
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    if session_id in session_loggers:
        session_loggers[session_id].log("client_disconnected")


@sio.on("start")
def handle_start(data):
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
    client.summary_client = summary_client  # Link for audio forwarding
    summary_client.start()
    client.start()


@sio.on("stop")
def handle_stop():
    global client
    session_id = request.sid
    log_print("INFO", "Stop requested", session_id=session_id)

    if client:
        client.stop()


@sio.on("request")
def handle_request():
    global client
    session_id = request.sid
    log_print("INFO", "Manual request triggered", session_id=session_id)

    if client and client.running:
        client.request()
    else:
        log_print("WARN", "Request ignored - no running client", session_id=session_id)


@sio.on("request_summary")
def handle_request_summary():
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


if __name__ == "__main__":
    log_print("INFO", "=" * 50)
    log_print("INFO", "Starting web_realtime server")
    log_print("INFO", f"Log directory: {LOG_DIR}")
    log_print("INFO", "=" * 50)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
