# pip install flask flask-socketio websocket-client pydub pyaudio
import base64
import json
import os
import threading

import pyaudio
import websocket
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from pydub import AudioSegment

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")

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
        #box { display: none; width: 50%; padding: 20px; }
        .option { border: 1px solid #333; padding: 12px 16px; margin: 6px 0; font-size: 16px; border-radius: 8px; transition: all 0.2s; }
        .option.active { border-color: #4ade80; background: #1a1a1a; }
        .option .word { font-weight: 600; color: #4ade80; }
        .option .desc { color: #aaa; margin-left: 8px; }
    </style>
</head>
<body>
    <div id="menu">
        <button onclick="start('manual')">Manual</button>
        <button onclick="start('auto')">Auto (5s)</button>
    </div>
    <div id="box"></div>
    <script>
        const socket = io();
        let mode = 'manual';
        let options = [];
        let currentIdx = 0;
        let buffer = '';
        let lastSpace = 0;

        socket.on('word', data => { buffer += data; });
        socket.on('done', () => {
            parseOptions();
            render();
        });
        socket.on('clear', () => {
            buffer = '';
            options = [];
            currentIdx = 0;
            document.getElementById('box').innerHTML = '';
        });

        function parseOptions() {
            options = buffer.split('\\n').filter(l => l.includes(':')).map(l => {
                const [word, ...desc] = l.split(':');
                return { word: word.trim(), desc: desc.join(':').trim() };
            });
            currentIdx = 0;
        }

        function render() {
            const box = document.getElementById('box');
            box.innerHTML = options.map((o, i) =>
                `<div class="option ${i === currentIdx ? 'active' : ''}">
                    <span class="word">${o.word}</span><span class="desc">${o.desc}</span>
                </div>`
            ).join('');
        }

        function start(m) {
            mode = m;
            document.getElementById('menu').style.display = 'none';
            document.getElementById('box').style.display = 'block';
            socket.emit('start', m);
        }

        document.addEventListener('keydown', e => {
            if (e.code !== 'Space') return;
            e.preventDefault();

            const now = Date.now();
            if (now - lastSpace < 300) {
                // Double space - new request
                socket.emit('request');
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
    def __init__(self, socketio, mode="manual"):
        self.sio = socketio
        self.mode = mode
        self.ws = None
        self.running = False
        self.chunks_sent = 0

    def on_open(self, ws):
        self.sio.emit("status", "Connected")
        ws.send(json.dumps({
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "input_audio_format": "pcm16",
                "turn_detection": None,
                "instructions": "Listen to audio. Identify difficult words. Be very concise.",
            },
        }))
        threading.Thread(target=self.stream_audio, daemon=True).start()

    def on_message(self, ws, message):
        event = json.loads(message)
        etype = event.get("type", "")
        if etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.sio.emit("word", delta)
        elif etype == "response.done":
            self.sio.emit("done")

    def on_error(self, ws, error):
        self.sio.emit("status", f"Error: {error}")

    def on_close(self, ws, status, msg):
        self.sio.emit("status", "Disconnected")
        self.running = False

    def stream_audio(self):
        audio = AudioSegment.from_file(AUDIO_FILE)
        audio = audio.set_frame_rate(RATE).set_channels(1).set_sample_width(2)
        raw = audio.raw_data

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)
        self.sio.emit("status", f"Playing... ({self.mode} mode)")

        chunk_bytes = CHUNK * 2
        chunks_per_interval = int(AUTO_INTERVAL / 0.2)

        for i in range(0, len(raw), chunk_bytes):
            if not self.running:
                break
            chunk = raw[i:i + chunk_bytes]
            stream.write(chunk)
            self.ws.send(json.dumps({
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(chunk).decode(),
            }))
            self.chunks_sent += 1

            if self.mode == "auto" and self.chunks_sent % chunks_per_interval == 0:
                self.request()

        if self.running:
            self.request()
        stream.stop_stream()
        stream.close()
        pa.terminate()
        self.sio.emit("status", "Done")

    def request(self):
        self.sio.emit("clear")
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self.ws.send(json.dumps({
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": """Be very concise. Pick 3-5 interesting/less common words from the audio.
ONLY use words you actually heard. Do NOT make up or hallucinate words.
If no interesting words, output nothing.
One word per line. Format:
word: short description (under 10 words)""",
            },
        }))

    def start(self):
        self.running = True
        self.chunks_sent = 0
        self.ws = websocket.WebSocketApp(
            URL,
            header=[f"Authorization: Bearer {OPENAI_API_KEY}", "OpenAI-Beta: realtime=v1"],
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
        )
        threading.Thread(target=self.ws.run_forever, daemon=True).start()

    def stop(self):
        self.running = False
        if self.ws:
            self.ws.close()


client = None


@app.route("/")
def index():
    return render_template_string(HTML)


@sio.on("start")
def handle_start(mode):
    global client
    if client:
        client.stop()
    client = RealtimeClient(sio, mode)
    client.start()


@sio.on("stop")
def handle_stop():
    global client
    if client:
        client.stop()


@sio.on("request")
def handle_request():
    global client
    if client and client.running:
        client.request()


if __name__ == "__main__":
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
