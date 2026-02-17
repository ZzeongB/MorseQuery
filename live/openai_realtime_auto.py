# pip install websocket-client pydub pyaudio
import base64
import json
import os
import threading
import time

import pyaudio
import websocket
from pydub import AudioSegment

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
AUDIO_FILE = "../mp3/clips/U6fI3brP8V4_clip_885_1240.mp3"

# Audio config
RATE = 24000
CHUNK = 4800  # 200ms
AUTO_INTERVAL = 5  # Request every N seconds


class RealtimeClient:
    def __init__(self):
        self.ws = None
        self.running = True
        self.chunks_sent = 0

    def on_open(self, ws):
        print("Connected.")
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,
                        "instructions": "You are listening to the conversation/lecture. Listen carefully and identify difficult or unfamiliar English words. Always provide very concise information.",
                    },
                }
            )
        )
        threading.Thread(target=self.stream_audio, daemon=True).start()

    def on_message(self, ws, message):
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "response.text.delta":
            print(event.get("delta", ""), end="", flush=True)
        elif etype == "response.done":
            print("\n---")
        elif etype == "error":
            print(f"\n[Error]: {event.get('error', {}).get('message', '')}")

    def on_error(self, ws, error):
        print(f"Error: {error}")

    def on_close(self, ws, close_status, close_msg):
        print("Disconnected.")
        self.running = False

    def stream_audio(self):
        """Stream audio file to API + play locally + auto request"""
        print(f"Loading: {AUDIO_FILE}")
        audio = AudioSegment.from_file(AUDIO_FILE)
        audio = audio.set_frame_rate(RATE).set_channels(1).set_sample_width(2)
        raw = audio.raw_data

        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, output=True)

        print(
            f"Playing {len(audio)/1000:.1f}s... Auto-requesting every {AUTO_INTERVAL}s"
        )

        chunk_bytes = CHUNK * 2
        chunks_per_interval = int(AUTO_INTERVAL / 0.2)  # 0.2s per chunk

        for i in range(0, len(raw), chunk_bytes):
            if not self.running:
                break
            chunk = raw[i : i + chunk_bytes]
            stream.write(chunk)
            self.ws.send(
                json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode(),
                    }
                )
            )
            self.chunks_sent += 1

            # Auto request every interval
            if self.chunks_sent % chunks_per_interval == 0:
                self.request_description()

        # Final request at end
        self.request_description()

        stream.stop_stream()
        stream.close()
        pa.terminate()
        print("Done.")
        time.sleep(3)  # Wait for final response
        self.ws.close()

    def request_description(self):
        """Auto request description"""
        self.ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        self.ws.send(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "modalities": ["text"],
                        "instructions": """Be very concise. From the audio you just heard, pick the most interesting or less common word.
You MUST always give at least 1 word - pick the least common one even if it's not very difficult.
Reply ONLY in this format (1-3 words max):
word: short description (under 10 words)

Example:
leverage: use something to maximum advantage
nuance: subtle difference in meaning""",
                    },
                }
            )
        )
        print("\n[Auto-requesting...]")

    def run(self):
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
        self.ws.run_forever()


if __name__ == "__main__":
    RealtimeClient().run()
