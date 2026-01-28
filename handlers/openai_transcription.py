"""OpenAI realtime transcription SocketIO event handlers.

Based on transcription-openai.py - maintains persistent WebSocket connection
and streams audio chunks as they arrive.
"""

import asyncio
import base64
import json
import os
import struct
import subprocess
import tempfile
import threading
from typing import List

import nest_asyncio
import numpy as np
import websockets
from flask import request
from flask_socketio import emit
from handlers.search import process_pending_search

from src.core.config import OPENAI_API_KEY

# Enable nested asyncio
nest_asyncio.apply()

# OpenAI Realtime API configuration (from transcription-openai.py)
MODEL_NAME = "gpt-4o-transcribe"
TARGET_SR = 24_000
RT_URL = "wss://api.openai.com/v1/realtime?intent=transcription"

EV_DELTA = "conversation.item.input_audio_transcription.delta"
EV_DONE = "conversation.item.input_audio_transcription.completed"

# Store OpenAI live sessions (like Gemini)
openai_live_sessions = {}


# ── helpers (from transcription-openai.py) ──────────────────────────────────
def float_to_16bit_pcm(float32_array):
    clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
    pcm16 = b"".join(struct.pack("<h", int(x * 32767)) for x in clipped)
    return pcm16


def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    return base64.b64encode(pcm_bytes).decode("ascii")


def convert_webm_to_pcm(audio_data: bytes, file_format: str) -> np.ndarray:
    """Convert any audio format to float32 PCM array using ffmpeg."""
    temp_input = None
    temp_output = None
    try:
        suffix_map = {
            "webm": ".webm",
            "wav": ".wav",
            "mp3": ".mp3",
            "mp4": ".mp4",
            "m4a": ".m4a",
            "ogg": ".ogg",
            "flac": ".flac",
        }
        suffix = suffix_map.get(file_format, ".webm")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_data)
            temp_input = f.name

        with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as f:
            temp_output = f.name

        # Convert to raw PCM: mono, 24kHz, 16-bit signed little-endian
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                temp_input,
                "-ar",
                str(TARGET_SR),
                "-ac",
                "1",
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                temp_output,
            ],
            capture_output=True,
            check=True,
        )

        # Read raw PCM and convert to float32
        with open(temp_output, "rb") as f:
            raw_data = f.read()

        samples = np.frombuffer(raw_data, dtype=np.int16)
        return samples.astype(np.float32) / 32768.0

    except subprocess.CalledProcessError as e:
        print(f"[OpenAI] ffmpeg error: {e.stderr.decode() if e.stderr else e}")
        raise
    finally:
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
        if temp_output and os.path.exists(temp_output):
            os.unlink(temp_output)


def _session_config(model: str, vad: float = 0.5) -> dict:
    """Session config (from transcription-openai.py)."""
    return {
        "type": "transcription_session.update",
        "session": {
            "input_audio_format": "pcm16",
            "turn_detection": {"type": "server_vad", "threshold": vad},
            "input_audio_transcription": {"model": model},
        },
    }


def run_openai_live_loop(session_id, socketio, transcription_sessions):
    """Background thread that runs the OpenAI Realtime async event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def openai_session_handler():
        """Main async handler for OpenAI Realtime session."""
        if not OPENAI_API_KEY:
            socketio.emit(
                "error",
                {"message": "OpenAI API key not configured"},
                room=session_id,
            )
            return

        session = transcription_sessions.get(session_id)
        if not session:
            return

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1",
        }

        try:
            print(f"[OpenAI] Connecting for session {session_id}...")
            socketio.emit(
                "status",
                {"message": "Connecting to OpenAI Realtime..."},
                room=session_id,
            )

            async with websockets.connect(
                RT_URL, additional_headers=headers, max_size=None
            ) as ws:
                print(f"[OpenAI] Connected for session {session_id}")

                # Send session config
                await ws.send(json.dumps(_session_config(MODEL_NAME)))

                # Store session state
                session.openai_ws = ws
                session.openai_active = True
                session.openai_audio_queue = asyncio.Queue()
                session.last_audio_timestamp = None  # Track last audio chunk timestamp

                socketio.emit(
                    "status",
                    {"message": "OpenAI Realtime connected. Start speaking."},
                    room=session_id,
                )

                async def send_audio():
                    """Send audio chunks from queue to OpenAI."""
                    while session.openai_active:
                        try:
                            # Queue contains (pcm_float, timestamp) tuples
                            item = await asyncio.wait_for(
                                session.openai_audio_queue.get(), timeout=1.0
                            )
                            pcm_float, audio_timestamp = item

                            # Update last audio timestamp
                            session.last_audio_timestamp = audio_timestamp
                            # print("[OpenAI] Last audio timestamp:", audio_timestamp)

                            # import time
                            # from datetime import datetime

                            # # 현재 시간 (unix timestamp, seconds)
                            # current_ts = time.time()
                            # print("Current timestamp (unix):", current_ts)

                            # # OpenAI audio timestamp (ISO 8601, UTC) → unix timestamp
                            # openai_dt = datetime.fromisoformat(
                            #     audio_timestamp.replace("Z", "+00:00")
                            # )
                            # openai_ts = openai_dt.timestamp()

                            # print("OpenAI timestamp (unix):", openai_ts)

                            # # 차이 계산
                            # diff_sec = current_ts - openai_ts
                            # print(
                            #     f"Time difference: {diff_sec:.3f} seconds ({diff_sec*1000:.1f} ms)"
                            # )

                            # Send as base64 encoded PCM16
                            payload = {
                                "type": "input_audio_buffer.append",
                                "audio": base64_encode_audio(pcm_float),
                            }
                            await ws.send(json.dumps(payload))
                        except asyncio.TimeoutError:
                            continue
                        except websockets.ConnectionClosed:
                            print("[OpenAI] Connection closed during send")
                            break
                        except Exception as e:
                            print(f"[OpenAI] Send error: {e}")
                            break

                async def receive_transcripts():
                    """Receive transcription events from OpenAI."""
                    current: List[str] = []

                    try:
                        async for msg in ws:
                            if not session.openai_active:
                                break

                            ev = json.loads(msg)
                            typ = ev.get("type")

                            if typ == EV_DELTA:
                                delta = ev.get("delta")
                                if delta:
                                    current.append(delta)
                                    # Emit partial transcription
                                    partial_text = "".join(current)
                                    socketio.emit(
                                        "transcription",
                                        {
                                            "text": partial_text,
                                            "source": "openai-realtime",
                                            "is_complete": False,
                                        },
                                        room=session_id,
                                    )

                            elif typ == EV_DONE:
                                # Sentence complete
                                text = "".join(current).strip()
                                current.clear()

                                if text:
                                    print(f"[OpenAI] Transcription: {text}")
                                    session.add_text(text)

                                    session._log_event(
                                        "openai_transcription",
                                        {
                                            "text": text,
                                            "word_count": len(text.split()),
                                            "last_audio_timestamp": session.last_audio_timestamp,
                                        },
                                    )

                                    socketio.emit(
                                        "transcription",
                                        {
                                            "text": text,
                                            "source": "openai",
                                            "is_complete": True,
                                        },
                                        room=session_id,
                                    )

                                    # Check pending search - wait for 2 EV_DONEs after spacebar
                                    if session.pending_search:
                                        ev_done_count = session.pending_search.get("ev_done_count", 0) + 1
                                        session.pending_search["ev_done_count"] = ev_done_count
                                        print(
                                            f"[OpenAI] Pending search: EV_DONE {ev_done_count}/2"
                                        )
                                        if ev_done_count >= 2:
                                            print("[OpenAI] 2 EV_DONEs received, processing search")
                                            process_pending_search(
                                                session, session_id, socketio
                                            )

                            elif typ == "error":
                                error_msg = ev.get("error", {}).get(
                                    "message", "Unknown"
                                )
                                print(f"[OpenAI] Server error: {error_msg}")
                                socketio.emit(
                                    "error",
                                    {"message": f"OpenAI: {error_msg}"},
                                    room=session_id,
                                )

                    except websockets.ConnectionClosed:
                        print("[OpenAI] Connection closed")
                    except Exception as e:
                        print(f"[OpenAI] Receive error: {e}")

                    # Flush remaining
                    if current:
                        text = "".join(current).strip()
                        if text:
                            session.add_text(text)
                            socketio.emit(
                                "transcription",
                                {"text": text, "source": "openai", "is_complete": True},
                                room=session_id,
                            )

                # Run send and receive concurrently
                await asyncio.gather(send_audio(), receive_transcripts())

        except Exception as e:
            print(f"[OpenAI] Session error: {e}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error",
                {"message": f"OpenAI connection error: {str(e)}"},
                room=session_id,
            )
        finally:
            session.openai_active = False
            session.openai_ws = None
            if session_id in openai_live_sessions:
                del openai_live_sessions[session_id]
            socketio.emit(
                "status",
                {"message": "OpenAI Realtime disconnected"},
                room=session_id,
            )

    try:
        loop.run_until_complete(openai_session_handler())
    finally:
        loop.close()


def register_openai_transcription_handlers(socketio, transcription_sessions):
    """Register OpenAI transcription event handlers."""

    @socketio.on("start_openai_transcription")
    def handle_start_openai_transcription():
        """Start OpenAI Realtime session."""
        session_id = request.sid

        if not OPENAI_API_KEY:
            emit("error", {"message": "OpenAI API key not configured"})
            return

        # Check if already connected
        if session_id in openai_live_sessions:
            emit("status", {"message": "OpenAI already connected"})
            return

        # Start background thread for OpenAI session
        thread = threading.Thread(
            target=run_openai_live_loop,
            args=(session_id, socketio, transcription_sessions),
        )
        thread.daemon = True
        thread.start()

        openai_live_sessions[session_id] = thread
        emit("status", {"message": "Starting OpenAI Realtime..."})

    @socketio.on("stop_openai_transcription")
    def handle_stop_openai_transcription():
        """Stop OpenAI Realtime session."""
        session_id = request.sid
        session = transcription_sessions.get(session_id)

        if session:
            session.openai_active = False

        if session_id in openai_live_sessions:
            del openai_live_sessions[session_id]

        emit("status", {"message": "OpenAI Realtime stopped"})

    @socketio.on("audio_chunk_openai")
    def handle_audio_chunk_openai(data):
        """Process audio chunk - convert and queue for OpenAI."""
        session_id = request.sid
        session = transcription_sessions.get(session_id)

        if not session:
            emit("error", {"message": "No active session"})
            return

        if not session.openai_active or not hasattr(session, "openai_audio_queue"):
            # Auto-start if not connected
            if session_id not in openai_live_sessions:
                handle_start_openai_transcription()
                emit("status", {"message": "Auto-starting OpenAI connection..."})
            return

        try:
            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")
            audio_timestamp = data.get(
                "timestamp"
            )  # Client timestamp when audio was recorded

            # Convert to float32 PCM
            pcm_float = convert_webm_to_pcm(audio_data, file_format)

            if len(pcm_float) > 0:
                # Queue the audio with timestamp for sending
                try:
                    session.openai_audio_queue.put_nowait((pcm_float, audio_timestamp))
                except asyncio.QueueFull:
                    print("[OpenAI] Audio queue full, dropping chunk")

        except Exception as e:
            print(f"[OpenAI] Audio processing error: {e}")
            import traceback

            traceback.print_exc()
            emit("error", {"message": f"Audio error: {str(e)}"})
