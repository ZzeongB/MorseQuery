"""Gemini Live API SocketIO event handlers."""

import asyncio
import base64
import os
import tempfile
import threading

from flask import request
from flask_socketio import emit
from google import genai
from pydub import AudioSegment

from src.core.config import (
    GEMINI_MODEL,
    GEMINI_SEND_SAMPLE_RATE,
    GEMINI_TRANSCRIPTION_CONFIG,
    GOOGLE_API_KEY,
)
from src.core.session import TranscriptionSession
from handlers.search import process_pending_search

# Initialize Gemini client
gemini_client = None
if GOOGLE_API_KEY:
    gemini_client = genai.Client(
        http_options={"api_version": "v1beta"}, api_key=GOOGLE_API_KEY
    )
    print("[Gemini] Client initialized")
else:
    print("[Gemini] Warning: GOOGLE_API_KEY not set, Gemini Live will not work")

# Store Gemini Live sessions
gemini_live_sessions = {}


def run_gemini_live_loop(session_id, socketio, transcription_sessions):
    """Background thread that runs the Gemini Live async event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def gemini_session_handler():
        """Main async handler for Gemini Live session."""
        if not gemini_client:
            socketio.emit(
                "error",
                {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."},
                room=session_id,
            )
            return

        session = transcription_sessions.get(session_id)
        if not session:
            return

        try:
            print(f"[Gemini] Attempting to connect for session {session_id}...")
            socketio.emit(
                "status", {"message": "Connecting to Gemini Live..."}, room=session_id
            )

            async with gemini_client.aio.live.connect(
                model=GEMINI_MODEL, config=GEMINI_TRANSCRIPTION_CONFIG
            ) as gemini_session:
                print(f"[Gemini] Connected successfully for session {session_id}")
                session.gemini_session = gemini_session
                session.gemini_active = True
                session.gemini_audio_queue = asyncio.Queue(maxsize=10)

                socketio.emit(
                    "status",
                    {"message": "Gemini Live connected. Start speaking or play audio."},
                    room=session_id,
                )
                socketio.emit(
                    "gemini_connected", {"status": "connected"}, room=session_id
                )

                async def send_audio():
                    """Send audio chunks from queue to Gemini."""
                    while session.gemini_active:
                        try:
                            audio_data = await asyncio.wait_for(
                                session.gemini_audio_queue.get(), timeout=1.0
                            )
                            await gemini_session.send(input=audio_data)
                        except asyncio.TimeoutError:
                            continue
                        except Exception as e:
                            print(f"[Gemini] Send error: {e}")
                            break

                async def receive_responses():
                    """Receive and process responses from Gemini."""
                    while session.gemini_active:
                        try:
                            turn = gemini_session.receive()
                            async for response in turn:
                                text = None

                                # Method 1: direct .text attribute
                                text = getattr(response, "text", None)

                                # Method 2: check for parts
                                if not text:
                                    parts = getattr(response, "parts", None)
                                    if parts:
                                        for part in parts:
                                            part_text = getattr(part, "text", None)
                                            if part_text:
                                                text = (text or "") + part_text

                                # Method 3: check for content
                                if not text:
                                    content = getattr(response, "content", None)
                                    if content:
                                        text = str(content)

                                # Method 4: check server_content (Live API specific)
                                if not text:
                                    server_content = getattr(
                                        response, "server_content", None
                                    )
                                    if server_content:
                                        model_turn = getattr(
                                            server_content, "model_turn", None
                                        )
                                        if model_turn:
                                            parts = getattr(model_turn, "parts", None)
                                            if parts:
                                                for part in parts:
                                                    part_text = getattr(
                                                        part, "text", None
                                                    )
                                                    if part_text:
                                                        text = (text or "") + part_text

                                if text:
                                    text = text.strip()
                                    if text:
                                        print(f"[Gemini] Transcription: {text}")

                                        # Add to session (for GPT keyword extraction)
                                        session.add_text(text)

                                        # Emit as standard transcription (like Whisper)
                                        socketio.emit(
                                            "transcription",
                                            {
                                                "text": text,
                                                "source": "gemini-live",
                                            },
                                            room=session_id,
                                        )

                                        # Log transcription
                                        session._log_event(
                                            "gemini_transcription",
                                            {"text": text, "word_count": len(text.split())},
                                        )

                                        # Check for pending search and process it
                                        if session.pending_search:
                                            print(f"[Gemini] Pending search detected, processing...")
                                            process_pending_search(session, session_id, socketio)
                                else:
                                    data = getattr(response, "data", None)
                                    if data:
                                        print(
                                            f"[Gemini DEBUG] Got audio data: {len(data)} bytes"
                                        )

                        except Exception as e:
                            if session.gemini_active:
                                print(f"[Gemini] Receive error: {e}")
                                import traceback

                                traceback.print_exc()
                            break

                send_task = asyncio.create_task(send_audio())
                receive_task = asyncio.create_task(receive_responses())

                while session.gemini_active:
                    await asyncio.sleep(0.5)

                send_task.cancel()
                receive_task.cancel()

        except Exception as e:
            print(f"[Gemini] Session error: {e}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Gemini error: {str(e)}"}, room=session_id
            )
        finally:
            session.gemini_active = False
            session.gemini_session = None
            socketio.emit(
                "gemini_disconnected", {"status": "disconnected"}, room=session_id
            )

    try:
        loop.run_until_complete(gemini_session_handler())
    finally:
        loop.close()


def register_gemini_handlers(socketio, transcription_sessions):
    """Register Gemini-related event handlers."""

    @socketio.on("start_gemini_live")
    def handle_start_gemini_live():
        """Start Gemini Live session for real-time transcription."""
        session_id = request.sid

        if not gemini_client:
            emit(
                "error",
                {"message": "Gemini client not initialized. Set GOOGLE_API_KEY."},
            )
            return

        if session_id not in transcription_sessions:
            transcription_sessions[session_id] = TranscriptionSession(session_id)

        session = transcription_sessions[session_id]

        if session.gemini_active:
            emit("status", {"message": "Gemini Live already active"})
            return

        # Reset Gemini state
        session.gemini_raw_output = ""
        session.gemini_parse_buffer = ""  # Reset buffer for new session
        session.gemini_captions = ""
        session.gemini_summary = {"overall_context": "", "current_segment": ""}
        session.gemini_terms = []
        session.gemini_recent_terms = []  # Reset recent terms
        session.gemini_recent_term_index = 0

        session._log_event("gemini_live_start", {"model": GEMINI_MODEL})

        emit("status", {"message": "Starting Gemini Live session..."})

        thread = threading.Thread(
            target=run_gemini_live_loop,
            args=(session_id, socketio, transcription_sessions),
        )
        thread.daemon = True
        thread.start()
        gemini_live_sessions[session_id] = thread

    @socketio.on("stop_gemini_live")
    def handle_stop_gemini_live():
        """Stop Gemini Live session."""
        session_id = request.sid

        if session_id in transcription_sessions:
            session = transcription_sessions[session_id]
            session.gemini_active = False

            session._log_event(
                "gemini_live_stop",
                {
                    "total_terms": len(session.gemini_terms_history),
                    "final_captions_length": len(session.gemini_captions),
                },
            )

        emit("status", {"message": "Gemini Live session stopped"})
        emit("gemini_disconnected", {"status": "disconnected"})

    @socketio.on("audio_chunk_gemini_live")
    def handle_audio_chunk_gemini_live(data):
        """Process audio chunk with Gemini Live."""
        session_id = request.sid

        if session_id not in transcription_sessions:
            emit("error", {"message": "No active session"})
            return

        session = transcription_sessions[session_id]

        if not session.gemini_active or not session.gemini_audio_queue:
            emit("error", {"message": "Gemini Live not active. Start Gemini first."})
            return

        try:
            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{file_format}", delete=False
                ) as temp_audio:
                    temp_audio.write(audio_data)
                    temp_path = temp_audio.name

                audio = AudioSegment.from_file(temp_path, format=file_format)
                audio = (
                    audio.set_channels(1)
                    .set_frame_rate(GEMINI_SEND_SAMPLE_RATE)
                    .set_sample_width(2)
                )
                raw_data = audio.raw_data

                audio_input = {"data": raw_data, "mime_type": "audio/pcm"}

                def add_to_queue():
                    try:
                        session.gemini_audio_queue.put_nowait(audio_input)
                    except asyncio.QueueFull:
                        print("[Gemini] Audio queue full, dropping chunk")

                if session.gemini_audio_queue:
                    add_to_queue()

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"[Gemini] Audio chunk error: {e}")
            import traceback

            traceback.print_exc()
            emit("error", {"message": f"Gemini audio error: {str(e)}"})
