"""Whisper transcription SocketIO event handlers."""

import base64
import os
import tempfile
import threading
from datetime import datetime, timedelta

import numpy as np
import whisper
from flask import request
from flask_socketio import emit
from pydub import AudioSegment

from handlers.search import process_pending_search

# Global Whisper model (lazy loaded)
whisper_model = None


def get_whisper_model():
    """Get or load the Whisper model."""
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model("tiny")
    return whisper_model


def process_whisper_background(
    audio_data, file_format, session_id, socketio, transcription_sessions
):
    """Background thread for Whisper processing."""
    global whisper_model
    temp_path = None
    audio_received_time = datetime.utcnow()

    try:
        if not audio_data or len(audio_data) < 100:
            print(
                f"[Whisper] Audio data too small: {len(audio_data)} bytes - skipping silently"
            )
            return

        extension_map = {
            "webm": ".webm",
            "wav": ".wav",
            "mp3": ".mp3",
            "mp4": ".mp4",
            "m4a": ".m4a",
            "ogg": ".ogg",
            "flac": ".flac",
        }

        suffix = extension_map.get(file_format, ".webm")

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name

        if whisper_model is None:
            socketio.emit(
                "status", {"message": "Loading Whisper model..."}, room=session_id
            )
            whisper_model = whisper.load_model("tiny")

        socketio.emit(
            "status", {"message": "Transcribing with Whisper..."}, room=session_id
        )

        transcription_start_time = datetime.utcnow()
        result = whisper_model.transcribe(temp_path)
        text = result["text"].strip()

        transcription_end_time = datetime.utcnow()
        transcription_duration = (
            transcription_end_time - transcription_start_time
        ).total_seconds()
        total_duration = (transcription_end_time - audio_received_time).total_seconds()

        if text:
            session = transcription_sessions[session_id]
            session.add_text(text)

            session._log_event(
                "whisper_transcription",
                {
                    "text": text,
                    "word_count": len(text.split()),
                    "transcription_duration_seconds": transcription_duration,
                    "total_processing_seconds": total_duration,
                    "audio_size_bytes": len(audio_data),
                    "format": file_format,
                },
            )

            socketio.emit(
                "transcription", {"text": text, "source": "whisper"}, room=session_id
            )

            # Check for pending search and process it
            if session.pending_search:
                print("[Whisper] Pending search detected, processing...")
                process_pending_search(session, session_id, socketio)
        else:
            socketio.emit(
                "status", {"message": "No speech detected in audio"}, room=session_id
            )

    except RuntimeError as e:
        error_msg = str(e)
        if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
            print("[Whisper] Empty or invalid audio data - skipping silently")
        elif "Linear(in_features=" in error_msg or "out_features=" in error_msg:
            print(f"[Whisper] Model or audio format error: {error_msg}")
            socketio.emit(
                "error",
                {
                    "message": "Audio format incompatible. Please try recording again or use a different audio source."
                },
                room=session_id,
            )
        else:
            print(f"Whisper RuntimeError: {error_msg}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Whisper error: {error_msg}"}, room=session_id
            )

    except ValueError as e:
        error_msg = str(e)
        print(f"[Whisper] Value error (likely audio format issue): {error_msg}")
        socketio.emit(
            "error",
            {"message": "Audio format error. Please try a different recording format."},
            room=session_id,
        )

    except Exception as e:
        print(f"Whisper error: {str(e)}")
        import traceback

        traceback.print_exc()
        socketio.emit("error", {"message": f"Whisper error: {str(e)}"}, room=session_id)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


def register_whisper_handlers(socketio, transcription_sessions):
    """Register Whisper-related event handlers."""

    @socketio.on("start_whisper")
    def handle_start_whisper():
        """Initialize Whisper model."""
        global whisper_model
        if whisper_model is None:
            emit("status", {"message": "Loading Whisper model..."})
            whisper_model = whisper.load_model("tiny")
            emit("status", {"message": "Whisper model loaded (tiny)"})
        else:
            emit("status", {"message": "Whisper already loaded"})

    @socketio.on("audio_chunk_whisper")
    def handle_audio_chunk_whisper(data):
        """Process audio chunk with Whisper."""
        session_id = request.sid

        try:
            # Mark Whisper as actively streaming
            if session_id in transcription_sessions:
                transcription_sessions[session_id].whisper_active = True

            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")

            emit(
                "status",
                {"message": f"Received audio ({len(audio_data)} bytes). Processing..."},
            )

            thread = threading.Thread(
                target=process_whisper_background,
                args=(
                    audio_data,
                    file_format,
                    session_id,
                    socketio,
                    transcription_sessions,
                ),
            )
            thread.daemon = True
            thread.start()

        except Exception as e:
            print(f"Whisper handler error: {str(e)}")
            import traceback

            traceback.print_exc()
            emit("error", {"message": f"Whisper error: {str(e)}"})

    @socketio.on("audio_chunk_realtime")
    def handle_audio_chunk_realtime(data):
        """Process audio chunk with real-time accumulation."""
        global whisper_model
        session_id = request.sid

        try:
            audio_received_time = datetime.utcnow()
            audio_data = base64.b64decode(data["audio"])
            file_format = data.get("format", "webm")
            phrase_timeout = data.get("phrase_timeout", 3)
            is_final = data.get("is_final", False)

            session = transcription_sessions[session_id]
            session.whisper_active = True  # Mark Whisper as actively streaming
            now = datetime.utcnow()

            phrase_complete = False
            if session.phrase_time and now - session.phrase_time > timedelta(
                seconds=phrase_timeout
            ):
                session.phrase_bytes = b""
                phrase_complete = True

            session.phrase_time = now

            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=f".{file_format}", delete=False
                ) as temp_audio:
                    temp_audio.write(audio_data)
                    temp_path = temp_audio.name

                audio = AudioSegment.from_file(temp_path, format=file_format)
                audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
                raw_data = audio.raw_data

                session.phrase_bytes += raw_data

                if len(session.phrase_bytes) < 100:
                    print(
                        f"[Real-time] Accumulated audio too small: {len(session.phrase_bytes)} bytes, skipping"
                    )
                    return

                audio_np = (
                    np.frombuffer(session.phrase_bytes, dtype=np.int16).astype(
                        np.float32
                    )
                    / 32768.0
                )

                if len(audio_np) == 0:
                    print("[Real-time] Audio array is empty, skipping transcription")
                    return

                if whisper_model is None:
                    socketio.emit(
                        "status",
                        {"message": "Loading Whisper model..."},
                        room=session_id,
                    )
                    whisper_model = whisper.load_model("tiny")

                socketio.emit("status", {"message": "Transcribing..."}, room=session_id)

                transcription_start_time = datetime.utcnow()

                try:
                    result = whisper_model.transcribe(audio_np, fp16=False)
                    text = result["text"].strip()

                    transcription_end_time = datetime.utcnow()
                    transcription_duration = (
                        transcription_end_time - transcription_start_time
                    ).total_seconds()
                    total_duration = (
                        transcription_end_time - audio_received_time
                    ).total_seconds()

                except RuntimeError as whisper_error:
                    error_msg = str(whisper_error)
                    if (
                        "cannot reshape tensor" in error_msg
                        or "0 elements" in error_msg
                    ):
                        print(f"[Real-time] Empty or invalid audio: {error_msg}")
                        socketio.emit(
                            "status",
                            {"message": "Audio too short, waiting for more..."},
                            room=session_id,
                        )
                        return
                    elif (
                        "Linear(in_features=" in error_msg
                        or "out_features=" in error_msg
                    ):
                        print(f"[Real-time] Model or audio format error: {error_msg}")
                        socketio.emit(
                            "error",
                            {
                                "message": "Audio format incompatible. Please try a different recording method."
                            },
                            room=session_id,
                        )
                        session.phrase_bytes = b""
                        return
                    else:
                        raise
                except ValueError as whisper_error:
                    error_msg = str(whisper_error)
                    print(f"[Real-time] Value error: {error_msg}")
                    socketio.emit(
                        "error",
                        {"message": "Audio format error. Please try recording again."},
                        room=session_id,
                    )
                    session.phrase_bytes = b""
                    return

                if text:
                    if phrase_complete or is_final:
                        session.transcription_lines.append(text)
                        session.add_text(text)

                        session._log_event(
                            "whisper_transcription",
                            {
                                "text": text,
                                "word_count": len(text.split()),
                                "transcription_duration_seconds": transcription_duration,
                                "total_processing_seconds": total_duration,
                                "mode": "realtime",
                                "is_complete": True,
                                "phrase_complete": phrase_complete,
                                "is_final": is_final,
                            },
                        )

                        socketio.emit(
                            "transcription",
                            {
                                "text": text,
                                "source": "whisper-realtime",
                                "is_complete": True,
                            },
                            room=session_id,
                        )

                        # Check for pending search and process it
                        if session.pending_search:
                            print("[Whisper Real-time] Pending search detected, processing...")
                            process_pending_search(session, session_id, socketio)
                    else:
                        session.transcription_lines[-1] = text
                        print(
                            f"[Real-time] INCOMPLETE phrase (not added to session): '{text}'"
                        )
                        socketio.emit(
                            "transcription",
                            {
                                "text": text,
                                "source": "whisper-realtime",
                                "is_complete": False,
                            },
                            room=session_id,
                        )

            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            print(f"Real-time handler error: {str(e)}")
            import traceback

            traceback.print_exc()
            socketio.emit(
                "error", {"message": f"Real-time error: {str(e)}"}, room=session_id
            )
