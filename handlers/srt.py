"""SRT file SocketIO event handlers."""

import os

from flask import request
from flask_socketio import emit

from src.core.session import TranscriptionSession
from src.core.srt_parser import parse_srt


def register_srt_handlers(socketio, transcription_sessions):
    """Register SRT-related event handlers."""

    @socketio.on("clear_srt")
    def handle_clear_srt():
        """Clear SRT data from session."""
        session_id = request.sid

        if session_id in transcription_sessions:
            session = transcription_sessions[session_id]
            if hasattr(session, "srt_entries"):
                del session.srt_entries
            if hasattr(session, "last_srt_text"):
                del session.last_srt_text
            session._log_event("srt_cleared", {})
            print(f"[SRT] Cleared SRT data for session {session_id}")

    @socketio.on("check_srt_for_media")
    def handle_check_srt_for_media(data):
        """Check if matching SRT file exists for uploaded media file."""
        session_id = request.sid
        media_filename = data.get("filename", "")

        if not media_filename:
            return

        base_name = os.path.splitext(media_filename)[0]

        srt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "srt")
        srt_path = None

        for root, dirs, files in os.walk(srt_dir):
            for file in files:
                if file.endswith(".srt"):
                    file_base = os.path.splitext(file)[0]
                    if file_base == base_name or file_base.startswith(base_name):
                        srt_path = os.path.join(root, file)
                        break
            if srt_path:
                break

        if srt_path and os.path.exists(srt_path):
            try:
                with open(srt_path, "r", encoding="utf-8") as f:
                    srt_content = f.read()

                entries = parse_srt(srt_content)
                print(
                    f"[SRT] Auto-loaded matching SRT: {srt_path} ({len(entries)} entries)"
                )

                if session_id not in transcription_sessions:
                    transcription_sessions[session_id] = TranscriptionSession(
                        session_id
                    )

                session = transcription_sessions[session_id]
                session.srt_entries = entries
                session.srt_index = 0

                session._log_event(
                    "srt_auto_loaded",
                    {
                        "media_file": media_filename,
                        "srt_file": os.path.basename(srt_path),
                        "entry_count": len(entries),
                    },
                )

                emit(
                    "srt_loaded",
                    {
                        "count": len(entries),
                        "status": f"Auto-loaded SRT: {os.path.basename(srt_path)}",
                        "auto": True,
                    },
                )
            except Exception as e:
                print(f"[SRT] Error auto-loading SRT: {e}")
        else:
            print(f"[SRT] No matching SRT found for: {media_filename}")
            emit("srt_not_found", {"filename": media_filename})

    @socketio.on("load_srt")
    def handle_load_srt(data):
        """Load transcription from SRT file content."""
        session_id = request.sid
        srt_content = data.get("content", "")

        if session_id not in transcription_sessions:
            transcription_sessions[session_id] = TranscriptionSession(session_id)

        session = transcription_sessions[session_id]

        try:
            entries = parse_srt(srt_content)
            print(f"[SRT] Parsed {len(entries)} subtitle entries")

            session.srt_entries = entries
            session.srt_index = 0

            session._log_event(
                "srt_loaded",
                {
                    "entry_count": len(entries),
                    "total_duration_ms": entries[-1][1] if entries else 0,
                },
            )

            emit("srt_loaded", {"count": len(entries), "status": "SRT file loaded"})
        except Exception as e:
            print(f"[SRT] Error parsing SRT: {e}")
            emit("error", {"message": f"SRT parse error: {str(e)}"})

    @socketio.on("srt_time_update")
    def handle_srt_time_update(data):
        """Send transcription based on current video time."""
        session_id = request.sid
        current_time_ms = data.get("time_ms", 0)

        if session_id not in transcription_sessions:
            return

        session = transcription_sessions[session_id]

        if not hasattr(session, "srt_entries") or not session.srt_entries:
            return

        for start_ms, end_ms, text in session.srt_entries:
            if start_ms <= current_time_ms <= end_ms:
                if (
                    not hasattr(session, "last_srt_text")
                    or session.last_srt_text != text
                ):
                    session.last_srt_text = text
                    session.add_text(text)

                    session._log_event(
                        "srt_transcription",
                        {
                            "video_time_ms": current_time_ms,
                            "subtitle_start_ms": start_ms,
                            "subtitle_end_ms": end_ms,
                            "text": text,
                            "word_count": len(text.split()),
                        },
                    )

                    emit(
                        "transcription",
                        {"text": text, "source": "srt", "is_complete": True},
                    )
                break
