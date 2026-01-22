"""SocketIO event handlers for MorseQuery."""

from handlers.connection import register_connection_handlers
from handlers.gemini import register_gemini_handlers
from handlers.search import register_search_handlers
from handlers.srt import register_srt_handlers
from handlers.whisper import register_whisper_handlers


def register_all_handlers(socketio, transcription_sessions):
    """Register all SocketIO event handlers.

    Args:
        socketio: Flask-SocketIO instance
        transcription_sessions: Dict to store session data
    """
    register_connection_handlers(socketio, transcription_sessions)
    register_whisper_handlers(socketio, transcription_sessions)
    register_gemini_handlers(socketio, transcription_sessions)
    register_search_handlers(socketio, transcription_sessions)
    register_srt_handlers(socketio, transcription_sessions)
