"""SocketIO event handlers for MorseQuery Simple."""

from handlers.connection import register_connection_handlers
from handlers.openai_transcription import register_openai_transcription_handlers
from handlers.search import register_search_handlers
from handlers.auto_inference import register_auto_inference_handlers
from handlers.context_manager import register_context_manager_handlers


def register_all_handlers(socketio, transcription_sessions):
    """Register all SocketIO event handlers.

    Args:
        socketio: Flask-SocketIO instance
        transcription_sessions: Dict to store session data
    """
    register_connection_handlers(socketio, transcription_sessions)
    register_openai_transcription_handlers(socketio, transcription_sessions)
    register_search_handlers(socketio, transcription_sessions)
    register_auto_inference_handlers(socketio, transcription_sessions)
    register_context_manager_handlers(socketio, transcription_sessions)
