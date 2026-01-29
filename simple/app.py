"""MorseQuery Simple - Simplified real-time audio streaming search.

Audio sources: Microphone, File Upload, YouTube Link
Speech recognition: OpenAI Realtime only
Search mode: GPT 4o mini only
Options: Search results disabled, Show all keywords disabled, Longpress grounding enabled
"""

from flask import Flask, render_template
from flask_socketio import SocketIO

from handlers import register_all_handlers

# Initialize Flask app
app = Flask(__name__)
app.config["SECRET_KEY"] = "morsequery-simple-secret-key"

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,
    ping_interval=25,
    max_http_buffer_size=100000000,
)

# Store transcription sessions
transcription_sessions = {}

# Register all SocketIO event handlers
register_all_handlers(socketio, transcription_sessions)


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5002)
