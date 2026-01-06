"""
Flask backend for user study data collection.
Stores participant data server-side instead of relying on browser downloads.
"""

import json
import os
import uuid
from datetime import datetime

import whisper
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from pydub import AudioSegment

app = Flask(__name__)
CORS(app)

# Data directory
DATA_DIR = "logs/study_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Transcript directory
TRANSCRIPT_DIR = "data/transcripts"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

# Audio directory for uploaded files
AUDIO_DIR = "data/audio"
os.makedirs(AUDIO_DIR, exist_ok=True)

# Base directory for serving files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Initialize Whisper model (lazy loading)
whisper_model = None


def get_session_filepath(session_id):
    """Find session file by session_id (handles timestamp-sessionid format)"""
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(f"-{session_id}.json"):
            return os.path.join(DATA_DIR, filename)
    # Fallback to old format (without timestamp)
    old_format = os.path.join(DATA_DIR, f"{session_id}.json")
    if os.path.exists(old_format):
        return old_format
    return None

# YouTube video configuration
YOUTUBE_VIDEOS = [
    {
        "video_id": "U6fI3brP8V4",
        "start_time": 885,  # 14:45
        "end_time": 1240,  # 20:40
        "title": "Video 1",
    },
    {
        "video_id": "bAkuNXtgrLA",
        "start_time": 680,  # 11:20
        "end_time": 1010,  # 16:50
        "title": "Video 2",
    },
    {
        "video_id": "cEVAjm_ETtY",
        "start_time": 1721,  # 28:41
        "end_time": 2040,  # 34:00
        "title": "Video 3",
    },
    {
        "video_id": "MxovSnvSO4E",
        "start_time": 355,  # 5:55
        "end_time": 657,  # 10:57
        "title": "Video 4",
    },
]


# @app.route("/")
# def index():
#     """Serve Page 1: Data Collection"""
#     return send_from_directory(BASE_DIR, "templates/study/study_page1_backend.html")


# @app.route("/label")
# def label():
#     """Serve Page 2: Retrospective Labeling"""
#     return send_from_directory(BASE_DIR, "templates/study/study_page2_backend.html")


# @app.route("/youtube-instruction")
@app.route("/")
def youtube_instruction():
    """Serve YouTube Study Instructions"""
    return send_from_directory(
        BASE_DIR, "templates/study/study_instruction_youtube.html"
    )


@app.route("/youtube")
def youtube_study():
    """Serve YouTube Study Page 1: Data Collection"""
    return send_from_directory(BASE_DIR, "templates/study/study_page1_youtube.html")


@app.route("/youtube-label")
def youtube_label():
    """Serve YouTube Study Page 2: Retrospective Labeling"""
    return send_from_directory(BASE_DIR, "templates/study/study_page2_youtube.html")


@app.route("/api/youtube/videos", methods=["GET"])
def get_youtube_videos():
    """Get YouTube video configuration"""
    return jsonify({"videos": YOUTUBE_VIDEOS})


@app.route("/video/<path:filename>")
def video(filename):
    """Serve video files with proper MIME type"""
    video_path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404
    return send_file(video_path, mimetype="video/mp4")


@app.route("/api/session/create", methods=["POST"])
def create_session():
    """Create new participant session"""
    session_id = str(uuid.uuid4())
    timestamp = datetime.now()
    session_data = {
        "session_id": session_id,
        "created_at": timestamp.isoformat(),
        "gestures": [],
        "labeled_data": [],
    }

    # Save session with timestamp-sessionid format
    timestamp_str = timestamp.strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp_str}-{session_id}.json"
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"session_id": session_id})


@app.route("/api/gesture/log", methods=["POST"])
def log_gesture():
    """Log a gesture timestamp (Page 1)"""
    data = request.json
    session_id = data.get("session_id")
    gesture = {
        "gesture_timestamp": data.get("gesture_timestamp"),
        "video_time": data.get("video_time"),
        "video_id": data.get("video_id"),  # For YouTube videos
        "video_index": data.get("video_index"),  # Which video (0-3, -1 for tutorial)
        "is_tutorial": data.get("is_tutorial", False),
    }

    # Load session
    filepath = get_session_filepath(session_id)
    if not filepath:
        return jsonify({"error": "Session not found"}), 404

    with open(filepath, "r") as f:
        session_data = json.load(f)

    # Add gesture
    session_data["gestures"].append(gesture)

    # Save session
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"success": True})


@app.route("/api/gestures/get/<session_id>", methods=["GET"])
def get_gestures(session_id):
    """Get all gestures for a session (Page 2)"""
    filepath = get_session_filepath(session_id)

    if not filepath:
        return jsonify({"error": "Session not found"}), 404

    with open(filepath, "r") as f:
        session_data = json.load(f)

    return jsonify({"gestures": session_data["gestures"]})


@app.route("/api/label/submit", methods=["POST"])
def submit_label():
    """Submit labeled data for a gesture (Page 2)"""
    data = request.json
    session_id = data.get("session_id")

    # Load session
    filepath = get_session_filepath(session_id)
    if not filepath:
        return jsonify({"error": "Session not found"}), 404

    with open(filepath, "r") as f:
        session_data = json.load(f)

    # Add labeled data (only YouTube study fields)
    labeled_gesture = {
        "gesture_timestamp": data.get("gesture_timestamp"),
        "video_time": data.get("video_time"),
        "video_id": data.get("video_id"),
        "video_index": data.get("video_index"),
        "is_tutorial": data.get("is_tutorial", False),
        "target_source": data.get("target_source", ""),
        "target_word": data.get("target_word"),
        "target_word_timestamp": data.get("target_word_timestamp"),
        "target_words_discrete": data.get("target_words_discrete"),
        "selected_words_with_time": data.get("selected_words_with_time"),
        "pressed_timestamp": data.get("pressed_timestamp"),
        "intent_types": data.get("intent_types", []),
        "intent_other_text": data.get("intent_other_text", ""),
    }

    session_data["labeled_data"].append(labeled_gesture)
    session_data["updated_at"] = datetime.now().isoformat()

    # Save session
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"success": True})


@app.route("/api/session/complete/<session_id>", methods=["POST"])
def complete_session(session_id):
    """Mark session as complete and return final data"""
    filepath = get_session_filepath(session_id)

    if not filepath:
        return jsonify({"error": "Session not found"}), 404

    with open(filepath, "r") as f:
        session_data = json.load(f)

    session_data["completed_at"] = datetime.now().isoformat()

    # Save final session
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2)

    return jsonify({"labeled_data": session_data["labeled_data"]})


@app.route("/api/transcript/generate", methods=["POST"])
def generate_transcript():
    """Generate transcript from audio file using Whisper"""
    global whisper_model

    if "audio_file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio_file"]
    video_id = request.form.get("video_id")
    start_time = float(request.form.get("start_time", 0))
    end_time = float(request.form.get("end_time", 0))

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    try:
        # Save uploaded audio file
        audio_filename = f"{video_id}.mp3"
        audio_path = os.path.join(AUDIO_DIR, audio_filename)
        audio_file.save(audio_path)

        # Extract the specified time range
        print(f"Loading audio file: {audio_path}")
        audio = AudioSegment.from_file(audio_path)

        # Convert seconds to milliseconds for pydub
        start_ms = int(start_time * 1000)
        end_ms = int(end_time * 1000)

        print(f"Extracting segment: {start_time}s to {end_time}s")
        audio_segment = audio[start_ms:end_ms]

        # Save the segment
        segment_path = os.path.join(AUDIO_DIR, f"{video_id}_segment.mp3")
        audio_segment.export(segment_path, format="mp3")

        # Load Whisper model if not already loaded
        if whisper_model is None:
            print("Loading Whisper model...")
            whisper_model = whisper.load_model("base")

        # Transcribe the segment
        print("Transcribing with Whisper...")
        result = whisper_model.transcribe(segment_path, word_timestamps=True)

        # Format transcript data with word-level timestamps
        segments = []
        words = []

        for segment in result.get("segments", []):
            # Adjust timestamps to be relative to the original video start_time
            adjusted_start = segment["start"] + start_time
            adjusted_end = segment["end"] + start_time

            # Process word-level timestamps
            segment_words = []
            for word in segment.get("words", []):
                word_start = word["start"] + start_time
                word_end = word["end"] + start_time
                word_data = {
                    "word": word["word"],
                    "start": word_start,
                    "end": word_end,
                    "start_formatted": format_timestamp(word_start),
                    "end_formatted": format_timestamp(word_end),
                }
                segment_words.append(word_data)
                words.append(word_data)

            segments.append(
                {
                    "start": adjusted_start,
                    "end": adjusted_end,
                    "start_formatted": format_timestamp(adjusted_start),
                    "end_formatted": format_timestamp(adjusted_end),
                    "text": segment["text"].strip(),
                    "words": segment_words,
                }
            )

        # Save transcript to JSON
        transcript_data = {
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "segments": segments,
            "words": words,
            "full_text": result.get("text", ""),
        }

        transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f"Transcript saved to: {transcript_path}")

        return jsonify(
            {
                "success": True,
                "message": "Transcript generated successfully",
                "transcript_path": transcript_path,
                "segments_count": len(segments),
            }
        )

    except Exception as e:
        print(f"Error generating transcript: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def format_timestamp(seconds):
    """Format seconds to MM:SS"""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


@app.route("/api/transcript/get", methods=["GET"])
def get_transcript():
    """Get transcript segments for a time range"""
    video_id = request.args.get("video_id")
    start_time = float(request.args.get("start", 0))
    end_time = float(request.args.get("end", 60))

    if not video_id:
        return jsonify({"error": "video_id is required"}), 400

    # Load transcript JSON for this video
    transcript_path = os.path.join(TRANSCRIPT_DIR, f"{video_id}.json")

    if not os.path.exists(transcript_path):
        return jsonify({"error": "Transcript not found"}), 404

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript_data = json.load(f)

    # Filter segments within the time range
    segments = []
    for segment in transcript_data.get("segments", []):
        seg_start = segment.get("start", 0)
        seg_end = segment.get("end", 0)

        # Include segment if it overlaps with the requested time range
        if seg_start <= end_time and seg_end >= start_time:
            segments.append(
                {
                    "start": seg_start,
                    "end": seg_end,
                    "start_formatted": segment.get("start_formatted", ""),
                    "end_formatted": segment.get("end_formatted", ""),
                    "text": segment.get("text", ""),
                    "words": segment.get("words", []),
                }
            )

    return jsonify({"segments": segments})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
    app.run(host="0.0.0.0", port=5002, debug=True)
