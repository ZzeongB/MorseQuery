import base64
import json
import os
import tempfile
import threading
from datetime import datetime

import nltk
import numpy as np
import pandas as pd
import requests
import whisper
from dotenv import load_dotenv
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from google.cloud import speech
from nltk.stem import WordNetLemmatizer
from pydub import AudioSegment

load_dotenv()

# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Initialize NLTK lemmatizer
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    print("[NLTK] Downloading WordNet data...")
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

lemmatizer = WordNetLemmatizer()

# Common English contractions
CONTRACTIONS = {
    "you're": "you are",
    "we're": "we are",
    "they're": "they are",
    "i'm": "i am",
    "he's": "he is",
    "she's": "she is",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "there's": "there is",
    "who's": "who is",
    "where's": "where is",
    "won't": "will not",
    "can't": "cannot",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "wouldn't": "would not",
    "shouldn't": "should not",
    "couldn't": "could not",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = "morsequery-secret-key"
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    ping_timeout=120,
    ping_interval=25,
    max_http_buffer_size=100000000,
)

# Initialize Whisper model (base model for speed)
whisper_model = None

# Google Custom Search API config
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_SEARCH_ENGINE_ID = os.getenv("GOOGLE_CUSTOM_SEARCH_ENGINE_ID")

# Load OpenLexicon for keyword extraction
lexicon_df = None
lexicon_dict = {}


def load_lexicon():
    """Load OpenLexicon.xlsx and create lookup dictionary"""
    global lexicon_df, lexicon_dict
    try:
        lexicon_path = os.path.join(
            os.path.dirname(__file__), "data/lexicon/OpenLexicon.xlsx"
        )
        print(f"Loading OpenLexicon from {lexicon_path}...")
        lexicon_df = pd.read_excel(lexicon_path)

        # Create dictionary for faster lookup: {word: LgSUBTLWF_value}
        lexicon_dict = {}
        for _, row in lexicon_df.iterrows():
            word = str(row["ortho"]).lower()
            freq_value = row["English_Lexicon_Project__LgSUBTLWF"]
            lexicon_dict[word] = freq_value

        print(f"Lexicon loaded: {len(lexicon_dict)} words")
    except Exception as e:
        print(f"Warning: Could not load OpenLexicon.xlsx: {e}")
        print("Keyword extraction will use fallback method")


# Load lexicon on startup
load_lexicon()

# Store transcription history per session
transcription_sessions = {}


def preprocess_word(word):
    """Preprocess word for lexicon lookup

    1. Expand contractions (you're -> you are)
    2. Lemmatize to base form (declares -> declare, remains -> remain)
    3. Return the most searchable form

    Args:
        word: Original word from transcription

    Returns:
        Preprocessed word in base form, or None if should be skipped
    """
    # Remove punctuation and lowercase
    cleaned = "".join(c for c in word if c.isalnum()).lower()

    if not cleaned or len(cleaned) < 3:
        return None

    # Expand contractions first
    if cleaned in CONTRACTIONS:
        # Return the first word of the expansion
        # e.g., "you're" -> "you are" -> "you"
        expanded = CONTRACTIONS[cleaned].split()[0]
        cleaned = expanded

    # Lemmatize: try as verb, then noun, then adjective
    # Verbs: declares -> declare, remains -> remain
    lemma_v = lemmatizer.lemmatize(cleaned, pos="v")
    if lemma_v != cleaned:
        return lemma_v

    # Nouns: cats -> cat
    lemma_n = lemmatizer.lemmatize(cleaned, pos="n")
    if lemma_n != cleaned:
        return lemma_n

    # Adjectives: better -> good
    lemma_a = lemmatizer.lemmatize(cleaned, pos="a")
    if lemma_a != cleaned:
        return lemma_a

    # No lemmatization needed, return as-is
    return cleaned


class TranscriptionSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.words = []
        self.word_timestamps = []  # Track when each word was added
        self.full_text = ""
        self.sentences = []
        # Real-time transcription state (like transcribe_demo.py)
        self.phrase_bytes = b""
        self.phrase_time = None
        self.transcription_lines = [""]

        # Logging system
        self.session_start_time = datetime.utcnow()
        self.event_log = []  # List of all events (words added, searches performed)

        # Log session start
        self._log_event(
            "session_start",
            {
                "session_id": session_id,
                "start_time": self.session_start_time.isoformat(),
            },
        )

    def _log_event(self, event_type, data):
        """Internal method to log events"""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "data": data,
        }
        self.event_log.append(event)
        print(f"[LOG] {event_type}: {data}")

    def add_text(self, text):
        if text.strip():
            from datetime import datetime

            self.full_text += " " + text
            self.sentences.append(text)

            # Add words with timestamps
            new_words = text.split()
            current_time = datetime.utcnow()
            for word in new_words:
                self.words.append(word)
                self.word_timestamps.append(current_time)

                # Log each word addition
                # self._log_event(
                #     "word_added",
                #     {
                #         "word": word,
                #         "timestamp": current_time.isoformat(),
                #         "word_index": len(self.words) - 1,
                #     },
                # )

    def get_top_keyword(self, context_window=20):
        """Extract most relevant keyword using OpenLexicon frequency filtering

        Args:
            context_window: Number of recent words to consider (default: 20)

        Returns:
            The most important keyword (lowest frequency or not in lexicon)
        """
        if len(self.words) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        # Get recent words (spacebar 근처 단어들)
        recent_words = (
            self.words[-context_window:]
            if len(self.words) > context_window
            else self.words
        )

        # Clean words and check against lexicon
        candidate_words = []

        for word in recent_words:
            # Preprocess word (expand contractions, lemmatize)
            cleaned = preprocess_word(word)

            # Skip if preprocessing returned None
            if not cleaned:
                continue

            # Check lexicon frequency (using preprocessed form)
            freq_value = lexicon_dict.get(cleaned)
            print(f"[Lexicon] '{word}' -> preprocessed: '{cleaned}'")

            # Keep words that are:
            # 1. Not in lexicon (freq_value is None)
            # 2. Have NaN frequency (pd.isna)
            # 3. Have frequency < 3.0 (rare words)
            if freq_value is None:
                # Not in lexicon - likely important/rare
                candidate_words.append(
                    (cleaned, -1, word)
                )  # -1 for sorting (highest priority)
                print(f"[Lexicon] '{cleaned}' not in lexicon -> candidate")
            elif pd.isna(freq_value):
                # NaN frequency - keep as candidate
                candidate_words.append((cleaned, 0, word))  # 0 for sorting
                print(f"[Lexicon] '{cleaned}' has NaN frequency -> candidate")
            elif freq_value < 3.0:
                # Low frequency (rare word) - keep as candidate
                candidate_words.append((cleaned, freq_value, word))
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} < 3.0 -> candidate")
            else:
                # High frequency (common word) - skip
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} >= 3.0 -> skipped")

        if not candidate_words:
            # No candidates found - fallback to last word
            fallback_word = self.words[-1] if self.words else ""
            print(
                f"[Lexicon] ⚠️ No important words found (all freq >= 3.0), using fallback: '{fallback_word}'"
            )
            return fallback_word

        # Sort by frequency (lowest first, with not-in-lexicon/-1 first)
        candidate_words.sort(key=lambda x: x[1])

        # Return the rarest word (or most recent if tied)
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        print(f"\n[Lexicon] Selected keyword: '{selected_word}' (freq={selected_freq})")
        print(
            f"[Lexicon] All candidates: {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        return selected_word

    def get_top_keyword_with_time_threshold(self, time_threshold=5):
        """Extract most important keyword from words within the last N seconds

        Finds the word with HIGHEST importance (LOWEST frequency) among words
        from the last N seconds that have OpenLexicon frequency < 3.0

        Args:
            time_threshold: Number of seconds to look back (default: 5)

        Returns:
            The most important keyword (lowest frequency) within the time window
        """
        from datetime import datetime, timedelta

        if len(self.words) == 0 or len(self.word_timestamps) == 0:
            return ""

        if len(self.words) == 1:
            return self.words[0]

        # Get current time and calculate threshold
        current_time = datetime.utcnow()
        threshold_time = current_time - timedelta(seconds=time_threshold)

        # Filter words within the time threshold
        recent_words = []
        for word, timestamp in zip(self.words, self.word_timestamps):
            if timestamp >= threshold_time:
                recent_words.append(word)

        if not recent_words:
            # No recent words - fallback to last word
            print(
                f"[Recent] No words found in last {time_threshold}s, using last word as fallback"
            )
            return self.words[-1] if self.words else ""

        print(
            f"\n[Recent] Found {len(recent_words)} words in last {time_threshold}s: {recent_words}"
        )

        # Only keep words with frequency < 3.0 (high importance threshold)
        candidate_words = []

        for word in recent_words:
            # Preprocess word (expand contractions, lemmatize)
            cleaned = preprocess_word(word)

            # Skip if preprocessing returned None
            if not cleaned:
                continue

            # Check lexicon frequency (using preprocessed form)
            freq_value = lexicon_dict.get(cleaned)
            print(f"[Recent] '{word}' -> preprocessed: '{cleaned}'")

            # Only keep words with importance threshold below 3.0:
            # 1. Not in lexicon (freq_value is None) -> highest importance
            # 2. Have NaN frequency (pd.isna) -> high importance
            # 3. Have frequency < 3.0 (rare words) -> high importance
            if freq_value is None:
                # Not in lexicon - highest importance
                candidate_words.append((cleaned, -1, word))
                print(
                    f"[Recent] '{cleaned}' not in lexicon (highest importance) -> candidate"
                )
            elif pd.isna(freq_value):
                # NaN frequency - very high importance
                candidate_words.append((cleaned, 0, word))
                print(
                    f"[Recent] '{cleaned}' has NaN frequency (very high importance) -> candidate"
                )
            elif freq_value < 3.0:
                # Low frequency = high importance
                candidate_words.append((cleaned, freq_value, word))
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} < 3.0 (high importance) -> candidate"
                )
            else:
                # High frequency (common word) - skip
                print(
                    f"[Recent] '{cleaned}' freq={freq_value:.3f} >= 3.0 (low importance) -> skipped"
                )

        if not candidate_words:
            # No high-importance words found - fallback to last recent word
            fallback_word = recent_words[-1] if recent_words else ""
            print(
                f"[Recent] ⚠️ No important words found (all freq >= 3.0) in last {time_threshold}s"
            )
            print(f"[Recent] Using fallback: '{fallback_word}'")

            # Log the analysis even when using fallback
            self._log_event(
                "keyword_extraction_recent",
                {
                    "time_threshold_seconds": time_threshold,
                    "recent_words": recent_words,
                    "candidates": [],
                    "selected_keyword": fallback_word,
                    "selected_frequency": None,
                    "selection_reason": "fallback_no_important_words",
                    "is_fallback": True,
                },
            )

            return fallback_word

        # Sort by frequency: LOWEST first = HIGHEST importance first
        # -1 (not in lexicon) < 0 (NaN) < 0.x (rare) < 3.0 (common, already filtered out)
        candidate_words.sort(key=lambda x: x[1])

        # Return the word with HIGHEST importance (LOWEST frequency)
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        freq_label = (
            "not in lexicon" if selected_freq == -1 else f"freq={selected_freq}"
        )

        print(
            f"\n[Recent] ✓ Selected HIGHEST importance keyword: '{selected_word}' ({freq_label}) from last {time_threshold}s"
        )
        print(
            f"[Recent] All candidates (sorted by importance): {[(w, f) for w, f, _ in candidate_words[:5]]}"
        )

        # Log the keyword extraction analysis
        self._log_event(
            "keyword_extraction_recent",
            {
                "time_threshold_seconds": time_threshold,
                "recent_words": recent_words,
                "candidates": [
                    {
                        "word": word,
                        "cleaned": cleaned,
                        "frequency": freq,
                        "importance_rank": idx + 1,
                    }
                    for idx, (cleaned, freq, word) in enumerate(candidate_words)
                ],
                "selected_keyword": selected_word,
                "selected_frequency": selected_freq,
                "selection_reason": "highest_importance",
                "is_fallback": False,
            },
        )

        return selected_word

    def log_search_action(self, search_mode, search_type, keyword, num_results=0):
        """Log when user performs a search (spacebar action)"""
        self._log_event(
            "search_action",
            {
                "search_mode": search_mode,
                "search_type": search_type,
                "keyword": keyword,
                "num_results": num_results,
                "total_words_at_search": len(self.words),
                "transcription_length": len(self.full_text),
            },
        )

    def save_logs_to_file(self):
        """Save session logs to a JSON file"""
        try:
            timestamp = self.session_start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"
            filepath = os.path.join(LOGS_DIR, filename)

            log_data = {
                "session_id": self.session_id,
                "session_start": self.session_start_time.isoformat(),
                "session_end": datetime.utcnow().isoformat(),
                "total_words": len(self.words),
                "total_events": len(self.event_log),
                "full_transcription": self.full_text,
                "events": self.event_log,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)

            print(f"[LOG] Session logs saved to: {filepath}")
            return filepath
        except Exception as e:
            print(f"[LOG] Error saving logs: {e}")
            return None


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def handle_connect():
    session_id = request.sid
    transcription_sessions[session_id] = TranscriptionSession(session_id)
    emit("connected", {"status": "Connected to MorseQuery"})
    print(f"[Session] New session connected: {session_id}")


@socketio.on("disconnect")
def handle_disconnect():
    session_id = request.sid
    if session_id in transcription_sessions:
        # Save logs before deleting session
        session = transcription_sessions[session_id]
        session._log_event(
            "session_end",
            {
                "total_words": len(session.words),
                "total_searches": sum(
                    1 for e in session.event_log if e["event_type"] == "search_action"
                ),
            },
        )
        log_file = session.save_logs_to_file()
        print(
            f"[Session] Session disconnected: {session_id}, logs saved to: {log_file}"
        )
        del transcription_sessions[session_id]


@socketio.on("start_whisper")
def handle_start_whisper():
    """Initialize Whisper model"""
    global whisper_model
    if whisper_model is None:
        emit("status", {"message": "Loading Whisper model..."})
        whisper_model = whisper.load_model("base")
        emit("status", {"message": "Whisper model loaded"})
    else:
        emit("status", {"message": "Whisper already loaded"})


def process_whisper_background(audio_data, file_format, session_id):
    """Background thread for Whisper processing"""
    global whisper_model
    temp_path = None

    try:
        # Check if audio data is valid
        if not audio_data or len(audio_data) < 100:
            print(
                f"[Whisper] Audio data too small: {len(audio_data)} bytes - skipping silently"
            )
            # Silently skip without showing error to user
            return

        # Map common formats to extensions
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

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name

        print(f"Processing audio file: {temp_path}, size: {len(audio_data)} bytes")

        # Transcribe with Whisper
        if whisper_model is None:
            socketio.emit(
                "status", {"message": "Loading Whisper model..."}, room=session_id
            )
            whisper_model = whisper.load_model("base")

        socketio.emit(
            "status", {"message": "Transcribing with Whisper..."}, room=session_id
        )
        result = whisper_model.transcribe(temp_path)
        text = result["text"].strip()

        print(f"Transcription result: {text}")

        if text:
            transcription_sessions[session_id].add_text(text)
            socketio.emit(
                "transcription", {"text": text, "source": "whisper"}, room=session_id
            )
        else:
            socketio.emit(
                "status", {"message": "No speech detected in audio"}, room=session_id
            )

    except RuntimeError as e:
        # Handle specific tensor reshape errors (empty/invalid audio)
        error_msg = str(e)
        if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
            print("[Whisper] Empty or invalid audio data - skipping silently")
            # Silently skip without showing error to user
            # This is normal in real-time recording scenarios
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
        # Handle audio format/conversion errors
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
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@socketio.on("audio_chunk_whisper")
def handle_audio_chunk_whisper(data):
    """Process audio chunk with Whisper"""
    session_id = request.sid

    try:
        print("\n" + "=" * 50)
        print("[Whisper] Received audio chunk request")
        print(f"Session ID: {session_id}")

        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])

        # Detect file type from data or use provided format
        file_format = data.get("format", "webm")
        print(f"Format: {file_format}, Data size: {len(audio_data)} bytes")

        emit(
            "status",
            {"message": f"Received audio ({len(audio_data)} bytes). Processing..."},
        )

        # Process in background thread to avoid blocking WebSocket
        thread = threading.Thread(
            target=process_whisper_background,
            args=(audio_data, file_format, session_id),
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
    """Process audio chunk with real-time accumulation (like transcribe_demo.py)"""
    session_id = request.sid

    try:
        from datetime import datetime, timedelta

        print("\n" + "=" * 50)
        print("[Real-time] Received audio chunk")
        print(f"Session ID: {session_id}")

        # Decode base64 audio data
        audio_data = base64.b64decode(data["audio"])
        file_format = data.get("format", "webm")
        phrase_timeout = data.get("phrase_timeout", 3)  # seconds
        is_final = data.get("is_final", False)

        print(
            f"Format: {file_format}, Data size: {len(audio_data)} bytes, is_final: {is_final}"
        )

        session = transcription_sessions[session_id]
        now = datetime.utcnow()

        # Check if this is a new phrase (phrase_timeout logic from demo)
        phrase_complete = False
        if session.phrase_time and now - session.phrase_time > timedelta(
            seconds=phrase_timeout
        ):
            session.phrase_bytes = b""
            phrase_complete = True

        session.phrase_time = now

        # Convert audio to raw PCM format
        temp_path = None
        try:
            # Save to temporary file for conversion
            with tempfile.NamedTemporaryFile(
                suffix=f".{file_format}", delete=False
            ) as temp_audio:
                temp_audio.write(audio_data)
                temp_path = temp_audio.name

            # Load audio and convert to raw PCM
            audio = AudioSegment.from_file(temp_path, format=file_format)
            # Convert to mono, 16kHz, 16-bit PCM (like demo)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            raw_data = audio.raw_data

            # Accumulate audio data
            session.phrase_bytes += raw_data

            print(f"Accumulated audio size: {len(session.phrase_bytes)} bytes")

            # Check if accumulated audio is valid
            if len(session.phrase_bytes) < 100:
                print(
                    f"[Real-time] Accumulated audio too small: {len(session.phrase_bytes)} bytes, skipping transcription"
                )
                return

            # Convert to numpy array for Whisper
            audio_np = (
                np.frombuffer(session.phrase_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )

            # Check if audio array is valid
            if len(audio_np) == 0:
                print("[Real-time] Audio array is empty, skipping transcription")
                return

            # Transcribe with Whisper
            global whisper_model
            if whisper_model is None:
                socketio.emit(
                    "status", {"message": "Loading Whisper model..."}, room=session_id
                )
                whisper_model = whisper.load_model("base")

            socketio.emit("status", {"message": "Transcribing..."}, room=session_id)

            try:
                result = whisper_model.transcribe(audio_np, fp16=False)
                text = result["text"].strip()
                print(f"Transcription: {text}")
            except RuntimeError as whisper_error:
                error_msg = str(whisper_error)
                if "cannot reshape tensor" in error_msg or "0 elements" in error_msg:
                    print(f"[Real-time] Empty or invalid audio: {error_msg}")
                    socketio.emit(
                        "status",
                        {"message": "Audio too short, waiting for more..."},
                        room=session_id,
                    )
                    return
                elif "Linear(in_features=" in error_msg or "out_features=" in error_msg:
                    print(f"[Real-time] Model or audio format error: {error_msg}")
                    socketio.emit(
                        "error",
                        {
                            "message": "Audio format incompatible. Please try a different recording method."
                        },
                        room=session_id,
                    )
                    # Reset phrase bytes to recover
                    session.phrase_bytes = b""
                    return
                else:
                    raise  # Re-raise if it's a different RuntimeError
            except ValueError as whisper_error:
                error_msg = str(whisper_error)
                print(f"[Real-time] Value error: {error_msg}")
                socketio.emit(
                    "error",
                    {"message": "Audio format error. Please try recording again."},
                    room=session_id,
                )
                # Reset phrase bytes to recover
                session.phrase_bytes = b""
                return

            if text:
                # Update or append transcription (like demo)
                if phrase_complete or is_final:
                    session.transcription_lines.append(text)
                    session.add_text(text)
                    socketio.emit(
                        "transcription",
                        {
                            "text": text,
                            "source": "whisper-realtime",
                            "is_complete": True,
                        },
                        room=session_id,
                    )
                else:
                    session.transcription_lines[-1] = text
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


@socketio.on("start_google_streaming")
def handle_start_google_streaming():
    """Start Google Speech-to-Text streaming"""
    emit("status", {"message": "Google Speech-to-Text ready"})


@socketio.on("audio_chunk_google")
def handle_audio_chunk_google(data):
    """Process audio chunk with Google Speech-to-Text"""
    session_id = request.sid

    try:
        client = speech.SpeechClient()

        # Decode audio
        audio_data = base64.b64decode(data["audio"])
        file_format = data.get("format", "webm")

        print(
            f"Processing Google STT: format={file_format}, size={len(audio_data)} bytes"
        )

        # Map format to Google encoding
        encoding_map = {
            "webm": speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            "mp3": speech.RecognitionConfig.AudioEncoding.MP3,
            "flac": speech.RecognitionConfig.AudioEncoding.FLAC,
            "wav": speech.RecognitionConfig.AudioEncoding.LINEAR16,
            "m4a": speech.RecognitionConfig.AudioEncoding.MP3,  # Treat m4a as MP3
            "mp4": speech.RecognitionConfig.AudioEncoding.MP3,
            "ogg": speech.RecognitionConfig.AudioEncoding.OGG_OPUS,
        }

        encoding = encoding_map.get(
            file_format, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS
        )

        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=48000,
            language_code="en-US",
            enable_automatic_punctuation=True,
        )

        emit("status", {"message": "Transcribing with Google STT..."})
        response = client.recognize(config=config, audio=audio)

        transcribed = False
        for result in response.results:
            text = result.alternatives[0].transcript
            if text:
                transcription_sessions[session_id].add_text(text)
                emit("transcription", {"text": text, "source": "google"})
                transcribed = True
                print(f"Google STT result: {text}")

        if not transcribed:
            emit("status", {"message": "No speech detected in audio"})

    except Exception as e:
        print(f"Google STT error: {str(e)}")
        import traceback

        traceback.print_exc()
        emit("error", {"message": f"Google STT error: {str(e)}"})


@socketio.on("search_request")
def handle_search_request(data):
    """Handle search request triggered by spacebar"""
    session_id = request.sid
    search_mode = data.get("mode", "tfidf")  # 'instant', 'recent', or 'tfidf'
    search_type = data.get("type", "text")  # 'text' or 'image'

    print(f"\n[Search Request] Mode: {search_mode}, Type: {search_type}")

    if session_id not in transcription_sessions:
        emit("error", {"message": "No active session"})
        return

    # Get keyword based on search mode
    if search_mode == "instant":
        # Use the keyword provided by client (last word)
        keyword = data.get("keyword", "")
        print(f"[Instant Search] Using client-provided keyword: {keyword}")
    elif search_mode == "recent":
        # Recent mode: important word from last 5 seconds
        time_threshold = data.get("time_threshold", 5)  # default 5 seconds
        keyword = transcription_sessions[
            session_id
        ].get_top_keyword_with_time_threshold(time_threshold)
        print(
            f"[Recent Search] Calculated keyword from last {time_threshold}s: {keyword}"
        )
    else:
        # TF-IDF mode: calculate important keyword
        keyword = transcription_sessions[session_id].get_top_keyword()
        print(f"[TF-IDF Search] Calculated keyword: {keyword}")

    if not keyword:
        emit("error", {"message": "No keywords available for search"})
        return

    # Clean the keyword (remove punctuation)
    keyword_clean = "".join(c for c in keyword if c.isalnum() or c.isspace()).strip()

    emit("search_keyword", {"keyword": keyword_clean, "mode": search_mode})

    # Perform Google Custom Search
    try:
        search_results = google_custom_search(keyword_clean, search_type)

        # Calculate number of results
        if search_type == "both" and isinstance(search_results, dict):
            num_results = len(search_results.get("text", [])) + len(
                search_results.get("image", [])
            )
        else:
            num_results = len(search_results) if isinstance(search_results, list) else 0

        # Log the search action
        transcription_sessions[session_id].log_search_action(
            search_mode=search_mode,
            search_type=search_type,
            keyword=keyword_clean,
            num_results=num_results,
        )

        emit(
            "search_results",
            {
                "keyword": keyword_clean,
                "mode": search_mode,
                "type": search_type,
                "results": search_results,
            },
        )
        print(f"[Search] Found {num_results} results for '{keyword_clean}'")
    except Exception as e:
        print(f"Search error: {str(e)}")
        emit("error", {"message": f"Search error: {str(e)}"})


def google_custom_search(query, search_type="text"):
    """Perform Google Custom Search"""
    url = "https://www.googleapis.com/customsearch/v1"

    # Handle 'both' search type by making two separate API calls
    if search_type == "both":
        text_results = google_custom_search(query, "text")
        image_results = google_custom_search(query, "image")
        return {"text": text_results, "image": image_results}

    params = {
        "key": GOOGLE_SEARCH_API_KEY,
        "cx": GOOGLE_SEARCH_ENGINE_ID,
        "q": query,
        "num": 5,
    }

    if search_type == "image":
        params["searchType"] = "image"

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    results = []

    if "items" in data:
        for item in data["items"]:
            if search_type == "image":
                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "thumbnail": item.get("image", {}).get("thumbnailLink", ""),
                        "context": item.get("snippet", ""),
                    }
                )
            else:
                # Extract image from pagemap if available
                pagemap = item.get("pagemap", {})
                image_url = None

                # Try cse_image first, then cse_thumbnail, then metatags og:image
                if "cse_image" in pagemap and len(pagemap["cse_image"]) > 0:
                    image_url = pagemap["cse_image"][0].get("src", "")
                elif "cse_thumbnail" in pagemap and len(pagemap["cse_thumbnail"]) > 0:
                    image_url = pagemap["cse_thumbnail"][0].get("src", "")
                elif "metatags" in pagemap and len(pagemap["metatags"]) > 0:
                    image_url = pagemap["metatags"][0].get("og:image", "")

                results.append(
                    {
                        "title": item.get("title", ""),
                        "link": item.get("link", ""),
                        "snippet": item.get("snippet", ""),
                        "image": image_url,  # Add image from pagemap
                    }
                )

    return results


@socketio.on("clear_session")
def handle_clear_session():
    """Clear transcription session"""
    session_id = request.sid

    # Save logs from old session before clearing
    if session_id in transcription_sessions:
        old_session = transcription_sessions[session_id]
        old_session._log_event(
            "session_cleared",
            {
                "total_words": len(old_session.words),
                "total_searches": sum(
                    1
                    for e in old_session.event_log
                    if e["event_type"] == "search_action"
                ),
            },
        )
        log_file = old_session.save_logs_to_file()
        print(f"[Session] Session cleared: {session_id}, logs saved to: {log_file}")

    # Create new session
    transcription_sessions[session_id] = TranscriptionSession(session_id)
    emit("session_cleared", {"status": "Session cleared"})


if __name__ == "__main__":
    socketio.run(app, debug=True, host="0.0.0.0", port=5001)
