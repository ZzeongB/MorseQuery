import os
import io
import base64
import tempfile
import threading
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import whisper
from google.cloud import speech
import numpy as np
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'morsequery-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading',
                    ping_timeout=120, ping_interval=25, max_http_buffer_size=100000000)

# Initialize Whisper model (base model for speed)
whisper_model = None

# Google Custom Search API config
GOOGLE_SEARCH_API_KEY = os.getenv('GOOGLE_CUSTOM_SEARCH_API_KEY')
GOOGLE_SEARCH_ENGINE_ID = os.getenv('GOOGLE_CUSTOM_SEARCH_ENGINE_ID')

# Load OpenLexicon for keyword extraction
lexicon_df = None
lexicon_dict = {}

def load_lexicon():
    """Load OpenLexicon.xlsx and create lookup dictionary"""
    global lexicon_df, lexicon_dict
    try:
        lexicon_path = os.path.join(os.path.dirname(__file__), 'OpenLexicon.xlsx')
        print(f"Loading OpenLexicon from {lexicon_path}...")
        lexicon_df = pd.read_excel(lexicon_path)

        # Create dictionary for faster lookup: {word: LgSUBTLWF_value}
        lexicon_dict = {}
        for _, row in lexicon_df.iterrows():
            word = str(row['ortho']).lower()
            freq_value = row['English_Lexicon_Project__LgSUBTLWF']
            lexicon_dict[word] = freq_value

        print(f"Lexicon loaded: {len(lexicon_dict)} words")
    except Exception as e:
        print(f"Warning: Could not load OpenLexicon.xlsx: {e}")
        print("Keyword extraction will use fallback method")

# Load lexicon on startup
load_lexicon()

# Store transcription history per session
transcription_sessions = {}


class TranscriptionSession:
    def __init__(self):
        self.words = []
        self.full_text = ""
        self.sentences = []
        # Real-time transcription state (like transcribe_demo.py)
        self.phrase_bytes = b''
        self.phrase_time = None
        self.transcription_lines = ['']

    def add_text(self, text):
        if text.strip():
            self.full_text += " " + text
            self.sentences.append(text)
            self.words.extend(text.split())

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
        recent_words = self.words[-context_window:] if len(self.words) > context_window else self.words

        # Clean words and check against lexicon
        candidate_words = []

        for word in recent_words:
            # Remove punctuation and lowercase
            cleaned = ''.join(c for c in word if c.isalnum()).lower()

            # Skip very short words
            if not cleaned or len(cleaned) < 3:
                continue

            # Check lexicon frequency
            freq_value = lexicon_dict.get(cleaned)

            # Keep words that are:
            # 1. Not in lexicon (freq_value is None)
            # 2. Have NaN frequency (pd.isna)
            # 3. Have frequency < 1.0 (rare words)
            if freq_value is None:
                # Not in lexicon - likely important/rare
                candidate_words.append((cleaned, -1, word))  # -1 for sorting (highest priority)
                print(f"[Lexicon] '{cleaned}' not in lexicon -> candidate")
            elif pd.isna(freq_value):
                # NaN frequency - keep as candidate
                candidate_words.append((cleaned, 0, word))  # 0 for sorting
                print(f"[Lexicon] '{cleaned}' has NaN frequency -> candidate")
            elif freq_value < 1.0:
                # Low frequency (rare word) - keep as candidate
                candidate_words.append((cleaned, freq_value, word))
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} < 1.0 -> candidate")
            else:
                # High frequency (common word) - skip
                print(f"[Lexicon] '{cleaned}' freq={freq_value:.3f} >= 1.0 -> skipped")

        if not candidate_words:
            # No candidates found - fallback to last word
            print("[Lexicon] No rare words found, using last word as fallback")
            return self.words[-1] if self.words else ""

        # Sort by frequency (lowest first, with not-in-lexicon/-1 first)
        candidate_words.sort(key=lambda x: x[1])

        # Return the rarest word (or most recent if tied)
        selected_word = candidate_words[0][0]
        selected_freq = candidate_words[0][1]

        print(f"\n[Lexicon] Selected keyword: '{selected_word}' (freq={selected_freq})")
        print(f"[Lexicon] All candidates: {[(w, f) for w, f, _ in candidate_words[:5]]}")

        return selected_word


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    session_id = request.sid
    transcription_sessions[session_id] = TranscriptionSession()
    emit('connected', {'status': 'Connected to MorseQuery'})


@socketio.on('disconnect')
def handle_disconnect():
    session_id = request.sid
    if session_id in transcription_sessions:
        del transcription_sessions[session_id]


@socketio.on('start_whisper')
def handle_start_whisper():
    """Initialize Whisper model"""
    global whisper_model
    if whisper_model is None:
        emit('status', {'message': 'Loading Whisper model...'})
        whisper_model = whisper.load_model("base")
        emit('status', {'message': 'Whisper model loaded'})
    else:
        emit('status', {'message': 'Whisper already loaded'})


def process_whisper_background(audio_data, file_format, session_id):
    """Background thread for Whisper processing"""
    global whisper_model
    temp_path = None

    try:
        # Map common formats to extensions
        extension_map = {
            'webm': '.webm',
            'wav': '.wav',
            'mp3': '.mp3',
            'mp4': '.mp4',
            'm4a': '.m4a',
            'ogg': '.ogg',
            'flac': '.flac'
        }

        suffix = extension_map.get(file_format, '.webm')

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_path = temp_audio.name

        print(f"Processing audio file: {temp_path}, size: {len(audio_data)} bytes")

        # Transcribe with Whisper
        if whisper_model is None:
            socketio.emit('status', {'message': 'Loading Whisper model...'}, room=session_id)
            whisper_model = whisper.load_model("base")

        socketio.emit('status', {'message': 'Transcribing with Whisper...'}, room=session_id)
        result = whisper_model.transcribe(temp_path)
        text = result['text'].strip()

        print(f"Transcription result: {text}")

        if text:
            transcription_sessions[session_id].add_text(text)
            socketio.emit('transcription', {'text': text, 'source': 'whisper'}, room=session_id)
        else:
            socketio.emit('status', {'message': 'No speech detected in audio'}, room=session_id)

    except Exception as e:
        print(f"Whisper error: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f'Whisper error: {str(e)}'}, room=session_id)

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)


@socketio.on('audio_chunk_whisper')
def handle_audio_chunk_whisper(data):
    """Process audio chunk with Whisper"""
    session_id = request.sid

    try:
        print("\n" + "="*50)
        print(f"[Whisper] Received audio chunk request")
        print(f"Session ID: {session_id}")

        # Decode base64 audio data
        audio_data = base64.b64decode(data['audio'])

        # Detect file type from data or use provided format
        file_format = data.get('format', 'webm')
        print(f"Format: {file_format}, Data size: {len(audio_data)} bytes")

        emit('status', {'message': f'Received audio ({len(audio_data)} bytes). Processing...'})

        # Process in background thread to avoid blocking WebSocket
        thread = threading.Thread(
            target=process_whisper_background,
            args=(audio_data, file_format, session_id)
        )
        thread.daemon = True
        thread.start()

    except Exception as e:
        print(f"Whisper handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Whisper error: {str(e)}'})


@socketio.on('audio_chunk_realtime')
def handle_audio_chunk_realtime(data):
    """Process audio chunk with real-time accumulation (like transcribe_demo.py)"""
    session_id = request.sid

    try:
        from datetime import datetime, timedelta
        from pydub import AudioSegment

        print("\n" + "="*50)
        print("[Real-time] Received audio chunk")
        print(f"Session ID: {session_id}")

        # Decode base64 audio data
        audio_data = base64.b64decode(data['audio'])
        file_format = data.get('format', 'webm')
        phrase_timeout = data.get('phrase_timeout', 3)  # seconds
        is_final = data.get('is_final', False)

        print(f"Format: {file_format}, Data size: {len(audio_data)} bytes, is_final: {is_final}")

        session = transcription_sessions[session_id]
        now = datetime.utcnow()

        # Check if this is a new phrase (phrase_timeout logic from demo)
        phrase_complete = False
        if session.phrase_time and now - session.phrase_time > timedelta(seconds=phrase_timeout):
            session.phrase_bytes = b''
            phrase_complete = True

        session.phrase_time = now

        # Convert audio to raw PCM format
        temp_path = None
        try:
            # Save to temporary file for conversion
            with tempfile.NamedTemporaryFile(suffix=f'.{file_format}', delete=False) as temp_audio:
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

            # Convert to numpy array for Whisper
            audio_np = np.frombuffer(session.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # Transcribe with Whisper
            global whisper_model
            if whisper_model is None:
                socketio.emit('status', {'message': 'Loading Whisper model...'}, room=session_id)
                whisper_model = whisper.load_model("base")

            socketio.emit('status', {'message': 'Transcribing...'}, room=session_id)
            result = whisper_model.transcribe(audio_np, fp16=False)
            text = result['text'].strip()

            print(f"Transcription: {text}")

            if text:
                # Update or append transcription (like demo)
                if phrase_complete or is_final:
                    session.transcription_lines.append(text)
                    session.add_text(text)
                    socketio.emit('transcription', {
                        'text': text,
                        'source': 'whisper-realtime',
                        'is_complete': True
                    }, room=session_id)
                else:
                    session.transcription_lines[-1] = text
                    socketio.emit('transcription', {
                        'text': text,
                        'source': 'whisper-realtime',
                        'is_complete': False
                    }, room=session_id)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    except Exception as e:
        print(f"Real-time handler error: {str(e)}")
        import traceback
        traceback.print_exc()
        socketio.emit('error', {'message': f'Real-time error: {str(e)}'}, room=session_id)


@socketio.on('start_google_streaming')
def handle_start_google_streaming():
    """Start Google Speech-to-Text streaming"""
    emit('status', {'message': 'Google Speech-to-Text ready'})


@socketio.on('audio_chunk_google')
def handle_audio_chunk_google(data):
    """Process audio chunk with Google Speech-to-Text"""
    session_id = request.sid

    try:
        client = speech.SpeechClient()

        # Decode audio
        audio_data = base64.b64decode(data['audio'])
        file_format = data.get('format', 'webm')

        print(f"Processing Google STT: format={file_format}, size={len(audio_data)} bytes")

        # Map format to Google encoding
        encoding_map = {
            'webm': speech.RecognitionConfig.AudioEncoding.WEBM_OPUS,
            'mp3': speech.RecognitionConfig.AudioEncoding.MP3,
            'flac': speech.RecognitionConfig.AudioEncoding.FLAC,
            'wav': speech.RecognitionConfig.AudioEncoding.LINEAR16,
            'm4a': speech.RecognitionConfig.AudioEncoding.MP3,  # Treat m4a as MP3
            'mp4': speech.RecognitionConfig.AudioEncoding.MP3,
            'ogg': speech.RecognitionConfig.AudioEncoding.OGG_OPUS
        }

        encoding = encoding_map.get(file_format, speech.RecognitionConfig.AudioEncoding.WEBM_OPUS)

        audio = speech.RecognitionAudio(content=audio_data)
        config = speech.RecognitionConfig(
            encoding=encoding,
            sample_rate_hertz=48000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )

        emit('status', {'message': 'Transcribing with Google STT...'})
        response = client.recognize(config=config, audio=audio)

        transcribed = False
        for result in response.results:
            text = result.alternatives[0].transcript
            if text:
                transcription_sessions[session_id].add_text(text)
                emit('transcription', {'text': text, 'source': 'google'})
                transcribed = True
                print(f"Google STT result: {text}")

        if not transcribed:
            emit('status', {'message': 'No speech detected in audio'})

    except Exception as e:
        print(f"Google STT error: {str(e)}")
        import traceback
        traceback.print_exc()
        emit('error', {'message': f'Google STT error: {str(e)}'})


@socketio.on('search_request')
def handle_search_request(data):
    """Handle search request triggered by spacebar"""
    session_id = request.sid
    search_mode = data.get('mode', 'tfidf')  # 'instant' or 'tfidf'
    search_type = data.get('type', 'text')  # 'text' or 'image'

    print(f"\n[Search Request] Mode: {search_mode}, Type: {search_type}")

    if session_id not in transcription_sessions:
        emit('error', {'message': 'No active session'})
        return

    # Get keyword based on search mode
    if search_mode == 'instant':
        # Use the keyword provided by client (last word)
        keyword = data.get('keyword', '')
        print(f"[Instant Search] Using client-provided keyword: {keyword}")
    else:
        # TF-IDF mode: calculate important keyword
        keyword = transcription_sessions[session_id].get_top_keyword()
        print(f"[TF-IDF Search] Calculated keyword: {keyword}")

    if not keyword:
        emit('error', {'message': 'No keywords available for search'})
        return

    # Clean the keyword (remove punctuation)
    keyword_clean = ''.join(c for c in keyword if c.isalnum() or c.isspace()).strip()

    emit('search_keyword', {'keyword': keyword_clean, 'mode': search_mode})

    # Perform Google Custom Search
    try:
        search_results = google_custom_search(keyword_clean, search_type)
        emit('search_results', {
            'keyword': keyword_clean,
            'mode': search_mode,
            'type': search_type,
            'results': search_results
        })
        print(f"[Search] Found {len(search_results)} results for '{keyword_clean}'")
    except Exception as e:
        print(f"Search error: {str(e)}")
        emit('error', {'message': f'Search error: {str(e)}'})


def google_custom_search(query, search_type='text'):
    """Perform Google Custom Search"""
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        'key': GOOGLE_SEARCH_API_KEY,
        'cx': GOOGLE_SEARCH_ENGINE_ID,
        'q': query,
        'num': 5
    }

    if search_type == 'image':
        params['searchType'] = 'image'

    response = requests.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    results = []

    if 'items' in data:
        for item in data['items']:
            if search_type == 'image':
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'thumbnail': item.get('image', {}).get('thumbnailLink', ''),
                    'context': item.get('snippet', '')
                })
            else:
                results.append({
                    'title': item.get('title', ''),
                    'link': item.get('link', ''),
                    'snippet': item.get('snippet', '')
                })

    return results


@socketio.on('clear_session')
def handle_clear_session():
    """Clear transcription session"""
    session_id = request.sid
    transcription_sessions[session_id] = TranscriptionSession()
    emit('session_cleared', {'status': 'Session cleared'})


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)
