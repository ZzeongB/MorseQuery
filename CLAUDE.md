# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MorseQuery is a real-time audio streaming search prototype. It transcribes audio (from microphone or file) using either Whisper or Google Speech-to-Text, then automatically extracts keywords using TF-IDF and performs Google searches when the user presses spacebar.

## Development Commands

### Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Running the Application
```bash
python app.py
```
The app runs on `http://localhost:5001` (note: port 5001, not 5000).

### Required Environment Variables
Create a `.env` file with:
- `GOOGLE_APPLICATION_CREDENTIALS` - path to Google Cloud service account JSON
- `GOOGLE_CUSTOM_SEARCH_API_KEY` - Google Custom Search API key
- `GOOGLE_CUSTOM_SEARCH_ENGINE_ID` - Google Custom Search Engine ID

### External Dependencies
- `ffmpeg` must be installed system-wide (required by Whisper for audio processing)

## Architecture

### WebSocket Communication Pattern
The app uses Flask-SocketIO for real-time bidirectional communication:
- **Client → Server events**: `start_whisper`, `start_google_streaming`, `audio_chunk_whisper`, `audio_chunk_google`, `audio_chunk_realtime`, `search_request`, `clear_session`
- **Server → Client events**: `connected`, `status`, `transcription`, `search_keyword`, `search_results`, `error`

### Session Management
Each WebSocket connection gets a unique `TranscriptionSession` (stored in `transcription_sessions` dict keyed by `request.sid`):
- Tracks words, sentences, and full text
- Maintains phrase state for real-time transcription mode
- Provides TF-IDF keyword extraction via `get_top_keyword()`

### Audio Processing Modes

**1. Whisper Chunk Mode** (`audio_chunk_whisper`)
- Processes discrete audio chunks in background threads
- Each chunk is saved to a temp file, transcribed independently
- Uses `process_whisper_background()` to avoid blocking the WebSocket

**2. Whisper Real-time Mode** (`audio_chunk_realtime`)
- Accumulates audio bytes in `session.phrase_bytes`
- Converts to mono 16kHz PCM format using pydub
- Implements phrase timeout logic (default 3s) to detect phrase boundaries
- Sends progressive updates (incomplete) and final transcriptions (complete)

**3. Google STT Mode** (`audio_chunk_google`)
- Uses Google Cloud Speech-to-Text synchronous `recognize()` API
- Maps audio format to appropriate `AudioEncoding` enum
- Processes each chunk independently (not true streaming)

### Keyword Extraction Logic
The `TranscriptionSession.get_top_keyword()` method:
1. Filters stop words and short words (<3 chars)
2. If 2+ sentences exist: uses TF-IDF with sklearn's `TfidfVectorizer` to find most relevant term
3. Fallback: uses simple word frequency counting (`Counter`)
4. Returns cleaned keyword for search

### Search Flow
When user presses spacebar:
1. Client sends `search_request` with mode ('instant' vs 'tfidf') and type ('text' vs 'image')
2. Server extracts keyword based on mode
3. Calls `google_custom_search()` which hits Google Custom Search API
4. Returns formatted results (with thumbnails for image search)

### Frontend Structure
- `templates/index.html` - Single page application
- `static/js/app.js` - WebRTC MediaRecorder, Socket.IO client logic
- `static/css/style.css` - UI styling

Frontend sends audio chunks as base64-encoded data via WebSocket with format metadata.

## Key Implementation Details

### Audio Format Handling
The app supports multiple audio formats (webm, wav, mp3, m4a, mp4, ogg, flac). Format detection:
- For Whisper: saves to temp file with correct extension
- For Google STT: maps to `RecognitionConfig.AudioEncoding` enum
- For real-time mode: uses pydub to convert any format to raw 16kHz mono PCM

### Threading Model
- Main Flask-SocketIO runs in `threading` async mode
- Whisper processing happens in daemon background threads to prevent blocking
- Each session is isolated by Socket.IO's room/session ID system

### Phrase Timeout Logic (Real-time Mode)
Inspired by `whisper_real_time/transcribe_demo.py`:
- Tracks last audio timestamp (`phrase_time`)
- If gap > `phrase_timeout` seconds, resets accumulated bytes and marks phrase complete
- Allows continuous streaming with natural phrase segmentation

## Important Notes

- The Whisper model loads lazily on first use (downloads ~140MB for base model)
- Google Cloud APIs require billing to be enabled even for free tier
- Port 5001 is hardcoded in `app.py:490` (not the default Flask 5000)
- HTTPS is required for microphone access in browsers (use self-signed cert for dev)
- The `whisper_real_time/` directory contains a separate demo script not integrated into the main app
