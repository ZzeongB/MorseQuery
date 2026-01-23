# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MorseQuery is a real-time audio streaming search prototype. It transcribes audio (from microphone or file) using Whisper, Google Speech-to-Text, or Gemini Live API, then automatically extracts keywords and performs Google searches when the user presses spacebar.

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
- `GOOGLE_API_KEY` - Google API key (for Gemini Live)
- `OPENAI_API_KEY` - OpenAI API key (for GPT-based keyword extraction)

### External Dependencies
- `ffmpeg` must be installed system-wide (required by Whisper for audio processing)

## Architecture

### Project Structure
```
app.py                  # Flask entry point, SocketIO initialization
handlers/               # SocketIO event handlers (modular)
├── connection.py       # Session lifecycle (connect, disconnect, clear)
├── whisper.py          # Whisper transcription (chunk + real-time modes)
├── gemini.py           # Gemini Live API integration
├── search.py           # Search request handling, keyword dispatch
└── srt.py              # SRT subtitle file support
src/core/               # Core business logic
├── session.py          # TranscriptionSession class, keyword extraction
├── config.py           # Environment variables, constants
├── lexicon.py          # OpenLexicon loading, word preprocessing
├── search.py           # Google Custom Search API wrapper
└── gemini_parser.py    # Parse Gemini streaming output
```

### WebSocket Communication Pattern
The app uses Flask-SocketIO for real-time bidirectional communication:
- **Client → Server events**: `start_whisper`, `start_google_streaming`, `audio_chunk_whisper`, `audio_chunk_google`, `audio_chunk_realtime`, `search_request`, `next_keyword`, `clear_session`
- **Server → Client events**: `connected`, `status`, `transcription`, `search_keyword`, `search_results`, `all_keywords`, `gemini_connected`, `gemini_captions`, `gemini_summary`, `gemini_terms`, `error`

### Session Management
Each WebSocket connection gets a unique `TranscriptionSession` (stored in `transcription_sessions` dict keyed by `request.sid`):
- Tracks words, timestamps, sentences, and full text
- Maintains phrase state for real-time transcription mode
- Stores GPT keyword pairs and Gemini terms for navigation
- Provides multiple keyword extraction methods
- Logs all events to `/logs/` as JSON files

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

**4. Gemini Live Mode** (`handlers/gemini.py`)
- Bidirectional audio streaming with real-time analysis
- Runs in separate asyncio event loop
- Produces captions, summaries, and extracted terms

### Keyword Extraction Modes
The `search_request` handler supports multiple modes (`handlers/search.py`):

| Mode | Description | Implementation |
|------|-------------|----------------|
| `instant` | Use keyword provided by client | Direct passthrough |
| `recent` | OpenLexicon filtering on recent words | `session.get_top_keyword_with_time_threshold()` |
| `gpt` | GPT-4o-mini predicts top 3 keywords | `session.get_top_keyword_gpt()` |
| `gemini` | Use terms extracted by Gemini Live | `session.get_gemini_terms_for_search()` |
| (default) | OpenLexicon filtering on context window | `session.get_top_keyword()` |

**OpenLexicon Logic** (`src/core/lexicon.py`):
1. Preprocesses words (contractions → lemmatization)
2. Looks up frequency in OpenLexicon.xlsx
3. Prioritizes: words not in lexicon > NaN freq > freq < 3.0
4. Returns lowest-frequency (most important) word

**GPT Logic** (`src/core/session.py:get_top_keyword_gpt`):
1. Gets words from last N seconds
2. Sends context to GPT-4o-mini with few-shot prompt
3. Returns top 3 keyword-description pairs
4. Supports double-spacebar navigation through keywords

### Search Flow
When user presses spacebar:
1. Client sends `search_request` with mode, type, and options
2. Server extracts keyword based on mode
3. Calls `google_custom_search()` which hits Google Custom Search API
4. Returns formatted results (with thumbnails for image search)
5. Logs action to session event log

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
- Gemini Live runs in a separate asyncio event loop
- Each session is isolated by Socket.IO's room/session ID system

### Phrase Timeout Logic (Real-time Mode)
Inspired by `whisper_real_time/transcribe_demo.py`:
- Tracks last audio timestamp (`phrase_time`)
- If gap > `phrase_timeout` seconds, resets accumulated bytes and marks phrase complete
- Allows continuous streaming with natural phrase segmentation

### Session Logging
Every session captures events to `/logs/` directory:
- `session_start` / `session_end`
- `whisper_transcription` (text, word count, timing)
- `keyword_extraction_*` (method, selected keyword, frequency)
- `search_action` (mode, type, keyword, results count)

## Important Notes

- The Whisper model uses the 'tiny' model for low latency real-time transcription (downloads ~75MB on first use)
  - Tiny model is faster but less accurate than base/small models
  - Optimized for quick response time to minimize delay between speech and keyword extraction
- FP16 warnings are suppressed (CPU doesn't support FP16, falls back to FP32 automatically)
- Google Cloud APIs require billing to be enabled even for free tier
- Port 5001 is hardcoded in `app.py` (not the default Flask 5000)
- HTTPS is required for microphone access in browsers (use self-signed cert for dev)
- The `whisper_real_time/` directory contains a separate demo script not integrated into the main app
