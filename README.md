# MorseQuery

Real-time audio transcription and smart keyword search.

## Features

- **Speech Recognition**: Whisper (local) or Google Speech-to-Text (cloud)
- **Smart Keyword Extraction**: Uses OpenLexicon to identify rare/important words
- **Instant Search**: Press spacebar to search Google for the most relevant keyword
- **Text & Image Search**: Toggle between search modes

## Quick Start

### 1. Install

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setup Credentials

Create `.env` file:

```env
GOOGLE_APPLICATION_CREDENTIALS=your-credentials.json
GOOGLE_CUSTOM_SEARCH_API_KEY=your_api_key
GOOGLE_CUSTOM_SEARCH_ENGINE_ID=your_engine_id
```

### 3. Run

```bash
python app.py
```

Visit: `http://localhost:5001`

## Usage

1. Choose Whisper or Google STT
2. Start microphone or upload file
3. Watch real-time transcription
4. Press **SPACEBAR** to search for the most important keyword

## How It Works

**Keyword Extraction**: Analyzes recent words using OpenLexicon frequency data. Selects rare words (frequency < 1.0) or words not in the lexicon as search keywords, filtering out common words.

## Requirements

- Python 3.8+
- Google Cloud credentials (for Speech-to-Text)
- Google Custom Search API key
- `ffmpeg` (for Whisper)