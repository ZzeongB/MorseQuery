# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The Search App is a playback-based keyword search interface for user studies. It plays audio files with configurable interruptions at target words, allowing users to discover and search for keywords during playback.

Part of the larger MorseQuery project (see parent directory CLAUDE.md for other apps).

## Development Commands

### Running the App
```bash
cd search
python app.py    # http://localhost:5003
```

### Data Processing Scripts
```bash
# Transcribe MP3 files with Whisper
python scripts/study_data/transcribe_mp3_with_whisper.py

# Extract semantic keywords (GPT-based, pipeline 1)
python scripts/study_data/extract_semantic_words.py

# Extract semantic keywords (pipeline 2)
python scripts/study_data/build_semantic_words2.py

# Extract jargon/technical terms
python scripts/study_data/extract_jargon_words.py

# Generate target words for study interruptions
python scripts/study_data/generate_target_words.py
python scripts/study_data/build_target_word_pipeline.py

# Generate quiz questions
python scripts/study_data/generate_true_false_quiz.py
```

### Dependencies
- `ffmpeg` must be installed system-wide (required by pydub for audio processing)
- Parent directory's `requirements.txt` covers all Python dependencies
- OpenLexicon.xlsx must exist at `data/lexicon/OpenLexicon.xlsx` for keyword frequency analysis

## Architecture

### Flask REST API (`app.py`)

**Main Routes:**
- `GET /` - Main playback interface
- `GET /timestamp-check` - Timestamp verification page
- `GET /timestamp-edit` - Word timestamp editing interface
- `GET /debug` - Debug interface

**API Endpoints:**
- `GET /api/files` - List MP3 files with associated transcripts
- `GET /api/transcript/<video_id>` - Get transcript with keywords and word frequencies
- `POST /api/transcript/<transcript_id>/timestamps` - Save edited word timestamps
- `GET /api/reverse/<filename>?end=<seconds>` - Get reversed audio segment (base64)
- `GET /api/speedup/<filename>?start=<s>&duration=<d>&rate=<r>` - Get sped-up audio (base64)
- `GET /api/study/config` - Get study configuration
- `GET /api/study/interruptions/<video_id>` - Get interruption config for video
- `POST /api/study/log` - Log study event (JSONL format)
- `POST /api/study/session` - Save complete session data (JSON)

### Data Directory Structure (`data/`)

```
data/
├── mp3/                    # Audio clip files (e.g., video_clip_680_1010.mp3)
├── transcripts/            # Whisper JSON output with word-level timestamps
├── sentences/              # Cached sentence-level segmentation (auto-generated)
├── semantic_words/         # Pipeline 1 keywords + jargon (.json, .jargon.json)
├── semantic_words2/        # Pipeline 2 keywords
├── lexicon/
│   └── OpenLexicon.xlsx    # Word frequency database
└── study/
    ├── study_config.json   # Study participant/feature configuration
    ├── audio_windows.json  # Per-video timing configuration
    └── target_words/       # Interruption configs per video
```

### Keyword Extraction Logic

**OpenLexicon Frequency-Based** (`extract_keywords()` in app.py):
1. Tokenize and filter stopwords
2. Look up word frequency in OpenLexicon.xlsx
3. Prioritize: words not in lexicon (-1) > low frequency (< 3.0)
4. Return top-K rarest words

**GPT-Based** (scripts/study_data/extract_semantic_words.py):
- Sends context to GPT with prompts from `scripts/prompts/`
- Returns semantic keywords with descriptions and timestamps

### Timestamp Editing Flow

When saving edited timestamps via `POST /api/transcript/<id>/timestamps`:
1. Creates new transcript with `_ts_edit` suffix
2. Updates word-level timestamps in segments
3. Regenerates sentence cache
4. Copies and remaps keyword files (semantic_words, jargon) to new timestamps
5. Remaps target word interruption configs

### Audio Processing

- `pydub.AudioSegment` for loading/manipulating MP3
- Reversed audio: `audio[:end_ms].reverse()`
- Speed-up with pitch preservation: ffmpeg `atempo` filter chain (chains multiple 2x filters for rates > 2.0)
- Clip filename format: `{video_id}_clip_{start_sec}_{end_sec}.mp3`

### Study Mode

Study sessions use interruptions defined in `data/study/target_words/{video_id}.json`:
```json
{
  "video_id": "...",
  "audio_start_time": 65.0,
  "interruptions": [
    {
      "target_word": "photosynthesis",
      "target_word_time": 120.5,
      "delay_seconds": 2.0,
      "search_start_time": 122.5
    }
  ]
}
```

Logs saved to `logs/study/` as:
- `{participant}_{feature}_{audio}_{timestamp}.jsonl` - Event log
- `{participant}_{feature}_{audio}_{timestamp}.json` - Session summary

## Key Implementation Details

### Path Resolution (`config.py`)

Uses `_prefer_search_dir()` to check `search/data/` first, then fall back to `../data/` for legacy compatibility.

### Transcript ID Resolution

- `preferred_transcript_id()`: Returns `_ts_edit` version if it exists
- `edited_transcript_id()`: Strips existing edit suffixes and adds `_ts_edit`

### Word Index Building

`build_word_index()` creates a flat list of words with:
- `segment_index`, `word_index` - Position in transcript
- `start`, `end` - Timestamps relative to clip start (not absolute)
- `text` - Word text

All timestamps in API responses are relative to clip start (adjusted by `clip_start` from transcript).

## Frontend Structure

- `templates/index.html` - Main playback UI
- `templates/timestamp_check.html` - Verify word alignments
- `templates/timestamp_edit.html` - Edit word timestamps
- `templates/debug.html` - Debug interface
- `static/js/index.js` - Playback logic, keyword display, study mode
- `static/css/index.css` - Styles for playback interface
