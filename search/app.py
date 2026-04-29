"""Playback Search App - three playback modes for keyword search."""

import base64
import io
import json
import re
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from pydub import AudioSegment

from config import (
    INTERRUPTIONS_DIR,
    KEYWORDS2_DIR,
    KEYWORDS_DIR,
    LEXICON_PATH,
    LOGS_DIR,
    MP3_DIR,
    SENTENCES_DIR,
    STUDY_AUDIO_WINDOWS_PATH,
    STUDY_DIR,
    TRANSCRIPT_DIR,
)

app = Flask(__name__)
TERM_RE = re.compile(r"[a-z0-9']+")


def _slugify_filename_part(value: str, default: str = "unknown") -> str:
    """Normalize a filename segment to a safe, predictable token."""
    text = str(value or "").strip()
    if not text:
        return default
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or default


def _build_study_log_basename(data: dict) -> str:
    """Build the shared basename for study JSON and JSONL logs."""
    provided = data.get("logBaseName")
    if provided:
        return _slugify_filename_part(Path(str(provided)).stem)

    participant = _slugify_filename_part(data.get("participant"))
    feature = _slugify_filename_part(data.get("feature"))
    audio_value = data.get("audio") or data.get("videoId")
    audio = _slugify_filename_part(Path(str(audio_value or "unknown")).stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{participant}_{feature}_{audio}_{timestamp}"

# Stopwords for keyword extraction
STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "you",
    "your",
    "yours",
    "yourself",
    "yourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "who",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "why",
    "how",
    "all",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "no",
    "nor",
    "not",
    "only",
    "own",
    "same",
    "so",
    "than",
    "too",
    "very",
    "s",
    "t",
    "can",
    "will",
    "just",
    "don",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
    "ain",
    "aren",
    "couldn",
    "didn",
    "doesn",
    "hadn",
    "hasn",
    "haven",
    "isn",
    "ma",
    "mightn",
    "mustn",
    "needn",
    "shan",
    "shouldn",
    "wasn",
    "weren",
    "won",
    "wouldn",
    "yeah",
    "okay",
    "ok",
    "like",
    "know",
    "think",
    "going",
    "want",
    "got",
    "get",
    "go",
    "come",
    "say",
    "said",
    "would",
    "could",
    "one",
    "two",
    "let",
    "something",
    "thing",
    "things",
    "really",
    "actually",
    "basically",
    "well",
}

# Lexicon cache
_lexicon: dict[str, float] = {}


def load_lexicon() -> dict[str, float]:
    """Load OpenLexicon.xlsx and return word->frequency dict."""
    global _lexicon
    if _lexicon:
        return _lexicon

    if not LEXICON_PATH.exists():
        return {}

    import pandas as pd

    df = pd.read_excel(LEXICON_PATH)

    for _, row in df.iterrows():
        word = str(row["ortho"]).lower()
        freq = row["English_Lexicon_Project__LgSUBTLWF"]
        if pd.notna(freq):
            _lexicon[word] = float(freq)
        else:
            _lexicon[word] = 0.0

    return _lexicon


def extract_keywords(text: str, top_k: int = 3) -> list[str]:
    """Extract rare keywords from text using OpenLexicon."""
    words = re.findall(r"[a-zA-Z]+", text.lower())
    words = [w for w in words if len(w) > 2 and w not in STOPWORDS]

    lexicon = load_lexicon()

    candidates = []
    for w in words:
        freq = lexicon.get(w)
        if freq is None:
            candidates.append((w, -1.0))  # Not in lexicon = rare
        elif freq < 3.0:
            candidates.append((w, freq))

    candidates.sort(key=lambda x: x[1])

    seen = set()
    result = []
    for w, _ in candidates:
        if w not in seen:
            seen.add(w)
            result.append(w)
            if len(result) >= top_k:
                break

    return result


def load_json_file(path):
    with open(path) as f:
        return json.load(f)


def load_study_audio_windows() -> dict:
    if not STUDY_AUDIO_WINDOWS_PATH.exists():
        return {"default_target_window_seconds": 300, "default_max_interruptions": 10, "videos": {}}
    return load_json_file(STUDY_AUDIO_WINDOWS_PATH)


def build_sentence_units(segments: list[dict]) -> list[dict]:
    """Build sentence-wise units using period-delimited word timestamps."""
    units = []
    current_words = []
    current_start = None
    fallback_segment_start = None
    fallback_segment_end = None

    def flush_sentence():
        nonlocal current_words, current_start, fallback_segment_start, fallback_segment_end
        if not current_words:
            return

        last_word = current_words[-1]
        units.append(
            {
                "text": " ".join(word["word"] for word in current_words).strip(),
                "start": current_start if current_start is not None else fallback_segment_start or 0,
                "end": last_word.get("end", fallback_segment_end or current_start or 0),
            }
        )

        current_words = []
        current_start = None
        fallback_segment_start = None
        fallback_segment_end = None

    for seg in segments:
        words = seg.get("words", [])
        if not words:
            flush_sentence()
            units.append(
                {
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"],
                }
            )
            continue

        for word in words:
            if not current_words:
                current_start = word["start"]
                fallback_segment_start = seg["start"]

            fallback_segment_end = seg["end"]
            current_words.append(word)

            if "." in word["word"]:
                flush_sentence()

    flush_sentence()
    return units


def load_or_create_sentences(video_id: str, segments: list[dict]) -> list[dict]:
    """Load sentence cache or create it from transcript segments."""
    SENTENCES_DIR.mkdir(parents=True, exist_ok=True)
    sentence_path = SENTENCES_DIR / f"{video_id}.json"

    if sentence_path.exists():
        with open(sentence_path) as f:
            return json.load(f)

    sentences = build_sentence_units(segments)
    with open(sentence_path, "w") as f:
        json.dump(sentences, f, indent=2)

    return sentences


def parse_clip_times(filename: str) -> tuple[float, float]:
    """Parse clip start/end times from filename like 'xxx_clip_680_1010.mp3'."""
    match = re.search(r"_clip_(\d+)_(\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


def normalize_token(text: str) -> str:
    tokens = TERM_RE.findall((text or "").lower())
    return tokens[0] if len(tokens) == 1 else ""


def build_word_index(segments: list[dict], *, clip_start: float) -> list[dict]:
    items = []
    for segment_index, segment in enumerate(segments):
        for word_index, word in enumerate(segment.get("words", [])):
            items.append(
                {
                    "segment_index": segment_index,
                    "word_index": word_index,
                    "text": word.get("word", ""),
                    "start": float(word.get("start", 0.0)) - clip_start,
                    "end": float(word.get("end", 0.0)) - clip_start,
                }
            )
    return items


def remap_time_entries(path: Path, words: list[dict]) -> None:
    if not path.exists():
        return

    entries = load_json_file(path)
    if not isinstance(entries, list):
        return

    occurrences: dict[str, list[float]] = {}
    for word in words:
        token = normalize_token(str(word.get("text", "")))
        if not token:
            continue
        occurrences.setdefault(token, []).append(float(word["start"]))

    used_indices: dict[str, set[int]] = {}
    updated_entries = []
    for entry in entries:
        if not isinstance(entry, dict):
            updated_entries.append(entry)
            continue

        token = normalize_token(str(entry.get("word", "")))
        original_time = float(entry.get("time", 0.0))
        entry_copy = dict(entry)
        candidates = occurrences.get(token, [])
        if candidates:
            used = used_indices.setdefault(token, set())
            ranked = sorted(
                range(len(candidates)),
                key=lambda idx: (abs(candidates[idx] - original_time), idx),
            )
            chosen_idx = next((idx for idx in ranked if idx not in used), ranked[0])
            used.add(chosen_idx)
            entry_copy["time"] = candidates[chosen_idx]
        updated_entries.append(entry_copy)

    path.write_text(json.dumps(updated_entries, indent=2), encoding="utf-8")


def apply_word_timestamp_updates(transcript_id: str, updated_words: list[dict]) -> None:
    transcript_path = TRANSCRIPT_DIR / f"{transcript_id}.json"
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript not found: {transcript_id}")

    data = load_json_file(transcript_path)
    clip_start = float(data.get("start_time", 0.0))
    segments = data.get("segments", [])
    indexed_words = build_word_index(segments, clip_start=clip_start)

    if len(updated_words) != len(indexed_words):
        raise ValueError("Word count mismatch while saving timestamps")

    flat_words_for_mapping = []
    for source_word, incoming_word in zip(indexed_words, updated_words):
        segment = segments[source_word["segment_index"]]
        word = segment["words"][source_word["word_index"]]
        rel_start = max(0.0, float(incoming_word["start"]))
        rel_end = max(rel_start, float(incoming_word["end"]))
        abs_start = rel_start + clip_start
        abs_end = rel_end + clip_start
        word["start"] = abs_start
        word["end"] = abs_end
        flat_words_for_mapping.append(
            {
                "text": word.get("word", ""),
                "start": rel_start,
            }
        )

    for segment in segments:
        segment_words = segment.get("words", [])
        if segment_words:
            segment["start"] = segment_words[0]["start"]
            segment["end"] = segment_words[-1]["end"]
            segment["text"] = " ".join(str(word.get("word", "")).strip() for word in segment_words).strip()

    data["text"] = " ".join(str(segment.get("text", "")).strip() for segment in segments).strip()
    transcript_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    sentence_path = SENTENCES_DIR / f"{transcript_id}.json"
    sentence_path.parent.mkdir(parents=True, exist_ok=True)
    sentence_path.write_text(
        json.dumps(build_sentence_units(load_transcript_segments_for_api(data)), indent=2),
        encoding="utf-8",
    )

    remap_time_entries(KEYWORDS_DIR / f"{transcript_id}.json", flat_words_for_mapping)
    remap_time_entries(KEYWORDS_DIR / f"{transcript_id}.jargon.json", flat_words_for_mapping)
    remap_time_entries(KEYWORDS2_DIR / f"{transcript_id}.json", flat_words_for_mapping)


def load_transcript_segments_for_api(data: dict) -> list[dict]:
    clip_start = float(data.get("start_time", 0.0))
    segments = []
    lexicon = load_lexicon()
    for seg in data.get("segments", []):
        words = []
        for w in seg.get("words", []):
            word_text = w["word"].strip().lower()
            word_clean = re.sub(r"[^a-z]", "", word_text)
            freq = lexicon.get(word_clean, -1)
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": w["start"] - clip_start,
                    "end": w["end"] - clip_start,
                    "freq": freq,
                }
            )

        segments.append(
            {
                "text": seg["text"],
                "start": seg["start"] - clip_start,
                "end": seg["end"] - clip_start,
                "keywords": extract_keywords(seg["text"]),
                "words": words,
            }
        )
    return segments


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/timestamp-check")
def timestamp_check():
    return render_template("timestamp_check.html")


@app.route("/timestamp-edit")
def timestamp_edit():
    return render_template("timestamp_edit.html")


@app.route("/debug")
def debug():
    return render_template("debug.html")


@app.route("/api/files")
def get_files():
    """Return list of available mp3 files with their video IDs."""
    files = []
    for mp3_path in MP3_DIR.glob("*.mp3"):
        video_id = mp3_path.stem.split("_clip_")[0]
        transcript_paths = sorted(TRANSCRIPT_DIR.glob(f"{video_id}*.json"))
        if not transcript_paths:
            continue

        clip_start, clip_end = parse_clip_times(mp3_path.name)
        for transcript_path in transcript_paths:
            files.append(
                {
                    "filename": mp3_path.name,
                    "video_id": video_id,
                    "transcript_id": transcript_path.stem,
                    "transcript_label": transcript_path.stem,
                    "clip_start": clip_start,
                    "clip_end": clip_end,
                }
            )
    return jsonify(files)


@app.route("/api/transcript/<video_id>")
def get_transcript(video_id: str):
    """Return transcript with keywords extracted for each segment."""
    transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
    if not transcript_path.exists():
        return jsonify({"error": "Transcript not found"}), 404

    with open(transcript_path) as f:
        data = json.load(f)

    clip_start = data.get("start_time", 0)
    segments = load_transcript_segments_for_api(data)

    # Load custom keywords if exists
    keywords_path = KEYWORDS_DIR / f"{video_id}.json"
    custom_keywords = []
    if keywords_path.exists():
        with open(keywords_path) as f:
            custom_keywords = json.load(f)

    jargon_path = KEYWORDS_DIR / f"{video_id}.jargon.json"
    jargon_keywords = []
    if jargon_path.exists():
        with open(jargon_path) as f:
            jargon_keywords = json.load(f)

    # Load custom keywords2 if exists
    keywords2_path = KEYWORDS2_DIR / f"{video_id}.json"
    custom_keywords2 = []
    if keywords2_path.exists():
        with open(keywords2_path) as f:
            custom_keywords2 = json.load(f)

    sentences = load_or_create_sentences(video_id, segments)

    return jsonify(
        {
            "video_id": video_id,
            "clip_start": clip_start,
            "segments": segments,
            "sentences": sentences,
            "custom_keywords": custom_keywords,
            "jargon_keywords": jargon_keywords,
            "custom_keywords2": custom_keywords2,
        }
    )


@app.route("/api/transcript/<transcript_id>/timestamps", methods=["POST"])
def save_transcript_timestamps(transcript_id: str):
    data = request.get_json()
    if not data or not isinstance(data.get("words"), list):
        return jsonify({"error": "Invalid payload"}), 400

    try:
        apply_word_timestamp_updates(transcript_id, data["words"])
    except FileNotFoundError:
        return jsonify({"error": "Transcript not found"}), 404
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({"status": "saved"})


@app.route("/mp3/<filename>")
def serve_mp3(filename: str):
    """Serve mp3 file."""
    return send_from_directory(MP3_DIR, filename)


@app.route("/api/reverse/<filename>")
def get_reversed_audio(filename: str):
    """Return reversed audio from 0 to end_sec as base64."""
    end_sec = request.args.get("end", type=float, default=0)
    if end_sec <= 0:
        return jsonify({"error": "Invalid end time"}), 400

    mp3_path = MP3_DIR / filename
    if not mp3_path.exists():
        return jsonify({"error": "File not found"}), 404

    audio = AudioSegment.from_mp3(mp3_path)
    end_ms = int(end_sec * 1000)
    chunk = audio[:end_ms]
    reversed_chunk = chunk.reverse()

    # Export to mp3 bytes
    buffer = io.BytesIO()
    reversed_chunk.export(buffer, format="mp3")
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return jsonify(
        {
            "audio": b64,
            "duration": len(reversed_chunk) / 1000,
        }
    )


@app.route("/api/speedup/<filename>")
def get_speedup_audio(filename: str):
    """Return sped-up audio from start_sec for duration_sec as base64."""
    start_sec = request.args.get("start", type=float, default=0)
    duration_sec = request.args.get("duration", type=float, default=30)  # Only process 30 sec
    rate = request.args.get("rate", type=float, default=2.0)
    preserve_pitch = request.args.get("preserve_pitch", default="true").lower() == "true"

    # Clamp rate to reasonable values
    rate = max(1.1, min(4.0, rate))

    mp3_path = MP3_DIR / filename
    if not mp3_path.exists():
        return jsonify({"error": "File not found"}), 404

    audio = AudioSegment.from_mp3(mp3_path)
    start_ms = int(start_sec * 1000)
    end_ms = int((start_sec + duration_sec) * 1000)
    end_ms = min(end_ms, len(audio))  # Don't exceed audio length
    chunk = audio[start_ms:end_ms]

    # Track if this is the last segment
    is_last = end_ms >= len(audio)

    if preserve_pitch:
        # Use ffmpeg's atempo filter to preserve pitch
        # atempo only supports 0.5 to 2.0, so we chain multiple filters for higher rates
        buffer_in = io.BytesIO()
        chunk.export(buffer_in, format="mp3")
        buffer_in.seek(0)

        # Build atempo filter chain (each atempo is limited to 0.5-2.0)
        atempo_filters = []
        remaining_rate = rate
        while remaining_rate > 2.0:
            atempo_filters.append("atempo=2.0")
            remaining_rate /= 2.0
        atempo_filters.append(f"atempo={remaining_rate}")
        filter_str = ",".join(atempo_filters)

        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
            tmp_in.write(buffer_in.read())
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace(".mp3", "_out.mp3")

        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-filter:a", filter_str,
                    "-vn", tmp_out_path
                ],
                capture_output=True,
                check=True
            )

            with open(tmp_out_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")

            # Calculate duration
            speedup_chunk = AudioSegment.from_mp3(tmp_out_path)
            duration = len(speedup_chunk) / 1000
        finally:
            import os
            if os.path.exists(tmp_in_path):
                os.remove(tmp_in_path)
            if os.path.exists(tmp_out_path):
                os.remove(tmp_out_path)
    else:
        # Simple speedup by altering frame_rate (changes pitch)
        speedup_chunk = chunk._spawn(
            chunk.raw_data,
            overrides={"frame_rate": int(chunk.frame_rate * rate)}
        ).set_frame_rate(chunk.frame_rate)

        # Export to mp3 bytes
        buffer = io.BytesIO()
        speedup_chunk.export(buffer, format="mp3")
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode("utf-8")
        duration = len(speedup_chunk) / 1000

    # original_duration = how much original audio time this covers
    original_duration = (end_ms - start_ms) / 1000

    return jsonify(
        {
            "audio": b64,
            "duration": duration,
            "rate": rate,
            "original_duration": original_duration,
            "is_last": is_last,
        }
    )


# ============== Study API Endpoints ==============


@app.route("/api/study/config")
def get_study_config():
    """Return study configuration."""
    config_path = STUDY_DIR / "study_config.json"
    if not config_path.exists():
        return jsonify({"error": "Study config not found"}), 404

    with open(config_path) as f:
        config = json.load(f)
    return jsonify(config)


@app.route("/api/study/interruptions/<video_id>")
def get_study_interruptions(video_id: str):
    """Return interruption config for a specific video."""
    interruptions_path = INTERRUPTIONS_DIR / f"{video_id}.json"
    if not interruptions_path.exists():
        return jsonify({"error": f"Interruptions for {video_id} not found"}), 404

    data = load_json_file(interruptions_path)
    interruptions = data.get("interruptions", [])
    if not interruptions:
        return jsonify({"error": f"No interruptions configured for {video_id}"}), 400

    study_windows = load_study_audio_windows()
    video_config = study_windows.get("videos", {}).get(video_id, {})
    audio_start_time = float(data.get("audio_start_time", video_config.get("audio_start_time", 0)))
    target_window_seconds = float(
        data.get(
            "target_window_seconds",
            video_config.get(
                "target_window_seconds",
                study_windows.get("default_target_window_seconds", 300),
            ),
        )
    )
    search_window_end_time = float(
        data.get("search_window_end_time", audio_start_time + target_window_seconds)
    )
    playback_end_time = max(
        float(item.get("target_word_time", 0)) + float(item.get("delay_seconds", 0))
        for item in interruptions
    )

    data["audio_start_time"] = audio_start_time
    data["target_window_seconds"] = target_window_seconds
    data["search_window_end_time"] = search_window_end_time
    data["playback_end_time"] = playback_end_time
    return jsonify(data)


@app.route("/api/study/log", methods=["POST"])
def log_study_event():
    """Log a single study event to JSONL file."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create logs directory if not exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{_build_study_log_basename(data)}.jsonl"

    # Append event to log file
    with open(log_path, "a") as f:
        event = {"ts": datetime.now().isoformat(), **data}
        f.write(json.dumps(event) + "\n")

    return jsonify({"status": "logged", "file": log_path.name})


@app.route("/api/study/session", methods=["POST"])
def save_study_session():
    """Save complete study session data."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Create logs directory if not exists
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Save session summary
    session_path = LOGS_DIR / f"{_build_study_log_basename(data)}.json"
    with open(session_path, "w") as f:
        json.dump(data, f, indent=2)

    return jsonify({"status": "saved", "file": session_path.name})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
