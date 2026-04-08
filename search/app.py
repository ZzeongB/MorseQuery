"""Playback Search App - three playback modes for keyword search."""

import base64
import io
import json
import re
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from pydub import AudioSegment

app = Flask(__name__)

BASE_DIR = Path(__file__).parent.parent
MP3_DIR = BASE_DIR / "mp3"
TRANSCRIPT_DIR = BASE_DIR / "data" / "transcripts"
KEYWORDS_DIR = BASE_DIR / "data" / "keywords"
LEXICON_PATH = BASE_DIR / "data" / "lexicon" / "OpenLexicon.xlsx"

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


def parse_clip_times(filename: str) -> tuple[float, float]:
    """Parse clip start/end times from filename like 'xxx_clip_680_1010.mp3'."""
    match = re.search(r"_clip_(\d+)_(\d+)", filename)
    if match:
        return float(match.group(1)), float(match.group(2))
    return 0.0, 0.0


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/files")
def get_files():
    """Return list of available mp3 files with their video IDs."""
    files = []
    for mp3_path in MP3_DIR.glob("*.mp3"):
        video_id = mp3_path.stem.split("_clip_")[0]
        transcript_path = TRANSCRIPT_DIR / f"{video_id}.json"
        if transcript_path.exists():
            clip_start, clip_end = parse_clip_times(mp3_path.name)
            files.append(
                {
                    "filename": mp3_path.name,
                    "video_id": video_id,
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

    # Process segments: convert timestamps and extract keywords
    segments = []
    for seg in data.get("segments", []):
        # Convert absolute time to mp3-relative time
        mp3_start = seg["start"] - clip_start
        mp3_end = seg["end"] - clip_start

        keywords = extract_keywords(seg["text"])

        # Process words with timestamps
        words = []
        for w in seg.get("words", []):
            words.append(
                {
                    "word": w["word"].strip(),
                    "start": w["start"] - clip_start,
                    "end": w["end"] - clip_start,
                }
            )

        segments.append(
            {
                "text": seg["text"],
                "start": mp3_start,
                "end": mp3_end,
                "keywords": keywords,
                "words": words,
            }
        )

    # Load custom keywords if exists
    keywords_path = KEYWORDS_DIR / f"{video_id}.json"
    custom_keywords = []
    if keywords_path.exists():
        with open(keywords_path) as f:
            custom_keywords = json.load(f)

    return jsonify(
        {
            "video_id": video_id,
            "clip_start": clip_start,
            "segments": segments,
            "custom_keywords": custom_keywords,
        }
    )


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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
