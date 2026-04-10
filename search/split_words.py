"""Split MP3 into word-level clips, filtering out common words using OpenLexicon."""

import json
import re
from pathlib import Path
from pydub import AudioSegment
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
MP3_PATH = BASE_DIR / "mp3" / "bAkuNXtgrLA_clip_680_1010.mp3"
TRANSCRIPT_PATH = BASE_DIR / "data" / "transcripts" / "bAkuNXtgrLA.json"
LEXICON_PATH = BASE_DIR / "data" / "lexicon" / "OpenLexicon.xlsx"
OUTPUT_DIR = BASE_DIR / "data" / "word_clips" / "bAkuNXtgrLA"

FREQ_SKIP_THRESHOLD = 4.0


def load_lexicon() -> dict[str, float]:
    """Load OpenLexicon.xlsx and return word->frequency dict."""
    if not LEXICON_PATH.exists():
        print(f"Warning: Lexicon not found at {LEXICON_PATH}")
        return {}

    df = pd.read_excel(LEXICON_PATH)
    lexicon = {}
    for _, row in df.iterrows():
        word = str(row["ortho"]).lower()
        freq = row["English_Lexicon_Project__LgSUBTLWF"]
        if pd.notna(freq):
            lexicon[word] = float(freq)
        else:
            lexicon[word] = 0.0
    return lexicon


def should_skip_word(word: str, freq: float) -> bool:
    """Check if word should be skipped."""
    # Skip short words
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if len(cleaned) <= 2:
        return True
    # Skip high frequency words
    if freq >= FREQ_SKIP_THRESHOLD:
        return True
    return False


def main():
    # Load transcript
    with open(TRANSCRIPT_PATH) as f:
        data = json.load(f)

    clip_start = data.get("start_time", 0)  # 680.0

    # Load audio
    print(f"Loading audio: {MP3_PATH}")
    audio = AudioSegment.from_mp3(MP3_PATH)

    # Load lexicon
    print("Loading lexicon...")
    lexicon = load_lexicon()

    # Extract all words with timestamps
    all_words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            word_text = w["word"].strip()
            word_clean = re.sub(r"[^a-z]", "", word_text.lower())
            freq = lexicon.get(word_clean, -1)

            all_words.append({
                "word": word_text,
                "start": w["start"] - clip_start,  # Convert to mp3-relative time
                "end": w["end"] - clip_start,
                "freq": freq,
            })

    # Filter out common words
    important_words = [w for w in all_words if not should_skip_word(w["word"], w["freq"])]

    print(f"Total words: {len(all_words)}")
    print(f"Important words (after filtering): {len(important_words)}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save words in reverse order (for backward playback test)
    reversed_words = list(reversed(important_words))

    print("\n--- Words to be saved (in reverse/backward order) ---")
    for i, w in enumerate(reversed_words):
        freq_str = f"{w['freq']:.2f}" if w['freq'] >= 0 else "N/A"
        print(f"{i+1:03d}. [{w['start']:.2f}s - {w['end']:.2f}s] {w['word']:<20} (freq: {freq_str})")

    print("\n--- Saving clips ---")
    for i, w in enumerate(reversed_words):
        start_ms = int(w["start"] * 1000)
        end_ms = int(w["end"] * 1000)

        # Add small padding
        start_ms = max(0, start_ms - 50)
        end_ms = min(len(audio), end_ms + 50)

        clip = audio[start_ms:end_ms]

        # Filename: order_word_originalTime.mp3
        word_clean = re.sub(r"[^a-zA-Z0-9]", "", w["word"])
        filename = f"{i+1:03d}_{word_clean}_{w['start']:.2f}s.mp3"
        output_path = OUTPUT_DIR / filename

        clip.export(output_path, format="mp3")
        print(f"Saved: {filename} ({end_ms - start_ms}ms)")

    print(f"\nDone! {len(reversed_words)} clips saved to {OUTPUT_DIR}")

    # Also save a manifest JSON
    manifest = {
        "source": str(MP3_PATH),
        "total_words": len(all_words),
        "filtered_words": len(important_words),
        "freq_threshold": FREQ_SKIP_THRESHOLD,
        "words": [{
            "index": i + 1,
            "word": w["word"],
            "start": w["start"],
            "end": w["end"],
            "freq": w["freq"],
        } for i, w in enumerate(reversed_words)]
    }

    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
