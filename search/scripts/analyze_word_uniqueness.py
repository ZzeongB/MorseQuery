"""
Analyze word uniqueness/frequency across SRT files for user study task material validation.
Uses SUBTLEX-US corpus (Brysbaert & New, 2009) - 51 million words from movie/TV subtitles.
Uniqueness score = 7 - Lg10WF (Lg10WF scale: 0=rare, ~6=common like "the").

Reference:
Brysbaert, M., & New, B. (2009). Moving beyond Kučera and Francis: A critical evaluation
of current word frequency norms. Behavior Research Methods, 41(4), 977-990.
"""

import os
import re
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import brown, stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find("corpora/brown")
except LookupError:
    nltk.download("brown", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def is_noun(word):
    """Check if a word is a noun using NLTK POS tagger.

    Noun tags: NN (singular), NNS (plural), NNP (proper singular), NNPS (proper plural)
    """
    tagged = nltk.pos_tag([word])
    pos = tagged[0][1]
    return pos.startswith("NN")


def get_pos_tag(word):
    """Get POS tag for a word."""
    tagged = nltk.pos_tag([word])
    return tagged[0][1]


# === SRT Parsing ===


def parse_srt(filepath):
    """Parse SRT file and extract subtitle entries with timestamps."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # SRT format: index, timestamp, text, blank line
    pattern = r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.*?)(?=\n\n|\Z)"
    matches = re.findall(pattern, content, re.DOTALL)

    entries = []
    for match in matches:
        idx, start_time, end_time, text = match
        # Convert timestamp to seconds
        start_sec = timestamp_to_seconds(start_time)
        end_sec = timestamp_to_seconds(end_time)
        # Clean text (remove [music], speaker markers, etc.)
        clean_text = clean_subtitle_text(text)
        if clean_text:
            entries.append(
                {
                    "index": int(idx),
                    "start": start_sec,
                    "end": end_sec,
                    "text": clean_text,
                }
            )
    return entries


def timestamp_to_seconds(ts):
    """Convert SRT timestamp (HH:MM:SS,mmm) to seconds."""
    parts = ts.replace(",", ":").split(":")
    hours, minutes, seconds, ms = (
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
    )
    return hours * 3600 + minutes * 60 + seconds + ms / 1000


def clean_subtitle_text(text):
    """Remove non-speech content like [music], speaker markers, etc."""
    # Remove [music], [applause], etc.
    text = re.sub(r"\[.*?\]", "", text)
    # Remove >> speaker markers
    text = re.sub(r">>", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_words_with_timestamps(entries):
    """Extract individual words with estimated timestamps from subtitle entries."""
    words = []
    for entry in entries:
        text = entry["text"]
        # Tokenize
        tokens = word_tokenize(text.lower())
        # Filter to alphabetic words only
        alpha_tokens = [t for t in tokens if t.isalpha() and len(t) > 1]

        if not alpha_tokens:
            continue

        # Distribute timestamps across words (rough estimate)
        duration = entry["end"] - entry["start"]
        time_per_word = duration / len(alpha_tokens) if alpha_tokens else 0

        for i, word in enumerate(alpha_tokens):
            word_time = entry["start"] + i * time_per_word
            words.append(
                {"word": word, "time": word_time, "entry_index": entry["index"]}
            )

    return words


def extract_6min_segment(words, start_time=0, duration=360):
    """Extract words from a 6-minute segment starting at start_time."""
    end_time = start_time + duration
    return [w for w in words if start_time <= w["time"] < end_time]


# === SUBTLEX-US Loading ===

_subtlex_data = None


def load_subtlex():
    """Load SUBTLEX-US frequency data."""
    global _subtlex_data
    if _subtlex_data is not None:
        return _subtlex_data

    subtlex_path = Path(
        "/Users/jeongin/morsequery/search/data/lexicon/SUBTLEXus74286wordstextversion.txt"
    )

    print(f"Loading SUBTLEX-US from {subtlex_path}...")
    df = pd.read_csv(subtlex_path, sep="\t", encoding="utf-8")

    # Create dictionary: word -> Lg10WF (log10 word frequency)
    # Lg10WF range: 0 (rare) to ~6.3 (common like "the")
    _subtlex_data = {}
    for _, row in df.iterrows():
        word = str(row["Word"]).lower()
        lg10wf = row["Lg10WF"]
        _subtlex_data[word] = lg10wf

    print(f"Loaded {len(_subtlex_data):,} words from SUBTLEX-US")
    return _subtlex_data


# === Word Frequency Analysis ===


def get_subtlex_frequency(word):
    """Get word frequency from SUBTLEX-US (Lg10WF scale).

    Lg10WF = log10(frequency per million + 1)
    Scale: 0 = very rare, ~6.3 = very common (like 'the').

    Based on 51 million words from 8,388 US film/TV subtitles.
    """
    subtlex = load_subtlex()
    return subtlex.get(word.lower(), 0.0)  # 0 = not found (very rare)


def get_uniqueness_score(word):
    """Calculate uniqueness score from SUBTLEX-US Lg10WF.

    Uniqueness = 7 - Lg10WF.
    Higher uniqueness = more rare word.

    Examples:
    - 'the' (Lg10WF ~6.2) -> uniqueness ~0.8
    - 'computer' (Lg10WF ~3.8) -> uniqueness ~3.2
    - 'ayatollah' (Lg10WF ~1.5) -> uniqueness ~5.5
    - Unknown word (Lg10WF 0) -> uniqueness 6.8
    """
    lg10wf = get_subtlex_frequency(word)
    return max(0, 7 - lg10wf)


def build_brown_corpus_freq():
    """Build frequency dictionary from NLTK Brown corpus."""
    freq = Counter()
    for word in brown.words():
        freq[word.lower()] += 1
    total = sum(freq.values())
    # Normalize to frequency per million
    return {w: (c / total) * 1e6 for w, c in freq.items()}


# === Analysis Functions ===


def analyze_srt_file(srt_path, brown_freq):
    """Analyze word uniqueness for entire SRT file."""
    entries = parse_srt(srt_path)
    all_words = extract_words_with_timestamps(entries)

    if not all_words:
        print(f"Warning: No words found in {srt_path}")
        return None

    # Get total duration
    max_time = max(w["time"] for w in all_words)

    # Get unique words
    unique_words = list(set(w["word"] for w in all_words))

    # Calculate frequencies
    stop_words = set(stopwords.words("english"))

    word_data = []
    for word in unique_words:
        # Skip stopwords for target word candidates
        is_stopword = word in stop_words

        # Get POS tag and check if noun
        pos_tag = get_pos_tag(word)
        word_is_noun = pos_tag.startswith("NN")

        # Get frequency scores from SUBTLEX-US
        lg10wf = get_subtlex_frequency(word)
        uniqueness = get_uniqueness_score(word)

        # Brown corpus frequency (per million)
        brown_fpm = brown_freq.get(word, 0)

        # Count occurrences in entire SRT
        count_in_srt = sum(1 for w in all_words if w["word"] == word)

        # Get first occurrence time
        first_occurrence = next(w["time"] for w in all_words if w["word"] == word)

        word_data.append(
            {
                "word": word,
                "subtlex_lg10wf": lg10wf,  # SUBTLEX-US Lg10WF (0-6.3 scale)
                "uniqueness": uniqueness,
                "brown_fpm": brown_fpm,
                "count": count_in_srt,
                "first_time": first_occurrence,
                "is_stopword": is_stopword,
                "pos_tag": pos_tag,
                "is_noun": word_is_noun,
            }
        )

    # Sort by uniqueness (descending)
    word_data.sort(key=lambda x: x["uniqueness"], reverse=True)

    return {
        "srt_file": os.path.basename(srt_path),
        "total_duration": max_time,
        "total_words": len(all_words),
        "unique_words": len(unique_words),
        "word_data": word_data,
        "all_words": all_words,  # Raw word list with timestamps for 1-min bucket check
    }


def get_minute_bucket(time_seconds):
    """Get the 1-minute bucket for a given time. E.g., 83s -> 1 (1:00-2:00)"""
    return int(time_seconds // 60)


def find_target_word_candidates(
    analysis_results,
    all_words_by_file,
    uniqueness_range=(5.0, 6.8),
    min_count=1,
    nouns_only=True,
):
    """
    Find target word candidates with consistent uniqueness across all files.

    Criteria:
    - Not a stopword
    - Must be a noun (if nouns_only=True)
    - Uniqueness score within specified range (moderately rare)
    - The word must appear only ONCE in its 1-minute window (no repetition of same word)
    """
    candidates_per_file = {}

    for result in analysis_results:
        if result is None:
            continue

        file_name = result["srt_file"]
        all_words = all_words_by_file.get(file_name, [])

        # First pass: find all potential candidates based on uniqueness and POS
        potential_candidates = []
        for wd in result["word_data"]:
            if wd["is_stopword"]:
                continue
            if nouns_only and not wd["is_noun"]:
                continue
            if uniqueness_range[0] <= wd["uniqueness"] <= uniqueness_range[1]:
                potential_candidates.append(wd)

        # Second pass: filter to only keep words that appear once in their 1-min window
        final_candidates = []
        for wd in potential_candidates:
            word = wd["word"]
            word_time = wd["first_time"]
            minute_bucket = get_minute_bucket(word_time)
            bucket_start = minute_bucket * 60
            bucket_end = bucket_start + 60

            # Count how many times this word appears in its 1-minute bucket
            count_in_bucket = sum(
                1
                for w in all_words
                if w["word"] == word and bucket_start <= w["time"] < bucket_end
            )

            if count_in_bucket == 1:
                wd["minute_bucket"] = minute_bucket
                wd["count_in_minute"] = count_in_bucket
                final_candidates.append(wd)

        # Sort by time
        final_candidates.sort(key=lambda x: x["first_time"])

        candidates_per_file[file_name] = final_candidates

    return candidates_per_file


# === Plotting ===


def plot_word_frequencies_chronological(analysis_result, candidates, output_dir):
    """Create frequency plot for a single SRT file with words in chronological order.
    X-axis shows time (30s intervals), target word labels shown above red bars.
    """
    if analysis_result is None:
        return

    file_name = analysis_result["srt_file"]
    word_data = analysis_result["word_data"]

    # Filter out stopwords
    non_stopword_data = [w for w in word_data if not w["is_stopword"]]

    # Sort by time (chronological order)
    plot_data = sorted(non_stopword_data, key=lambda x: x["first_time"])

    # Get candidate words for this file
    candidate_words = set(c["word"] for c in candidates.get(file_name, []))

    # Prepare plot data
    words = [d["word"] for d in plot_data]
    times = [d["first_time"] for d in plot_data]
    uniqueness_scores = [d["uniqueness"] for d in plot_data]
    colors = ["red" if w in candidate_words else "steelblue" for w in words]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10))

    # Plot bars using time as x-axis
    bar_width = 1.2
    bars = ax.bar(
        times,
        uniqueness_scores,
        width=bar_width,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )

    # Add text labels above red bars (target word candidates)
    for i, (word, time, score) in enumerate(zip(words, times, uniqueness_scores)):
        if word in candidate_words:
            ax.text(
                time,
                score + 0.15,
                word,
                ha="center",
                va="bottom",
                fontsize=7,
                rotation=90,
                fontweight="bold",
                color="darkred",
            )

    # Set x-axis with 1-minute intervals
    max_time = max(times) if times else 360
    total_duration = analysis_result.get("total_duration", max_time)
    x_ticks = list(range(0, int(max_time) + 60, 60))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t//60}m" for t in x_ticks], fontsize=10)

    # Labels and title
    ax.set_xlabel("Time (minutes)", fontsize=12)
    ax.set_ylabel("Uniqueness Score (0-7, higher = more rare)", fontsize=12)
    ax.set_title(
        f'Word Uniqueness Analysis: {file_name}\n'
        f'Total duration: {total_duration/60:.1f} min | Total words: {analysis_result["total_words"]} | '
        f'Unique words: {len(plot_data)} | '
        f'Red = Target word candidates ({len(candidate_words)}, 1-min unique)',
        fontsize=14,
    )

    # Add uniqueness range indicators
    ax.axhline(
        y=5.0, color="green", linestyle="--", alpha=0.5, label="Target range (5.0-6.8)"
    )
    ax.axhline(y=6.8, color="green", linestyle="--", alpha=0.5)
    ax.fill_between([0, max_time + 10], 5.0, 6.8, alpha=0.1, color="green")

    # Add vertical grid lines at 1-minute intervals
    for t in x_ticks:
        ax.axvline(x=t, color="gray", linestyle="-", alpha=0.4, linewidth=1)

    ax.legend(loc="upper right")
    ax.set_ylim(0, 8.5)  # Extra space for labels
    ax.set_xlim(-5, max_time + 10)

    plt.tight_layout()

    # Save
    output_path = os.path.join(
        output_dir, f'{file_name.replace(".srt", "")}_word_uniqueness_chronological.png'
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved chronological plot: {output_path}")
    return output_path


def plot_word_frequencies(analysis_result, candidates, output_dir, show_top_n=100):
    """Create frequency plot for a single SRT file."""
    if analysis_result is None:
        return

    file_name = analysis_result["srt_file"]
    word_data = analysis_result["word_data"]

    # Filter out stopwords for cleaner visualization
    non_stopword_data = [w for w in word_data if not w["is_stopword"]]

    # Take top N by uniqueness
    plot_data = non_stopword_data[:show_top_n]

    # Get candidate words for this file
    candidate_words = set(c["word"] for c in candidates.get(file_name, []))

    # Prepare plot data
    words = [d["word"] for d in plot_data]
    uniqueness_scores = [d["uniqueness"] for d in plot_data]
    colors = ["red" if w in candidate_words else "steelblue" for w in words]

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 8))

    bars = ax.bar(
        range(len(words)),
        uniqueness_scores,
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add word labels
    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=90, fontsize=7)

    # Labels and title
    ax.set_xlabel("Words (sorted by uniqueness)", fontsize=12)
    ax.set_ylabel("Uniqueness Score (0-8, higher = more rare)", fontsize=12)
    ax.set_title(
        f'Word Uniqueness Analysis: {file_name}\n'
        f'6-min segment | Total words: {analysis_result["total_words"]} | '
        f'Unique words: {analysis_result["unique_words"]} | '
        f'Red = Target word candidates',
        fontsize=14,
    )

    # Add uniqueness range indicators
    ax.axhline(
        y=5.0, color="green", linestyle="--", alpha=0.5, label="Target range (5.0-6.8)"
    )
    ax.axhline(y=6.8, color="green", linestyle="--", alpha=0.5)
    ax.fill_between(range(len(words)), 5.0, 6.8, alpha=0.1, color="green")

    ax.legend(loc="upper right")
    ax.set_ylim(0, 8)

    plt.tight_layout()

    # Save
    output_path = os.path.join(
        output_dir, f'{file_name.replace(".srt", "")}_word_uniqueness.png'
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot: {output_path}")
    return output_path


def plot_comparison_across_files(analysis_results, candidates, output_dir):
    """Create comparison plot showing uniqueness distribution across all files."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for i, result in enumerate(analysis_results):
        if result is None or i >= 5:
            continue

        ax = axes[i]
        file_name = result["srt_file"]

        # Get non-stopword data
        word_data = [w for w in result["word_data"] if not w["is_stopword"]]

        # Get candidate words
        candidate_words = set(c["word"] for c in candidates.get(file_name, []))

        # Separate candidates and non-candidates
        candidate_uniqueness = [
            w["uniqueness"] for w in word_data if w["word"] in candidate_words
        ]
        non_candidate_uniqueness = [
            w["uniqueness"] for w in word_data if w["word"] not in candidate_words
        ]

        # Histogram
        ax.hist(
            non_candidate_uniqueness,
            bins=20,
            range=(0, 8),
            alpha=0.7,
            color="steelblue",
            label="All words",
            edgecolor="black",
        )
        ax.hist(
            candidate_uniqueness,
            bins=20,
            range=(0, 8),
            alpha=0.8,
            color="red",
            label="Target candidates",
            edgecolor="black",
        )

        ax.axvline(x=5.0, color="green", linestyle="--", alpha=0.7)
        ax.axvline(x=6.8, color="green", linestyle="--", alpha=0.7)

        ax.set_xlabel("Uniqueness Score")
        ax.set_ylabel("Word Count")
        ax.set_title(
            f"{file_name}\n(n={len(word_data)} unique words, {len(candidate_uniqueness)} candidates)"
        )
        ax.legend(loc="upper right", fontsize=8)

    # Hide unused subplot
    if len(analysis_results) < 6:
        axes[5].axis("off")

    plt.suptitle(
        "Word Uniqueness Distribution Comparison Across SRT Files\n"
        "Green lines indicate target word candidate range (5.0-6.8)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, "comparison_uniqueness_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved comparison plot: {output_path}")
    return output_path


def print_summary(analysis_results, candidates):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("WORD UNIQUENESS ANALYSIS SUMMARY")
    print("=" * 80)

    for result in analysis_results:
        if result is None:
            continue

        file_name = result["srt_file"]
        word_data = result["word_data"]
        non_stopword = [w for w in word_data if not w["is_stopword"]]

        # Calculate stats
        uniqueness_scores = [w["uniqueness"] for w in non_stopword]
        mean_uniqueness = np.mean(uniqueness_scores)
        std_uniqueness = np.std(uniqueness_scores)

        file_candidates = candidates.get(file_name, [])

        # Count nouns
        nouns = [w for w in non_stopword if w["is_noun"]]

        total_duration = result.get("total_duration", 0)

        print(f"\n{file_name}:")
        print(f"  Total duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
        print(f"  Total words: {result['total_words']}")
        print(f"  Unique words (non-stopword): {len(non_stopword)}")
        print(f"  Nouns: {len(nouns)}")
        print(f"  Mean uniqueness: {mean_uniqueness:.2f} (std: {std_uniqueness:.2f})")
        print(
            f"  Target word candidates (nouns, uniqueness 5.0-6.8, 1-min unique): {len(file_candidates)}"
        )

        if file_candidates:
            print(f"  Candidates (sorted by time):")
            for c in file_candidates:
                minute = c.get("minute_bucket", int(c["first_time"] // 60))
                print(
                    f"    [{minute}:00-{minute+1}:00] {c['word']} ({c['pos_tag']}): "
                    f"uniqueness={c['uniqueness']:.2f}, Lg10WF={c['subtlex_lg10wf']:.2f}, "
                    f"time={c['first_time']:.1f}s"
                )


# === Main ===


def main():
    # Paths
    srt_dir = Path("/Users/jeongin/morsequery/search/data/mp3/srt")
    output_dir = Path("/Users/jeongin/morsequery/search/data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get SRT files
    srt_files = sorted(srt_dir.glob("*.srt"))
    print(f"Found {len(srt_files)} SRT files")

    # Build Brown corpus frequency dictionary
    print("Building Brown corpus frequency dictionary...")
    brown_freq = build_brown_corpus_freq()
    print(f"Brown corpus: {len(brown_freq)} unique words")

    # Analyze each file (entire SRT)
    print("\nAnalyzing SRT files (entire duration)...")
    analysis_results = []
    all_words_by_file = {}
    for srt_path in srt_files:
        print(f"  Processing {srt_path.name}...")
        result = analyze_srt_file(srt_path, brown_freq)
        analysis_results.append(result)
        if result:
            all_words_by_file[result["srt_file"]] = result["all_words"]

    # Find target word candidates
    print("\nFinding target word candidates...")
    candidates = find_target_word_candidates(
        analysis_results, all_words_by_file, uniqueness_range=(5.0, 6.8)
    )

    # Print summary
    print_summary(analysis_results, candidates)

    # Create plots
    print("\nGenerating plots...")
    for result in analysis_results:
        plot_word_frequencies(result, candidates, output_dir)
        plot_word_frequencies_chronological(result, candidates, output_dir)

    plot_comparison_across_files(analysis_results, candidates, output_dir)

    # Save detailed results to JSON
    output_json = output_dir / "word_uniqueness_analysis.json"

    # Convert to serializable format
    serializable_results = []
    for result in analysis_results:
        if result:
            serializable_results.append(
                {
                    "srt_file": result["srt_file"],
                    "total_duration": result["total_duration"],
                    "total_words": result["total_words"],
                    "unique_words": result["unique_words"],
                    "word_data": result["word_data"],
                }
            )

    with open(output_json, "w") as f:
        json.dump(
            {
                "analysis_results": serializable_results,
                "candidates": {k: v for k, v in candidates.items()},
            },
            f,
            indent=2,
        )

    print(f"\nSaved detailed results to: {output_json}")
    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main()
