import json
import os
import statistics
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer

# Set matplotlib to use a non-GUI backend
matplotlib.use("Agg")

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
    # Versions without apostrophes
    "youre": "you are",
    "theyre": "they are",
    "im": "i am",
    "hes": "he is",
    "shes": "she is",
    "its": "it is",
    "thats": "that is",
    "whats": "what is",
    "theres": "there is",
    "whos": "who is",
    "wheres": "where is",
    "wont": "will not",
    "cant": "cannot",
    "dont": "do not",
    "doesnt": "does not",
    "didnt": "did not",
    "havent": "have not",
    "hasnt": "has not",
    "hadnt": "had not",
    "wouldnt": "would not",
    "shouldnt": "should not",
    "couldnt": "could not",
    "isnt": "is not",
    "arent": "are not",
    "wasnt": "was not",
    "werent": "were not",
}

# Define colors for each participant
PARTICIPANT_COLORS = {
    "pilot-p1": "#FF6B6B",  # Red
    "pilot-p2": "#4ECDC4",  # Teal
}

# Load the data
participants = {}
for participant_id in ["pilot-p1", "pilot-p2"]:
    file_path = f"logs/study_data/{participant_id}.json"
    try:
        with open(file_path, "r") as f:
            participants[participant_id] = json.load(f)["labeled_data"]
        print(f"Loaded {participant_id}: {len(participants[participant_id])} gestures")
    except Exception as e:
        print(f"Warning: Could not load {participant_id}: {e}")

if not participants:
    print("ERROR: No participant data loaded!")
    exit(1)

print("\n" + "=" * 80)
print("PILOT P1 + P2 COMBINED ANALYSIS")
print("=" * 80)

# 1. Number of gestures per video (per participant)
print("\n1. NUMBER OF GESTURES PER VIDEO (PER PARTICIPANT)")
print("-" * 80)

video_counts_per_participant = {}
for participant_id, labeled_data in participants.items():
    video_counts = Counter()
    for item in labeled_data:
        video_id = item["video_id"]
        video_index = item["video_index"]
        is_tutorial = item["is_tutorial"]

        if is_tutorial:
            key = "Tutorial"
        else:
            key = f"Video {video_index}"

        video_counts[key] += 1

    video_counts_per_participant[participant_id] = video_counts

    print(f"\n  {participant_id.upper()}:")
    for video, count in sorted(
        video_counts.items(), key=lambda x: (x[0].startswith("Tutorial"), x[0])
    ):
        print(f"    {video}: {count} gestures")

    total = sum(video_counts.values())
    tutorial = sum(
        count for video, count in video_counts.items() if video.startswith("Tutorial")
    )
    non_tutorial = total - tutorial
    print(f"    Total: {total}, Tutorial: {tutorial}, Non-tutorial: {non_tutorial}")

# 2. Target word length (per participant, excluding target_source="other")
print("\n\n2. TARGET WORD LENGTH (Number of words per participant)")
print("   NOTE: Excluding entries with target_source='other'")
print("-" * 80)

word_counts_per_participant = {}
long_targets = []  # Track unusually long targets

for participant_id, labeled_data in participants.items():
    word_counts = []
    for item in labeled_data:
        target_word = item["target_word"]
        target_source = item.get("target_source", "")

        # Skip if target_word is invalid or if target_source is "other"
        if not target_word or target_word == "잘못 누름" or target_source == "other":
            continue

        num_words = len(target_word.split())
        word_counts.append(num_words)

        # Track unusually long targets (4+ words)
        if num_words >= 4:
            long_targets.append(
                {
                    "participant": participant_id,
                    "target_word": target_word,
                    "num_words": num_words,
                    "video_index": item["video_index"],
                }
            )

    word_counts_per_participant[participant_id] = word_counts

    print(f"\n  {participant_id.upper()}:")
    print(
        f"    Min: {min(word_counts)}, Max: {max(word_counts)}, Mean: {statistics.mean(word_counts):.2f}, Median: {statistics.median(word_counts):.2f}"
    )

    word_count_dist = Counter(word_counts)
    print("    Distribution:")
    for num_words in sorted(word_count_dist.keys()):
        count = word_count_dist[num_words]
        percentage = (count / len(word_counts)) * 100
        print(
            f"      {num_words} word{'s' if num_words > 1 else ''}: {count} ({percentage:.1f}%)"
        )

# Print unusually long targets
if long_targets:
    print("\n  UNUSUALLY LONG TARGETS (4+ words):")
    print("-" * 80)
    for item in sorted(long_targets, key=lambda x: x["num_words"], reverse=True):
        print(
            f"    {item['participant']}, Video {item['video_index']}: \"{item['target_word']}\" ({item['num_words']} words)"
        )

# 3. Intent types (per participant)
print("\n\n3. INTENT TYPES (PER PARTICIPANT)")
print("-" * 80)

intent_counts_per_participant = {}
all_intent_types = set()

for participant_id, labeled_data in participants.items():
    intent_type_counts = Counter()
    multi_intent_count = 0

    for item in labeled_data:
        intent_types = item["intent_types"]
        num_intents = len(intent_types)

        if num_intents > 1:
            multi_intent_count += 1

        for intent in intent_types:
            intent_type_counts[intent] += 1
            all_intent_types.add(intent)

    intent_counts_per_participant[participant_id] = intent_type_counts

    print(f"\n  {participant_id.upper()}:")
    print(f"    Unique intent types: {len(intent_type_counts)}")
    print("    Distribution:")
    for intent, count in intent_type_counts.most_common():
        percentage = (count / len(labeled_data)) * 100
        print(f"      {intent}: {count} ({percentage:.1f}%)")

    print(
        f"    Gestures with multiple intents: {multi_intent_count} ({(multi_intent_count/len(labeled_data))*100:.1f}%)"
    )

# Print all "other" values
print("\n  ALL 'OTHER' INTENT VALUES:")
print("-" * 80)
for participant_id, labeled_data in participants.items():
    other_intents = []
    for item in labeled_data:
        if "other" in item["intent_types"]:
            other_intents.append(
                {
                    "video_index": item["video_index"],
                    "target_word": item["target_word"],
                    "intent_types": item["intent_types"],
                    "intent_other_text": item.get("intent_other_text", ""),
                }
            )

    if other_intents:
        print(f"\n  {participant_id.upper()} (Total: {len(other_intents)}):")
        for i, item in enumerate(other_intents, 1):
            print(f"    {i}. Video {item['video_index']}: \"{item['target_word']}\"")
            print(f"       intents: {item['intent_types']}")
            if item["intent_other_text"]:
                print(f"       other_text: \"{item['intent_other_text']}\"")
    else:
        print(f"\n  {participant_id.upper()}: No 'other' intents found")

# 4. Time gap between pressed_timestamp and target_word_timestamp (per participant)
print("\n\n4. TIME GAP (pressed_timestamp - target_word_timestamp, per participant)")
print("-" * 80)

time_gaps_per_participant = {}
long_time_gaps = []  # Track unusually long time gaps

for participant_id, labeled_data in participants.items():
    time_gaps = []
    for item in labeled_data:
        target_word_timestamp = item.get("target_word_timestamp")
        if target_word_timestamp is not None:
            pressed_timestamp = item["pressed_timestamp"]
            time_gap = pressed_timestamp - target_word_timestamp
            time_gaps.append(time_gap)

            # Track unusually long time gaps (>8 seconds)
            if time_gap > 4.0:
                long_time_gaps.append(
                    {
                        "participant": participant_id,
                        "time_gap": time_gap,
                        "target_word": item["target_word"],
                        "video_index": item["video_index"],
                        "pressed_time": pressed_timestamp,
                    }
                )

    time_gaps_per_participant[participant_id] = time_gaps

    print(f"\n  {participant_id.upper()}:")
    print(f"    Valid entries: {len(time_gaps)}")
    print(f"    Min: {min(time_gaps):.2f}s, Max: {max(time_gaps):.2f}s")
    print(
        f"    Mean: {statistics.mean(time_gaps):.2f}s, Median: {statistics.median(time_gaps):.2f}s"
    )
    print(f"    Std dev: {statistics.stdev(time_gaps):.2f}s")

    ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, float("inf"))]
    print("    Distribution:")
    for start, end in ranges:
        count = sum(1 for gap in time_gaps if start <= gap < end)
        percentage = (count / len(time_gaps)) * 100
        if end == float("inf"):
            print(f"      {start}+ seconds: {count} ({percentage:.1f}%)")
        else:
            print(f"      {start}-{end} seconds: {count} ({percentage:.1f}%)")

# Print unusually long time gaps
if long_time_gaps:
    print("\n  UNUSUALLY LONG TIME GAPS (>8 seconds):")
    print("-" * 80)
    for item in sorted(long_time_gaps, key=lambda x: x["time_gap"], reverse=True):
        print(
            f"    {item['participant']}, Video {item['video_index']}: \"{item['target_word']}\" - {item['time_gap']:.2f}s (pressed at {item['pressed_time']:.1f}s)"
        )

# 5. Naive approach evaluation
print("\n\n5. NAIVE APPROACH EVALUATION (WITH LEMMATIZATION)")
print("-" * 80)
print("  Approach: Within 5 seconds, pick word with lowest openlexicon value")
print("  Improvements: Contraction expansion + Lemmatization (verb/noun/adj)")
print()

# Load OpenLexicon
lexicon_path = "data/lexicon/OpenLexicon.xlsx"
print(f"  Loading OpenLexicon from {lexicon_path}...")
try:
    lexicon_df = pd.read_excel(lexicon_path)
    lexicon_dict = {}
    for _, row in lexicon_df.iterrows():
        word = str(row["ortho"]).lower()
        freq_value = row["English_Lexicon_Project__LgSUBTLWF"]
        lexicon_dict[word] = freq_value
    print(f"  Lexicon loaded: {len(lexicon_dict)} words\n")
except Exception as e:
    print(f"  ERROR: Could not load OpenLexicon: {e}")
    print("\n" + "=" * 80)
    exit(1)

# Load transcripts
transcripts = {}
transcript_dir = "data/transcripts"
print(f"  Loading transcripts from {transcript_dir}/...")
for video_id in ["U6fI3brP8V4", "bAkuNXtgrLA", "cEVAjm_ETtY", "MxovSnvSO4E"]:
    transcript_path = os.path.join(transcript_dir, f"{video_id}.json")
    try:
        with open(transcript_path, "r") as f:
            transcripts[video_id] = json.load(f)
    except Exception as e:
        print(f"    Warning: Could not load {video_id}: {e}")

print(f"  Transcripts loaded: {len(transcripts)} videos\n")


def preprocess_word(word):
    """Preprocess word for lexicon lookup"""
    # Remove punctuation and lowercase
    cleaned = "".join(c for c in word if c.isalnum()).lower()

    if not cleaned or len(cleaned) < 3:
        return None

    # Expand contractions first
    if cleaned in CONTRACTIONS:
        expanded = CONTRACTIONS[cleaned].split()[0]
        cleaned = expanded

    # Lemmatize: try as verb, then noun, then adjective
    lemma_v = lemmatizer.lemmatize(cleaned, pos="v")
    if lemma_v != cleaned:
        return lemma_v

    lemma_n = lemmatizer.lemmatize(cleaned, pos="n")
    if lemma_n != cleaned:
        return lemma_n

    lemma_a = lemmatizer.lemmatize(cleaned, pos="a")
    if lemma_a != cleaned:
        return lemma_a

    return cleaned


def get_openlexicon_value(word):
    """Get openlexicon frequency value for a word (lower = rarer)"""
    # First check if it's a contraction - if so, skip it
    cleaned_word = "".join(c for c in word if c.isalnum()).lower()
    if cleaned_word in CONTRACTIONS:
        return float("inf")  # Skip contractions

    cleaned = preprocess_word(word)
    if not cleaned:
        return float("inf")  # Skip punctuation-only words

    # Check if word is in lexicon
    if cleaned in lexicon_dict:
        return lexicon_dict[cleaned]
    else:
        # Not in lexicon = very rare, assign lowest value
        return -1.0


def find_words_in_window(transcript, pressed_time, window_seconds=5.0):
    """Find all words within window_seconds before pressed_time"""
    words_in_window = []

    if "segments" not in transcript:
        return words_in_window

    for segment in transcript["segments"]:
        if "words" not in segment:
            continue

        for word_obj in segment["words"]:
            word_start = word_obj["start"]
            word_text = word_obj["word"].strip()

            # Skip contractions
            cleaned_word = "".join(c for c in word_text if c.isalnum()).lower()
            if cleaned_word in CONTRACTIONS:
                continue

            # Check if word is within the time window
            if pressed_time - window_seconds <= word_start <= pressed_time:
                openlexicon_val = get_openlexicon_value(word_text)
                words_in_window.append(
                    {
                        "word": word_text,
                        "start": word_start,
                        "openlexicon": openlexicon_val,
                    }
                )

    return words_in_window


# Evaluate naive approach per participant
print("  Evaluating naive approach on labeled gestures...\n")

results_per_participant = {}

for participant_id, labeled_data in participants.items():
    correct_matches = 0
    total_evaluated = 0
    no_words_found = 0
    details = []

    for item in labeled_data:
        # Skip tutorial and invalid entries
        if item["is_tutorial"]:
            continue
        if item.get("target_word_timestamp") is None:
            continue
        if item["target_source"] != "transcript":
            continue

        video_id = item["video_id"]
        pressed_timestamp = item["pressed_timestamp"]
        target_word = item["target_word"]

        # Get transcript
        if video_id not in transcripts:
            continue

        transcript = transcripts[video_id]

        # Find words in 5-second window
        words_in_window = find_words_in_window(
            transcript, pressed_timestamp, window_seconds=5.0
        )

        if not words_in_window:
            no_words_found += 1
            continue

        # Find word with lowest openlexicon value
        best_word = min(words_in_window, key=lambda x: x["openlexicon"])

        # Clean both target and predicted for comparison
        target_cleaned = preprocess_word(target_word.split()[0]) if target_word else ""
        predicted_cleaned = preprocess_word(best_word["word"])

        # Check if match
        is_match = target_cleaned == predicted_cleaned
        if is_match:
            correct_matches += 1

        total_evaluated += 1

        # Store details for error analysis
        details.append(
            {
                "video_id": video_id,
                "pressed_time": pressed_timestamp,
                "target": target_word,
                "predicted": best_word["word"],
                "predicted_openlexicon": best_word["openlexicon"],
                "is_match": is_match,
                "num_candidates": len(words_in_window),
            }
        )

    results_per_participant[participant_id] = {
        "correct": correct_matches,
        "total": total_evaluated,
        "no_words": no_words_found,
        "details": details,
    }

    # Print results
    print(f"  {participant_id.upper()}:")
    print(f"    Total gestures evaluated: {total_evaluated}")
    print(f"    Correct matches: {correct_matches}")
    if total_evaluated > 0:
        accuracy = (correct_matches / total_evaluated) * 100
        print(f"    Accuracy: {accuracy:.1f}%")
    else:
        print("    Accuracy: N/A (no valid gestures)")
    print(f"    No words found in window: {no_words_found}")

    # Analyze errors
    all_errors = [d for d in details if not d["is_match"]]

    # Count multi-word target errors
    multi_word_errors = [err for err in all_errors if len(err["target"].split()) > 1]
    single_word_errors = [err for err in all_errors if len(err["target"].split()) == 1]

    if all_errors:
        print("\n    Error Analysis:")
        print(f"      Total errors: {len(all_errors)}")
        print(
            f"      Multi-word target errors: {len(multi_word_errors)} ({len(multi_word_errors)/len(all_errors)*100:.1f}%)"
        )
        print(
            f"      Single-word target errors: {len(single_word_errors)} ({len(single_word_errors)/len(all_errors)*100:.1f}%)"
        )

    # Show sample errors
    sample_errors = all_errors[:10]
    if sample_errors:
        print(f"\n    Sample errors (showing first {len(sample_errors)}):")
        for i, err in enumerate(sample_errors, 1):
            num_words = len(err["target"].split())
            word_indicator = f" [{num_words}w]" if num_words > 1 else ""
            print(
                f"      {i}. Target: '{err['target']}'{word_indicator} | Predicted: '{err['predicted']}' (openlexicon={err['predicted_openlexicon']:.2f})"
            )
            print(
                f"         Time: {err['pressed_time']:.1f}s | Candidates: {err['num_candidates']}"
            )
    print()

print("\n" + "=" * 80)

# Generate combined distribution graphs
print("\nGenerating combined distribution graphs...")
print("-" * 80)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Pilot P1 + P2 Combined Analysis", fontsize=16, fontweight="bold")

# 1. Gestures per video (grouped bar chart)
ax1 = axes[0, 0]

# Get all unique video keys
all_videos = set()
for counts in video_counts_per_participant.values():
    all_videos.update(counts.keys())

# Sort videos
sorted_videos = sorted(all_videos, key=lambda x: (x.startswith("Tutorial"), x))

# Prepare data
x = range(len(sorted_videos))
width = 0.35
participant_ids = list(participants.keys())

for i, participant_id in enumerate(participant_ids):
    counts = [
        video_counts_per_participant[participant_id].get(video, 0)
        for video in sorted_videos
    ]
    bars = ax1.bar(
        [xi + i * width for xi in x],
        counts,
        width,
        label=participant_id,
        color=PARTICIPANT_COLORS[participant_id],
        alpha=0.8,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

ax1.set_xlabel("Video", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Gestures", fontsize=12, fontweight="bold")
ax1.set_title("1. Number of Gestures per Video", fontsize=14, fontweight="bold")
ax1.set_xticks([xi + width / 2 for xi in x])
ax1.set_xticklabels(sorted_videos, rotation=45, ha="right")
ax1.legend()
ax1.grid(axis="y", alpha=0.3)

# 2. Target word length distribution (overlaid bars)
ax2 = axes[0, 1]

# Get all word lengths
all_word_lengths = set()
for word_counts in word_counts_per_participant.values():
    all_word_lengths.update(word_counts)

sorted_word_lengths = sorted(all_word_lengths)

for i, participant_id in enumerate(participant_ids):
    word_count_dist = Counter(word_counts_per_participant[participant_id])
    counts = [word_count_dist.get(length, 0) for length in sorted_word_lengths]

    bars = ax2.bar(
        [x + i * width for x in sorted_word_lengths],
        counts,
        width,
        label=participant_id,
        color=PARTICIPANT_COLORS[participant_id],
        alpha=0.7,
    )

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

ax2.set_xlabel("Number of Words in Target", fontsize=12, fontweight="bold")
ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax2.set_title(
    "2. Target Word Length Distribution\n(excluding target_source='other')",
    fontsize=14,
    fontweight="bold",
)
ax2.legend()
ax2.grid(axis="y", alpha=0.3)

# 3. Intent types distribution (grouped horizontal bars)
ax3 = axes[1, 0]

# Get all intent types
sorted_intents = sorted(all_intent_types)

y = range(len(sorted_intents))
height_bar = 0.35

for i, participant_id in enumerate(participant_ids):
    counts = [
        intent_counts_per_participant[participant_id].get(intent, 0)
        for intent in sorted_intents
    ]
    bars = ax3.barh(
        [yi + i * height_bar for yi in y],
        counts,
        height_bar,
        label=participant_id,
        color=PARTICIPANT_COLORS[participant_id],
        alpha=0.8,
    )

    # Add value labels
    for bar, count in zip(bars, counts):
        width_val = bar.get_width()
        if width_val > 0:
            ax3.text(
                width_val,
                bar.get_y() + bar.get_height() / 2.0,
                f" {int(count)}",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
            )

ax3.set_xlabel("Frequency", fontsize=12, fontweight="bold")
ax3.set_ylabel("Intent Type", fontsize=12, fontweight="bold")
ax3.set_title("3. Intent Types Distribution", fontsize=14, fontweight="bold")
ax3.set_yticks([yi + height_bar / 2 for yi in y])
ax3.set_yticklabels(sorted_intents)
ax3.legend()
ax3.grid(axis="x", alpha=0.3)

# 4. Time gap distribution (overlaid bars)
ax4 = axes[1, 1]

ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, float("inf"))]
range_labels = []
for start, end in ranges:
    if end == float("inf"):
        range_labels.append(f"{start}+s")
    else:
        range_labels.append(f"{start}-{end}s")

x_pos = range(len(range_labels))
bar_width = 0.35

for i, participant_id in enumerate(participant_ids):
    time_gaps = time_gaps_per_participant[participant_id]
    counts = [
        sum(1 for gap in time_gaps if start <= gap < end) for start, end in ranges
    ]

    bars = ax4.bar(
        [x + i * bar_width for x in x_pos],
        counts,
        bar_width,
        label=participant_id,
        color=PARTICIPANT_COLORS[participant_id],
        alpha=0.7,
    )

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        if height > 0:
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

ax4.set_xlabel("Time Gap (seconds)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax4.set_title(
    "4. Time Gap Distribution (Pressed - Target Word)", fontsize=14, fontweight="bold"
)
ax4.set_xticks([x + bar_width / 2 for x in x_pos])
ax4.set_xticklabels(range_labels, rotation=45, ha="right")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

# Add statistics text box
stats_lines = []
for participant_id, time_gaps in time_gaps_per_participant.items():
    mean = statistics.mean(time_gaps)
    median = statistics.median(time_gaps)
    stats_lines.append(f"{participant_id}: μ={mean:.2f}s, Med={median:.2f}s")

stats_text = "\n".join(stats_lines)
ax4.text(
    0.98,
    0.97,
    stats_text,
    transform=ax4.transAxes,
    fontsize=9,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig("pilot_combined_distributions.png", dpi=300, bbox_inches="tight")
print("  Saved graph to: pilot_combined_distributions.png")

print("\n" + "=" * 80)
print("COMBINED ANALYSIS COMPLETE!")
print("=" * 80)
