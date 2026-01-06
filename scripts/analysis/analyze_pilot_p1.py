import json
import os
import statistics
from collections import Counter

import matplotlib
import matplotlib.pyplot as plt
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
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

# Common English contractions (with and without apostrophes for matching)
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
    # Versions without apostrophes (for matching after cleaning)
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

# Load the data
with open("logs/study_data/pilot-p1.json", "r") as f:
    data = json.load(f)

labeled_data = data["labeled_data"]

print("=" * 80)
print("PILOT-P1 ANALYSIS (IMPROVED WITH LEMMATIZATION)")
print("=" * 80)

# 1. Number of gestures per video
print("\n1. NUMBER OF GESTURES PER VIDEO")
print("-" * 80)
video_counts = Counter()
for item in labeled_data:
    video_id = item["video_id"]
    video_index = item["video_index"]
    is_tutorial = item["is_tutorial"]

    if is_tutorial:
        key = f"Tutorial (index={video_index})"
    else:
        key = f"Video {video_index}"

    video_counts[key] += 1

for video, count in sorted(
    video_counts.items(), key=lambda x: (x[0].startswith("Tutorial"), x[0])
):
    print(f"  {video}: {count} gestures")

print(f"\n  Total gestures: {len(labeled_data)}")
print(f"  Tutorial gestures: {sum(1 for item in labeled_data if item['is_tutorial'])}")
print(
    f"  Non-tutorial gestures: {sum(1 for item in labeled_data if not item['is_tutorial'])}"
)

# 2. Number of words in target_word (split by space)
print("\n\n2. TARGET WORD LENGTH (Number of words)")
print("-" * 80)
word_counts = []
for item in labeled_data:
    target_word = item["target_word"]
    if target_word and target_word != "잘못 누름":
        num_words = len(target_word.split())
        word_counts.append(num_words)

print(f"  Min words: {min(word_counts)}")
print(f"  Max words: {max(word_counts)}")
print(f"  Mean words: {statistics.mean(word_counts):.2f}")
print(f"  Median words: {statistics.median(word_counts):.2f}")

word_count_dist = Counter(word_counts)
print("\n  Distribution:")
for num_words in sorted(word_count_dist.keys()):
    count = word_count_dist[num_words]
    percentage = (count / len(word_counts)) * 100
    print(
        f"    {num_words} word{'s' if num_words > 1 else ''}: {count} ({percentage:.1f}%)"
    )

# 3. Number of intent types
print("\n\n3. INTENT TYPES")
print("-" * 80)
intent_type_counts = Counter()
multi_intent_count = 0
for item in labeled_data:
    intent_types = item["intent_types"]
    num_intents = len(intent_types)

    if num_intents > 1:
        multi_intent_count += 1

    for intent in intent_types:
        intent_type_counts[intent] += 1

print(f"  Total unique intent types: {len(intent_type_counts)}")
print("\n  Intent type distribution:")
for intent, count in intent_type_counts.most_common():
    percentage = (count / len(labeled_data)) * 100
    print(f"    {intent}: {count} ({percentage:.1f}%)")

print(
    f"\n  Gestures with multiple intents: {multi_intent_count} ({(multi_intent_count/len(labeled_data))*100:.1f}%)"
)

# 4. Time gap between pressed_timestamp and target_word_timestamp
print("\n\n4. TIME GAP (pressed_timestamp - target_word_timestamp)")
print("-" * 80)
time_gaps = []
for item in labeled_data:
    target_word_timestamp = item.get("target_word_timestamp")
    if target_word_timestamp is not None:
        pressed_timestamp = item["pressed_timestamp"]
        time_gap = pressed_timestamp - target_word_timestamp
        time_gaps.append(time_gap)

print(f"  Valid entries: {len(time_gaps)} / {len(labeled_data)}")
print(f"  Min gap: {min(time_gaps):.2f} seconds")
print(f"  Max gap: {max(time_gaps):.2f} seconds")
print(f"  Mean gap: {statistics.mean(time_gaps):.2f} seconds")
print(f"  Median gap: {statistics.median(time_gaps):.2f} seconds")
print(f"  Std dev: {statistics.stdev(time_gaps):.2f} seconds")

ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, float("inf"))]
print("\n  Distribution:")
for start, end in ranges:
    count = sum(1 for gap in time_gaps if start <= gap < end)
    percentage = (count / len(time_gaps)) * 100
    if end == float("inf"):
        print(f"    {start}+ seconds: {count} ({percentage:.1f}%)")
    else:
        print(f"    {start}-{end} seconds: {count} ({percentage:.1f}%)")

# 5. Naive approach evaluation with improved preprocessing
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
    """Preprocess word for lexicon lookup (from app.py logic)

    1. Remove punctuation and lowercase
    2. Expand contractions (you're -> you are)
    3. Lemmatize to base form (declares -> declare, remains -> remain)
    """
    # Remove punctuation and lowercase
    cleaned = "".join(c for c in word if c.isalnum()).lower()

    if not cleaned or len(cleaned) < 3:
        return None

    # Expand contractions first
    if cleaned in CONTRACTIONS:
        # Return the first word of the expansion
        expanded = CONTRACTIONS[cleaned].split()[0]
        cleaned = expanded

    # Lemmatize: try as verb, then noun, then adjective
    # Verbs: declares -> declare, remains -> remain
    lemma_v = lemmatizer.lemmatize(cleaned, pos="v")
    if lemma_v != cleaned:
        return lemma_v

    # Nouns: cats -> cat
    lemma_n = lemmatizer.lemmatize(cleaned, pos="n")
    if lemma_n != cleaned:
        return lemma_n

    # Adjectives: better -> good
    lemma_a = lemmatizer.lemmatize(cleaned, pos="a")
    if lemma_a != cleaned:
        return lemma_a

    # No lemmatization needed, return as-is
    return cleaned


def get_openlexicon_value(word):
    """Get openlexicon frequency value for a word (lower = rarer)"""
    # First check if it's a contraction - if so, skip it
    cleaned_word = "".join(c for c in word if c.isalnum()).lower()
    if cleaned_word in CONTRACTIONS:
        # DEBUG: Print when we detect a contraction
        # print(f"[DEBUG] Skipping contraction: {word} -> {cleaned_word}")
        return float("inf")  # Skip contractions by assigning very high value

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

            # Skip contractions (you're, I'm, etc.)
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


# Evaluate naive approach
print("  Evaluating naive approach on labeled gestures...\n")

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

# Print results
print("  Results:")
print(f"    Total gestures evaluated: {total_evaluated}")
print(f"    Correct matches: {correct_matches}")
print(f"    Accuracy: {(correct_matches/total_evaluated)*100:.1f}%")
print(f"    No words found in window: {no_words_found}")

print("\n  Sample errors (showing first 30):")
errors = [d for d in details if not d["is_match"]][:30]
for i, err in enumerate(errors, 1):
    print(
        f"    {i}. Target: '{err['target']}' | Predicted: '{err['predicted']}' (openlexicon={err['predicted_openlexicon']:.2f})"
    )
    print(
        f"       Time: {err['pressed_time']:.1f}s | Candidates: {err['num_candidates']}"
    )

print("\n" + "=" * 80)

# Generate distribution graphs
print("\nGenerating distribution graphs...")
print("-" * 80)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Pilot P1 Analysis - Distribution Graphs", fontsize=16, fontweight="bold")

# 1. Gestures per video
ax1 = axes[0, 0]
video_labels = []
video_values = []
for video, count in sorted(
    video_counts.items(), key=lambda x: (x[0].startswith("Tutorial"), x[0])
):
    video_labels.append(video.replace("Tutorial (index=-1)", "Tutorial"))
    video_values.append(count)

bars1 = ax1.bar(
    range(len(video_labels)),
    video_values,
    color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
)
ax1.set_xlabel("Video", fontsize=12, fontweight="bold")
ax1.set_ylabel("Number of Gestures", fontsize=12, fontweight="bold")
ax1.set_title("1. Number of Gestures per Video", fontsize=14, fontweight="bold")
ax1.set_xticks(range(len(video_labels)))
ax1.set_xticklabels(video_labels, rotation=45, ha="right")
ax1.grid(axis="y", alpha=0.3)

# Add value labels on bars
for i, (bar, value) in enumerate(zip(bars1, video_values)):
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(value)}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

# 2. Target word length distribution
ax2 = axes[0, 1]
word_lengths = sorted(word_count_dist.keys())
word_length_counts = [word_count_dist[l] for l in word_lengths]

bars2 = ax2.bar(
    word_lengths, word_length_counts, color="#FF6B6B", alpha=0.7, edgecolor="black"
)
ax2.set_xlabel("Number of Words in Target", fontsize=12, fontweight="bold")
ax2.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax2.set_title("2. Target Word Length Distribution", fontsize=14, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

# Add value labels and percentages
for i, (bar, count) in enumerate(zip(bars2, word_length_counts)):
    height = bar.get_height()
    percentage = (count / len(word_counts)) * 100
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(count)}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

# 3. Intent types distribution
ax3 = axes[1, 0]
intent_names = [intent for intent, _ in intent_type_counts.most_common()]
intent_counts = [count for _, count in intent_type_counts.most_common()]

bars3 = ax3.barh(
    intent_names, intent_counts, color=["#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"]
)
ax3.set_xlabel("Frequency", fontsize=12, fontweight="bold")
ax3.set_ylabel("Intent Type", fontsize=12, fontweight="bold")
ax3.set_title("3. Intent Types Distribution", fontsize=14, fontweight="bold")
ax3.grid(axis="x", alpha=0.3)

# Add value labels and percentages
for i, (bar, count, intent) in enumerate(zip(bars3, intent_counts, intent_names)):
    width = bar.get_width()
    percentage = (count / len(labeled_data)) * 100
    ax3.text(
        width,
        bar.get_y() + bar.get_height() / 2.0,
        f" {int(count)} ({percentage:.1f}%)",
        ha="left",
        va="center",
        fontweight="bold",
    )

# 4. Time gap distribution
ax4 = axes[1, 1]
range_labels = []
range_counts = []
range_colors = []

for start, end in ranges:
    count = sum(1 for gap in time_gaps if start <= gap < end)
    range_counts.append(count)
    if end == float("inf"):
        range_labels.append(f"{start}+s")
    else:
        range_labels.append(f"{start}-{end}s")

    # Color by urgency (closer = more urgent = warmer color)
    if end <= 2:
        range_colors.append("#FF6B6B")  # Red for 0-2s
    elif end <= 3:
        range_colors.append("#FFA07A")  # Light red for 2-3s
    elif end <= 5:
        range_colors.append("#FFD700")  # Gold for 3-5s
    else:
        range_colors.append("#87CEEB")  # Sky blue for 5+s

bars4 = ax4.bar(
    range_labels, range_counts, color=range_colors, alpha=0.7, edgecolor="black"
)
ax4.set_xlabel("Time Gap (seconds)", fontsize=12, fontweight="bold")
ax4.set_ylabel("Frequency", fontsize=12, fontweight="bold")
ax4.set_title(
    "4. Time Gap Distribution (Pressed - Target Word)", fontsize=14, fontweight="bold"
)
ax4.grid(axis="y", alpha=0.3)
ax4.tick_params(axis="x", rotation=45)

# Add value labels and percentages
for i, (bar, count) in enumerate(zip(bars4, range_counts)):
    height = bar.get_height()
    percentage = (count / len(time_gaps)) * 100 if len(time_gaps) > 0 else 0
    ax4.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(count)}\n({percentage:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
        fontweight="bold",
    )

# Add statistics text box to time gap plot
stats_text = f"Mean: {statistics.mean(time_gaps):.2f}s\nMedian: {statistics.median(time_gaps):.2f}s\nStd Dev: {statistics.stdev(time_gaps):.2f}s"
ax4.text(
    0.98,
    0.97,
    stats_text,
    transform=ax4.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

plt.tight_layout()
plt.savefig("pilot_p1_distributions.png", dpi=300, bbox_inches="tight")
print("  Saved graph to: pilot_p1_distributions.png")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)
