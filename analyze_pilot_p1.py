import json
from collections import Counter
import statistics
import pandas as pd
import os
import re

# Load the data
with open('logs/study_data/pilot-p1.json', 'r') as f:
    data = json.load(f)

labeled_data = data['labeled_data']

print("=" * 80)
print("PILOT-P1 ANALYSIS")
print("=" * 80)

# 1. Number of gestures per video
print("\n1. NUMBER OF GESTURES PER VIDEO")
print("-" * 80)
video_counts = Counter()
for item in labeled_data:
    video_id = item['video_id']
    video_index = item['video_index']
    is_tutorial = item['is_tutorial']

    if is_tutorial:
        key = f"Tutorial (index={video_index})"
    else:
        key = f"Video {video_index}"

    video_counts[key] += 1

for video, count in sorted(video_counts.items(), key=lambda x: (x[0].startswith('Tutorial'), x[0])):
    print(f"  {video}: {count} gestures")

print(f"\n  Total gestures: {len(labeled_data)}")
print(f"  Tutorial gestures: {sum(1 for item in labeled_data if item['is_tutorial'])}")
print(f"  Non-tutorial gestures: {sum(1 for item in labeled_data if not item['is_tutorial'])}")

# 2. Number of words in target_word (split by space)
print("\n\n2. TARGET WORD LENGTH (Number of words)")
print("-" * 80)
word_counts = []
for item in labeled_data:
    target_word = item['target_word']
    if target_word and target_word != "잘못 누름":  # Exclude "wrong press" entries
        num_words = len(target_word.split())
        word_counts.append(num_words)

print(f"  Min words: {min(word_counts)}")
print(f"  Max words: {max(word_counts)}")
print(f"  Mean words: {statistics.mean(word_counts):.2f}")
print(f"  Median words: {statistics.median(word_counts):.2f}")

# Distribution
word_count_dist = Counter(word_counts)
print(f"\n  Distribution:")
for num_words in sorted(word_count_dist.keys()):
    count = word_count_dist[num_words]
    percentage = (count / len(word_counts)) * 100
    print(f"    {num_words} word{'s' if num_words > 1 else ''}: {count} ({percentage:.1f}%)")

# 3. Number of intent types
print("\n\n3. INTENT TYPES")
print("-" * 80)
intent_type_counts = Counter()
multi_intent_count = 0
for item in labeled_data:
    intent_types = item['intent_types']
    num_intents = len(intent_types)

    if num_intents > 1:
        multi_intent_count += 1

    for intent in intent_types:
        intent_type_counts[intent] += 1

print(f"  Total unique intent types: {len(intent_type_counts)}")
print(f"\n  Intent type distribution:")
for intent, count in intent_type_counts.most_common():
    percentage = (count / len(labeled_data)) * 100
    print(f"    {intent}: {count} ({percentage:.1f}%)")

print(f"\n  Gestures with multiple intents: {multi_intent_count} ({(multi_intent_count/len(labeled_data))*100:.1f}%)")

# Count combinations
intent_combinations = Counter()
for item in labeled_data:
    intent_types = tuple(sorted(item['intent_types']))
    intent_combinations[intent_types] += 1

print(f"\n  Top intent combinations:")
for combo, count in intent_combinations.most_common(10):
    percentage = (count / len(labeled_data)) * 100
    print(f"    {', '.join(combo)}: {count} ({percentage:.1f}%)")

# 4. Time gap between pressed_timestamp and target_word_timestamp
print("\n\n4. TIME GAP (pressed_timestamp - target_word_timestamp)")
print("-" * 80)
time_gaps = []
for item in labeled_data:
    target_word_timestamp = item.get('target_word_timestamp')
    if target_word_timestamp is not None:  # Skip entries without timestamp
        pressed_timestamp = item['pressed_timestamp']
        time_gap = pressed_timestamp - target_word_timestamp
        time_gaps.append(time_gap)

print(f"  Valid entries: {len(time_gaps)} / {len(labeled_data)}")
print(f"  Min gap: {min(time_gaps):.2f} seconds")
print(f"  Max gap: {max(time_gaps):.2f} seconds")
print(f"  Mean gap: {statistics.mean(time_gaps):.2f} seconds")
print(f"  Median gap: {statistics.median(time_gaps):.2f} seconds")
print(f"  Std dev: {statistics.stdev(time_gaps):.2f} seconds")

# Distribution by ranges
print(f"\n  Distribution:")
ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10), (10, float('inf'))]
for start, end in ranges:
    count = sum(1 for gap in time_gaps if start <= gap < end)
    percentage = (count / len(time_gaps)) * 100
    if end == float('inf'):
        print(f"    {start}+ seconds: {count} ({percentage:.1f}%)")
    else:
        print(f"    {start}-{end} seconds: {count} ({percentage:.1f}%)")

# 5. Naive approach evaluation
print("\n\n5. NAIVE APPROACH EVALUATION")
print("-" * 80)
print("  Approach: Within 5 seconds before gesture, pick word with lowest openlexicon value")
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
        with open(transcript_path, 'r') as f:
            transcripts[video_id] = json.load(f)
    except Exception as e:
        print(f"    Warning: Could not load {video_id}: {e}")

print(f"  Transcripts loaded: {len(transcripts)} videos\n")

def preprocess_word(word):
    """Clean word for lexicon lookup"""
    # Remove punctuation and lowercase
    cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
    return cleaned if cleaned else None

def get_openlexicon_value(word):
    """Get openlexicon frequency value for a word (lower = rarer)"""
    cleaned = preprocess_word(word)
    if not cleaned:
        return float('inf')  # Skip punctuation-only words

    # Check if word is in lexicon
    if cleaned in lexicon_dict:
        return lexicon_dict[cleaned]
    else:
        # Not in lexicon = very rare, assign lowest value
        return -1.0

def find_words_in_window(transcript, pressed_time, window_seconds=5.0):
    """Find all words within window_seconds before pressed_time"""
    words_in_window = []

    if 'segments' not in transcript:
        return words_in_window

    for segment in transcript['segments']:
        if 'words' not in segment:
            continue

        for word_obj in segment['words']:
            word_start = word_obj['start']
            word_text = word_obj['word'].strip()

            # Check if word is within the time window
            if pressed_time - window_seconds <= word_start <= pressed_time:
                openlexicon_val = get_openlexicon_value(word_text)
                words_in_window.append({
                    'word': word_text,
                    'start': word_start,
                    'openlexicon': openlexicon_val
                })

    return words_in_window

# Evaluate naive approach
print("  Evaluating naive approach on labeled gestures...\n")

correct_matches = 0
total_evaluated = 0
no_words_found = 0
details = []

for item in labeled_data:
    # Skip tutorial and invalid entries
    if item['is_tutorial']:
        continue
    if item.get('target_word_timestamp') is None:
        continue
    if item['target_source'] != 'transcript':
        continue

    video_id = item['video_id']
    pressed_timestamp = item['pressed_timestamp']
    target_word = item['target_word']

    # Get transcript
    if video_id not in transcripts:
        continue

    transcript = transcripts[video_id]

    # Find words in 5-second window
    words_in_window = find_words_in_window(transcript, pressed_timestamp, window_seconds=5.0)

    if not words_in_window:
        no_words_found += 1
        continue

    # Find word with lowest openlexicon value
    best_word = min(words_in_window, key=lambda x: x['openlexicon'])

    # Clean both target and predicted for comparison
    target_cleaned = preprocess_word(target_word.split()[0]) if target_word else ""
    predicted_cleaned = preprocess_word(best_word['word'])

    # Check if match
    is_match = target_cleaned == predicted_cleaned
    if is_match:
        correct_matches += 1

    total_evaluated += 1

    # Store details for error analysis
    details.append({
        'video_id': video_id,
        'pressed_time': pressed_timestamp,
        'target': target_word,
        'predicted': best_word['word'],
        'predicted_openlexicon': best_word['openlexicon'],
        'is_match': is_match,
        'num_candidates': len(words_in_window)
    })

# Print results
print(f"  Results:")
print(f"    Total gestures evaluated: {total_evaluated}")
print(f"    Correct matches: {correct_matches}")
print(f"    Accuracy: {(correct_matches/total_evaluated)*100:.1f}%")
print(f"    No words found in window: {no_words_found}")

print(f"\n  Sample errors (showing first 10):")
errors = [d for d in details if not d['is_match']][:10]
for i, err in enumerate(errors, 1):
    print(f"    {i}. Target: '{err['target']}' | Predicted: '{err['predicted']}' (openlexicon={err['predicted_openlexicon']:.2f})")
    print(f"       Time: {err['pressed_time']:.1f}s | Candidates: {err['num_candidates']}")

print("\n" + "=" * 80)
