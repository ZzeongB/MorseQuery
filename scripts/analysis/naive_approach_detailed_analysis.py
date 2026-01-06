import json
import pandas as pd
import os
import re
from collections import Counter

# Load data
with open('logs/study_data/pilot-p1.json', 'r') as f:
    data = json.load(f)
labeled_data = data['labeled_data']

# Load OpenLexicon
lexicon_path = "data/lexicon/OpenLexicon.xlsx"
lexicon_df = pd.read_excel(lexicon_path)
lexicon_dict = {}
for _, row in lexicon_df.iterrows():
    word = str(row["ortho"]).lower()
    freq_value = row["English_Lexicon_Project__LgSUBTLWF"]
    lexicon_dict[word] = freq_value

# Load transcripts
transcripts = {}
transcript_dir = "data/transcripts"
for video_id in ["U6fI3brP8V4", "bAkuNXtgrLA", "cEVAjm_ETtY", "MxovSnvSO4E"]:
    transcript_path = os.path.join(transcript_dir, f"{video_id}.json")
    with open(transcript_path, 'r') as f:
        transcripts[video_id] = json.load(f)

def preprocess_word(word):
    """Clean word for lexicon lookup"""
    cleaned = re.sub(r'[^a-zA-Z]', '', word).lower()
    return cleaned if cleaned else None

def get_openlexicon_value(word):
    """Get openlexicon frequency value for a word"""
    cleaned = preprocess_word(word)
    if not cleaned:
        return float('inf')
    if cleaned in lexicon_dict:
        return lexicon_dict[cleaned]
    else:
        return -1.0  # Not in lexicon

def find_words_in_window(transcript, pressed_time, window_seconds=5.0):
    """Find all words within window"""
    words_in_window = []
    if 'segments' not in transcript:
        return words_in_window

    for segment in transcript['segments']:
        if 'words' not in segment:
            continue
        for word_obj in segment['words']:
            word_start = word_obj['start']
            word_text = word_obj['word'].strip()
            if pressed_time - window_seconds <= word_start <= pressed_time:
                openlexicon_val = get_openlexicon_value(word_text)
                words_in_window.append({
                    'word': word_text,
                    'start': word_start,
                    'openlexicon': openlexicon_val,
                    'time_to_press': pressed_time - word_start
                })
    return words_in_window

# Detailed error analysis
print("=" * 80)
print("NAIVE APPROACH: DETAILED ERROR ANALYSIS")
print("=" * 80)

errors_by_category = {
    'missing_from_lexicon': [],
    'multi_word_target': [],
    'wrong_word_picked': [],
    'correct_match': []
}

common_wrong_picks = Counter()

for item in labeled_data:
    if item['is_tutorial'] or item.get('target_word_timestamp') is None:
        continue
    if item['target_source'] != 'transcript':
        continue

    video_id = item['video_id']
    pressed_timestamp = item['pressed_timestamp']
    target_word = item['target_word']

    if video_id not in transcripts:
        continue

    transcript = transcripts[video_id]
    words_in_window = find_words_in_window(transcript, pressed_timestamp, window_seconds=5.0)

    if not words_in_window:
        continue

    # Find word with lowest openlexicon
    best_word = min(words_in_window, key=lambda x: x['openlexicon'])

    # Get target first word
    target_first_word = target_word.split()[0] if target_word else ""
    target_cleaned = preprocess_word(target_first_word)
    predicted_cleaned = preprocess_word(best_word['word'])

    is_match = target_cleaned == predicted_cleaned

    # Categorize error
    if is_match:
        errors_by_category['correct_match'].append({
            'target': target_word,
            'predicted': best_word['word'],
            'openlexicon': best_word['openlexicon']
        })
    else:
        # Check if predicted word is missing from lexicon
        if best_word['openlexicon'] == -1.0:
            errors_by_category['missing_from_lexicon'].append({
                'target': target_word,
                'predicted': best_word['word'],
                'time_gap': best_word['time_to_press']
            })
            common_wrong_picks[best_word['word']] += 1
        # Check if target has multiple words
        elif len(target_word.split()) > 1:
            errors_by_category['multi_word_target'].append({
                'target': target_word,
                'predicted': best_word['word'],
                'openlexicon': best_word['openlexicon']
            })
        else:
            errors_by_category['wrong_word_picked'].append({
                'target': target_word,
                'predicted': best_word['word'],
                'openlexicon': best_word['openlexicon'],
                'target_openlexicon': get_openlexicon_value(target_word)
            })

# Print analysis
print("\n1. ERROR BREAKDOWN BY CATEGORY")
print("-" * 80)
total = sum(len(v) for v in errors_by_category.values())
for category, items in errors_by_category.items():
    count = len(items)
    percentage = (count / total) * 100 if total > 0 else 0
    print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")

print("\n\n2. MISSING FROM LEXICON ERRORS")
print("-" * 80)
print("  These are words not in OpenLexicon that got incorrectly prioritized:")
print(f"\n  Most common wrong picks:")
for word, count in common_wrong_picks.most_common(10):
    print(f"    '{word}': {count} times")

print(f"\n  Sample errors:")
for i, err in enumerate(errors_by_category['missing_from_lexicon'][:5], 1):
    print(f"    {i}. Target: '{err['target']}' → Predicted: '{err['predicted']}' (gap: {err['time_gap']:.2f}s)")

print("\n\n3. MULTI-WORD TARGET ERRORS")
print("-" * 80)
print(f"  Total: {len(errors_by_category['multi_word_target'])}")
print("  The naive approach only picks single words, but many targets are phrases.")
print(f"\n  Sample errors:")
for i, err in enumerate(errors_by_category['multi_word_target'][:5], 1):
    print(f"    {i}. Target: '{err['target']}'")
    print(f"       Predicted: '{err['predicted']}' (openlexicon={err['openlexicon']:.2f})")

print("\n\n4. WRONG WORD PICKED (Both in Lexicon)")
print("-" * 80)
print(f"  Total: {len(errors_by_category['wrong_word_picked'])}")
print("  Cases where both target and predicted are in lexicon, but wrong word chosen.")
print(f"\n  Sample errors:")
for i, err in enumerate(errors_by_category['wrong_word_picked'][:5], 1):
    print(f"    {i}. Target: '{err['target']}' (openlexicon={err['target_openlexicon']:.2f})")
    print(f"       Predicted: '{err['predicted']}' (openlexicon={err['openlexicon']:.2f})")

print("\n\n5. CORRECT MATCHES")
print("-" * 80)
print(f"  Total: {len(errors_by_category['correct_match'])}")
print(f"\n  Sample correct predictions:")
for i, item in enumerate(errors_by_category['correct_match'][:10], 1):
    print(f"    {i}. '{item['target']}' → '{item['predicted']}' (openlexicon={item['openlexicon']:.2f})")

print("\n" + "=" * 80)
