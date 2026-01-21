import json

# Read pilot-p2 data
with open('logs/study_data/pilot-p2.json', 'r') as f:
    pilot_data = json.load(f)

# Load all transcripts
transcripts = {}
video_ids = ['U6fI3brP8V4', 'bAkuNXtgrLA', 'cEVAjm_ETtY', 'MxovSnvSO4E']
for vid in video_ids:
    with open(f'data/transcripts/{vid}.json', 'r') as f:
        transcripts[vid] = json.load(f)

# Analyze each labeled_data entry
for idx, entry in enumerate(pilot_data['labeled_data']):
    if entry['target_source'] == 'transcript':
        video_id = entry['video_id']
        target_timestamp = entry['target_word_timestamp']
        target_word = entry['target_word']

        # Get transcript for this video
        transcript = transcripts[video_id]
        words = transcript.get('words', [])

        # Find words from ~3 seconds before target timestamp
        start_time = target_timestamp - 3.0
        context_words = []

        for word_info in words:
            word_start = word_info['start']
            if start_time <= word_start <= target_timestamp + 1:
                context_words.append({
                    'word': word_info['word'],
                    'start': word_start
                })

        print(f"\n{'='*80}")
        print(f"Entry #{idx+1} (video_index: {entry.get('video_index', 'N/A')}, is_tutorial: {entry['is_tutorial']})")
        print(f"Video ID: {video_id}")
        print(f"Target timestamp: {target_timestamp:.2f}s")
        print(f"Target word(s): {target_word}")
        print(f"Intent types: {', '.join(entry['intent_types'])}")
        if entry['intent_other_text']:
            print(f"Intent detail: {entry['intent_other_text']}")
        print(f"\nContext (~3 seconds before):")

        # Display the words
        if context_words:
            word_text = ' '.join([w['word'] for w in context_words])
            print(f"  [{context_words[0]['start']:.2f}s - {context_words[-1]['start']:.2f}s]: {word_text}")
        else:
            print("  No context words found in this range")

        print(f"{'='*80}")

print(f"\n\nTotal labeled entries with transcript source: {sum(1 for e in pilot_data['labeled_data'] if e['target_source'] == 'transcript')}")
