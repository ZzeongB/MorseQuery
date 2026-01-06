import json
from collections import defaultdict

# Load the results
with open('gpt_prediction_results.json', 'r') as f:
    data = json.load(f)

results = data['results']
summary = data['summary']

# Categorize results
successful_multi_word = []
successful_single_word = []
failed_predictions = []

for r in results:
    actual_word_count = len(r['actual'].replace(',', '').replace('.', '').split())

    if r['is_match']:
        if actual_word_count > 1:
            successful_multi_word.append(r)
        else:
            successful_single_word.append(r)
    else:
        failed_predictions.append(r)

# Print summary statistics
print("="*100)
print("GPT PREDICTION RESULTS ANALYSIS")
print("="*100)
print(f"\nOverall Accuracy: {summary['accuracy']:.1%} ({summary['matches']}/{summary['total']})")
print(f"\nSuccessful single-word predictions: {len(successful_single_word)}")
print(f"Successful multi-word predictions: {len(successful_multi_word)}")
print(f"Failed predictions: {len(failed_predictions)}")

# ============================================================================
# 1. SUCCESSFUL MULTI-WORD PREDICTIONS
# ============================================================================
print("\n" + "="*100)
print("1. SUCCESSFUL MULTI-WORD PREDICTIONS")
print("="*100)

if successful_multi_word:
    for i, r in enumerate(successful_multi_word, 1):
        print(f"\n--- Example {i} ---")
        print(f"Context: ...{r['context'][-80:]}")
        print(f"Actual:    '{r['actual']}'")
        print(f"Predicted: '{r['predicted']}'")
        print(f"Intent: {r['intent']}")
        if r.get('intent_other_text'):
            print(f"User's reason: {r['intent_other_text']}")
        print(f"Overlap: {r['overlap']:.1%}")
else:
    print("\nNo successful multi-word predictions found.")

# ============================================================================
# 2. FAILED PREDICTIONS
# ============================================================================
print("\n" + "="*100)
print("2. FAILED PREDICTIONS")
print("="*100)

if failed_predictions:
    # Sort by overlap (to see near-misses first)
    failed_predictions.sort(key=lambda x: x['overlap'], reverse=True)

    for i, r in enumerate(failed_predictions, 1):
        print(f"\n--- Failure {i} ---")
        print(f"Context: ...{r['context'][-80:]}")
        print(f"Actual:    '{r['actual']}'")
        print(f"Predicted: '{r['predicted']}'")
        print(f"Intent: {r['intent']}")
        if r.get('intent_other_text'):
            print(f"User's reason: {r['intent_other_text']}")
        print(f"Overlap: {r['overlap']:.1%}")

        # Analyze why it failed
        actual_words = set(r['actual'].lower().replace(',', '').replace('.', '').split())
        predicted_words = set(r['predicted'].lower().replace(',', '').replace('.', '').split())
        missing = actual_words - predicted_words
        extra = predicted_words - actual_words

        if missing:
            print(f"Missing words: {', '.join(missing)}")
        if extra:
            print(f"Extra words: {', '.join(extra)}")
else:
    print("\nNo failed predictions!")

# ============================================================================
# 3. ANALYSIS BY INTENT TYPE
# ============================================================================
print("\n" + "="*100)
print("3. ACCURACY BY INTENT TYPE")
print("="*100)

for intent, stats in sorted(summary['intent_breakdown'].items()):
    acc = stats['matches'] / stats['total'] if stats['total'] > 0 else 0
    print(f"{intent:20s}: {acc:6.1%} ({stats['matches']}/{stats['total']})")

# ============================================================================
# 4. WORD COUNT ANALYSIS
# ============================================================================
print("\n" + "="*100)
print("4. ACCURACY BY WORD COUNT")
print("="*100)

word_count_stats = defaultdict(lambda: {'total': 0, 'matches': 0})

for r in results:
    word_count = len(r['actual'].replace(',', '').replace('.', '').split())
    word_count_stats[word_count]['total'] += 1
    if r['is_match']:
        word_count_stats[word_count]['matches'] += 1

for word_count in sorted(word_count_stats.keys()):
    stats = word_count_stats[word_count]
    acc = stats['matches'] / stats['total'] if stats['total'] > 0 else 0
    word_label = "word" if word_count == 1 else "words"
    print(f"{word_count} {word_label:6s}: {acc:6.1%} ({stats['matches']}/{stats['total']})")

# ============================================================================
# 5. SAVE ORGANIZED RESULTS
# ============================================================================
organized_results = {
    'summary': {
        'overall_accuracy': summary['accuracy'],
        'total_tests': summary['total'],
        'total_matches': summary['matches'],
        'successful_single_word': len(successful_single_word),
        'successful_multi_word': len(successful_multi_word),
        'failed': len(failed_predictions)
    },
    'successful_multi_word_examples': successful_multi_word,
    'failed_examples': failed_predictions,
    'successful_single_word_examples': successful_single_word,
    'intent_breakdown': summary['intent_breakdown'],
    'word_count_breakdown': {
        str(k): v for k, v in word_count_stats.items()
    }
}

with open('gpt_results_organized.json', 'w') as f:
    json.dump(organized_results, f, indent=2, ensure_ascii=False)

print("\n" + "="*100)
print("Organized results saved to: gpt_results_organized.json")
print("="*100)

# ============================================================================
# 6. SAMPLE SUCCESSFUL SINGLE-WORD PREDICTIONS (for comparison)
# ============================================================================
print("\n" + "="*100)
print("5. SAMPLE SUCCESSFUL SINGLE-WORD PREDICTIONS (for comparison)")
print("="*100)

if successful_single_word:
    # Show first 5 examples
    for i, r in enumerate(successful_single_word[:5], 1):
        print(f"\n--- Example {i} ---")
        print(f"Context: ...{r['context'][-80:]}")
        print(f"Actual:    '{r['actual']}'")
        print(f"Predicted: '{r['predicted']}'")
        print(f"Intent: {r['intent']}")
        if r.get('intent_other_text'):
            print(f"User's reason: {r['intent_other_text']}")
        print(f"Overlap: {r['overlap']:.1%}")

    if len(successful_single_word) > 5:
        print(f"\n... and {len(successful_single_word) - 5} more successful single-word predictions")
else:
    print("\nNo successful single-word predictions found.")
