import json

# Load results
with open('gpt_prediction_results.json', 'r') as f:
    data = json.load(f)

results = data['results']

# Filter failures
failures = [r for r in results if not r['is_match']]

print("="*100)
print(f"TOTAL FAILURES: {len(failures)} out of {len(results)} ({len(failures)/len(results)*100:.1f}%)")
print("="*100)

for i, failure in enumerate(failures, 1):
    print(f"\n{'='*100}")
    print(f"FAILURE #{i}")
    print(f"{'='*100}")
    print(f"Context:         ...{failure['context'][-80:]}")
    print(f"\nActual:          '{failure['actual']}'")
    print(f"Predicted:       '{failure['predicted']}'")
    print(f"\nIntent:          {failure['intent']}")
    if failure.get('intent_other_text'):
        print(f"User's reason:   {failure['intent_other_text']}")
    print(f"\nOverlap:         {failure['overlap']:.1%}")

    # Analyze the mismatch
    actual_words = set(failure['actual'].lower().replace(',', '').replace('.', '').split())
    predicted_words = set(failure['predicted'].lower().replace(',', '').replace('.', '').split())

    missing = actual_words - predicted_words
    extra = predicted_words - actual_words
    common = actual_words & predicted_words

    if common:
        print(f"Common words:    {', '.join(common)}")
    if missing:
        print(f"Missing words:   {', '.join(missing)}")
    if extra:
        print(f"Extra words:     {', '.join(sorted(extra))}")

print(f"\n{'='*100}")
print("SUMMARY BY FAILURE TYPE")
print(f"{'='*100}")

# Group by intent
from collections import defaultdict
intent_failures = defaultdict(list)

for f in failures:
    intent_failures[f['intent']].append(f)

for intent, fails in sorted(intent_failures.items()):
    print(f"\n{intent}: {len(fails)} failures")
    for f in fails:
        print(f"  - '{f['actual']}' → '{f['predicted'][:30]}...'")
