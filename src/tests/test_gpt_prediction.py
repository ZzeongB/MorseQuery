import json
import os

from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Read pilot-p2 data
with open("logs/study_data/pilot-p2.json", "r") as f:
    pilot_data = json.load(f)

# Load all transcripts
transcripts = {}
video_ids = ["U6fI3brP8V4", "bAkuNXtgrLA", "cEVAjm_ETtY", "MxovSnvSO4E"]
for vid in video_ids:
    with open(f"data/transcripts/{vid}.json", "r") as f:
        transcripts[vid] = json.load(f)


def get_context_words(video_id, target_timestamp, window=3.0):
    """Get words from window seconds before target timestamp"""
    transcript = transcripts[video_id]
    words = transcript.get("words", [])

    start_time = target_timestamp - window
    context_words = []

    for word_info in words:
        word_start = word_info["start"]
        if start_time <= word_start <= target_timestamp + 1:
            context_words.append(word_info["word"])

    return " ".join(context_words)


# Filter entries with transcript source
transcript_entries = [
    e for e in pilot_data["labeled_data"] if e["target_source"] == "transcript"
]

# for idx, e in enumerate(transcript_entries):
#     print(f"{idx}: {e['target_word']}")

# Select few-shot examples (mix of single and multi-word)
few_shot_indices = [
    1,  # "I get paid for this,"
    2,  # "spark gap," - 2 words
    6,  # "metaphorically" - 1 word
    9,  # "cortex." - 1 word
    14,  # "Lonnie Sujonsen"
    25,  # "chronic fatique",
    28,  # "pancreas"
]

# Create few-shot prompt
few_shot_examples = []
print("Few shot examples")
for idx in few_shot_indices:
    entry = transcript_entries[idx]
    context = get_context_words(entry["video_id"], entry["target_word_timestamp"])
    target = entry["target_word"]
    intent = ", ".join(entry["intent_types"])
    intent_other_text = entry.get("intent_other_text", "")

    few_shot_examples.append(
        {
            "context": context,
            "target": target,
            "intent": intent,
            "intent_other_text": intent_other_text,
        }
    )

    print("\t", target)


# Create the prompt template
def create_prompt(context_text, user_intent_detail=None):
    prompt = """You are analyzing video lecture transcripts. Users watch lectures and select specific words or phrases they want to look up or get more information about.

Given the transcript context, predict which word(s) the user selected. The selected words should be:
- Technical terms or unfamiliar vocabulary
- Concepts that need clarification
- Names or specific references
- Words that might need visual aids
- Jokes or phrases that went by too quickly (when intent is 'other')

Few-shot examples:

"""

    for i, ex in enumerate(few_shot_examples, 1):
        prompt += f"Example {i}:\n"
        prompt += f"Context: {ex['context']}\n"
        prompt += f"User selected: {ex['target']}\n"
        prompt += f"Intent: {ex['intent']}\n"
        if ex.get("intent_other_text"):
            prompt += f"User's reason: {ex['intent_other_text']}\n"
        prompt += "\n"

    prompt += f"""Now predict for this new context:

Context: {context_text}"""

    if user_intent_detail:
        prompt += f"\nUser's reason: {user_intent_detail}"

    prompt += "\nUser selected: "

    print("Prompt: ", prompt)

    return prompt


# Test on remaining entries
test_indices = [i for i in range(len(transcript_entries)) if i not in few_shot_indices]

results = []
print("Testing GPT predictions...\n")
print("=" * 100)

for idx in test_indices:
    entry = transcript_entries[idx]
    context = get_context_words(entry["video_id"], entry["target_word_timestamp"])
    actual_target = entry["target_word"]
    intent = ", ".join(entry["intent_types"])
    intent_other_text = entry.get("intent_other_text", "")

    # Get GPT prediction
    # Pass intent_other_text only if intent contains "other" and text is not empty
    user_intent_detail = (
        intent_other_text if ("other" in intent and intent_other_text) else None
    )
    prompt = create_prompt(context, user_intent_detail)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=50,
        )

        predicted = response.choices[0].message.content.strip()

        # Simple matching: check if key words overlap
        actual_words = set(
            actual_target.lower().replace(",", "").replace(".", "").split()
        )
        predicted_words = set(
            predicted.lower().replace(",", "").replace(".", "").split()
        )

        # Calculate overlap
        if len(actual_words) == 0:
            overlap = 0
        else:
            overlap = len(actual_words.intersection(predicted_words)) / len(
                actual_words
            )

        # Consider it a match if overlap >= 50% or if main word is captured
        is_match = overlap >= 0.5

        results.append(
            {
                "index": idx,
                "context": context,
                "actual": actual_target,
                "predicted": predicted,
                "intent": intent,
                "intent_other_text": intent_other_text,
                "overlap": overlap,
                "is_match": is_match,
            }
        )

        print(f"\nTest #{len(results)} (Entry #{idx})")
        print(f"Context: ...{context[-100:]}")
        print(f"Actual: {actual_target}")
        print(f"Predicted: {predicted}")
        print(f"Intent: {intent}")
        if intent_other_text:
            print(f"User's reason: {intent_other_text}")
        print(f"Match: {'✓' if is_match else '✗'} (overlap: {overlap:.1%})")
        print("-" * 100)

    except Exception as e:
        print(f"Error processing entry {idx}: {e}")
        continue

# Calculate accuracy
total = len(results)
matches = sum(1 for r in results if r["is_match"])
accuracy = matches / total if total > 0 else 0

print("\n" + "=" * 100)
print("\nRESULTS SUMMARY:")
print(f"Total tests: {total}")
print(f"Matches: {matches}")
print(f"Accuracy: {accuracy:.1%}")
print(f"\nFew-shot examples used: {len(few_shot_examples)}")
print("Example types: single-word and multi-word phrases")

# Breakdown by intent type
from collections import defaultdict

intent_stats = defaultdict(lambda: {"total": 0, "matches": 0})

for r in results:
    intents = r["intent"].split(", ")
    for intent in intents:
        intent_stats[intent]["total"] += 1
        if r["is_match"]:
            intent_stats[intent]["matches"] += 1

print("\nAccuracy by intent type:")
for intent, stats in sorted(intent_stats.items()):
    intent_acc = stats["matches"] / stats["total"] if stats["total"] > 0 else 0
    print(f"  {intent}: {intent_acc:.1%} ({stats['matches']}/{stats['total']})")

# Save detailed results
with open("gpt_prediction_results.json", "w") as f:
    json.dump(
        {
            "few_shot_examples": few_shot_examples,
            "results": results,
            "summary": {
                "total": total,
                "matches": matches,
                "accuracy": accuracy,
                "intent_breakdown": dict(intent_stats),
            },
        },
        f,
        indent=2,
    )

print("\nDetailed results saved to: gpt_prediction_results.json")
