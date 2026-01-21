#!/usr/bin/env python3
"""Test script to verify contraction filtering works"""

# Contractions dictionary (same as in main script)
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
}

def test_contraction_detection(word):
    """Test if contraction detection works"""
    cleaned_word = "".join(c for c in word if c.isalnum()).lower()
    is_contraction = cleaned_word in CONTRACTIONS
    return cleaned_word, is_contraction

# Test cases
test_words = ["you're", "don't", "can't", "that's", "they're", "hello", "world", "I'm", "it's"]

print("Testing contraction detection:")
print("=" * 60)
for word in test_words:
    cleaned, is_contraction = test_contraction_detection(word)
    status = "✓ CONTRACTION" if is_contraction else "✗ not contraction"
    print(f"{word:15} -> {cleaned:15} {status}")

print("\n" + "=" * 60)
