# Pilot P1 Analysis Summary

## Overview
- **Session ID**: 8a657952-8552-4976-a09f-120af43de82c
- **Date**: 2025-12-22
- **Total Gestures**: 81 (2 tutorial, 79 non-tutorial)

## 1. Gestures Per Video

| Video | Gestures | Video ID |
|-------|----------|----------|
| Video 0 | 16 | U6fI3brP8V4 |
| Video 1 | 26 | bAkuNXtgrLA |
| Video 2 | 25 | cEVAjm_ETtY |
| Video 3 | 12 | MxovSnvSO4E |
| Tutorial | 2 | U6fI3brP8V4 |

**Key Findings**:
- Most gestures occurred during Video 1 (26 gestures)
- Videos had fairly balanced gesture counts (12-26 gestures each)

## 2. Target Word Length

**Statistics**:
- Min: 1 word
- Max: 11 words
- Mean: 2.35 words
- Median: 2.00 words

**Distribution**:
- 1 word: 31 (39.2%)
- 2 words: 26 (32.9%)
- 3 words: 7 (8.9%)
- 4 words: 7 (8.9%)
- 5+ words: 8 (10.1%)

**Key Findings**:
- Most target words/phrases are short (1-2 words = 72.1%)
- Longer phrases (5+ words) are rare (10.1%)

## 3. Intent Types

**Overall Distribution**:
- lookup_meaning: 60 (74.1%)
- other: 28 (34.6%)
- background_info: 29 (35.8%)
- find_images: 14 (17.3%)

**Multi-Intent Gestures**: 37 (45.7%)

**Top Intent Combinations**:
1. lookup_meaning only: 24 (29.6%)
2. other only: 17 (21.0%)
3. background_info + lookup_meaning: 14 (17.3%)
4. background_info + find_images + lookup_meaning: 11 (13.6%)
5. lookup_meaning + other: 8 (9.9%)

**Key Findings**:
- Primary intent is "lookup_meaning" (74.1% of gestures)
- Nearly half of gestures (45.7%) have multiple intents
- "other" intent often indicates "Didn't hear accurately"
- Combined lookup + background info is common for technical terms

## 4. Time Gap Analysis

**Statistics**:
- Valid entries: 79/81
- Min gap: 0.28 seconds
- Max gap: 15.52 seconds
- Mean gap: 2.14 seconds
- Median gap: 1.67 seconds
- Std dev: 1.92 seconds

**Distribution**:
- 0-1 seconds: 6 (7.6%)
- 1-2 seconds: 46 (58.2%)
- 2-3 seconds: 20 (25.3%)
- 3-4 seconds: 3 (3.8%)
- 4-5 seconds: 2 (2.5%)
- 5+ seconds: 2 (2.6%)

**Key Findings**:
- Most gestures occur within 1-2 seconds after hearing the target word (58.2%)
- 91.1% of gestures occur within 3 seconds
- Very few outliers (only 2 gestures beyond 5 seconds)
- Median reaction time is 1.67 seconds

## 5. Naive Approach Evaluation

**Approach**: Within 5 seconds before gesture, select the word with the lowest OpenLexicon frequency value

**Results**:
- Total gestures evaluated: 77
- Correct matches: 21
- **Accuracy: 27.3%**
- No words found in window: 0

**Analysis of Failures**:

The naive approach fails primarily due to:

1. **Missing Lexicon Entries**: Words not in OpenLexicon get assigned -1.0 (treated as "very rare"), but many of these are actually common contractions:
   - "I'm", "here's", "don't", "there's", "m,"
   - These get incorrectly prioritized over actual rare words

2. **Multi-word Targets**: The approach only picks single words, but 60.8% of targets are multi-word phrases

3. **Context Ignorance**: The approach doesn't consider:
   - Part of speech (verbs vs. nouns)
   - Semantic relevance
   - User's likely intent

**Sample Errors**:
- Target: "Newton," → Predicted: "I'm" (openlexicon=-1.00)
- Target: "obsessed," → Predicted: "biography" (openlexicon=1.88)
- Target: "nails" → Predicted: "there's" (openlexicon=-1.00)

**Error Breakdown**:
- Missing from Lexicon: 36 (46.8%) - Biggest issue
- Multi-word Targets: 13 (16.9%)
- Wrong Word Picked: 7 (9.1%)
- Correct Matches: 21 (27.3%)

**Most Common Wrong Picks** (words not in lexicon):
- "phalamus": 4 times (likely misspelled "thalamus")
- "that's": 4 times
- "I'm": 3 times
- "here's": 3 times
- "you're": 3 times
- Contractions and misspelled technical terms dominate errors

**Why It Succeeds** (27.3% accuracy):
The approach works well for genuinely rare technical terms not in the lexicon:
- "Rydberg", "subcortical", "hippocampus", "LgN"
- These are unique in context and correctly prioritized

**Recommendations for Improvement**:
1. **Add contraction handling**: Expand contractions before lookup ("I'm" → "I am")
2. **Fix technical term spelling**: Use fuzzy matching or alternative lexicons for scientific terms
3. **Multi-word phrase extraction**: Consider n-grams (2-3 words)
4. **Add stopword filtering**: Filter common contractions even if not in lexicon
5. **Weight by temporal distance**: Closer words more likely (exponential decay)
6. **Part-of-speech filtering**: Prefer nouns/adjectives over verbs/pronouns
7. **Semantic clustering**: Group words by topic and prefer domain-relevant terms

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total Gestures | 81 |
| Avg Time Gap | 2.14s |
| Avg Target Length | 2.35 words |
| Primary Intent | lookup_meaning (74.1%) |
| Naive Approach Accuracy | 27.3% |
| Optimal Window | 1-3 seconds (83.5% of gestures) |
