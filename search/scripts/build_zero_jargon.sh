#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../venv/bin/python}"
COUNT_FIELD="${COUNT_FIELD:-duplicate_count_minute_bucket}"

if [ "$#" -eq 0 ]; then
  stems=(6 7 8 9 10)
else
  stems=("$@")
fi

for stem in "${stems[@]}"; do
  "$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/count_semantic_word_duplicates.py" \
    --semantic-json "$ROOT_DIR/data/semantic_words/${stem}.json" \
    --transcript-json "$ROOT_DIR/data/transcripts/${stem}.json" \
    --output-json "$ROOT_DIR/data/semantic_words/${stem}.counts.json"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/filter_zero_count_semantic_words.py" \
    --input-json "$ROOT_DIR/data/semantic_words/${stem}.counts.json" \
    --output-json "$ROOT_DIR/data/semantic_words/${stem}.zero.json" \
    --count-field "$COUNT_FIELD"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/extract_jargon_words.py" \
    --semantic-json "$ROOT_DIR/data/semantic_words/${stem}.zero.json" \
    --transcript-json "$ROOT_DIR/data/transcripts/${stem}.json" \
    --output-json "$ROOT_DIR/data/semantic_words/${stem}.zero.jargon.json"

  "$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/attach_word_uniqueness.py" \
    --input-json "$ROOT_DIR/data/semantic_words/${stem}.zero.jargon.json" \
    --output-json "$ROOT_DIR/data/semantic_words/${stem}.zero.jargon.json"
done
