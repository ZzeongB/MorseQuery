#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../venv/bin/python}"

if [ "$#" -eq 0 ]; then
  stems=(6 7 8 9 10)
else
  stems=("$@")
fi

for stem in "${stems[@]}"; do
  "$PYTHON_BIN" "$ROOT_DIR/scripts/filter_semantic_words_constraints.py" \
    --semantic-json "$ROOT_DIR/data/semantic_words/${stem}.json" \
    --transcript-json "$ROOT_DIR/data/transcripts/${stem}.json" \
    --min-word-gap 3 \
    --min-time-gap-seconds 2.0
done
