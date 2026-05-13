#!/bin/zsh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/../venv/bin/python}"

if [ "$#" -eq 0 ]; then
  stems=(6 7 8 9 10)
else
  stems=("$@")
fi

"$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/generate_target_words.py" \
  --keywords-dir "$ROOT_DIR/data/semantic_words" \
  --keywords-suffix ".zero.jargon.json" \
  --audio-windows "$ROOT_DIR/data/study/audio_windows.json" \
  --output-dir "$ROOT_DIR/data/study/target_words" \
  --selection-mode minute_slots \
  --stems "${stems[@]}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/study_data/build_semantic_words2.py" \
  --semantic-dir "$ROOT_DIR/data/semantic_words" \
  --target-dir "$ROOT_DIR/data/study/target_words" \
  --output-dir "$ROOT_DIR/data/semantic_words2" \
  --stems "${stems[@]}"

"$PYTHON_BIN" "$ROOT_DIR/scripts/analysis/plot_jargon_word_distribution.py"
