from pathlib import Path

_PARTS_DIR = Path(__file__).resolve().parent / "_parts" / "explore_context"

_PART_PATH = _PARTS_DIR / "lexicon_processing.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
_PART_PATH = _PARTS_DIR / "similarity_experiments.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
