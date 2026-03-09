from pathlib import Path

_PARTS_DIR = Path(__file__).resolve().parent / "_parts" / "web_realtime"

_PART_PATH = _PARTS_DIR / "core_runtime.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
_PART_PATH = _PARTS_DIR / "dialogue_compression.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
_PART_PATH = _PARTS_DIR / "monitoring_controls.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
_PART_PATH = _PARTS_DIR / "socket_handlers.py"
exec(compile(_PART_PATH.read_text(), str(_PART_PATH), "exec"), globals(), globals())
