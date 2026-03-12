"""User settings management for persistent configuration."""

import json
from pathlib import Path
from typing import Any, Optional

import pyaudio

SETTINGS_FILE = Path(__file__).parent / "settings.json"

_DEFAULT_VOICE_ID = "67192626-0be9-4f45-b660-580d318c994d"

_DEFAULT_SETTINGS = {
    "session_id": "default",
    "voice_ids": {
        "keyword": _DEFAULT_VOICE_ID,
        "sum0": _DEFAULT_VOICE_ID,
        "sum1": _DEFAULT_VOICE_ID,
    },
    "preferred_mic_keywords": ["Jabra", "jabra"],
}

_cached_settings: Optional[dict] = None


def _load_settings() -> dict:
    """Load settings from JSON file."""
    global _cached_settings
    if _cached_settings is not None:
        return _cached_settings

    if not SETTINGS_FILE.exists():
        _cached_settings = _DEFAULT_SETTINGS.copy()
        _save_settings(_cached_settings)
        return _cached_settings

    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            # Merge with defaults to ensure all keys exist
            merged = _DEFAULT_SETTINGS.copy()
            merged.update(loaded)
            _cached_settings = merged
            return _cached_settings
    except Exception:
        _cached_settings = _DEFAULT_SETTINGS.copy()
        return _cached_settings


def _save_settings(settings: dict) -> None:
    """Save settings to JSON file."""
    global _cached_settings
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
        _cached_settings = settings
    except Exception as e:
        print(f"[WARN] Failed to save settings: {e}")


def get_setting(key: str, default: Any = None) -> Any:
    """Get a setting value."""
    settings = _load_settings()
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> None:
    """Set a setting value and persist to file."""
    settings = _load_settings()
    settings[key] = value
    _save_settings(settings)


def get_session_id() -> str:
    """Get the configured session ID."""
    return str(get_setting("session_id", "default"))


def set_session_id(session_id: str) -> None:
    """Set the session ID."""
    set_setting("session_id", str(session_id).strip() or "default")


def get_voice_id(key: str = "keyword") -> str:
    """Get the configured voice ID for a specific client.

    Args:
        key: One of "keyword", "sum0", "sum1" (default: "keyword")

    Returns:
        Voice ID string
    """
    voice_ids = get_setting("voice_ids", {})
    if isinstance(voice_ids, dict) and key in voice_ids:
        return str(voice_ids[key])
    return _DEFAULT_VOICE_ID


def set_voice_id(key: str, voice_id: str) -> None:
    """Set the voice ID for a specific client.

    Args:
        key: One of "keyword", "sum0", "sum1"
        voice_id: Cartesia voice ID
    """
    voice_ids = get_setting("voice_ids", {})
    if not isinstance(voice_ids, dict):
        voice_ids = _DEFAULT_SETTINGS["voice_ids"].copy()
    voice_ids[key] = str(voice_id).strip()
    set_setting("voice_ids", voice_ids)


def get_all_voice_ids() -> dict[str, str]:
    """Get all configured voice IDs."""
    voice_ids = get_setting("voice_ids", {})
    if isinstance(voice_ids, dict):
        return {k: str(v) for k, v in voice_ids.items()}
    return _DEFAULT_SETTINGS["voice_ids"].copy()


def get_preferred_mic_keywords() -> list[str]:
    """Get the list of preferred mic name keywords."""
    keywords = get_setting("preferred_mic_keywords", [])
    if isinstance(keywords, list):
        return [str(k) for k in keywords if k]
    return []


def find_preferred_mic_index() -> Optional[int]:
    """Find the index of the preferred microphone.

    Returns the index of the first input device whose name contains
    any of the preferred keywords, or None if not found.
    """
    keywords = get_preferred_mic_keywords()
    if not keywords:
        return None

    try:
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    name = str(info.get("name", ""))
                    for keyword in keywords:
                        if keyword in name:
                            return i
        finally:
            pa.terminate()
    except Exception:
        pass
    return None


def reload_settings() -> dict:
    """Force reload settings from file."""
    global _cached_settings
    _cached_settings = None
    return _load_settings()
