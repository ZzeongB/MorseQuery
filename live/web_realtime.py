"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import base64
import io
import json
import os
import re
import struct
import subprocess
import sys
import threading
import time
import wave
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import openai
import pyaudio
from pydub import AudioSegment, effects, silence
from pydub.utils import make_chunks
from clients import (
    ConversationReconstructorClient,
    DialogueStore,
    RealtimeClient,
    SummaryClient,
    TranscriptReconstructorClient,
)
from clients.context_judge_client import ContextJudgeClient
from clients.tts_client import TTSClient

# Mic level monitoring
_mic_monitor_streams: dict[int, tuple[pyaudio.Stream, pyaudio.PyAudio]] = {}
_mic_monitor_lock = threading.Lock()
_mic_monitor_running = False
_mic_monitor_thread: Optional[threading.Thread] = None

# Noise gate calibration monitoring
_noise_gate_monitor_running = False
_noise_gate_monitor_thread: Optional[threading.Thread] = None

# Global PyAudio lock to prevent segfaults from concurrent access
_pyaudio_lock = threading.Lock()
from config import LOG_DIR, TEMPLATES_DIR
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO
from handlers.grounding import handle_search_grounding
from logger import get_logger, get_session_dir, get_session_subdir, log_print

STATIC_DIR = Path(__file__).parent / "static"
app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
sio = SocketIO(app, cors_allowed_origins="*")

# AirPods ANC/Transparency auto-switch during TTS playback.
_airpods_lock = threading.Lock()
_airpods_active_tts_count = 0
_airpods_last_mode: Optional[str] = None
_airpods_mode_switch_enabled = True
_airpods_keyword_hold = False
_airpods_script_path = Path(__file__).resolve().parents[1] / "airpod.py"


def _run_airpods_mode(mode: str, reason: str, wait: bool = False) -> None:
    """Best-effort mode switch using airpod.py."""
    if mode not in {"anc", "transparency"}:
        return
    if sys.platform != "darwin":
        return
    if not _airpods_script_path.exists():
        log_print(
            "WARN",
            "airpod.py not found; skipping AirPods mode switch",
            path=str(_airpods_script_path),
            mode=mode,
        )
        return

    def _apply() -> None:
        try:
            cmd = [sys.executable, str(_airpods_script_path), mode]
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                log_print(
                    "WARN",
                    "AirPods mode command failed",
                    mode=mode,
                    reason=reason,
                    returncode=result.returncode,
                    stderr=(result.stderr or "").strip()[:200],
                )
                return
            log_print("INFO", "AirPods mode switched", mode=mode, reason=reason)
        except Exception as e:
            log_print(
                "WARN",
                f"AirPods mode switch failed: {e}",
                mode=mode,
                reason=reason,
            )

    if wait:
        _apply()
    else:
        threading.Thread(target=_apply, daemon=True).start()


def _set_airpods_mode(mode: str, reason: str, wait: bool = False) -> None:
    with _airpods_lock:
        global _airpods_last_mode, _airpods_mode_switch_enabled
        if not _airpods_mode_switch_enabled:
            return
        if _airpods_last_mode == mode:
            return
        _airpods_last_mode = mode
    _run_airpods_mode(mode, reason, wait=wait)


def _on_tts_started(reason: str) -> None:
    with _airpods_lock:
        global _airpods_active_tts_count
        _airpods_active_tts_count += 1
        should_switch = _airpods_active_tts_count == 1
    if should_switch:
        # Block TTS start briefly until ANC command is applied.
        _set_airpods_mode("anc", reason, wait=True)


def _on_tts_finished(reason: str) -> None:
    with _airpods_lock:
        global _airpods_active_tts_count, _airpods_keyword_hold
        if _airpods_active_tts_count > 0:
            _airpods_active_tts_count -= 1
        should_switch = _airpods_active_tts_count == 0 and not _airpods_keyword_hold
    if should_switch:
        _set_airpods_mode("transparency", reason)


def _set_keyword_anc_hold(active: bool, reason: str) -> None:
    with _airpods_lock:
        global _airpods_keyword_hold
        _airpods_keyword_hold = active
    if active:
        # Keep ANC locked from keyword phase until summary end/dismiss.
        _set_airpods_mode("anc", reason, wait=True)
    else:
        # If no playback is active, release immediately.
        with _airpods_lock:
            should_release = _airpods_active_tts_count == 0
        if should_release:
            _set_airpods_mode("transparency", reason)


def _reset_tts_airpods_state(reason: str) -> None:
    with _airpods_lock:
        global _airpods_active_tts_count, _airpods_keyword_hold
        _airpods_active_tts_count = 0
        _airpods_keyword_hold = False
    _set_airpods_mode("transparency", reason)


_original_sio_emit = sio.emit


def _emit_with_airpods(event, *args, **kwargs):
    if event == "tts_playing":
        _on_tts_started("socketio_tts_playing")
    elif event == "tts_done":
        # Summary/reconstruction flow ended.
        _set_keyword_anc_hold(False, "socketio_tts_done")
        _on_tts_finished("socketio_tts_done")
        # Enforce transparency on summary completion even if counters drift.
        _set_airpods_mode("transparency", "socketio_tts_done_force", wait=True)
    return _original_sio_emit(event, *args, **kwargs)


sio.emit = _emit_with_airpods  # type: ignore[assignment]

# Active clients (per session in production, global for simplicity here)
_clients_lock = threading.Lock()
client: Optional[RealtimeClient] = None
summary_clients: list[SummaryClient] = []  # One per summary mic
context_judge: Optional[ContextJudgeClient] = None  # Context-aware TTS judge
conversation_reconstructor: Optional[ConversationReconstructorClient] = None
transcript_reconstructor: Optional[TranscriptReconstructorClient] = None
keyword_tts_client: Optional[TTSClient] = None
_keyword_tts_request_token = 0
_active_runtime_sid: Optional[str] = None

# Dialogue stores for VAD transcripts (one per summary speaker)
_dialogue_stores: dict[str, DialogueStore] = {}  # key: "A" or "B"
_dialogue_segment_start_time: float = 0.0
_DIALOGUE_BUFFER_RETENTION_SEC = 15 * 60  # always-on rolling buffer
_MISS_PADDING_BEFORE_SEC = 1.0
_FAST_CATCHUP_DEFAULT_THRESHOLD_SEC = 10.0
_FAST_CATCHUP_DEFAULT_SPEED = 1.6
_FAST_CATCHUP_DEFAULT_GAP_SEC = 0.5
_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB = -45.0
_FAST_CATCHUP_TARGET_RMS_DBFS = -18.0
_FAST_CATCHUP_TARGET_PEAK_DBFS = -1.5
_FAST_CATCHUP_CHAIN_MAX_STEPS = 5
_FAST_CATCHUP_CHAIN_MIN_LAG_SEC = 2.0
_FAST_CATCHUP_EXTRA_GAIN_DB = 9.5
_FAST_CATCHUP_DENOISE_ENABLED = True
_FAST_CATCHUP_DENOISE_CHUNK_MS = 20
_FAST_CATCHUP_DENOISE_NOISE_ATTEN_DB = 12.0
_fast_catchup_threshold_sec_runtime = _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
_SEGMENT_TAIL_GRACE_SEC = 1.4
_SEGMENT_POST_END_WAIT_SEC = 0.45
_SEGMENT_DIALOGUE_QUIET_WINDOW_SEC = 1.0
_SEGMENT_DIALOGUE_MAX_WAIT_SEC = 4.0
_SEGMENT_DIALOGUE_POLL_SEC = 0.12
_POST_TTS_FOLLOWUP_WAIT_SEC = 0.3
_BEFORE_CONTEXT_RECENT_WINDOW_SEC = 60.0

# Persist 3-path compression results (time/input/output) for every segment/session.
_three_path_results_lock = threading.Lock()
_all_sessions_three_path_file = (
    LOG_DIR / "sessions" / "all_sessions_three_path_results.json"
)

# Aggregate multiple summary-agent outputs into one request batch per segment
_judge_batch_lock = threading.Lock()
_judge_batch: dict[int, dict] = {}
_judge_completed_segments: set[int] = set()
_JUDGE_BATCH_TIMEOUT_SEC = 0.7

# Conversation context tracking (from keyword RealtimeClient VAD transcripts)
_segment_ctx_lock = threading.Lock()
_segment_seq = 0
_segment_windows: dict[int, dict] = {}
_recent_vad_transcripts = deque(maxlen=80)  # (ts, text)
_recent_vad_boundaries = deque(maxlen=200)  # (ts, event_type)
_before_context_lock = threading.Lock()
_before_context_summary = ""
_post_tts_followup_active = False
_post_tts_followup_inflight = False
_post_tts_followup_cursor_ts = 0.0
_post_tts_followup_live_window_open = False
_segment_compression_inflight: set[int] = set()
_transcript_compression_mode = "realtime"


def _reset_segment_tracking() -> None:
    global _segment_seq, _before_context_summary
    global \
        _post_tts_followup_active, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts
    global _post_tts_followup_live_window_open, _segment_compression_inflight
    with _segment_ctx_lock:
        _segment_seq = 0
        _segment_windows.clear()
        _recent_vad_transcripts.clear()
        _recent_vad_boundaries.clear()
        _post_tts_followup_active = False
        _post_tts_followup_inflight = False
        _post_tts_followup_cursor_ts = 0.0
        _post_tts_followup_live_window_open = False
        _segment_compression_inflight.clear()
    with _before_context_lock:
        _before_context_summary = ""


def _on_vad_transcript(transcript: str) -> None:
    now = time.time()
    text = (transcript or "").strip()
    if not text:
        return

    with _segment_ctx_lock:
        _recent_vad_transcripts.append((now, text))
        # Bind next_sentence to the earliest ended segment that does not have one yet.
        pending = sorted(
            (
                (seg_id, info)
                for seg_id, info in _segment_windows.items()
                if info.get("end_ts") is not None and not info.get("next_sentence")
            ),
            key=lambda x: x[0],
        )
        for seg_id, info in pending:
            if now >= float(info.get("end_ts", 0.0)):
                info["next_sentence"] = text
                log_print(
                    "INFO",
                    "Captured next_sentence for segment",
                    segment_id=seg_id,
                    chars=len(text),
                )
                break


def _on_vad_boundary(event_type: str, payload: dict) -> None:
    """Track fast speech boundaries from server VAD events."""
    ts = float((payload or {}).get("received_at_ts") or time.time())
    et = str(event_type or "").strip().lower()
    if et not in {"speech_started", "speech_stopped"}:
        return
    with _segment_ctx_lock:
        _recent_vad_boundaries.append((ts, et))


def _snap_end_to_vad_stop(
    target_end_ts: float,
    wait_sec: float = 0.45,
    max_lookback_sec: float = 20.0,
) -> float:
    """Snap an end timestamp to the nearest recent speech_stopped boundary."""
    target = float(target_end_ts or 0.0)
    if target <= 0:
        return target
    deadline = time.time() + max(0.0, float(wait_sec))
    while True:
        with _segment_ctx_lock:
            candidates = [
                ts
                for ts, et in _recent_vad_boundaries
                if et == "speech_stopped"
                and ts >= target
                and ts <= target + max_lookback_sec
            ]
        if candidates:
            return min(candidates)
        if time.time() >= deadline:
            break
        time.sleep(0.03)
    return target


def _get_current_keywords_with_desc(limit: int = 3) -> list[dict]:
    """Get latest extracted keywords from RealtimeClient."""
    with _clients_lock:
        current_client = client
    if not current_client:
        return []
    try:
        return current_client.get_recent_keywords(limit=limit)
    except Exception:
        return []


def _get_dialogue_before_start_ts(start_ts: float) -> str:
    """Get full speaker-labeled dialogue before a given timestamp."""
    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_entries_between(0.0, start_ts))

    all_entries.sort(key=lambda e: e.timestamp)
    if not all_entries:
        return ""
    lines = [f"{e.speaker_id}: {e.text}" for e in all_entries]
    return "\n".join(lines)


def _get_recent_dialogue_before_start_ts(
    start_ts: float,
    window_sec: float = _BEFORE_CONTEXT_RECENT_WINDOW_SEC,
) -> str:
    """Get recent speaker-labeled dialogue in [start_ts-window_sec, start_ts]."""
    lo = max(0.0, float(start_ts) - max(1.0, float(window_sec)))
    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_entries_between(lo, start_ts))
    all_entries.sort(key=lambda e: e.timestamp)
    if not all_entries:
        return ""
    lines = [f"{e.speaker_id}: {e.text}" for e in all_entries]
    return "\n".join(lines)


def _get_before_context_summary() -> str:
    with _before_context_lock:
        return str(_before_context_summary or "")


def _set_before_context_summary(text: str) -> None:
    global _before_context_summary
    with _before_context_lock:
        _before_context_summary = str(text or "").strip()


def _get_all_dialogue_entries() -> list[dict]:
    """Get all dialogue entries across speakers in chronological order."""
    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_dialogue_chronological())
    all_entries.sort(key=lambda e: e.timestamp)
    return [
        {
            "timestamp": e.timestamp,
            "timestamp_iso_utc": datetime.fromtimestamp(
                e.timestamp, tz=timezone.utc
            ).isoformat(),
            "speaker": e.speaker_id,
            "source_id": e.source_id,
            "text": e.text,
        }
        for e in all_entries
    ]


def _save_session_full_dialogue_json(session_id: str, reason: str) -> None:
    """Append full session dialogue snapshots under dialogue/."""
    try:
        entries = _get_all_dialogue_entries()
        formatted_dialogue = "\n".join(
            f"{e['speaker']}: {e['text']}" for e in entries if e.get("text")
        )
        payload = {
            "session_id": session_id,
            "captured_at": datetime.now().isoformat(),
            "reason": reason,
            "entry_count": len(entries),
            "formatted_dialogue": formatted_dialogue,
            "entries": entries,
        }

        logs_dir = get_session_subdir(session_id, "dialogue")
        logs_dir.mkdir(parents=True, exist_ok=True)
        filepath = logs_dir / f"{session_id}_session_full_snapshots.json"
        _append_json_list(filepath, payload)

        log_print(
            "INFO",
            "Appended session full dialogue JSON",
            session_id=session_id,
            reason=reason,
            path=str(filepath),
            entry_count=len(entries),
        )
    except Exception as e:
        log_print(
            "ERROR",
            f"Failed to save session full dialogue JSON: {e}",
            session_id=session_id,
            reason=reason,
        )


def _build_before_context_via_gpt_mini(
    *,
    previous_context: str,
    recent_dialogue: str,
    keyword_block: str,
) -> str:
    """Build concise and accurate before_context from dialogue + keywords."""
    if (
        not previous_context.strip()
        and not recent_dialogue.strip()
        and not keyword_block.strip()
    ):
        return ""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You create context hints for downstream dialogue compression.\n"
                        "Must be concise, accurate, and grounded only in the provided inputs.\n"
                        "Do not invent facts.\n"
                        "Include: Topic/intent, and current conversation state."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Build before_context from the following.\n\n"
                        f"[Previous summarized context]\n{previous_context or '(none)'}\n\n"
                        f"[Recent dialogue (last 30-60s)]\n{recent_dialogue or '(none)'}\n\n"
                    ),
                },
            ],
            temperature=0.2,
        )
        text = str((response.choices[0].message.content or "")).strip()
        normalized = "\n".join(
            line.strip() for line in text.splitlines() if line.strip()
        )
        if normalized:
            _set_before_context_summary(normalized)
        return normalized
    except Exception as e:
        log_print("WARN", f"before_context gpt-mini build failed: {e}")
        return ""


def _build_highlevel_before_context(start_ts: float) -> str:
    """Build before-context using previous context + recent dialogue + keywords."""
    previous_context = _get_before_context_summary()
    recent_dialogue = _get_recent_dialogue_before_start_ts(start_ts)
    keyword_items = _get_current_keywords_with_desc(limit=3)
    keyword_hint_parts = []
    for item in keyword_items:
        word = str(item.get("word", "") or "").strip()
        desc = str(item.get("desc", "") or "").strip()
        if not word:
            continue
        if desc:
            keyword_hint_parts.append(f"{word}: {desc}")
        else:
            keyword_hint_parts.append(word)

    if not keyword_hint_parts:
        keyword_block = ""
    else:
        keyword_block = "; ".join(keyword_hint_parts)

    llm_context = _build_before_context_via_gpt_mini(
        previous_context=previous_context,
        recent_dialogue=recent_dialogue,
        keyword_block=keyword_block,
    )
    if llm_context:
        return llm_context

    if keyword_block:
        return f"Current keywords: {keyword_block}"
    return ""


def _collect_context_before(segment_id: int) -> str:
    with _segment_ctx_lock:
        window = _segment_windows.get(segment_id, {})
        start_ts = window.get("start_ts")
        if start_ts is None:
            return ""

        start_ts_value = float(start_ts)
    return _collect_context_before_start_ts(start_ts_value)


def _collect_context_before_start_ts(start_ts: float) -> str:
    return _build_highlevel_before_context(start_ts)


def _consume_next_sentence(segment_id: int) -> str:
    # Wait briefly to capture the first post-segment transcript.
    deadline = time.time() + 1.2
    while time.time() < deadline:
        with _segment_ctx_lock:
            window = _segment_windows.get(segment_id, {})
            text = str(window.get("next_sentence", "") or "").strip()
        if text:
            return text
        time.sleep(0.05)
    return ""


# -------------------------
# Dialogue Store and Transcript Compression
# -------------------------


def _reset_dialogue_stores() -> None:
    """Reset all dialogue stores."""
    global _dialogue_segment_start_time
    for store in _dialogue_stores.values():
        store.clear()
    _dialogue_stores.clear()
    _dialogue_segment_start_time = 0.0


def _make_vad_transcript_callback(speaker_id: str, session_id: str, source_id: str):
    """Create a VAD transcript callback that adds entries to the DialogueStore.

    Args:
        speaker_id: Speaker identifier ("A" or "B")
        session_id: Session ID for logging

    Returns:
        Callback function taking (transcript: str)
    """

    def _callback(transcript: str) -> None:
        if not transcript or not transcript.strip():
            return

        text = transcript.strip()

        # Get or create dialogue store for this speaker
        if speaker_id not in _dialogue_stores:
            _dialogue_stores[speaker_id] = DialogueStore()

        store = _dialogue_stores[speaker_id]
        entry = store.add_entry(
            speaker_id=speaker_id,
            text=text,
            source_id=source_id,
        )
        _append_session_transcript_entry(
            session_id=session_id,
            record={
                "type": "utterance",
                "captured_at": datetime.now().isoformat(),
                "timestamp": entry.timestamp,
                "timestamp_iso_utc": datetime.fromtimestamp(
                    entry.timestamp, tz=timezone.utc
                ).isoformat(),
                "speaker": entry.speaker_id,
                "source_id": entry.source_id,
                "text": entry.text,
            },
        )
        prune_cutoff = time.time() - _DIALOGUE_BUFFER_RETENTION_SEC
        _ = store.prune_before(prune_cutoff)

        log_print(
            "INFO",
            "VAD transcript added to DialogueStore",
            session_id=session_id,
            speaker=speaker_id,
            text=text[:50],
            timestamp=entry.timestamp,
        )

        # If summary pipeline is already running, trigger follow-up compression
        # immediately when new VAD arrives (until near-end cutoff).
        should_trigger_followup = False
        with _segment_ctx_lock:
            should_trigger_followup = (
                _post_tts_followup_active
                and _post_tts_followup_live_window_open
                and (not _post_tts_followup_inflight)
                and (len(_segment_compression_inflight) == 0)
                and entry.timestamp >= float(_post_tts_followup_cursor_ts or 0.0)
            )
        if should_trigger_followup:
            threading.Thread(
                target=_trigger_post_tts_followup_if_needed,
                args=(session_id, "vad_arrived"),
                daemon=True,
            ).start()

    return _callback


def _on_segment_compression_finished(segment_id: int, session_id: str) -> None:
    """Mark base segment compression completion and optionally release follow-up."""
    should_trigger_followup = False
    with _segment_ctx_lock:
        _segment_compression_inflight.discard(segment_id)
        should_trigger_followup = (
            _post_tts_followup_active
            and _post_tts_followup_live_window_open
            and (not _post_tts_followup_inflight)
            and (len(_segment_compression_inflight) == 0)
        )
    if should_trigger_followup:
        threading.Thread(
            target=_trigger_post_tts_followup_if_needed,
            args=(session_id, "segment_complete"),
            daemon=True,
        ).start()


_EVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
    "you",
    "we",
    "they",
    "i",
    "he",
    "she",
    "this",
    "those",
    "these",
}


def _tokenize_for_fidelity(text: str) -> list[str]:
    cleaned = re.sub(r"\b[AB]\s*:\s*", " ", (text or "").lower())
    tokens = re.findall(r"[a-z0-9']+", cleaned)
    return [t for t in tokens if t and (len(t) > 2 or t.isdigit())]


def _compute_dialogue_fidelity(original: str, compressed: str) -> float:
    """Approximate faithfulness score in [0, 1] using lexical overlap + speaker preservation."""
    orig_tokens = _tokenize_for_fidelity(original)
    comp_tokens = _tokenize_for_fidelity(compressed)
    if not orig_tokens or not comp_tokens:
        return 0.0

    orig_content = [t for t in orig_tokens if t not in _EVAL_STOPWORDS]
    comp_content = [t for t in comp_tokens if t not in _EVAL_STOPWORDS]
    if not orig_content or not comp_content:
        return 0.0

    orig_set = set(orig_content)
    comp_set = set(comp_content)
    overlap = len(orig_set & comp_set)
    if overlap <= 0:
        return 0.0

    precision = overlap / max(1, len(comp_set))
    recall = overlap / max(1, len(orig_set))
    f1 = (2 * precision * recall) / max(1e-9, (precision + recall))

    has_a_in_src = bool(re.search(r"(?:^|\n)\s*A\s*:", original))
    has_b_in_src = bool(re.search(r"(?:^|\n)\s*B\s*:", original))
    has_a_in_out = bool(re.search(r"(?:^|\n)\s*A\s*:", compressed))
    has_b_in_out = bool(re.search(r"(?:^|\n)\s*B\s*:", compressed))
    speaker_bonus = 0.0
    if has_a_in_src and has_a_in_out:
        speaker_bonus += 0.05
    if has_b_in_src and has_b_in_out:
        speaker_bonus += 0.05

    word_count = len(comp_tokens)
    length_penalty = 1.0
    if word_count > 35:
        length_penalty = 0.9
    if word_count > 50:
        length_penalty = 0.75

    score = min(1.0, (f1 + speaker_bonus) * length_penalty)
    return max(0.0, score)


def _extract_speakers(text: str) -> set[str]:
    return set(re.findall(r"(?:^|\n)\s*([AB])\s*:", text or ""))


_SUSPICIOUS_SINGLETON_TOKENS = {
    "you",
    "yeah",
    "yep",
    "yup",
    "uh",
    "um",
    "hmm",
    "hm",
    "ah",
    "oh",
    "ok",
    "okay",
    "right",
}


def _is_suspicious_singleton_utterance(text: str) -> bool:
    words = re.findall(r"[a-zA-Z']+", str(text or ""))
    if len(words) != 1:
        return False
    return words[0].lower() in _SUSPICIOUS_SINGLETON_TOKENS


def _build_literal_fallback(dialogue: str, max_words: int = 12) -> str:
    """Return short literal line(s) from source dialogue."""
    out_lines: list[str] = []
    for raw_line in (dialogue or "").splitlines():
        line = raw_line.strip()
        if not (line.startswith("A:") or line.startswith("B:")):
            continue
        speaker, _, text = line.partition(":")
        text = " ".join(text.strip().split())
        if not text:
            continue
        if _is_suspicious_singleton_utterance(text):
            continue
        words = text.split()
        out_lines.append(f"{speaker.strip()}: {' '.join(words[:max_words])}")
        if len(out_lines) >= 2:
            break
    return "\n".join(out_lines)


def _coerce_compressed_to_source(dialogue: str, compressed: str) -> str:
    """Constrain compressed output to source speakers and minimal lexical grounding."""
    src = (dialogue or "").strip()
    out = (compressed or "").strip()
    if not src:
        return ""
    if not out:
        return _build_literal_fallback(src)

    source_speakers = _extract_speakers(src)
    output_lines = []
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not (line.startswith("A:") or line.startswith("B:")):
            continue
        spk = line[0]
        if source_speakers and spk not in source_speakers:
            continue
        _, _, utterance = line.partition(":")
        if _is_suspicious_singleton_utterance(utterance):
            continue
        output_lines.append(line)
        if len(output_lines) >= 2:
            break

    # If source has one speaker, force one-speaker output.
    if len(source_speakers) == 1 and output_lines:
        only_speaker = next(iter(source_speakers))
        output_lines = [ln for ln in output_lines if ln.startswith(f"{only_speaker}:")]

    candidate = "\n".join(output_lines).strip()
    if not candidate:
        return _build_literal_fallback(src)

    # If lexical grounding is too weak, use literal fallback to avoid hallucination.
    fidelity = _compute_dialogue_fidelity(src, candidate)
    if fidelity < 0.12:
        fallback = _build_literal_fallback(src)
        return fallback or candidate
    return candidate


def _get_combined_dialogue() -> str:
    """Get combined dialogue from all stores, sorted chronologically."""
    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_dialogue_chronological())

    # Sort all entries by timestamp
    all_entries.sort(key=lambda e: e.timestamp)

    if not all_entries:
        return ""

    lines = [f"{e.speaker_id}: {e.text}" for e in all_entries]
    return "\n".join(lines)


def _get_dialogue_since_segment_start() -> str:
    """Get dialogue since the last segment start."""
    if _dialogue_segment_start_time <= 0:
        return _get_combined_dialogue()

    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_entries_since(_dialogue_segment_start_time))

    all_entries.sort(key=lambda e: e.timestamp)

    if not all_entries:
        return ""

    lines = [f"{e.speaker_id}: {e.text}" for e in all_entries]
    return "\n".join(lines)


def _get_dialogue_by_time_window(
    start_ts: float, end_ts: float
) -> tuple[str, list[dict]]:
    """Slice always-on transcript buffer by timestamp window."""
    all_entries = []
    for store in _dialogue_stores.values():
        all_entries.extend(store.get_entries_between(start_ts, end_ts))

    all_entries.sort(key=lambda e: e.timestamp)
    if not all_entries:
        return "", []

    lines = [f"{e.speaker_id}: {e.text}" for e in all_entries]
    entries_payload = [
        {
            "timestamp": e.timestamp,
            "timestamp_iso_utc": datetime.fromtimestamp(
                e.timestamp, tz=timezone.utc
            ).isoformat(),
            "speaker": e.speaker_id,
            "source_id": e.source_id,
            "text": e.text,
        }
        for e in all_entries
    ]
    return "\n".join(lines), entries_payload


def _append_json_list(filepath: Path, record: dict) -> None:
    """Append one record into a JSON list file."""
    existing: list = []
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                existing = loaded
        except Exception:
            existing = []
    existing.append(record)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def _append_session_transcript_entry(session_id: str, record: dict) -> None:
    """Append one utterance record to the session full transcript."""
    try:
        logs_dir = get_session_subdir(session_id, "dialogue")
        logs_dir.mkdir(parents=True, exist_ok=True)
        filepath = logs_dir / f"{session_id}_session_full_transcript.json"
        _append_json_list(filepath, record)
    except Exception as e:
        log_print(
            "ERROR",
            f"Failed to append session transcript entry: {e}",
            session_id=session_id,
        )


def _save_three_path_results_record(record: dict, session_id: str) -> None:
    """Save (time,input,output) 3-path comparison record.

    Writes both:
    1) session-scoped file
    2) all-sessions aggregate file
    """
    session_file = get_session_dir(session_id) / "three_path_results.json"
    with _three_path_results_lock:
        _append_json_list(session_file, record)
        _append_json_list(_all_sessions_three_path_file, record)


def _compress_dialogue_api(
    dialogue: str,
    segment_id: int,
    model: str,
    before_context: str = "",
    keyword_context: str = "",
    fallback_models: list[str] | None = None,
) -> dict:
    """Compress dialogue using OpenAI chat-completions API.

    Args:
        dialogue: The formatted dialogue string (A: ...\nB: ...)
        segment_id: The segment identifier

    Returns:
        Dictionary with compressed_text, elapsed_ms, method, and fidelity score
    """
    start_time = time.time()
    candidates = [model] + [m for m in (fallback_models or []) if m and m != model]
    last_error = ""

    for candidate in candidates:
        try:
            response = openai.chat.completions.create(
                model=candidate,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a dialogue compressor. Compress the dialogue into a brief "
                            "catch-up (max 30 words total). Preserve speaker labels (A: and B:). "
                            "Remove filler and keep only core ideas. Output only dialogue lines. "
                            "Each speaker line must be 12 words or fewer. "
                            "If a token/word is clearly odd or context-mismatched (likely transcript error), ignore it. "
                            "Drop content that is unrelated to current context or user-viewed keywords. "
                            "Treat isolated one-word utterances as likely transcription errors unless clearly meaningful. "
                            "Never invent claims/questions not in Dialogue. "
                            "If Dialogue has one speaker, output only that speaker."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Compress this dialogue.\n\n"
                            f"Before context (hints only):\n{before_context.strip() or '(none)'}\n\n"
                            f"Current viewed keywords:\n{keyword_context.strip() or '(none)'}\n\n"
                            f"Dialogue:\n{dialogue}\n\n"
                            "Rules: Context is hint-only. Do not import extra facts from context. "
                            "If dialogue is short/noisy, output short literal fragment only. "
                            "Drop context-mismatched weird tokens rather than guessing replacements. "
                            "Exclude lines/phrases unrelated to the active context or viewed keywords. "
                            "Treat one-word fragments as possible transcription errors and drop them when uncertain. "
                            "Hard limit: each A:/B: line <= 12 words."
                        ),
                    },
                ],
                max_tokens=100,
                temperature=0.6,
            )

            compressed = response.choices[0].message.content or ""
            elapsed_ms = (time.time() - start_time) * 1000

            # Normalize output to A:/B: format
            lines = []
            for raw_line in compressed.splitlines():
                line = raw_line.strip()
                if line.startswith("A:") or line.startswith("B:"):
                    lines.append(line)
                if len(lines) >= 3:
                    break

            normalized = "\n".join(lines) if lines else compressed[:400]
            normalized = _coerce_compressed_to_source(dialogue, normalized)
            fidelity = _compute_dialogue_fidelity(dialogue, normalized)
            method = f"api_{candidate.replace('-', '_')}"

            return {
                "compressed_text": normalized,
                "elapsed_ms": elapsed_ms,
                "fidelity_score": fidelity,
                "requested_model": model,
                "model": candidate,
                "fallback_used": candidate != model,
                "method": method,
                "segment_id": segment_id,
            }
        except Exception as e:
            last_error = str(e)
            is_last = candidate == candidates[-1]
            log_level = "ERROR" if is_last else "WARN"
            log_print(
                log_level,
                f"API compression failed for model {candidate}: {e}",
                segment_id=segment_id,
                model=candidate,
                requested_model=model,
            )
            if is_last:
                break

    elapsed_ms = (time.time() - start_time) * 1000
    method = f"api_{model.replace('-', '_')}"
    return {
        "compressed_text": "",
        "elapsed_ms": elapsed_ms,
        "fidelity_score": 0.0,
        "requested_model": model,
        "model": model,
        "fallback_used": False,
        "method": method,
        "segment_id": segment_id,
        "error": last_error or "unknown_error",
    }


def _save_dialogue_json(
    dialogue: str,
    segment_id: int,
    session_id: str,
    entries_override: list[dict] | None = None,
    window: dict | None = None,
) -> None:
    """Save dialogue transcript to JSON file for debugging.

    Args:
        dialogue: The formatted dialogue string
        segment_id: The segment identifier
        session_id: Session ID for file naming
    """
    try:
        # Get segment-scoped entries with timestamps (speaker + source + text).
        all_entries = list(entries_override or [])
        if not all_entries:
            for store in _dialogue_stores.values():
                entries = (
                    store.get_entries_since(_dialogue_segment_start_time)
                    if _dialogue_segment_start_time > 0
                    else store.get_dialogue_chronological()
                )
                for entry in entries:
                    all_entries.append(
                        {
                            "timestamp": entry.timestamp,
                            "timestamp_iso_utc": datetime.fromtimestamp(
                                entry.timestamp, tz=timezone.utc
                            ).isoformat(),
                            "speaker": entry.speaker_id,
                            "source_id": entry.source_id,
                            "text": entry.text,
                        }
                    )

        # Sort by timestamp
        all_entries.sort(key=lambda e: e["timestamp"])

        # Create dialogue log
        dialogue_log = {
            "segment_id": segment_id,
            "session_id": session_id,
            "captured_at": datetime.now().isoformat(),
            "segment_start_time": _dialogue_segment_start_time,
            "window": window or {},
            "entry_count": len(all_entries),
            "formatted_dialogue": dialogue,
            "entries": all_entries,
        }

        # Save to session-scoped dialogue directory
        logs_dir = get_session_subdir(session_id, "dialogue")
        logs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_seg{segment_id}_{session_id}_dialogue.json"
        filepath = logs_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(dialogue_log, f, indent=2, ensure_ascii=False)

        sio.emit(
            "dialogue_transcript_ready",
            {
                "segment_id": segment_id,
                "entry_count": len(all_entries),
                "formatted_dialogue": dialogue,
                "entries": all_entries,
                "path": str(filepath),
            },
        )

        log_print(
            "INFO",
            f"Dialogue saved to {filepath}",
            segment_id=segment_id,
            entry_count=len(all_entries),
        )
    except Exception as e:
        log_print(
            "ERROR",
            f"Failed to save dialogue JSON: {e}",
            segment_id=segment_id,
        )


def _trigger_parallel_compression_for_dialogue(
    dialogue: str,
    segment_id: int,
    session_id: str,
    before_context: str = "",
    trigger_source: str = "segment",
    window: dict | None = None,
    entries: list[dict] | None = None,
) -> None:
    """Run 3-path compression for provided dialogue and auto-select output."""
    session_logger = get_logger(session_id)
    if trigger_source == "segment":
        with _segment_ctx_lock:
            _segment_compression_inflight.add(segment_id)

    if not dialogue or not dialogue.strip():
        log_print(
            "INFO",
            "No dialogue to compress",
            session_id=session_id,
            segment_id=segment_id,
            trigger_source=trigger_source,
        )
        sio.emit(
            "judge_rejected",
            {
                "reason": "All caught up",
                "segment_id": segment_id,
                "summary": "",
                "no_dialogue": True,
            },
            to=session_id,
        )
        if trigger_source == "segment":
            _on_segment_compression_finished(segment_id, session_id)
        return

    # Save dialogue to JSON for debugging
    _save_dialogue_json(
        dialogue,
        segment_id,
        session_id,
        entries_override=entries,
        window=window,
    )

    log_print(
        "INFO",
        "Triggering parallel compression",
        session_id=session_id,
        segment_id=segment_id,
        dialogue_chars=len(dialogue),
        before_context_chars=len(before_context or ""),
        trigger_source=trigger_source,
    )
    keyword_items = _get_current_keywords_with_desc(limit=3)
    keyword_context = "; ".join(
        (
            f"{str(k.get('word', '')).strip()}: {str(k.get('desc', '')).strip()}"
            if str(k.get("desc", "")).strip()
            else str(k.get("word", "")).strip()
        )
        for k in keyword_items
        if str(k.get("word", "")).strip()
    )

    results = {
        "segment_id": segment_id,
        "original_dialogue": dialogue,
        "realtime": None,
        "api_mini": None,
        "api_nano": None,
    }
    results_lock = threading.Lock()
    results_ready = threading.Event()
    expected_paths = 3

    def _on_realtime_result(payload: dict) -> None:
        raw_text = str(payload.get("compressed_text", "") or "")
        text = _coerce_compressed_to_source(dialogue, raw_text)
        elapsed_ms = float(payload.get("elapsed_ms", 0.0) or 0.0)
        fidelity = _compute_dialogue_fidelity(dialogue, text)
        with results_lock:
            results["realtime"] = {
                "text": text,
                "elapsed_ms": elapsed_ms,
                "fidelity_score": fidelity,
                "method": "realtime_api",
            }
            done_count = sum(
                1
                for key in ("realtime", "api_mini", "api_nano")
                if results.get(key) is not None
            )
            if done_count >= expected_paths:
                results_ready.set()

    def _run_api_path(
        model: str, key: str, fallback_models: list[str] | None = None
    ) -> None:
        api_result = _compress_dialogue_api(
            dialogue,
            segment_id,
            model,
            before_context=before_context,
            keyword_context=keyword_context,
            fallback_models=fallback_models,
        )
        compressed_text = str(api_result.get("compressed_text", "") or "")
        log_print(
            "INFO",
            "API compression result",
            session_id=session_id,
            segment_id=segment_id,
            trigger_source=trigger_source,
            requested_model=api_result.get("requested_model", model),
            model=api_result.get("model", model),
            fallback_used=api_result.get("fallback_used", False),
            elapsed_ms=api_result.get("elapsed_ms"),
            fidelity_score=api_result.get("fidelity_score"),
            text_preview=compressed_text[:200],
            has_error=bool(api_result.get("error")),
        )
        session_logger.log(
            "api_compression_result",
            segment_id=segment_id,
            trigger_source=trigger_source,
            key=key,
            requested_model=api_result.get("requested_model", model),
            model=api_result.get("model", model),
            fallback_used=api_result.get("fallback_used", False),
            elapsed_ms=api_result.get("elapsed_ms"),
            fidelity_score=api_result.get("fidelity_score"),
            compressed_text=compressed_text,
            error=api_result.get("error"),
        )
        with results_lock:
            results[key] = {
                "text": api_result.get("compressed_text", ""),
                "elapsed_ms": api_result.get("elapsed_ms", 0),
                "fidelity_score": api_result.get("fidelity_score", 0.0),
                "method": api_result.get("method", f"api_{model}"),
            }
            done_count = sum(
                1
                for k in ("realtime", "api_mini", "api_nano")
                if results.get(k) is not None
            )
            if done_count >= expected_paths:
                results_ready.set()

        # Emit API result immediately
        sio.emit("transcript_compressed_api", api_result)

    # Path 1: Realtime API (if available)
    with _clients_lock:
        tr = transcript_reconstructor
    if tr and tr.running:
        # Set temporary callback to capture result
        original_callback = tr.on_reconstruction_callback
        tr.on_reconstruction_callback = _on_realtime_result
        tr.reconstruct_transcript(
            dialogue,
            segment_id,
            before_context=before_context,
            keyword_context=keyword_context,
        )

        # Restore original callback after a delay
        def _restore_callback():
            time.sleep(5.0)
            with _clients_lock:
                if transcript_reconstructor:
                    transcript_reconstructor.on_reconstruction_callback = (
                        original_callback
                    )

        threading.Thread(target=_restore_callback, daemon=True).start()
    else:
        # No Realtime client - mark as skipped
        with results_lock:
            results["realtime"] = {
                "text": "",
                "elapsed_ms": 0,
                "fidelity_score": 0.0,
                "skipped": True,
                "method": "realtime_api",
            }

    # Path 2/3: API calls (background threads)
    threading.Thread(
        target=_run_api_path,
        args=("gpt-4o-mini", "api_mini", ["gpt-4.1-mini"]),
        daemon=True,
    ).start()
    threading.Thread(
        target=_run_api_path,
        args=("gpt-4o-nano", "api_nano", ["gpt-4.1-nano"]),
        daemon=True,
    ).start()

    # Wait for all results and emit comparison (with timeout)
    def _emit_comparison():
        results_ready.wait(timeout=10.0)
        with results_lock:
            candidates = []
            for key in ("realtime", "api_mini", "api_nano"):
                item = results.get(key)
                if not item or not item.get("text"):
                    continue
                candidates.append(
                    {
                        "key": key,
                        "text": item.get("text", ""),
                        "elapsed_ms": float(item.get("elapsed_ms", 0.0) or 0.0),
                        "method": str(item.get("method", key)),
                    }
                )

            selected = None
            by_key = {c["key"]: c for c in candidates}
            mode = _transcript_compression_mode
            if mode == "fastest":
                selected = min(
                    candidates,
                    key=lambda c: float(c.get("elapsed_ms", 0.0) or 0.0),
                    default=None,
                )
            elif mode in ("realtime", "api_mini", "api_nano"):
                selected = by_key.get(mode)
                if not selected:
                    selected = (
                        by_key.get("realtime")
                        or by_key.get("api_mini")
                        or by_key.get("api_nano")
                    )
            else:
                selected = (
                    by_key.get("realtime")
                    or by_key.get("api_mini")
                    or by_key.get("api_nano")
                )

            comparison = {
                "segment_id": segment_id,
                "trigger_source": trigger_source,
                "window": window or {},
                "entry_count": len(entries or []),
                "before_context": before_context,
                "original_dialogue": dialogue,
                "realtime": results.get("realtime"),
                "api_mini": results.get("api_mini"),
                "api_nano": results.get("api_nano"),
                "selection_mode": mode,
                "selected": selected,
            }
        three_path_record = {
            "time": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "segment_id": segment_id,
            "trigger_source": trigger_source,
            "input": dialogue,
            "before_context": before_context,
            "keyword_context": keyword_context,
            "output": {
                "realtime": (comparison.get("realtime") or {}).get("text", ""),
                "api_mini": (comparison.get("api_mini") or {}).get("text", ""),
                "api_nano": (comparison.get("api_nano") or {}).get("text", ""),
                "selected": (comparison.get("selected") or {}).get("method", ""),
            },
            "meta": {
                "realtime_elapsed_ms": (comparison.get("realtime") or {}).get(
                    "elapsed_ms"
                ),
                "api_mini_elapsed_ms": (comparison.get("api_mini") or {}).get(
                    "elapsed_ms"
                ),
                "api_nano_elapsed_ms": (comparison.get("api_nano") or {}).get(
                    "elapsed_ms"
                ),
                "window": comparison.get("window") or {},
                "entry_count": comparison.get("entry_count", 0),
            },
        }
        _save_three_path_results_record(three_path_record, session_id)
        sio.emit("transcript_compression_comparison", comparison)
        realtime_info = comparison.get("realtime") or {}
        api_mini_info = comparison.get("api_mini") or {}
        api_nano_info = comparison.get("api_nano") or {}
        log_print(
            "INFO",
            "Compression comparison emitted",
            session_id=session_id,
            segment_id=segment_id,
            selected=(selected or {}).get("method"),
            trigger_source=trigger_source,
            realtime_elapsed_ms=realtime_info.get("elapsed_ms"),
            api_mini_elapsed_ms=api_mini_info.get("elapsed_ms"),
            api_nano_elapsed_ms=api_nano_info.get("elapsed_ms"),
        )

        chosen_text = str((selected or {}).get("text", "") or "")
        chosen_method = str((selected or {}).get("method", "") or "")

        if chosen_text:
            _synthesize_compressed_dialogue(
                chosen_text,
                segment_id,
                chosen_method,
                trigger_source=trigger_source,
            )
            if trigger_source == "segment":
                _on_segment_compression_finished(segment_id, session_id)
            return

        # All paths returned empty/invalid output: end summary flow explicitly.
        sio.emit(
            "judge_rejected",
            {
                "reason": "All caught up",
                "segment_id": segment_id,
                "summary": "",
                "no_dialogue": True,
            },
            to=session_id,
        )
        if trigger_source == "segment":
            _on_segment_compression_finished(segment_id, session_id)

    threading.Thread(target=_emit_comparison, daemon=True).start()


def _trigger_parallel_compression(segment_id: int, session_id: str) -> None:
    """Compatibility path: use last listening-segment dialogue."""
    dialogue = ""
    entries: list[dict] = []
    window_payload: dict = {}

    # Prefer segment-window slicing to avoid races with subsequent listening sessions.
    with _segment_ctx_lock:
        segment_window = dict(_segment_windows.get(segment_id, {}))
    start_ts = float(segment_window.get("start_ts") or 0.0)
    end_ts = float(segment_window.get("end_ts") or 0.0)

    if start_ts > 0:
        if end_ts <= 0:
            end_ts = time.time()
        slice_end_ts = end_ts + _SEGMENT_TAIL_GRACE_SEC
        window_payload = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "slice_end_ts": slice_end_ts,
            "tail_grace_sec": _SEGMENT_TAIL_GRACE_SEC,
        }

        # Watermark gate: wait until observed transcript count stops growing
        # for a quiet window, then compress.
        wait_start = time.time()
        stable_since = wait_start
        target_watermark = 0
        poll_count = 0

        while True:
            dialogue, entries = _get_dialogue_by_time_window(start_ts, slice_end_ts)
            entry_count = len(entries)
            now = time.time()
            poll_count += 1

            if entry_count > target_watermark:
                target_watermark = entry_count
                stable_since = now
                if target_watermark > 0:
                    log_print(
                        "INFO",
                        "Segment watermark advanced",
                        session_id=session_id,
                        segment_id=segment_id,
                        target_watermark=target_watermark,
                        elapsed_ms=int((now - wait_start) * 1000),
                    )

            watermark_reached = entry_count >= target_watermark
            quiet_elapsed = now - stable_since
            quiet_enough = quiet_elapsed >= _SEGMENT_DIALOGUE_QUIET_WINDOW_SEC
            timed_out = (now - wait_start) >= _SEGMENT_DIALOGUE_MAX_WAIT_SEC

            if (watermark_reached and quiet_enough) or timed_out:
                if timed_out:
                    log_print(
                        "WARN",
                        "Segment dialogue wait timeout; compressing with current transcripts",
                        session_id=session_id,
                        segment_id=segment_id,
                        received_entries=entry_count,
                        target_watermark=target_watermark,
                        max_wait_sec=_SEGMENT_DIALOGUE_MAX_WAIT_SEC,
                    )
                elif target_watermark > 0:
                    log_print(
                        "INFO",
                        "Segment watermark settled; starting compression",
                        session_id=session_id,
                        segment_id=segment_id,
                        settled_entries=entry_count,
                        quiet_ms=int(quiet_elapsed * 1000),
                        polls=poll_count,
                    )
                break

            time.sleep(_SEGMENT_DIALOGUE_POLL_SEC)
    else:
        dialogue = _get_dialogue_since_segment_start()

    before_context = _collect_context_before(segment_id)

    _trigger_parallel_compression_for_dialogue(
        dialogue=dialogue,
        segment_id=segment_id,
        session_id=session_id,
        before_context=before_context,
        trigger_source="segment",
        window=window_payload,
        entries=entries,
    )


def _trigger_parallel_compression_after_delay(
    segment_id: int,
    session_id: str,
    delay_sec: float = _SEGMENT_POST_END_WAIT_SEC,
) -> None:
    """Delay compression start to allow late VAD transcripts to arrive."""
    if delay_sec > 0:
        time.sleep(delay_sec)
    _trigger_parallel_compression(segment_id, session_id)


def _try_fast_catchup_for_segment(segment_id: int, session_id: str) -> bool:
    """Try iterative source-audio fast catch-up for one listening segment."""
    with _segment_ctx_lock:
        segment_window = dict(_segment_windows.get(segment_id, {}))
    start_ts = float(segment_window.get("start_ts") or 0.0)
    end_ts = float(segment_window.get("end_ts") or 0.0)
    if start_ts <= 0:
        return False
    if end_ts <= start_ts:
        end_ts = time.time()
    end_ts = _snap_end_to_vad_stop(end_ts)

    # Round 1: the original missed segment [start, end]
    cursor_start = start_ts
    cursor_end = end_ts
    emitted_any = False

    for step in range(_FAST_CATCHUP_CHAIN_MAX_STEPS):
        lag_sec = max(0.0, cursor_end - cursor_start)
        if lag_sec <= _FAST_CATCHUP_CHAIN_MIN_LAG_SEC:
            break

        dialogue, _entries = _get_dialogue_by_time_window(cursor_start, cursor_end)
        output_sec = _synthesize_fast_catchup_dialogue(
            dialogue=dialogue,
            segment_id=segment_id,
            session_id=session_id,
            start_ts=cursor_start,
            end_ts=cursor_end,
            speed=_FAST_CATCHUP_DEFAULT_SPEED,
            gap_sec=_FAST_CATCHUP_DEFAULT_GAP_SEC,
            silence_thresh_db=_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB,
            trigger_source=f"segment_fast_catchup_source_step{step + 1}",
        )
        if output_sec <= 0:
            return emitted_any
        emitted_any = True

        # Wait approximately for emitted audio playback, then catch the newly
        # accumulated lag window [previous_end, now].
        time.sleep(min(8.0, output_sec))
        cursor_start = cursor_end
        cursor_end = _snap_end_to_vad_stop(time.time())

    return emitted_any


def _trigger_post_tts_followup_if_needed(session_id: str, reason: str = "") -> None:
    """If new VAD arrived after end_listening, compress+TTS it as follow-up."""
    global \
        _segment_seq, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts, \
        _post_tts_followup_active
    global _post_tts_followup_live_window_open

    if reason == "user_cancel":
        with _segment_ctx_lock:
            _post_tts_followup_active = False
            _post_tts_followup_inflight = False
            _post_tts_followup_live_window_open = False
        return

    should_wait = reason not in {"summary_tts_near_end", "vad_arrived"}
    if should_wait and _POST_TTS_FOLLOWUP_WAIT_SEC > 0:
        time.sleep(_POST_TTS_FOLLOWUP_WAIT_SEC)

    with _segment_ctx_lock:
        if not _post_tts_followup_active or _post_tts_followup_inflight:
            return
        if len(_segment_compression_inflight) > 0:
            return
        start_ts = float(_post_tts_followup_cursor_ts or 0.0)
        if start_ts <= 0:
            _post_tts_followup_active = False
            return
        end_ts = time.time()
        if end_ts <= start_ts:
            return
        _post_tts_followup_inflight = True

    dialogue, entries = _get_dialogue_by_time_window(start_ts, end_ts)
    entry_count = len(entries)

    with _segment_ctx_lock:
        if entries:
            last_ts = float(entries[-1].get("timestamp") or end_ts)
            _post_tts_followup_cursor_ts = max(last_ts + 0.001, end_ts)
        else:
            _post_tts_followup_active = False
            _post_tts_followup_inflight = False
            _post_tts_followup_cursor_ts = end_ts
            return

        _segment_seq += 1
        segment_id = _segment_seq
        _segment_windows[segment_id] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "next_sentence": "",
        }

    before_context = _collect_context_before_start_ts(start_ts)

    log_print(
        "INFO",
        "Triggering post-TTS follow-up compression",
        session_id=session_id,
        segment_id=segment_id,
        reason=reason or "browser_tts_done",
        entries=entry_count,
    )

    def _run_followup():
        global _post_tts_followup_inflight
        try:
            _trigger_parallel_compression_for_dialogue(
                dialogue=dialogue,
                segment_id=segment_id,
                session_id=session_id,
                before_context=before_context,
                trigger_source="post_tts_followup",
                window={
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                },
                entries=entries,
            )
        finally:
            with _segment_ctx_lock:
                _post_tts_followup_inflight = False

    threading.Thread(target=_run_followup, daemon=True).start()


def _apply_fast_catchup_to_segment(
    audio: AudioSegment,
    *,
    speed: float = _FAST_CATCHUP_DEFAULT_SPEED,
    gap_sec: float = _FAST_CATCHUP_DEFAULT_GAP_SEC,
    silence_thresh_db: float = _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB,
) -> AudioSegment:
    """Apply gap removal + speed-up to an AudioSegment."""
    if audio is None or len(audio) <= 0:
        return audio
    speed = max(1.0, min(3.0, float(speed or _FAST_CATCHUP_DEFAULT_SPEED)))
    gap_sec = max(0.1, min(2.0, float(gap_sec or _FAST_CATCHUP_DEFAULT_GAP_SEC)))
    min_silence_len_ms = int(gap_sec * 1000.0)
    keep_silence_ms = 35

    # Pre-denoise before silence trim/speed-up:
    # 1) speech-band filters, 2) attenuate very low-level chunks.
    if _FAST_CATCHUP_DENOISE_ENABLED:
        try:
            audio = audio.high_pass_filter(90).low_pass_filter(7600)
            chunk_ms = max(10, int(_FAST_CATCHUP_DENOISE_CHUNK_MS))
            noise_chunks = make_chunks(audio, chunk_ms)
            noise_floor = (
                min(-34.0, float(audio.dBFS) - 12.0)
                if audio.dBFS != float("-inf")
                else -40.0
            )
            denoised = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
            for chunk in noise_chunks:
                if chunk.dBFS == float("-inf") or chunk.dBFS < noise_floor:
                    denoised += chunk - float(_FAST_CATCHUP_DENOISE_NOISE_ATTEN_DB)
                else:
                    denoised += chunk
            if len(denoised) > 0:
                audio = denoised
        except Exception:
            pass

    try:
        spans = silence.detect_nonsilent(
            audio,
            min_silence_len=min_silence_len_ms,
            silence_thresh=float(silence_thresh_db),
        )
        if spans:
            compact = AudioSegment.silent(duration=0, frame_rate=audio.frame_rate)
            for start_ms, end_ms in spans:
                lo = max(0, start_ms - keep_silence_ms)
                hi = min(len(audio), end_ms + keep_silence_ms)
                compact += audio[lo:hi]
            audio = compact
    except Exception:
        pass

    if speed > 1.01:
        try:
            audio = effects.speedup(
                audio, playback_speed=speed, chunk_size=120, crossfade=20
            )
        except Exception:
            pass

    # Match perceived level close to Cartesia TTS:
    # - raise RMS toward target
    # - clamp by peak headroom to avoid clipping
    try:
        if audio.max_dBFS != float("-inf"):
            rms_gain = float(_FAST_CATCHUP_TARGET_RMS_DBFS) - float(audio.dBFS)
            peak_headroom = float(_FAST_CATCHUP_TARGET_PEAK_DBFS) - float(audio.max_dBFS)
            gain_db = min(rms_gain, peak_headroom)
            if gain_db > 0.0:
                audio = audio.apply_gain(gain_db)
    except Exception:
        pass
    # User-requested extra loudness boost (~3x amplitude).
    try:
        audio = audio.apply_gain(float(_FAST_CATCHUP_EXTRA_GAIN_DB))
    except Exception:
        pass
    return audio


def _save_fast_catchup_audio(
    *,
    session_id: str,
    segment_id: int,
    wav_bytes: bytes,
    speed: float,
    gap_sec: float,
) -> Optional[str]:
    """Persist fast-catchup audio for debugging/replay."""
    if not wav_bytes:
        return None
    try:
        out_dir = get_session_subdir(session_id, "fast_catchup")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{timestamp}_seg{segment_id}_x{speed:.2f}_gap{gap_sec:.2f}_fastcatchup.wav"
        )
        path = out_dir / filename
        with open(path, "wb") as f:
            f.write(wav_bytes)
        return str(path)
    except Exception as e:
        log_print(
            "WARN",
            f"Failed to save fast catch-up audio: {e}",
            session_id=session_id,
            segment_id=segment_id,
        )
        return None


def _synthesize_fast_catchup_dialogue(
    dialogue: str,
    segment_id: int,
    session_id: str,
    *,
    start_ts: float = 0.0,
    end_ts: float = 0.0,
    speed: float = _FAST_CATCHUP_DEFAULT_SPEED,
    gap_sec: float = _FAST_CATCHUP_DEFAULT_GAP_SEC,
    silence_thresh_db: float = _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB,
    trigger_source: str = "missed_fast_catchup",
) -> float:
    """Emit fast catch-up from original source audio (not TTS)."""
    with _clients_lock:
        current_client = client
    if not current_client:
        return 0.0

    try:
        pcm = current_client.get_audio_window_pcm(start_ts, end_ts)
    except Exception:
        return 0.0
    if not pcm:
        return 0.0

    source_audio = AudioSegment(
        data=pcm,
        sample_width=2,
        frame_rate=24000,
        channels=1,
    )
    processed_audio = _apply_fast_catchup_to_segment(
        source_audio,
        speed=speed,
        gap_sec=gap_sec,
        silence_thresh_db=silence_thresh_db,
    )
    if processed_audio is None or len(processed_audio) <= 0:
        return 0.0
    out_io = io.BytesIO()
    processed_audio.export(out_io, format="wav")
    processed_wav = out_io.getvalue()
    if not processed_wav:
        return 0.0
    saved_path = _save_fast_catchup_audio(
        session_id=session_id,
        segment_id=segment_id,
        wav_bytes=processed_wav,
        speed=speed,
        gap_sec=gap_sec,
    )

    method = f"fast_catchup_source_{speed:.2f}x_nogap{gap_sec:.2f}s"
    sio.emit(
        "compressed_dialogue_tts",
        {
            "segment_id": segment_id,
            "audio": base64.b64encode(processed_wav).decode("utf-8"),
            "text": str(dialogue or ""),
            "format": "wav",
            "sample_rate": 24000,
            "method": method,
            "trigger_source": trigger_source,
            "turn_count": 0,
            "saved_path": saved_path,
        },
        to=session_id,
    )
    log_print(
        "INFO",
        "Fast catch-up emitted (source audio)",
        session_id=session_id,
        segment_id=segment_id,
        source_duration_ms=len(source_audio),
        output_duration_ms=len(processed_audio),
        speed=speed,
        gap_sec=gap_sec,
        silence_thresh_db=silence_thresh_db,
        saved_path=saved_path,
    )
    return max(0.0, len(processed_audio) / 1000.0)


def _synthesize_compressed_dialogue(
    compressed_text: str,
    segment_id: int,
    method: str = "unknown",
    trigger_source: str = "segment",
) -> None:
    """Synthesize TTS for compressed dialogue and emit as merged audio.

    Args:
        compressed_text: The compressed dialogue text (A: ...\nB: ...)
        segment_id: The segment identifier
        method: The compression method used (for logging)
    """
    if not compressed_text or not compressed_text.strip():
        log_print(
            "WARN",
            "No compressed text for TTS",
            segment_id=segment_id,
        )
        return

    turns = _parse_reconstructed_turns(compressed_text)
    if not turns:
        log_print(
            "WARN",
            "No turns parsed from compressed text",
            segment_id=segment_id,
            text=compressed_text[:100],
        )
        return

    audio_items: list[
        tuple[bytes, str, str, int]
    ] = []  # (audio_bytes, speaker, text, idx)

    for i, (speaker, text) in enumerate(turns):
        with _clients_lock:
            tts = _get_tts_client_for_speaker(speaker)
        if not tts:
            continue

        audio_bytes = tts.synthesize(text, language="en")
        if not audio_bytes:
            continue
        audio_items.append((audio_bytes, speaker, text, i))

    if not audio_items:
        log_print(
            "WARN",
            "No audio synthesized for compressed dialogue",
            segment_id=segment_id,
        )
        return

    # Merge audio items into single WAV
    def _merge_wav_items_compressed(
        items: list[tuple[bytes, str, str, int]],
    ) -> Optional[bytes]:
        """Concatenate multiple mono 16-bit 24kHz WAV bytes into one WAV."""
        if not items:
            return None
        pcm_chunks: list[bytes] = []
        for audio_bytes, _, _, _ in items:
            try:
                with io.BytesIO(audio_bytes) as wav_io:
                    with wave.open(wav_io, "rb") as wf:
                        if (
                            wf.getnchannels() != 1
                            or wf.getsampwidth() != 2
                            or wf.getframerate() != 24000
                        ):
                            log_print(
                                "WARN",
                                "Unexpected compressed TTS WAV format; skipping merge",
                                segment_id=segment_id,
                            )
                            return None
                        pcm_chunks.append(wf.readframes(wf.getnframes()))
            except Exception as e:
                log_print(
                    "ERROR",
                    f"Failed to parse compressed TTS WAV: {e}",
                    segment_id=segment_id,
                )
                return None

        merged_pcm = b"".join(pcm_chunks)
        if not merged_pcm:
            return None
        out_io = io.BytesIO()
        with wave.open(out_io, "wb") as out_wav:
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(24000)
            out_wav.writeframes(merged_pcm)
        return out_io.getvalue()

    merged_wav = _merge_wav_items_compressed(audio_items)
    if merged_wav:
        sio.emit(
            "compressed_dialogue_tts",
            {
                "segment_id": segment_id,
                "audio": base64.b64encode(merged_wav).decode("utf-8"),
                "text": compressed_text,
                "format": "wav",
                "sample_rate": 24000,
                "method": method,
                "trigger_source": trigger_source,
                "turn_count": len(audio_items),
            },
        )
        log_print(
            "INFO",
            f"compressed_dialogue_tts emitted: count={len(audio_items)}, bytes={len(merged_wav)}",
            segment_id=segment_id,
            method=method,
        )


def _parse_reconstructed_turns(text: str) -> list[tuple[str, str]]:
    """Parse reconstructed dialogue lines preserving order."""
    turns: list[tuple[str, str]] = []
    normalized = str(text or "")
    # Handle escaped newlines from model output logs/serialization.
    normalized = (
        normalized.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    )

    # Parse turn blocks robustly across multiline output.
    pattern = (
        r"(?:^|\n)\s*(?:SPEAKER\s*)?([AB])\s*(?::|-|–|—)\s*(.+?)"
        r"(?=(?:\n\s*(?:SPEAKER\s*)?[AB]\s*(?::|-|–|—)\s*)|\Z)"
    )
    for m in re.finditer(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
        speaker = m.group(1).upper()
        utterance = " ".join(m.group(2).strip().split())
        if not utterance:
            continue
        turns.append((speaker, utterance))
        if len(turns) >= 3:
            break
    return turns


def _get_tts_client_for_speaker(speaker: str) -> Optional[TTSClient]:
    idx = 0 if speaker == "A" else 1
    if idx < len(summary_clients):
        sc = summary_clients[idx]
        if sc.tts_client:
            return sc.tts_client
    if summary_clients:
        for sc in summary_clients:
            if sc.tts_client:
                return sc.tts_client
    return keyword_tts_client


def _on_reconstruction_result(payload: dict) -> None:
    """Emit reconstructed turns (TTS disabled - handled by transcript compression flow)."""
    conversation = str((payload or {}).get("conversation", "")).strip()
    sum0 = str((payload or {}).get("sum0", "")).strip()
    sum1 = str((payload or {}).get("sum1", "")).strip()
    segment_id = int((payload or {}).get("segment_id", 0) or 0)
    turns = _parse_reconstructed_turns(conversation)
    if not turns:
        # Fallback: if model format drifts, still provide ordered A/B turns.
        if sum0:
            turns.append(("A", sum0))
        if sum1:
            turns.append(("B", sum1))
        if not turns and conversation:
            turns.append(("A", conversation[:240]))
        if not turns:
            return

    sio.emit(
        "conversation_reconstructed_turns",
        {
            "segment_id": segment_id,
            "turns": [{"speaker": s, "text": t} for s, t in turns],
        },
    )
    sio.emit(
        "conversation_reconstruct_done",
        {
            "segment_id": segment_id,
            "turn_count": len(turns),
        },
    )

    # TTS synthesis disabled here - now handled by transcript compression flow
    # via _trigger_parallel_compression() -> _synthesize_compressed_dialogue()
    # which uses actual VAD transcripts instead of summaries


def _is_active_session(session_id: str) -> bool:
    return _active_runtime_sid is None or session_id == _active_runtime_sid


def _normalize_client_ts(raw: object) -> float:
    """Normalize client timestamp to unix seconds.

    Accepts seconds or milliseconds epoch.
    """
    try:
        ts = float(raw)  # type: ignore[arg-type]
    except Exception:
        return 0.0
    if ts <= 0:
        return 0.0
    # 13-digit epoch milliseconds.
    if ts >= 1_000_000_000_000:
        return ts / 1000.0
    return ts


def _flush_judge_batch(segment_id: int, session_id: str) -> None:
    """Flush aggregated summaries for a segment into judge/reconstructor requests."""
    with _judge_batch_lock:
        batch = _judge_batch.get(segment_id)
        judge = context_judge
        reconstructor = conversation_reconstructor
        if (
            not batch
            or batch.get("sent")
            or (not judge and not reconstructor)
            or segment_id in _judge_completed_segments
        ):
            return

        summaries_by_source: dict[str, str] = batch.get("summaries", {})
        if not summaries_by_source:
            return

        batch["sent"] = True
        _judge_batch.pop(segment_id, None)
        _judge_completed_segments.add(segment_id)

    ordered_sources = sorted(summaries_by_source.keys())
    merged_parts = [
        summaries_by_source[source].strip()
        for source in ordered_sources
        if summaries_by_source[source].strip()
    ]
    merged_summary = " ".join(merged_parts)

    log_print(
        "INFO",
        "Flushing batched summaries",
        session_id=session_id,
        segment_id=segment_id,
        count=len(summaries_by_source),
    )
    expected_tts_count = int(batch.get("tts_expected_count", len(merged_parts)))
    if judge:
        judge.judge_summary(
            merged_summary,
            segment_id,
            expected_tts_count=max(0, expected_tts_count),
        )

    if reconstructor:
        sum0 = summaries_by_source.get("sum0", "").strip()
        sum1 = summaries_by_source.get("sum1", "").strip()
        context_before = _collect_context_before(segment_id)
        next_sentence = _consume_next_sentence(segment_id)
        reconstructor.reconstruct_conversation(
            context_before=context_before,
            sum0=sum0,
            sum1=sum1,
            next_sentence=next_sentence,
            segment_id=segment_id,
        )


def _make_summary_batch_callback(source_id: str, session_id: str, tts_enabled: bool):
    """Create callback that batches summary outputs before judge request."""

    def _callback(summary: str, segment_id: int) -> None:
        if not summary or not summary.strip():
            return

        flush_now = False
        with _judge_batch_lock:
            if segment_id in _judge_completed_segments:
                return
            expected = max(1, len(summary_clients))
            batch = _judge_batch.setdefault(
                segment_id,
                {
                    "summaries": {},
                    "expected": expected,
                    "tts_sources": set(),
                    "tts_expected_count": 0,
                    "sent": False,
                    "timer": None,
                },
            )
            batch["expected"] = expected
            batch["summaries"][source_id] = summary.strip()
            if tts_enabled:
                batch["tts_sources"].add(source_id)
                batch["tts_expected_count"] = len(batch["tts_sources"])

            # First arrival: start timeout to avoid waiting forever.
            if batch.get("timer") is None:
                timer = threading.Timer(
                    _JUDGE_BATCH_TIMEOUT_SEC,
                    _flush_judge_batch,
                    args=(segment_id, session_id),
                )
                timer.daemon = True
                batch["timer"] = timer
                timer.start()

            if len(batch["summaries"]) >= batch["expected"]:
                flush_now = True

        if flush_now:
            _flush_judge_batch(segment_id, session_id)

    return _callback


@app.route("/")
def index():
    """Serve the main page."""
    log_print("INFO", "Index page requested")
    return render_template("realtime.html")


@app.route("/api/devices")
def api_devices():
    """Return list of available audio input devices."""
    with _pyaudio_lock:
        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxInputChannels"],
                    }
                )

        pa.terminate()
    log_print("INFO", f"Found {len(devices)} input devices")
    return jsonify(devices)


@app.route("/api/output_devices")
def api_output_devices():
    """Return list of available audio output devices."""
    with _pyaudio_lock:
        pa = pyaudio.PyAudio()
        devices = []

        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if info["maxOutputChannels"] > 0:
                devices.append(
                    {
                        "index": i,
                        "name": info["name"],
                        "channels": info["maxOutputChannels"],
                    }
                )

        pa.terminate()
    log_print("INFO", f"Found {len(devices)} output devices")
    return jsonify(devices)


def _mic_monitor_loop(device_indices: list[int], select_ids: list[str]):
    """Background thread to monitor mic levels using PyAudio."""
    global _mic_monitor_running, _mic_monitor_streams

    CHUNK = 1024
    RATE = 16000
    FORMAT = pyaudio.paInt16

    # Build mapping: device_idx -> list of select_ids
    device_to_selects: dict[int, list[str]] = {}
    for device_idx, select_id in zip(device_indices, select_ids):
        if device_idx not in device_to_selects:
            device_to_selects[device_idx] = []
        device_to_selects[device_idx].append(select_id)

    # Open streams for unique devices only
    streams = []
    for device_idx in device_to_selects.keys():
        try:
            with _pyaudio_lock:
                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=FORMAT,
                    channels=1,
                    rate=RATE,
                    input=True,
                    input_device_index=device_idx,
                    frames_per_buffer=CHUNK,
                )
            streams.append((device_idx, stream, pa))
            log_print("INFO", f"Mic monitor opened for device {device_idx}")
        except Exception as e:
            log_print(
                "ERROR", f"Failed to open mic monitor for device {device_idx}: {e}"
            )

    while _mic_monitor_running and streams:
        levels = {}
        for device_idx, stream, pa in streams:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                # Calculate RMS level
                samples = struct.unpack(f"<{CHUNK}h", data)
                rms = (sum(s * s for s in samples) / CHUNK) ** 0.5
                # Normalize to 0-100 (max 32768 for 16-bit audio)
                level = min(100, int((rms / 8000) * 100))
                # Apply level to all select_ids that use this device
                for select_id in device_to_selects.get(device_idx, []):
                    levels[select_id] = level
            except Exception as e:
                log_print("ERROR", f"Mic monitor read error: {e}")

        if levels:
            sio.emit("mic_levels", levels)

        sio.sleep(0.05)  # 50ms interval

    # Cleanup
    for device_idx, stream, pa in streams:
        try:
            stream.stop_stream()
            stream.close()
            pa.terminate()
        except:
            pass

    log_print("INFO", "Mic monitor stopped")


def start_mic_monitor(device_indices: list[int], select_ids: list[str]):
    """Start monitoring mic levels for given device indices."""
    global _mic_monitor_running, _mic_monitor_thread

    stop_mic_monitor()

    if not device_indices:
        return

    # Small delay to ensure old thread has stopped
    time.sleep(0.1)

    _mic_monitor_running = True
    _mic_monitor_thread = sio.start_background_task(
        _mic_monitor_loop, device_indices, select_ids
    )
    log_print("INFO", f"Starting mic monitor for devices: {device_indices}")


def stop_mic_monitor():
    """Stop mic level monitoring."""
    global _mic_monitor_running, _mic_monitor_thread

    _mic_monitor_running = False
    if _mic_monitor_thread:
        _mic_monitor_thread = None


def _noise_gate_monitor_loop(device_indices: list[int], mic_ids: list[str]):
    """Background thread to monitor mic RMS for noise gate calibration."""
    global _noise_gate_monitor_running

    CHUNK = 1024
    RATE = 24000
    FORMAT = pyaudio.paInt16

    # Build mapping: device_idx -> mic_id
    device_to_mic: dict[int, str] = {}
    for device_idx, mic_id in zip(device_indices, mic_ids):
        device_to_mic[device_idx] = mic_id

    streams = []

    try:
        # Open streams for each unique device
        for device_idx in device_to_mic.keys():
            try:
                with _pyaudio_lock:
                    pa = pyaudio.PyAudio()
                    stream = pa.open(
                        format=FORMAT,
                        channels=1,
                        rate=RATE,
                        input=True,
                        input_device_index=device_idx,
                        frames_per_buffer=CHUNK,
                    )
                streams.append((device_idx, stream, pa))
                log_print("INFO", f"Noise gate monitor opened for device {device_idx}")
            except Exception as e:
                log_print(
                    "ERROR",
                    f"Failed to open noise gate monitor for device {device_idx}: {e}",
                )

        while _noise_gate_monitor_running and streams:
            levels = {}
            for device_idx, stream, pa in streams:
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    # Calculate RMS
                    audio_data = np.frombuffer(data, dtype=np.int16)
                    rms = float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))
                    mic_id = device_to_mic.get(device_idx)
                    if mic_id:
                        levels[mic_id] = rms
                except Exception as e:
                    log_print("ERROR", f"Noise gate monitor read error: {e}")

            if levels:
                sio.emit("noise_gate_levels", levels)

            sio.sleep(0.05)  # 50ms interval

    except Exception as e:
        log_print("ERROR", f"Noise gate monitor failed: {e}")
    finally:
        for device_idx, stream, pa in streams:
            try:
                stream.stop_stream()
                stream.close()
                pa.terminate()
            except:
                pass
        log_print("INFO", "Noise gate monitor stopped")


def start_noise_gate_monitor(device_indices: list[int], mic_ids: list[str]):
    """Start monitoring mic RMS for noise gate calibration."""
    global _noise_gate_monitor_running, _noise_gate_monitor_thread

    stop_noise_gate_monitor()
    time.sleep(0.1)

    _noise_gate_monitor_running = True
    _noise_gate_monitor_thread = sio.start_background_task(
        _noise_gate_monitor_loop, device_indices, mic_ids
    )
    log_print("INFO", f"Starting noise gate monitor for devices {device_indices}")


def stop_noise_gate_monitor():
    """Stop noise gate calibration monitoring."""
    global _noise_gate_monitor_running, _noise_gate_monitor_thread

    _noise_gate_monitor_running = False
    if _noise_gate_monitor_thread:
        _noise_gate_monitor_thread = None


@sio.on("start_noise_gate_monitor")
def handle_start_noise_gate_monitor(data: dict):
    """Start noise gate calibration monitoring."""
    session_id = request.sid
    device_indices = data.get("device_indices", [])
    mic_ids = data.get("mic_ids", [])
    if device_indices:
        log_print(
            "INFO",
            f"Start noise gate monitor: devices={device_indices}, mics={mic_ids}",
            session_id=session_id,
        )
        start_noise_gate_monitor(device_indices, mic_ids)


@sio.on("stop_noise_gate_monitor")
def handle_stop_noise_gate_monitor():
    """Stop noise gate calibration monitoring."""
    session_id = request.sid
    log_print("INFO", "Stop noise gate monitor", session_id=session_id)
    stop_noise_gate_monitor()


@sio.on("start_mic_monitor")
def handle_start_mic_monitor(data: dict):
    """Start mic level monitoring for selected devices."""
    session_id = request.sid
    device_indices = data.get("device_indices", [])
    select_ids = data.get("select_ids", [])
    log_print(
        "INFO",
        f"Start mic monitor: devices={device_indices}, ids={select_ids}",
        session_id=session_id,
    )
    start_mic_monitor(device_indices, select_ids)


@sio.on("stop_mic_monitor")
def handle_stop_mic_monitor():
    """Stop mic level monitoring."""
    session_id = request.sid
    log_print("INFO", "Stop mic monitor", session_id=session_id)
    stop_mic_monitor()


@sio.on("connect")
def handle_connect():
    """Handle client connection."""
    session_id = request.sid
    log_print("INFO", "Client connected", session_id=session_id)
    get_logger(session_id).log("client_connected")


@sio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    global \
        client, \
        summary_clients, \
        context_judge, \
        conversation_reconstructor, \
        transcript_reconstructor, \
        keyword_tts_client, \
        _active_runtime_sid
    session_id = request.sid
    log_print("INFO", "Client disconnected", session_id=session_id)
    logger = get_logger(session_id)
    logger.log("client_disconnected")

    # Stop mic monitors
    stop_mic_monitor()
    stop_noise_gate_monitor()

    # Ignore disconnects from non-active sockets; runtime is shared globals.
    if _active_runtime_sid and session_id != _active_runtime_sid:
        log_print(
            "INFO",
            "Ignoring disconnect from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    # Stop running clients when active user disconnects (e.g., page refresh)
    with _clients_lock:
        if client:
            client.stop()
            client = None
        for sc in summary_clients:
            sc.stop()
        summary_clients = []
        if context_judge:
            context_judge.stop()
            context_judge = None
        if conversation_reconstructor:
            conversation_reconstructor.stop()
            conversation_reconstructor = None
        if transcript_reconstructor:
            transcript_reconstructor.stop()
            transcript_reconstructor = None
        keyword_tts_client = None
        target_session_id = _active_runtime_sid or session_id
        _save_session_full_dialogue_json(
            session_id=target_session_id,
            reason="disconnect",
        )
        _active_runtime_sid = None
    _reset_tts_airpods_state("disconnect")
    with _judge_batch_lock:
        _judge_batch.clear()
        _judge_completed_segments.clear()
    _reset_segment_tracking()
    _reset_dialogue_stores()


@sio.on("start")
def handle_start(data: dict):
    """Start audio streaming and keyword extraction."""
    global \
        client, \
        summary_clients, \
        context_judge, \
        conversation_reconstructor, \
        transcript_reconstructor, \
        keyword_tts_client, \
        _active_runtime_sid, \
        _airpods_mode_switch_enabled, \
        _transcript_compression_mode, \
        _fast_catchup_threshold_sec_runtime
    session_id = request.sid
    log_print("INFO", "Start requested", session_id=session_id, data=data)

    # Stop mic monitor when session starts
    stop_mic_monitor()
    stop_noise_gate_monitor()
    with _airpods_lock:
        _airpods_mode_switch_enabled = bool(
            (data or {}).get("airpods_mode_switch_enabled", True)
        )
    _reset_tts_airpods_state("start")

    with _clients_lock:
        previous_session_id = _active_runtime_sid
        if client:
            log_print("INFO", "Stopping previous client", session_id=session_id)
            client.stop()
        for sc in summary_clients:
            sc.stop()
        summary_clients = []
        if context_judge:
            context_judge.stop()
            context_judge = None
        if conversation_reconstructor:
            conversation_reconstructor.stop()
            conversation_reconstructor = None
        if transcript_reconstructor:
            transcript_reconstructor.stop()
            transcript_reconstructor = None
        if previous_session_id:
            _save_session_full_dialogue_json(
                session_id=previous_session_id,
                reason="start_new_session",
            )
        with _judge_batch_lock:
            _judge_batch.clear()
            _judge_completed_segments.clear()
        _reset_segment_tracking()
        _reset_dialogue_stores()
        _active_runtime_sid = session_id

        source = data.get("source", "mic")

        # Get source selections for keyword and summary agents.
        keyword_mic = data.get("keyword_mic")  # int or None
        summary_mics = data.get("summary_mics", [])  # list of ints
        keyword_source = str(data.get("keyword_source", "") or "").strip() or None
        raw_summary_sources = data.get("summary_sources", [])
        summary_sources = [
            str(x).strip() for x in raw_summary_sources if str(x or "").strip()
        ]
        if not summary_sources:
            fallback_sum0 = str(
                data.get("summary0_source", "") or data.get("summary_0_source", "")
            ).strip()
            fallback_sum1 = str(
                data.get("summary1_source", "") or data.get("summary_1_source", "")
            ).strip()
            summary_sources = [x for x in [fallback_sum0, fallback_sum1] if x]
        voice_ids = data.get("voice_ids", [])  # list of voice IDs for each summary mic
        keyword_voice_id = data.get("keyword_voice_id")  # voice ID for keyword TTS
        tts_output_device = data.get("tts_output_device")  # output device index for TTS

        # Create keyword TTS client (uses default voice if not specified)
        keyword_tts_kwargs = {
            "socketio": sio,
            "session_id": f"{session_id}_keyword_tts",
            "output_device_index": tts_output_device,
        }
        if keyword_voice_id:
            keyword_tts_kwargs["voice_id"] = keyword_voice_id
        keyword_tts_client = TTSClient(**keyword_tts_kwargs)

        # Keep the field for backward compatibility, but summary-generation path is disabled.
        judge_enabled = data.get("judge_enabled", False)
        reconstructor_enabled = data.get("reconstructor_enabled", True)
        requested_mode = str(data.get("transcript_compression_mode", "realtime"))
        if requested_mode in ("fastest", "realtime", "api_mini", "api_nano"):
            _transcript_compression_mode = requested_mode
        else:
            _transcript_compression_mode = "realtime"
        try:
            _fast_catchup_threshold_sec_runtime = float(
                data.get(
                    "fast_catchup_threshold_sec",
                    _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC,
                )
                or _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
            )
        except Exception:
            _fast_catchup_threshold_sec_runtime = _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
        _fast_catchup_threshold_sec_runtime = max(
            1.0, min(30.0, _fast_catchup_threshold_sec_runtime)
        )

        # Noise gate settings
        noise_gate_data = data.get("noise_gate", {})
        enable_noise_gate = noise_gate_data.get("enabled", False)
        noise_gate_config = None
        if enable_noise_gate:
            from clients.audio_filter import NoiseGateConfig

            threshold = noise_gate_data.get("threshold", 500)
            # Calculate margin_multiplier based on threshold
            # We set noise_floor to threshold/2 and margin to 2.0 so threshold = noise_floor * 2
            noise_gate_config = NoiseGateConfig(
                min_threshold=threshold,  # Use threshold directly as min
                margin_multiplier=1.0,  # Direct threshold mode
            )
            log_print(
                "INFO",
                f"Noise gate enabled with threshold={threshold}",
                session_id=session_id,
            )

        client = RealtimeClient(
            sio,
            source,
            session_id,
            device_index=keyword_mic,
            mp3_file=keyword_source,
            enable_noise_gate=enable_noise_gate,
            noise_gate_config=noise_gate_config,
        )

        # If noise gate is enabled with fixed threshold, set it
        if enable_noise_gate and client.noise_gate:
            client.set_noise_threshold(noise_gate_data.get("threshold", 500))
        client.add_vad_transcript_callback(_on_vad_transcript)
        client.add_vad_boundary_callback(_on_vad_boundary)

        # Create one SummaryClient per summary source.
        summary_input_count = (
            len(summary_mics) if source == "mic" else len(summary_sources)
        )
        for i in range(summary_input_count):
            voice_id = voice_ids[i] if i < len(voice_ids) else None
            mic_kwargs = {}
            if source == "mic":
                mic_kwargs["device_indices"] = [summary_mics[i]]
            else:
                mic_kwargs["audio_file"] = summary_sources[i]
                mic_kwargs["noise_cut_threshold"] = (
                    int(noise_gate_data.get("threshold", 500))
                    if enable_noise_gate
                    else 0
                )
            sc = SummaryClient(
                sio,
                session_id=f"{session_id}_sum{i}",
                source=source,
                enable_tts=bool(voice_id),
                prepare_tts_on_callback=judge_enabled,
                mic_id=f"summary_{i}",
                voice_id=voice_id,
                output_device_index=tts_output_device,
                **mic_kwargs,
            )
            # Add VAD transcript callback for this speaker
            speaker_id = "A" if i == 0 else "B"
            sc.add_vad_transcript_callback(
                _make_vad_transcript_callback(
                    speaker_id=speaker_id,
                    session_id=session_id,
                    source_id=f"sum{i}",
                )
            )
            summary_clients.append(sc)
            sc.start()

        if summary_clients and reconstructor_enabled:
            # Create TranscriptReconstructorClient for VAD transcript compression.
            transcript_reconstructor = TranscriptReconstructorClient(
                sio,
                session_id=f"{session_id}_transcript_reconstructor",
                source=source,
            )
            transcript_reconstructor.start()
            log_print(
                "INFO",
                "TranscriptReconstructorClient created and connected",
                session_id=session_id,
            )

        # Summary callbacks are intentionally not connected in VAD-only mode.

        # Connect RealtimeClient to SummaryClients for transcript forwarding
        client.set_summary_clients(summary_clients)

        client.start()


@sio.on("stop")
def handle_stop():
    """Stop audio streaming."""
    global client
    session_id = request.sid
    log_print("INFO", "Stop requested", session_id=session_id)
    if not _is_active_session(session_id):
        log_print(
            "INFO",
            "Ignoring stop from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    with _clients_lock:
        if client:
            client.stop()
        target_session_id = _active_runtime_sid or session_id
        _save_session_full_dialogue_json(
            session_id=target_session_id,
            reason="stop",
        )
    _reset_tts_airpods_state("stop")


@sio.on("request")
def handle_request():
    """Handle manual keyword extraction request."""
    session_id = request.sid
    log_print("INFO", "Manual request triggered", session_id=session_id)
    if not _is_active_session(session_id):
        log_print(
            "INFO",
            "Ignoring request from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    with _clients_lock:
        if client and client.running:
            client.request()
        else:
            log_print(
                "WARN", "Request ignored - no running client", session_id=session_id
            )


@sio.on("start_listening")
def handle_start_listening():
    """Start a listening segment for later summarization."""
    global _segment_seq, _dialogue_segment_start_time
    global \
        _post_tts_followup_active, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts
    global _post_tts_followup_live_window_open
    session_id = request.sid
    if not _is_active_session(session_id):
        log_print(
            "INFO",
            "Ignoring start_listening from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    with _clients_lock:
        if summary_clients:
            # Set dialogue segment start time for VAD transcript tracking
            _dialogue_segment_start_time = time.time()

            with _segment_ctx_lock:
                _segment_seq += 1
                _segment_windows[_segment_seq] = {
                    "start_ts": time.time(),
                    "end_ts": None,
                    "next_sentence": "",
                }
                _post_tts_followup_active = False
                _post_tts_followup_inflight = False
                _post_tts_followup_cursor_ts = 0.0
                _post_tts_followup_live_window_open = False
            for sc in summary_clients:
                sc.start_listening()
            # Also notify context judge
            if context_judge:
                context_judge.start_listening()
            if conversation_reconstructor:
                conversation_reconstructor.start_listening()
            if transcript_reconstructor:
                transcript_reconstructor.start_listening()
            log_print(
                "INFO",
                "Start listening",
                session_id=session_id,
                clients=len(summary_clients),
            )
        else:
            log_print(
                "WARN",
                "start_listening ignored - no summary clients",
                session_id=session_id,
            )


@sio.on("end_listening")
def handle_end_listening():
    """End listening segment and request summary."""
    global \
        _post_tts_followup_active, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts
    global _post_tts_followup_live_window_open
    session_id = request.sid
    if not _is_active_session(session_id):
        log_print(
            "INFO",
            "Ignoring end_listening from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    with _clients_lock:
        if summary_clients:
            current_segment_duration_sec = 0.0
            with _segment_ctx_lock:
                current_segment_id = _segment_seq
                if _segment_seq > 0 and _segment_seq in _segment_windows:
                    ended_at = time.time()
                    started_at = float(
                        _segment_windows[_segment_seq].get("start_ts") or ended_at
                    )
                    _segment_windows[_segment_seq]["end_ts"] = ended_at
                    current_segment_duration_sec = max(0.0, ended_at - started_at)
                    _post_tts_followup_active = True
                    _post_tts_followup_inflight = False
                    _post_tts_followup_cursor_ts = ended_at
                    _post_tts_followup_live_window_open = True
            for sc in summary_clients:
                sc.end_listening()
            # Also notify context judge
            if context_judge:
                context_judge.end_listening()
            if conversation_reconstructor:
                conversation_reconstructor.end_listening()
            if transcript_reconstructor:
                transcript_reconstructor.end_listening()

            threshold_sec = float(_fast_catchup_threshold_sec_runtime)
            should_fast_catchup = (
                current_segment_duration_sec > 0
                and current_segment_duration_sec <= threshold_sec
            )
            if should_fast_catchup:
                with _segment_ctx_lock:
                    # Fast catch-up path should not trigger post-TTS follow-up compression.
                    _post_tts_followup_active = False
                    _post_tts_followup_inflight = False
                    _post_tts_followup_live_window_open = False

                def _run_fast_catchup_or_fallback():
                    ok = _try_fast_catchup_for_segment(current_segment_id, session_id)
                    if ok:
                        return
                    _trigger_parallel_compression_after_delay(
                        current_segment_id, session_id
                    )

                threading.Thread(
                    target=_run_fast_catchup_or_fallback,
                    daemon=True,
                ).start()
            else:
                threading.Thread(
                    target=_trigger_parallel_compression_after_delay,
                    args=(current_segment_id, session_id),
                    daemon=True,
                ).start()

            log_print(
                "INFO",
                "End listening, requesting summary",
                session_id=session_id,
                clients=len(summary_clients),
                segment_duration_sec=current_segment_duration_sec,
                fast_catchup=should_fast_catchup,
                threshold_sec=threshold_sec,
            )
        else:
            # No summary clients - signal completion immediately
            sio.emit(
                "summary_done", {"is_empty": True, "no_summary": True, "segment_id": 0}
            )
            log_print(
                "WARN",
                "end_listening ignored - no summary clients",
                session_id=session_id,
            )


@sio.on("request_missed_summary")
def handle_request_missed_summary(data: dict):
    """Summarize missed window from always-on transcript buffer.

    Expected payload:
    - miss_start_ts: epoch sec/ms
    - miss_end_ts: epoch sec/ms
    - padding_before_sec: optional, default 1.0
    - padding_after_sec: optional, default 0.0
    - fast_catchup_enabled: optional, default true
    - fast_catchup_threshold_sec: optional, default 10.0
    - fast_catchup_speed: optional, default 1.6
    - fast_catchup_gap_sec: optional, default 0.5
    - fast_catchup_silence_thresh_db: optional, default -45
    """
    global _segment_seq

    session_id = request.sid
    if not _is_active_session(session_id):
        log_print(
            "INFO",
            "Ignoring request_missed_summary from non-active session",
            session_id=session_id,
            active_session_id=_active_runtime_sid,
        )
        return

    payload = data or {}
    start_ts = _normalize_client_ts(payload.get("miss_start_ts"))
    end_ts = _normalize_client_ts(payload.get("miss_end_ts"))
    if start_ts <= 0 or end_ts <= 0:
        sio.emit(
            "missed_summary_error",
            {"error": "invalid_timestamps", "data": payload},
            to=session_id,
        )
        return

    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts

    pad_before = float(
        payload.get("padding_before_sec", _MISS_PADDING_BEFORE_SEC) or 0.0
    )
    pad_after = float(payload.get("padding_after_sec", 0.0) or 0.0)
    pad_before = max(0.0, min(5.0, pad_before))
    pad_after = max(0.0, min(3.0, pad_after))

    window_start = max(0.0, start_ts - pad_before)
    window_end = max(window_start, end_ts + pad_after)
    miss_duration_sec = max(0.0, float(end_ts - start_ts))
    dialogue, entries = _get_dialogue_by_time_window(window_start, window_end)
    before_context = _collect_context_before_start_ts(window_start)

    fast_catchup_enabled = bool(payload.get("fast_catchup_enabled", True))
    fast_catchup_threshold_sec = float(
        payload.get(
            "fast_catchup_threshold_sec", _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
        )
        or _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
    )
    fast_catchup_speed = float(
        payload.get("fast_catchup_speed", _FAST_CATCHUP_DEFAULT_SPEED)
        or _FAST_CATCHUP_DEFAULT_SPEED
    )
    fast_catchup_gap_sec = float(
        payload.get("fast_catchup_gap_sec", _FAST_CATCHUP_DEFAULT_GAP_SEC)
        or _FAST_CATCHUP_DEFAULT_GAP_SEC
    )
    fast_catchup_silence_thresh_db = float(
        payload.get(
            "fast_catchup_silence_thresh_db", _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB
        )
        or _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB
    )
    fast_catchup_threshold_sec = max(0.0, min(30.0, fast_catchup_threshold_sec))
    fast_catchup_speed = max(1.0, min(3.0, fast_catchup_speed))
    fast_catchup_gap_sec = max(0.1, min(2.0, fast_catchup_gap_sec))

    with _segment_ctx_lock:
        _segment_seq += 1
        segment_id = _segment_seq

    sio.emit(
        "missed_transcript_window",
        {
            "segment_id": segment_id,
            "miss_start_ts": start_ts,
            "miss_end_ts": end_ts,
            "window_start_ts": window_start,
            "window_end_ts": window_end,
            "padding_before_sec": pad_before,
            "padding_after_sec": pad_after,
            "entry_count": len(entries),
            "dialogue": dialogue,
        },
        to=session_id,
    )

    if not dialogue.strip():
        sio.emit(
            "missed_summary_empty",
            {
                "segment_id": segment_id,
                "window_start_ts": window_start,
                "window_end_ts": window_end,
                "entry_count": 0,
            },
            to=session_id,
        )
        return

    log_print(
        "INFO",
        "request_missed_summary accepted",
        session_id=session_id,
        segment_id=segment_id,
        miss_start_ts=start_ts,
        miss_end_ts=end_ts,
        window_start_ts=window_start,
        window_end_ts=window_end,
        miss_duration_sec=miss_duration_sec,
        entries=len(entries),
    )

    should_fast_catchup = (
        fast_catchup_enabled
        and miss_duration_sec > 0
        and miss_duration_sec <= fast_catchup_threshold_sec
    )
    if should_fast_catchup:
        log_print(
            "INFO",
            "Using fast catch-up for missed window",
            session_id=session_id,
            segment_id=segment_id,
            miss_duration_sec=miss_duration_sec,
            threshold_sec=fast_catchup_threshold_sec,
            speed=fast_catchup_speed,
            gap_sec=fast_catchup_gap_sec,
        )

        def _run_fast_catchup_with_fallback():
            output_sec = _synthesize_fast_catchup_dialogue(
                dialogue=dialogue,
                segment_id=segment_id,
                session_id=session_id,
                start_ts=window_start,
                end_ts=window_end,
                speed=fast_catchup_speed,
                gap_sec=fast_catchup_gap_sec,
                silence_thresh_db=fast_catchup_silence_thresh_db,
                trigger_source="missed_fast_catchup_source",
            )
            if output_sec > 0:
                return
            _trigger_parallel_compression_for_dialogue(
                dialogue=dialogue,
                segment_id=segment_id,
                session_id=session_id,
                before_context=before_context,
                trigger_source="missed_timestamp",
                window={
                    "miss_start_ts": start_ts,
                    "miss_end_ts": end_ts,
                    "window_start_ts": window_start,
                    "window_end_ts": window_end,
                    "padding_before_sec": pad_before,
                    "padding_after_sec": pad_after,
                },
                entries=entries,
            )

        threading.Thread(target=_run_fast_catchup_with_fallback, daemon=True).start()
        return

    threading.Thread(
        target=_trigger_parallel_compression_for_dialogue,
        kwargs={
            "dialogue": dialogue,
            "segment_id": segment_id,
            "session_id": session_id,
            "before_context": before_context,
            "trigger_source": "missed_timestamp",
            "window": {
                "miss_start_ts": start_ts,
                "miss_end_ts": end_ts,
                "window_start_ts": window_start,
                "window_end_ts": window_end,
                "padding_before_sec": pad_before,
                "padding_after_sec": pad_after,
            },
            "entries": entries,
        },
        daemon=True,
    ).start()


@sio.on("search_grounding")
def handle_grounding(data: dict):
    """Handle search grounding request (long-press)."""
    session_id = request.sid
    handle_search_grounding(sio, session_id, data)


@sio.on("keyword_tts")
def handle_keyword_tts(data: dict):
    """Synthesize keyword definition audio and play on server."""
    global _keyword_tts_request_token
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    payload = data or {}
    text = str(payload.get("text", "")).strip()
    keyword = str(payload.get("keyword", "")).strip()
    if not text:
        return
    if not keyword:
        # Fallback: derive keyword from "keyword. description" text payload.
        keyword = text.split(".", 1)[0].strip()

    with _clients_lock:
        tts = keyword_tts_client
        _keyword_tts_request_token += 1
        request_token = _keyword_tts_request_token
    if not tts:
        return

    # Latest-only behavior: if a newer navigation request arrives while synthesis
    # is in-flight, this request is dropped before playback.
    def _synthesize_and_play():
        def _watch_keyword_playback_done() -> None:
            # keyword path uses emit_done=False, so tts_done is not emitted.
            while tts.is_playing:
                time.sleep(0.05)
            with _clients_lock:
                if request_token != _keyword_tts_request_token:
                    return
                if client and client.running and keyword:
                    client.mark_keyword_tts_completed(keyword)
            sio.emit("keyword_tts_done", to=session_id)
            # Keep ANC hold active across keyword navigation/loading.

        audio_bytes = tts.synthesize(text, language="en")
        if not audio_bytes:
            return

        with _clients_lock:
            if request_token != _keyword_tts_request_token:
                return

        tts.stop_playback(wait=True, timeout_sec=1.2)
        tts.queue_audio_bytes(audio_bytes, text)
        # Switch ANC right before keyword playback starts (not at request time).
        _set_keyword_anc_hold(True, "keyword_tts_before_playback")
        # If previous playback is still tearing down after cancel, first call can miss.
        # Retry briefly so the latest navigation click starts audio without a second tap.
        if tts.play_queued(reason="keyword", emit_done=False):
            threading.Thread(target=_watch_keyword_playback_done, daemon=True).start()
            return
        for _ in range(30):  # up to ~300ms
            time.sleep(0.01)
            with _clients_lock:
                if request_token != _keyword_tts_request_token:
                    return
            if tts.play_queued(reason="keyword", emit_done=False):
                threading.Thread(
                    target=_watch_keyword_playback_done, daemon=True
                ).start()
                return

    threading.Thread(target=_synthesize_and_play, daemon=True).start()
    log_print("INFO", "keyword_tts requested", session_id=session_id, chars=len(text))


@sio.on("keyword_tts_preload")
def handle_keyword_tts_preload(data: dict):
    """Pre-synthesize keyword TTS and keep it in cache (no playback)."""
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    payload = data or {}
    texts = payload.get("texts", [])
    if not isinstance(texts, list):
        return

    # Normalize + dedupe while preserving order.
    unique_texts = []
    seen = set()
    for raw in texts:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        unique_texts.append(text)

    if not unique_texts:
        return

    with _clients_lock:
        tts = keyword_tts_client
    if not tts:
        return

    def _preload():
        for text in unique_texts:
            tts.synthesize(text, language="en")
        log_print(
            "INFO",
            "keyword_tts preloaded",
            session_id=session_id,
            count=len(unique_texts),
        )

    threading.Thread(target=_preload, daemon=True).start()


@sio.on("cancel_keyword_tts")
def handle_cancel_keyword_tts():
    """Cancel keyword TTS only (do not affect summary flow)."""
    global _keyword_tts_request_token
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    with _clients_lock:
        _keyword_tts_request_token += 1
        if keyword_tts_client:
            keyword_tts_client.stop_playback(wait=True, timeout_sec=1.2)
    log_print("INFO", "cancel_keyword_tts handled", session_id=session_id)


@sio.on("cancel_tts")
def handle_cancel_tts():
    """Cancel pending/playing TTS (both keyword and summary)."""
    global _keyword_tts_request_token
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    with _clients_lock:
        _keyword_tts_request_token += 1
        # Cancel keyword TTS
        if keyword_tts_client:
            keyword_tts_client.stop_playback(wait=True, timeout_sec=1.2)

        # Cancel summary TTS
        if context_judge:
            context_judge.cancel_tts(reason="doubleclick_cancel")
        else:
            # Fallback: clear any summary client queues directly
            for sc in summary_clients:
                if sc.tts_client:
                    sc.tts_client.stop_playback()
    _set_keyword_anc_hold(False, "cancel_tts")
    _reset_tts_airpods_state("cancel_tts")
    log_print("INFO", "cancel_tts handled", session_id=session_id)


@sio.on("browser_tts_playback_start")
def handle_browser_tts_playback_start(data: dict):
    """Browser-side TTS playback start (summary/reconstruction)."""
    session_id = request.sid
    if not _is_active_session(session_id):
        return {"ok": False, "active": False}
    source = str((data or {}).get("source", "")).strip() or "browser"
    _on_tts_started(f"browser_tts_playback_start:{source}")
    return {"ok": True}


@sio.on("browser_tts_playback_done")
def handle_browser_tts_playback_done(data: dict):
    """Browser-side TTS playback done (summary/reconstruction)."""
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    reason = str((data or {}).get("reason", "")).strip() or "browser_done"
    # Browser summary/reconstruction playback ended.
    _set_keyword_anc_hold(False, f"browser_tts_done:{reason}")
    _on_tts_finished(f"browser_tts_playback_done:{reason}")
    # Enforce transparency on summary completion even if counters drift.
    _set_airpods_mode(
        "transparency", f"browser_tts_playback_done_force:{reason}", wait=True
    )


@sio.on("browser_tts_near_end")
def handle_browser_tts_near_end(data: dict):
    """Browser-side summary/reconstruction TTS near-end signal (~2s before done)."""
    session_id = request.sid
    if not _is_active_session(session_id):
        return
    reason = str((data or {}).get("reason", "")).strip() or "summary_tts_near_end"
    global _post_tts_followup_live_window_open
    with _segment_ctx_lock:
        _post_tts_followup_live_window_open = False
    threading.Thread(
        target=_trigger_post_tts_followup_if_needed,
        args=(session_id, reason),
        daemon=True,
    ).start()


@sio.on("set_airpods_mode_switch")
def handle_set_airpods_mode_switch(data: dict):
    """Update AirPods ANC/Transparency auto-switch enable flag."""
    global _airpods_mode_switch_enabled
    enabled = bool((data or {}).get("enabled", True))
    with _airpods_lock:
        _airpods_mode_switch_enabled = enabled
    log_print(
        "INFO",
        "AirPods mode auto-switch updated",
        session_id=request.sid,
        enabled=enabled,
    )


if __name__ == "__main__":
    log_print("INFO", "=" * 50)
    log_print("INFO", "Starting web_realtime server")
    log_print("INFO", "=" * 50)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
    sio.run(app, host="0.0.0.0", port=5002, debug=False)
