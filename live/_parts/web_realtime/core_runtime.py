"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import os
import re
import subprocess
import sys
import threading
import time
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openai
import pyaudio

try:
    from audiotsm import wsola as _audiotsm_wsola
    from audiotsm.io.wav import WavReader as _AudioTSMWavReader
    from audiotsm.io.wav import WavWriter as _AudioTSMWavWriter
    _AUDIO_TSM_AVAILABLE = True
except Exception:
    _AUDIO_TSM_AVAILABLE = False
from clients import (
    ConversationReconstructorClient,
    DialogueStore,
    RealtimeClient,
    SummaryClient,
    TranscriptReconstructorClient,
)
from clients.context_judge_client import ContextJudgeClient
from clients.streaming_tts_client import StreamingTTSClient, DEFAULT_VOICE_ID
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
from flask import Flask
from flask_socketio import SocketIO
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
keyword_tts_stream_client: Optional[StreamingTTSClient] = None
_keyword_tts_request_token = 0
_active_runtime_sid: Optional[str] = None

# Dialogue stores for VAD transcripts (one per summary speaker)
_dialogue_stores: dict[str, DialogueStore] = {}  # key: "A" or "B"
_dialogue_segment_start_time: float = 0.0

# Diarization: track which mic is the primary speaker when both are active
_diarization_lock = threading.Lock()
_diarization_enabled = True  # Can be toggled via start event
_diarization_rms_ratio_threshold = 1.5  # Must be X times louder to win

_DIALOGUE_BUFFER_RETENTION_SEC = 15 * 60  # always-on rolling buffer
_MISS_PADDING_BEFORE_SEC = 1.0
_FAST_CATCHUP_DEFAULT_THRESHOLD_SEC = 1.0
_FAST_CATCHUP_DEFAULT_SPEED = 1.5
_FAST_CATCHUP_DEFAULT_GAP_SEC = 0.0
_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB = -45.0
_FAST_CATCHUP_WINDOW_MODE_DEFAULT = "vad_utterance"
_FAST_CATCHUP_TARGET_RMS_DBFS = -18.0
_FAST_CATCHUP_TARGET_PEAK_DBFS = -1.5
_FAST_CATCHUP_CHAIN_MAX_STEPS = 5
_FAST_CATCHUP_CHAIN_MIN_LAG_SEC = 2.0
_FAST_CATCHUP_EXTRA_GAIN_DB = 9.5
_FAST_CATCHUP_DENOISE_ENABLED = True
_FAST_CATCHUP_DENOISE_CHUNK_MS = 20
_FAST_CATCHUP_DENOISE_NOISE_ATTEN_DB = 12.0
_fast_catchup_threshold_sec_runtime = _FAST_CATCHUP_DEFAULT_THRESHOLD_SEC
_fast_catchup_speed_runtime = _FAST_CATCHUP_DEFAULT_SPEED
_fast_catchup_gap_sec_runtime = _FAST_CATCHUP_DEFAULT_GAP_SEC
_fast_catchup_window_mode_runtime = _FAST_CATCHUP_WINDOW_MODE_DEFAULT
_fast_catchup_chain_enabled_runtime = False
_summary_followup_enabled_runtime = False
_missed_summary_latency_bridge_enabled_runtime = False
_SEGMENT_TAIL_GRACE_SEC = 1.4
_SEGMENT_POST_END_WAIT_SEC = 0.0  # No delay - start summarization immediately
# To trigger summarization X seconds before keyword_tts ends, set this > 0
# (Currently keyword_tts duration is not predictable, so this is mainly for the
#  delay after end_listening before compression starts)

# Early summarization: trigger compression this many seconds before keyword_tts ends
# Note: This is approximate since streaming TTS duration is not known in advance
_KEYWORD_TTS_EARLY_SUMMARIZATION_SEC = 1.0
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
_recent_vad_utterances = deque(maxlen=200)  # (speech_started_ts, speech_stopped_ts)
_vad_open_speech_start_ts: Optional[float] = None
_before_context_lock = threading.Lock()
_before_context_summary = ""
_post_tts_followup_active = False
_post_tts_followup_inflight = False
_post_tts_followup_cursor_ts = 0.0
_post_tts_followup_live_window_open = False
_segment_compression_inflight: set[int] = set()
_pending_fast_catchup_segments: dict[int, str] = {}
_pending_fast_catchup_inflight: set[int] = set()
_pending_latency_bridge_by_session: dict[str, dict] = {}
_transcript_compression_mode = "realtime"


def _has_pending_fast_catchup(session_id: Optional[str] = None) -> bool:
    with _segment_ctx_lock:
        if not _pending_fast_catchup_segments:
            return False
        if not session_id:
            return True
        return any(sid == session_id for sid in _pending_fast_catchup_segments.values())


def _emit_fast_catchup_pending(session_id: str, pending: bool, segment_id: int) -> None:
    try:
        sio.emit(
            "fast_catchup_pending",
            {"pending": bool(pending), "segment_id": int(segment_id)},
            to=session_id,
        )
    except Exception:
        pass


def _reset_segment_tracking() -> None:
    global _segment_seq, _before_context_summary
    global _vad_open_speech_start_ts
    global \
        _post_tts_followup_active, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts
    global \
        _post_tts_followup_live_window_open, \
        _segment_compression_inflight, \
        _pending_fast_catchup_segments, \
        _pending_fast_catchup_inflight, \
        _pending_latency_bridge_by_session
    with _segment_ctx_lock:
        _segment_seq = 0
        _segment_windows.clear()
        _recent_vad_transcripts.clear()
        _recent_vad_boundaries.clear()
        _recent_vad_utterances.clear()
        _vad_open_speech_start_ts = None
        _post_tts_followup_active = False
        _post_tts_followup_inflight = False
        _post_tts_followup_cursor_ts = 0.0
        _post_tts_followup_live_window_open = False
        _segment_compression_inflight.clear()
        _pending_fast_catchup_segments.clear()
        _pending_fast_catchup_inflight.clear()
        _pending_latency_bridge_by_session.clear()
    with _before_context_lock:
        _before_context_summary = ""


def _on_vad_transcript(transcript: str) -> None:
    now = time.time()
    text = (transcript or "").strip()
    if not text:
        return

    # Save RealtimeClient VAD transcript to file
    if _active_runtime_sid:
        _append_session_transcript_entry(
            session_id=_active_runtime_sid,
            record={
                "type": "utterance",
                "captured_at": datetime.now().isoformat(),
                "timestamp": now,
                "timestamp_iso_utc": datetime.fromtimestamp(
                    now, tz=timezone.utc
                ).isoformat(),
                "speaker": "realtime",
                "source_id": "realtime",
                "text": text,
            },
            source_id="realtime",
        )

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
    global _vad_open_speech_start_ts
    ts = float((payload or {}).get("received_at_ts") or time.time())
    et = str(event_type or "").strip().lower()
    if et not in {"speech_started", "speech_stopped"}:
        return
    trigger_segments: list[tuple[int, str]] = []
    with _segment_ctx_lock:
        _recent_vad_boundaries.append((ts, et))
        if et == "speech_started":
            _vad_open_speech_start_ts = ts
        else:
            start_ts = _vad_open_speech_start_ts
            if start_ts is None:
                # Recover if start event was dropped: ignore invalid pair.
                return
            if ts > start_ts:
                _recent_vad_utterances.append((start_ts, ts))
            _vad_open_speech_start_ts = None
            if _pending_fast_catchup_segments:
                for (
                    segment_id,
                    pending_session_id,
                ) in _pending_fast_catchup_segments.items():
                    if segment_id in _pending_fast_catchup_inflight:
                        continue
                    _pending_fast_catchup_inflight.add(segment_id)
                    trigger_segments.append((segment_id, pending_session_id))

    for segment_id, pending_session_id in trigger_segments:
        threading.Thread(
            target=_run_pending_fast_catchup_segment,
            args=(segment_id, pending_session_id),
            daemon=True,
        ).start()


def _get_completed_vad_utterance_windows(
    *,
    segment_start_ts: float,
    segment_end_ts: float,
    min_start_ts: float,
) -> tuple[list[tuple[float, float]], bool]:
    """Get completed utterances overlapping segment range and pending-open status."""
    windows: list[tuple[float, float]] = []
    has_open_utterance = False
    with _segment_ctx_lock:
        utterances = list(_recent_vad_utterances)
        open_start = _vad_open_speech_start_ts

    for utt_start, utt_end in utterances:
        if utt_end <= min_start_ts:
            continue
        if utt_start >= segment_end_ts:
            continue
        if utt_end <= segment_start_ts:
            continue
        win_start = max(segment_start_ts, min_start_ts, utt_start)
        win_end = float(utt_end)
        if win_end > win_start:
            windows.append((win_start, win_end))

    if (
        open_start is not None
        and open_start < segment_end_ts
        and open_start >= min_start_ts
    ):
        has_open_utterance = True

    return windows, has_open_utterance


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


def _wait_for_pending_speech_transcript(
    max_wait_sec: float = 2.0,
    poll_interval_sec: float = 0.1,
    session_id: Optional[str] = None,
) -> bool:
    """Wait for pending speech (speech_started but not yet stopped) to complete.

    If there's an open speech, waits for:
    1. speech_stopped event
    2. Transcript to arrive

    Args:
        max_wait_sec: Maximum time to wait
        poll_interval_sec: Polling interval
        session_id: Session ID for logging

    Returns:
        True if we waited and transcript arrived, False if timeout or no pending
    """
    start_time = time.time()
    initial_open_ts = None

    with _segment_ctx_lock:
        initial_open_ts = _vad_open_speech_start_ts

    if initial_open_ts is None:
        # No pending speech
        return False

    log_print(
        "INFO",
        f"Waiting for pending speech transcript (started at {initial_open_ts:.2f})",
        session_id=session_id,
    )

    # Track initial transcript count to detect new arrivals
    initial_transcript_count = len(_recent_vad_transcripts)

    while (time.time() - start_time) < max_wait_sec:
        with _segment_ctx_lock:
            current_open_ts = _vad_open_speech_start_ts
            current_transcript_count = len(_recent_vad_transcripts)

        # Check if speech ended (open_ts changed to None or different)
        speech_ended = (current_open_ts is None) or (current_open_ts != initial_open_ts)

        # Check if new transcript arrived
        new_transcript_arrived = current_transcript_count > initial_transcript_count

        if speech_ended and new_transcript_arrived:
            elapsed_ms = (time.time() - start_time) * 1000
            log_print(
                "INFO",
                f"Pending speech transcript arrived (waited {elapsed_ms:.0f}ms)",
                session_id=session_id,
            )
            return True

        if speech_ended:
            # Speech ended but transcript not yet arrived, keep waiting briefly
            pass

        time.sleep(poll_interval_sec)

    elapsed_ms = (time.time() - start_time) * 1000
    log_print(
        "WARN",
        f"Timeout waiting for pending speech transcript ({elapsed_ms:.0f}ms)",
        session_id=session_id,
    )
    return False


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


def _get_last_n_turns_before(start_ts: float, n_turns: int = 3) -> str:
    """Get the last N dialogue turns before start_ts.

    Args:
        start_ts: Timestamp to search before
        n_turns: Number of turns to retrieve (default 3)

    Returns:
        Formatted dialogue string with speaker labels
    """
    all_entries = []
    for store in _dialogue_stores.values():
        # Get all entries before start_ts
        entries = store.get_entries_between(0.0, start_ts)
        all_entries.extend(entries)

    # Sort by timestamp descending, take last N
    all_entries.sort(key=lambda e: e.timestamp, reverse=True)
    last_n = all_entries[:n_turns]

    # Reverse back to chronological order
    last_n.reverse()

    if not last_n:
        return ""

    lines = [f"{e.speaker_id}: {e.text}" for e in last_n]
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


def _should_skip_transcript_diarization(source_id: str) -> bool:
    """Check if this transcript should be skipped due to diarization.

    When multiple mics detect speech simultaneously, only the louder one
    should produce transcripts. The quieter one is likely picking up
    crosstalk and should be ignored.

    This function uses speech segment RMS (average during the speech) and
    timestamps to detect overlapping speech windows, rather than checking
    instantaneous state which may have already changed by callback time.

    Args:
        source_id: Source identifier (e.g., "sum0", "sum1")

    Returns:
        True if this transcript should be skipped (quieter mic)
    """
    if not _diarization_enabled:
        return False

    with _clients_lock:
        clients = list(summary_clients)

    if len(clients) < 2:
        return False

    # Find this client and others
    this_client = None
    other_clients = []
    for sc in clients:
        sc_source_id = f"sum{clients.index(sc)}"
        if sc_source_id == source_id:
            this_client = sc
        else:
            other_clients.append(sc)

    if not this_client or not other_clients:
        return False

    # Get VAD states - use segment RMS and timestamps instead of instantaneous state
    this_state = this_client.get_vad_state()
    this_rms = float(this_state.get("last_speech_segment_rms", 0.0))
    this_stop_ts = this_state.get("speech_stop_ts")

    # If no valid RMS data, let it through
    if this_rms <= 0:
        return False

    now = time.time()
    # Window to consider speech as "concurrent" (seconds)
    overlap_window_sec = 3.0

    # Check if any other mic had concurrent speech with higher RMS
    for other in other_clients:
        other_state = other.get_vad_state()
        other_rms = float(other_state.get("last_speech_segment_rms", 0.0))
        other_stop_ts = other_state.get("speech_stop_ts")

        if other_rms <= 0:
            continue

        # Check if other mic's speech was recent/concurrent
        # Either currently speaking OR stopped within the overlap window
        is_other_speaking = other_state.get("is_speaking", False)
        is_other_recent = (
            other_stop_ts is not None
            and (now - float(other_stop_ts)) < overlap_window_sec
        )
        is_this_recent = (
            this_stop_ts is not None
            and (now - float(this_stop_ts)) < overlap_window_sec
        )

        # If both had recent speech, compare their segment RMS
        if (is_other_speaking or is_other_recent) and is_this_recent:
            ratio = other_rms / this_rms
            if ratio >= _diarization_rms_ratio_threshold:
                # Other mic was significantly louder - skip this transcript
                log_print(
                    "INFO",
                    f"Diarization: skipping {source_id} transcript (quieter mic)",
                    source_id=source_id,
                    this_rms=round(this_rms, 1),
                    other_rms=round(other_rms, 1),
                    ratio=round(ratio, 2),
                    this_stop_ts=this_stop_ts,
                    other_stop_ts=other_stop_ts,
                )
                return True

    return False


def _make_vad_transcript_callback(speaker_id: str, session_id: str, source_id: str):
    """Create a VAD transcript callback that adds entries to the DialogueStore.

    Uses RMS-based diarization to filter crosstalk when multiple mics are active.

    Args:
        speaker_id: Speaker identifier ("A" or "B")
        session_id: Session ID for logging
        source_id: Source identifier (e.g., "sum0", "sum1")

    Returns:
        Callback function taking (transcript: str, start_ts: Optional[float])
    """

    def _callback(transcript: str, start_ts: Optional[float] = None) -> None:
        if not transcript or not transcript.strip():
            return

        # Diarization check: skip if this is the quieter mic
        if _should_skip_transcript_diarization(source_id):
            return

        text = transcript.strip()

        # Add directly to DialogueStore
        if speaker_id not in _dialogue_stores:
            _dialogue_stores[speaker_id] = DialogueStore()

        store = _dialogue_stores[speaker_id]
        entry = store.add_entry(
            speaker_id=speaker_id,
            text=text,
            source_id=source_id,
        )

        # Build record with optional start_time
        record = {
            "type": "utterance",
            "captured_at": datetime.now().isoformat(),
            "timestamp": entry.timestamp,
            "timestamp_iso_utc": datetime.fromtimestamp(
                entry.timestamp, tz=timezone.utc
            ).isoformat(),
            "speaker": entry.speaker_id,
            "source_id": entry.source_id,
            "text": entry.text,
        }
        if start_ts is not None:
            record["start_time"] = start_ts
            record["start_time_iso_utc"] = datetime.fromtimestamp(
                start_ts, tz=timezone.utc
            ).isoformat()

        _append_session_transcript_entry(
            session_id=session_id,
            record=record,
            source_id=source_id,
        )
        prune_cutoff = time.time() - _DIALOGUE_BUFFER_RETENTION_SEC
        _ = store.prune_before(prune_cutoff)

        # Trigger follow-up compression if needed
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
