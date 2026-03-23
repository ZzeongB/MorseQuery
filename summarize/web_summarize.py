"""Web interface for summarization with quiz questions.

Usage:
    python web_summarize.py

Features:
- Two-mic summarization (no keyword extraction)
- Quiz questions displayed at fixed times
- ANC control during summarization
- Audio recording only (no real-time keyword extraction)
"""

import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pyaudio
from flask import Flask, render_template
from flask_socketio import SocketIO

from config import LOG_DIR, TEMPLATES_DIR, STATIC_DIR, QUIZ_DIR
from logger import get_logger, get_session_subdir, log_print
from clients import SummaryClient, TranscriptSyncMode
from clients.streaming_tts_client import StreamingTTSClient, DEFAULT_VOICE_ID

app = Flask(__name__, template_folder=str(TEMPLATES_DIR), static_folder=str(STATIC_DIR))
sio = SocketIO(app, cors_allowed_origins="*")

# Global PyAudio lock
_pyaudio_lock = threading.Lock()

# AirPods ANC/Transparency control
_airpods_lock = threading.Lock()
_airpods_last_mode: Optional[str] = None
_airpods_mode_switch_enabled = True
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
    if _active_session_id:
        event_type = "anc_on" if mode == "anc" else "anc_off"
        get_logger(_active_session_id).log(event_type, reason=reason)
    _run_airpods_mode(mode, reason, wait=wait)


# Active clients
_clients_lock = threading.Lock()
summary_clients: list[SummaryClient] = []
_active_session_id: Optional[str] = None

# Quiz state
_quiz_lock = threading.Lock()
_quiz_questions: list[dict] = []
_quiz_current_index: int = 0
_quiz_timers: list[threading.Timer] = []
_quiz_started: bool = False
_quiz_start_time: float = 0.0

# Fixed quiz times after start
QUIZ_TIMES_MINUTES = [0.5, 5, 8]
QUIZ_TIME_LIMIT_SEC = 60  # 1 minute to answer each question

# Dialogue stores for VAD transcripts (one per summary speaker)
_dialogue_stores: dict[str, list] = {}  # key: "A" or "B"
_quiz_segment_started_at: dict[int, float] = {}
_tts_lock = threading.Lock()
_tts_client: Optional[StreamingTTSClient] = None


def _reset_dialogue_stores() -> None:
    """Reset all dialogue stores."""
    _dialogue_stores.clear()
    _quiz_segment_started_at.clear()


def _reset_tts_client() -> None:
    global _tts_client
    with _tts_lock:
        if _tts_client is not None:
            _tts_client.close()
        _tts_client = None


def _init_tts_client(session_id: str, voice_id: Optional[str]) -> None:
    global _tts_client
    _reset_tts_client()
    normalized_voice_id = (voice_id or "").strip()
    if not normalized_voice_id:
        return
    with _tts_lock:
        _tts_client = StreamingTTSClient(
            sio,
            session_id=session_id,
            voice_id=normalized_voice_id,
            emit_to=session_id,
        )


def _clip_text(text: str, limit: int = 180) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _build_segment_summary(start_ts: float, end_ts: float) -> str:
    summaries_by_source: dict[str, str] = {}
    for source_id, speaker in (("sum0", "A"), ("sum1", "B")):
        entries = [
            entry["text"]
            for entry in _dialogue_stores.get(speaker, [])
            if start_ts <= float(entry.get("timestamp", 0.0)) <= end_ts
        ]
        if not entries:
            continue
        # Match live's batch concept: one compact summary per source, then merge.
        summaries_by_source[source_id] = _clip_text(" ".join(entries[-3:]), 140)

    ordered = []
    for source_id, speaker in (("sum0", "A"), ("sum1", "B")):
        summary = summaries_by_source.get(source_id)
        if summary:
            ordered.append(f"Speaker {speaker}: {summary}")
    return " ".join(ordered).strip()


def _emit_summary_tts(session_id: str, question_index: int, text: str) -> None:
    normalized = (text or "").strip()
    if not normalized:
        return

    sio.emit(
        "summary_text",
        {"question_index": question_index, "text": normalized},
        to=session_id,
    )

    logger = get_logger(session_id)
    logger.log(
        "summary_text_generated",
        question_index=question_index,
        text=normalized,
    )

    with _tts_lock:
        client = _tts_client

    if client is None:
        return

    stream_id = f"summary_{question_index}_{int(time.time() * 1000)}"

    def _run() -> None:
        success = client.synthesize_streaming(normalized, stream_id=stream_id)
        if not success:
            sio.emit(
                "summary_error",
                {"error": "TTS streaming failed"},
                to=session_id,
            )

    threading.Thread(target=_run, daemon=True).start()


def _make_vad_transcript_callback(speaker_id: str, session_id: str, source_id: str):
    """Create a VAD transcript callback that stores entries."""

    def _callback(
        transcript: str,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
    ) -> None:
        if not transcript or not transcript.strip():
            return

        text = transcript.strip()
        timestamp = time.time()

        entry = {
            "timestamp": timestamp,
            "timestamp_iso_utc": datetime.fromtimestamp(
                timestamp, tz=timezone.utc
            ).isoformat(),
            "speaker": speaker_id,
            "source_id": source_id,
            "text": text,
            "start_time": start_ts,
            "end_time": end_ts,
        }

        if speaker_id not in _dialogue_stores:
            _dialogue_stores[speaker_id] = []

        _dialogue_stores[speaker_id].append(entry)

        # Emit transcript to frontend
        sio.emit("vad_transcript", entry, to=session_id)

        log_print(
            "INFO",
            f"VAD transcript ({speaker_id}): {text[:50]}...",
            session_id=session_id,
            source_id=source_id,
        )

    return _callback


def _load_quiz_from_file(filepath: str) -> list[dict]:
    """Load quiz questions from a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "questions" in data:
            return data["questions"]
        else:
            log_print("WARN", f"Invalid quiz format in {filepath}")
            return []
    except Exception as e:
        log_print("ERROR", f"Failed to load quiz file: {e}", filepath=filepath)
        return []


def _resolve_correct_answer(question: dict) -> tuple[object, object]:
    """Return normalized correct answer metadata for a question.

    Supported quiz formats:
    - correct_answer: option index (0-based)
    - correct_answer: option text
    - correct_option: option index (0-based)
    - correct_option_text: option text
    """
    options = question.get("options") or question.get("choices") or []

    if "correct_answer" in question:
        raw = question.get("correct_answer")
    elif "correct_option" in question:
        raw = question.get("correct_option")
    else:
        raw = question.get("correct_option_text")

    if raw is None:
        return None, None

    if isinstance(raw, int):
        correct_text = None
        if 0 <= raw < len(options):
            option = options[raw]
            correct_text = option.get("text") if isinstance(option, dict) else option
        return raw, correct_text

    if isinstance(raw, str):
        raw_text = raw.strip()
        for idx, option in enumerate(options):
            option_text = option.get("text") if isinstance(option, dict) else option
            if isinstance(option_text, str) and option_text.strip() == raw_text:
                return idx, option_text
        return None, raw_text

    return None, raw


def _schedule_all_quiz_questions(session_id: str) -> None:
    """Schedule all quiz questions at fixed times (2, 5, 8 minutes)."""
    global _quiz_timers

    with _quiz_lock:
        # Cancel any existing timers
        for timer in _quiz_timers:
            timer.cancel()
        _quiz_timers.clear()

        # Schedule each question at fixed times
        for i, time_minutes in enumerate(QUIZ_TIMES_MINUTES):
            if i >= len(_quiz_questions):
                break

            delay_sec = time_minutes * 60

            def show_question(question_idx=i):
                _show_quiz_question_by_index(session_id, question_idx)

            timer = threading.Timer(delay_sec, show_question)
            timer.daemon = True
            timer.start()
            _quiz_timers.append(timer)

            log_print(
                "INFO",
                f"Scheduled quiz question {i + 1} at {time_minutes} minutes",
                session_id=session_id,
            )


def _show_quiz_question_by_index(session_id: str, question_index: int) -> None:
    """Show a specific quiz question by index."""
    global _quiz_current_index

    with _quiz_lock:
        if not _quiz_questions or question_index >= len(_quiz_questions):
            log_print("INFO", "Quiz question index out of range", session_id=session_id, index=question_index)
            return

        question = _quiz_questions[question_index]
        _quiz_current_index = question_index
        _quiz_segment_started_at[question_index] = time.time()

    # Turn on ANC when showing question
    _set_airpods_mode("anc", "quiz_question_shown", wait=True)

    # Start listening on summary clients
    for sc in summary_clients:
        sc.start_listening()

    # Get quiz time for this question
    quiz_time_minutes = QUIZ_TIMES_MINUTES[question_index] if question_index < len(QUIZ_TIMES_MINUTES) else 0

    # Emit question to frontend with time limit
    sio.emit("quiz_question", {
        "index": question_index,
        "total": min(len(_quiz_questions), len(QUIZ_TIMES_MINUTES)),
        "question": question,
        "time_limit_sec": QUIZ_TIME_LIMIT_SEC,
        "quiz_time_minutes": quiz_time_minutes,
    }, to=session_id)

    logger = get_logger(session_id)
    logger.log("quiz_question_shown", index=question_index, question=question, quiz_time_minutes=quiz_time_minutes)

    log_print(
        "INFO",
        f"Quiz question {question_index + 1} shown at {quiz_time_minutes} minutes",
        session_id=session_id,
    )


def _start_quiz_timer(session_id: str) -> None:
    """Start the quiz timers for all questions at fixed times."""
    global _quiz_started, _quiz_start_time, _quiz_current_index

    with _quiz_lock:
        _quiz_started = True
        _quiz_start_time = time.time()
        _quiz_current_index = 0

    _schedule_all_quiz_questions(session_id)

    log_print(
        "INFO",
        f"Quiz timers started for times: {QUIZ_TIMES_MINUTES} minutes",
        session_id=session_id,
    )


def _stop_quiz_timer() -> None:
    """Stop all quiz timers."""
    global _quiz_timers, _quiz_started

    with _quiz_lock:
        for timer in _quiz_timers:
            timer.cancel()
        _quiz_timers.clear()
        _quiz_started = False


# -------------------------
# Routes
# -------------------------

@app.route("/")
def index():
    return render_template("summarize.html")


@app.route("/quiz/<filename>")
def get_quiz(filename):
    """Get quiz questions from a file."""
    filepath = QUIZ_DIR / filename
    if not filepath.exists():
        return {"error": "Quiz file not found"}, 404

    questions = _load_quiz_from_file(str(filepath))
    return {"questions": questions}


# -------------------------
# Socket.IO handlers
# -------------------------

@sio.on("connect")
def handle_connect():
    session_id = str(getattr(sio, "request", None) and getattr(sio.request, "sid", "default") or "default")
    log_print("INFO", "Client connected", session_id=session_id)
    sio.emit("connected", {"session_id": session_id})


@sio.on("disconnect")
def handle_disconnect():
    log_print("INFO", "Client disconnected")


@sio.on("get_devices")
def handle_get_devices():
    """Return list of available audio devices."""
    devices = []
    try:
        with _pyaudio_lock:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxInputChannels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": info.get("name", f"Device {i}"),
                        "channels": info.get("maxInputChannels", 0),
                    })
            pa.terminate()
    except Exception as e:
        log_print("ERROR", f"Failed to get devices: {e}")

    sio.emit("devices", {"devices": devices})


@sio.on("get_output_devices")
def handle_get_output_devices():
    """Return list of available output devices."""
    devices = []
    try:
        with _pyaudio_lock:
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if info.get("maxOutputChannels", 0) > 0:
                    devices.append({
                        "index": i,
                        "name": info.get("name", f"Device {i}"),
                        "channels": info.get("maxOutputChannels", 0),
                    })
            pa.terminate()
    except Exception as e:
        log_print("ERROR", f"Failed to get output devices: {e}")

    sio.emit("output_devices", {"devices": devices})


@sio.on("start")
def handle_start(data):
    """Start the summarization session with quiz questions."""
    global summary_clients, _active_session_id, _quiz_questions

    session_id = data.get("session_id", "default")
    source = data.get("source", "mic")
    summary_mics = data.get("summary_mics", [])  # List of device indices
    quiz_file = data.get("quiz_file")  # Optional quiz file path
    tts_voice_id = data.get("tts_voice_id") or DEFAULT_VOICE_ID
    noise_threshold = data.get("noise_threshold", 0)

    _active_session_id = session_id

    # Load quiz questions if provided
    if quiz_file:
        filepath = QUIZ_DIR / quiz_file if not Path(quiz_file).is_absolute() else Path(quiz_file)
        _quiz_questions = _load_quiz_from_file(str(filepath))
        log_print("INFO", f"Loaded {len(_quiz_questions)} quiz questions", filepath=str(filepath))

    logger = get_logger(session_id)
    logger.log(
        "session_start",
        source=source,
        summary_mics=summary_mics,
        quiz_file=quiz_file,
        tts_voice_id=tts_voice_id,
        quiz_times_minutes=QUIZ_TIMES_MINUTES,
        quiz_time_limit_sec=QUIZ_TIME_LIMIT_SEC,
        quiz_question_count=len(_quiz_questions),
    )

    # Reset dialogue stores
    _reset_dialogue_stores()
    _init_tts_client(session_id, tts_voice_id)

    # Create summary clients for each mic
    with _clients_lock:
        # Stop existing clients
        for sc in summary_clients:
            sc.stop()
        summary_clients.clear()

        speaker_ids = ["A", "B"]
        for i, mic_idx in enumerate(summary_mics[:2]):  # Max 2 mics
            speaker_id = speaker_ids[i] if i < len(speaker_ids) else f"Speaker{i}"
            source_id = f"sum{i}"
            mic_id = f"summary{i}"

            sc = SummaryClient(
                socketio=sio,
                session_id=f"{session_id}_{source_id}",
                source="mic",
                device_indices=[mic_idx],
                audio_file=None,
                noise_cut_threshold=noise_threshold,
                mic_id=mic_id,
            )

            # Add VAD transcript callback
            sc.add_vad_transcript_callback(
                _make_vad_transcript_callback(speaker_id, session_id, source_id)
            )

            summary_clients.append(sc)
            sc.start()

            log_print(
                "INFO",
                f"Started SummaryClient for {speaker_id}",
                session_id=session_id,
                mic_idx=mic_idx,
            )

    # Start quiz timer if questions are loaded
    if _quiz_questions:
        _start_quiz_timer(session_id)

    sio.emit("status", {"status": "started", "session_id": session_id})


@sio.on("stop")
def handle_stop():
    """Stop the summarization session."""
    global summary_clients, _active_session_id

    session_id = _active_session_id

    # Stop quiz timer
    _stop_quiz_timer()

    # Stop summary clients
    with _clients_lock:
        for sc in summary_clients:
            sc.stop()
        summary_clients.clear()
    _reset_tts_client()

    # Turn off ANC
    _set_airpods_mode("transparency", "session_stop")

    if session_id:
        logger = get_logger(session_id)
        logger.log("session_stop")

    _active_session_id = None

    log_print("INFO", "Session stopped", session_id=session_id)
    sio.emit("status", {"status": "stopped"})


@sio.on("quiz_answer")
def handle_quiz_answer(data):
    """Handle quiz answer submission (auto-submitted after time limit)."""
    session_id = _active_session_id
    if not session_id:
        return

    question_index = data.get("question_index", 0)
    answer = data.get("answer")  # Can be None if no answer selected
    timeout = data.get("timeout", False)  # True if time ran out
    answer_completed_at = time.time()

    # Check if answer is correct
    is_correct = None
    correct_answer = None
    correct_answer_text = None
    with _quiz_lock:
        if question_index < len(_quiz_questions):
            question = _quiz_questions[question_index]
            correct_answer, correct_answer_text = _resolve_correct_answer(question)
            if correct_answer is not None and answer is not None:
                is_correct = answer == correct_answer

    # End listening on summary clients
    for sc in summary_clients:
        sc.end_listening()

    # Turn off ANC
    _set_airpods_mode("transparency", "quiz_answered")

    logger = get_logger(session_id)
    logger.log(
        "quiz_answered",
        question_index=question_index,
        answer=answer,
        correct_answer=correct_answer,
        correct_answer_text=correct_answer_text,
        is_correct=is_correct,
        timeout=timeout,
    )

    start_ts = _quiz_segment_started_at.get(question_index, max(answer_completed_at - QUIZ_TIME_LIMIT_SEC, 0.0))
    summary_text = _build_segment_summary(start_ts, answer_completed_at)
    if not summary_text:
        summary_text = "No clear speech captured during this question."
    _emit_summary_tts(session_id, question_index, summary_text)

    log_print(
        "INFO",
        f"Quiz answer submitted for question {question_index}: answer={answer}, correct={is_correct}, timeout={timeout}",
        session_id=session_id,
    )

    # Check if all questions completed
    total_questions = min(len(_quiz_questions), len(QUIZ_TIMES_MINUTES))
    if question_index >= total_questions - 1:
        sio.emit("quiz_complete", {}, to=session_id)
        log_print("INFO", "All quiz questions completed", session_id=session_id)


@sio.on("show_next_question")
def handle_show_next_question():
    """Manually trigger the next quiz question."""
    session_id = _active_session_id
    if not session_id:
        return

    _show_quiz_question(session_id)


@sio.on("set_anc")
def handle_set_anc(data):
    """Manually set ANC mode."""
    mode = data.get("mode", "transparency")
    reason = data.get("reason", "manual")
    _set_airpods_mode(mode, reason, wait=True)


@sio.on("start_listening")
def handle_start_listening():
    """Manually start listening on summary clients."""
    _set_airpods_mode("anc", "manual_start_listening", wait=True)
    for sc in summary_clients:
        sc.start_listening()
    sio.emit("listening_status", {"listening": True})


@sio.on("end_listening")
def handle_end_listening():
    """Manually end listening on summary clients."""
    for sc in summary_clients:
        sc.end_listening()
    _set_airpods_mode("transparency", "manual_end_listening")
    sio.emit("listening_status", {"listening": False})


# -------------------------
# Main
# -------------------------

if __name__ == "__main__":
    log_print("INFO", "Starting Summarize web server on port 5003")
    sio.run(app, host="0.0.0.0", port=5003, debug=False, allow_unsafe_werkzeug=True)
