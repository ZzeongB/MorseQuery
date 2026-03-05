"""Web interface for OpenAI Realtime API with keyword extraction.

Usage:
    python web_realtime.py

Dependencies:
    pip install flask flask-socketio websocket-client pydub pyaudio
"""

import base64
import io
import os
import subprocess
import struct
import sys
import threading
import time
import wave
from collections import deque
from pathlib import Path
import re
from typing import Optional

import numpy as np
import pyaudio

from clients import ConversationReconstructorClient, RealtimeClient, SummaryClient
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
from config import TEMPLATES_DIR
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from handlers.grounding import handle_search_grounding
from logger import get_logger, log_print

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
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
    return _original_sio_emit(event, *args, **kwargs)


sio.emit = _emit_with_airpods  # type: ignore[assignment]

# Active clients (per session in production, global for simplicity here)
_clients_lock = threading.Lock()
client: Optional[RealtimeClient] = None
summary_clients: list[SummaryClient] = []  # One per summary mic
context_judge: Optional[ContextJudgeClient] = None  # Context-aware TTS judge
conversation_reconstructor: Optional[ConversationReconstructorClient] = None
keyword_tts_client: Optional[TTSClient] = None
_keyword_tts_request_token = 0
_active_runtime_sid: Optional[str] = None

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


def _reset_segment_tracking() -> None:
    global _segment_seq
    with _segment_ctx_lock:
        _segment_seq = 0
        _segment_windows.clear()
        _recent_vad_transcripts.clear()


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


def _collect_context_before(segment_id: int) -> str:
    with _segment_ctx_lock:
        window = _segment_windows.get(segment_id, {})
        start_ts = window.get("start_ts")
        if start_ts is None:
            return ""

        # Use the recent context (last ~25s before start, max 3 lines).
        candidates = [
            text
            for ts, text in _recent_vad_transcripts
            if ts < float(start_ts) and ts >= float(start_ts) - 25.0
        ]
    return " ".join(candidates[-3:]).strip()


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


def _parse_reconstructed_turns(text: str) -> list[tuple[str, str]]:
    """Parse reconstructed dialogue lines preserving order."""
    turns: list[tuple[str, str]] = []
    normalized = str(text or "")
    # Handle escaped newlines from model output logs/serialization.
    normalized = normalized.replace("\\n", "\n").replace("\r\n", "\n").replace("\r", "\n")

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
    """Emit reconstructed turns and synthesize per-turn TTS in sequence."""
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

    def _merge_wav_items(audio_items: list[tuple[bytes, str, str, int]]) -> Optional[bytes]:
        """Concatenate multiple mono 16-bit 24kHz WAV bytes into one WAV."""
        if not audio_items:
            return None
        pcm_chunks: list[bytes] = []
        for audio_bytes, _, _, _ in audio_items:
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
                                "Unexpected reconstruction TTS WAV format; skipping merge",
                                segment_id=segment_id,
                                channels=wf.getnchannels(),
                                sampwidth=wf.getsampwidth(),
                                framerate=wf.getframerate(),
                            )
                            return None
                        pcm_chunks.append(wf.readframes(wf.getnframes()))
            except Exception as e:
                log_print(
                    "ERROR",
                    f"Failed to parse reconstruction TTS WAV for merge: {e}",
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

    def _synthesize_turns():
        audio_items: list[tuple[bytes, str, str, int]] = []  # (audio_bytes, speaker, text, idx)
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
                "No audio synthesized for reconstruction",
                segment_id=segment_id,
            )
            sio.emit(
                "conversation_tts_done",
                {
                    "segment_id": segment_id,
                    "count": 0,
                },
            )
            return

        # Emit only after all per-turn TTS synthesis completes.
        sio.emit(
            "conversation_tts_done",
            {
                "segment_id": segment_id,
                "count": len(audio_items),
            },
        )

        merged_wav = _merge_wav_items(audio_items)
        if merged_wav:
            sio.emit(
                "conversation_tts_merged",
                {
                    "segment_id": segment_id,
                    "audio": base64.b64encode(merged_wav).decode("utf-8"),
                    "format": "wav",
                    "sample_rate": 24000,
                    "count": len(audio_items),
                },
            )
            log_print(
                "INFO",
                f"conversation_tts_merged emitted: count={len(audio_items)}, bytes={len(merged_wav)}",
                segment_id=segment_id,
            )
            return

        # Fallback: emit per-turn audio if merge fails.
        log_print(
            "INFO",
            f"Merge unavailable; emitting {len(audio_items)} conversation_tts events",
            segment_id=segment_id,
        )
        for audio_bytes, speaker, text, i in audio_items:
            audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
            sio.emit(
                "conversation_tts",
                {
                    "segment_id": segment_id,
                    "turn_index": i,
                    "speaker": speaker,
                    "text": text,
                    "audio": audio_b64,
                    "format": "wav",
                    "sample_rate": 24000,
                },
            )
            log_print(
                "INFO",
                f"conversation_tts emitted: turn={i}, speaker={speaker}, audio_bytes={len(audio_bytes)}",
                segment_id=segment_id,
            )

    threading.Thread(target=_synthesize_turns, daemon=True).start()


def _is_active_session(session_id: str) -> bool:
    return _active_runtime_sid is None or session_id == _active_runtime_sid


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
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxInputChannels"],
                })

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
                devices.append({
                    "index": i,
                    "name": info["name"],
                    "channels": info["maxOutputChannels"],
                })

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
            log_print("ERROR", f"Failed to open mic monitor for device {device_idx}: {e}")

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
                log_print("ERROR", f"Failed to open noise gate monitor for device {device_idx}: {e}")

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
        log_print("INFO", f"Start noise gate monitor: devices={device_indices}, mics={mic_ids}", session_id=session_id)
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
    log_print("INFO", f"Start mic monitor: devices={device_indices}, ids={select_ids}", session_id=session_id)
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
    global client, summary_clients, context_judge, conversation_reconstructor, keyword_tts_client, _active_runtime_sid
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
        keyword_tts_client = None
        _active_runtime_sid = None
    _reset_tts_airpods_state("disconnect")
    with _judge_batch_lock:
        _judge_batch.clear()
        _judge_completed_segments.clear()
    _reset_segment_tracking()


@sio.on("start")
def handle_start(data: dict):
    """Start audio streaming and keyword extraction."""
    global client, summary_clients, context_judge, conversation_reconstructor, keyword_tts_client, _active_runtime_sid, _airpods_mode_switch_enabled
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
        with _judge_batch_lock:
            _judge_batch.clear()
            _judge_completed_segments.clear()
        _reset_segment_tracking()
        _active_runtime_sid = session_id

        source = data.get("source", "mic")

        # Get source selections for keyword and summary agents.
        keyword_mic = data.get("keyword_mic")  # int or None
        summary_mics = data.get("summary_mics", [])  # list of ints
        keyword_source = str(data.get("keyword_source", "") or "").strip() or None
        raw_summary_sources = data.get("summary_sources", [])
        summary_sources = [
            str(x).strip()
            for x in raw_summary_sources
            if str(x or "").strip()
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

        # Judge agent settings (when disabled, summaries play without judgment)
        judge_enabled = data.get("judge_enabled", False)
        reconstructor_enabled = data.get("reconstructor_enabled", True)

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
                margin_multiplier=1.0,    # Direct threshold mode
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

        # Create one SummaryClient per summary source.
        summary_input_count = len(summary_mics) if source == "mic" else len(summary_sources)
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
            summary_clients.append(sc)
            sc.start()

        # Create ContextJudgeClient if we have summary clients with TTS
        # Uses the first summary mic for audio context
        # When judge_enabled=False, summaries play directly without judgment
        if source == "mic" and summary_mics and summary_clients and judge_enabled:
            judge_tts_clients = [
                sc.tts_client for sc in summary_clients if sc.tts_client is not None
            ]
            context_judge = ContextJudgeClient(
                sio,
                session_id=f"{session_id}_judge",
                device_indices=[summary_mics[0]],
                tts_clients=judge_tts_clients,
            )
            context_judge.start()

            log_print(
                "INFO",
                "ContextJudgeClient created and connected",
                session_id=session_id,
            )
        elif summary_clients:
            log_print(
                "INFO",
                "Judge agent disabled - summaries will play directly",
                session_id=session_id,
            )

        if summary_clients and reconstructor_enabled:
            reconstructor_kwargs = {}
            if source == "mic" and summary_mics:
                reconstructor_kwargs["device_indices"] = [summary_mics[0]]
            elif source == "mp3":
                if keyword_source:
                    reconstructor_kwargs["audio_file"] = keyword_source
                elif summary_sources:
                    reconstructor_kwargs["audio_file"] = summary_sources[0]

            conversation_reconstructor = ConversationReconstructorClient(
                sio,
                session_id=f"{session_id}_reconstructor",
                source=source,
                on_reconstruction_callback=_on_reconstruction_result,
                **reconstructor_kwargs,
            )
            conversation_reconstructor.start()
            log_print(
                "INFO",
                "ConversationReconstructorClient created and connected",
                session_id=session_id,
            )

        # Connect summary callbacks to aggregated batcher when any agent is enabled.
        if summary_clients and (judge_enabled or reconstructor_enabled):
            for i, sc in enumerate(summary_clients):
                sc.set_summary_callback(
                    _make_summary_batch_callback(
                        source_id=f"sum{i}",
                        session_id=session_id,
                        tts_enabled=sc.tts_client is not None,
                    )
                )
                if context_judge:
                    sc.set_tts_ready_callback(context_judge.on_tts_ready)

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
            log_print("WARN", "Request ignored - no running client", session_id=session_id)


@sio.on("start_listening")
def handle_start_listening():
    """Start a listening segment for later summarization."""
    global _segment_seq
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
            with _segment_ctx_lock:
                _segment_seq += 1
                _segment_windows[_segment_seq] = {
                    "start_ts": time.time(),
                    "end_ts": None,
                    "next_sentence": "",
                }
            for sc in summary_clients:
                sc.start_listening()
            # Also notify context judge
            if context_judge:
                context_judge.start_listening()
            if conversation_reconstructor:
                conversation_reconstructor.start_listening()
            log_print("INFO", "Start listening", session_id=session_id, clients=len(summary_clients))
        else:
            log_print(
                "WARN",
                "start_listening ignored - no summary clients",
                session_id=session_id,
            )


@sio.on("end_listening")
def handle_end_listening():
    """End listening segment and request summary."""
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
            with _segment_ctx_lock:
                if _segment_seq > 0 and _segment_seq in _segment_windows:
                    _segment_windows[_segment_seq]["end_ts"] = time.time()
            for sc in summary_clients:
                sc.end_listening()
            # Also notify context judge
            if context_judge:
                context_judge.end_listening()
            if conversation_reconstructor:
                conversation_reconstructor.end_listening()
            log_print("INFO", "End listening, requesting summary", session_id=session_id, clients=len(summary_clients))
        else:
            # No summary clients - signal completion immediately
            sio.emit("summary_done", {"is_empty": True, "no_summary": True, "segment_id": 0})
            log_print(
                "WARN",
                "end_listening ignored - no summary clients",
                session_id=session_id,
            )


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
    text = str((data or {}).get("text", "")).strip()
    if not text:
        return

    with _clients_lock:
        tts = keyword_tts_client
        _keyword_tts_request_token += 1
        request_token = _keyword_tts_request_token
    if not tts:
        return
    _set_keyword_anc_hold(True, "keyword_tts_requested")

    # Latest-only behavior: if a newer navigation request arrives while synthesis
    # is in-flight, this request is dropped before playback.
    def _synthesize_and_play():
        def _watch_keyword_playback_done() -> None:
            # keyword path uses emit_done=False, so tts_done is not emitted.
            while tts.is_playing:
                time.sleep(0.05)
            # Keep ANC hold active across keyword navigation/loading.

        audio_bytes = tts.synthesize(text, language="en")
        if not audio_bytes:
            return

        with _clients_lock:
            if request_token != _keyword_tts_request_token:
                return

        tts.stop_playback(wait=True, timeout_sec=1.2)
        tts.queue_audio_bytes(audio_bytes, text)
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
                threading.Thread(target=_watch_keyword_playback_done, daemon=True).start()
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
