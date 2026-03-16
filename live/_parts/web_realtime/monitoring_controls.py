import base64
import io
import os
import re
import struct
import tempfile
import threading
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np
import pyaudio
from flask import jsonify, render_template, request
from pydub import AudioSegment, effects, silence
from pydub.utils import make_chunks

from clients.streaming_tts_client import StreamingTTSClient


def _trigger_post_tts_followup_if_needed(session_id: str, reason: str = "") -> None:
    """If new VAD arrived after end_listening, compress+TTS it as follow-up."""
    global \
        _segment_seq, \
        _post_tts_followup_inflight, \
        _post_tts_followup_cursor_ts, \
        _post_tts_followup_active
    global _post_tts_followup_live_window_open
    global _vad_then_commit_pending

    # VAD_THEN_COMMIT uses the follow-up path even when general summary follow-up is off.
    if not _summary_followup_enabled_runtime and not _vad_then_commit_pending:
        with _segment_ctx_lock:
            _post_tts_followup_active = False
            _post_tts_followup_inflight = False
            _post_tts_followup_live_window_open = False
        return

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

    # Log follow-up check to session log
    session_logger = get_logger(session_id)
    session_logger.log(
        "post_tts_followup_check",
        reason=reason or "browser_tts_done",
        start_ts=start_ts,
        end_ts=end_ts,
        window_sec=round(end_ts - start_ts, 3),
        entry_count=entry_count,
        has_new_transcript=entry_count > 0,
        entries=[
            {
                "speaker": e.get("speaker"),
                "text": e.get("text"),
                "timestamp": e.get("timestamp"),
            }
            for e in entries
        ] if entries else [],
    )

    with _segment_ctx_lock:
        if entries:
            last_ts = float(entries[-1].get("timestamp") or end_ts)
            _post_tts_followup_cursor_ts = max(last_ts + 0.001, end_ts)
        else:
            _post_tts_followup_active = False
            _post_tts_followup_inflight = False
            _post_tts_followup_cursor_ts = end_ts
            session_logger.log(
                "post_tts_followup_skipped",
                reason="no_new_transcript",
                start_ts=start_ts,
                end_ts=end_ts,
            )
            return

        _segment_seq += 1
        segment_id = _segment_seq
        _segment_windows[segment_id] = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "next_sentence": "",
            "fast_catchup_cursor_ts": start_ts,
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
            # Re-check if follow-up is still active (may have been cancelled by anc_off)
            with _segment_ctx_lock:
                if not _post_tts_followup_active:
                    log_print(
                        "INFO",
                        "Skipping post_tts_followup - cancelled before compression",
                        session_id=session_id,
                        segment_id=segment_id,
                    )
                    session_logger.log(
                        "post_tts_followup_cancelled",
                        segment_id=segment_id,
                        reason="cancelled_before_compression",
                    )
                    return
            # Log follow-up compression start
            session_logger.log(
                "post_tts_followup_start",
                segment_id=segment_id,
                entry_count=entry_count,
                dialogue=dialogue,
                start_ts=start_ts,
                end_ts=end_ts,
            )
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
) -> tuple[AudioSegment, str]:
    """Apply gap removal + speed-up to an AudioSegment."""
    if audio is None or len(audio) <= 0:
        return audio, "none_empty_audio"
    speed = max(1.0, min(3.0, float(speed or _FAST_CATCHUP_DEFAULT_SPEED)))
    gap_sec = max(0.0, min(2.0, float(gap_sec if gap_sec is not None else _FAST_CATCHUP_DEFAULT_GAP_SEC)))
    min_silence_len_ms = int(gap_sec * 1000.0)
    keep_silence_ms = 35
    speedup_method = "none"

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

    if gap_sec > 0:
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
        audio, speedup_method = _speedup_with_audiotsm(audio, speed)

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
    return audio, speedup_method


def _speedup_with_audiotsm(audio: AudioSegment, speed: float) -> tuple[AudioSegment, str]:
    """Time-scale audio using audiotsm WSOLA; fallback only if unavailable/error."""
    if audio is None or len(audio) <= 0:
        return audio, "none_empty_audio"
    speed = max(1.0, min(3.0, float(speed or _FAST_CATCHUP_DEFAULT_SPEED)))
    if speed <= 1.01:
        return audio, "none_speed<=1.01"

    if not _AUDIO_TSM_AVAILABLE:
        # Temporary fallback to keep runtime resilient until dependency is installed.
        try:
            return effects.speedup(
                audio, playback_speed=speed, chunk_size=120, crossfade=20
            ), "pydub_speedup_no_audiotsm"
        except Exception:
            return audio, "none_pydub_fallback_failed"

    try:
        with tempfile.TemporaryDirectory(prefix="fastcatchup_tsm_") as tmpdir:
            in_wav = os.path.join(tmpdir, "in.wav")
            out_wav = os.path.join(tmpdir, "out.wav")
            audio.export(in_wav, format="wav")
            with _AudioTSMWavReader(in_wav) as reader:
                with _AudioTSMWavWriter(
                    out_wav,
                    reader.channels,
                    reader.samplerate,
                ) as writer:
                    # WSOLA tends to preserve speech intelligibility for small speed-ups.
                    tsm = _audiotsm_wsola(
                        channels=reader.channels,
                        speed=speed,
                        frame_length=1024,
                    )
                    tsm.run(reader, writer)
            return AudioSegment.from_file(out_wav, format="wav"), "audiotsm_wsola"
    except Exception as e:
        log_print(
            "WARN",
            f"audiotsm speed-up failed, fallback to pydub: {e}",
        )
        try:
            return effects.speedup(
                audio, playback_speed=speed, chunk_size=120, crossfade=20
            ), "pydub_speedup_after_audiotsm_error"
        except Exception:
            return audio, "none_pydub_fallback_failed"


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
    processed_audio, speedup_method = _apply_fast_catchup_to_segment(
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

    method = (
        f"fast_catchup_source_{speed:.2f}x_nogap{gap_sec:.2f}s_{speedup_method}"
    )
    sio.emit(
        "compressed_dialogue_tts",
        {
            "segment_id": segment_id,
            "audio": base64.b64encode(processed_wav).decode("utf-8"),
            "text": str(dialogue or ""),
            "format": "wav",
            "sample_rate": 24000,
            "method": method,
            "speedup_method": speedup_method,
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
        speedup_method=speedup_method,
        gap_sec=gap_sec,
        silence_thresh_db=silence_thresh_db,
        saved_path=saved_path,
    )
    return max(0.0, len(processed_audio) / 1000.0)


def _synthesize_compressed_dialogue(
    compressed_text: str,
    segment_id: int,
    session_id: str,
    method: str = "unknown",
    trigger_source: str = "segment",
    skip_logging: bool = False,
) -> None:
    """Synthesize TTS for compressed dialogue and emit as merged audio.

    Args:
        compressed_text: The compressed dialogue text (A: ...\nB: ...)
        segment_id: The segment identifier
        method: The compression method used (for logging)
        skip_logging: If True, skip creating JSON log files for recon streams
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

    # Emit turns data for frontend text UI
    turns_payload = [
        {"speaker": speaker, "text": text}
        for speaker, text in turns
    ]
    sio.emit(
        "conversation_reconstructed_turns",
        {
            "segment_id": segment_id,
            "turns": turns_payload,
            "method": method,
            "trigger_source": trigger_source,
        },
        to=session_id,
    )

    for i, (speaker, text) in enumerate(turns):
        with _clients_lock:
            tts = _get_tts_client_for_speaker(speaker)
        if not tts:
            continue

        stream_id = f"recon_{session_id}_{segment_id}_{i}"
        stream_client = StreamingTTSClient(
            socketio=sio,
            session_id=f"{session_id}_recon_stream_{segment_id}_{i}",
            voice_id=tts.voice_id,
            model_id=tts.model_id,
            chunk_event="streaming_tts_chunk",
            done_event="streaming_tts_done",
            emit_to=session_id,
            event_extra={
                "stream_id": stream_id,
                "type": "reconstruction",
                "segment_id": segment_id,
                "method": method,
                "trigger_source": trigger_source,
                "turn_index": i,
                "speaker": speaker,
            },
            skip_logging=skip_logging,
        )
        started = stream_client.start_stream()
        pushed = started and stream_client.push_text(text)
        if started and pushed:
            stream_client.finish_input()
            # Emit tts_playing on first turn to trigger text UI
            if i == 0:
                sio.emit(
                    "tts_playing",
                    {"reason": "reconstruction", "segment_id": segment_id},
                    to=session_id,
                )
            while stream_client.is_streaming:
                time.sleep(0.02)
            stream_client.close()
        else:
            stream_client.close()
            log_print(
                "WARN",
                "Streaming synthesis failed for compressed turn",
                segment_id=segment_id,
                speaker=speaker,
                turn_index=i,
            )

    sio.emit(
        "conversation_tts_done",
        {
            "segment_id": segment_id,
            "count": len(turns),
            "method": method,
            "trigger_source": trigger_source,
        },
        to=session_id,
    )
    log_print(
        "INFO",
        "compressed_dialogue_tts streamed",
        segment_id=segment_id,
        method=method,
        turns=len(turns),
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
