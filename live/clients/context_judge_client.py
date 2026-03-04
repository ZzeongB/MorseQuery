"""OpenAI Realtime API client for context-aware TTS judgment.

This client continuously streams audio to OpenAI Realtime API and judges
whether a summary TTS should be played based on:
1. Catch-up Value (Q1): Does the missed content have important information?
2. Interrupt Timing (Q3): Is this a good moment to interrupt?

TTS and judgment run in parallel for reduced latency:
- If TTS ready first: wait for judge, play if approved
- If judge approves first: play as soon as TTS ready
"""

import base64
import io
import json
import re
import threading
import time
import wave
from typing import Optional

import pyaudio
import websocket
from config import AUDIO_CHUNK, AUDIO_RATE, OPENAI_API_KEY, OPENAI_REALTIME_URL
from flask_socketio import SocketIO
from logger import get_logger, log_print

from .prompt import JUDGE_SESSION_INSTRUCTIONS, build_judgment_prompt
from .tts_client import TTSClient

MIN_SEGMENT_DURATION_FOR_JUDGE_SEC = 2


class ContextJudgeClient:
    """Realtime client that judges whether to play TTS based on audio context.

    Continuously streams audio to OpenAI Realtime API. When judge_summary() is called,
    commits current audio buffer and requests LLM judgment on whether to play TTS.

    TTS synthesis and judgment run in parallel. Playback occurs when both:
    1. TTS synthesis is complete (audio queued)
    2. Judge approves playback
    """

    def __init__(
        self,
        socketio: SocketIO,
        session_id: str = "default",
        device_indices: list[int] | None = None,
        tts_client: Optional[TTSClient] = None,
        tts_clients: list[TTSClient] | None = None,
    ):
        self.sio = socketio
        self.session_id = session_id
        self.device_indices = device_indices or []
        self.tts_clients = tts_clients or ([tts_client] if tts_client else [])

        self.ws: Optional[websocket.WebSocketApp] = None
        self.running = False
        self.connected = False

        self.response_buffer = ""
        self.logger = get_logger(session_id)

        # Audio streaming
        self.pa: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

        # Rolling buffer for last 3 seconds
        self.recent_audio_buffer: list[bytes] = []
        self.chunks_for_3_seconds = int(3 * AUDIO_RATE / AUDIO_CHUNK)

        # Listening state (synchronized with SummaryClient)
        self.listening = False
        self.segment_id = 0
        self._segment_start_times: dict[int, float] = {}
        self._segment_end_times: dict[int, float] = {}

        # Pending summary for judgment
        self.pending_summary: Optional[str] = None
        self.pending_segment_id: int = 0

        # Parallel TTS/judgment state
        self._state_lock = threading.Lock()
        self.tts_ready_count: int = 0  # Number of TTS clients that have finished
        self.tts_expected_count: int = len(self.tts_clients)  # Expected TTS ready callbacks
        self.judge_decided: bool = False  # Judge has made a decision
        self.judge_approved: bool = False  # Judge decision (True=play, False=skip)
        self.judge_reason: str = ""  # Judge reason

        # Thread synchronization
        self._lock = threading.Lock()
        self._shutdown_event = threading.Event()
        self._tts_play_lock = threading.Lock()
        self._tts_play_thread: Optional[threading.Thread] = None

        log_print(
            "INFO",
            "ContextJudgeClient created",
            session_id=session_id,
            devices=device_indices,
            tts_clients=len(self.tts_clients),
        )
        self.logger.log(
            "context_judge_created",
            devices=device_indices,
            tts_clients=len(self.tts_clients),
        )

    # -------------------------
    # Public APIs
    # -------------------------

    def start_listening(self) -> None:
        """Mark the start of a listening segment (synchronized with SummaryClient)."""
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "start_listening ignored (not connected)",
                session_id=self.session_id,
            )
            return

        self.segment_id += 1
        self.listening = True
        self.response_buffer = ""
        self._segment_start_times[self.segment_id] = time.time()

        # Reset parallel state for new segment
        with self._state_lock:
            self.tts_ready_count = 0
            self.tts_expected_count = len(self.tts_clients)
            self.judge_decided = False
            self.judge_approved = False
            self.judge_reason = ""

        # Clear buffer and re-append last 3 seconds
        with self._lock:
            ws = self.ws
            if ws:
                try:
                    ws.send(json.dumps({"type": "input_audio_buffer.clear"}))
                    # Re-append last 3 seconds of audio for context
                    for chunk in self.recent_audio_buffer:
                        audio_b64 = base64.b64encode(chunk).decode()
                        ws.send(
                            json.dumps(
                                {
                                    "type": "input_audio_buffer.append",
                                    "audio": audio_b64,
                                }
                            )
                        )
                except Exception:
                    pass

        self.logger.log("judge_start_listening", segment_id=self.segment_id)
        log_print(
            "INFO",
            f"ContextJudgeClient start_listening segment={self.segment_id}",
            session_id=self.session_id,
        )

    def end_listening(self) -> None:
        """Mark the end of a listening segment (synchronized with SummaryClient)."""
        if not self.running or not self.connected:
            return

        self.listening = False
        self._segment_end_times[self.segment_id] = time.time()
        self.logger.log("judge_end_listening", segment_id=self.segment_id)
        log_print(
            "INFO",
            f"ContextJudgeClient end_listening segment={self.segment_id}",
            session_id=self.session_id,
        )

    def judge_summary(self, summary: str, segment_id: int) -> None:
        """Start judgment for the given summary (runs in parallel with TTS).

        Called by SummaryClient after generating a summary.
        Commits current audio buffer and requests LLM judgment.

        Args:
            summary: The summary text to judge
            segment_id: The segment ID this summary belongs to
        """
        if not self.running or not self.connected:
            log_print(
                "WARN",
                "judge_summary ignored (not connected)",
                session_id=self.session_id,
            )
            return

        if not summary or not summary.strip():
            log_print(
                "INFO",
                "judge_summary ignored (empty summary)",
                session_id=self.session_id,
            )
            return

        # Guard against duplicate judge requests for the same in-flight segment.
        with self._state_lock:
            if (
                self.pending_summary is not None
                and not self.judge_decided
                and self.pending_segment_id == segment_id
            ):
                log_print(
                    "WARN",
                    "Duplicate judge_summary ignored (in-flight)",
                    session_id=self.session_id,
                    segment_id=segment_id,
                )
                self.logger.log(
                    "judge_request_ignored_duplicate",
                    segment_id=segment_id,
                )
                return

        # Reset state for new judgment
        with self._state_lock:
            self.tts_ready_count = 0
            self.tts_expected_count = len(self.tts_clients)
            self.judge_decided = False
            self.judge_approved = False
            self.judge_reason = ""

        self.pending_summary = summary
        self.pending_segment_id = segment_id
        self.response_buffer = ""
        segment_duration_sec = self._get_segment_duration_sec(segment_id)

        if (
            segment_duration_sec is not None
            and segment_duration_sec < MIN_SEGMENT_DURATION_FOR_JUDGE_SEC
        ):
            reason = (
                f"Segment too short ({segment_duration_sec:.2f}s < "
                f"{MIN_SEGMENT_DURATION_FOR_JUDGE_SEC:.2f}s)."
            )
            should_clear_tts = False
            with self._state_lock:
                self.judge_decided = True
                self.judge_approved = False
                self.judge_reason = reason
                # If any TTS already queued/ready, clear immediately
                should_clear_tts = self.tts_ready_count > 0

            # If TTS already queued/ready, clear immediately to prevent stale playback.
            if should_clear_tts:
                self._clear_tts()

            self.logger.log(
                "judge_auto_reject_short_segment",
                segment_id=segment_id,
                duration_sec=segment_duration_sec,
                min_duration_sec=MIN_SEGMENT_DURATION_FOR_JUDGE_SEC,
            )
            log_print(
                "INFO",
                "Auto-reject short segment before LLM judgment",
                session_id=self.session_id,
                segment_id=segment_id,
                duration_sec=segment_duration_sec,
                min_duration_sec=MIN_SEGMENT_DURATION_FOR_JUDGE_SEC,
            )
            self.sio.emit(
                "judge_rejected",
                {
                    "reason": reason,
                    "summary": summary,
                    "segment_id": segment_id,
                },
            )
            return

        log_print(
            "INFO",
            f"Judgment request: {summary[:60]}...",
            session_id=self.session_id,
            segment_id=segment_id,
            segment_duration_sec=segment_duration_sec,
        )
        self.logger.log(
            "judge_request",
            summary=summary,
            segment_id=segment_id,
            segment_duration_sec=segment_duration_sec,
        )

        # Build judgment prompt with summary text
        prompt = build_judgment_prompt(
            summary,
            segment_duration_sec=segment_duration_sec,
            min_segment_duration_sec=MIN_SEGMENT_DURATION_FOR_JUDGE_SEC,
        )

        with self._lock:
            ws = self.ws
            if not ws:
                log_print(
                    "WARN",
                    "judge_summary: no websocket",
                    session_id=self.session_id,
                )
                return

            try:
                # Commit current audio buffer
                ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

                # Request judgment from LLM
                ws.send(
                    json.dumps(
                        {
                            "type": "response.create",
                            "response": {
                                "modalities": ["text"],
                                "instructions": prompt,
                                "temperature": 0.6,
                                "max_output_tokens": 120,
                            },
                        }
                    )
                )
            except Exception as e:
                self.logger.log("judge_request_failed", error=str(e))
                log_print(
                    "ERROR",
                    f"judge_summary send failed: {e}",
                    session_id=self.session_id,
                )

    def _get_segment_duration_sec(self, segment_id: int) -> Optional[float]:
        """Return start~end duration for a segment if available."""
        start = self._segment_start_times.get(segment_id)
        if start is None:
            return None
        end = self._segment_end_times.get(segment_id, time.time())
        return max(0.0, end - start)

    def on_tts_ready(self, success: bool) -> None:
        """Called when TTS synthesis completes (callback from TTSClient).

        Waits for ALL TTS clients to be ready before playing.
        If judge already approved and all TTS ready, plays immediately.
        If judge already rejected, clears queue.
        If judge pending, waits for judge decision.

        Args:
            success: True if TTS was successfully queued
        """
        log_print(
            "INFO",
            f"TTS ready callback: success={success}",
            session_id=self.session_id,
        )
        self.logger.log("tts_ready", success=success)

        if not success:
            log_print(
                "WARN",
                "TTS synthesis failed, skipping playback",
                session_id=self.session_id,
            )
            return

        with self._state_lock:
            self.tts_ready_count += 1
            all_tts_ready = self.tts_ready_count >= self.tts_expected_count

            log_print(
                "INFO",
                f"TTS ready count: {self.tts_ready_count}/{self.tts_expected_count}",
                session_id=self.session_id,
            )

            if self.judge_decided:
                # Judge already made a decision
                if self.judge_approved:
                    if all_tts_ready:
                        log_print(
                            "INFO",
                            "All TTS ready + judge approved -> playing",
                            session_id=self.session_id,
                        )
                        self._play_tts(self.judge_reason)
                    else:
                        log_print(
                            "INFO",
                            f"TTS ready ({self.tts_ready_count}/{self.tts_expected_count}), waiting for remaining TTS",
                            session_id=self.session_id,
                        )
                else:
                    log_print(
                        "INFO",
                        "TTS ready + judge rejected -> clearing queue",
                        session_id=self.session_id,
                    )
                    self._clear_tts()
            else:
                # Judge still pending, TTS will wait
                log_print(
                    "INFO",
                    f"TTS ready ({self.tts_ready_count}/{self.tts_expected_count}), waiting for judge decision",
                    session_id=self.session_id,
                )

    # -------------------------
    # WebSocket handlers
    # -------------------------

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        """Handle WebSocket connection opened."""
        self.connected = True
        log_print(
            "INFO",
            "ContextJudgeClient WebSocket connected",
            session_id=self.session_id,
        )
        self.logger.log("judge_ws_connected")

        # Configure session for audio input with no VAD (we control timing)
        ws.send(
            json.dumps(
                {
                    "type": "session.update",
                    "session": {
                        "modalities": ["text", "audio"],
                        "input_audio_format": "pcm16",
                        "turn_detection": None,  # No VAD - we control timing
                        "instructions": JUDGE_SESSION_INSTRUCTIONS,
                    },
                }
            )
        )

        # Start audio streaming if devices are configured
        if self.device_indices:
            threading.Thread(target=self._stream_audio, daemon=True).start()

    def on_message(self, _ws: websocket.WebSocketApp, message: str) -> None:
        """Handle incoming WebSocket messages."""
        event = json.loads(message)
        etype = event.get("type", "")

        if etype == "session.created":
            session_info = event.get("session", {})
            self.logger.log(
                "judge_openai_session_created", session_id=session_info.get("id")
            )
            return

        if etype == "session.updated":
            self.logger.log("judge_openai_session_updated")
            return

        if etype == "response.text.delta":
            delta = event.get("delta", "")
            if delta:
                self.response_buffer += delta
            return

        if etype == "response.done":
            self._handle_judgment_response()
            return

        if etype == "error":
            err = event.get("error", {}).get("message", "Unknown error")
            log_print(
                "ERROR",
                f"ContextJudgeClient OpenAI error: {err}",
                session_id=self.session_id,
            )
            self.logger.log("judge_openai_error", error=err)
            return

    def _handle_judgment_response(self) -> None:
        """Parse LLM judgment response and decide whether to play TTS."""
        response = self.response_buffer.strip()
        self.response_buffer = ""

        log_print(
            "INFO",
            f"Judgment response: {response}",
            session_id=self.session_id,
        )
        self.logger.log(
            "judge_response",
            response=response,
            pending_summary=self.pending_summary,
            segment_id=self.pending_segment_id,
        )

        approved, reason, valid_format = self._parse_judgment_response(response)
        if not valid_format:
            log_print(
                "WARN",
                "Invalid judge output format, forcing NO",
                session_id=self.session_id,
                raw=response,
            )
            self.logger.log("judge_invalid_format", raw=response)

        with self._state_lock:
            self.judge_decided = True
            self.judge_approved = approved
            self.judge_reason = reason
            all_tts_ready = self.tts_ready_count >= self.tts_expected_count

            if approved:
                log_print(
                    "INFO",
                    f"Judgment APPROVED: {reason}",
                    session_id=self.session_id,
                )
                self.logger.log(
                    "judge_approved",
                    reason=reason,
                    segment_id=self.pending_segment_id,
                    tts_ready_count=self.tts_ready_count,
                    tts_expected_count=self.tts_expected_count,
                )

                if all_tts_ready:
                    # All TTS already ready, play immediately
                    log_print(
                        "INFO",
                        "Judge approved + all TTS ready -> playing",
                        session_id=self.session_id,
                    )
                    self._play_tts(reason)
                else:
                    # TTS not ready, will play when all on_tts_ready callbacks complete
                    log_print(
                        "INFO",
                        f"Judge approved, waiting for TTS ({self.tts_ready_count}/{self.tts_expected_count})",
                        session_id=self.session_id,
                    )

                self.sio.emit(
                    "judge_approved",
                    {
                        "reason": reason,
                        "summary": self.pending_summary,
                        "segment_id": self.pending_segment_id,
                    },
                )
            else:
                log_print(
                    "INFO",
                    f"Judgment REJECTED: {reason}",
                    session_id=self.session_id,
                )
                self.logger.log(
                    "judge_rejected",
                    reason=reason,
                    segment_id=self.pending_segment_id,
                    tts_ready_count=self.tts_ready_count,
                    tts_expected_count=self.tts_expected_count,
                )

                if self.tts_ready_count > 0:
                    # Some TTS already ready, clear queue
                    log_print(
                        "INFO",
                        "Judge rejected + TTS ready -> clearing queue",
                        session_id=self.session_id,
                    )
                    self._clear_tts()
                # If TTS not ready, it will be cleared when on_tts_ready is called

                self.sio.emit(
                    "judge_rejected",
                    {
                        "reason": reason,
                        "summary": self.pending_summary,
                        "segment_id": self.pending_segment_id,
                    },
                )

        # Clear pending state
        self.pending_summary = None
        self.pending_segment_id = 0

    def _parse_judgment_response(self, response: str) -> tuple[bool, str, bool]:
        """Parse judge response.

        Returns:
            (approved, reason, valid_format)
            valid_format=False means the output did not follow supported formats.
        """
        # JSON format fallback:
        # {"Q1":"YES","Q3":"YES","FINAL":"YES","REASON":"..."}
        try:
            parsed = json.loads(response)
            if isinstance(parsed, dict):
                q1_raw = str(parsed.get("Q1", "")).upper()
                q3_raw = str(parsed.get("Q2", parsed.get("Q3", ""))).upper()
                final_raw = str(parsed.get("FINAL", "")).upper()
                reason = str(parsed.get("REASON", "")).strip()

                if (
                    q1_raw in ("YES", "NO")
                    and q3_raw in ("YES", "NO")
                    and final_raw in ("YES", "NO")
                ):
                    q1 = q1_raw == "YES"
                    q3 = q3_raw == "YES"
                    final = final_raw == "YES"

                    # FINAL=YES when BOTH Q1 AND Q3 are YES
                    expected_final = q1 and q3
                    approved = expected_final
                    if final != expected_final:
                        reason = (
                            f"Inconsistent FINAL. Parsed reason: {reason}"
                            if reason
                            else "Inconsistent FINAL."
                        )
                    return approved, reason, True
        except Exception:
            pass

        # Strict format:
        # Q1=YES;Q2=YES;FINAL=NO;REASON=...
        strict_pattern = (
            r"Q1\s*=\s*(YES|NO)\s*;\s*"
            r"Q[23]\s*=\s*(YES|NO)\s*;\s*"
            r"FINAL\s*=\s*(YES|NO)\s*;\s*"
            r"REASON\s*=\s*(.+)$"
        )
        strict_match = re.match(strict_pattern, response, re.IGNORECASE)
        if strict_match:
            q1 = strict_match.group(1).upper() == "YES"
            q3 = strict_match.group(2).upper() == "YES"
            final = strict_match.group(3).upper() == "YES"
            reason = strict_match.group(4).strip()

            # FINAL=YES when BOTH Q1 AND Q3 are YES
            expected_final = q1 and q3
            approved = expected_final
            if final != expected_final:
                reason = (
                    f"Inconsistent FINAL. Parsed reason: {reason}"
                    if reason
                    else "Inconsistent FINAL."
                )
            return approved, reason, True

        # Tolerant key-value fallback for malformed JSON/kv responses.
        # Examples handled:
        # {"Q1":YES,"Q2":NO,"FINAL=YES,"REASON=..."}
        # Q1:YES, Q2:NO, FINAL:YES, REASON:...
        upper_response = response.upper()
        q1_match = re.search(r"\bQ1\b[^A-Z]*(YES|NO)\b", upper_response)
        q3_match = re.search(r"\bQ[23]\b[^A-Z]*(YES|NO)\b", upper_response)
        final_match = re.search(r"\bFINAL\b[^A-Z]*(YES|NO)\b", upper_response)

        reason = ""
        reason_match = re.search(
            r"\bREASON\b\s*[:=]\s*\"?(.+?)\"?\s*\}?$", response, re.IGNORECASE
        )
        if reason_match:
            reason = reason_match.group(1).strip()

        if q1_match and q3_match:
            q1 = q1_match.group(1) == "YES"
            q3 = q3_match.group(1) == "YES"
            # FINAL=YES when BOTH Q1 AND Q3 are YES
            expected_final = q1 and q3
            approved = expected_final

            if final_match:
                final = final_match.group(1) == "YES"
                if final != expected_final:
                    reason = (
                        f"Inconsistent FINAL. Parsed reason: {reason}"
                        if reason
                        else "Inconsistent FINAL."
                    )

            return approved, reason, True

        # Legacy fallback: "YES: reason" / "NO: reason"
        upper = response.upper()
        if upper.startswith("YES"):
            reason = response.split(":", 1)[1].strip() if ":" in response else ""
            return True, reason, True
        if upper.startswith("NO"):
            reason = response.split(":", 1)[1].strip() if ":" in response else ""
            return False, reason, True

        # Invalid output: fail closed (NO)
        return False, "Invalid judge output format", False

    def _play_tts(self, reason: str) -> None:
        """Play queued TTS audio.

        Args:
            reason: Reason for playing (for logging)
        """
        with self._tts_play_lock:
            if self._tts_play_thread and self._tts_play_thread.is_alive():
                return
            self._tts_play_thread = threading.Thread(
                target=self._play_tts_sequential,
                args=(reason,),
                daemon=True,
            )
            self._tts_play_thread.start()

    def _play_tts_sequential(self, reason: str) -> None:
        """Play queued TTS from multiple clients with no gap.

        Collects all audio from all TTS clients first, then plays through
        a single PyAudio stream to eliminate gaps between items.
        """
        # 1. Collect all audio from all TTS clients
        all_audio: list[tuple[bytes, str]] = []
        for client in self.tts_clients:
            with client._queue_lock:
                all_audio.extend(client.audio_queue)
                client.audio_queue.clear()

        if not all_audio:
            log_print(
                "DEBUG",
                "No audio collected for playback",
                session_id=self.session_id,
            )
            return

        log_print(
            "INFO",
            f"Playing {len(all_audio)} TTS items with single stream",
            session_id=self.session_id,
            reason=reason,
        )

        # 2. Play all audio through a single PyAudio stream (no gap)
        pa = None
        stream = None
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,  # TTS_SAMPLE_RATE
                output=True,
            )

            for audio_bytes, text in all_audio:
                if self._shutdown_event.is_set():
                    break
                pcm_data = self._extract_pcm_from_wav(audio_bytes)
                if pcm_data:
                    # Write in chunks so stop requests can interrupt
                    chunk_size = 4096
                    for i in range(0, len(pcm_data), chunk_size):
                        if self._shutdown_event.is_set():
                            break
                        stream.write(pcm_data[i : i + chunk_size])
                    log_print(
                        "DEBUG",
                        f"Played TTS: {text[:30]}...",
                        session_id=self.session_id,
                    )

        except Exception as e:
            log_print(
                "ERROR",
                f"PyAudio playback error: {e}",
                session_id=self.session_id,
            )
        finally:
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass

        # Emit tts_done once after ALL audio is done
        if not self._shutdown_event.is_set():
            self.sio.emit("tts_done")

    def _extract_pcm_from_wav(self, wav_bytes: bytes) -> Optional[bytes]:
        """Extract raw PCM data from WAV bytes."""
        try:
            with io.BytesIO(wav_bytes) as wav_io:
                with wave.open(wav_io, "rb") as wav_file:
                    return wav_file.readframes(wav_file.getnframes())
        except Exception as e:
            log_print(
                "ERROR",
                f"Failed to extract PCM from WAV: {e}",
                session_id=self.session_id,
            )
            return None

    def _clear_tts(self) -> None:
        """Clear queued TTS audio."""
        for client in self.tts_clients:
            client.stop_playback()

    def cancel_tts(self, reason: str = "user_cancel") -> None:
        """Cancel ongoing/pending TTS playback and queued audio."""
        with self._state_lock:
            self.judge_decided = True
            self.judge_approved = False
            self.judge_reason = reason

        for client in self.tts_clients:
            client.stop_playback()

        self.logger.log("judge_tts_canceled", reason=reason)
        log_print(
            "INFO",
            "Judge TTS canceled",
            session_id=self.session_id,
            reason=reason,
            tts_clients=len(self.tts_clients),
        )

    def on_error(self, _ws: websocket.WebSocketApp, error: Exception) -> None:
        """Handle WebSocket error."""
        log_print(
            "ERROR",
            f"ContextJudgeClient websocket error: {error}",
            session_id=self.session_id,
        )
        self.logger.log("judge_ws_error", error=str(error))

    def on_close(self, _ws: websocket.WebSocketApp, status: int, msg: str) -> None:
        """Handle WebSocket connection closed."""
        self.connected = False
        self.running = False
        log_print("INFO", "ContextJudgeClient closed", session_id=self.session_id)
        self.logger.log("judge_ws_closed", status=status, message=msg)

        with self._lock:
            self.ws = None

    # -------------------------
    # Audio streaming
    # -------------------------

    def _stream_audio(self) -> None:
        """Stream audio from configured microphone."""
        if not self.device_indices:
            log_print(
                "WARN",
                "ContextJudgeClient: no device configured",
                session_id=self.session_id,
            )
            return

        if not self.running:
            log_print(
                "WARN",
                "ContextJudgeClient: not running, skipping audio stream",
                session_id=self.session_id,
            )
            return

        device_idx = self.device_indices[0]
        pa = None
        stream = None

        try:
            pa = pyaudio.PyAudio()
            info = pa.get_device_info_by_index(device_idx)
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=AUDIO_CHUNK,
            )
            device_name = info["name"]
            log_print(
                "INFO",
                f"ContextJudgeClient opened mic {device_idx}: {device_name}",
                session_id=self.session_id,
            )
            self.logger.log("judge_audio_start", device=device_name)

            # Store references for cleanup
            self.pa = pa
            self.stream = stream

            commit_interval = 3.0
            last_commit = 0.0

            while self.running and self.connected and not self._shutdown_event.is_set():
                try:
                    data = stream.read(AUDIO_CHUNK, exception_on_overflow=False)
                except Exception as e:
                    if self._shutdown_event.is_set():
                        break
                    log_print(
                        "WARN",
                        f"ContextJudgeClient error reading device {device_idx}: {e}",
                        session_id=self.session_id,
                    )
                    continue

                # Maintain rolling buffer for last 3 seconds
                self.recent_audio_buffer.append(data)
                if len(self.recent_audio_buffer) > self.chunks_for_3_seconds:
                    self.recent_audio_buffer.pop(0)

                audio_b64 = base64.b64encode(data).decode()

                with self._lock:
                    ws = self.ws
                    if ws and not self._shutdown_event.is_set():
                        try:
                            ws.send(
                                json.dumps(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "audio": audio_b64,
                                    }
                                )
                            )

                            # Periodic commit (skip during listening to keep segment intact)
                            now = time.time()
                            if (
                                now - last_commit >= commit_interval
                                and not self.listening
                            ):
                                ws.send(
                                    json.dumps({"type": "input_audio_buffer.commit"})
                                )
                                last_commit = now
                        except Exception:
                            break

        except Exception as e:
            log_print(
                "ERROR",
                f"ContextJudgeClient failed to open device {device_idx}: {e}",
                session_id=self.session_id,
            )
        finally:
            # Clean up PyAudio resources safely
            if stream:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
            if pa:
                try:
                    pa.terminate()
                except Exception:
                    pass
            self.stream = None
            self.pa = None

            log_print(
                "INFO",
                "ContextJudgeClient audio streaming stopped",
                session_id=self.session_id,
            )

    # -------------------------
    # Lifecycle
    # -------------------------

    def start(self) -> None:
        """Start the context judge client."""
        self._shutdown_event.clear()
        self.running = True
        self.logger.log("context_judge_start")

        with self._lock:
            self.ws = websocket.WebSocketApp(
                OPENAI_REALTIME_URL,
                header=[
                    f"Authorization: Bearer {OPENAI_API_KEY}",
                    "OpenAI-Beta: realtime=v1",
                ],
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close,
            )
            ws = self.ws
        threading.Thread(target=ws.run_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop the context judge client."""
        self.running = False
        self.connected = False
        self.listening = False
        self._shutdown_event.set()

        self.logger.log("context_judge_stop")

        # Wait briefly for audio thread to exit cleanly
        time.sleep(0.15)

        with self._lock:
            ws = self.ws
            self.ws = None

        if ws:
            try:
                ws.close()
            except Exception:
                pass
