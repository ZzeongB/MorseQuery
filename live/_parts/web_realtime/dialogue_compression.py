from clients.prompt import (
    DIALOGUE_COMPRESSION_SYSTEM_PROMPT,
    build_dialogue_compression_user_prompt,
)


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
                        "content": DIALOGUE_COMPRESSION_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": build_dialogue_compression_user_prompt(
                            dialogue=dialogue,
                            before_context=before_context,
                            keyword_context=keyword_context,
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


def _emit_missed_summary_latency_bridge(
    *,
    segment_id: int,
    session_id: str,
    window: dict | None,
) -> bool:
    """After summary TTS synthesis, fast-catch up post-summary latency audio."""
    payload = window or {}
    if not bool(payload.get("missed_summary_latency_bridge_enabled", False)):
        log_print(
            "INFO",
            "Latency bridge skipped: disabled",
            session_id=session_id,
            segment_id=segment_id,
            trigger_source=payload.get("trigger_source", ""),
        )
        return False

    bridge_decision_ts = float(payload.get("bridge_decision_ts") or 0.0)
    summary_started_ts = float(payload.get("summary_started_ts") or 0.0)
    window_end_ts = float(payload.get("window_end_ts") or 0.0)
    segment_end_ts = float(payload.get("end_ts") or 0.0)
    base_end_ts = max(window_end_ts, segment_end_ts)
    bridge_start_ts = max(bridge_decision_ts, summary_started_ts, base_end_ts)
    if bridge_start_ts <= 0:
        log_print(
            "INFO",
            "Latency bridge skipped: invalid start",
            session_id=session_id,
            segment_id=segment_id,
            summary_started_ts=summary_started_ts,
            bridge_decision_ts=bridge_decision_ts,
            window_end_ts=window_end_ts,
            segment_end_ts=segment_end_ts,
        )
        return False

    bridge_end_ts = time.time()
    lag_sec = max(0.0, bridge_end_ts - bridge_start_ts)
    bridge_window_sec = max(0.0, bridge_end_ts - bridge_start_ts)
    if lag_sec < 0.2:
        log_print(
            "INFO",
            "Latency bridge skipped: tiny lag",
            session_id=session_id,
            segment_id=segment_id,
            lag_sec=lag_sec,
            summary_started_ts=summary_started_ts,
            bridge_decision_ts=bridge_decision_ts,
            window_end_ts=window_end_ts,
            segment_end_ts=segment_end_ts,
            bridge_start_ts=bridge_start_ts,
            bridge_end_ts=bridge_end_ts,
            bridge_window_sec=bridge_window_sec,
        )
        return False

    speed = max(
        1.0,
        min(
            3.0,
            float(
                payload.get("fast_catchup_speed", _fast_catchup_speed_runtime)
                or _fast_catchup_speed_runtime
            ),
        ),
    )
    gap_sec = max(
        0.0,
        min(
            2.0,
            float(
                payload.get("fast_catchup_gap_sec", _fast_catchup_gap_sec_runtime)
                or _fast_catchup_gap_sec_runtime
            ),
        ),
    )
    silence_thresh_db = float(
        payload.get(
            "fast_catchup_silence_thresh_db", _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB
        )
        or _FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB
    )
    utterance_windows, has_open_utterance = _get_completed_vad_utterance_windows(
        segment_start_ts=bridge_start_ts,
        segment_end_ts=bridge_end_ts,
        min_start_ts=bridge_start_ts,
    )
    if len(utterance_windows) <= 0 and has_open_utterance:
        wait_for_stop_sec = max(
            0.0,
            min(
                3.0,
                float(payload.get("bridge_wait_for_speech_stop_sec", 1.8) or 1.8),
            ),
        )
        waited_end_ts = _snap_end_to_vad_stop(
            bridge_end_ts,
            wait_sec=wait_for_stop_sec,
            max_lookback_sec=8.0,
        )
        if waited_end_ts > bridge_end_ts:
            bridge_end_ts = waited_end_ts
            lag_sec = max(0.0, bridge_end_ts - bridge_start_ts)
            bridge_window_sec = max(0.0, bridge_end_ts - bridge_start_ts)
            utterance_windows, has_open_utterance = (
                _get_completed_vad_utterance_windows(
                    segment_start_ts=bridge_start_ts,
                    segment_end_ts=bridge_end_ts,
                    min_start_ts=bridge_start_ts,
                )
            )
            log_print(
                "INFO",
                "Latency bridge waited for speech_stopped",
                session_id=session_id,
                segment_id=segment_id,
                wait_for_stop_sec=wait_for_stop_sec,
                bridge_end_ts=bridge_end_ts,
                speech_done_count=len(utterance_windows),
                has_open_utterance=has_open_utterance,
            )

    speech_done_count = len(utterance_windows)
    if speech_done_count <= 0:
        log_print(
            "INFO",
            "Latency bridge skipped: no speech_done window",
            session_id=session_id,
            segment_id=segment_id,
            summary_started_ts=summary_started_ts,
            bridge_decision_ts=bridge_decision_ts,
            window_end_ts=window_end_ts,
            segment_end_ts=segment_end_ts,
            bridge_start_ts=bridge_start_ts,
            bridge_end_ts=bridge_end_ts,
            bridge_window_sec=bridge_window_sec,
            lag_sec=lag_sec,
            speech_done_count=speech_done_count,
            has_open_utterance=has_open_utterance,
        )
        return False

    emitted_count = 0
    output_sec_total = 0.0
    for idx, (utt_start, utt_end) in enumerate(utterance_windows, start=1):
        utt_dialogue, _utt_entries = _get_dialogue_by_time_window(utt_start, utt_end)
        output_sec = _synthesize_fast_catchup_dialogue(
            dialogue=utt_dialogue,
            segment_id=segment_id,
            session_id=session_id,
            start_ts=utt_start,
            end_ts=utt_end,
            speed=speed,
            gap_sec=gap_sec,
            silence_thresh_db=silence_thresh_db,
            trigger_source=f"missed_summary_latency_bridge_utt{idx}",
        )
        if output_sec <= 0:
            continue
        emitted_count += 1
        output_sec_total += output_sec

    if emitted_count <= 0:
        log_print(
            "INFO",
            "Latency bridge skipped: no audio from speech_done windows",
            session_id=session_id,
            segment_id=segment_id,
            summary_started_ts=summary_started_ts,
            bridge_decision_ts=bridge_decision_ts,
            window_end_ts=window_end_ts,
            segment_end_ts=segment_end_ts,
            bridge_start_ts=bridge_start_ts,
            bridge_end_ts=bridge_end_ts,
            bridge_window_sec=bridge_window_sec,
            lag_sec=lag_sec,
            speech_done_count=speech_done_count,
            has_open_utterance=has_open_utterance,
        )
        return False

    log_print(
        "INFO",
        "Missed-summary latency bridge emitted",
        session_id=session_id,
        segment_id=segment_id,
        bridge_lag_sec=lag_sec,
        summary_started_ts=summary_started_ts,
        bridge_decision_ts=bridge_decision_ts,
        window_end_ts=window_end_ts,
        segment_end_ts=segment_end_ts,
        bridge_start_ts=bridge_start_ts,
        bridge_end_ts=bridge_end_ts,
        bridge_window_sec=bridge_window_sec,
        speech_done_count=speech_done_count,
        emitted_count=emitted_count,
        output_sec=output_sec_total,
        has_open_utterance=has_open_utterance,
        speed=speed,
        gap_sec=gap_sec,
    )
    return True


def _register_pending_latency_bridge(
    *,
    session_id: str,
    segment_id: int,
    trigger_source: str,
    window: dict | None,
) -> None:
    payload = dict(window or {})
    if not bool(payload.get("missed_summary_latency_bridge_enabled", False)):
        return
    if trigger_source not in ("segment", "missed_timestamp"):
        return
    with _segment_ctx_lock:
        _pending_latency_bridge_by_session[session_id] = {
            "segment_id": int(segment_id),
            "trigger_source": trigger_source,
            "window": payload,
            "registered_at_ts": time.time(),
        }
    log_print(
        "INFO",
        "Latency bridge armed",
        session_id=session_id,
        segment_id=segment_id,
        trigger_source=trigger_source,
        bridge_decision_ts=payload.get("bridge_decision_ts"),
        window_end_ts=payload.get("window_end_ts"),
    )


def _has_pending_latency_bridge(session_id: str) -> bool:
    with _segment_ctx_lock:
        return session_id in _pending_latency_bridge_by_session


def _emit_pending_latency_bridge_if_needed(session_id: str, reason: str = "") -> bool:
    with _segment_ctx_lock:
        pending = _pending_latency_bridge_by_session.pop(session_id, None)
    if not pending:
        return False
    try:
        window = dict(pending.get("window") or {})
        window["trigger_source"] = pending.get("trigger_source", "")
        return _emit_missed_summary_latency_bridge(
            segment_id=int(pending.get("segment_id") or 0),
            session_id=session_id,
            window=window,
        )
    except Exception as e:
        log_print(
            "WARN",
            f"Latency bridge emit failed: {e}",
            session_id=session_id,
            segment_id=int(pending.get("segment_id") or 0),
            reason=reason or "browser_tts_done",
        )
        return False


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

    # Wait for any pending speech transcript before starting compression
    _wait_for_pending_speech_transcript(
        max_wait_sec=2.0,
        poll_interval_sec=0.1,
        session_id=session_id,
    )

    # Build enhanced before context with last 3 turns
    start_ts = float((window or {}).get("start_ts") or time.time())
    last_3_turns = _get_last_n_turns_before(start_ts, n_turns=2)
    if last_3_turns:
        if before_context:
            before_context = f"{before_context}\n\n[Recent transcript before this segment]\n{last_3_turns}"
        else:
            before_context = f"[Recent transcript before this segment]\n{last_3_turns}"

    summary_started_ts = time.time()
    window_payload = dict(window or {})
    if summary_started_ts > 0:
        window_payload["summary_started_ts"] = summary_started_ts
    window_payload.setdefault("bridge_decision_ts", summary_started_ts)
    window_payload.setdefault(
        "missed_summary_latency_bridge_enabled",
        bool(_missed_summary_latency_bridge_enabled_runtime),
    )
    window_payload.setdefault("fast_catchup_speed", float(_fast_catchup_speed_runtime))
    window_payload.setdefault(
        "fast_catchup_gap_sec", float(_fast_catchup_gap_sec_runtime)
    )
    window_payload.setdefault(
        "fast_catchup_silence_thresh_db",
        float(_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB),
    )
    window_payload["trigger_source"] = trigger_source
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
        window=window_payload,
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
        args=("gpt-4.1-nano", "api_nano", ["gpt-4.1-nano"]),
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
                "summary_started_ts": summary_started_ts,
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
                session_id,
                chosen_method,
                trigger_source=trigger_source,
            )
            _register_pending_latency_bridge(
                session_id=session_id,
                segment_id=segment_id,
                trigger_source=trigger_source,
                window=window_payload,
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
        bridge_decision_ts = float(segment_window.get("bridge_decision_ts") or 0.0)
        if bridge_decision_ts > 0:
            window_payload["bridge_decision_ts"] = bridge_decision_ts

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
    _clear_fast_catchup_pending_for_segment(segment_id, session_id)
    if delay_sec > 0:
        time.sleep(delay_sec)
    _trigger_parallel_compression(segment_id, session_id)


def _run_pending_fast_catchup_segment(segment_id: int, session_id: str) -> None:
    """Retry fast-catchup for pending segment when a new VAD stop arrives."""
    try:
        ok = _try_fast_catchup_for_segment(segment_id, session_id)
        if ok:
            return
        _clear_fast_catchup_pending_for_segment(segment_id, session_id)
        _trigger_parallel_compression_after_delay(segment_id, session_id)
    finally:
        with _segment_ctx_lock:
            _pending_fast_catchup_inflight.discard(segment_id)


def _clear_fast_catchup_pending_for_segment(segment_id: int, session_id: str) -> None:
    """Clear pending fast-catchup state when switching to compression fallback."""
    should_emit_clear = False
    with _segment_ctx_lock:
        _pending_fast_catchup_inflight.discard(segment_id)
        removed = _pending_fast_catchup_segments.pop(segment_id, None) is not None
        session_has_pending = any(
            sid == session_id for sid in _pending_fast_catchup_segments.values()
        )
        should_emit_clear = removed and (not session_has_pending)
    if should_emit_clear:
        _emit_fast_catchup_pending(session_id, False, segment_id)


def _try_fast_catchup_for_segment(segment_id: int, session_id: str) -> bool:
    """Try source-audio fast catch-up using completed VAD utterance windows."""
    with _segment_ctx_lock:
        segment_window = dict(_segment_windows.get(segment_id, {}))
    start_ts = float(segment_window.get("start_ts") or 0.0)
    end_ts = float(segment_window.get("end_ts") or 0.0)
    cursor_ts = float(segment_window.get("fast_catchup_cursor_ts") or start_ts)
    if start_ts <= 0:
        return False
    if end_ts <= start_ts:
        end_ts = time.time()

    window_mode = str(
        _fast_catchup_window_mode_runtime or _FAST_CATCHUP_WINDOW_MODE_DEFAULT
    )
    if window_mode == "time_window":
        window_start = max(start_ts, cursor_ts)
        window_end = max(window_start, end_ts)
        lag_sec = max(0.0, window_end - window_start)
        if lag_sec <= 0.05:
            return False

        dialogue, _entries = _get_dialogue_by_time_window(window_start, window_end)
        output_sec = _synthesize_fast_catchup_dialogue(
            dialogue=dialogue,
            segment_id=segment_id,
            session_id=session_id,
            start_ts=window_start,
            end_ts=window_end,
            speed=float(_fast_catchup_speed_runtime),
            gap_sec=float(_fast_catchup_gap_sec_runtime),
            silence_thresh_db=_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB,
            trigger_source="segment_fast_catchup_time_window",
        )
        emitted_any = output_sec > 0
        should_emit_clear = False
        with _segment_ctx_lock:
            if segment_id in _segment_windows:
                _segment_windows[segment_id]["fast_catchup_cursor_ts"] = window_end
            cleared_pending = (
                _pending_fast_catchup_segments.pop(segment_id, None) is not None
            )
            session_has_pending = any(
                sid == session_id for sid in _pending_fast_catchup_segments.values()
            )
            should_emit_clear = cleared_pending and (not session_has_pending)
        if should_emit_clear:
            _emit_fast_catchup_pending(session_id, False, segment_id)
        return emitted_any

    utterance_windows, has_open_utterance = _get_completed_vad_utterance_windows(
        segment_start_ts=start_ts,
        segment_end_ts=end_ts,
        min_start_ts=cursor_ts,
    )

    if not utterance_windows and has_open_utterance:
        if not _fast_catchup_chain_enabled_runtime:
            log_print(
                "INFO",
                "Fast catch-up chain disabled; skipping pending open utterance",
                session_id=session_id,
                segment_id=segment_id,
                start_ts=start_ts,
                end_ts=end_ts,
                cursor_ts=cursor_ts,
            )
            return False
        newly_pending = False
        with _segment_ctx_lock:
            newly_pending = segment_id not in _pending_fast_catchup_segments
            _pending_fast_catchup_segments[segment_id] = session_id
        if newly_pending:
            _emit_fast_catchup_pending(session_id, True, segment_id)
        log_print(
            "INFO",
            "Fast catch-up pending open utterance",
            session_id=session_id,
            segment_id=segment_id,
            start_ts=start_ts,
            end_ts=end_ts,
            cursor_ts=cursor_ts,
        )
        return True

    if not utterance_windows:
        cleared_pending = False
        session_has_pending = False
        with _segment_ctx_lock:
            cleared_pending = (
                _pending_fast_catchup_segments.pop(segment_id, None) is not None
            )
            session_has_pending = any(
                sid == session_id for sid in _pending_fast_catchup_segments.values()
            )
        if cleared_pending and not session_has_pending:
            _emit_fast_catchup_pending(session_id, False, segment_id)
        return False

    speed = float(_fast_catchup_speed_runtime)
    gap_sec = float(_fast_catchup_gap_sec_runtime)
    emitted_any = False
    last_emitted_end_ts = cursor_ts

    for idx, (cursor_start, cursor_end) in enumerate(utterance_windows, start=1):
        lag_sec = max(0.0, cursor_end - cursor_start)
        if lag_sec <= _FAST_CATCHUP_CHAIN_MIN_LAG_SEC:
            last_emitted_end_ts = max(last_emitted_end_ts, cursor_end)
            continue

        dialogue, _entries = _get_dialogue_by_time_window(cursor_start, cursor_end)
        output_sec = _synthesize_fast_catchup_dialogue(
            dialogue=dialogue,
            segment_id=segment_id,
            session_id=session_id,
            start_ts=cursor_start,
            end_ts=cursor_end,
            speed=speed,
            gap_sec=gap_sec,
            silence_thresh_db=_FAST_CATCHUP_DEFAULT_SILENCE_THRESH_DB,
            trigger_source=f"segment_fast_catchup_vad_utt{idx}",
        )
        if output_sec <= 0:
            break
        emitted_any = True
        last_emitted_end_ts = max(last_emitted_end_ts, cursor_end)

    with _segment_ctx_lock:
        if segment_id in _segment_windows:
            _segment_windows[segment_id]["fast_catchup_cursor_ts"] = last_emitted_end_ts
        had_pending = segment_id in _pending_fast_catchup_segments
        if has_open_utterance:
            _pending_fast_catchup_segments[segment_id] = session_id
            session_has_pending = True
        else:
            _pending_fast_catchup_segments.pop(segment_id, None)
            session_has_pending = any(
                sid == session_id for sid in _pending_fast_catchup_segments.values()
            )
    if has_open_utterance and not had_pending:
        _emit_fast_catchup_pending(session_id, True, segment_id)
    if (not has_open_utterance) and had_pending and (not session_has_pending):
        _emit_fast_catchup_pending(session_id, False, segment_id)
    return emitted_any
