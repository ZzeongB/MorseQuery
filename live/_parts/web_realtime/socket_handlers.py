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
        _fast_catchup_threshold_sec_runtime, \
        _fast_catchup_speed_runtime, \
        _fast_catchup_gap_sec_runtime, \
        _fast_catchup_chain_enabled_runtime, \
        _summary_followup_enabled_runtime, \
        _missed_summary_latency_bridge_enabled_runtime
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
        try:
            _fast_catchup_speed_runtime = float(
                data.get("fast_catchup_speed", _FAST_CATCHUP_DEFAULT_SPEED)
                or _FAST_CATCHUP_DEFAULT_SPEED
            )
        except Exception:
            _fast_catchup_speed_runtime = _FAST_CATCHUP_DEFAULT_SPEED
        _fast_catchup_speed_runtime = max(1.0, min(3.0, _fast_catchup_speed_runtime))
        try:
            _fast_catchup_gap_sec_runtime = float(
                data.get("fast_catchup_gap_sec", _FAST_CATCHUP_DEFAULT_GAP_SEC)
                or _FAST_CATCHUP_DEFAULT_GAP_SEC
            )
        except Exception:
            _fast_catchup_gap_sec_runtime = _FAST_CATCHUP_DEFAULT_GAP_SEC
        _fast_catchup_gap_sec_runtime = max(
            0.0, min(2.0, _fast_catchup_gap_sec_runtime)
        )
        _fast_catchup_chain_enabled_runtime = bool(
            data.get("fast_catchup_chain_enabled", False)
        )
        _summary_followup_enabled_runtime = bool(
            data.get("summary_followup_enabled", False)
        )
        _missed_summary_latency_bridge_enabled_runtime = bool(
            data.get("missed_summary_latency_bridge_enabled", False)
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
                    "fast_catchup_cursor_ts": time.time(),
                }
                _post_tts_followup_active = False
                _post_tts_followup_inflight = False
                _post_tts_followup_cursor_ts = 0.0
                _post_tts_followup_live_window_open = False
                _pending_fast_catchup_segments.clear()
                _pending_fast_catchup_inflight.clear()
            _emit_fast_catchup_pending(session_id, False, _segment_seq)
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
                    _post_tts_followup_active = bool(_summary_followup_enabled_runtime)
                    _post_tts_followup_inflight = False
                    _post_tts_followup_cursor_ts = ended_at
                    _post_tts_followup_live_window_open = bool(
                        _summary_followup_enabled_runtime
                    )
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
                with _segment_ctx_lock:
                    if current_segment_id > 0 and current_segment_id in _segment_windows:
                        _segment_windows[current_segment_id]["bridge_decision_ts"] = time.time()
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
                speed=_fast_catchup_speed_runtime,
                gap_sec=_fast_catchup_gap_sec_runtime,
                catchup_chain_enabled=_fast_catchup_chain_enabled_runtime,
                summary_followup_enabled=_summary_followup_enabled_runtime,
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
    - fast_catchup_speed: optional, default 1.5
    - fast_catchup_gap_sec: optional, default 0.0
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
    missed_summary_latency_bridge_enabled = bool(
        payload.get(
            "missed_summary_latency_bridge_enabled",
            _missed_summary_latency_bridge_enabled_runtime,
        )
    )
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
    fast_catchup_gap_sec = max(0.0, min(2.0, fast_catchup_gap_sec))

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
            fallback_decision_ts = time.time()
            _trigger_parallel_compression_for_dialogue(
                dialogue=dialogue,
                segment_id=segment_id,
                session_id=session_id,
                before_context=before_context,
                trigger_source="missed_timestamp",
                window={
                    "bridge_decision_ts": fallback_decision_ts,
                    "miss_start_ts": start_ts,
                    "miss_end_ts": end_ts,
                    "window_start_ts": window_start,
                    "window_end_ts": window_end,
                    "padding_before_sec": pad_before,
                    "padding_after_sec": pad_after,
                    "missed_summary_latency_bridge_enabled": missed_summary_latency_bridge_enabled,
                    "fast_catchup_speed": fast_catchup_speed,
                    "fast_catchup_gap_sec": fast_catchup_gap_sec,
                    "fast_catchup_silence_thresh_db": fast_catchup_silence_thresh_db,
                },
                entries=entries,
            )

        threading.Thread(target=_run_fast_catchup_with_fallback, daemon=True).start()
        return

    summarize_decision_ts = time.time()
    threading.Thread(
        target=_trigger_parallel_compression_for_dialogue,
        kwargs={
            "dialogue": dialogue,
            "segment_id": segment_id,
            "session_id": session_id,
            "before_context": before_context,
            "trigger_source": "missed_timestamp",
            "window": {
                "bridge_decision_ts": summarize_decision_ts,
                "miss_start_ts": start_ts,
                "miss_end_ts": end_ts,
                "window_start_ts": window_start,
                "window_end_ts": window_end,
                "padding_before_sec": pad_before,
                "padding_after_sec": pad_after,
                "missed_summary_latency_bridge_enabled": missed_summary_latency_bridge_enabled,
                "fast_catchup_speed": fast_catchup_speed,
                "fast_catchup_gap_sec": fast_catchup_gap_sec,
                "fast_catchup_silence_thresh_db": fast_catchup_silence_thresh_db,
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
        return {"ok": False, "active": False}
    reason = str((data or {}).get("reason", "")).strip() or "browser_done"
    if reason == "user_cancel":
        with _segment_ctx_lock:
            _pending_latency_bridge_by_session.pop(session_id, None)
    if reason != "user_cancel" and _has_pending_fast_catchup(session_id):
        _emit_fast_catchup_pending(session_id, True, 0)
        return {"ok": True, "defer_finish": True}
    if reason != "user_cancel" and _has_pending_latency_bridge(session_id):
        emitted = _emit_pending_latency_bridge_if_needed(session_id, reason)
        if emitted:
            return {"ok": True, "defer_finish": True}
    # Browser summary/reconstruction playback ended.
    _set_keyword_anc_hold(False, f"browser_tts_done:{reason}")
    _on_tts_finished(f"browser_tts_playback_done:{reason}")
    # Enforce transparency on summary completion even if counters drift.
    _set_airpods_mode(
        "transparency", f"browser_tts_playback_done_force:{reason}", wait=True
    )
    return {"ok": True, "defer_finish": False}


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
    sio.run(
        app,
        host="0.0.0.0",
        port=5002,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
