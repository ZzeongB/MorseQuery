function handleTap() {
    const now = Date.now();
    unlockAudioFeedback();
    if (now - lastSpace < 300) {
        // Double tap
        if (summaryRequested || summaryInProgress && !keywordTtsPlaying) {
            playConfirmedFeedback();
            cancelSummaryPlayback();
            lastSpace = 0;
            return;
        }
        if (infoVisible && options.length > 0) {
            playConfirmedFeedback();
            const wasKeywordPlaying = keywordTtsPlaying;
            cancelKeywordTts();
            options = [];
            currentIdx = 0;
            infoVisible = false;
            document.getElementById('optionsList').innerHTML = '';

            if (wasKeywordPlaying) {
                showSkippedIndicator('Keyword canceled');
                setTimeout(() => {
                    startSummarizing();
                }, 500);
            } else {
                startSummarizing();
            }
        } else {
            playConfirmedFeedback();
            hideSummary();
            startListeningIfNeeded();
            socket.emit('request');
            showLoadingIndicator('Inferring', 'inferring', 320);

            if (inferencingTimer) clearTimeout(inferencingTimer);
            inferencingTimer = setTimeout(() => {
                hideLoadingIndicator();
                showSkippedIndicator('No new keywords this round');
                inferencingTimer = null;
            }, INFERENCING_TIMEOUT_MS);
        }
        lastSpace = 0;
    } else {
        // Single tap - wait to see if double
        showTapIndicator();
        lastSpace = now;
        setTimeout(() => {
            if (lastSpace !== 0 && Date.now() - lastSpace >= 280) {
                if (options.length > 0) {
                    playTapFeedback();
                    recoverListeningForKeywordNavigation();
                    cancelKeywordTts();
                    currentIdx = (currentIdx + 1) % options.length;
                    render();
                    if (keywordOutputMode === 'audio') {
                        playCurrentKeywordTts();
                    }
                }
                lastSpace = 0;
            }
        }, 300);
    }
}

// ============================================================================
// Socket Event Handlers
// ============================================================================

socket.on('noise_gate_levels', data => {
    if (noiseGateEnabled) {
        for (const [micId, rms] of Object.entries(data)) {
            updateNoiseGateUI(micId, rms);
        }
    }
});

socket.on('mic_levels', data => {
    for (const [selectId, level] of Object.entries(data)) {
        const levelBar = document.getElementById(selectId + 'Level');
        if (levelBar) {
            levelBar.style.width = level + '%';
        }
    }
});

socket.on('keywords', data => {
    if (inferencingTimer) {
        clearTimeout(inferencingTimer);
        inferencingTimer = null;
    }
    if (!Array.isArray(data) || data.length === 0) {
        hideLoadingIndicator();
        return;
    }
    hideSummary();
    options = options.concat(data);
    const MAX_KEYWORDS = 3;
    if (options.length > MAX_KEYWORDS) {
        options = options.slice(-MAX_KEYWORDS);
    }
    currentIdx = options.length - 1;
    infoVisible = true;

    preloadKeywordTtsForItems(data);

    if (keywordOutputMode === 'audio') {
        playCurrentKeywordTts();
    } else {
        hideLoadingIndicator();
        render();
    }
});

socket.on('summary_chunk', data => {
    // Summary display removed
});

socket.on('clear', () => {
    // Don't clear keywords - keep last 3 visible
});

socket.on('session_ended', () => {
    options = [];
    currentIdx = 0;
    infoVisible = false;
    summaryRequested = false;
    summaryInProgress = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    listeningActive = false;
    summaryTriggeredForListeningSession = false;
    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    clearKeywordAutoSummarizeTimer();
    keywordTtsPreloadedTexts.clear();
    pendingSummaryTexts = [];
    streamingTtsBuffers.clear();
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    document.getElementById('optionsList').innerHTML = '';
    hideLoadingIndicator();
    hideSummaryText();
    hideReconstructedTurns();
    clearReconstructedState();
    summarySegmentState.clear();
    pendingEmptySummarySignal = null;
    pendingSkippedIndicator = null;
});

socket.on('summary_done', data => {
    const segmentId = Number((data && data.segment_id) || 0);
    if (!isSummaryCycleActive()) {
        // Ignore stale/unsolicited summary events when user is not in a summary flow.
        if (segmentId > 0) summarySegmentState.delete(segmentId);
        return;
    }

    const text = (!data.is_empty && !data.no_summary) ? (data.summary || '').trim() : '';
    markSummaryDoneForSegment(segmentId, !!text);

    if (text) {
        pendingSummaryTexts.push(text);
        if (summaryFinalizeTimer) {
            clearTimeout(summaryFinalizeTimer);
        }
        summaryFinalizeTimer = setTimeout(() => {
            if (reconstructorEnabled && (pendingReconstructedTurns.length > 0 || pendingReconstructedSegmentId > 0)) return;
            if (!summaryInProgress && summaryRequested) {
                summaryRequested = false;
                summaryInProgress = false;
                pendingSummarizeIndicatorAfterKeyword = false;
                awaitingJudgeDecision = false;
                hideLoadingIndicator();
                hideSummaryText();
                pendingSummaryTexts = [];
                showSkippedIndicator('Summary unavailable');
            }
        }, 8000);
        return;
    }

    if (hasAllSummariesEmpty(segmentId)) {
        if (reconstructorEnabled && (pendingReconstructedTurns.length > 0 || pendingReconstructedSegmentId > 0)) return;
        queueEmptySummarySignal(segmentId);
        return;
    }

    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
    }
    summaryFinalizeTimer = setTimeout(() => {
        if (pendingSummaryTexts.length > 0 || !summaryRequested) return;
        if (reconstructorEnabled && (pendingReconstructedTurns.length > 0 || pendingReconstructedSegmentId > 0)) return;
        summaryRequested = false;
        summaryInProgress = false;
        pendingSummarizeIndicatorAfterKeyword = false;
        awaitingJudgeDecision = false;
        hideLoadingIndicator();
        hideSummaryText();
        pendingSummaryTexts = [];
        showSkippedIndicator('No summary detected');
        if (segmentId > 0) summarySegmentState.delete(segmentId);
    }, 1500);
});

socket.on('fast_catchup_pending', data => {
    fastCatchupPending = !!(data && data.pending);
});

socket.on('tts_playing', data => {
    const reason = (data && data.reason) || '';
    stopLoadingAudioFeedback();

    if (reason === 'keyword') {
        keywordTtsPlaying = true;
        scheduleAutoSummarizeFromKeywordPlayback(keywordPlaybackToken);
        if (keywordOutputMode === 'audio' && options.length > 0) {
            hideLoadingIndicator();
            render();
        }
        return;
    }

    if (reconstructorEnabled) {
        return;
    }

    summaryInProgress = true;
    summaryRequested = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    hideLoadingIndicator();
    setTimeout(() => {
        if (ttsPlaying || ttsQueue.length > 0) return;
        playTtsStartFeedback('summary');
    }, 260);
});

socket.on('tts_done', () => {
    summaryInProgress = false;
    summaryRequested = false;
    listeningActive = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    playTtsEndFeedback('summary');
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    hideLoadingIndicator();
    hideSummaryText();
    pendingSummaryTexts = [];
});

socket.on('judge_rejected', data => {
    summaryRequested = false;
    summaryInProgress = false;
    allowPostFollowupTts = false;
    listeningActive = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    hideLoadingIndicator();
    hideSummaryText();
    pendingSummaryTexts = [];
    const reason = data.reason || 'All caught up';
    showSkippedIndicator(reason);
});

socket.on('conversation_reconstructed', data => {
    console.log('conversation_reconstructed', data);
});

socket.on('conversation_reconstruct_done', data => {
    console.log('conversation_reconstruct_done', data.segment_id, data.turn_count);
});

socket.on('conversation_reconstructed_turns', data => {
    const turns = Array.isArray(data && data.turns) ? data.turns : [];
    const segId = Number((data && data.segment_id) || 0);
    pendingReconstructedTurns = turns;
    pendingReconstructedSegmentId = segId;
    baseReconstructedTurns = turns;
    baseReconstructedSegmentId = segId;
    followupReconstructedGroups = [];
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    summaryInProgress = true;
    if (turns.length === 0) {
        renderedReconstructedSegmentId = 0;
        renderReconstructedTurns([]);
    }
});

socket.on('summary_tts', data => {
    if (!data.audio) return;
    if (
        reconstructorEnabled &&
        pendingReconstructedTurns.length > 0 &&
        pendingReconstructedSegmentId > 0
    ) {
        return;
    }
    stopLoadingAudioFeedback();

    awaitingJudgeDecision = false;
    summaryRequested = false;
    listeningActive = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }

    summaryInProgress = true;
    summaryTextAllowedThisPlayback = true;
    enqueueTtsAudio(data.audio, { type: 'summary' });
});

socket.on('summary_tts_error', data => {
    console.error('TTS error:', data.error);
    summaryInProgress = false;
    allowPostFollowupTts = false;
    summaryRequested = false;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    hideLoadingIndicator();
    hideSummaryText();
    pendingSummaryTexts = [];
    showSkippedIndicator('Summary audio failed');
});

socket.on('keyword_tts', data => {
    if (!data.audio) return;
    enqueueTtsAudio(data.audio, { type: 'keyword' });
});

socket.on('streaming_tts_chunk', data => {
    if (!data || !data.audio) return;
    const streamId = String(data.stream_id || '').trim();
    if (!streamId) return;

    const audioData = atob(data.audio);
    const chunk = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        chunk[i] = audioData.charCodeAt(i);
    }

    const existing = streamingTtsBuffers.get(streamId) || {
        chunks: [],
        meta: {
            type: data.type || 'summary',
            segmentId: Number(data.segment_id || 0),
            method: data.method || '',
            triggerSource: data.trigger_source || '',
        },
    };
    existing.chunks.push(chunk);
    streamingTtsBuffers.set(streamId, existing);
});

socket.on('streaming_tts_done', data => {
    const streamId = String((data && data.stream_id) || '').trim();
    if (!streamId) return;

    const buffered = streamingTtsBuffers.get(streamId);
    streamingTtsBuffers.delete(streamId);
    if (!buffered || !Array.isArray(buffered.chunks) || buffered.chunks.length === 0) {
        return;
    }
    if (data && data.stopped) {
        return;
    }

    const pcm = concatUint8Arrays(buffered.chunks);
    const wav = pcm16ToWavBytes(pcm, Number(data.sample_rate || 24000), 1);
    const ttsType = buffered.meta && buffered.meta.type ? buffered.meta.type : 'summary';

    if (ttsType === 'reconstruction') {
        stopLoadingAudioFeedback();
        summaryRequested = false;
        listeningActive = false;
        summaryInProgress = true;
        awaitingJudgeDecision = false;
        if (summaryFinalizeTimer) {
            clearTimeout(summaryFinalizeTimer);
            summaryFinalizeTimer = null;
        }
    }

    enqueueTtsBytes(wav, 'audio/wav', {
        type: ttsType,
        segmentId: Number((buffered.meta && buffered.meta.segmentId) || 0),
    });
});

socket.on('conversation_tts_done', data => {
    console.log('conversation_tts_done', data.segment_id, data.count);
});

socket.on('conversation_tts_merged', data => {
    console.log(
        'conversation_tts_merged received',
        data.segment_id,
        data.count,
        data.audio ? data.audio.length : 0
    );
    if (!data.audio) return;
    const segId = Number((data && data.segment_id) || 0);
    stopLoadingAudioFeedback();
    summaryRequested = false;
    listeningActive = false;
    summaryInProgress = true;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    enqueueTtsAudio(data.audio, { type: 'reconstruction', segmentId: segId });
});

socket.on('compressed_dialogue_tts', data => {
    if (!data || !data.audio) return;
    const segId = Number((data && data.segment_id) || 0);
    const method = (data && data.method) || 'unknown';
    const triggerSource = (data && data.trigger_source) || '';
    const text = (data && data.text) || '';
    const isSpeedup = method.startsWith('fast_catchup_source_');

    if (triggerSource === 'post_tts_followup' && !allowPostFollowupTts) {
        console.log('drop late post_tts_followup tts', segId, method);
        return;
    }
    if (
        triggerSource === 'post_tts_followup' &&
        (baseReconstructedSegmentId <= 0 || segId <= baseReconstructedSegmentId)
    ) {
        console.log('drop early/invalid post_tts_followup tts', segId, method, baseReconstructedSegmentId);
        return;
    }

    if (!isSpeedup) {
        const turns = parseCompressedDialogueTurns(text);
        if (turns.length > 0) {
            if (triggerSource === 'post_tts_followup') {
                upsertFollowupReconstructedGroup(segId, turns);
            } else {
                baseReconstructedTurns = turns;
                baseReconstructedSegmentId = segId;
                followupReconstructedGroups = [];
            }
            pendingReconstructedTurns = turns;
            pendingReconstructedSegmentId = segId;
            renderedReconstructedSegmentId = 0;
        }
    }

    stopLoadingAudioFeedback();
    summaryRequested = false;
    listeningActive = false;
    summaryInProgress = true;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    console.log('compressed_dialogue_tts received', segId, method, data.turn_count || 0);
    enqueueTtsAudio(data.audio, {
        type: 'reconstruction',
        segmentId: segId,
        triggerSource,
        visualMode: isSpeedup ? 'none' : 'reconstruction',
    });
});

socket.on('transcript_compression_comparison', data => {
    if (!data) return;
    const segId = Number((data && data.segment_id) || 0);
    const selected = data.selected || null;
    console.log('transcript_compression_comparison', data);
    if (selected && selected.method) {
        console.log(
            `[seg=${segId}] selected=${selected.method} latency=${Math.round(selected.elapsed_ms || 0)}ms`
        );
    }
});

socket.on('dialogue_transcript_ready', data => {
    if (!data) return;
    console.log(
        'dialogue_transcript_ready',
        Number(data.segment_id || 0),
        Number(data.entry_count || 0),
        data.path || ''
    );
});

socket.on('conversation_tts', data => {
    console.log('conversation_tts received', data.segment_id, data.turn_index, data.audio ? data.audio.length : 0);
    if (!data.audio) return;
    const segId = Number((data && data.segment_id) || 0);
    stopLoadingAudioFeedback();
    summaryRequested = false;
    listeningActive = false;
    summaryInProgress = true;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    enqueueTtsAudio(data.audio, { type: 'reconstruction', segmentId: segId });
});

socket.on('keyword_tts_error', data => {
    console.error('Keyword TTS error:', data.error);
    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    if (pendingSummarizeIndicatorAfterKeyword && summaryRequested && summaryInProgress) {
        pendingSummarizeIndicatorAfterKeyword = false;
        hideInfo();
        hideSummaryText();
        hideReconstructedTurns();
        showLoadingIndicator('Summarizing...', 'summarizing', 220);
    }
    maybeEmitPendingSkippedIndicator();
});

socket.on('keyword_tts_done', () => {
    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    const keywordPlaybackBusy = isKeywordPlaybackBusy();
    // Fallback auto-trigger: if duration estimate missed, trigger summarize on actual TTS completion.
    if (
        autoPreSummarizeEnabled &&
        dismissMode === 'summary' &&
        listeningActive &&
        !summaryTriggeredForListeningSession &&
        !summaryRequested &&
        !keywordPlaybackBusy
    ) {
        startSummarizing();
    }
    if (
        pendingSummarizeIndicatorAfterKeyword &&
        summaryRequested &&
        summaryInProgress &&
        !keywordPlaybackBusy
    ) {
        pendingSummarizeIndicatorAfterKeyword = false;
        hideInfo();
        hideSummaryText();
        hideReconstructedTurns();
        showLoadingIndicator('Summarizing...', 'summarizing', 220);
    }
    if (!ttsPlaying && ttsQueue.length === 0) {
        playTtsEndFeedback('keyword');
    }
    if (!ttsPlaying && ttsQueue.length > 0) {
        playNextTts();
    }
    maybeEmitPendingEmptySummarySignal();
    maybeEmitPendingSkippedIndicator();
});

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener('keydown', e => {
    if (e.repeat) return;
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
        return;
    }
    unlockAudioFeedback();
    e.preventDefault();
});

document.addEventListener('keyup', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.isContentEditable) {
        return;
    }
    if (e.key === 'PageDown') {
        clearAllActiveUiAndAudio();
        return;
    }
    handleTap();
});

document.addEventListener('contextmenu', e => {
    e.preventDefault();
});

document.addEventListener('mouseup', e => {
    if (e.button !== 2) return;
    handleTap();
});

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initHeightSlider();
    fetchDevices();
    fetchOutputDevices();
});

window.addEventListener('resize', applyWidgetHeight);
