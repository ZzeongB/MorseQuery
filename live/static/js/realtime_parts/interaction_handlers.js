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
    recentlyFinishedStreamingTts.clear();
    awaitingJudgeDecision = false;
    summaryCueSequenceActive = false;
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
    if (ignoreIncomingSummaryEvents) return;
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
    if (ignoreIncomingSummaryEvents && reason !== 'keyword') {
        return;
    }
    stopLoadingAudioFeedback();

    if (reason === 'keyword') {
        keywordTtsPlaying = true;
        // Don't schedule auto-summarize here - wait until keyword TTS is done
        if (keywordOutputMode === 'audio' && options.length > 0) {
            hideLoadingIndicator();
            render();
        }
        return;
    }

    if (reason === 'reconstruction') {
        summaryInProgress = true;
        summaryRequested = false;
        pendingSummarizeIndicatorAfterKeyword = false;
        awaitingJudgeDecision = false;
        if (summaryFinalizeTimer) {
            clearTimeout(summaryFinalizeTimer);
            summaryFinalizeTimer = null;
        }
        hideLoadingIndicator();
        summaryTextAllowedThisPlayback = true;
        if (pendingReconstructedTurns.length > 0) {
            renderReconstructedTurns(pendingReconstructedTurns);
            renderedReconstructedSegmentId = pendingReconstructedSegmentId;
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
    if (ignoreIncomingSummaryEvents) return;
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
    if (ignoreIncomingSummaryEvents) return;
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

    const sampleRate = Number(data.sample_rate || 24000);
    const ttsType = data.type || 'summary';
    const meta = {
        type: ttsType,
        segmentId: Number(data.segment_id || 0),
        method: data.method || '',
        triggerSource: data.trigger_source || '',
    };

    // Check if this stream is already in the queue
    let queueItem = streamingTtsQueue.find(item => item.streamId === streamId);

    if (!queueItem) {
        // Create new queue item for this stream
        queueItem = {
            streamId,
            sampleRate,
            meta,
            ttsType,
            chunks: [],
            receivingDone: false,
            player: null,
        };
        streamingTtsQueue.push(queueItem);
        console.log(`[StreamingTTS] New stream queued: ${streamId}, queue length: ${streamingTtsQueue.length}`);
    }

    // Buffer the chunk
    queueItem.chunks.push(chunk);

    // If no current player, start playing this stream
    if (!currentStreamingPlayer) {
        startStreamingTtsPlayback(queueItem);
    } else if (currentStreamingPlayer === queueItem.player) {
        // This is the current stream, push chunk immediately
        queueItem.player.pushChunk(chunk);
        queueItem.chunks.pop(); // Remove from buffer since pushed
    }
    // Otherwise, chunk stays buffered until this stream's turn
});

const RECENT_STREAM_DONE_TTL_MS = 15000;
const recentlyFinishedStreamingTts = new Map(); // streamId -> finishedAt(ms)

function markStreamingTtsFinished(streamId) {
    if (!streamId) return;
    const now = Date.now();
    recentlyFinishedStreamingTts.set(streamId, now);
    for (const [id, ts] of recentlyFinishedStreamingTts) {
        if (now - ts > RECENT_STREAM_DONE_TTL_MS) {
            recentlyFinishedStreamingTts.delete(id);
        }
    }
}

function wasStreamingTtsFinishedRecently(streamId) {
    const ts = recentlyFinishedStreamingTts.get(streamId);
    if (!ts) return false;
    if (Date.now() - ts > RECENT_STREAM_DONE_TTL_MS) {
        recentlyFinishedStreamingTts.delete(streamId);
        return false;
    }
    return true;
}

function createStreamingPlayer(queueItem) {
    const { streamId, sampleRate, meta, ttsType } = queueItem;
    const player = new StreamingTtsPlayer(streamId, sampleRate, meta);

    player.onFirstChunk = () => {
        hideLoadingIndicator();
        if (ttsType === 'keyword') {
            keywordTtsPlaying = true;
            playTtsStartFeedback('keyword');
        } else if (ttsType === 'reconstruction' || ttsType === 'summary') {
            stopLoadingAudioFeedback();
            summaryInProgress = true;
            if (!summaryCueSequenceActive) {
                summaryCueSequenceActive = true;
                playTtsStartFeedback('summary');
            }
        }
    };

    player.onEndCallback = () => {
        console.log(`[StreamingTTS] Playback ended: ${streamId}`);

        // Remove this stream from queue
        const idx = streamingTtsQueue.findIndex(item => item.streamId === streamId);
        if (idx >= 0) {
            streamingTtsQueue.splice(idx, 1);
        }
        markStreamingTtsFinished(streamId);
        streamingTtsPlayers.delete(streamId);
        currentStreamingPlayer = null;

        if (ttsType === 'keyword') {
            keywordTtsPlaying = false;
            keywordTtsCurrentText = '';
            playTtsEndFeedback('keyword');
            hideInfo();
            socket.emit('keyword_tts_done');

            // Show "Summarizing..." if it was deferred
            if (
                pendingSummarizeIndicatorAfterKeyword &&
                summaryRequested &&
                summaryInProgress
            ) {
                pendingSummarizeIndicatorAfterKeyword = false;
                hideSummaryText();
                hideReconstructedTurns();
                showLoadingIndicator('Summarizing...', 'summarizing', 220);
            }

            // Auto-start summarizing if enabled (after playback done)
            maybeStartSummarizingAfterKeyword();
        } else if (ttsType === 'reconstruction' || ttsType === 'summary') {
            // Check if there are more summary streams in queue
            const hasMoreSummary = streamingTtsQueue.some(
                item => item.ttsType === 'reconstruction' || item.ttsType === 'summary'
            );
            if (!hasMoreSummary) {
                summaryCueSequenceActive = false;
                playTtsEndFeedback('summary');
                summaryInProgress = false;
                hideSummaryText();
                hideReconstructedTurns();
            }
        }

        // Play next stream in queue
        playNextStreamingTts();
    };

    return player;
}

function startStreamingTtsPlayback(queueItem) {
    if (queueItem.player) return; // Already started

    const player = createStreamingPlayer(queueItem);
    queueItem.player = player;
    streamingTtsPlayers.set(queueItem.streamId, player);
    currentStreamingPlayer = player;

    console.log(`[StreamingTTS] Starting playback: ${queueItem.streamId}`);

    // Push all buffered chunks
    for (const chunk of queueItem.chunks) {
        player.pushChunk(chunk);
    }
    queueItem.chunks = [];

    // If already done receiving, signal finish
    if (queueItem.receivingDone) {
        console.log(`[StreamingTTS] Stream already done receiving: ${queueItem.streamId}, signaling finish`);
        player.finish();
    }
}

function playNextStreamingTts() {
    if (streamingTtsQueue.length === 0) {
        currentStreamingPlayer = null;
        console.log('[StreamingTTS] Queue empty');
        return;
    }

    const next = streamingTtsQueue[0];
    console.log(`[StreamingTTS] Playing next: ${next.streamId}`);
    startStreamingTtsPlayback(next);
}

socket.on('streaming_tts_done', data => {
    const streamId = String((data && data.stream_id) || '').trim();
    if (!streamId) return;

    const queueItem = streamingTtsQueue.find(item => item.streamId === streamId);
    if (!queueItem) {
        if (!data || !data.stopped) {
            // Normal races can deliver "done" after playback cleanup, or for empty streams with no chunks.
            if (!wasStreamingTtsFinishedRecently(streamId)) {
                console.log(`[StreamingTTS] Ignore late/empty done: ${streamId}`);
            }
        }
        return;
    }

    console.log(`[StreamingTTS] Receiving done: ${streamId}, stopped: ${!!(data && data.stopped)}`);

    if (data && data.stopped) {
        // Stream was stopped/cancelled
        if (queueItem.player) {
            queueItem.player.stop();
        }
        const idx = streamingTtsQueue.findIndex(item => item.streamId === streamId);
        if (idx >= 0) streamingTtsQueue.splice(idx, 1);
        markStreamingTtsFinished(streamId);
        streamingTtsPlayers.delete(streamId);
        if (currentStreamingPlayer === queueItem.player) {
            currentStreamingPlayer = null;
            if (!streamingTtsQueue.some(
                item => item.ttsType === 'reconstruction' || item.ttsType === 'summary'
            )) {
                summaryCueSequenceActive = false;
            }
            playNextStreamingTts();
        }
        return;
    }

    // Mark as done receiving chunks
    queueItem.receivingDone = true;

    // If this stream is currently playing, signal finish to wait for playback
    if (queueItem.player && currentStreamingPlayer === queueItem.player) {
        queueItem.player.finish();
        // onEndCallback will handle cleanup and playing next
    }
    // If not yet playing, it will be handled when its turn comes
});

socket.on('conversation_tts_done', data => {
    console.log('conversation_tts_done', data.segment_id, data.count);
});

socket.on('conversation_tts_merged', data => {
    if (ignoreIncomingSummaryEvents) return;
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
    if (ignoreIncomingSummaryEvents) return;
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
    if (ignoreIncomingSummaryEvents) return;
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
    // Server-side done can arrive before browser playback ends for streaming keyword TTS.
    // In that case, defer completion handling until playback actually ends.
    const streamingKeywordStillPlaying = streamingTtsQueue.some(item => item && item.ttsType === 'keyword');
    if (streamingKeywordStillPlaying) {
        console.log('[StreamingTTS] Ignore keyword_tts_done until keyword playback ends');
        return;
    }

    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    const keywordPlaybackBusy = isKeywordPlaybackBusy();
    // Fallback auto-trigger: if duration estimate missed, trigger summarize on actual TTS completion.
    if (!keywordPlaybackBusy) {
        maybeStartSummarizingAfterKeyword();
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
    if (e.key === 'PageDown' || e.key === 'Escape') {
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
