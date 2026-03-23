function handleTap() {
    unlockAudioFeedback();
    showTapIndicator();
    if (!pauseResumeEnabled) return;
    if (ttsPaused) {
        playTapFeedback();
        resumeTtsPlayback();
        return;
    }
    if (ttsPlaying || (currentStreamingPlayer && currentStreamingPlayer.isPlaying)) {
        playTapFeedback();
        pauseTtsPlayback();
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
    // Ignore keywords received after dismiss
    if (ignoreIncomingKeywordEvents) {
        return;
    }
    // Check if response is from a stale request (requestId mismatch)
    const responseRequestId = (data && data.requestId) || 0;
    if (responseRequestId > 0 && responseRequestId !== keywordRequestToken) {
        console.log('Ignoring stale keyword response', responseRequestId, keywordRequestToken);
        return;
    }
    // Extract keywords array from response
    const keywords = (data && data.keywords) || data;
    if (!Array.isArray(keywords) || keywords.length === 0) {
        hideLoadingIndicator();
        return;
    }
    hideSummary();
    options = options.concat(keywords);
    const MAX_KEYWORDS = 3;
    if (options.length > MAX_KEYWORDS) {
        options = options.slice(-MAX_KEYWORDS);
    }
    currentIdx = options.length - 1;
    infoVisible = true;

    preloadKeywordTtsForItems(keywords);

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
    // Quiz UI is session-scoped; ignore generic clear events.
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
    keywordTtsPlaybackStartTime = 0;
    keywordTtsCurrentText = '';
    ignoreIncomingKeywordEvents = false;
    clearKeywordAutoSummarizeTimer();
    keywordTtsPreloadedTexts.clear();
    clearQuizTimers();
    clearQueuedQuizRound();
    quizBank = [];
    sessionQuizOrder = [];
    quizRoundNumber = 0;
    quizScheduleIndex = 0;
    quizCurrentIndex = 0;
    quizRoundActive = false;
    quizRoundEndsAt = 0;
    quizSessionStartAt = 0;
    quizAncActive = false;
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

socket.on('quiz_ready', data => {
    const questions = Array.isArray(data && data.questions) ? data.questions : [];
    if (questions.length === 0) {
        console.warn('quiz_ready received without questions');
        return;
    }
    quizBank = questions;
    quizRoundDurationSec = Math.max(1, Number((data && data.duration_sec) || 60));
    sessionQuizOrder = buildSessionQuizOrder();
    quizRoundNumber = 0;
    quizScheduleIndex = 0;
    quizCurrentIndex = 0;
    if (!quizSessionStartAt) {
        quizSessionStartAt = Date.now();
    }
    scheduleNextQuizRound();
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
                endQuizAncSession();
                scheduleNextQuizRound();
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
        endQuizAncSession();
        scheduleNextQuizRound();
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
        keywordTtsPlaybackStartTime = Date.now();
        console.log('[KeywordTTS] tts_playing received, keywordTtsPlaybackStartTime:', keywordTtsPlaybackStartTime);
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
    endQuizAncSession();
    scheduleNextQuizRound();
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
    // Notify server to turn off ANC
    socket.emit('browser_tts_playback_done', { reason: 'judge_rejected' });
    endQuizAncSession();
    scheduleNextQuizRound();
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
    endQuizAncSession();
    scheduleNextQuizRound();
});

socket.on('keyword_tts', data => {
    if (ignoreIncomingKeywordEvents) return;
    if (!data.audio) return;
    enqueueTtsAudio(data.audio, { type: 'keyword' });
});

socket.on('streaming_tts_chunk', data => {
    if (!data || !data.audio) return;
    const streamId = String(data.stream_id || '').trim();
    if (!streamId) return;

    const ttsType = data.type || 'summary';
    // Ignore keyword streams after keyword dismiss
    if (ignoreIncomingKeywordEvents && ttsType === 'keyword') {
        return;
    }
    // Ignore summary/reconstruction streams after dismiss
    if (ignoreIncomingSummaryEvents && ttsType !== 'keyword') {
        return;
    }

    const audioData = atob(data.audio);
    const chunk = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        chunk[i] = audioData.charCodeAt(i);
    }

    const sampleRate = Number(data.sample_rate || 24000);
    const keyword = data.keyword || '';
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
            keyword,
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
            keywordTtsPlaybackStartTime = Date.now();
            console.log('[KeywordTTS] Started, keywordTtsPlaybackStartTime:', keywordTtsPlaybackStartTime);
            playTtsStartFeedback('keyword');
            // Notify server that keyword TTS playback started
            socket.emit('keyword_tts_playback_start', { keyword: queueItem.keyword || '' });
        } else if (ttsType === 'reconstruction' || ttsType === 'summary') {
            stopLoadingAudioFeedback();
            summaryInProgress = true;
            if (!summaryCueSequenceActive) {
                summaryCueSequenceActive = true;
                // Notify server that streaming summary playback started (for AirPods mode)
                socket.emit('browser_tts_playback_start', {
                    source: ttsType,
                    segment_id: Number((meta && meta.segmentId) || 0),
                });
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
            keywordTtsPlaybackStartTime = 0;
            keywordTtsCurrentText = '';
            playTtsEndFeedback('keyword');
            hideInfo();
            socket.emit('keyword_tts_playback_done', { keyword: queueItem.keyword || '' });

            // If this was a resumed playback, switch to transparency mode (ANC off)
            if (ttsResumedPlayback) {
                socket.emit('pause_tts');
                console.log('StreamingTTS keyword onEndCallback: ANC off (resumed playback)');
                ttsResumedPlayback = false;
                return;  // Skip summarizing for resumed playback
            }

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
                // Notify server that streaming summary playback ended (for AirPods transparency)
                socket.emit('browser_tts_playback_done', { reason: 'streaming_summary_done' });
                endQuizAncSession();
                scheduleNextQuizRound();
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
        // Check if there are items in regular TTS queue (e.g., follow-up TTS)
        if (ttsQueue.length > 0 && !ttsPlaying) {
            console.log('[StreamingTTS] Playing queued regular TTS after streaming done');
            playNextTts();
        }
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
    // Track that transcript came in during listening (for dismiss-with-summary logic)
    if (listeningActive && Number(data.entry_count || 0) > 0) {
        hasTranscriptDuringListening = true;
    }
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
    keywordTtsPlaybackStartTime = 0;
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
    keywordTtsPlaybackStartTime = 0;
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
    // PageDown & Esc: dismiss (single click)
    if (e.key === 'PageDown' || e.key === 'Escape') {
        handleDismissKey();
        return;
    }
    // Other keys: pause/resume (single) & keyword request (double)
    handleTap();
});

document.addEventListener('contextmenu', e => {
    e.preventDefault();
});

document.addEventListener('mouseup', e => {
    // Right-click (button 2): pause/resume (single) & keyword request (double)
    if (e.button === 2) {
        handleTap();
        return;
    }
});

// ============================================================================
// Dismiss Key Handler (PageDown & Esc)
// ============================================================================

function dismissAllTtsAndUi() {
    // FIRST: Invalidate pending keyword requests and ignore responses
    // This must happen BEFORE cancel_tts to prevent race conditions where
    // keywords event arrives and triggers keyword_tts (which would turn ANC back on)
    keywordRequestToken += 1;
    ignoreIncomingKeywordEvents = true;

    // Clear keyword loading state
    if (inferencingTimer) {
        clearTimeout(inferencingTimer);
        inferencingTimer = null;
    }

    // Stop all TTS playback (keyword + summary) and trigger anc_off on server
    // This emits 'cancel_tts' which handles both keyword and summary TTS server-side
    stopTtsPlayback();

    // Explicitly turn off ANC (covers inferring cancel case where no TTS has started yet)
    socket.emit('browser_tts_playback_done', { reason: 'dismiss_all' });

    // Local keyword TTS cleanup (don't emit cancel_keyword_tts since cancel_tts already handles it)
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    stopKeywordStreamingPlaybackLocally();
    removeQueuedKeywordBrowserTts();
    keywordTtsPlaying = false;
    keywordTtsPlaybackStartTime = 0;
    keywordTtsCurrentText = '';

    // Clear summary state
    summaryRequested = false;
    summaryInProgress = false;
    allowPostFollowupTts = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    ignoreIncomingSummaryEvents = true;
    listeningActive = false;
    summaryTriggeredForListeningSession = false;
    summaryCueSequenceActive = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }

    // Clear UI
    options = [];
    currentIdx = 0;
    infoVisible = false;
    document.getElementById('optionsList').innerHTML = '';
    ttsResumedPlayback = false;
    hideLoadingIndicator();
    hideSummaryText();
    hideReconstructedTurns();
    clearReconstructedState();
    pendingSummaryTexts = [];

    showDismissIndicator();
    endQuizAncSession();
}

function handleDismissKey() {
    unlockAudioFeedback();
    playTapFeedback();

    // Check if any TTS is playing (keyword or summary) or keyword loading is in progress
    const keywordPlaying = isKeywordPlaybackBusy();
    const keywordLoading = inferencingTimer !== null;
    const summaryPlaying = summaryRequested || summaryInProgress || isTtsPlayingOrPaused();

    if (keywordPlaying || keywordLoading || summaryPlaying) {
        // Check if we should still run summarizing logic
        // (missed conversation since keyword TTS start >= 5 seconds OR transcript came in during listening)
        const durationMs = keywordTtsPlaybackStartTime > 0 ? (Date.now() - keywordTtsPlaybackStartTime) : 0;
        const shouldSummarize = listeningActive &&
            !summaryTriggeredForListeningSession &&
            !summaryRequested &&
            (durationMs >= 5000 || hasTranscriptDuringListening);

        console.log('[Dismiss] keywordTtsPlaybackStartTime:', keywordTtsPlaybackStartTime,
            'durationMs:', durationMs, 'listeningActive:', listeningActive,
            'summaryTriggeredForListeningSession:', summaryTriggeredForListeningSession,
            'summaryRequested:', summaryRequested,
            'hasTranscriptDuringListening:', hasTranscriptDuringListening,
            'shouldSummarize:', shouldSummarize);

        if (shouldSummarize) {
            // FIRST: Invalidate pending keyword requests and ignore responses
            // This prevents keywords event from triggering keyword_tts during summarizing
            keywordRequestToken += 1;
            ignoreIncomingKeywordEvents = true;

            // Clear keyword loading state
            if (inferencingTimer) {
                clearTimeout(inferencingTimer);
                inferencingTimer = null;
            }

            // Stop keyword TTS only, then trigger summarizing
            // Cancel keyword TTS on server (not summary TTS), keep ANC on for summarizing
            socket.emit('cancel_keyword_tts', { keep_anc: true });

            // Local keyword TTS cleanup
            keywordPlaybackToken += 1;
            clearKeywordAutoSummarizeTimer();
            stopKeywordStreamingPlaybackLocally();
            removeQueuedKeywordBrowserTts();
            keywordTtsPlaying = false;
            keywordTtsPlaybackStartTime = 0;
            keywordTtsCurrentText = '';

            // Clear keyword UI
            options = [];
            currentIdx = 0;
            infoVisible = false;
            document.getElementById('optionsList').innerHTML = '';
            ttsResumedPlayback = false;

            // Trigger summarizing (force: bypass dismissMode check)
            startSummarizing({ force: true });
            showDismissIndicator();
        } else {
            // Normal dismiss - stop everything
            dismissAllTtsAndUi();
        }
    }
    // Dismiss only - do nothing if nothing is playing
}

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    initHeightSlider();
    fetchDevices();
    fetchOutputDevices();
});

window.addEventListener('resize', applyWidgetHeight);
