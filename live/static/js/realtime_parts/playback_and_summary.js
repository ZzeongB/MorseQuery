function enqueueTtsAudio(audioBase64, meta = null) {
    console.log('enqueueTtsAudio called, ttsPlaying:', ttsPlaying, 'queueLen:', ttsQueue.length, 'streamingPlayer:', !!currentStreamingPlayer);
    const audioData = atob(audioBase64);
    const audioArray = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
    }

    const blob = new Blob([audioArray], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    ttsQueue.push({ audioData, url, meta });

    // Don't start playback if streaming TTS is playing - will be triggered when streaming finishes
    if (!ttsPlaying && !currentStreamingPlayer && !isBlockedByKeywordPlayback(meta)) {
        playNextTts();
    } else {
        console.log('enqueueTtsAudio: blocked, ttsPlaying:', ttsPlaying, 'streaming:', !!currentStreamingPlayer);
    }
}

function enqueueTtsBytes(audioBytes, mimeType = 'audio/wav', meta = null) {
    if (!audioBytes || audioBytes.length === 0) return;
    const blob = new Blob([audioBytes], { type: mimeType });
    const url = URL.createObjectURL(blob);
    ttsQueue.push({ audioBytes, url, meta });

    // Don't start playback if streaming TTS is playing - will be triggered when streaming finishes
    if (!ttsPlaying && !currentStreamingPlayer && !isBlockedByKeywordPlayback(meta)) {
        playNextTts();
    }
}

function concatUint8Arrays(chunks) {
    const total = chunks.reduce((sum, item) => sum + item.length, 0);
    const merged = new Uint8Array(total);
    let offset = 0;
    for (const chunk of chunks) {
        merged.set(chunk, offset);
        offset += chunk.length;
    }
    return merged;
}

function pcm16ToWavBytes(pcmBytes, sampleRate = 24000, channels = 1) {
    const dataSize = pcmBytes.length;
    const wav = new Uint8Array(44 + dataSize);
    const view = new DataView(wav.buffer);
    const bytesPerSample = 2;
    const blockAlign = channels * bytesPerSample;
    const byteRate = sampleRate * blockAlign;

    view.setUint32(0, 0x52494646, false); // RIFF
    view.setUint32(4, 36 + dataSize, true);
    view.setUint32(8, 0x57415645, false); // WAVE
    view.setUint32(12, 0x666d7420, false); // fmt
    view.setUint32(16, 16, true); // PCM fmt chunk size
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, channels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, 16, true); // bits per sample
    view.setUint32(36, 0x64617461, false); // data
    view.setUint32(40, dataSize, true);
    wav.set(pcmBytes, 44);
    return wav;
}

function isSummaryOrReconstructionMeta(meta) {
    return !!meta && (meta.type === 'summary' || meta.type === 'reconstruction');
}

function isBlockedByKeywordPlayback(meta) {
    return autoPreSummarizeEnabled && keywordTtsPlaying && isSummaryOrReconstructionMeta(meta);
}

// ============================================================================
// Keyword TTS Functions
// ============================================================================

function clearKeywordAutoSummarizeTimer() {
    if (keywordAutoSummarizeTimer) {
        clearTimeout(keywordAutoSummarizeTimer);
        keywordAutoSummarizeTimer = null;
    }
}

function startListeningIfNeeded() {
    if (listeningActive) return;
    socket.emit('start_listening');
    listeningActive = true;
    listeningStartTime = Date.now();
    hasTranscriptDuringListening = false;
    summaryTriggeredForListeningSession = false;
}

function endListeningIfNeeded() {
    if (!listeningActive) return false;
    socket.emit('end_listening', { mode: transcriptSyncMode });
    listeningActive = false;
    summaryTriggeredForListeningSession = true;
    return true;
}

function recoverListeningForKeywordNavigation() {
    if (summaryTriggeredForListeningSession) return;
    startListeningIfNeeded();
}

function estimateKeywordTtsDurationMs(text) {
    const normalized = String(text || '').trim();
    if (!normalized) return 0;
    const words = normalized.split(/\s+/).filter(Boolean).length;
    const punctuationPauses = (normalized.match(/[.,!?;:]/g) || []).length;
    const estimatedSeconds = Math.max(2.0, (words / KEYWORD_ESTIMATED_WPS) + (punctuationPauses * 0.12));
    return Math.round(estimatedSeconds * 1000);
}

function scheduleAutoSummarizeFromKeywordPlayback(playbackToken = keywordPlaybackToken) {
    clearKeywordAutoSummarizeTimer();
    if (!autoPreSummarizeEnabled) return;
    if (dismissMode !== 'summary') return;
    if (summaryRequested || summaryInProgress) return;
    const durationMs = estimateKeywordTtsDurationMs(keywordTtsCurrentText);
    if (durationMs <= 0) return;
    const delayMs = Math.max(0, durationMs - keywordSummaryLeadMs);
    keywordAutoSummarizeTimer = setTimeout(() => {
        if (playbackToken !== keywordPlaybackToken) return;
        keywordAutoSummarizeTimer = null;
        startSummarizing({ silentWhileKeyword: true });
    }, delayMs);
}

function requestKeywordTts(items) {
    if (!Array.isArray(items) || items.length === 0) return;
    keywordTtsPlaying = true;
    for (const item of items) {
        const word = (item.word || '').trim();
        const desc = (item.desc || '').trim();
        if (!word || !desc) continue;
        socket.emit('keyword_tts', { text: `${word}. ${desc}`, keyword: word });
    }
}

function preloadKeywordTtsForItems(items) {
    if (!Array.isArray(items) || items.length === 0) return;

    const texts = [];
    for (const item of items) {
        const word = (item.word || '').trim();
        const desc = (item.desc || '').trim();
        if (!word || !desc) continue;

        const text = `${word}. ${desc}`;
        if (keywordTtsPreloadedTexts.has(text)) continue;
        keywordTtsPreloadedTexts.add(text);
        texts.push(text);
    }

    if (texts.length > 0) {
        socket.emit('keyword_tts_preload', { texts });
    }
}

function playCurrentKeywordTts() {
    if (options.length === 0) return;
    if (isSummaryCycleActive() || summaryTriggeredForListeningSession) return;
    if (isKeywordPlaybackBusy()) return;
    const item = options[currentIdx];
    if (!item) return;
    const word = (item.word || '').trim();
    const desc = (item.desc || '').trim();
    if (!word || !desc) return;
    clearKeywordAutoSummarizeTimer();
    keywordPlaybackToken += 1;
    keywordTtsPlaying = true;
    keywordTtsCurrentText = `${word}. ${desc}`;
    // Don't schedule auto-summarize here - wait until keyword TTS is done (handled in onEndCallback)
    socket.emit('keyword_tts', { text: keywordTtsCurrentText, keyword: word });
}

function stopKeywordStreamingPlaybackLocally() {
    const wasCurrentKeyword =
        currentStreamingPlayer &&
        currentStreamingPlayer.meta &&
        currentStreamingPlayer.meta.type === 'keyword';

    for (const item of streamingTtsQueue) {
        if (!item) continue;
        if (item.ttsType !== 'keyword') continue;
        if (item.player) {
            try {
                item.player.stop();
            } catch (e) {}
        }
        streamingTtsPlayers.delete(item.streamId);
    }

    streamingTtsQueue = streamingTtsQueue.filter(item => item && item.ttsType !== 'keyword');

    if (wasCurrentKeyword) {
        currentStreamingPlayer = null;
    }

    if (!currentStreamingPlayer && streamingTtsQueue.length > 0) {
        playNextStreamingTts();
    }
}

function removeQueuedKeywordBrowserTts() {
    if (!Array.isArray(ttsQueue) || ttsQueue.length === 0) return;
    const kept = [];
    for (const item of ttsQueue) {
        if (item && item.meta && item.meta.type === 'keyword') {
            if (item.url) {
                try { URL.revokeObjectURL(item.url); } catch (e) {}
            }
            continue;
        }
        kept.push(item);
    }
    ttsQueue = kept;
}

function maybeStartSummarizingAfterKeyword() {
    if (!autoPreSummarizeEnabled) return;
    if (dismissMode !== 'summary') return;
    if (summaryTriggeredForListeningSession) return;
    if (!listeningActive) return;
    if (summaryRequested) return;
    if (isKeywordPlaybackBusy()) return;
    // Skip auto-summarize if this was a resumed playback
    if (ttsResumedPlayback) {
        console.log('maybeStartSummarizingAfterKeyword: skipped (resumed playback)');
        ttsResumedPlayback = false;
        return;
    }
    startSummarizing();
}

function cancelKeywordTts() {
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    stopKeywordStreamingPlaybackLocally();
    removeQueuedKeywordBrowserTts();
    socket.emit('cancel_keyword_tts');
    keywordTtsPlaying = false;
    keywordTtsPlaybackStartTime = 0;
    keywordTtsCurrentText = '';
}

// ============================================================================
// UI Rendering Functions
// ============================================================================

function render() {
    const list = document.getElementById('optionsList');
    if (options.length === 0) {
        list.innerHTML = '';
        return;
    }

    if (view === 'single') {
        const o = options[currentIdx];
        list.innerHTML = `
            <div class="option-row">
                <div class="keyword-box active">${o.word}</div>
                <div class="desc-box active">${o.desc}</div>
            </div>
        `;
    } else {
        if (descView === 'all') {
            list.innerHTML = options.map((o, i) => {
                const isActive = i === currentIdx;
                return `
                    <div class="option-row">
                        <div class="keyword-box ${isActive ? 'active' : ''}" onclick="selectKeyword(${i})">${o.word}</div>
                        <div class="desc-box ${isActive ? 'active' : ''}">${o.desc}</div>
                    </div>
                `;
            }).join('');
        } else {
            list.innerHTML = options.map((o, i) => {
                const isActive = i === currentIdx;
                return `
                    <div class="option-row">
                        <div class="keyword-box ${isActive ? 'active' : ''}" onclick="selectKeyword(${i})">${o.word}</div>
                        ${isActive ? `<div class="desc-box active">${o.desc}</div>` : ''}
                    </div>
                `;
            }).join('');
        }
    }
}

function selectKeyword(idx) {
    if (idx < 0 || idx >= options.length) return;
    currentIdx = idx;
    render();
    if (keywordOutputMode === 'audio') {
        recoverListeningForKeywordNavigation();
        cancelKeywordTts();
        playCurrentKeywordTts();
    }
}

function hideInfo() {
    document.getElementById('optionsList').innerHTML = '';
    options = [];
    infoVisible = false;
}

function clearQuizTimers() {
    if (quizRoundTimer) {
        clearTimeout(quizRoundTimer);
        quizRoundTimer = null;
    }
    if (quizCountdownTimer) {
        clearInterval(quizCountdownTimer);
        quizCountdownTimer = null;
    }
}

function clearQueuedQuizRound() {
    if (quizLaunchTimer) {
        clearTimeout(quizLaunchTimer);
        quizLaunchTimer = null;
    }
}

function formatQuizRemaining() {
    const remainMs = Math.max(0, quizRoundEndsAt - Date.now());
    const totalSec = Math.ceil(remainMs / 1000);
    const minutes = Math.floor(totalSec / 60);
    const seconds = totalSec % 60;
    return `${minutes}:${String(seconds).padStart(2, '0')}`;
}

function buildSessionQuizOrder() {
    // Server already sends the correct quiz set (A or B) based on quiz_set parameter
    const base = Array.isArray(quizBank) ? quizBank : [];
    if (base.length === 0) return [];
    const cloned = base.map(item => {
        const options = Array.isArray(item.options) ? item.options.slice() : [];
        const indexed = options.map((text, idx) => ({ text, idx }));
        for (let i = indexed.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indexed[i], indexed[j]] = [indexed[j], indexed[i]];
        }
        return {
            id: item.id,
            text: item.text,
            options: indexed.map(entry => entry.text),
            correct_answer: indexed.findIndex(entry => entry.idx === Number(item.correct_answer)),
        };
    });
    for (let i = cloned.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [cloned[i], cloned[j]] = [cloned[j], cloned[i]];
    }
    return cloned;
}

function updateQuizTimerUi() {
    const timerEl = document.getElementById('quizTimer');
    if (timerEl) {
        timerEl.textContent = formatQuizRemaining();
    }
}

function renderQuizRoundComplete(message = 'Summarizing the last minute...') {
    const list = document.getElementById('optionsList');
    list.innerHTML = `
        <div class="quiz-card">
            <div class="quiz-header">
                <div>
                    <div class="quiz-label">Quiz Round ${quizRoundNumber}</div>
                </div>
                <div class="quiz-timer">0:00</div>
            </div>
            <div class="quiz-question">${message}</div>
        </div>
    `;
}

function renderQuizQuestion() {
    const list = document.getElementById('optionsList');
    if (!quizRoundActive || quizCurrentIndex >= sessionQuizOrder.length) {
        hideInfo();
        return;
    }

    const item = sessionQuizOrder[quizCurrentIndex];
    const optionsHtml = item.options.map((option, idx) => `
        <button class="quiz-option" onclick="submitQuizAnswer(${idx})">${option}</button>
    `).join('');

    list.innerHTML = `
        <div class="quiz-card">
            <div class="quiz-header">
                <div>
                    <div class="quiz-label">Quiz Set ${quizSetSelection} · Round ${quizRoundNumber}</div>
                </div>
                <div class="quiz-timer" id="quizTimer">${formatQuizRemaining()}</div>
            </div>
            <div class="quiz-question">${item.text}</div>
            <div class="quiz-options">${optionsHtml}</div>
        </div>
    `;
}

function scheduleNextQuizRound() {
    clearQueuedQuizRound();
    if (!Array.isArray(sessionQuizOrder) || sessionQuizOrder.length === 0) return;
    if (quizScheduleIndex >= quizPlannedOffsetsSec.length) return;
    const targetAt = quizSessionStartAt + (quizPlannedOffsetsSec[quizScheduleIndex] * 1000);
    const delayMs = Math.max(0, targetAt - Date.now());
    quizLaunchTimer = setTimeout(() => {
        quizLaunchTimer = null;
        startNextQuizRound(quizScheduleIndex);
    }, delayMs);
}

function startNextQuizRound(expectedIndex = quizScheduleIndex) {
    clearQueuedQuizRound();
    clearQuizTimers();
    if (!Array.isArray(sessionQuizOrder) || sessionQuizOrder.length === 0) return;
    if (expectedIndex !== quizScheduleIndex) return;
    if (quizScheduleIndex >= quizPlannedOffsetsSec.length) return;
    if (quizRoundActive) return;
    if (summaryRequested || summaryInProgress || awaitingJudgeDecision) {
        quizLaunchTimer = setTimeout(() => {
            quizLaunchTimer = null;
            startNextQuizRound(expectedIndex);
        }, 1000);
        return;
    }

    quizRoundNumber += 1;
    quizScheduleIndex += 1;
    scheduleNextQuizRound();
    quizRoundActive = true;
    quizRoundEndsAt = Date.now() + (quizRoundDurationSec * 1000);

    hideSummary();
    if (!quizAncActive) {
        socket.emit('quiz_session_start');
        quizAncActive = true;
    }
    startListeningIfNeeded();
    socket.emit('quiz_round_start', {
        quiz_set: quizSetSelection,
        round_number: quizRoundNumber,
        question_ids: sessionQuizOrder.map(item => item.id),
        duration_sec: quizRoundDurationSec,
    });
    renderQuizQuestion();
    updateQuizTimerUi();

    quizCountdownTimer = setInterval(() => {
        updateQuizTimerUi();
        if (Date.now() >= quizRoundEndsAt) {
            finishQuizRound();
        }
    }, 250);
    quizRoundTimer = setTimeout(() => {
        finishQuizRound();
    }, quizRoundDurationSec * 1000);
}

function finishQuizRound() {
    if (!quizRoundActive) return;
    const answeredCount = Math.min(quizCurrentIndex, sessionQuizOrder.length);
    socket.emit('quiz_round_end', {
        quiz_set: quizSetSelection,
        round_number: quizRoundNumber,
        answered_count: answeredCount,
        shown_count: answeredCount,
        total_available: sessionQuizOrder.length,
    });
    quizRoundActive = false;
    clearQuizTimers();
    hideInfo();
    startSummarizing({ force: true });
}

function submitQuizAnswer(selectedIdx) {
    if (!quizRoundActive) return;
    const item = sessionQuizOrder[quizCurrentIndex];
    if (!item) return;
    socket.emit('quiz_question_answered', {
        quiz_set: quizSetSelection,
        round_number: quizRoundNumber,
        question_id: item.id,
        question_position: quizCurrentIndex + 1,
        selected_index: selectedIdx,
        correct_index: Number(item.correct_answer),
    });
    const buttons = document.querySelectorAll('.quiz-option');
    buttons.forEach((button) => {
        button.disabled = true;
    });
    setTimeout(() => {
        if (!quizRoundActive) return;
        quizCurrentIndex += 1;
        renderQuizQuestion();
    }, 400);
}

function endQuizAncSession() {
    if (!quizAncActive) return;
    socket.emit('quiz_session_done');
    quizAncActive = false;
}

// ============================================================================
// Loading/Indicator Functions
// ============================================================================

function showLoadingIndicator(label = 'Searching', mode = 'inferring', audioDelayMs = 0) {
    if (skippedIndicatorTimer) {
        clearTimeout(skippedIndicatorTimer);
        skippedIndicatorTimer = null;
    }
    document.getElementById('tapDot1').style.display = 'none';
    document.getElementById('tapDot2').style.display = 'none';
    const indicator = document.getElementById('loadingIndicator');
    indicator.classList.remove('summarizing', 'skipped');
    if (mode === 'inferring') {
        hideSummaryText();
        hideReconstructedTurns();
    }
    if (mode === 'summarizing') {
        indicator.classList.add('summarizing');
    }
    document.querySelector('#loadingIndicator .label').textContent = label;
    indicator.classList.add('active');
    startLoadingAudioFeedback(mode, audioDelayMs);
}

function normalizeSkipMessage(message) {
    const text = (message || '').trim();
    switch (text) {
        case 'Summary unavailable':
            return 'Summary ready shortly';
        case 'No summary detected':
            return 'All caught up';
        case 'Summary audio failed':
            return 'Summary ready in text';
        case 'Summary canceled':
            return 'Summary skipped';
        case 'Summary not needed':
            return 'All caught up';
        case 'No keywords found':
            return 'No new keywords this round';
        default:
            return text || 'All caught up';
    }
}

function hideLoadingIndicator() {
    const indicator = document.getElementById('loadingIndicator');
    indicator.classList.remove('active', 'summarizing', 'skipped');
    stopLoadingAudioFeedback();
}

function showSkippedIndicator(message = 'All caught up') {
    return showSkippedIndicatorWithOptions(message, { playCompleteFeedback: true });
}

function showSkippedIndicatorWithOptions(message = 'All caught up', opts = {}) {
    if (isKeywordPlaybackBusy()) {
        pendingSkippedIndicator = { message, opts };
        return;
    }

    const shouldPlayComplete = opts.playCompleteFeedback !== false;
    const indicator = document.getElementById('loadingIndicator');
    indicator.classList.remove('summarizing');
    indicator.classList.add('active', 'skipped');
    document.querySelector('#loadingIndicator .label').textContent = normalizeSkipMessage(message);
    if (shouldPlayComplete) {
        playCompleteFeedback();
    }
    if (skippedIndicatorTimer) clearTimeout(skippedIndicatorTimer);
    skippedIndicatorTimer = setTimeout(() => {
        skippedIndicatorTimer = null;
        hideLoadingIndicator();
    }, 2000);
}

function maybeEmitPendingSkippedIndicator() {
    if (!pendingSkippedIndicator) return;
    if (isKeywordPlaybackBusy()) return;
    const pending = pendingSkippedIndicator;
    pendingSkippedIndicator = null;
    showSkippedIndicatorWithOptions(pending.message, pending.opts || {});
}

function getExpectedSummaryClientCount() {
    return Math.max(1, Number(configuredSummaryClientCount) || 1);
}

function isSummaryCycleActive() {
    return (
        summaryRequested ||
        summaryInProgress ||
        pendingSummarizeIndicatorAfterKeyword ||
        awaitingJudgeDecision
    );
}

function markSummaryDoneForSegment(segmentId, hasText) {
    if (segmentId <= 0) return;
    const prev = summarySegmentState.get(segmentId) || { received: 0, nonEmpty: 0 };
    prev.received += 1;
    if (hasText) prev.nonEmpty += 1;
    summarySegmentState.set(segmentId, prev);
}

function hasAllSummariesEmpty(segmentId) {
    if (segmentId <= 0) return false;
    const state = summarySegmentState.get(segmentId);
    if (!state) return false;
    return state.received >= getExpectedSummaryClientCount() && state.nonEmpty === 0;
}

function isKeywordPlaybackBusy() {
    if (keywordTtsPlaying) return true;
    if (activeBrowserTtsType === 'keyword') return true;
    if (ttsQueue.some(item => item && item.meta && item.meta.type === 'keyword')) return true;
    if (streamingTtsQueue.some(item => item && item.ttsType === 'keyword')) return true;
    return !!(
        currentStreamingPlayer &&
        currentStreamingPlayer.meta &&
        currentStreamingPlayer.meta.type === 'keyword'
    );
}

function emitAllCaughtUpForEmptySummary(segmentId) {
    summaryRequested = false;
    summaryInProgress = false;
    allowPostFollowupTts = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    hideLoadingIndicator();
    hideSummaryText();
    hideReconstructedTurns();
    pendingSummaryTexts = [];
    clearReconstructedState();
    options = [];
    currentIdx = 0;
    infoVisible = false;
    document.getElementById('optionsList').innerHTML = '';
    showSkippedIndicatorWithOptions('All caught up!', { playCompleteFeedback: false });
    playTtsStartFeedback('summary');
    setTimeout(() => playTtsEndFeedback('summary'), 320);
    socket.emit('browser_tts_playback_done', { reason: 'all_caught_up' });
    endQuizAncSession();
    summarySegmentState.delete(segmentId);
    scheduleNextQuizRound();
}

function maybeEmitPendingEmptySummarySignal() {
    if (!pendingEmptySummarySignal) return;
    if (isKeywordPlaybackBusy()) return;
    const pending = pendingEmptySummarySignal;
    pendingEmptySummarySignal = null;
    emitAllCaughtUpForEmptySummary(pending.segmentId);
}

function queueEmptySummarySignal(segmentId) {
    pendingEmptySummarySignal = { segmentId };
    maybeEmitPendingEmptySummarySignal();
}

function showTapIndicator() {
    const dot1 = document.getElementById('tapDot1');
    const dot2 = document.getElementById('tapDot2');
    if (!dot1) return;
    dot1.style.display = '';
    dot2.style.display = '';
    dot1.classList.remove('show');
    dot2.classList.remove('show');
    void dot1.offsetWidth;
    dot1.classList.add('show');
}

function showDoubletapIndicator() {
    const dot1 = document.getElementById('tapDot1');
    const dot2 = document.getElementById('tapDot2');
    if (!dot1 || !dot2) return;
    dot1.style.display = '';
    dot2.style.display = '';
    dot1.classList.remove('show');
    dot2.classList.remove('show');
    void dot1.offsetWidth;
    dot1.classList.add('show');
    dot2.classList.add('show');
}

// ============================================================================
// Summary/Reconstruction Functions
// ============================================================================

function hideSummary() {
    if (summaryTimer) {
        clearTimeout(summaryTimer);
        summaryTimer = null;
    }
    hideSummaryText();
    hideReconstructedTurns();
    pendingSummaryTexts = [];
}

function showSummaryText() {
    if (!summaryTextAllowedThisPlayback) return;
    if (!showSummaryTextEnabled || pendingSummaryTexts.length === 0) return;
    const container = document.getElementById('summaryBubbles');
    const bubble1 = document.getElementById('summaryBubble1');
    const bubble2 = document.getElementById('summaryBubble2');
    if (!container || !bubble1 || !bubble2) return;

    if (pendingSummaryTexts[0]) {
        bubble1.textContent = pendingSummaryTexts[0];
        bubble1.classList.add('active');
    } else {
        bubble1.classList.remove('active');
    }

    if (pendingSummaryTexts[1]) {
        bubble2.textContent = pendingSummaryTexts[1];
        bubble2.classList.add('active');
    } else {
        bubble2.classList.remove('active');
    }

    container.classList.add('active');
}

function hideSummaryText(resetPlaybackFlag = true) {
    if (resetPlaybackFlag) {
        summaryTextAllowedThisPlayback = false;
    }
    const container = document.getElementById('summaryBubbles');
    const bubble1 = document.getElementById('summaryBubble1');
    const bubble2 = document.getElementById('summaryBubble2');
    if (container) container.classList.remove('active');
    if (bubble1) {
        bubble1.textContent = '';
        bubble1.classList.remove('active');
    }
    if (bubble2) {
        bubble2.textContent = '';
        bubble2.classList.remove('active');
    }
}

function renderReconstructedTurns(turns) {
    if (!showSummaryTextEnabled) {
        const container = document.getElementById('reconstructedTurns');
        if (container) {
            container.classList.remove('active');
            container.innerHTML = '';
        }
        return;
    }
    // Keep UI text hidden unless explicitly allowed for summary text mode.
    if (!summaryTextAllowedThisPlayback) {
        const container = document.getElementById('reconstructedTurns');
        if (container) {
            container.classList.remove('active');
            container.innerHTML = '';
        }
        return;
    }
    const html = buildReconstructedTurnsHtml(turns);
    const container = document.getElementById('reconstructedTurns');
    if (!container) return;
    if (!html) {
        container.classList.remove('active');
        container.innerHTML = '';
        return;
    }
    container.innerHTML = html;
    container.classList.add('active');
}

function buildReconstructedTurnsHtml(turns) {
    if (!Array.isArray(turns) || turns.length === 0) return '';
    return turns.map(turn => {
        const speaker = (turn.speaker || '').toUpperCase() === 'B' ? 'B' : 'A';
        const cls = speaker === 'A' ? 'a' : 'b';
        const rawText = (turn.text || '').trim();
        const text = rawText.replace(/^\s*[AB]\s*:\s*/i, '').trim();
        if (!text) return '';
        return `<div class="reconstructed-turn ${cls}">${text}</div>`;
    }).join('');
}

function renderReconstructedStack(baseTurns, groups) {
    if (!showSummaryTextEnabled) {
        const container = document.getElementById('reconstructedTurns');
        if (container) {
            container.classList.remove('active');
            container.innerHTML = '';
        }
        return;
    }
    if (!summaryTextAllowedThisPlayback) {
        const container = document.getElementById('reconstructedTurns');
        if (container) {
            container.classList.remove('active');
            container.innerHTML = '';
        }
        return;
    }
    const container = document.getElementById('reconstructedTurns');
    if (!container) return;

    const baseHtml = buildReconstructedTurnsHtml(baseTurns);
    const followupItems = [];
    const safeGroups = Array.isArray(groups) ? groups : [];
    for (const group of safeGroups) {
        const turns = Array.isArray(group?.turns) ? group.turns : [];
        const html = buildReconstructedTurnsHtml(turns);
        if (!html) continue;
        followupItems.push(html);
    }

    if (!baseHtml && followupItems.length === 0) {
        container.classList.remove('active');
        container.innerHTML = '';
        return;
    }

    let html = '';
    if (baseHtml) {
        html += `<div class="reconstructed-group">${baseHtml}</div>`;
    }
    for (const groupHtml of followupItems) {
        html += `<div class="reconstructed-group">${groupHtml}</div>`;
    }
    container.innerHTML = html;
    container.classList.add('active');
}

function hideReconstructedTurns() {
    renderReconstructedTurns([]);
}

function clearReconstructedState() {
    pendingReconstructedTurns = [];
    pendingReconstructedSegmentId = 0;
    renderedReconstructedSegmentId = 0;
    baseReconstructedTurns = [];
    baseReconstructedSegmentId = 0;
    followupReconstructedGroups = [];
}

function upsertFollowupReconstructedGroup(segmentId, turns) {
    const idx = followupReconstructedGroups.findIndex(g => Number(g.segmentId) === Number(segmentId));
    const payload = { segmentId: Number(segmentId) || 0, turns: Array.isArray(turns) ? turns : [] };
    if (idx >= 0) {
        followupReconstructedGroups[idx] = payload;
    } else {
        followupReconstructedGroups.push(payload);
    }
}

function parseCompressedDialogueTurns(text) {
    const normalized = String(text || '').replace(/\\n/g, '\n');
    const lines = normalized
        .split('\n')
        .map(line => line.trim())
        .filter(Boolean);
    const turns = [];
    for (const line of lines) {
        const match = line.match(/^(?:speaker\s*)?([AB])\s*[:\-–—]\s*(.+)$/i);
        if (!match) continue;
        const speaker = match[1].toUpperCase();
        const utterance = (match[2] || '').trim();
        if (!utterance) continue;
        turns.push({ speaker, text: utterance });
        if (turns.length >= 3) break;
    }
    return turns;
}

function showDismissIndicator() {
    const el = document.getElementById('dismissIndicator');
    if (!el) return;
    if (dismissTimer) clearTimeout(dismissTimer);
    el.classList.remove('fade-out');
    el.classList.add('show');
    dismissTimer = setTimeout(() => {
        el.classList.add('fade-out');
        setTimeout(() => {
            el.classList.remove('show', 'fade-out');
        }, 200);
    }, 1500);
}

// ============================================================================
// Session Control Functions
// ============================================================================

function start(v) {
    view = v;
    document.getElementById('menu').style.display = 'none';
    document.getElementById('main-content').style.display = 'block';

    stopAllMicMonitors();
    stopNoiseGateMonitor();
    summarySegmentState.clear();
    pendingEmptySummarySignal = null;
    pendingSkippedIndicator = null;
    fastCatchupPending = false;
    clearQuizTimers();
    clearQueuedQuizRound();
    quizBank = [];
    sessionQuizOrder = [];
    quizRoundNumber = 0;
    quizScheduleIndex = 0;
    quizCurrentIndex = 0;
    quizRoundActive = false;
    quizRoundEndsAt = 0;
    quizSessionStartAt = Date.now();
    quizAncActive = false;

    const params = { source: source };
    const fastCatchupThresholdEl = document.getElementById('fastCatchupThresholdSec');
    const fastCatchupSpeedEl = document.getElementById('fastCatchupSpeed');
    const fastCatchupGapEl = document.getElementById('fastCatchupGapSec');
    let fastCatchupThresholdSec = Number(fastCatchupThresholdEl ? fastCatchupThresholdEl.value : 1);
    let fastCatchupSpeed = Number(fastCatchupSpeedEl ? fastCatchupSpeedEl.value : 1.5);
    let fastCatchupGapSec = Number(fastCatchupGapEl ? fastCatchupGapEl.value : 0.0);
    if (!Number.isFinite(fastCatchupThresholdSec)) fastCatchupThresholdSec = 1;
    if (!Number.isFinite(fastCatchupSpeed)) fastCatchupSpeed = 1.5;
    if (!Number.isFinite(fastCatchupGapSec)) fastCatchupGapSec = 0.0;
    fastCatchupThresholdSec = Math.min(30, Math.max(0, Math.round(fastCatchupThresholdSec)));  // 0 = disabled
    fastCatchupSpeed = Math.min(3.0, Math.max(1.0, Math.round(fastCatchupSpeed * 10) / 10));
    fastCatchupGapSec = Math.min(2.0, Math.max(0.0, Math.round(fastCatchupGapSec * 10) / 10));
    if (fastCatchupThresholdEl) fastCatchupThresholdEl.value = String(fastCatchupThresholdSec);
    if (fastCatchupSpeedEl) fastCatchupSpeedEl.value = fastCatchupSpeed.toFixed(1);
    if (fastCatchupGapEl) fastCatchupGapEl.value = fastCatchupGapSec.toFixed(1);
    params.fast_catchup_threshold_sec = fastCatchupThresholdSec;
    params.fast_catchup_speed = fastCatchupSpeed;
    params.fast_catchup_gap_sec = fastCatchupGapSec;
    params.fast_catchup_window_mode = fastCatchupWindowMode;
    if (source === 'mic') {
        const kw = document.getElementById('keywordMic').value;
        const s1 = document.getElementById('summaryMic1').value;
        const s2 = document.getElementById('summaryMic2').value;
        const v1 = document.getElementById('voice1').value;
        const v2 = document.getElementById('voice2').value;
        const kwVoice = document.getElementById('keywordVoice').value;
        const ttsOut = document.getElementById('ttsOutput').value;
        params.keyword_mic = kw ? parseInt(kw) : null;
        params.summary_mics = [s1, s2].filter(x => x).map(x => parseInt(x));
        configuredSummaryClientCount = params.summary_mics.length;
        params.voice_ids = [v1, v2];
        params.keyword_voice_id = kwVoice || null;
        params.tts_output_device = ttsOut ? parseInt(ttsOut) : null;
    } else if (source === 'mp3') {
        const sum0 = document.getElementById('summaryMp3_0').value.trim();
        const sum1 = document.getElementById('summaryMp3_1').value.trim();
        const keywordPath = document.getElementById('keywordMp3').value.trim();
        const v1 = document.getElementById('voice1').value;
        const v2 = document.getElementById('voice2').value;
        const kwVoice = document.getElementById('keywordVoice').value;
        const ttsOut = document.getElementById('ttsOutput').value;
        params.summary_sources = [sum0, sum1].filter(x => x);
        configuredSummaryClientCount = params.summary_sources.length;
        params.keyword_source = keywordPath || null;
        params.voice_ids = [v1, v2];
        params.keyword_voice_id = kwVoice || null;
        params.tts_output_device = ttsOut ? parseInt(ttsOut) : null;
    }

    if (noiseGateEnabled) {
        params.noise_gate = {
            enabled: true,
            threshold: noiseGateThreshold
        };
    }

    params.judge_enabled = judgeEnabled;
    params.reconstructor_enabled = reconstructorEnabled;
    params.single_keyword_mode = singleKeywordMode;
    params.transcript_compression_mode = transcriptCompressionMode;
    params.fast_catchup_chain_enabled = fastCatchupChainEnabled;
    params.summary_followup_enabled = summaryFollowupEnabled;
    params.skip_first_transcript_enabled = skipFirstTranscriptEnabled;
    params.missed_summary_latency_bridge_enabled = missedSummaryLatencyBridgeEnabled;
    params.airpods_mode_switch_enabled = airpodsModeSwitchEnabled;
    params.quiz_set = quizSetSelection || 'A';

    socket.emit('start', params);
}

function startSummarizing(opts = {}) {
    // force: bypass dismissMode check (used when dismiss triggers summarizing due to duration/transcript conditions)
    const force = !!opts.force;
    if (force || dismissMode === 'summary') {
        if (summaryRequested) return;
        if (summaryTriggeredForListeningSession) return;
        if (!listeningActive) return;
        ignoreIncomingSummaryEvents = false;
        clearKeywordAutoSummarizeTimer();
        hideReconstructedTurns();
        pendingSummaryTexts = [];
        hideSummaryText();
        summaryRequested = true;
        summaryInProgress = true;
        allowPostFollowupTts = true;
        awaitingJudgeDecision = false;
        const silentWhileKeyword = !!opts.silentWhileKeyword;
        const shouldDeferIndicatorUntilKeywordDone =
            silentWhileKeyword && autoPreSummarizeEnabled && keywordTtsPlaying;
        pendingSummarizeIndicatorAfterKeyword = shouldDeferIndicatorUntilKeywordDone;
        if (!shouldDeferIndicatorUntilKeywordDone) {
            showLoadingIndicator('Summarizing...', 'summarizing');
        }
        unlockAudioForTts();
        endListeningIfNeeded();
    }
}

function dismissKeywords() {
    options = [];
    currentIdx = 0;
    infoVisible = false;
    document.getElementById('optionsList').innerHTML = '';
    // Clear resumed playback flag if set
    ttsResumedPlayback = false;
    // Don't auto-summarize when dismissing - just clear keywords
}

function clearAllActiveUiAndAudio() {
    stopTtsPlayback();
    cancelKeywordTts();
    clearKeywordAutoSummarizeTimer();
    if (inferencingTimer) {
        clearTimeout(inferencingTimer);
        inferencingTimer = null;
    }
    if (summaryFinalizeTimer) {
        clearTimeout(summaryFinalizeTimer);
        summaryFinalizeTimer = null;
    }
    stopLoadingAudioFeedback();

    options = [];
    currentIdx = 0;
    infoVisible = false;
    summaryRequested = false;
    summaryInProgress = false;
    pendingSummarizeIndicatorAfterKeyword = false;
    awaitingJudgeDecision = false;
    listeningActive = false;
    summaryTriggeredForListeningSession = false;
    ignoreIncomingSummaryEvents = true;
    keywordTtsPlaying = false;
    keywordTtsPlaybackStartTime = 0;
    keywordTtsCurrentText = '';
    pendingSummaryTexts = [];
    clearQuizTimers();
    clearQueuedQuizRound();
    quizRoundActive = false;
    quizRoundEndsAt = 0;
    endQuizAncSession();
    clearReconstructedState();
    summaryCueSequenceActive = false;
    lastSpace = 0;

    document.getElementById('optionsList').innerHTML = '';
    hideLoadingIndicator();
    hideSummaryText();
    hideReconstructedTurns();
    document.getElementById('tapDot1').classList.remove('show');
    document.getElementById('tapDot2').classList.remove('show');
    document.getElementById('tapDot1').style.display = '';
    document.getElementById('tapDot2').style.display = '';
}

// ============================================================================
// Input Handling
// ============================================================================
