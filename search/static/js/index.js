const audio = document.getElementById('audio-player');
const fileSelect = document.getElementById('file-select');
const timeDisplay = document.getElementById('time-display');
const transcriptDiv = document.getElementById('transcript');
const jumpSecondsInput = document.getElementById('jump-seconds');
const toggleTranscriptBtn = document.getElementById('toggle-transcript');
const replaySpeedupEnabled = document.getElementById('replay-speedup-enabled');
const replaySpeedupRate = document.getElementById('replay-speedup-rate');

const studyPanel = document.getElementById('study-panel');
const studyParticipant = document.getElementById('study-participant');
const studyFeature = document.getElementById('study-feature');
const studyAudio = document.getElementById('study-audio');
const btnStartStudy = document.getElementById('btn-start-study');
const btnStopStudy = document.getElementById('btn-stop-study');
const interruptionOverlay = document.getElementById('interruption-overlay');
const targetWordDisplay = document.getElementById('target-word-display');
const taskNumberDisplay = document.getElementById('task-number');
const taskTotalDisplay = document.getElementById('task-total');
const taskTimerDisplay = document.getElementById('task-timer');
const feedbackOverlay = document.getElementById('feedback-overlay');
const feedbackContent = document.getElementById('feedback-content');
const prepOverlay = document.getElementById('prep-overlay');
const prepCountDisplay = document.getElementById('prep-count-display');

const modeButtons = {
    discontinuous: document.getElementById('mode-discontinuous'),
    keyword: document.getElementById('mode-keyword'),
    keyword2: document.getElementById('mode-keyword2'),
    word: document.getElementById('mode-word'),
    sentence: document.getElementById('mode-sentence'),
};

let currentMode = 'discontinuous';
let transcript = null;
let customKeywords = [];
let customKeywords2 = [];
let allWords = [];
let navigableWords = [];
let sentenceUnits = [];
let currentFilename = null;
let currentVideoId = null;

let keywordIndex = -1;
let keyword2Index = -1;
let wordIndex = -1;
let sentenceIndex = -1;
let navigationAnchorPending = {
    keyword: false,
    keyword2: false,
    word: false,
    sentence: false,
};

let isSpeedupReplay = false;
let speedupReplayAudio = null;

let isWordBackwardMode = false;
let currentWordEndTime = 0;

let studyMode = false;
let studyConfig = null;
let studyInterruptions = null;
let currentTaskIndex = 0;
let taskActive = false;
let taskStartTime = null;
let taskTimerInterval = null;
let sessionStartTime = null;
let sessionTasks = [];
let activeTargetTime = null;
let maxPlayedTime = 0;
let suppressNextSeekLog = false;
let lastLoggedAudioTime = 0;
let prepCountdownInterval = null;
let prepCountdownTimeout = null;

const FREQ_SKIP_THRESHOLD = 4.0;

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

function getVideoIdFromAudioFilename(audioFile) {
    if (!audioFile) return null;
    const dotIdx = audioFile.lastIndexOf('.');
    return dotIdx >= 0 ? audioFile.slice(0, dotIdx) : audioFile;
}

function resetPlaybackProgressLock() {
    maxPlayedTime = 0;
}

function getMaxSeekTime() {
    return Math.max(0, maxPlayedTime);
}

function clampToPlayedTime(targetTime) {
    return Math.max(0, Math.min(targetTime, getMaxSeekTime()));
}

async function loadFiles() {
    if (!fileSelect) return;
    const res = await fetch('/api/files');
    const files = await res.json();
    files.forEach((f) => {
        const opt = document.createElement('option');
        opt.value = JSON.stringify(f);
        opt.textContent = f.filename;
        fileSelect.appendChild(opt);
    });
}

async function loadTranscript(videoId) {
    currentVideoId = videoId;
    const res = await fetch(`/api/transcript/${videoId}`);
    transcript = await res.json();
    customKeywords = transcript.custom_keywords || [];
    customKeywords2 = transcript.custom_keywords2 || [];

    allWords = [];
    if (transcript.segments) {
        transcript.segments.forEach((seg) => {
            if (seg.words) {
                seg.words.forEach((w) => {
                    allWords.push({
                        word: w.word.trim(),
                        start: w.start,
                        end: w.end,
                        freq: w.freq !== undefined ? w.freq : -1,
                    });
                });
            }
        });
    }
    navigableWords = allWords.filter((w) => !shouldSkipWord(w));
    sentenceUnits = transcript.sentences || buildSentenceUnits();

    resetAllIndices();
    syncNavigationIndices(audio.currentTime || 0);
    renderTranscript();
}

function resetAllIndices() {
    keywordIndex = -1;
    keyword2Index = -1;
    wordIndex = -1;
    sentenceIndex = -1;
    navigationAnchorPending = {
        keyword: false,
        keyword2: false,
        word: false,
        sentence: false,
    };
}

function shouldSkipWord(wordObj) {
    const normalized = wordObj.word.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normalized.length <= 2) return true;
    if (wordObj.freq >= FREQ_SKIP_THRESHOLD) return true;
    return false;
}

function buildSentenceUnits() {
    if (!transcript || !transcript.segments) return [];

    const units = [];
    let currentWords = [];
    let currentStart = null;
    let fallbackSegmentStart = null;
    let fallbackSegmentEnd = null;

    const flushSentence = () => {
        if (currentWords.length === 0) return;

        const lastWord = currentWords[currentWords.length - 1];
        units.push({
            text: currentWords.map((w) => w.word).join(' ').trim(),
            start: currentStart ?? fallbackSegmentStart ?? 0,
            end: lastWord.end ?? fallbackSegmentEnd ?? currentStart ?? 0,
        });

        currentWords = [];
        currentStart = null;
        fallbackSegmentStart = null;
        fallbackSegmentEnd = null;
    };

    transcript.segments.forEach((seg) => {
        const segmentWords = seg.words || [];

        if (segmentWords.length === 0) {
            flushSentence();
            units.push({
                text: seg.text,
                start: seg.start,
                end: seg.end,
            });
            return;
        }

        segmentWords.forEach((word) => {
            if (currentWords.length === 0) {
                currentStart = word.start;
                fallbackSegmentStart = seg.start;
            }

            fallbackSegmentEnd = seg.end;
            currentWords.push(word);

            if (word.word.includes('.')) {
                flushSentence();
            }
        });
    });

    flushSentence();
    return units;
}

function findNearestIndex(items, targetTime, getTime) {
    if (!items || items.length === 0) return -1;

    let bestIndex = 0;
    let bestDistance = Math.abs(getTime(items[0]) - targetTime);

    for (let i = 1; i < items.length; i += 1) {
        const distance = Math.abs(getTime(items[i]) - targetTime);
        if (distance < bestDistance) {
            bestDistance = distance;
            bestIndex = i;
        }
    }

    return bestIndex;
}

function syncNavigationIndices(targetTime = audio.currentTime) {
    keywordIndex = findNearestIndex(customKeywords, targetTime, (item) => item.time);
    keyword2Index = findNearestIndex(customKeywords2, targetTime, (item) => item.time);
    wordIndex = findNearestIndex(navigableWords, targetTime, (item) => item.start);
    sentenceIndex = findNearestIndex(sentenceUnits, targetTime, (item) => item.start);
    navigationAnchorPending = {
        keyword: keywordIndex >= 0,
        keyword2: keyword2Index >= 0,
        word: wordIndex >= 0,
        sentence: sentenceIndex >= 0,
    };
}

function getIndexedTarget(items, currentIndex, currentTime, getTime, direction, useAnchor = false) {
    if (!items || items.length === 0) return { targetIndex: -1, target: null };

    const safeIndex = currentIndex >= 0 ? currentIndex : findNearestIndex(items, currentTime, getTime);
    if (safeIndex < 0) return { targetIndex: -1, target: null };

    let targetIndex = safeIndex;
    if (!useAnchor) {
        if (direction < 0) {
            targetIndex = Math.max(0, safeIndex - 1);
        } else if (direction > 0) {
            targetIndex = Math.min(items.length - 1, safeIndex + 1);
        }
    }

    return {
        targetIndex,
        target: items[targetIndex],
    };
}

function renderTranscript() {
    if (!transcript) return;

    const jumpSeconds = jumpSecondsInput ? (parseInt(jumpSecondsInput.value, 10) || 15) : 15;

    if (currentMode === 'discontinuous') {
        const blocks = [];

        transcript.segments.forEach((seg) => {
            const blockIndex = Math.floor(seg.start / jumpSeconds);
            const blockStart = blockIndex * jumpSeconds;

            if (blocks.length === 0 || blocks[blocks.length - 1].start !== blockStart) {
                blocks.push({ start: blockStart, end: blockStart + jumpSeconds, texts: [seg.text] });
            } else {
                blocks[blocks.length - 1].texts.push(seg.text);
            }
        });

        transcriptDiv.innerHTML = blocks.map((block, i) => (
            `<div class="segment" data-idx="${i}" data-start="${block.start}" data-end="${block.end}">
                <span class="time">[${formatTime(block.start)}]</span> ${block.texts.join(' ')}
            </div>`
        )).join('');
    } else if (currentMode === 'keyword' || currentMode === 'keyword2') {
        let fullText = transcript.segments.map((seg) => seg.text).join(' ');
        const keywords = currentMode === 'keyword' ? customKeywords : customKeywords2;

        keywords.forEach((kw) => {
            const escaped = kw.word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            const regex = new RegExp(`\\b(${escaped})\\b`, 'gi');
            fullText = fullText.replace(regex, '<span class="kw">$1</span>');
        });

        transcriptDiv.innerHTML = `<div class="segment continuous">${fullText}</div>`;
    } else if (currentMode === 'word') {
        let html = '<div class="segment continuous">';
        allWords.forEach((w, i) => {
            const skip = shouldSkipWord(w);
            const cls = skip ? '' : 'class="kw"';
            html += `<span ${cls} data-word-idx="${i}" data-start="${w.start}">${w.word} </span>`;
        });
        html += '</div>';
        transcriptDiv.innerHTML = html;
    } else if (currentMode === 'sentence') {
        transcriptDiv.innerHTML = sentenceUnits.map((sentence, i) => (
            `<div class="segment" data-seg-idx="${i}" data-start="${sentence.start}" data-end="${sentence.end}">
                <span class="time">[${formatTime(sentence.start)}]</span> ${sentence.text}
            </div>`
        )).join('');
    }
}

function updateCurrentSegment() {
    const currentTime = audio.currentTime;

    if (currentMode === 'discontinuous' || currentMode === 'sentence') {
        document.querySelectorAll('#transcript .segment').forEach((el) => {
            const start = parseFloat(el.dataset.start);
            const end = parseFloat(el.dataset.end);
            if (currentTime >= start && currentTime < end) {
                el.classList.add('current');
                el.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                el.classList.remove('current');
            }
        });
    }

    let suffix = '';
    if (isSpeedupReplay) {
        const rate = replaySpeedupRate ? replaySpeedupRate.value : '1.5';
        suffix = ` [${rate}x]`;
    }
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audio.duration || 0)}${suffix}`;
}

function setMode(mode) {
    currentMode = mode;
    Object.keys(modeButtons).forEach((m) => {
        if (modeButtons[m]) {
            modeButtons[m].classList.toggle('active', m === mode);
        }
    });

    const jumpBackSetting = document.getElementById('jump-back-setting');
    if (jumpBackSetting) {
        jumpBackSetting.style.display = mode === 'discontinuous' ? 'block' : 'none';
    }

    stopSpeedupReplay();
    stopWordBackwardMode();
    resetAllIndices();
    syncNavigationIndices(activeTargetTime ?? audio.currentTime ?? 0);
    renderTranscript();
}

function stopWordBackwardMode() {
    isWordBackwardMode = false;
    currentWordEndTime = 0;
}

function startSpeedupReplay() {
    if (replaySpeedupEnabled && !replaySpeedupEnabled.checked) return;
    const rate = replaySpeedupRate ? (parseFloat(replaySpeedupRate.value) || 1.5) : 1.5;
    audio.playbackRate = rate;
    isSpeedupReplay = true;
}

function stopSpeedupReplay() {
    audio.playbackRate = 1.0;
    isSpeedupReplay = false;
}

function applySearchPlaybackRate() {
    if (taskActive) {
        startSpeedupReplay();
    } else {
        stopSpeedupReplay();
    }
}

function jumpBack() {
    const seconds = jumpSecondsInput ? (parseInt(jumpSecondsInput.value, 10) || 15) : 15;
    setAudioTime(audio.currentTime - seconds, 'jump_back', { seconds });
    applySearchPlaybackRate();
    audio.play();
}

function jumpForward() {
    const seconds = jumpSecondsInput ? (parseInt(jumpSecondsInput.value, 10) || 15) : 15;
    setAudioTime(audio.currentTime + seconds, 'jump_forward', { seconds });
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousKeyword(useKeyword2 = false) {
    const keywords = useKeyword2 ? customKeywords2 : customKeywords;
    const anchorKey = useKeyword2 ? 'keyword2' : 'keyword';

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    const activeIndex = useKeyword2 ? keyword2Index : keywordIndex;
    const { targetIndex, target } = getIndexedTarget(
        keywords,
        activeIndex,
        currentTime,
        (item) => item.time,
        -1,
        navigationAnchorPending[anchorKey],
    );
    if (!target) return;

    navigationAnchorPending[anchorKey] = false;
    if (useKeyword2) keyword2Index = targetIndex;
    else keywordIndex = targetIndex;

    setAudioTime(target.time - 0.5, useKeyword2 ? 'keyword2_prev' : 'keyword_prev', {
        keyword: target.word,
        keywordTime: target.time,
        keywordIndex: targetIndex,
    });
    applySearchPlaybackRate();
    audio.play();
}

function jumpToNextKeyword(useKeyword2 = false) {
    const keywords = useKeyword2 ? customKeywords2 : customKeywords;
    const anchorKey = useKeyword2 ? 'keyword2' : 'keyword';

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    const activeIndex = useKeyword2 ? keyword2Index : keywordIndex;
    const { targetIndex, target } = getIndexedTarget(
        keywords,
        activeIndex,
        currentTime,
        (item) => item.time,
        1,
        navigationAnchorPending[anchorKey],
    );
    if (!target || target.time > getMaxSeekTime() + 0.5) return;

    navigationAnchorPending[anchorKey] = false;
    setAudioTime(target.time - 0.5, useKeyword2 ? 'keyword2_next' : 'keyword_next', {
        keyword: target.word,
        keywordTime: target.time,
        keywordIndex: targetIndex,
    });
    if (useKeyword2) keyword2Index = targetIndex;
    else keywordIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousWord() {
    if (navigableWords.length === 0) return;

    const { targetIndex, target } = getIndexedTarget(
        navigableWords,
        wordIndex,
        audio.currentTime,
        (item) => item.start,
        -1,
        navigationAnchorPending.word,
    );
    if (!target) return;

    navigationAnchorPending.word = false;
    wordIndex = targetIndex;
    setAudioTime(target.start, 'word_prev', {
        word: target.word,
        wordStart: target.start,
        wordIndex: targetIndex,
    });
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousWordAuto() {
    if (navigableWords.length === 0 || wordIndex <= 0) {
        stopWordBackwardMode();
        return;
    }

    wordIndex -= 1;
    const target = navigableWords[wordIndex];
    setAudioTime(target.start, 'word_prev_auto', {
        word: target.word,
        wordStart: target.start,
        wordIndex,
    });
    currentWordEndTime = target.end;
}

function jumpToNextWord() {
    if (navigableWords.length === 0) return;

    stopWordBackwardMode();

    const { targetIndex, target } = getIndexedTarget(
        navigableWords,
        wordIndex,
        audio.currentTime,
        (item) => item.start,
        1,
        navigationAnchorPending.word,
    );
    if (!target || target.start > getMaxSeekTime()) return;

    navigationAnchorPending.word = false;
    setAudioTime(target.start, 'word_next', {
        word: target.word,
        wordStart: target.start,
        wordIndex: targetIndex,
    });
    wordIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousSentence() {
    if (sentenceUnits.length === 0) return;

    const { targetIndex, target } = getIndexedTarget(
        sentenceUnits,
        sentenceIndex,
        audio.currentTime,
        (item) => item.start,
        -1,
        navigationAnchorPending.sentence,
    );
    if (!target) return;

    navigationAnchorPending.sentence = false;
    sentenceIndex = targetIndex;
    setAudioTime(target.start, 'sentence_prev', {
        sentenceIndex: targetIndex,
        sentenceStart: target.start,
    });
    applySearchPlaybackRate();
    audio.play();
}

function jumpToNextSentence() {
    if (sentenceUnits.length === 0) return;

    const { targetIndex, target } = getIndexedTarget(
        sentenceUnits,
        sentenceIndex,
        audio.currentTime,
        (item) => item.start,
        1,
        navigationAnchorPending.sentence,
    );
    if (!target || target.start > getMaxSeekTime()) return;

    navigationAnchorPending.sentence = false;
    setAudioTime(target.start, 'sentence_next', {
        sentenceIndex: targetIndex,
        sentenceStart: target.start,
    });
    sentenceIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

async function loadStudyConfig() {
    try {
        const res = await fetch('/api/study/config');
        if (!res.ok) {
            console.error('Study config not found');
            return;
        }
        studyConfig = await res.json();

        studyFeature.innerHTML = '<option value="">-- Select Feature --</option>';
        studyConfig.features.forEach((f) => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f.charAt(0).toUpperCase() + f.slice(1);
            studyFeature.appendChild(opt);
        });

        studyAudio.innerHTML = '<option value="">-- Select Audio --</option>';
        studyConfig.audio_files.forEach((f) => {
            const opt = document.createElement('option');
            opt.value = f;
            opt.textContent = f;
            studyAudio.appendChild(opt);
        });
    } catch (err) {
        console.error('Failed to load study config:', err);
    }
}

async function logStudyEvent(event, data = {}) {
    try {
        await fetch('/api/study/log', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                participant: studyParticipant.value,
                event,
                ...data,
            }),
        });
    } catch (err) {
        console.error('Failed to log event:', err);
    }
}

function logNavigationEvent(action, fromTime, toTime, extra = {}) {
    if (!studyMode) return;
    logStudyEvent('navigation', {
        action,
        mode: currentMode,
        fromTime,
        toTime,
        taskActive,
        playbackRate: audio.playbackRate,
        ...extra,
    });
}

function setAudioTime(targetTime, action, extra = {}) {
    const fromTime = audio.currentTime;
    const toTime = clampToPlayedTime(targetTime);
    suppressNextSeekLog = true;
    audio.currentTime = toTime;
    logNavigationEvent(action, fromTime, toTime, extra);
    return toTime;
}

async function startStudySession() {
    const participant = studyParticipant.value.trim();
    const feature = studyFeature.value;
    const audioFile = studyAudio.value;

    if (!participant || !feature || !audioFile) {
        alert('Please fill in all fields (Participant, Feature, Audio)');
        return;
    }

    const videoId = getVideoIdFromAudioFilename(audioFile);

    try {
        const res = await fetch(`/api/study/interruptions/${videoId}`);
        if (!res.ok) {
            alert(`Interruptions config not found for ${videoId}`);
            return;
        }
        studyInterruptions = await res.json();
    } catch (err) {
        alert('Failed to load interruptions config');
        return;
    }

    studyMode = true;
    currentTaskIndex = 0;
    taskActive = false;
    sessionTasks = [];
    sessionStartTime = Date.now();

    setMode(feature);

    currentFilename = audioFile;
    audio.src = `/mp3/${audioFile}`;
    resetPlaybackProgressLock();
    await loadTranscript(videoId);

    document.body.classList.add('study-active');
    btnStartStudy.style.display = 'none';
    btnStopStudy.style.display = 'inline-block';
    taskTotalDisplay.textContent = studyInterruptions.interruptions.length;

    await logStudyEvent('session_start', {
        feature,
        audio: audioFile,
        videoId,
        totalTasks: studyInterruptions.interruptions.length,
    });

    audio.play();
}

async function stopStudySession() {
    activeTargetTime = null;
    if (taskTimerInterval) {
        clearInterval(taskTimerInterval);
        taskTimerInterval = null;
    }
    if (prepCountdownInterval) {
        clearInterval(prepCountdownInterval);
        prepCountdownInterval = null;
    }
    if (prepCountdownTimeout) {
        clearTimeout(prepCountdownTimeout);
        prepCountdownTimeout = null;
    }

    interruptionOverlay.classList.remove('active');
    feedbackOverlay.classList.remove('active', 'success', 'failure');
    prepOverlay.classList.remove('active');
    audio.pause();

    const completed = sessionTasks.filter((t) => t.outcome === 'spacebar').length;
    const timeouts = sessionTasks.filter((t) => t.outcome === 'timeout').length;

    await logStudyEvent('session_complete', {
        totalTasks: studyInterruptions ? studyInterruptions.interruptions.length : 0,
        completed,
        timeouts,
        duration: Date.now() - sessionStartTime,
    });

    await fetch('/api/study/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            participant: studyParticipant.value,
            feature: studyFeature.value,
            audio: studyAudio.value,
            sessionStartTime: new Date(sessionStartTime).toISOString(),
            sessionEndTime: new Date().toISOString(),
            totalTasks: studyInterruptions ? studyInterruptions.interruptions.length : 0,
            completed,
            timeouts,
            tasks: sessionTasks,
        }),
    });

    studyMode = false;
    taskActive = false;
    studyInterruptions = null;

    document.body.classList.remove('study-active');
    btnStartStudy.style.display = 'inline-block';
    btnStopStudy.style.display = 'none';
}

function checkForInterruption() {
    if (!studyMode || taskActive || !studyInterruptions) return;
    if (currentTaskIndex >= studyInterruptions.interruptions.length) return;

    const currentInterruption = studyInterruptions.interruptions[currentTaskIndex];
    const triggerTime = currentInterruption.target_word_time + currentInterruption.delay_seconds;

    if (audio.currentTime >= triggerTime) {
        triggerInterruption(currentInterruption, triggerTime);
    }
}

async function triggerInterruption(interruption, triggerTime) {
    taskActive = true;
    taskStartTime = Date.now();
    activeTargetTime = audio.currentTime;

    stopSpeedupReplay();
    audio.pause();
    syncNavigationIndices(activeTargetTime);

    taskNumberDisplay.textContent = currentTaskIndex + 1;
    targetWordDisplay.textContent = interruption.target_word;
    taskTimerDisplay.textContent = studyConfig.task_timeout_seconds;

    interruptionOverlay.classList.add('active');

    await logStudyEvent('interruption_triggered', {
        taskIndex: currentTaskIndex + 1,
        targetWord: interruption.target_word,
        targetWordTime: interruption.target_word_time,
        triggerTime,
    });

    let remaining = studyConfig.task_timeout_seconds;
    taskTimerInterval = setInterval(() => {
        remaining -= 1;
        taskTimerDisplay.textContent = remaining;

        if (remaining <= 0) {
            clearInterval(taskTimerInterval);
            taskTimerInterval = null;
            handleTaskTimeout(interruption, triggerTime);
        }
    }, 1000);
}

async function handleSpacebarConfirmation() {
    if (!taskActive || !studyInterruptions) return;

    if (taskTimerInterval) {
        clearInterval(taskTimerInterval);
        taskTimerInterval = null;
    }

    stopSpeedupReplay();
    audio.pause();

    const interruption = studyInterruptions.interruptions[currentTaskIndex];
    const responseTimeMs = Date.now() - taskStartTime;
    const userPosition = audio.currentTime;
    const targetTime = interruption.target_word_time;
    const distanceSeconds = userPosition - targetTime;

    const taskResult = {
        taskIndex: currentTaskIndex + 1,
        targetWord: interruption.target_word,
        targetTime,
        userPosition,
        distanceSeconds,
        responseTimeMs,
        outcome: 'spacebar',
    };
    sessionTasks.push(taskResult);

    await logStudyEvent('task_completed', taskResult);

    interruptionOverlay.classList.remove('active');
    feedbackOverlay.classList.remove('active', 'success', 'failure');
    resumeAfterTask(interruption);
}

async function handleTaskTimeout(interruption, triggerTime) {
    const taskResult = {
        taskIndex: currentTaskIndex + 1,
        targetWord: interruption.target_word,
        targetTime: interruption.target_word_time,
        triggerTime,
        outcome: 'timeout',
    };
    sessionTasks.push(taskResult);

    await logStudyEvent('task_timeout', taskResult);

    stopSpeedupReplay();
    interruptionOverlay.classList.remove('active');
    showFeedback(false, interruption, taskResult, true);
}

function showFeedback(success, interruption, taskResult, isTimeout = false) {
    feedbackOverlay.classList.remove('success', 'failure');

    if (success) {
        feedbackContent.textContent = 'Good!';
        feedbackOverlay.classList.add('success');
    } else {
        feedbackContent.textContent = isTimeout ? 'Time Up!' : 'Try Again';
        feedbackOverlay.classList.add('failure');
    }

    feedbackOverlay.classList.add('active');

    setTimeout(() => {
        feedbackOverlay.classList.remove('active', 'success', 'failure');
        resumeAfterTask(interruption);
    }, studyConfig.feedback_display_ms);
}

function resumeAfterTask(interruption) {
    stopSpeedupReplay();
    taskActive = false;
    activeTargetTime = null;
    currentTaskIndex += 1;

    if (currentTaskIndex >= studyInterruptions.interruptions.length) {
        completeStudySession();
        return;
    }

    const triggerTime = interruption.target_word_time + interruption.delay_seconds;
    const resumeTime = Math.max(0, triggerTime - studyConfig.resume_offset_seconds);
    const actualResumeTime = setAudioTime(resumeTime, 'task_resume', {
        targetWord: interruption.target_word,
        triggerTime,
    });
    syncNavigationIndices(actualResumeTime);
    audio.pause();

    const prepDelaySeconds = studyConfig.prep_delay_seconds ?? 3;
    let remaining = prepDelaySeconds;
    prepCountDisplay.textContent = remaining;
    prepOverlay.classList.add('active');

    if (prepCountdownInterval) clearInterval(prepCountdownInterval);
    if (prepCountdownTimeout) clearTimeout(prepCountdownTimeout);

    prepCountdownInterval = setInterval(() => {
        remaining -= 1;
        if (remaining > 0) {
            prepCountDisplay.textContent = remaining;
        }
    }, 1000);

    prepCountdownTimeout = setTimeout(() => {
        if (prepCountdownInterval) {
            clearInterval(prepCountdownInterval);
            prepCountdownInterval = null;
        }
        prepCountdownTimeout = null;
        prepOverlay.classList.remove('active');
        if (!studyMode || taskActive) return;
        audio.play();
    }, prepDelaySeconds * 1000);
}

async function completeStudySession() {
    stopSpeedupReplay();
    activeTargetTime = null;
    if (prepCountdownInterval) {
        clearInterval(prepCountdownInterval);
        prepCountdownInterval = null;
    }
    if (prepCountdownTimeout) {
        clearTimeout(prepCountdownTimeout);
        prepCountdownTimeout = null;
    }
    prepOverlay.classList.remove('active');
    const completed = sessionTasks.filter((t) => t.outcome === 'spacebar').length;
    const timeouts = sessionTasks.filter((t) => t.outcome === 'timeout').length;

    await logStudyEvent('session_complete', {
        totalTasks: studyInterruptions.interruptions.length,
        completed,
        timeouts,
        duration: Date.now() - sessionStartTime,
    });

    await fetch('/api/study/session', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            participant: studyParticipant.value,
            feature: studyFeature.value,
            audio: studyAudio.value,
            sessionStartTime: new Date(sessionStartTime).toISOString(),
            sessionEndTime: new Date().toISOString(),
            totalTasks: studyInterruptions.interruptions.length,
            completed,
            timeouts,
            tasks: sessionTasks,
        }),
    });

    alert(`Study session complete!\n\nCompleted: ${completed}\nTimeouts: ${timeouts}`);

    studyMode = false;
    taskActive = false;
    activeTargetTime = null;

    document.body.classList.remove('study-active');
    btnStartStudy.style.display = 'inline-block';
    btnStopStudy.style.display = 'none';
}

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT') return;

    if (e.code === 'Space' && studyMode && taskActive) {
        e.preventDefault();
        handleSpacebarConfirmation();
        return;
    }

    if (e.code === 'ArrowLeft') {
        e.preventDefault();

        switch (currentMode) {
            case 'discontinuous':
                jumpBack();
                break;
            case 'keyword':
                jumpToPreviousKeyword(false);
                break;
            case 'keyword2':
                jumpToPreviousKeyword(true);
                break;
            case 'word':
                jumpToPreviousWord();
                break;
            case 'sentence':
                jumpToPreviousSentence();
                break;
            default:
                break;
        }
    }

    if (e.code === 'ArrowRight') {
        e.preventDefault();

        switch (currentMode) {
            case 'discontinuous':
                jumpForward();
                break;
            case 'keyword':
                jumpToNextKeyword(false);
                break;
            case 'keyword2':
                jumpToNextKeyword(true);
                break;
            case 'word':
                jumpToNextWord();
                break;
            case 'sentence':
                jumpToNextSentence();
                break;
            default:
                break;
        }
    }
});

if (fileSelect) {
    fileSelect.addEventListener('change', async () => {
        if (!fileSelect.value) return;
        const file = JSON.parse(fileSelect.value);
        currentFilename = file.filename;
        audio.src = `/mp3/${file.filename}`;
        resetPlaybackProgressLock();
        resetAllIndices();
        stopSpeedupReplay();
        stopWordBackwardMode();
        await loadTranscript(file.video_id);
    });
}

audio.addEventListener('timeupdate', () => {
    maxPlayedTime = Math.max(maxPlayedTime, audio.currentTime);
    lastLoggedAudioTime = audio.currentTime;
    updateCurrentSegment();
    checkForInterruption();
});

audio.addEventListener('seeking', () => {
    const clampedTime = clampToPlayedTime(audio.currentTime);
    if (audio.currentTime > clampedTime + 0.01) {
        audio.currentTime = clampedTime;
    }
});

audio.addEventListener('seeked', () => {
    if (suppressNextSeekLog) {
        suppressNextSeekLog = false;
        return;
    }
    syncNavigationIndices(audio.currentTime);
    if (!studyMode) return;
    logNavigationEvent('manual_seek', lastLoggedAudioTime, audio.currentTime);
});

Object.keys(modeButtons).forEach((mode) => {
    if (modeButtons[mode]) {
        modeButtons[mode].addEventListener('click', () => setMode(mode));
    }
});

if (toggleTranscriptBtn) {
    toggleTranscriptBtn.addEventListener('click', () => {
        const isHidden = transcriptDiv.style.display === 'none';
        transcriptDiv.style.display = isHidden ? 'block' : 'none';
        toggleTranscriptBtn.textContent = isHidden ? 'Hide Transcript' : 'Show Transcript';
    });
}

if (jumpSecondsInput) {
    jumpSecondsInput.addEventListener('change', () => {
        if (currentMode === 'discontinuous') {
            renderTranscript();
        }
    });
}

btnStartStudy.addEventListener('click', startStudySession);
btnStopStudy.addEventListener('click', stopStudySession);

loadFiles();
loadStudyConfig();
setMode('discontinuous');
