const audio = document.getElementById('audio-player');
const fileSelect = document.getElementById('file-select');
const timeDisplay = document.getElementById('time-display');
const transcriptDiv = document.getElementById('transcript');
const jumpSecondsInput = document.getElementById('jump-seconds');
const toggleTranscriptBtn = document.getElementById('toggle-transcript');
const replaySpeedupEnabled = document.getElementById('replay-speedup-enabled');
const replaySpeedupRate = document.getElementById('replay-speedup-rate');
const isDebugPage = document.body.classList.contains('debug-page');

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
let studySessionId = null;
let studyLogBaseName = null;
let sessionTasks = [];
let activeTargetTime = null;
let maxPlayedTime = 0;
let suppressNextSeekLog = false;
let lastLoggedAudioTime = 0;
let prepCountdownInterval = null;
let prepCountdownTimeout = null;
let blockedSeekAudioContext = null;
let lastBlockedSeekCueAt = 0;
let lastListeningStartedAt = null;
let lastListeningAudioTime = 0;

const FREQ_SKIP_THRESHOLD = 4.0;
const TASK_SEARCH_WINDOW_SECONDS = 120;

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

function slugifyStudyFilenamePart(value, fallback = 'unknown') {
    const normalized = String(value ?? '').trim().replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '');
    return normalized || fallback;
}

function buildStudyLogBaseName(participant, feature, audioFile, startedAtMs) {
    const timestamp = new Date(startedAtMs).toISOString().replace(/[-:]/g, '').replace('T', '_').slice(0, 15);
    const audioStem = getVideoIdFromAudioFilename(audioFile) || audioFile || 'unknown';
    return [
        slugifyStudyFilenamePart(participant),
        slugifyStudyFilenamePart(feature),
        slugifyStudyFilenamePart(audioStem),
        timestamp,
    ].join('_');
}

function resetPlaybackProgressLock(initialTime = 0) {
    maxPlayedTime = initialTime;
}

function getMaxSeekTime() {
    return Math.max(0, maxPlayedTime);
}

function clampToPlayedTime(targetTime) {
    return Math.max(0, Math.min(targetTime, getMaxSeekTime()));
}

function getStudyPlaybackBounds() {
    if (!studyMode || !studyInterruptions) return null;

    return {
        min: studyInterruptions.audio_start_time ?? 0,
        max: studyInterruptions.playback_end_time ?? Infinity,
    };
}

function clampToStudyPlaybackBounds(targetTime) {
    const bounds = getStudyPlaybackBounds();
    if (!bounds) return targetTime;
    return Math.max(bounds.min, Math.min(targetTime, bounds.max));
}

function getActiveSearchInterval() {
    if (!studyMode || !taskActive || activeTargetTime === null) return null;
    const playbackBounds = getStudyPlaybackBounds();
    const minTime = playbackBounds ? playbackBounds.min : 0;

    return {
        min: Math.max(minTime, activeTargetTime - TASK_SEARCH_WINDOW_SECONDS),
        max: activeTargetTime,
    };
}

function clampToActiveSearchInterval(targetTime) {
    const interval = getActiveSearchInterval();
    if (!interval) return targetTime;
    return Math.max(interval.min, Math.min(targetTime, interval.max));
}

function getSearchableItems(items, getTime) {
    const interval = getActiveSearchInterval();
    if (!interval) return items;
    return items.filter((item) => {
        const time = getTime(item);
        return time >= interval.min && time <= interval.max;
    });
}

function sortItemsByTime(items, getTime) {
    return [...items].sort((a, b) => getTime(a) - getTime(b));
}

function getClampedNavigationTime(targetTime) {
    return clampToPlayedTime(
        clampToActiveSearchInterval(clampToStudyPlaybackBounds(targetTime)),
    );
}

function playBlockedSeekCue() {
    const now = performance.now();
    if (now - lastBlockedSeekCueAt < 80) return;
    lastBlockedSeekCueAt = now;

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) return;

    if (!blockedSeekAudioContext) {
        blockedSeekAudioContext = new AudioContextCtor();
    }
    if (blockedSeekAudioContext.state === 'suspended') {
        blockedSeekAudioContext.resume().catch(() => {});
    }

    const context = blockedSeekAudioContext;
    const startAt = context.currentTime;
    const notes = [
        { frequency: 1568, duration: 0.1, delay: 0 },
        { frequency: 2093, duration: 0.16, delay: 0.11 },
    ];
    const voices = [
        { type: 'triangle', detune: 0, gain: 0.26 },
        { type: 'sawtooth', detune: -8, gain: 0.08 },
        { type: 'sine', detune: 7, gain: 0.06 },
    ];

    notes.forEach(({ frequency, duration, delay }) => {
        const noteStart = startAt + delay;
        voices.forEach(({ type, detune, gain: peakGain }) => {
            const oscillator = context.createOscillator();
            const gain = context.createGain();

            oscillator.type = type;
            oscillator.frequency.setValueAtTime(frequency, noteStart);
            oscillator.detune.setValueAtTime(detune, noteStart);
            gain.gain.setValueAtTime(0.0001, noteStart);
            gain.gain.linearRampToValueAtTime(peakGain, noteStart + 0.01);
            gain.gain.exponentialRampToValueAtTime(0.0001, noteStart + duration);

            oscillator.connect(gain);
            gain.connect(context.destination);
            oscillator.start(noteStart);
            oscillator.stop(noteStart + duration);
        });
    });
}

function showBlockedNavigationCue() {
    playBlockedSeekCue();
}

function blockForwardNavigation(action, extra = {}) {
    showBlockedNavigationCue();
    logNavigationEvent(action, audio.currentTime, audio.currentTime, {
        blocked: true,
        maxSeekTime: getMaxSeekTime(),
        ...extra,
    });
}

function blockNavigation(action, targetTime, extra = {}) {
    showBlockedNavigationCue();
    logNavigationEvent(action, audio.currentTime, audio.currentTime, {
        blocked: true,
        requestedTime: targetTime,
        maxSeekTime: getMaxSeekTime(),
        ...extra,
    });
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
    customKeywords = sortItemsByTime(transcript.custom_keywords || [], (item) => item.time);
    customKeywords2 = sortItemsByTime(transcript.custom_keywords2 || [], (item) => item.time);

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
    allWords = sortItemsByTime(allWords, (item) => item.start);
    navigableWords = allWords.filter((w) => !shouldSkipWord(w));
    sentenceUnits = sortItemsByTime(transcript.sentences || buildSentenceUnits(), (item) => item.start);

    resetAllIndices();
    syncNavigationIndices(audio.currentTime || 0);
    renderTranscript();
}

function resetAllIndices() {
    keywordIndex = -1;
    keyword2Index = -1;
    wordIndex = -1;
    sentenceIndex = -1;
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

const PREV_TIME_TOLERANCE = 0.1; // Include the current item despite minor seek precision drift.
const NEXT_TIME_EPSILON = 0.001; // Advance to the first item strictly after the current timestamp.

function findCurrentIndexByTime(items, currentTime, getTime) {
    if (!items || items.length === 0) return -1;

    // Find the last item that starts at or before currentTime
    for (let i = items.length - 1; i >= 0; i -= 1) {
        if (getTime(items[i]) <= currentTime + PREV_TIME_TOLERANCE) {
            return i;
        }
    }

    return -1;
}

function findPrevIndexByTime(items, currentTime, getTime) {
    if (!items || items.length === 0) return { targetIndex: -1, blocked: false };

    // Find the current item index
    const currentIndex = findCurrentIndexByTime(items, currentTime, getTime);

    // Always go to the previous item (before the current one)
    if (currentIndex <= 0) {
        return { targetIndex: -1, blocked: true };
    }

    return { targetIndex: currentIndex - 1, blocked: false };
}

function findNextIndexByTime(items, currentTime, getTime) {
    if (!items || items.length === 0) return { targetIndex: -1, blocked: false };

    // Find the first item strictly after the current playback timestamp
    // that we can seek to without being clamped elsewhere.
    let nextIndex = -1;
    for (let i = 0; i < items.length; i += 1) {
        const itemTime = getTime(items[i]);
        if (itemTime <= currentTime + NEXT_TIME_EPSILON) {
            continue;
        }
        if (Math.abs(getClampedNavigationTime(itemTime) - itemTime) > 0.01) {
            continue;
        }
        if (itemTime > currentTime + NEXT_TIME_EPSILON) {
            nextIndex = i;
            break;
        }
    }

    if (nextIndex < 0) {
        return { targetIndex: -1, blocked: true };
    }

    return { targetIndex: nextIndex, blocked: false };
}

function findCurrentItemByTime(items, currentTime, getTime) {
    if (!items || items.length === 0) return null;

    let currentIndex = -1;
    for (let i = items.length - 1; i >= 0; i -= 1) {
        if (getTime(items[i]) <= currentTime + PREV_TIME_TOLERANCE) {
            currentIndex = i;
            break;
        }
    }

    return currentIndex >= 0 ? items[currentIndex] : null;
}

function syncNavigationIndices(targetTime = audio.currentTime) {
    const searchableKeywords = getSearchableItems(customKeywords, (item) => item.time);
    const searchableKeywords2 = getSearchableItems(customKeywords2, (item) => item.time);
    const searchableWords = getSearchableItems(navigableWords, (item) => item.start);
    const searchableSentences = getSearchableItems(sentenceUnits, (item) => item.start);

    keywordIndex = findNearestIndex(searchableKeywords, targetTime, (item) => item.time);
    keyword2Index = findNearestIndex(searchableKeywords2, targetTime, (item) => item.time);
    wordIndex = findNearestIndex(searchableWords, targetTime, (item) => item.start);
    sentenceIndex = findNearestIndex(searchableSentences, targetTime, (item) => item.start);
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
    if (timeDisplay) {
        timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audio.duration || 0)}${suffix}`;
    }
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

function moveToSearchIntervalStart(action, extra = {}) {
    const interval = getActiveSearchInterval();
    if (!interval) return false;
    setAudioTime(interval.min, action, {
        intervalStart: interval.min,
        ...extra,
    });
    return true;
}

function jumpBack() {
    const seconds = jumpSecondsInput ? (parseInt(jumpSecondsInput.value, 10) || 15) : 15;
    if (setAudioTimeFromArrow(audio.currentTime - seconds, 'jump_back', 'jump_back_blocked', { seconds }) === null) {
        if (!moveToSearchIntervalStart('jump_back_to_interval_start', { seconds })) return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function jumpForward() {
    const seconds = jumpSecondsInput ? (parseInt(jumpSecondsInput.value, 10) || 15) : 15;
    if (audio.currentTime + seconds > getMaxSeekTime()) {
        blockForwardNavigation('jump_forward_blocked', { seconds });
        return;
    }
    if (setAudioTimeFromArrow(audio.currentTime + seconds, 'jump_forward', 'jump_forward_blocked', { seconds }) === null) {
        return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousKeyword(useKeyword2 = false) {
    const keywords = getSearchableItems(
        useKeyword2 ? customKeywords2 : customKeywords,
        (item) => item.time,
    );

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    const currentKeyword = findCurrentItemByTime(keywords, currentTime, (item) => item.time);
    const { targetIndex, blocked } = findPrevIndexByTime(keywords, currentTime, (item) => item.time);

    console.log('[keyword_prev]', {
        mode: useKeyword2 ? 'keyword2' : 'keyword',
        currentTime,
        currentKeyword,
        targetKeyword: targetIndex >= 0 ? keywords[targetIndex] : null,
        blocked,
    });

    if (blocked || targetIndex < 0) {
        if (!moveToSearchIntervalStart(
            useKeyword2 ? 'keyword2_prev_to_interval_start' : 'keyword_prev_to_interval_start',
            { currentTime },
        )) {
            blockNavigation(useKeyword2 ? 'keyword2_prev_blocked' : 'keyword_prev_blocked', currentTime, {
                currentTime,
            });
            return;
        }
        applySearchPlaybackRate();
        audio.play();
        return;
    }

    const target = keywords[targetIndex];
    if (useKeyword2) keyword2Index = targetIndex;
    else keywordIndex = targetIndex;

    if (setAudioTimeFromArrow(
        target.time,
        useKeyword2 ? 'keyword2_prev' : 'keyword_prev',
        useKeyword2 ? 'keyword2_prev_blocked' : 'keyword_prev_blocked',
        {
            keyword: target.word,
            keywordTime: target.time,
            keywordIndex: targetIndex,
        },
    ) === null) {
        return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function jumpToNextKeyword(useKeyword2 = false) {
    const keywords = getSearchableItems(
        useKeyword2 ? customKeywords2 : customKeywords,
        (item) => item.time,
    );

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    const currentKeyword = findCurrentItemByTime(keywords, currentTime, (item) => item.time);
    const { targetIndex, blocked } = findNextIndexByTime(keywords, currentTime, (item) => item.time);

    console.log('[keyword_next]', {
        mode: useKeyword2 ? 'keyword2' : 'keyword',
        currentTime,
        currentKeyword,
        targetKeyword: targetIndex >= 0 ? keywords[targetIndex] : null,
        blocked,
    });

    if (blocked || targetIndex < 0) {
        blockNavigation(useKeyword2 ? 'keyword2_next_blocked' : 'keyword_next_blocked', currentTime, {
            currentTime,
        });
        return;
    }

    const target = keywords[targetIndex];
    const seekTime = target.time;
    if (seekTime > getMaxSeekTime()) {
        blockForwardNavigation(useKeyword2 ? 'keyword2_next_blocked' : 'keyword_next_blocked', {
            keyword: target.word,
            keywordTime: target.time,
            keywordIndex: targetIndex,
        });
        return;
    }

    if (setAudioTimeFromArrow(
        seekTime,
        useKeyword2 ? 'keyword2_next' : 'keyword_next',
        useKeyword2 ? 'keyword2_next_blocked' : 'keyword_next_blocked',
        {
            keyword: target.word,
            keywordTime: target.time,
            keywordIndex: targetIndex,
        },
    ) === null) {
        return;
    }
    if (useKeyword2) keyword2Index = targetIndex;
    else keywordIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousWord() {
    const searchableWords = getSearchableItems(navigableWords, (item) => item.start);
    if (searchableWords.length === 0) return;

    const currentTime = audio.currentTime;
    const { targetIndex, blocked } = findPrevIndexByTime(searchableWords, currentTime, (item) => item.start);

    if (blocked || targetIndex < 0) {
        if (!moveToSearchIntervalStart('word_prev_to_interval_start', { currentTime })) {
            blockNavigation('word_prev_blocked', currentTime, { currentTime });
            return;
        }
        applySearchPlaybackRate();
        audio.play();
        return;
    }

    const target = searchableWords[targetIndex];
    wordIndex = targetIndex;
    if (setAudioTimeFromArrow(target.start, 'word_prev', 'word_prev_blocked', {
        word: target.word,
        wordStart: target.start,
        wordIndex: targetIndex,
    }) === null) {
        return;
    }
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
    const searchableWords = getSearchableItems(navigableWords, (item) => item.start);
    if (searchableWords.length === 0) return;

    stopWordBackwardMode();

    const currentTime = audio.currentTime;
    const { targetIndex, blocked } = findNextIndexByTime(searchableWords, currentTime, (item) => item.start);

    if (blocked || targetIndex < 0) {
        blockNavigation('word_next_blocked', currentTime, { currentTime });
        return;
    }

    const target = searchableWords[targetIndex];
    if (target.start > getMaxSeekTime()) {
        blockForwardNavigation('word_next_blocked', {
            word: target.word,
            wordStart: target.start,
            wordIndex: targetIndex,
        });
        return;
    }

    if (setAudioTimeFromArrow(target.start, 'word_next', 'word_next_blocked', {
        word: target.word,
        wordStart: target.start,
        wordIndex: targetIndex,
    }) === null) {
        return;
    }
    wordIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

function jumpToPreviousSentence() {
    const searchableSentences = getSearchableItems(sentenceUnits, (item) => item.start);
    if (searchableSentences.length === 0) return;

    const currentTime = audio.currentTime;
    const { targetIndex, blocked } = findPrevIndexByTime(searchableSentences, currentTime, (item) => item.start);

    if (blocked || targetIndex < 0) {
        if (!moveToSearchIntervalStart('sentence_prev_to_interval_start', { currentTime })) {
            blockNavigation('sentence_prev_blocked', currentTime, { currentTime });
            return;
        }
        applySearchPlaybackRate();
        audio.play();
        return;
    }

    const target = searchableSentences[targetIndex];
    sentenceIndex = targetIndex;
    if (setAudioTimeFromArrow(target.start, 'sentence_prev', 'sentence_prev_blocked', {
        sentenceIndex: targetIndex,
        sentenceStart: target.start,
    }) === null) {
        return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function jumpToNextSentence() {
    const searchableSentences = getSearchableItems(sentenceUnits, (item) => item.start);
    if (searchableSentences.length === 0) return;

    const currentTime = audio.currentTime;
    const { targetIndex, blocked } = findNextIndexByTime(searchableSentences, currentTime, (item) => item.start);

    if (blocked || targetIndex < 0) {
        blockNavigation('sentence_next_blocked', currentTime, { currentTime });
        return;
    }

    const target = searchableSentences[targetIndex];
    if (target.start > getMaxSeekTime()) {
        blockForwardNavigation('sentence_next_blocked', {
            sentenceIndex: targetIndex,
            sentenceStart: target.start,
        });
        return;
    }

    if (setAudioTimeFromArrow(target.start, 'sentence_next', 'sentence_next_blocked', {
        sentenceIndex: targetIndex,
        sentenceStart: target.start,
    }) === null) {
        return;
    }
    sentenceIndex = targetIndex;
    applySearchPlaybackRate();
    audio.play();
}

function replayCurrentKeyword(useKeyword2 = false) {
    const keywords = getSearchableItems(
        useKeyword2 ? customKeywords2 : customKeywords,
        (item) => item.time,
    );

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    const currentIndex = findCurrentIndexByTime(keywords, currentTime, (item) => item.time);

    if (currentIndex < 0) {
        blockNavigation(useKeyword2 ? 'keyword2_replay_blocked' : 'keyword_replay_blocked', currentTime, {
            currentTime,
        });
        return;
    }

    const target = keywords[currentIndex];
    if (useKeyword2) keyword2Index = currentIndex;
    else keywordIndex = currentIndex;

    console.log('[keyword_replay]', {
        mode: useKeyword2 ? 'keyword2' : 'keyword',
        currentTime,
        targetKeyword: target,
    });

    if (setAudioTimeFromArrow(
        target.time,
        useKeyword2 ? 'keyword2_replay' : 'keyword_replay',
        useKeyword2 ? 'keyword2_replay_blocked' : 'keyword_replay_blocked',
        {
            keyword: target.word,
            keywordTime: target.time,
            keywordIndex: currentIndex,
        },
    ) === null) {
        return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function replayCurrentWord() {
    const searchableWords = getSearchableItems(navigableWords, (item) => item.start);
    if (searchableWords.length === 0) return;

    const currentTime = audio.currentTime;
    const currentIndex = findCurrentIndexByTime(searchableWords, currentTime, (item) => item.start);

    if (currentIndex < 0) {
        blockNavigation('word_replay_blocked', currentTime, { currentTime });
        return;
    }

    const target = searchableWords[currentIndex];
    wordIndex = currentIndex;

    console.log('[word_replay]', {
        currentTime,
        targetWord: target,
    });

    if (setAudioTimeFromArrow(target.start, 'word_replay', 'word_replay_blocked', {
        word: target.word,
        wordStart: target.start,
        wordIndex: currentIndex,
    }) === null) {
        return;
    }
    applySearchPlaybackRate();
    audio.play();
}

function replayCurrentSentence() {
    const searchableSentences = getSearchableItems(sentenceUnits, (item) => item.start);
    if (searchableSentences.length === 0) return;

    const currentTime = audio.currentTime;
    const currentIndex = findCurrentIndexByTime(searchableSentences, currentTime, (item) => item.start);

    if (currentIndex < 0) {
        blockNavigation('sentence_replay_blocked', currentTime, { currentTime });
        return;
    }

    const target = searchableSentences[currentIndex];
    sentenceIndex = currentIndex;

    console.log('[sentence_replay]', {
        currentTime,
        targetSentence: target,
    });

    if (setAudioTimeFromArrow(target.start, 'sentence_replay', 'sentence_replay_blocked', {
        sentenceIndex: currentIndex,
        sentenceStart: target.start,
    }) === null) {
        return;
    }
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
                sessionId: studySessionId,
                feature: studyFeature.value,
                audio: studyAudio.value,
                logBaseName: studyLogBaseName,
                event,
                ...data,
            }),
        });
    } catch (err) {
        console.error('Failed to log event:', err);
    }
}

function resetListeningWindow() {
    lastListeningStartedAt = Date.now();
    lastListeningAudioTime = audio.currentTime;
}

function flushListeningInterval(endedBy, extra = {}) {
    if (!studyMode || lastListeningStartedAt === null) return;

    const endedAt = Date.now();
    const listeningDurationMs = endedAt - lastListeningStartedAt;
    const endAudioTime = audio.currentTime;

    if (listeningDurationMs <= 0) {
        resetListeningWindow();
        return;
    }

    logStudyEvent('listening', {
        startedAt: new Date(lastListeningStartedAt).toISOString(),
        endedAt: new Date(endedAt).toISOString(),
        listeningDurationMs,
        startAudioTime: lastListeningAudioTime,
        endAudioTime,
        audioProgressSeconds: endAudioTime - lastListeningAudioTime,
        endedBy,
        taskActive,
        ...extra,
    });

    lastListeningStartedAt = endedAt;
    lastListeningAudioTime = endAudioTime;
}

function logUserAction(action, extra = {}) {
    if (!studyMode) return;

    flushListeningInterval('user_action', { nextAction: action });
    logStudyEvent('user_action', {
        action,
        audioTime: audio.currentTime,
        mode: currentMode,
        taskActive,
        playbackRate: audio.playbackRate,
        ...extra,
    });
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
    const toTime = getClampedNavigationTime(targetTime);
    suppressNextSeekLog = true;
    audio.currentTime = toTime;
    logNavigationEvent(action, fromTime, toTime, extra);
    return toTime;
}

function setAudioTimeFromArrow(targetTime, action, blockedAction, extra = {}) {
    const toTime = getClampedNavigationTime(targetTime);
    if (Math.abs(toTime - targetTime) > 0.01) {
        blockNavigation(blockedAction, targetTime, extra);
        return null;
    }
    return setAudioTime(targetTime, action, extra);
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

    currentFilename = audioFile;
    audio.src = `/mp3/${audioFile}`;

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
    studySessionId = `${participant}_${Date.now()}`;
    currentTaskIndex = 0;
    taskActive = false;
    sessionTasks = [];
    sessionStartTime = Date.now();
    studyLogBaseName = buildStudyLogBaseName(participant, feature, audioFile, sessionStartTime);
    const audioStartTime = studyInterruptions.audio_start_time ?? 0;
    audio.currentTime = audioStartTime;
    resetPlaybackProgressLock(audioStartTime);
    resetListeningWindow();

    setMode(feature);

    await loadTranscript(videoId);

    document.body.classList.add('study-active');
    btnStartStudy.style.display = 'none';
    btnStopStudy.style.display = 'inline-block';
    taskTotalDisplay.textContent = studyInterruptions.interruptions.length;

    await logStudyEvent('session_start', {
        feature,
        audio: audioFile,
        videoId,
        audioStartTime,
        playbackEndTime: studyInterruptions.playback_end_time,
        totalTasks: studyInterruptions.interruptions.length,
    });

    const playbackStartPromise = audio.play().catch((err) => {
        console.error('Failed to start study playback:', err);
        return err;
    });
    const playbackStartResult = await playbackStartPromise;
    if (playbackStartResult instanceof Error) {
        alert('Playback could not start automatically. Please click play and start again.');
    }
}

async function stopStudySession() {
    flushListeningInterval('session_stop');
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
            sessionId: studySessionId,
            feature: studyFeature.value,
            audio: studyAudio.value,
            logBaseName: studyLogBaseName,
            sessionStartTime: new Date(sessionStartTime).toISOString(),
            sessionEndTime: new Date().toISOString(),
            totalTasks: studyInterruptions ? studyInterruptions.interruptions.length : 0,
            completed,
            timeouts,
            tasks: sessionTasks,
        }),
    });

    studyMode = false;
    studySessionId = null;
    studyLogBaseName = null;
    taskActive = false;
    studyInterruptions = null;
    lastListeningStartedAt = null;

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
    flushListeningInterval('interruption');
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

    const delayTime = interruption.delay_seconds ?? null;
    const delayType = interruption.delay_type ?? (delayTime !== null ? 'custom_seconds' : null);
    await logStudyEvent('interruption', {
        taskIndex: currentTaskIndex + 1,
        word: interruption.target_word,
        delayType,
        delayTime,
        targetTime: interruption.target_word_time,
        triggerTime,
        audioTime: activeTargetTime,
        searchIntervalMin: Math.max(0, activeTargetTime - TASK_SEARCH_WINDOW_SECONDS),
        searchIntervalMax: activeTargetTime,
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

    const interruption = studyInterruptions.interruptions[currentTaskIndex];

    logUserAction('space', {
        taskIndex: currentTaskIndex + 1,
        targetTime: interruption.target_word_time,
        triggerAudioTime: activeTargetTime,
    });

    if (taskTimerInterval) {
        clearInterval(taskTimerInterval);
        taskTimerInterval = null;
    }

    stopSpeedupReplay();
    audio.pause();

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
        resetListeningWindow();
        audio.play();
    }, prepDelaySeconds * 1000);
}

async function completeStudySession() {
    stopSpeedupReplay();
    flushListeningInterval('session_complete');
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
            sessionId: studySessionId,
            feature: studyFeature.value,
            audio: studyAudio.value,
            logBaseName: studyLogBaseName,
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
    studySessionId = null;
    studyLogBaseName = null;
    taskActive = false;
    activeTargetTime = null;
    lastListeningStartedAt = null;

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
        if (studyMode && !taskActive && !isDebugPage) return;
        logUserAction('left');

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
        if (studyMode && !taskActive && !isDebugPage) return;
        logUserAction('right');

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

    if (e.code === 'ArrowUp') {
        e.preventDefault();
        if (studyMode && !taskActive && !isDebugPage) return;
        logUserAction('up');

        switch (currentMode) {
            case 'discontinuous':
                // No replay in discontinuous mode - do nothing
                break;
            case 'keyword':
                replayCurrentKeyword(false);
                break;
            case 'keyword2':
                replayCurrentKeyword(true);
                break;
            case 'word':
                replayCurrentWord();
                break;
            case 'sentence':
                replayCurrentSentence();
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
    const playbackBounds = getStudyPlaybackBounds();
    if (playbackBounds && audio.currentTime > playbackBounds.max) {
        audio.currentTime = playbackBounds.max;
        audio.pause();
    }

    const searchInterval = getActiveSearchInterval();
    if (searchInterval) {
        if (audio.currentTime < searchInterval.min) {
            audio.currentTime = searchInterval.min;
        } else if (audio.currentTime > searchInterval.max) {
            audio.currentTime = searchInterval.max;
            audio.pause();
            showBlockedNavigationCue();
        }
    }

    maxPlayedTime = Math.max(maxPlayedTime, audio.currentTime);
    lastLoggedAudioTime = audio.currentTime;
    updateCurrentSegment();
    checkForInterruption();
});

audio.addEventListener('seeking', () => {
    const clampedTime = clampToPlayedTime(
        clampToActiveSearchInterval(clampToStudyPlaybackBounds(audio.currentTime)),
    );
    if (Math.abs(audio.currentTime - clampedTime) > 0.01) {
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
    logUserAction('seek', { fromTime: lastLoggedAudioTime, toTime: audio.currentTime });
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
