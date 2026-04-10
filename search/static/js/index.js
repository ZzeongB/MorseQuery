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
let sessionTasks = [];

const FREQ_SKIP_THRESHOLD = 4.0;

function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
}

async function loadFiles() {
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

    resetAllIndices();
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

function renderTranscript() {
    if (!transcript) return;

    const jumpSeconds = parseInt(jumpSecondsInput.value, 10) || 15;

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
        transcriptDiv.innerHTML = transcript.segments.map((seg, i) => (
            `<div class="segment" data-seg-idx="${i}" data-start="${seg.start}" data-end="${seg.end}">
                <span class="time">[${formatTime(seg.start)}]</span> ${seg.text}
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
    if (isSpeedupReplay) suffix = ` [${replaySpeedupRate.value}x]`;
    timeDisplay.textContent = `${formatTime(currentTime)} / ${formatTime(audio.duration || 0)}${suffix}`;
}

function setMode(mode) {
    currentMode = mode;
    Object.keys(modeButtons).forEach((m) => {
        modeButtons[m].classList.toggle('active', m === mode);
    });

    document.getElementById('jump-back-setting').style.display =
        mode === 'discontinuous' ? 'block' : 'none';

    stopSpeedupReplay();
    stopWordBackwardMode();
    resetAllIndices();
    renderTranscript();
}

function stopWordBackwardMode() {
    isWordBackwardMode = false;
    currentWordEndTime = 0;
}

function startSpeedupReplay() {
    if (!replaySpeedupEnabled.checked) return;
    const rate = parseFloat(replaySpeedupRate.value) || 1.5;
    audio.playbackRate = rate;
    isSpeedupReplay = true;
}

function stopSpeedupReplay() {
    audio.playbackRate = 1.0;
    isSpeedupReplay = false;
}

function jumpBack() {
    const seconds = parseInt(jumpSecondsInput.value, 10) || 15;
    audio.currentTime = Math.max(0, audio.currentTime - seconds);
    startSpeedupReplay();
    audio.play();
}

function jumpForward() {
    const seconds = parseInt(jumpSecondsInput.value, 10) || 15;
    audio.currentTime = Math.min(audio.duration || 0, audio.currentTime + seconds);
    stopSpeedupReplay();
    audio.play();
}

function jumpToPreviousKeyword(useKeyword2 = false) {
    const keywords = useKeyword2 ? customKeywords2 : customKeywords;

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;
    let currentIdx = useKeyword2 ? keyword2Index : keywordIndex;

    if (currentIdx === -1) {
        for (let i = keywords.length - 1; i >= 0; i -= 1) {
            if (keywords[i].time < currentTime - 0.5) {
                currentIdx = i;
                break;
            }
        }
        if (currentIdx === -1) currentIdx = 0;
    } else {
        currentIdx -= 1;
    }

    if (currentIdx < 0) currentIdx = 0;

    if (useKeyword2) keyword2Index = currentIdx;
    else keywordIndex = currentIdx;

    const target = keywords[currentIdx];
    audio.currentTime = Math.max(0, target.time - 0.3);
    startSpeedupReplay();
    audio.play();
}

function jumpToNextKeyword(useKeyword2 = false) {
    const keywords = useKeyword2 ? customKeywords2 : customKeywords;

    if (keywords.length === 0) return;

    const currentTime = audio.currentTime;

    for (let i = 0; i < keywords.length; i += 1) {
        if (keywords[i].time > currentTime + 0.1) {
            audio.currentTime = Math.max(0, keywords[i].time - 0.3);
            if (useKeyword2) keyword2Index = -1;
            else keywordIndex = -1;
            stopSpeedupReplay();
            audio.play();
            return;
        }
    }
}

function jumpToPreviousWord() {
    if (allWords.length === 0) return;

    const currentTime = audio.currentTime;

    if (wordIndex === -1) {
        for (let i = allWords.length - 1; i >= 0; i -= 1) {
            if (allWords[i].start < currentTime - 0.3) {
                wordIndex = i;
                break;
            }
        }
        if (wordIndex === -1) wordIndex = 0;
    } else {
        let newIndex = wordIndex - 1;
        while (newIndex > 0 && shouldSkipWord(allWords[newIndex])) {
            newIndex -= 1;
        }
        if (newIndex >= 0) wordIndex = newIndex;
    }

    const target = allWords[wordIndex];
    audio.currentTime = Math.max(0, target.start);
    startSpeedupReplay();
    audio.play();
}

function jumpToPreviousWordAuto() {
    if (allWords.length === 0 || wordIndex <= 0) {
        stopWordBackwardMode();
        return;
    }

    let newIndex = wordIndex - 1;
    while (newIndex > 0 && shouldSkipWord(allWords[newIndex])) {
        newIndex -= 1;
    }

    if (newIndex < 0) {
        stopWordBackwardMode();
        return;
    }

    wordIndex = newIndex;
    const target = allWords[wordIndex];
    audio.currentTime = Math.max(0, target.start);
    currentWordEndTime = target.end;
}

function jumpToNextWord() {
    if (allWords.length === 0) return;

    stopWordBackwardMode();

    const currentTime = audio.currentTime;

    for (let i = 0; i < allWords.length; i += 1) {
        if (allWords[i].start > currentTime + 0.1 && !shouldSkipWord(allWords[i])) {
            audio.currentTime = Math.max(0, allWords[i].start);
            wordIndex = -1;
            stopSpeedupReplay();
            audio.play();
            return;
        }
    }
}

function jumpToPreviousSentence() {
    if (!transcript || !transcript.segments || transcript.segments.length === 0) return;

    const currentTime = audio.currentTime;
    const { segments } = transcript;

    if (sentenceIndex === -1) {
        for (let i = segments.length - 1; i >= 0; i -= 1) {
            if (segments[i].start < currentTime - 0.5) {
                sentenceIndex = i;
                break;
            }
        }
        if (sentenceIndex === -1) sentenceIndex = 0;
    } else {
        sentenceIndex -= 1;
    }

    if (sentenceIndex < 0) sentenceIndex = 0;

    const target = segments[sentenceIndex];
    audio.currentTime = Math.max(0, target.start);
    startSpeedupReplay();
    audio.play();
}

function jumpToNextSentence() {
    if (!transcript || !transcript.segments || transcript.segments.length === 0) return;

    const currentTime = audio.currentTime;
    const { segments } = transcript;

    for (let i = 0; i < segments.length; i += 1) {
        if (segments[i].start > currentTime + 0.1) {
            audio.currentTime = Math.max(0, segments[i].start);
            sentenceIndex = -1;
            stopSpeedupReplay();
            audio.play();
            return;
        }
    }
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

async function startStudySession() {
    const participant = studyParticipant.value.trim();
    const feature = studyFeature.value;
    const audioFile = studyAudio.value;

    if (!participant || !feature || !audioFile) {
        alert('Please fill in all fields (Participant, Feature, Audio)');
        return;
    }

    const videoId = audioFile.split('_clip_')[0];

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
    if (taskTimerInterval) {
        clearInterval(taskTimerInterval);
        taskTimerInterval = null;
    }

    interruptionOverlay.classList.remove('active');
    feedbackOverlay.classList.remove('active', 'success', 'failure');
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

    audio.pause();

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
    showFeedback(true, interruption, taskResult);
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
    taskActive = false;
    currentTaskIndex += 1;

    if (currentTaskIndex >= studyInterruptions.interruptions.length) {
        completeStudySession();
        return;
    }

    const triggerTime = interruption.target_word_time + interruption.delay_seconds;
    const resumeTime = Math.max(0, triggerTime - studyConfig.resume_offset_seconds);
    audio.currentTime = resumeTime;
    audio.play();
}

async function completeStudySession() {
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

fileSelect.addEventListener('change', async () => {
    if (!fileSelect.value) return;
    const file = JSON.parse(fileSelect.value);
    currentFilename = file.filename;
    audio.src = `/mp3/${file.filename}`;
    resetAllIndices();
    stopSpeedupReplay();
    stopWordBackwardMode();
    await loadTranscript(file.video_id);
});

audio.addEventListener('timeupdate', () => {
    updateCurrentSegment();
    checkForInterruption();
});

Object.keys(modeButtons).forEach((mode) => {
    modeButtons[mode].addEventListener('click', () => setMode(mode));
});

toggleTranscriptBtn.addEventListener('click', () => {
    const isHidden = transcriptDiv.style.display === 'none';
    transcriptDiv.style.display = isHidden ? 'block' : 'none';
    toggleTranscriptBtn.textContent = isHidden ? 'Hide Transcript' : 'Show Transcript';
});

jumpSecondsInput.addEventListener('change', () => {
    if (currentMode === 'discontinuous') {
        renderTranscript();
    }
});

btnStartStudy.addEventListener('click', startStudySession);
btnStopStudy.addEventListener('click', stopStudySession);

loadFiles();
loadStudyConfig();
setMode('discontinuous');
