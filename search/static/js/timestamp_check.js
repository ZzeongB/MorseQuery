const audio = document.getElementById('audio-player');
const fileSelect = document.getElementById('file-select');
const playbackRateSelect = document.getElementById('playback-rate');
const offsetInput = document.getElementById('offset-ms');
const loopPaddingInput = document.getElementById('loop-padding-ms');
const timeline = document.getElementById('timeline');
const wordList = document.getElementById('word-list');
const timeReadout = document.getElementById('time-readout');
const currentWordEl = document.getElementById('current-word');
const selectionWindowEl = document.getElementById('selection-window');
const selectionDetails = document.getElementById('selection-details');
const playSelectionButton = document.getElementById('play-selection');
const toggleLoopButton = document.getElementById('toggle-loop');

let transcriptData = null;
let words = [];
let selectedWordIndex = -1;
let activeWordIndex = -1;
let loopEnabled = false;
let selectionLoop = null;

function formatSeconds(value) {
    return Number.isFinite(value) ? value.toFixed(2) : '-';
}

function getOffsetSeconds() {
    return (Number(offsetInput.value) || 0) / 1000;
}

function getLoopPaddingSeconds() {
    return Math.max(0, (Number(loopPaddingInput.value) || 0) / 1000);
}

function clampTime(value) {
    const duration = Number.isFinite(audio.duration) ? audio.duration : Infinity;
    return Math.max(0, Math.min(value, duration));
}

function getAdjustedWindow(word) {
    const offset = getOffsetSeconds();
    return {
        start: clampTime(word.start + offset),
        end: clampTime(word.end + offset),
    };
}

function buildWords(transcript) {
    const items = [];
    (transcript.segments || []).forEach((segment, segmentIndex) => {
        (segment.words || []).forEach((word, wordIndexInSegment) => {
            items.push({
                id: `${segmentIndex}-${wordIndexInSegment}`,
                word: word.word,
                start: word.start,
                end: word.end,
                segmentText: segment.text,
                segmentIndex,
            });
        });
    });
    return items;
}

async function loadFiles() {
    const response = await fetch('/api/files');
    const files = await response.json();
    files.forEach((file) => {
        const option = document.createElement('option');
        option.value = JSON.stringify(file);
        option.textContent = file.transcript_id && file.transcript_id !== file.video_id
            ? `${file.filename} :: ${file.transcript_id}`
            : file.filename;
        fileSelect.appendChild(option);
    });
}

async function loadTranscript(transcriptId) {
    const response = await fetch(`/api/transcript/${transcriptId}`);
    transcriptData = await response.json();
    words = buildWords(transcriptData);
    selectedWordIndex = words.length > 0 ? 0 : -1;
    activeWordIndex = -1;
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
    syncActiveWord();
}

function renderWordList() {
    wordList.innerHTML = '';
    if (words.length === 0) {
        wordList.innerHTML = '<p class="empty">No word-level timestamps in this transcript.</p>';
        return;
    }

    const fragment = document.createDocumentFragment();
    words.forEach((word, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'word-chip';
        button.dataset.index = String(index);
        button.innerHTML = `
            <span class="word-text">${word.word}</span>
            <span class="word-time">${formatSeconds(word.start)}-${formatSeconds(word.end)}</span>
        `;
        button.addEventListener('click', () => {
            selectedWordIndex = index;
            seekToSelectedWord();
            renderWordList();
            renderTimeline();
            updateSelectionDetails();
        });
        fragment.appendChild(button);
    });
    wordList.appendChild(fragment);
    updateWordSelectionClasses();
}

function renderTimeline() {
    timeline.innerHTML = '';
    if (words.length === 0) {
        timeline.innerHTML = '<p class="empty">No timeline data.</p>';
        return;
    }

    const duration = Math.max(...words.map((word) => word.end), audio.duration || 0.001);
    const cursor = document.createElement('div');
    cursor.className = 'timeline-cursor';
    timeline.appendChild(cursor);

    words.forEach((word, index) => {
        const bar = document.createElement('button');
        bar.type = 'button';
        bar.className = 'timeline-word';
        bar.dataset.index = String(index);
        const left = (word.start / duration) * 100;
        const width = Math.max(0.35, ((word.end - word.start) / duration) * 100);
        bar.style.left = `${left}%`;
        bar.style.width = `${width}%`;
        bar.title = `${word.word} ${formatSeconds(word.start)}-${formatSeconds(word.end)}`;
        bar.textContent = word.word;
        bar.addEventListener('click', () => {
            selectedWordIndex = index;
            seekToSelectedWord();
            renderWordList();
            renderTimeline();
            updateSelectionDetails();
        });
        timeline.appendChild(bar);
    });

    updateWordSelectionClasses();
    updateTimelineCursor();
}

function updateWordSelectionClasses() {
    document.querySelectorAll('.word-chip').forEach((element) => {
        const index = Number(element.dataset.index);
        element.classList.toggle('selected', index === selectedWordIndex);
        element.classList.toggle('active', index === activeWordIndex);
    });

    document.querySelectorAll('.timeline-word').forEach((element) => {
        const index = Number(element.dataset.index);
        element.classList.toggle('selected', index === selectedWordIndex);
        element.classList.toggle('active', index === activeWordIndex);
    });
}

function updateTimelineCursor() {
    const cursor = timeline.querySelector('.timeline-cursor');
    if (!cursor || words.length === 0) return;
    const duration = Math.max(...words.map((word) => word.end), audio.duration || 0.001);
    cursor.style.left = `${(clampTime(audio.currentTime) / duration) * 100}%`;
}

function getWordAtTime(timeSeconds) {
    const offset = getOffsetSeconds();
    return words.findIndex((word) => {
        const start = word.start + offset;
        const end = word.end + offset;
        return timeSeconds >= start && timeSeconds <= end;
    });
}

function syncActiveWord() {
    activeWordIndex = getWordAtTime(audio.currentTime);
    const activeWord = words[activeWordIndex];
    currentWordEl.textContent = activeWord ? activeWord.word : '-';
    updateWordSelectionClasses();
    updateTimelineCursor();
    updateTimeReadout();
}

function updateTimeReadout() {
    timeReadout.textContent = `${formatSeconds(audio.currentTime)} / ${formatSeconds(audio.duration)}`;
}

function updateSelectionDetails() {
    const detailValues = selectionDetails.querySelectorAll('dd');
    const word = words[selectedWordIndex];
    if (!word) {
        detailValues.forEach((detail) => {
            detail.textContent = '-';
        });
        selectionWindowEl.textContent = '-';
        playSelectionButton.disabled = true;
        toggleLoopButton.disabled = true;
        return;
    }

    const adjusted = getAdjustedWindow(word);
    const values = [
        word.word,
        formatSeconds(word.start),
        formatSeconds(word.end),
        formatSeconds(word.end - word.start),
        `${word.segmentIndex + 1}`,
        formatSeconds(adjusted.start),
    ];
    detailValues.forEach((detail, index) => {
        detail.textContent = values[index];
    });

    const padding = getLoopPaddingSeconds();
    const loopStart = clampTime(adjusted.start - padding);
    const loopEnd = clampTime(adjusted.end + padding);
    selectionWindowEl.textContent = `${formatSeconds(loopStart)}-${formatSeconds(loopEnd)}`;
    playSelectionButton.disabled = false;
    toggleLoopButton.disabled = false;
    toggleLoopButton.textContent = loopEnabled ? 'Loop On' : 'Loop Off';
}

function seekToSelectedWord() {
    const word = words[selectedWordIndex];
    if (!word) return;
    const adjusted = getAdjustedWindow(word);
    audio.currentTime = adjusted.start;
}

function playSelection() {
    const word = words[selectedWordIndex];
    if (!word) return;
    const adjusted = getAdjustedWindow(word);
    const padding = getLoopPaddingSeconds();
    selectionLoop = {
        start: clampTime(adjusted.start - padding),
        end: clampTime(adjusted.end + padding),
    };
    audio.currentTime = selectionLoop.start;
    audio.play().catch(() => {});
}

function moveSelection(delta) {
    if (words.length === 0) return;
    selectedWordIndex = Math.max(0, Math.min(words.length - 1, selectedWordIndex + delta));
    seekToSelectedWord();
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
}

function handleLoopPlayback() {
    if (!selectionLoop) return;
    if (audio.currentTime >= selectionLoop.end) {
        if (loopEnabled) {
            audio.currentTime = selectionLoop.start;
            audio.play().catch(() => {});
        } else {
            selectionLoop = null;
            audio.pause();
        }
    }
}

function handleFileChange() {
    const selected = fileSelect.value;
    if (!selected) return;
    const file = JSON.parse(selected);
    audio.src = `/mp3/${encodeURIComponent(file.filename)}`;
    audio.load();
    loadTranscript(file.transcript_id || file.video_id);
}

document.querySelectorAll('[data-seek]').forEach((button) => {
    button.addEventListener('click', () => {
        audio.currentTime = clampTime(audio.currentTime + Number(button.dataset.seek));
    });
});

fileSelect.addEventListener('change', handleFileChange);
playbackRateSelect.addEventListener('change', () => {
    audio.playbackRate = Number(playbackRateSelect.value) || 1;
});
offsetInput.addEventListener('input', () => {
    updateSelectionDetails();
    syncActiveWord();
});
loopPaddingInput.addEventListener('input', updateSelectionDetails);

playSelectionButton.addEventListener('click', playSelection);
toggleLoopButton.addEventListener('click', () => {
    loopEnabled = !loopEnabled;
    updateSelectionDetails();
});

audio.addEventListener('timeupdate', () => {
    syncActiveWord();
    handleLoopPlayback();
});
audio.addEventListener('loadedmetadata', updateTimeReadout);

document.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, textarea')) return;

    if (event.code === 'Space') {
        event.preventDefault();
        if (audio.paused) {
            audio.play().catch(() => {});
        } else {
            audio.pause();
        }
        return;
    }

    if (event.key === 'j' || event.key === 'J') {
        audio.currentTime = clampTime(audio.currentTime - 0.2);
        return;
    }
    if (event.key === 'l' || event.key === 'L') {
        audio.currentTime = clampTime(audio.currentTime + 0.2);
        return;
    }
    if (event.key === ',') {
        event.preventDefault();
        moveSelection(-1);
        return;
    }
    if (event.key === '.') {
        event.preventDefault();
        moveSelection(1);
        return;
    }
    if (event.key === 'r' || event.key === 'R') {
        event.preventDefault();
        playSelection();
    }
});

loadFiles();
