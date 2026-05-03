const audio = document.getElementById('audio-player');
const fileSelect = document.getElementById('file-select');
const playbackRateSelect = document.getElementById('playback-rate');
const dragScaleInput = document.getElementById('drag-scale-ms');
const zoomRange = document.getElementById('zoom-range');
const zoomReadout = document.getElementById('zoom-readout');
const zoomOutButton = document.getElementById('zoom-out');
const zoomInButton = document.getElementById('zoom-in');
const timelineScroll = document.getElementById('timeline-scroll');
const timeline = document.getElementById('timeline');
const wordList = document.getElementById('word-list');
const timeReadout = document.getElementById('time-readout');
const currentWordEl = document.getElementById('current-word');
const dragDeltaEl = document.getElementById('drag-delta');
const saveStatusEl = document.getElementById('save-status');
const selectionDetails = document.getElementById('selection-details');
const playSelectionButton = document.getElementById('play-selection');
const resetButton = document.getElementById('reset-button');
const saveButton = document.getElementById('save-button');

let transcriptData = null;
let transcriptId = null;
let words = [];
let originalWords = [];
let selectedWordIndex = -1;
let activeWordIndex = -1;
let keywordStarts = new Set();
let jargonStarts = new Set();
let dragState = null;
let zoomLevel = Number(zoomRange?.value || 3);
let selectedFile = null;

function syncSelectedFileFromSelect() {
    if (!fileSelect.value) {
        selectedFile = null;
        return;
    }
    selectedFile = JSON.parse(fileSelect.value);
}

function getTimelineDuration() {
    return Math.max(...words.map((word) => word.end), audio.duration || 0.001);
}

function getTimelineWidth(duration = getTimelineDuration()) {
    const basePixelsPerSecond = 28;
    const minWidth = timelineScroll ? timelineScroll.clientWidth - 2 : 0;
    return Math.max(minWidth, duration * basePixelsPerSecond * zoomLevel);
}

function updateZoomReadout() {
    zoomReadout.textContent = `${zoomLevel}x`;
}

function formatSeconds(value) {
    return Number.isFinite(value) ? value.toFixed(2) : '-';
}

function formatSignedSeconds(value) {
    if (!Number.isFinite(value)) return '-';
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}s`;
}

function clampTime(value) {
    const duration = Number.isFinite(audio.duration) ? audio.duration : Infinity;
    return Math.max(0, Math.min(value, duration));
}

function buildTimeKey(value) {
    return Number(value).toFixed(2);
}

function buildWords(transcript) {
    const items = [];
    (transcript.segments || []).forEach((segment, segmentIndex) => {
        (segment.words || []).forEach((word, wordIndexInSegment) => {
            items.push({
                id: `${segmentIndex}-${wordIndexInSegment}`,
                word: word.word,
                start: Number(word.start),
                end: Number(word.end),
                segmentIndex,
            });
        });
    });
    return items;
}

function cloneWords(items) {
    return items.map((word) => ({ ...word }));
}

function markHighlightKinds(items) {
    return items.map((word) => {
        const key = buildTimeKey(word.start);
        return {
            ...word,
            isKeyword: keywordStarts.has(key),
            isJargon: jargonStarts.has(key),
        };
    });
}

function syncHighlightSets(transcript) {
    keywordStarts = new Set(
        (transcript.custom_keywords || []).map((item) => buildTimeKey(item.time)),
    );
    jargonStarts = new Set(
        (transcript.jargon_keywords || []).map((item) => buildTimeKey(item.time)),
    );
}

function getWordDelta(index) {
    if (index < 0 || index >= words.length || index >= originalWords.length) return 0;
    return words[index].start - originalWords[index].start;
}

function updateDirtyState() {
    const dirty = words.some((word, index) =>
        Math.abs(word.start - originalWords[index].start) > 0.0001 ||
        Math.abs(word.end - originalWords[index].end) > 0.0001,
    );
    resetButton.disabled = !dirty;
    saveButton.disabled = !dirty || !transcriptId;
    if (!dirty && saveStatusEl.textContent === 'Unsaved changes') {
        saveStatusEl.textContent = 'Idle';
    }
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

async function loadTranscript(nextTranscriptId) {
    const response = await fetch(`/api/transcript/${nextTranscriptId}`);
    transcriptData = await response.json();
    transcriptId = nextTranscriptId;
    syncHighlightSets(transcriptData);
    words = markHighlightKinds(buildWords(transcriptData));
    originalWords = cloneWords(words);
    selectedWordIndex = words.length > 0 ? 0 : -1;
    activeWordIndex = -1;
    dragState = null;
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
    syncActiveWord();
    saveStatusEl.textContent = 'Idle';
    updateDirtyState();
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
        if (word.isKeyword) button.classList.add('keyword');
        if (word.isJargon) button.classList.add('jargon');
        if (Math.abs(getWordDelta(index)) > 0.0001) button.classList.add('shifted');
        button.dataset.index = String(index);
        const badges = [];
        if (word.isKeyword) badges.push('<span class="badge">keyword</span>');
        if (word.isJargon) badges.push('<span class="badge">jargon</span>');
        button.innerHTML = `
            <span class="word-text">${word.word}</span>
            <span class="word-meta">
                <span>${formatSeconds(word.start)}-${formatSeconds(word.end)}</span>
                ${badges.join('')}
            </span>
        `;
        button.addEventListener('click', () => {
            selectedWordIndex = index;
            seekToSelectedWord({ play: true });
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

    const duration = getTimelineDuration();
    const timelineWidth = getTimelineWidth(duration);
    timeline.style.width = `${timelineWidth}px`;
    const cursor = document.createElement('div');
    cursor.className = 'timeline-cursor';
    timeline.appendChild(cursor);

    words.forEach((word, index) => {
        const bar = document.createElement('button');
        bar.type = 'button';
        bar.className = 'timeline-word';
        if (word.isKeyword) bar.classList.add('keyword');
        if (word.isJargon) bar.classList.add('jargon');
        if (Math.abs(getWordDelta(index)) > 0.0001) bar.classList.add('shifted');
        bar.dataset.index = String(index);
        const left = (word.start / duration) * timelineWidth;
        const width = Math.max(14, ((word.end - word.start) / duration) * timelineWidth);
        bar.style.left = `${left}px`;
        bar.style.width = `${width}px`;
        bar.title = `${word.word} ${formatSeconds(word.start)}-${formatSeconds(word.end)}`;
        bar.textContent = word.word;
        bar.addEventListener('click', () => {
            selectedWordIndex = index;
            seekToSelectedWord({ play: true });
            renderWordList();
            renderTimeline();
            updateSelectionDetails();
        });
        bar.addEventListener('pointerdown', (event) => beginDrag(event, index));
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
    const duration = getTimelineDuration();
    const timelineWidth = getTimelineWidth(duration);
    cursor.style.left = `${(clampTime(audio.currentTime) / duration) * timelineWidth}px`;
}

function getWordAtTime(timeSeconds) {
    return words.findIndex((word) => timeSeconds >= word.start && timeSeconds <= word.end);
}

function syncActiveWord() {
    activeWordIndex = getWordAtTime(audio.currentTime);
    currentWordEl.textContent = activeWordIndex >= 0 ? words[activeWordIndex].word : '-';
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
        playSelectionButton.disabled = true;
        dragDeltaEl.textContent = '0.00s';
        return;
    }

    const delta = getWordDelta(selectedWordIndex);
    const flags = [
        word.isKeyword ? 'keyword' : '',
        word.isJargon ? 'jargon' : '',
    ].filter(Boolean).join(', ') || '-';
    const values = [
        word.word,
        formatSeconds(word.start),
        formatSeconds(word.end),
        formatSeconds(word.end - word.start),
        formatSignedSeconds(delta),
        flags,
    ];
    detailValues.forEach((detail, index) => {
        detail.textContent = values[index];
    });
    dragDeltaEl.textContent = formatSignedSeconds(delta);
    playSelectionButton.disabled = false;
}

function seekToSelectedWord({ play = false } = {}) {
    const word = words[selectedWordIndex];
    if (!word) return;
    console.log(`seekToSelectedWord: index=${selectedWordIndex}, word="${word.word}", start=${word.start}, currentTime before=${audio.currentTime}`);
    audio.currentTime = clampTime(word.start);
    console.log(`seekToSelectedWord: currentTime after=${audio.currentTime}`);
    if (play) {
        audio.play().catch(() => {});
    }
}

function playSelection() {
    const word = words[selectedWordIndex];
    if (!word) return;
    audio.currentTime = clampTime(word.start);
    audio.play().catch(() => {});
    window.clearTimeout(playSelection._timeoutId);
    playSelection._timeoutId = window.setTimeout(() => audio.pause(), Math.max(120, (word.end - word.start) * 1000));
}

function moveSelection(delta) {
    if (words.length === 0) return;
    selectedWordIndex = Math.max(0, Math.min(words.length - 1, selectedWordIndex + delta));
    seekToSelectedWord();
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
}

function beginDrag(event, index) {
    event.preventDefault();
    event.stopPropagation();
    selectedWordIndex = index;
    const duration = getTimelineDuration();
    dragState = {
        index,
        pointerId: event.pointerId,
        startX: event.clientX,
        duration,
        timelineWidth: Math.max(1, getTimelineWidth(duration)),
        initialWords: cloneWords(words),
    };
    const bar = event.currentTarget;
    bar.classList.add('dragging');
    saveStatusEl.textContent = 'Unsaved changes';
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
}

function setZoomLevel(nextZoom) {
    zoomLevel = Math.max(1, Math.min(12, Number(nextZoom) || 1));
    if (zoomRange) zoomRange.value = String(zoomLevel);
    updateZoomReadout();
    renderTimeline();
}

function onDrag(event) {
    if (!dragState) return;
    const scale = Number(dragScaleInput.value) || 1;
    const deltaX = event.clientX - dragState.startX;
    const deltaSeconds = (deltaX / dragState.timelineWidth) * dragState.duration * scale;
    const anchorIndex = dragState.index;
    const anchorDuration = dragState.initialWords[anchorIndex].end - dragState.initialWords[anchorIndex].start;
    const minDelta = -dragState.initialWords[anchorIndex].start;
    const clampedDelta = Math.max(minDelta, deltaSeconds);

    words = dragState.initialWords.map((word, index) => {
        if (index < anchorIndex) return { ...word };
        return {
            ...word,
            start: word.start + clampedDelta,
            end: Math.max(word.start + clampedDelta, word.start + clampedDelta + (word.end - word.start)),
        };
    });

    if (words[anchorIndex]) {
        words[anchorIndex].end = words[anchorIndex].start + anchorDuration;
    }

    renderWordList();
    renderTimeline();
    updateSelectionDetails();
    updateDirtyState();
}

function endDrag(event) {
    if (!dragState || event.pointerId !== dragState.pointerId) return;
    const draggingBar = timeline.querySelector('.timeline-word.dragging');
    if (draggingBar) draggingBar.classList.remove('dragging');
    dragState = null;
    updateDirtyState();
}

function resetWords() {
    words = cloneWords(originalWords);
    saveStatusEl.textContent = 'Idle';
    renderWordList();
    renderTimeline();
    updateSelectionDetails();
    updateDirtyState();
}

async function saveWords() {
    if (!transcriptId) return;
    saveButton.disabled = true;
    saveStatusEl.textContent = 'Saving...';
    const response = await fetch(`/api/transcript/${encodeURIComponent(transcriptId)}/timestamps`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            words: words.map(({ start, end }) => ({ start, end })),
        }),
    });
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        saveStatusEl.textContent = payload.error || 'Save failed';
        updateDirtyState();
        return;
    }
    const payload = await response.json().catch(() => ({}));
    const savedTranscriptId = payload.transcript_id || transcriptId;
    saveStatusEl.textContent = `Saved as ${savedTranscriptId}`;
    await refreshFileOptions(savedTranscriptId);
    await loadTranscript(savedTranscriptId);
}

async function refreshFileOptions(selectTranscriptId) {
    const previousValue = fileSelect.value;
    fileSelect.innerHTML = '<option value="">Select an MP3 clip</option>';
    await loadFiles();

    if (selectTranscriptId) {
        const match = Array.from(fileSelect.options).find((option) => {
            if (!option.value) return false;
            try {
                return JSON.parse(option.value).transcript_id === selectTranscriptId;
            } catch {
                return false;
            }
        });
        if (match) {
            fileSelect.value = match.value;
            syncSelectedFileFromSelect();
            return;
        }
    }

    fileSelect.value = previousValue;
    syncSelectedFileFromSelect();
}

function handleFileChange() {
    const selected = fileSelect.value;
    if (!selected) return;
    const file = JSON.parse(selected);
    selectedFile = file;
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
zoomRange.addEventListener('input', () => setZoomLevel(zoomRange.value));
zoomOutButton.addEventListener('click', () => setZoomLevel(zoomLevel - 1));
zoomInButton.addEventListener('click', () => setZoomLevel(zoomLevel + 1));
playSelectionButton.addEventListener('click', playSelection);
resetButton.addEventListener('click', resetWords);
saveButton.addEventListener('click', saveWords);

audio.addEventListener('timeupdate', syncActiveWord);
audio.addEventListener('loadedmetadata', updateTimeReadout);
window.addEventListener('resize', () => renderTimeline());
timeline.addEventListener('pointermove', onDrag);
timeline.addEventListener('pointerup', endDrag);
timeline.addEventListener('pointercancel', endDrag);
document.addEventListener('pointermove', onDrag);
document.addEventListener('pointerup', endDrag);
document.addEventListener('pointercancel', endDrag);

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

updateZoomReadout();
loadFiles();
