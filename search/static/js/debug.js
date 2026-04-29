const debugWaveformScroll = document.getElementById('debug-waveform-scroll');
const debugWaveformTrack = document.getElementById('debug-waveform-track');
const debugPlayhead = document.getElementById('debug-playhead');
const debugWaveformStatus = document.getElementById('debug-waveform-status');

const DEBUG_BAR_COUNT = 480;
const DEBUG_BAR_WIDTH = 4;
const DEBUG_BAR_GAP = 2;
const DEBUG_TRACK_PADDING = 16;

let debugWaveformLevels = [];
let debugWaveformSource = null;
let debugAutoScrollEnabled = true;

function setDebugWaveformStatus(message) {
    debugWaveformStatus.textContent = message;
}

function clearDebugWaveform() {
    debugWaveformLevels = [];
    debugWaveformTrack.querySelectorAll('.debug-waveform-bar').forEach((bar) => bar.remove());
    debugPlayhead.style.left = `${DEBUG_TRACK_PADDING}px`;
    debugWaveformTrack.style.width = '100%';
}

function buildDebugWaveform(channelData) {
    const blockSize = Math.max(1, Math.floor(channelData.length / DEBUG_BAR_COUNT));
    const levels = [];

    for (let i = 0; i < DEBUG_BAR_COUNT; i += 1) {
        const start = i * blockSize;
        const end = Math.min(channelData.length, start + blockSize);
        let sum = 0;

        for (let cursor = start; cursor < end; cursor += 1) {
            sum += Math.abs(channelData[cursor]);
        }

        const mean = end > start ? sum / (end - start) : 0;
        levels.push(Math.min(1, Math.pow(mean * 3.2, 0.85)));
    }

    return levels;
}

async function decodeDebugWaveform(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`audio fetch failed: ${response.status}`);
    }

    const arrayBuffer = await response.arrayBuffer();
    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextCtor();

    try {
        const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
        return buildDebugWaveform(audioBuffer.getChannelData(0));
    } finally {
        await context.close();
    }
}

function renderDebugWaveform(levels) {
    clearDebugWaveform();
    debugWaveformLevels = levels;

    const trackWidth = Math.max(
        debugWaveformScroll.clientWidth - 20,
        DEBUG_TRACK_PADDING * 2 + levels.length * (DEBUG_BAR_WIDTH + DEBUG_BAR_GAP),
    );
    debugWaveformTrack.style.width = `${trackWidth}px`;

    const fragment = document.createDocumentFragment();
    levels.forEach((level) => {
        const bar = document.createElement('div');
        bar.className = 'debug-waveform-bar';
        bar.style.height = `${Math.max(8, Math.round(level * 140))}px`;
        fragment.appendChild(bar);
    });

    debugWaveformTrack.appendChild(fragment);
    debugWaveformTrack.appendChild(debugPlayhead);
    updateDebugPlayhead();
}

function updateDebugPlayhead() {
    const duration = audio.duration || 0;
    const usableWidth = debugWaveformTrack.clientWidth - DEBUG_TRACK_PADDING * 2;
    const ratio = duration > 0 ? audio.currentTime / duration : 0;
    const left = DEBUG_TRACK_PADDING + usableWidth * Math.min(1, Math.max(0, ratio));

    debugPlayhead.style.left = `${left}px`;

    if (debugAutoScrollEnabled) {
        const targetLeft = Math.max(0, left - debugWaveformScroll.clientWidth / 2);
        debugWaveformScroll.scrollTo({ left: targetLeft, behavior: 'smooth' });
    }
}

async function loadDebugWaveformForCurrentAudio() {
    const selectedValue = fileSelect?.value;
    if (!selectedValue) {
        clearDebugWaveform();
        setDebugWaveformStatus('Select audio to render waveform.');
        return;
    }

    let parsedFile;
    try {
        parsedFile = JSON.parse(selectedValue);
    } catch {
        setDebugWaveformStatus('Audio selection is not ready yet.');
        return;
    }

    const waveformUrl = `/mp3/${encodeURIComponent(parsedFile.filename)}`;
    if (debugWaveformSource === waveformUrl) {
        updateDebugPlayhead();
        return;
    }

    debugWaveformSource = waveformUrl;
    setDebugWaveformStatus(`Rendering ${parsedFile.filename}...`);

    try {
        const levels = await decodeDebugWaveform(waveformUrl);
        renderDebugWaveform(levels);
        setDebugWaveformStatus(parsedFile.filename);
    } catch (error) {
        clearDebugWaveform();
        setDebugWaveformStatus(`Waveform error: ${error.message}`);
    }
}

function ensureDebugDefaultSelection() {
    if (!fileSelect || fileSelect.value || fileSelect.options.length === 0) return;
    const firstRealOption = Array.from(fileSelect.options).find((option) => option.value);
    if (!firstRealOption) return;

    fileSelect.value = firstRealOption.value;
    fileSelect.dispatchEvent(new Event('change'));
}

if (fileSelect) {
    fileSelect.addEventListener('change', () => {
        debugWaveformSource = null;
        loadDebugWaveformForCurrentAudio();
    });
}

audio.addEventListener('loadedmetadata', updateDebugPlayhead);
audio.addEventListener('timeupdate', updateDebugPlayhead);
audio.addEventListener('seeked', updateDebugPlayhead);
audio.addEventListener('play', updateDebugPlayhead);

debugWaveformScroll.addEventListener('pointerdown', () => {
    debugAutoScrollEnabled = false;
});

debugWaveformScroll.addEventListener('pointerup', () => {
    debugAutoScrollEnabled = true;
});

debugWaveformScroll.addEventListener('mouseleave', () => {
    debugAutoScrollEnabled = true;
});

window.addEventListener('resize', () => {
    if (debugWaveformLevels.length > 0) {
        renderDebugWaveform(debugWaveformLevels);
    }
});

window.addEventListener('load', () => {
    ensureDebugDefaultSelection();
    loadDebugWaveformForCurrentAudio();
});
