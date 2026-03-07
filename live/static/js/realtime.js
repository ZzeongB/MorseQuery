/**
 * Realtime Words - Main JavaScript
 */

// Socket.IO connection
const socket = io();

// ============================================================================
// State Variables
// ============================================================================

let view = 'single';
let descView = 'single';
let source = 'mic';
let options = [];
let currentIdx = 0;
let summaryBuffer = '';
let summaryTimer = null;
let lastSpace = 0;
let infoVisible = false;
let dismissMode = 'summary'; // 'summary' or 'none'
let summaryRequested = false; // Track if we actually requested a summary
let summaryInProgress = false; // summarizing or tts playback in progress
let showSummaryTextEnabled = true;
let autoPreSummarizeEnabled = true;
let pendingSummaryTexts = [];
let summaryFinalizeTimer = null;
let awaitingJudgeDecision = false;
let allowPostFollowupTts = false;
let keywordOutputMode = 'audio'; // 'text' | 'audio'
let judgeEnabled = false; // Judge agent on/off (when off, summaries play without judgment)
let reconstructorEnabled = true; // Conversation reconstructor on/off
let transcriptCompressionMode = 'fastest'; // 'fastest' | 'realtime' | 'api_mini' | 'api_nano'
let keywordTtsPlaying = false; // Track if keyword TTS is playing
let keywordTtsCurrentText = '';
let keywordAutoSummarizeTimer = null;
let keywordPlaybackToken = 0;
let pendingSummarizeIndicatorAfterKeyword = false;
let keywordTtsPreloadedTexts = new Set(); // Track preloaded keyword TTS texts
let configuredSummaryClientCount = 2;
let summarySegmentState = new Map(); // segmentId -> {received, nonEmpty}
let pendingEmptySummarySignal = null; // {segmentId}
let pendingSkippedIndicator = null; // {message, opts}
let listeningActive = false;
let summaryTriggeredForListeningSession = false;
let inferencingTimer = null;
const INFERENCING_TIMEOUT_MS = 5000; // 5 seconds timeout
const KEYWORD_SUMMARY_LEAD_MS = 2000;
const SUMMARY_FOLLOWUP_LEAD_MS = 2000;
let keywordSummaryLeadMs = KEYWORD_SUMMARY_LEAD_MS;
const KEYWORD_ESTIMATED_WPS = 3.5; // Cartesia speed=1.2 approximation
let pendingReconstructedTurns = [];
let pendingReconstructedSegmentId = 0;
let renderedReconstructedSegmentId = 0;
let baseReconstructedTurns = [];
let baseReconstructedSegmentId = 0;
let followupReconstructedGroups = []; // [{segmentId, turns}]
let widgetHeight = 0; // 0 = bottom, 100 = top
let isDragging = false;
let availableDevices = [];
let availableOutputDevices = [];
const micSelectIds = ['summaryMic1', 'summaryMic2', 'keywordMic'];

// Noise gate calibration
let noiseGateEnabled = false;
let noiseGateThreshold = 500;  // RMS value (0-32768 scale)
let noiseGateCurrentRMS = { sum1: 0, sum2: 0 };
const NOISE_GATE_MAX_RMS = 5000;  // Max display scale

// Audio feedback
let audioFeedbackMode = 'on'; // 'on' | 'verbal' | 'off'
let airpodsModeSwitchEnabled = true;
let audioContext = null;
let audioUnlocked = false;
let loadingAudioInterval = null;
let loadingAudioToken = 0;
let loadingAudioStartTimer = null;
let verbalCueLastAt = {};
let browserSummaryPlaybackActive = false;
let activeBrowserTtsType = null;
let suppressNextBrowserTtsEndCue = false;
let ttsCueLastPlayedAt = {
    keyword_start: 0,
    keyword_end: 0,
    summary_start: 0,
    summary_end: 0,
};
const AUDIO_FEEDBACK_GAIN_MULTIPLIER = 6.0;
const AUDIO_FEEDBACK_MAX_GAIN = 0.8;
const AUDIO_CUE_DEDUP_WINDOWS = {
    keyword_start: 220,
    keyword_end: 220,
    summary_start: 1400,
    summary_end: 1400,
};

// TTS playback
let ttsQueue = [];
let ttsPlaying = false;
let currentTtsAudio = null;
let currentTtsUrl = null;
let currentTtsSource = null;  // Web Audio API source node
let currentSummaryNearEndTimer = null;
let skippedIndicatorTimer = null;
let dismissTimer = null;

// ============================================================================
// Utility Functions
// ============================================================================

function emitWithAck(event, payload = {}, timeoutMs = 8000) {
    return new Promise((resolve) => {
        let done = false;
        const timer = setTimeout(() => {
            if (done) return;
            done = true;
            resolve(null);
        }, timeoutMs);

        socket.emit(event, payload, (ack) => {
            if (done) return;
            done = true;
            clearTimeout(timer);
            resolve(ack || null);
        });
    });
}

async function waitForAncBeforeBrowserSummaryPlayback(meta) {
    if (!meta) return true;
    if (meta.type !== 'summary' && meta.type !== 'reconstruction') return true;
    if (browserSummaryPlaybackActive) return true;

    const ack = await emitWithAck('browser_tts_playback_start', {
        source: meta.type,
        segment_id: Number(meta.segmentId || 0),
    });
    const ok = !ack || ack.ok !== false;
    if (ok) {
        browserSummaryPlaybackActive = true;
    }
    return ok;
}

// ============================================================================
// Noise Gate Functions
// ============================================================================

function toggleNoiseGate() {
    noiseGateEnabled = document.getElementById('noiseGateEnabled').checked;
    const container = document.getElementById('noiseGateContainer');
    container.style.display = noiseGateEnabled ? 'flex' : 'none';

    if (noiseGateEnabled) {
        startNoiseGateMonitor();
        initNoiseGateMetersClick();
    } else {
        stopNoiseGateMonitor();
    }
}

function startNoiseGateMonitor() {
    const sum1Select = document.getElementById('summaryMic1');
    const sum2Select = document.getElementById('summaryMic2');
    const deviceIndices = [];
    const micIds = [];

    if (sum1Select.value) {
        deviceIndices.push(parseInt(sum1Select.value));
        micIds.push('sum1');
    }
    if (sum2Select.value) {
        deviceIndices.push(parseInt(sum2Select.value));
        micIds.push('sum2');
    }

    if (deviceIndices.length > 0) {
        socket.emit('start_noise_gate_monitor', { device_indices: deviceIndices, mic_ids: micIds });
    }
}

function stopNoiseGateMonitor() {
    socket.emit('stop_noise_gate_monitor');
}

function updateNoiseGateUI(micId, rms) {
    const suffix = micId === 'sum1' ? '1' : '2';
    noiseGateCurrentRMS[micId] = rms;

    // Update level bar (percentage of max scale)
    const levelPercent = Math.min(100, (rms / NOISE_GATE_MAX_RMS) * 100);
    const levelEl = document.getElementById('noiseGateLevel' + suffix);
    if (levelEl) levelEl.style.width = levelPercent + '%';

    // Update current level text
    const currentEl = document.getElementById('noiseGateCurrentLevel' + suffix);
    if (currentEl) currentEl.textContent = Math.round(rms);

    // Update gate state
    const isOpen = rms >= noiseGateThreshold;
    const stateEl = document.getElementById('noiseGateState' + suffix);
    if (stateEl) {
        stateEl.textContent = isOpen ? 'OPEN' : 'CLOSED';
        stateEl.className = 'gate-state ' + (isOpen ? 'open' : 'closed');
    }
}

function updateNoiseGateThresholdUI() {
    const thresholdPercent = Math.min(100, (noiseGateThreshold / NOISE_GATE_MAX_RMS) * 100);

    ['1', '2'].forEach(suffix => {
        const thresholdEl = document.getElementById('noiseGateThreshold' + suffix);
        const filteredEl = document.getElementById('noiseGateFilteredZone' + suffix);
        if (thresholdEl) thresholdEl.style.left = thresholdPercent + '%';
        if (filteredEl) filteredEl.style.width = thresholdPercent + '%';
    });

    document.getElementById('noiseGateThresholdValue').textContent = Math.round(noiseGateThreshold);
    document.getElementById('noiseGateThresholdSlider').value = noiseGateThreshold;

    updateNoiseGateUI('sum1', noiseGateCurrentRMS.sum1);
    updateNoiseGateUI('sum2', noiseGateCurrentRMS.sum2);
}

function updateNoiseGateThresholdFromSlider() {
    noiseGateThreshold = parseInt(document.getElementById('noiseGateThresholdSlider').value);
    updateNoiseGateThresholdUI();
}

function initNoiseGateMetersClick() {
    ['1', '2'].forEach(suffix => {
        const meter = document.getElementById('noiseGateMeter' + suffix);
        if (meter) {
            meter.onclick = function(e) {
                const rect = meter.getBoundingClientRect();
                const relativeX = e.clientX - rect.left;
                const percentage = Math.max(0, Math.min(100, (relativeX / rect.width) * 100));
                noiseGateThreshold = (percentage / 100) * NOISE_GATE_MAX_RMS;
                updateNoiseGateThresholdUI();
            };
        }
    });

    updateNoiseGateThresholdUI();
}

// ============================================================================
// Device/Mic Functions
// ============================================================================

async function fetchDevices() {
    try {
        const resp = await fetch('/api/devices');
        availableDevices = await resp.json();
        populateMicSelects();
    } catch (e) {
        console.error('Failed to fetch devices:', e);
    }
}

async function fetchOutputDevices() {
    try {
        const resp = await fetch('/api/output_devices');
        availableOutputDevices = await resp.json();
        populateOutputSelect();
    } catch (e) {
        console.error('Failed to fetch output devices:', e);
    }
}

function populateOutputSelect() {
    const sel = document.getElementById('ttsOutput');
    sel.innerHTML = '<option value="">Default</option>';
    availableOutputDevices.forEach(d => {
        sel.innerHTML += `<option value="${d.index}">[${d.index}] ${d.name}</option>`;
    });
}

function populateMicSelects() {
    micSelectIds.forEach((id, idx) => {
        const sel = document.getElementById(id);
        sel.innerHTML = '<option value="">Default</option>';
        availableDevices.forEach(d => {
            sel.innerHTML += `<option value="${d.index}">[${d.index}] ${d.name}</option>`;
        });
        // Auto-select: keywordMic always uses Macbook mic
        if (id === 'keywordMic') {
            const macbookMic = availableDevices.find(d => d.name.toLowerCase().includes('macbook'));
            if (macbookMic) {
                sel.value = macbookMic.index;
            } else if (availableDevices.length > 0) {
                sel.value = availableDevices[0].index;
            }
        } else if (availableDevices.length > idx) {
            sel.value = availableDevices[idx].index;
        }
        sel.addEventListener('change', updateMicMonitor);
    });
    updateMicMonitor();
}

function updateMicMonitor() {
    const deviceIndices = [];
    const selectIds = [];
    micSelectIds.forEach(id => {
        const sel = document.getElementById(id);
        if (sel.value) {
            deviceIndices.push(parseInt(sel.value));
            selectIds.push(id);
        }
    });
    if (deviceIndices.length > 0) {
        socket.emit('start_mic_monitor', { device_indices: deviceIndices, select_ids: selectIds });
    } else {
        socket.emit('stop_mic_monitor');
    }
}

function stopAllMicMonitors() {
    socket.emit('stop_mic_monitor');
    micSelectIds.forEach(id => {
        const levelBar = document.getElementById(id + 'Level');
        if (levelBar) levelBar.style.width = '0%';
    });
}

// ============================================================================
// Settings Toggle Functions
// ============================================================================

function setSource(s) {
    source = s;
    document.getElementById('btn-mic').classList.toggle('selected', s === 'mic');
    document.getElementById('btn-mp3').classList.toggle('selected', s === 'mp3');
    document.getElementById('micSelectSection').style.display = s === 'mic' ? 'block' : 'none';
    document.getElementById('mp3SourceSection').style.display = s === 'mp3' ? 'block' : 'none';
    if (s === 'mic' && availableDevices.length === 0) {
        fetchDevices();
    } else if (s === 'mp3') {
        stopAllMicMonitors();
    }
}

function setDismissMode(dm) {
    dismissMode = dm;
    document.getElementById('btn-dismiss-summary').classList.toggle('selected', dm === 'summary');
    document.getElementById('btn-dismiss-none').classList.toggle('selected', dm === 'none');
}

function setKeywordOutputMode(mode) {
    keywordOutputMode = mode;
    document.getElementById('btn-keyword-text').classList.toggle('selected', mode === 'text');
    document.getElementById('btn-keyword-audio').classList.toggle('selected', mode === 'audio');
}

function setSummaryTextMode(enabled) {
    showSummaryTextEnabled = enabled;
    document.getElementById('btn-summary-text-on').classList.toggle('selected', enabled);
    document.getElementById('btn-summary-text-off').classList.toggle('selected', !enabled);
    if (!enabled) {
        hideSummaryText();
    } else if (summaryInProgress && pendingSummaryTexts.length > 0) {
        showSummaryText();
    }
}

function setAutoPreSummarizeEnabled(enabled) {
    autoPreSummarizeEnabled = enabled;
    document.getElementById('btn-auto-presum-on').classList.toggle('selected', enabled);
    document.getElementById('btn-auto-presum-off').classList.toggle('selected', !enabled);
    if (!enabled) {
        clearKeywordAutoSummarizeTimer();
    }
}

function setKeywordSummaryLeadMs(ms) {
    const value = Number(ms);
    if (!Number.isFinite(value) || value < KEYWORD_SUMMARY_LEAD_MS) {
        keywordSummaryLeadMs = KEYWORD_SUMMARY_LEAD_MS;
        return;
    }
    keywordSummaryLeadMs = Math.round(value);
}

function setJudgeEnabled(enabled) {
    judgeEnabled = enabled;
    document.getElementById('btn-judge-on').classList.toggle('selected', enabled);
    document.getElementById('btn-judge-off').classList.toggle('selected', !enabled);
}

function setReconstructorEnabled(enabled) {
    reconstructorEnabled = enabled;
    document.getElementById('btn-reconstructor-on').classList.toggle('selected', enabled);
    document.getElementById('btn-reconstructor-off').classList.toggle('selected', !enabled);
}

function setTranscriptCompressionMode(mode) {
    transcriptCompressionMode = mode;
    document.getElementById('btn-compress-fastest').classList.toggle('selected', mode === 'fastest');
    document.getElementById('btn-compress-realtime').classList.toggle('selected', mode === 'realtime');
    document.getElementById('btn-compress-api-mini').classList.toggle('selected', mode === 'api_mini');
    document.getElementById('btn-compress-api-nano').classList.toggle('selected', mode === 'api_nano');
}

function isToneFeedbackEnabled() {
    return audioFeedbackMode !== 'off';
}

function isVerbalFeedbackEnabled() {
    return audioFeedbackMode === 'verbal';
}

function speakFeedback(text, dedupMs = 450) {
    if (!isVerbalFeedbackEnabled()) return;
    if (!('speechSynthesis' in window)) return;
    const key = (text || '').toLowerCase();
    const now = Date.now();
    const last = verbalCueLastAt[key] || 0;
    if (now - last < dedupMs) return;
    verbalCueLastAt[key] = now;
    try {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.0;
        utterance.pitch = 1.0;
        utterance.volume = 1.0;
        window.speechSynthesis.speak(utterance);
    } catch (e) {}
}

function setAudioFeedbackMode(mode) {
    const normalized = mode === 'verbal' ? 'verbal' : (mode === 'off' ? 'off' : 'on');
    audioFeedbackMode = normalized;
    document.getElementById('btn-audio-feedback-on').classList.toggle('selected', normalized === 'on');
    const verbalBtn = document.getElementById('btn-audio-feedback-verbal');
    if (verbalBtn) verbalBtn.classList.toggle('selected', normalized === 'verbal');
    document.getElementById('btn-audio-feedback-off').classList.toggle('selected', normalized === 'off');
    if (normalized === 'off') {
        stopLoadingAudioFeedback();
    }
    if (normalized === 'off' && 'speechSynthesis' in window) {
        try { window.speechSynthesis.cancel(); } catch (e) {}
        return;
    }
    // Mode button click is a user gesture: unlock WebAudio so tone can play in verbal mode too.
    unlockAudioFeedback();
}

// Backward compatibility for older button handlers/calls
function setAudioFeedbackEnabled(enabled) {
    setAudioFeedbackMode(enabled ? 'on' : 'off');
}

function setAirpodsModeSwitchEnabled(enabled) {
    airpodsModeSwitchEnabled = enabled;
    document.getElementById('btn-airpods-mode-on').classList.toggle('selected', enabled);
    document.getElementById('btn-airpods-mode-off').classList.toggle('selected', !enabled);
    socket.emit('set_airpods_mode_switch', { enabled });
}

function setDescView(dv) {
    descView = dv;
    document.getElementById('btn-desc-single').classList.toggle('selected', dv === 'single');
    document.getElementById('btn-desc-all').classList.toggle('selected', dv === 'all');
}

// ============================================================================
// Audio Feedback Functions
// ============================================================================

async function ensureAudioContext() {
    if (!isToneFeedbackEnabled()) return null;
    if (!audioContext || audioContext.state === 'closed') {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return null;
        audioContext = new Ctx();
        audioUnlocked = false;
    }
    if (audioContext.state === 'suspended') {
        try {
            await audioContext.resume();
            audioUnlocked = true;
        } catch (e) {}
    }
    if (audioContext.state === 'running') {
        audioUnlocked = true;
    }
    return audioContext;
}

async function unlockAudioFeedback() {
    const ctx = await ensureAudioContext();
    if (!ctx || audioUnlocked) return;
    try {
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        const now = ctx.currentTime;
        gain.gain.setValueAtTime(0.0001, now);
        osc.frequency.setValueAtTime(440, now);
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.start(now);
        osc.stop(now + 0.01);
        audioUnlocked = true;
    } catch (e) {}
}

async function playTone(freq, durationMs = 70, gainValue = 0.06, type = 'sine') {
    const ctx = await ensureAudioContext();
    if (!ctx) return;
    if (ctx.state !== 'running') return;

    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    const now = ctx.currentTime;
    const end = now + durationMs / 1000;
    const effectiveGain = Math.min(
        AUDIO_FEEDBACK_MAX_GAIN,
        gainValue * AUDIO_FEEDBACK_GAIN_MULTIPLIER
    );
    osc.type = type;
    osc.frequency.setValueAtTime(freq, now);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(effectiveGain, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, end);
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(end);
}

function playConfirmedFeedback() {
    if (!isToneFeedbackEnabled()) return;
    playTone(740, 55, 0.07, 'triangle');
    setTimeout(() => playTone(988, 60, 0.078, 'triangle'), 78);
    setTimeout(() => playTone(1318, 78, 0.086, 'triangle'), 162);
}

function playTapFeedback() {
    if (!isToneFeedbackEnabled()) return;
    playTone(900, 45, 0.048, 'sine');
}

function playCompleteFeedback() {
    if (!isToneFeedbackEnabled()) return;
    playTone(780, 60, 0.042, 'sine');
    setTimeout(() => playTone(620, 90, 0.045, 'sine'), 85);
}

function shouldPlayCue(cueId) {
    const now = Date.now();
    const last = ttsCueLastPlayedAt[cueId] || 0;
    const dedupMs = AUDIO_CUE_DEDUP_WINDOWS[cueId] || 220;
    if (now - last < dedupMs) return false;
    ttsCueLastPlayedAt[cueId] = now;
    return true;
}

function playTtsStartFeedback(type) {
    if (isVerbalFeedbackEnabled()) {
        if (type === 'keyword') {
            speakFeedback('Keyword start!', 400);
        } else {
            speakFeedback('Summary start!', 500);
        }
    }
    if (!isToneFeedbackEnabled()) return;
    if (type === 'keyword') {
        if (!shouldPlayCue('keyword_start')) return;
        playTone(1397, 52, 0.06, 'square');
        setTimeout(() => playTone(1760, 60, 0.066, 'square'), 76);
        return;
    }
    if (!shouldPlayCue('summary_start')) return;
    playTone(622, 70, 0.058, 'triangle');
    setTimeout(() => playTone(784, 90, 0.064, 'triangle'), 92);
}

function playTtsEndFeedback(type) {
    if (isVerbalFeedbackEnabled()) {
        if (type === 'summary') {
            speakFeedback('Summary end!', 500);
        }
    }
    if (!isToneFeedbackEnabled()) return;
    if (type === 'keyword') {
        if (!shouldPlayCue('keyword_end')) return;
        playTone(1760, 60, 0.058, 'square');
        setTimeout(() => playTone(1318, 88, 0.062, 'square'), 78);
        return;
    }
    if (!shouldPlayCue('summary_end')) return;
    playTone(784, 80, 0.056, 'triangle');
    setTimeout(() => playTone(523, 110, 0.062, 'triangle'), 98);
}

function startLoadingAudioFeedback(mode = 'inferring', startDelayMs = 0) {
    if (audioFeedbackMode === 'off') return;
    stopLoadingAudioFeedback();
    loadingAudioToken += 1;
    const token = loadingAudioToken;
    const isSummarizing = mode === 'summarizing';
    const safeDelay = Math.max(0, Number(startDelayMs) || 0);

    if (isVerbalFeedbackEnabled()) {
        speakFeedback(isSummarizing ? 'Summarizing!' : 'Keyword inferring!', 900);
    }

    const loop = () => {
        if (!isToneFeedbackEnabled() || token !== loadingAudioToken) return;
        if (isSummarizing) {
            playTone(560, 90, 0.038, 'triangle');
            setTimeout(() => {
                if (token !== loadingAudioToken) return;
                playTone(420, 110, 0.04, 'triangle');
            }, 130);
        } else {
            playTone(1046, 45, 0.033, 'sine');
            setTimeout(() => {
                if (token !== loadingAudioToken) return;
                playTone(1318, 48, 0.035, 'sine');
            }, 70);
        }
    };

    if (safeDelay > 0) {
        loadingAudioStartTimer = setTimeout(() => {
            if (!isToneFeedbackEnabled() || token !== loadingAudioToken) return;
            loop();
            loadingAudioInterval = setInterval(loop, isSummarizing ? 1200 : 820);
        }, safeDelay);
        return;
    }
    loop();
    loadingAudioInterval = setInterval(loop, isSummarizing ? 1200 : 820);
}

function stopLoadingAudioFeedback() {
    loadingAudioToken += 1;
    if (loadingAudioStartTimer) {
        clearTimeout(loadingAudioStartTimer);
        loadingAudioStartTimer = null;
    }
    if (loadingAudioInterval) {
        clearInterval(loadingAudioInterval);
        loadingAudioInterval = null;
    }
}

// ============================================================================
// Height Slider Functions
// ============================================================================

function updateHeightSlider(value) {
    widgetHeight = Math.max(0, Math.min(100, value));
    const thumb = document.getElementById('heightThumb');
    const overviewWidget = document.getElementById('overviewWidget');
    const label = document.getElementById('heightLabel');
    const livePreview = document.getElementById('live-preview');

    const thumbPos = 110 - (widgetHeight * 1.1);
    thumb.style.top = thumbPos + 'px';

    const maxBottom = 100;
    const overviewPos = (widgetHeight / 100) * maxBottom;
    overviewWidget.style.bottom = overviewPos + 'px';

    const maxScreenBottom = window.innerHeight - 150;
    const screenPos = 20 + (widgetHeight / 100) * maxScreenBottom;
    livePreview.style.bottom = screenPos + 'px';

    if (widgetHeight < 20) {
        label.textContent = 'Bottom';
    } else if (widgetHeight < 40) {
        label.textContent = 'Lower';
    } else if (widgetHeight < 60) {
        label.textContent = 'Middle';
    } else if (widgetHeight < 80) {
        label.textContent = 'Upper';
    } else {
        label.textContent = 'Top';
    }

    applyWidgetHeight();
}

function showLivePreview() {
    document.getElementById('height-overlay').classList.add('visible');
    document.getElementById('live-preview').classList.add('visible');
}

function hideLivePreview() {
    document.getElementById('height-overlay').classList.remove('visible');
    document.getElementById('live-preview').classList.remove('visible');
}

function applyWidgetHeight() {
    const mainContent = document.getElementById('main-content');
    const maxOffset = window.innerHeight - 200;
    const offset = (widgetHeight / 100) * maxOffset;
    mainContent.style.marginBottom = offset + 'px';
}

function initHeightSlider() {
    const track = document.getElementById('heightTrack');
    const thumb = document.getElementById('heightThumb');

    function handleDrag(e) {
        if (!isDragging) return;
        e.preventDefault();

        const rect = track.getBoundingClientRect();
        const clientY = e.type.includes('touch') ? e.touches[0].clientY : e.clientY;
        const relativeY = clientY - rect.top;
        const percentage = 100 - ((relativeY / rect.height) * 100);

        updateHeightSlider(percentage);
    }

    function startDrag(e) {
        isDragging = true;
        thumb.style.cursor = 'grabbing';
        showLivePreview();
        handleDrag(e);
    }

    function endDrag() {
        if (!isDragging) return;
        isDragging = false;
        thumb.style.cursor = 'grab';
        hideLivePreview();
    }

    thumb.addEventListener('mousedown', startDrag);
    track.addEventListener('mousedown', startDrag);
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', endDrag);

    thumb.addEventListener('touchstart', startDrag, { passive: false });
    track.addEventListener('touchstart', startDrag, { passive: false });
    document.addEventListener('touchmove', handleDrag, { passive: false });
    document.addEventListener('touchend', endDrag);

    updateHeightSlider(0);
}

// ============================================================================
// TTS Playback Functions
// ============================================================================

async function unlockAudioForTts() {
    if (audioUnlocked) return;

    try {
        const ctx = await ensureAudioContext();
        if (ctx && ctx.state === 'running') {
            audioUnlocked = true;
            console.log('Audio unlocked for TTS via AudioContext');
        }
    } catch (e) {
        console.warn('Audio unlock failed:', e);
    }
}

function handleTtsPlaybackStart(meta = null) {
    hideLoadingIndicator();

    if (meta && meta.type === 'summary') {
        if (autoPreSummarizeEnabled) {
            hideInfo();
        }
        if (pendingSummaryTexts.length > 0) {
            showSummaryText();
        }
        return;
    }

    if (meta && meta.type === 'reconstruction') {
        if (autoPreSummarizeEnabled) {
            hideInfo();
        }
        const segId = Number((meta && meta.segmentId) || 0);
        const triggerSource = (meta && meta.triggerSource) || '';
        if (
            pendingReconstructedTurns.length > 0 &&
            pendingReconstructedSegmentId === segId &&
            renderedReconstructedSegmentId !== segId
        ) {
            if (triggerSource === 'post_tts_followup' && baseReconstructedTurns.length > 0) {
                renderReconstructedStack(baseReconstructedTurns, followupReconstructedGroups);
            } else {
                renderReconstructedTurns(pendingReconstructedTurns);
            }
            renderedReconstructedSegmentId = segId;
        }
    }
}

function clearCurrentSummaryNearEndTimer() {
    if (!currentSummaryNearEndTimer) return;
    clearTimeout(currentSummaryNearEndTimer);
    currentSummaryNearEndTimer = null;
}

function scheduleSummaryNearEndSignal(meta, durationSec) {
    clearCurrentSummaryNearEndTimer();
    if (!isSummaryOrReconstructionMeta(meta)) return;
    const durationMs = Math.max(0, Math.round((Number(durationSec) || 0) * 1000));
    const delayMs = Math.max(0, durationMs - SUMMARY_FOLLOWUP_LEAD_MS);
    const segId = Number((meta && meta.segmentId) || 0);
    const ttsType = (meta && meta.type) || 'summary';
    currentSummaryNearEndTimer = setTimeout(() => {
        currentSummaryNearEndTimer = null;
        socket.emit('browser_tts_near_end', {
            reason: 'summary_tts_near_end',
            type: ttsType,
            segment_id: segId,
        });
    }, delayMs);
}

async function playNextTts() {
    if (ttsQueue.length === 0) {
        clearCurrentSummaryNearEndTimer();
        ttsPlaying = false;
        currentTtsAudio = null;
        currentTtsUrl = null;
        currentTtsSource = null;
        if (!suppressNextBrowserTtsEndCue && activeBrowserTtsType) {
            playTtsEndFeedback(activeBrowserTtsType);
        }
        suppressNextBrowserTtsEndCue = false;
        activeBrowserTtsType = null;
        if (browserSummaryPlaybackActive) {
            socket.emit('browser_tts_playback_done', { reason: 'queue_empty' });
            browserSummaryPlaybackActive = false;
        }
        summaryInProgress = false;
        allowPostFollowupTts = false;
        hideSummaryText();
        hideReconstructedTurns();
        pendingSummaryTexts = [];
        clearReconstructedState();
        maybeEmitPendingEmptySummarySignal();
        maybeEmitPendingSkippedIndicator();
        return;
    }
    if (isBlockedByKeywordPlayback(ttsQueue[0].meta)) {
        ttsPlaying = false;
        return;
    }

    ttsPlaying = true;
    summaryInProgress = true;
    const { audioData, url, meta } = ttsQueue.shift();
    const ttsType = (meta && meta.type === 'keyword') ? 'keyword' : 'summary';
    currentTtsUrl = url;
    console.log('playNextTts: playing audio via Web Audio API');

    try {
        const ctx = await ensureAudioContext();
        if (!ctx) {
            console.error('No AudioContext available');
            URL.revokeObjectURL(url);
            playNextTts();
            return;
        }

        const arrayBuffer = new ArrayBuffer(audioData.length);
        const view = new Uint8Array(arrayBuffer);
        for (let i = 0; i < audioData.length; i++) {
            view[i] = audioData.charCodeAt(i);
        }

        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        currentTtsSource = source;

        source.onended = () => {
            clearCurrentSummaryNearEndTimer();
            URL.revokeObjectURL(url);
            currentTtsAudio = null;
            currentTtsUrl = null;
            currentTtsSource = null;
            playNextTts();
        };

        await waitForAncBeforeBrowserSummaryPlayback(meta);
        if (activeBrowserTtsType !== ttsType) {
            playTtsStartFeedback(ttsType);
            activeBrowserTtsType = ttsType;
        }
        source.start(0);
        scheduleSummaryNearEndSignal(meta, audioBuffer.duration);
        console.log('playNextTts: Web Audio playback started');
        handleTtsPlaybackStart(meta);

    } catch (e) {
        console.error('Web Audio playback failed:', e);
        URL.revokeObjectURL(url);
        currentTtsAudio = null;
        currentTtsUrl = null;
        currentTtsSource = null;
        playNextTts();
    };
}

function stopTtsPlayback() {
    socket.emit('cancel_tts');
    suppressNextBrowserTtsEndCue = true;
    clearCurrentSummaryNearEndTimer();

    if (currentTtsAudio) {
        try {
            currentTtsAudio.pause();
            currentTtsAudio.currentTime = 0;
        } catch (e) {}
        currentTtsAudio = null;
    }
    if (currentTtsSource) {
        try {
            currentTtsSource.stop();
        } catch (e) {}
        currentTtsSource = null;
    }
    if (currentTtsUrl) {
        try { URL.revokeObjectURL(currentTtsUrl); } catch (e) {}
        currentTtsUrl = null;
    }
    while (ttsQueue.length > 0) {
        const item = ttsQueue.shift();
        if (item && item.url) {
            try { URL.revokeObjectURL(item.url); } catch (e) {}
        }
    }
    ttsPlaying = false;
    if (browserSummaryPlaybackActive) {
        socket.emit('browser_tts_playback_done', { reason: 'user_cancel' });
        browserSummaryPlaybackActive = false;
    }
}

function cancelSummaryPlayback() {
    stopTtsPlayback();
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
    clearReconstructedState();
    pendingSummaryTexts = [];
    showSkippedIndicator('Summary canceled');
}

function enqueueTtsAudio(audioBase64, meta = null) {
    console.log('enqueueTtsAudio called, ttsPlaying:', ttsPlaying, 'queueLen:', ttsQueue.length);
    const audioData = atob(audioBase64);
    const audioArray = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        audioArray[i] = audioData.charCodeAt(i);
    }

    const blob = new Blob([audioArray], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    ttsQueue.push({ audioData, url, meta });

    if (!ttsPlaying && !isBlockedByKeywordPlayback(meta)) {
        playNextTts();
    } else {
        console.log('enqueueTtsAudio: ttsPlaying=true, skipping playNextTts');
    }
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
    summaryTriggeredForListeningSession = false;
}

function endListeningIfNeeded() {
    if (!listeningActive) return false;
    socket.emit('end_listening');
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
    const item = options[currentIdx];
    if (!item) return;
    const word = (item.word || '').trim();
    const desc = (item.desc || '').trim();
    if (!word || !desc) return;
    clearKeywordAutoSummarizeTimer();
    keywordPlaybackToken += 1;
    keywordTtsPlaying = true;
    keywordTtsCurrentText = `${word}. ${desc}`;
    scheduleAutoSummarizeFromKeywordPlayback(keywordPlaybackToken);
    socket.emit('keyword_tts', { text: keywordTtsCurrentText, keyword: word });
}

function cancelKeywordTts() {
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    socket.emit('cancel_keyword_tts');
    keywordTtsPlaying = false;
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
    return ttsQueue.some(item => item && item.meta && item.meta.type === 'keyword');
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
    summarySegmentState.delete(segmentId);
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

function hideSummaryText() {
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

    const params = { source: source };
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
    params.transcript_compression_mode = transcriptCompressionMode;
    params.airpods_mode_switch_enabled = airpodsModeSwitchEnabled;

    socket.emit('start', params);
}

function startSummarizing(opts = {}) {
    if (dismissMode === 'summary') {
        if (summaryRequested) return;
        if (summaryTriggeredForListeningSession) return;
        if (!listeningActive) return;
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
    startSummarizing();
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
    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    pendingSummaryTexts = [];
    clearReconstructedState();
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
    if (pendingSummaryTexts.length > 0) {
        showSummaryText();
    }
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
    enqueueTtsAudio(data.audio, { type: 'reconstruction', segmentId: segId, triggerSource });
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
        showLoadingIndicator('Summarizing...', 'summarizing', 220);
    }
    maybeEmitPendingSkippedIndicator();
});

socket.on('keyword_tts_done', () => {
    keywordTtsPlaying = false;
    keywordTtsCurrentText = '';
    keywordPlaybackToken += 1;
    clearKeywordAutoSummarizeTimer();
    // Fallback auto-trigger: if duration estimate missed, trigger summarize on actual TTS completion.
    if (
        autoPreSummarizeEnabled &&
        dismissMode === 'summary' &&
        listeningActive &&
        !summaryTriggeredForListeningSession &&
        !summaryRequested
    ) {
        startSummarizing();
    }
    if (pendingSummarizeIndicatorAfterKeyword && summaryRequested && summaryInProgress) {
        pendingSummarizeIndicatorAfterKeyword = false;
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
