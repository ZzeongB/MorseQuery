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
let transcriptCompressionMode = 'api_mini'; // 'fastest' | 'realtime' | 'api_mini' | 'api_nano'
let fastCatchupChainEnabled = false;
let fastCatchupWindowMode = 'vad_utterance'; // 'vad_utterance' | 'time_window'
let summaryFollowupEnabled = false;
let missedSummaryLatencyBridgeEnabled = false;
let fastCatchupPending = false;
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
let ignoreIncomingSummaryEvents = false;
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
let streamingTtsBuffers = new Map(); // stream_id -> { chunks: Uint8Array[], meta }
let streamingTtsPlayers = new Map(); // stream_id -> StreamingTtsPlayer instance
let streamingTtsQueue = []; // Queue for sequential playback: { streamId, player, chunks: [] }
let currentStreamingPlayer = null; // Currently playing streaming TTS
let summaryCueSequenceActive = false; // True while a summary/reconstruction stream chain is active
let ttsPlaying = false;
let currentTtsAudio = null;
let currentTtsUrl = null;
let currentTtsSource = null;  // Web Audio API source node
let summaryTextAllowedThisPlayback = false;
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

function setFastCatchupChainEnabled(enabled) {
    fastCatchupChainEnabled = !!enabled;
    document.getElementById('btn-catchup-chain-on').classList.toggle('selected', fastCatchupChainEnabled);
    document.getElementById('btn-catchup-chain-off').classList.toggle('selected', !fastCatchupChainEnabled);
}

function setFastCatchupWindowMode(mode) {
    fastCatchupWindowMode = mode === 'time_window' ? 'time_window' : 'vad_utterance';
    document.getElementById('btn-catchup-window-vad').classList.toggle('selected', fastCatchupWindowMode === 'vad_utterance');
    document.getElementById('btn-catchup-window-time').classList.toggle('selected', fastCatchupWindowMode === 'time_window');
}

function setSummaryFollowupEnabled(enabled) {
    summaryFollowupEnabled = !!enabled;
    document.getElementById('btn-summary-followup-on').classList.toggle('selected', summaryFollowupEnabled);
    document.getElementById('btn-summary-followup-off').classList.toggle('selected', !summaryFollowupEnabled);
}

function setMissedSummaryLatencyBridgeEnabled(enabled) {
    missedSummaryLatencyBridgeEnabled = !!enabled;
    document.getElementById('btn-missed-bridge-on').classList.toggle('selected', missedSummaryLatencyBridgeEnabled);
    document.getElementById('btn-missed-bridge-off').classList.toggle('selected', !missedSummaryLatencyBridgeEnabled);
}

// ============================================================================
// Streaming TTS Player
// ============================================================================

class StreamingTtsPlayer {
    constructor(streamId, sampleRate = 24000, meta = {}) {
        this.streamId = streamId;
        this.sampleRate = sampleRate;
        this.meta = meta;
        this.audioContext = null;
        this.nextPlayTime = 0;
        this.isPlaying = false;
        this.isStopped = false;
        this.pendingChunks = [];
        this.scheduledSources = [];
        this.totalSamplesScheduled = 0;
        this.startedPlayback = false;
        this.onEndCallback = null;
    }

    async start() {
        if (this.audioContext) return;

        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) {
            console.error('StreamingTtsPlayer: No AudioContext available');
            return;
        }

        this.audioContext = new Ctx({ sampleRate: this.sampleRate });
        if (this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
            } catch (e) {
                console.error('StreamingTtsPlayer: Failed to resume AudioContext', e);
            }
        }

        this.nextPlayTime = this.audioContext.currentTime + 0.05; // Small buffer
        this.isPlaying = true;
        console.log(`StreamingTtsPlayer[${this.streamId}]: Started, sampleRate=${this.sampleRate}`);
    }

    async pushChunk(pcmBytes) {
        if (this.isStopped) return;

        if (!this.audioContext) {
            await this.start();
        }

        if (!this.audioContext || this.audioContext.state === 'closed') return;

        // PCM S16LE to Float32
        const samples = pcmBytes.length / 2;
        const float32 = new Float32Array(samples);
        const dataView = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);

        for (let i = 0; i < samples; i++) {
            const int16 = dataView.getInt16(i * 2, true); // little-endian
            float32[i] = int16 / 32768.0;
        }

        // Create AudioBuffer
        const audioBuffer = this.audioContext.createBuffer(1, samples, this.sampleRate);
        audioBuffer.getChannelData(0).set(float32);

        // Schedule playback
        const source = this.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audioContext.destination);

        // Calculate when to play
        const currentTime = this.audioContext.currentTime;
        if (this.nextPlayTime < currentTime) {
            this.nextPlayTime = currentTime + 0.02;
        }

        source.start(this.nextPlayTime);
        this.scheduledSources.push(source);

        const duration = samples / this.sampleRate;
        this.nextPlayTime += duration;
        this.totalSamplesScheduled += samples;

        // Trigger UI update on first chunk
        if (!this.startedPlayback) {
            this.startedPlayback = true;
            this.onFirstChunk();
        }
    }

    onFirstChunk() {
        // Override in usage to trigger UI updates
    }

    stop() {
        this.isStopped = true;
        this.isPlaying = false;

        for (const source of this.scheduledSources) {
            try {
                source.stop();
            } catch (e) {}
        }
        this.scheduledSources = [];

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => {});
        }
        this.audioContext = null;

        console.log(`StreamingTtsPlayer[${this.streamId}]: Stopped`);
    }

    async finish() {
        if (this.isStopped) return;

        // Wait for all scheduled audio to finish
        if (this.audioContext && this.nextPlayTime > this.audioContext.currentTime) {
            const remainingMs = (this.nextPlayTime - this.audioContext.currentTime) * 1000;
            await new Promise(resolve => setTimeout(resolve, remainingMs + 50));
        }

        // If stream was cancelled while waiting, don't fire end callback.
        if (this.isStopped) return;

        this.isPlaying = false;
        const totalDuration = this.totalSamplesScheduled / this.sampleRate;
        console.log(`StreamingTtsPlayer[${this.streamId}]: Finished, duration=${totalDuration.toFixed(2)}s`);

        if (this.onEndCallback) {
            this.onEndCallback();
        }

        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => {});
        }
        this.audioContext = null;
    }
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
    const isSummary = !!(meta && meta.type === 'summary');
    const isReconstruction = !!(meta && meta.type === 'reconstruction');
    const isSpeedup = !!(meta && meta.visualMode === 'none');
    summaryTextAllowedThisPlayback = isSummary || isReconstruction;

    if (isSummary) {
        // During summary playback, keyword text visual must always stay hidden.
        hideInfo();
        hideReconstructedTurns();
        if (pendingSummaryTexts.length > 0) {
            showSummaryText();
        }
        return;
    }

    if (isReconstruction) {
        if (isSpeedup) {
            hideSummaryText(false);
            hideReconstructedTurns();
            return;
        }
        hideSummaryText(false);
        if (baseReconstructedTurns.length > 0 || followupReconstructedGroups.length > 0) {
            renderReconstructedStack(baseReconstructedTurns, followupReconstructedGroups);
        } else {
            renderReconstructedTurns(pendingReconstructedTurns);
        }
        return;
    }

    hideSummaryText();
    hideReconstructedTurns();
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
        const finishedSummaryPlayback = browserSummaryPlaybackActive || activeBrowserTtsType === 'summary';
        const finishedKeywordPlayback = activeBrowserTtsType === 'keyword';
        clearCurrentSummaryNearEndTimer();
        ttsPlaying = false;
        currentTtsAudio = null;
        currentTtsUrl = null;
        currentTtsSource = null;
        if (browserSummaryPlaybackActive) {
            const ack = await emitWithAck('browser_tts_playback_done', { reason: 'queue_empty' });
            const deferFinish = !!(ack && ack.defer_finish);
            if (deferFinish) {
                fastCatchupPending = true;
                suppressNextBrowserTtsEndCue = false;
                activeBrowserTtsType = null;
                summaryInProgress = true;
                return;
            }
            browserSummaryPlaybackActive = false;
        }
        if (!suppressNextBrowserTtsEndCue && activeBrowserTtsType) {
            playTtsEndFeedback(activeBrowserTtsType);
        }
        suppressNextBrowserTtsEndCue = false;
        activeBrowserTtsType = null;
        if (finishedKeywordPlayback) {
            keywordTtsPlaying = false;
            keywordTtsCurrentText = '';
            if (
                pendingSummarizeIndicatorAfterKeyword &&
                summaryRequested &&
                summaryInProgress
            ) {
                pendingSummarizeIndicatorAfterKeyword = false;
                hideInfo();
                hideSummaryText();
                hideReconstructedTurns();
                showLoadingIndicator('Summarizing...', 'summarizing', 220);
            }
            if (
                autoPreSummarizeEnabled &&
                dismissMode === 'summary' &&
                listeningActive &&
                !summaryTriggeredForListeningSession &&
                !summaryRequested
            ) {
                startSummarizing();
            }
        }
        summaryInProgress = false;
        allowPostFollowupTts = false;
        hideSummaryText();
        hideReconstructedTurns();
        pendingSummaryTexts = [];
        clearReconstructedState();
        if (finishedSummaryPlayback) {
            hideInfo();
        }
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
    const { audioData, audioBytes, url, meta } = ttsQueue.shift();
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

        let arrayBuffer;
        if (audioBytes && audioBytes.buffer) {
            arrayBuffer = audioBytes.buffer.slice(
                audioBytes.byteOffset,
                audioBytes.byteOffset + audioBytes.byteLength,
            );
        } else {
            arrayBuffer = new ArrayBuffer(audioData.length);
            const view = new Uint8Array(arrayBuffer);
            for (let i = 0; i < audioData.length; i++) {
                view[i] = audioData.charCodeAt(i);
            }
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
    streamingTtsBuffers.clear();
    // Stop all streaming TTS players
    for (const [streamId, player] of streamingTtsPlayers) {
        player.stop();
    }
    streamingTtsPlayers.clear();
    // Clear streaming TTS queue
    for (const item of streamingTtsQueue) {
        if (item.player) {
            item.player.stop();
        }
    }
    streamingTtsQueue = [];
    currentStreamingPlayer = null;
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
    ignoreIncomingSummaryEvents = true;
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
