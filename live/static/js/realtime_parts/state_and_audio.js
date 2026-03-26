/**
 * Realtime Words - Main JavaScript
 */

// Socket.IO connection
const socket = io();

// ============================================================================
// State Variables
// ============================================================================

var view = 'single';
var descView = 'single';
var source = 'mic';
var options = [];
var currentIdx = 0;
var summaryBuffer = '';
var summaryTimer = null;
var lastSpace = 0;
var infoVisible = false;
var dismissMode = 'summary'; // 'summary' or 'none'
var summaryRequested = false; // Track if we actually requested a summary
var summaryInProgress = false; // summarizing or tts playback in progress
var showSummaryTextEnabled = false;
var autoPreSummarizeEnabled = true;
var pendingSummaryTexts = [];
var summaryFinalizeTimer = null;
var awaitingJudgeDecision = false;
var allowPostFollowupTts = false;
var keywordOutputMode = 'audio'; // 'text' | 'audio'
var judgeEnabled = false; // Judge agent on/off (when off, summaries play without judgment)
var reconstructorEnabled = false; // Conversation reconstructor on/off
var transcriptCompressionMode = 'api_mini'; // 'fastest' | 'realtime' | 'api_mini' | 'api_nano'
var fastCatchupChainEnabled = false;
var fastCatchupWindowMode = 'vad_utterance'; // 'vad_utterance' | 'time_window'
var summaryFollowupEnabled = false;
var skipFirstTranscriptEnabled = true; // Skip first transcript in summary (default: on)
var missedSummaryLatencyBridgeEnabled = false;
var fastCatchupPending = false;
var keywordTtsPlaying = false; // Track if keyword TTS is playing
var keywordTtsPlaybackStartTime = 0; // Timestamp when keyword TTS playback started
var keywordTtsCurrentText = '';
var keywordAutoSummarizeTimer = null;
var keywordPlaybackToken = 0;
var pendingSummarizeIndicatorAfterKeyword = false;
var keywordTtsPreloadedTexts = new Set(); // Track preloaded keyword TTS texts
var configuredSummaryClientCount = 2;
var summarySegmentState = new Map(); // segmentId -> {received, nonEmpty}
var pendingEmptySummarySignal = null; // {segmentId}
var pendingSkippedIndicator = null; // {message, opts}
var listeningActive = false;
var listeningStartTime = 0; // Timestamp when listening started (for duration tracking)
var hasTranscriptDuringListening = false; // Whether conversation transcript came in during listening
var summaryTriggeredForListeningSession = false;
var ignoreIncomingSummaryEvents = false;
var ignoreIncomingKeywordEvents = false;
var inferencingTimer = null;
var keywordRequestToken = 0; // Token to track keyword request validity
const INFERENCING_TIMEOUT_MS = 5000; // 5 seconds timeout
const KEYWORD_SUMMARY_LEAD_MS = 2000;
const SUMMARY_FOLLOWUP_LEAD_MS = 500;
var keywordSummaryLeadMs = KEYWORD_SUMMARY_LEAD_MS;
const KEYWORD_ESTIMATED_WPS = 3.5; // Cartesia speed=1.2 approximation
var pendingReconstructedTurns = [];
var pendingReconstructedSegmentId = 0;
var renderedReconstructedSegmentId = 0;
var baseReconstructedTurns = [];
var baseReconstructedSegmentId = 0;
var followupReconstructedGroups = []; // [{segmentId, turns}]
var widgetHeight = 0; // 0 = bottom, 100 = top
var isDragging = false;
var availableDevices = [];
var availableOutputDevices = [];
const micSelectIds = ['summaryMic1', 'summaryMic2', 'keywordMic'];

// Noise gate calibration
var noiseGateEnabled = false;
var noiseGateThreshold = 500;  // RMS value (0-32768 scale)
var noiseGateCurrentRMS = { sum1: 0, sum2: 0 };
const NOISE_GATE_MAX_RMS = 5000;  // Max display scale

// Audio feedback
var audioFeedbackMode = 'on'; // 'on' | 'verbal' | 'off'
var airpodsModeSwitchEnabled = true;
var singleClickNavEnabled = false; // Single-click navigate (default: off)
var pauseResumeEnabled = true; // Pause/Resume TTS on single tap (default: on)
var singleKeywordMode = true; // Extract only 1 keyword (default: on)
var transcriptSyncMode = 'vad'; // 'vad' | 'commit' | 'speech_wait' | 'vad_then_commit'
var audioContext = null;
var audioUnlocked = false;
var loadingAudioInterval = null;
var loadingAudioToken = 0;
var loadingAudioStartTimer = null;
var verbalCueLastAt = {};
var browserSummaryPlaybackActive = false;
var activeBrowserTtsType = null;
var suppressNextBrowserTtsEndCue = false;
var ttsCueLastPlayedAt = {
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
var ttsQueue = [];
var streamingTtsBuffers = new Map(); // stream_id -> { chunks: Uint8Array[], meta }
var streamingTtsPlayers = new Map(); // stream_id -> StreamingTtsPlayer instance
var streamingTtsQueue = []; // Queue for sequential playback: { streamId, player, chunks: [] }
var currentStreamingPlayer = null; // Currently playing streaming TTS
var summaryCueSequenceActive = false; // True while a summary/reconstruction stream chain is active
var ttsPlaying = false;
var currentTtsAudio = null;
var currentTtsUrl = null;
var currentTtsSource = null;  // Web Audio API source node

// TTS pause/resume state
var ttsPaused = false;
var ttsPausing = false;  // Flag to prevent onended from firing during pause
var ttsPausedOffset = 0;  // Playback offset in seconds when paused
var ttsPlaybackStartTime = 0;  // AudioContext time when playback started
var ttsPausedBuffer = null;  // AudioBuffer to resume from
var ttsPausedMeta = null;  // Metadata of paused audio
var ttsResumedPlayback = false;  // True if current playback was resumed (skip auto-summarize)
var summaryTextAllowedThisPlayback = false;
var currentSummaryNearEndTimer = null;
var skippedIndicatorTimer = null;
var dismissTimer = null;
var pageExitCleanupSent = false;
var quizBank = [];
var sessionQuizOrder = [];
var quizRoundDurationSec = 60;
var quizRoundNumber = 0;
var quizCurrentIndex = 0;
var quizRoundActive = false;
var quizRoundEndsAt = 0;
var quizRoundTimer = null;
var quizCountdownTimer = null;
var quizLaunchTimer = null;
var quizSessionStartAt = 0;
var quizPlannedOffsetsSec = [60, 240, 420];
var quizPlannedOffsetsRegular = [60, 240, 420];
var quizPlannedOffsetsTutorial = [30, 180];
var quizScheduleIndex = 0;
var quizAncActive = false;
var quizSetSelection = 'A';
var quizIsTutorial = false;

// N-back test state variables
var nbackN = 2;                        // N for n-back test (2-back)
var nbackTotalLetters = 32;            // Total letters in sequence
var nbackDisplayMs = 500;              // Letter display duration (ms)
var nbackIntervalMs = 2500;            // Interval between letters (ms)
var nbackSequence = [];                // Array of letters for current round
var nbackCurrentIndex = 0;             // Current letter index
var nbackRoundActive = false;          // Whether n-back round is active
var nbackDisplayTimer = null;          // Timer for letter display
var nbackIntervalTimer = null;         // Timer for interval between letters
var nbackLetterShownAt = 0;            // Timestamp when current letter was shown
var nbackResponseMade = false;         // Whether response was made for current letter
var nbackKeysPressed = [];             // Keys pressed during current trial
var nbackResults = [];                 // Results for each trial: {letter, isMatch, response, correct, responseTime}
var nbackLeftHeld = false;             // Whether left arrow is currently held down
var nbackRightHeld = false;            // Whether right arrow is currently held down

// Predefined n-back sequences (Set A and Set B, each with 3 rounds)
// Each sequence: 32 letters, ~8 matches (25%), matches marked with * in comments
var nbackPredefinedSets = {
    A: [
        // Set A, Round 1 (8 matches at indices: 4, 8, 12, 16, 20, 24, 27, 30)
        ["M", "N", "M", "L", "P", "R", "S", "R", "S", "K", "Q", "T", "V", "T", "W", "T", "J", "F", "G", "H", "G", "D"],
        // Set A, Round 2 (8 matches at indices: 5, 9, 13, 17, 21, 25, 28, 31)
        ["B", "D", "B", "C", "B", "F", "G", "H", "J", "H", "K", "L", "P", "Q", "P", "Q", "R", "S", "X", "Z", "X", "Y"],
        // Set A, Round 3 (8 matches at indices: 3, 7, 11, 15, 19, 23, 26, 29)
        ["K", "L", "M", "N", "P", "N", "R", "S", "T", "U", "T", "U", "V", "W", "X", "W", "Y", "W", "G", "B", "C", "B"]
    ],
    B: [
        // Set B, Round 1 (8 matches at indices: 6, 10, 14, 18, 22, 25, 28, 31)
        ["F", "G", "F", "G", "H", "J", "K", "L", "K", "M", "N", "P", "R", "P", "S", "P", "Q", "T", "V", "W", "V", "X"],
        // Set B, Round 2 (8 matches at indices: 2, 6, 10, 14, 18, 22, 26, 30)
        ["S", "T", "V", "X", "V", "Y", "V", "R", "Q", "P", "Q", "L", "M", "N", "O", "N", "O", "G", "H", "J", "K", "J"],
        // Set B, Round 3 (8 matches at indices: 4, 8, 12, 16, 20, 24, 28, 31)
        ["L", "M", "L", "N", "P", "Q", "R", "Q", "R", "S", "T", "V", "W", "V", "X", "V", "B", "C", "D", "F", "D", "Z"]
    ]
};
// Tutorial sequences (2 rounds for practice)
var nbackTutorialSequences = [
    ["G", "H", "J", "K", "L", "N", "L", "M", "P", "Q", "P", "Q", "R", "S", "T", "S", "U", "S", "V", "W", "X", "W"],
    ["X", "Z", "X", "Y", "X", "B", "C", "D", "F", "D", "G", "H", "J", "K", "L", "K", "L", "M", "N", "P", "R", "P"]
];

var nbackCurrentSet = 'A';             // Current set ('A' or 'B')
var nbackShuffledOrder = [0, 1, 2];    // Shuffled order of rounds within set
var nbackSetRoundIndex = 0;            // Current index in shuffledOrder (0, 1, or 2)
var nbackIsTutorial = false;           // Whether current round is tutorial
var nbackTutorialRoundIndex = 0;       // Current tutorial round (0 or 1)

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

function cleanupBeforePageExit() {
    if (pageExitCleanupSent) return;
    pageExitCleanupSent = true;

    try {
        navigator.sendBeacon('/api/cleanup', new Blob(['{}'], { type: 'application/json' }));
    } catch (e) {}
    try {
        stopTtsPlayback();
    } catch (e) {}
    try {
        socket.emit('cancel_keyword_tts');
        socket.emit('stop');
        socket.emit('stop_mic_monitor');
        socket.emit('stop_noise_gate_monitor');
    } catch (e) {}
}

window.addEventListener('pagehide', cleanupBeforePageExit);
window.addEventListener('beforeunload', cleanupBeforePageExit);

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
// Brown Noise Control Functions
// ============================================================================

var brownNoiseRunning = false;
var brownNoiseVolume = 50; // 0-100

function toggleBrownNoise() {
    if (brownNoiseRunning) {
        stopBrownNoise();
    } else {
        startBrownNoise();
    }
}

function startBrownNoise() {
    const volume = brownNoiseVolume / 100;
    socket.emit('brown_noise_start', { volume: volume }, function(response) {
        if (response && response.ok) {
            brownNoiseRunning = true;
            updateBrownNoiseUI();
        }
    });
}

function stopBrownNoise() {
    socket.emit('brown_noise_stop', {}, function(response) {
        if (response && response.ok) {
            brownNoiseRunning = false;
            updateBrownNoiseUI();
        }
    });
}

function updateBrownNoiseVolume() {
    const slider = document.getElementById('brownNoiseVolume');
    brownNoiseVolume = parseInt(slider.value);
    document.getElementById('brownNoiseVolumeValue').textContent = brownNoiseVolume + '%';

    // Only update server if noise is running
    if (brownNoiseRunning) {
        const volume = brownNoiseVolume / 100;
        socket.emit('brown_noise_volume', { volume: volume });
    }
}

function updateBrownNoiseUI() {
    const btn = document.getElementById('brownNoiseToggle');
    const volumeWrapper = document.getElementById('brownNoiseVolumeWrapper');

    if (brownNoiseRunning) {
        btn.classList.add('active');
        volumeWrapper.classList.add('active');
    } else {
        btn.classList.remove('active');
        volumeWrapper.classList.remove('active');
    }
}

function fetchBrownNoiseStatus() {
    socket.emit('brown_noise_status', {}, function(response) {
        if (response && response.ok) {
            brownNoiseRunning = response.running;
            brownNoiseVolume = Math.round((response.volume || 0.5) * 100);
            document.getElementById('brownNoiseVolume').value = brownNoiseVolume;
            document.getElementById('brownNoiseVolumeValue').textContent = brownNoiseVolume + '%';
            updateBrownNoiseUI();
        }
    });
}

// ============================================================================
// Options Collapse Toggle
// ============================================================================

function toggleOptionsCollapse() {
    const content = document.getElementById('optionsCollapseContent');
    const arrow = document.getElementById('optionsCollapseArrow');
    const isOpen = content.style.display !== 'none';

    content.style.display = isOpen ? 'none' : 'block';
    arrow.classList.toggle('open', !isOpen);
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
        // Auto-select: keywordMic prefers Jabra, then falls back to MacBook.
        if (id === 'keywordMic') {
            const jabraMic = availableDevices.find(d => d.name.toLowerCase().includes('jabra'));
            const macbookMic = availableDevices.find(d => d.name.toLowerCase().includes('macbook'));
            if (jabraMic) {
                sel.value = jabraMic.index;
            } else if (macbookMic) {
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
        hideReconstructedTurns();
    } else if (summaryInProgress && pendingSummaryTexts.length > 0) {
        showSummaryText();
    } else if (summaryInProgress) {
        if (baseReconstructedTurns.length > 0 || followupReconstructedGroups.length > 0) {
            renderReconstructedStack(baseReconstructedTurns, followupReconstructedGroups);
        } else if (pendingReconstructedTurns.length > 0) {
            renderReconstructedTurns(pendingReconstructedTurns);
        }
    }
}

function setQuizSetSelection(setName) {
    const upper = (setName || 'A').toUpperCase();
    const normalized = upper === 'T' ? 'T' : (upper === 'B' ? 'B' : 'A');
    quizSetSelection = normalized;
    quizIsTutorial = normalized === 'T';
    quizPlannedOffsetsSec = quizIsTutorial ? quizPlannedOffsetsTutorial : quizPlannedOffsetsRegular;
    const btnA = document.getElementById('btn-quiz-set-a');
    const btnB = document.getElementById('btn-quiz-set-b');
    const btnT = document.getElementById('btn-quiz-set-t');
    if (btnA) btnA.classList.toggle('selected', normalized === 'A');
    if (btnB) btnB.classList.toggle('selected', normalized === 'B');
    if (btnT) btnT.classList.toggle('selected', normalized === 'T');
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

function setSkipFirstTranscriptEnabled(enabled) {
    skipFirstTranscriptEnabled = !!enabled;
    document.getElementById('btn-skip-first-transcript-on').classList.toggle('selected', skipFirstTranscriptEnabled);
    document.getElementById('btn-skip-first-transcript-off').classList.toggle('selected', !skipFirstTranscriptEnabled);
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
        this.isPaused = false;
        this.pendingChunks = [];
        this.scheduledSources = [];
        this.totalSamplesScheduled = 0;
        this.startedPlayback = false;
        this.onEndCallback = null;
        this.allChunks = [];  // Store all chunks for replay on resume
        this.playbackStartTime = 0;  // AudioContext time when playback started
        this.samplesPlayedBeforePause = 0;  // Track how many samples were played before pause
        this.finishPending = false;  // True if finish() was called while paused
        this.pendingFinishTimer = null;  // Timer for delayed finish after resume
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
        this.playbackStartTime = this.audioContext.currentTime;
        this.isPlaying = true;
        this.startedPlayback = true;
        console.log(`StreamingTtsPlayer[${this.streamId}]: Started, sampleRate=${this.sampleRate}`);
    }

    async pushChunk(pcmBytes) {
        // Store chunk for replay on resume
        this.allChunks.push(new Uint8Array(pcmBytes));
        return this.pushChunkInternal(pcmBytes, true);
    }

    async pushChunkInternal(pcmBytes, triggerFirstChunk = true) {
        if (this.isStopped || this.isPaused) return;

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
        if (triggerFirstChunk && !this.startedPlayback) {
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
        this.isPaused = false;

        // Cancel any pending finish timer
        if (this.pendingFinishTimer) {
            clearTimeout(this.pendingFinishTimer);
            this.pendingFinishTimer = null;
        }

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

    pause() {
        if (this.isStopped || this.isPaused || !this.isPlaying) return false;

        // Calculate how many samples have been played
        if (this.audioContext && this.startedPlayback) {
            const elapsedTime = this.audioContext.currentTime - this.playbackStartTime;
            this.samplesPlayedBeforePause += Math.floor(elapsedTime * this.sampleRate);
        }

        // Stop all scheduled sources
        for (const source of this.scheduledSources) {
            try {
                source.stop();
            } catch (e) {}
        }
        this.scheduledSources = [];

        this.isPaused = true;
        this.isPlaying = false;
        console.log(`StreamingTtsPlayer[${this.streamId}]: Paused at sample ${this.samplesPlayedBeforePause}`);
        return true;
    }

    async resume() {
        if (this.isStopped || !this.isPaused) return false;

        if (this.allChunks.length === 0) {
            this.isPaused = false;
            return false;
        }

        // Reset playback state
        this.nextPlayTime = 0;
        this.totalSamplesScheduled = 0;
        this.scheduledSources = [];

        // Restart audio context if needed
        if (!this.audioContext || this.audioContext.state === 'closed') {
            const Ctx = window.AudioContext || window.webkitAudioContext;
            if (!Ctx) {
                this.isPaused = false;
                return false;
            }
            this.audioContext = new Ctx({ sampleRate: this.sampleRate });
        }

        if (this.audioContext.state === 'suspended') {
            try {
                await this.audioContext.resume();
            } catch (e) {
                console.error('StreamingTtsPlayer resume failed:', e);
                this.isPaused = false;
                return false;
            }
        }

        this.nextPlayTime = this.audioContext.currentTime + 0.05;
        this.playbackStartTime = this.audioContext.currentTime;
        this.isPaused = false;
        this.isPlaying = true;

        // Re-push chunks, skipping already-played samples
        let samplesToSkip = this.samplesPlayedBeforePause;
        let samplesScheduledInResume = 0;
        for (const chunk of this.allChunks) {
            const chunkSamples = chunk.length / 2;  // PCM16 = 2 bytes per sample
            if (samplesToSkip >= chunkSamples) {
                // Skip this entire chunk
                samplesToSkip -= chunkSamples;
                continue;
            }
            if (samplesToSkip > 0) {
                // Partial chunk - skip some samples from the beginning
                const bytesToSkip = samplesToSkip * 2;
                const remainingChunk = chunk.slice(bytesToSkip);
                samplesScheduledInResume += remainingChunk.length / 2;
                await this.pushChunkInternal(remainingChunk, false);
                samplesToSkip = 0;
            } else {
                // Full chunk
                samplesScheduledInResume += chunkSamples;
                await this.pushChunkInternal(chunk, false);
            }
        }

        // Calculate remaining duration based on samples scheduled
        const remainingDurationMs = (samplesScheduledInResume / this.sampleRate) * 1000;
        console.log(`StreamingTtsPlayer[${this.streamId}]: Resumed from sample ${this.samplesPlayedBeforePause}, remaining duration: ${remainingDurationMs.toFixed(0)}ms`);

        // If finish was called while paused, wait for actual playback to complete
        if (this.finishPending) {
            this.finishPending = false;
            // Wait for the calculated remaining duration plus buffer, then finish
            this.pendingFinishTimer = setTimeout(() => {
                this.pendingFinishTimer = null;
                if (!this.isStopped && !this.isPaused) {
                    this.finish();
                }
            }, remainingDurationMs + 200);  // Add 200ms buffer for scheduling delays
        }

        return true;
    }

    async finish() {
        if (this.isStopped) return;

        // If paused, mark finish as pending and return
        if (this.isPaused) {
            this.finishPending = true;
            console.log(`StreamingTtsPlayer[${this.streamId}]: finish() deferred (paused)`);
            return;
        }

        // Wait for all scheduled audio to finish
        if (this.audioContext && this.nextPlayTime > this.audioContext.currentTime) {
            const remainingMs = (this.nextPlayTime - this.audioContext.currentTime) * 1000;
            await new Promise(resolve => setTimeout(resolve, remainingMs + 50));
        }

        // If stream was cancelled or paused while waiting, don't fire end callback.
        if (this.isStopped || this.isPaused) {
            if (this.isPaused) this.finishPending = true;
            return;
        }

        this.finishPending = false;
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

function setSingleClickNavEnabled(enabled) {
    singleClickNavEnabled = enabled;
    document.getElementById('btn-single-click-nav-on').classList.toggle('selected', enabled);
    document.getElementById('btn-single-click-nav-off').classList.toggle('selected', !enabled);
}

function setPauseResumeEnabled(enabled) {
    pauseResumeEnabled = enabled;
    document.getElementById('btn-pause-resume-on').classList.toggle('selected', enabled);
    document.getElementById('btn-pause-resume-off').classList.toggle('selected', !enabled);
    console.log('Pause/Resume TTS:', enabled ? 'enabled' : 'disabled');
}

function setSingleKeywordMode(enabled) {
    singleKeywordMode = enabled;
    document.getElementById('btn-single-keyword-on').classList.toggle('selected', enabled);
    document.getElementById('btn-single-keyword-off').classList.toggle('selected', !enabled);
    socket.emit('set_single_keyword_mode', { enabled });
}

function setTranscriptSyncMode(mode) {
    const validModes = ['vad', 'commit', 'speech_wait', 'vad_then_commit'];
    transcriptSyncMode = validModes.includes(mode) ? mode : 'vad';
    document.getElementById('btn-transcript-sync-vad').classList.toggle('selected', transcriptSyncMode === 'vad');
    document.getElementById('btn-transcript-sync-commit').classList.toggle('selected', transcriptSyncMode === 'commit');
    document.getElementById('btn-transcript-sync-speech').classList.toggle('selected', transcriptSyncMode === 'speech_wait');
    document.getElementById('btn-transcript-sync-vad-then-commit').classList.toggle('selected', transcriptSyncMode === 'vad_then_commit');
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
            keywordTtsPlaybackStartTime = 0;
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
            endQuizAncSession();
            scheduleNextQuizRound();
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
        // Save audio buffer for pause/resume
        ttsPausedBuffer = audioBuffer;
        ttsPausedMeta = meta;

        const source = ctx.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(ctx.destination);
        currentTtsSource = source;

        source.onended = () => {
            // Skip cleanup if we're pausing (not naturally ending)
            if (ttsPausing || ttsPaused) {
                console.log('playNextTts onended: skipped (pausing/paused)');
                return;
            }
            clearCurrentSummaryNearEndTimer();
            URL.revokeObjectURL(url);
            currentTtsAudio = null;
            currentTtsUrl = null;
            currentTtsSource = null;
            clearTtsPausedState();
            playNextTts();
        };

        await waitForAncBeforeBrowserSummaryPlayback(meta);
        if (activeBrowserTtsType !== ttsType) {
            playTtsStartFeedback(ttsType);
            activeBrowserTtsType = ttsType;
        }
        // Track playback start time for pause/resume
        ttsPlaybackStartTime = ctx.currentTime;
        ttsPausedOffset = 0;
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
    clearTtsPausedState();

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

// ============================================================================
// TTS Pause/Resume Functions
// ============================================================================

function clearTtsPausedState() {
    ttsPaused = false;
    ttsPausing = false;
    ttsPausedOffset = 0;
    ttsPlaybackStartTime = 0;
    ttsPausedBuffer = null;
    ttsPausedMeta = null;
    ttsResumedPlayback = false;
}

function isTtsPlayingOrPaused() {
    return ttsPlaying || ttsPaused || currentStreamingPlayer !== null;
}

function pauseTtsPlayback() {
    // Clear auto-summarize timer when pausing
    clearKeywordAutoSummarizeTimer();

    // Handle streaming TTS pause
    if (currentStreamingPlayer && !currentStreamingPlayer.isStopped) {
        currentStreamingPlayer.pause();
        ttsPaused = true;
        // Notify server to switch to transparency mode (ANC off)
        socket.emit('pause_tts');
        console.log('pauseTtsPlayback: streaming TTS paused');
        return true;
    }

    // Handle regular TTS pause
    if (!ttsPlaying || !currentTtsSource || !audioContext) {
        return false;
    }

    try {
        // Calculate how far into the audio we are
        const currentTime = audioContext.currentTime;
        ttsPausedOffset += (currentTime - ttsPlaybackStartTime);
        // Set pausing flag to prevent onended from triggering
        ttsPausing = true;
        currentTtsSource.stop();
        ttsPaused = true;
        ttsPausing = false;
        ttsPlaying = false;
        // Notify server to switch to transparency mode (ANC off)
        socket.emit('pause_tts');
        console.log('pauseTtsPlayback: regular TTS paused at offset', ttsPausedOffset);
        return true;
    } catch (e) {
        console.error('pauseTtsPlayback failed:', e);
        ttsPausing = false;
        return false;
    }
}

async function resumeTtsPlayback() {
    if (!ttsPaused) {
        return false;
    }

    // Handle streaming TTS resume
    if (currentStreamingPlayer && currentStreamingPlayer.isPaused) {
        await currentStreamingPlayer.resume();
        ttsPaused = false;
        ttsResumedPlayback = true;  // Mark as resumed playback (skip auto-summarize)
        // Notify server to switch to ANC mode (ANC on)
        socket.emit('resume_tts');
        console.log('resumeTtsPlayback: streaming TTS resumed');
        return true;
    }

    // Handle regular TTS resume - continue from paused position
    if (ttsPausedBuffer && audioContext) {
        try {
            if (audioContext.state === 'suspended') {
                await audioContext.resume();
            }

            const source = audioContext.createBufferSource();
            source.buffer = ttsPausedBuffer;
            source.connect(audioContext.destination);
            currentTtsSource = source;

            // Calculate remaining duration for onended check
            const remainingDuration = ttsPausedBuffer.duration - ttsPausedOffset;

            source.onended = () => {
                // Skip cleanup if we're pausing again
                if (ttsPausing || ttsPaused) {
                    console.log('resumeTtsPlayback onended: skipped (pausing/paused)');
                    return;
                }
                clearCurrentSummaryNearEndTimer();
                if (currentTtsUrl) {
                    URL.revokeObjectURL(currentTtsUrl);
                }
                currentTtsAudio = null;
                currentTtsUrl = null;
                currentTtsSource = null;
                // Switch to transparency mode (ANC off) after resumed playback ends
                socket.emit('pause_tts');
                console.log('resumeTtsPlayback onended: ANC off (transparency)');
                clearTtsPausedState();
                playNextTts();
            };

            // Start from the paused offset
            ttsPlaybackStartTime = audioContext.currentTime;
            source.start(0, ttsPausedOffset);
            ttsPlaying = true;
            ttsPaused = false;
            ttsResumedPlayback = true;  // Mark as resumed playback (skip auto-summarize)
            // Notify server to switch to ANC mode (ANC on)
            socket.emit('resume_tts');
            console.log('resumeTtsPlayback: regular TTS resumed from offset', ttsPausedOffset);
            return true;
        } catch (e) {
            console.error('resumeTtsPlayback failed:', e);
            clearTtsPausedState();
            return false;
        }
    }

    return false;
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
