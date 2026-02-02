// MorseQuery Simple - Widget Frontend (Two-column layout)
const socket = io();

// State
let state = {
    isRecording: false,
    mediaRecorder: null,
    audioStream: null,
    audioChunks: [],
    allKeywords: [],         // Unified list (new + history, most recent first)
    currentKeywordIndex: 0,
    currentKeyword: null,
    currentDescription: null,
    groundingCache: {},      // Cache grounding results by keyword: { keyword: { text, citations, image } }
    autoInferenceMode: 'off',
    autoInferenceInterval: 3.0,
    currentConfig: 1,        // Current prompt configuration (1-6)
};

// DOM elements
const widget = document.querySelector('.widget');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const mainContent = document.getElementById('mainContent');
const keywordsList = document.getElementById('keywordsList');
const rightColumn = document.getElementById('rightColumn');
const descriptionPanel = document.getElementById('descriptionPanel');
const selectedKeyword = document.getElementById('selectedKeyword');
const keywordDescription = document.getElementById('keywordDescription');
const descriptionImage = document.getElementById('descriptionImage');
const groundingPanel = document.getElementById('groundingPanel');
const groundingKeyword = document.getElementById('groundingKeyword');
const groundingText = document.getElementById('groundingText');
const groundingImage = document.getElementById('groundingImage');
const citationList = document.getElementById('citationList');
const emptyPanel = document.getElementById('emptyPanel');
const micBtn = document.getElementById('micBtn');
const fileBtn = document.getElementById('fileBtn');
const clearBtn = document.getElementById('clearBtn');
const fileInput = document.getElementById('fileInput');
const audioPlayer = document.getElementById('audioPlayer');
const autoModeSelect = document.getElementById('autoModeSelect');
const autoIntervalInput = document.getElementById('autoIntervalInput');
const configSelect = document.getElementById('configSelect');

// Socket events
socket.on('connect', () => {
    setStatus('connected', 'Connected');
    socket.emit('start_openai_transcription');
});

socket.on('disconnect', () => {
    setStatus('disconnected', 'Disconnected');
});

socket.on('status', (data) => {
    setStatus('processing', data.message);
});

socket.on('error', (data) => {
    setStatus('error', data.message);
    setTimeout(() => setStatus('connected', 'Ready'), 3000);
});

socket.on('transcription', (data) => {
    // Transcription received, no status change needed
});

socket.on('keywords_extracted', (data) => {
    console.log('[Keywords]', data);
    hideWaitingKeywords();

    // Merge new keywords with history (new first, then history)
    const newKeywords = data.keywords || [];
    const history = data.history || [];

    // Build unified list: new keywords at top, then history
    state.allKeywords = [...newKeywords];

    // Add history items that aren't duplicates
    history.forEach(h => {
        const exists = state.allKeywords.some(k => k.keyword === h.keyword);
        if (!exists) {
            state.allKeywords.push(h);
        }
    });

    state.currentKeywordIndex = 0;

    renderKeywordsList();

    // Auto-select first keyword
    if (state.allKeywords.length > 0) {
        selectKeyword(0);
    }

    setStatus('connected', 'Keywords extracted');
});

socket.on('keyword_selected', (data) => {
    console.log('[Keyword Selected]', data);
    state.currentKeyword = data.keyword;
    state.currentDescription = data.description;
    state.currentKeywordIndex = data.index;

    displayDescription(data.keyword, data.description);
    highlightKeyword(data.index);
    hideGrounding();

    setStatus('connected', `${data.keyword} (${data.index + 1}/${data.total})`);
});

socket.on('grounding_result', (data) => {
    console.log('[Grounding]', data);
    hideWaitingDetail();

    // Cache the grounding result for this keyword
    state.groundingCache[data.keyword] = {
        text: data.text,
        citations: data.citations,
        image: data.image,
    };

    // Update the keyword in allKeywords to mark it has grounding
    const kwIndex = state.allKeywords.findIndex(k => k.keyword === data.keyword);
    if (kwIndex !== -1) {
        state.allKeywords[kwIndex].hasGrounding = true;
    }

    // Display in the description panel (replacing short description)
    displayKeywordDetail(data.keyword, data.text, data.citations, data.image);
    setStatus('connected', 'Detailed info loaded');
});

socket.on('keyword_history', (data) => {
    // Update history in allKeywords
    const history = data.history || [];
    const currentNew = state.allKeywords.filter(k => k.isNew);

    state.allKeywords = [...currentNew];
    history.forEach(h => {
        const exists = state.allKeywords.some(k => k.keyword === h.keyword);
        if (!exists) {
            state.allKeywords.push(h);
        }
    });

    renderKeywordsList();
});

socket.on('session_cleared', () => {
    state.allKeywords = [];
    state.currentKeywordIndex = 0;
    state.currentKeyword = null;
    state.currentDescription = null;
    state.groundingCache = {};

    renderKeywordsList();
    showEmptyState();

    setStatus('connected', 'Session cleared');
});

socket.on('auto_inference_status', (data) => {
    console.log('[Auto-Inference] Status:', data);
    state.autoInferenceMode = data.mode;
    state.autoInferenceInterval = data.interval;
    autoModeSelect.value = data.mode;
    autoIntervalInput.value = data.interval;
    updateAutoInferenceUI();
});

socket.on('config_status', (data) => {
    console.log('[Config] Status:', data);
    state.currentConfig = data.config_id;
    configSelect.value = data.config_id;
});

// Button events
micBtn.addEventListener('click', toggleMicrophone);
fileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
clearBtn.addEventListener('click', clearSession);

// Auto-inference events
autoModeSelect.addEventListener('change', () => {
    const mode = autoModeSelect.value;
    const interval = parseFloat(autoIntervalInput.value) || 3.0;
    socket.emit('set_auto_inference', { mode, interval });
    updateAutoInferenceUI();
});

autoIntervalInput.addEventListener('change', () => {
    if (state.autoInferenceMode === 'time') {
        const mode = autoModeSelect.value;
        const interval = parseFloat(autoIntervalInput.value) || 3.0;
        socket.emit('set_auto_inference', { mode, interval });
    }
});

// Configuration selector event
configSelect.addEventListener('change', () => {
    const configId = parseInt(configSelect.value);
    state.currentConfig = configId;
    socket.emit('set_config', { config_id: configId });
    console.log('[Config] Changed to:', configId);
});

function updateAutoInferenceUI() {
    const showInterval = autoModeSelect.value === 'time';
    autoIntervalInput.style.display = showInterval ? 'inline-block' : 'none';
    document.querySelector('.auto-interval-label').style.display = showInterval ? 'inline' : 'none';
}

// Spacebar / Right-click handling
const DOUBLE_PRESS_THRESHOLD_KEY = 300;
const DOUBLE_PRESS_THRESHOLD_MOUSE = 400;
const LONG_PRESS_THRESHOLD_KEY = 500;
const LONG_PRESS_THRESHOLD_MOUSE = 800;

let pressTimer = null;
let pressCount = 0;
let pressDownTime = 0;
let longPressAnimationFrame = null;
let longPressTriggered = false;
let currentLongPressThreshold = LONG_PRESS_THRESHOLD_KEY;
let currentDoublePressThreshold = DOUBLE_PRESS_THRESHOLD_KEY;
let currentInputType = 'key'; // 'key' or 'mouse'

// Spacebar events
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();

        if (!e.repeat && pressDownTime === 0) {
            pressDownTime = Date.now();
            longPressTriggered = false;
            currentLongPressThreshold = LONG_PRESS_THRESHOLD_KEY;
            currentDoublePressThreshold = DOUBLE_PRESS_THRESHOLD_KEY;
            currentInputType = 'key';
            startLongPressTimer();
        }
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        handlePressUp();
    }
});

// Right-click events
document.addEventListener('contextmenu', (e) => {
    e.preventDefault();

    if (pressDownTime === 0) {
        pressDownTime = Date.now();
        longPressTriggered = false;
        currentLongPressThreshold = LONG_PRESS_THRESHOLD_MOUSE;
        currentDoublePressThreshold = DOUBLE_PRESS_THRESHOLD_MOUSE;
        currentInputType = 'mouse';
        startLongPressTimer();
    }
});

document.addEventListener('mouseup', (e) => {
    if (e.button === 2 && pressDownTime > 0) {
        handlePressUp();
    }
});

// Unified press-up handler
function handlePressUp() {
    pressDownTime = 0;
    cancelLongPressTimer();

    if (longPressTriggered) {
        longPressTriggered = false;
        pressCount = 0;
        return;
    }

    pressCount++;

    if (pressCount === 2) {
        clearTimeout(pressTimer);
        pressTimer = null;
        pressCount = 0;
        handleDoublePress();
        return;
    }

    if (pressTimer) clearTimeout(pressTimer);
    pressTimer = setTimeout(() => {
        pressTimer = null;
        if (pressCount === 1) {
            handleSinglePress();
        }
        pressCount = 0;
    }, currentDoublePressThreshold);
}

function startLongPressTimer() {
    const startTime = Date.now();

    function updateProgress() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / currentLongPressThreshold, 1);

        showLongPressIndicator(progress);

        if (progress >= 0.5 && !longPressTriggered) {
            longPressTriggered = true;
            handleLongPress();
        } else if(progress >= 1) {
            cancelLongPressTimer();
        }
        else if (progress < 1) {
            longPressAnimationFrame = requestAnimationFrame(updateProgress);
        }
    }

    longPressAnimationFrame = requestAnimationFrame(updateProgress);
}

function cancelLongPressTimer() {
    if (longPressAnimationFrame) {
        cancelAnimationFrame(longPressAnimationFrame);
        longPressAnimationFrame = null;
    }
    hideLongPressIndicator();
}

// Press handlers
function handleSinglePress() {
    console.log(`[${currentInputType}] single`);
    showWaitingKeywords('Extracting keywords...');

    socket.emit('search_request', {
        client_timestamp: new Date().toISOString(),
    });
}

function handleDoublePress() {
    console.log(`[${currentInputType}] double`);

    if (state.allKeywords.length === 0) {
        setStatus('error', 'No keywords. Press SPACE first.');
        return;
    }

    // Move to next keyword locally
    const nextIndex = (state.currentKeywordIndex + 1) % state.allKeywords.length;
    selectKeyword(nextIndex);
}

function handleLongPress() {
    console.log(`[${currentInputType}] long`);

    const keyword = state.currentKeyword || (state.allKeywords[0]?.keyword);
    const description = state.currentDescription || (state.allKeywords[0]?.description);

    if (!keyword) {
        setStatus('error', 'No keyword selected');
        return;
    }

    showWaitingDetail(`Getting detailed info for "${keyword}"...`);

    socket.emit('search_grounding', {
        keyword: keyword,
        description: description || '',
    });
}

// UI functions
function setStatus(type, message) {
    statusDot.className = 'status-dot ' + type;
    statusText.textContent = message;
}

function showWaitingKeywords(message = 'Extracting keywords...') {
    widget.classList.add('waiting-keywords');
    setStatus('processing', message);
}

function hideWaitingKeywords() {
    widget.classList.remove('waiting-keywords');
}

function showWaitingDetail(message = 'Loading details...') {
    widget.classList.add('waiting-detail');
    setStatus('processing', message);
}

function hideWaitingDetail() {
    widget.classList.remove('waiting-detail');
}

function renderKeywordsList() {
    if (state.allKeywords.length === 0) {
        keywordsList.innerHTML = '<div class="keywords-empty">No keywords yet</div>';
        return;
    }

    keywordsList.innerHTML = state.allKeywords.map((kw, i) => `
        <div class="keyword-item ${i === state.currentKeywordIndex ? 'active' : ''}" data-index="${i}">
            <span class="keyword-text">${kw.keyword}</span>
        </div>
    `).join('');

    // Click to select
    keywordsList.querySelectorAll('.keyword-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            selectKeyword(index);
        });
    });
}

function selectKeyword(index) {
    if (index < 0 || index >= state.allKeywords.length) return;

    state.currentKeywordIndex = index;
    const kw = state.allKeywords[index];
    state.currentKeyword = kw.keyword;
    state.currentDescription = kw.description;
    state.currentImage = kw.image;

    highlightKeyword(index);

    // Check if we have cached grounding for this keyword
    const cached = state.groundingCache[kw.keyword];
    if (cached) {
        displayKeywordDetail(kw.keyword, cached.text, cached.citations, cached.image);
    } else {
        displayDescription(kw.keyword, kw.description, kw.image);
    }

    setStatus('connected', `${kw.keyword} (${index + 1}/${state.allKeywords.length})`);
}

function highlightKeyword(index) {
    keywordsList.querySelectorAll('.keyword-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
}

function displayDescription(keyword, description, imageUrl) {
    hideEmptyState();
    hideGrounding();

    descriptionPanel.classList.add('visible');
    descriptionPanel.classList.remove('has-grounding');
    selectedKeyword.textContent = keyword;

    // Build content with inline image
    let content = description || 'No description available';
    if (imageUrl) {
        content = `<img src="${imageUrl}" alt="${keyword}" onerror="this.style.display='none'"> ${content}`;
    }
    keywordDescription.innerHTML = content;

    // Clear citation list in description panel
    const citationArea = descriptionPanel.querySelector('.description-citations');
    if (citationArea) citationArea.innerHTML = '';
}

function displayKeywordDetail(keyword, text, citations, imageUrl) {
    hideEmptyState();
    hideGrounding();

    descriptionPanel.classList.add('visible');
    descriptionPanel.classList.add('has-grounding');
    selectedKeyword.textContent = keyword;

    // Remove citation markers from text
    let processedText = text;
    if (citations && citations.length > 0) {
        citations.forEach(c => {
            const marker = `[${c.index}]`;
            processedText = processedText.split(marker).join('');
        });
    }

    // Build content with inline image
    let content = parseMarkdown(processedText);
    if (imageUrl) {
        content = `<img src="${imageUrl}" alt="${keyword}" onerror="this.style.display='none'"> ${content}`;
    }
    keywordDescription.innerHTML = content;
}

function hideDescription() {
    descriptionPanel.classList.remove('visible');
}

function displayGrounding(keyword, text, citations, imageUrl) {
    hideDescription();
    hideEmptyState();

    groundingPanel.classList.add('visible');
    groundingKeyword.textContent = keyword;

    // Process citations in text
    let processedText = text;
    if (citations && citations.length > 0) {
        citations.forEach(c => {
            const marker = `[${c.index}]`;
            const link = `<a href="${c.uri}" target="_blank" class="citation-link">[${c.index}]</a>`;
            processedText = processedText.split(marker).join(link);
        });
    }

    groundingText.innerHTML = parseMarkdown(processedText);

    // Display image if available
    if (imageUrl) {
        groundingImage.innerHTML = `<img src="${imageUrl}" alt="${keyword}" onerror="this.parentElement.style.display='none'">`;
        groundingImage.style.display = 'block';
    } else {
        groundingImage.innerHTML = '';
        groundingImage.style.display = 'none';
    }

    // Citation list
    if (citations && citations.length > 0) {
        citationList.innerHTML = `
            <div class="citation-title">Sources</div>
            ${citations.map(c => `
                <a href="${c.uri}" target="_blank" class="citation-item">
                    [${c.index}] ${c.title || 'Source'}
                </a>
            `).join('')}
        `;
    } else {
        citationList.innerHTML = '';
    }
}

function hideGrounding() {
    groundingPanel.classList.remove('visible');
}

function showEmptyState() {
    hideDescription();
    hideGrounding();
    emptyPanel.classList.add('visible');
}

function hideEmptyState() {
    emptyPanel.classList.remove('visible');
}

function showLongPressIndicator(progress) {
    const pct = Math.round(progress * 100);
    statusDot.style.background = `conic-gradient(#f59e0b ${pct}%, #e5e7eb ${pct}%)`;
    if (progress < 1) {
        setStatus('processing', 'Hold for details...');
    }
}

function hideLongPressIndicator() {
    statusDot.style.background = '';
}

function parseMarkdown(text) {
    let html = text;
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*\n]+?)\*/g, '<em>$1</em>');
    html = html.replace(/^### (.+)$/gm, '<h4>$1</h4>');
    html = html.replace(/^## (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^# (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^[\*\-] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

// Audio functions
async function toggleMicrophone() {
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, sampleRate: 48000 }
        });

        state.audioStream = stream;

        const options = { mimeType: 'audio/webm;codecs=opus' };
        state.mediaRecorder = new MediaRecorder(stream, options);
        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) state.audioChunks.push(e.data);
        };

        state.mediaRecorder.onstop = () => {
            const blob = new Blob(state.audioChunks, { type: 'audio/webm' });
            sendAudio(blob);
            state.audioChunks = [];
        };

        state.mediaRecorder.start();
        state.recordingInterval = setInterval(() => {
            if (state.mediaRecorder?.state === 'recording') {
                state.mediaRecorder.stop();
                state.mediaRecorder.start();
            }
        }, 3000);

        state.isRecording = true;
        micBtn.classList.add('active');

    } catch (err) {
        setStatus('error', 'Mic access denied');
    }
}

function stopRecording() {
    if (state.mediaRecorder?.state !== 'inactive') {
        state.mediaRecorder.stop();
    }
    if (state.recordingInterval) {
        clearInterval(state.recordingInterval);
    }
    if (state.audioStream) {
        state.audioStream.getTracks().forEach(t => t.stop());
    }

    state.isRecording = false;
    micBtn.classList.remove('active');
}

function sendAudio(blob) {
    const reader = new FileReader();
    reader.onloadend = () => {
        socket.emit('audio_chunk_openai', {
            audio: reader.result.split(',')[1],
            format: 'webm',
            timestamp: new Date().toISOString(),
        });
    };
    reader.readAsDataURL(blob);
}

function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const url = URL.createObjectURL(file);
    audioPlayer.src = url;

    const audioContext = new AudioContext({ sampleRate: 48000 });
    const source = audioContext.createMediaElementSource(audioPlayer);
    const dest = audioContext.createMediaStreamDestination();
    source.connect(dest);
    source.connect(audioContext.destination);

    const recorder = new MediaRecorder(dest.stream, { mimeType: 'audio/webm;codecs=opus' });
    let chunks = [];

    recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/webm' });
        sendAudio(blob);
        chunks = [];
    };

    recorder.start();
    const interval = setInterval(() => {
        if (recorder.state === 'recording') {
            recorder.stop();
            recorder.start();
        }
    }, 2000);

    audioPlayer.play();

    audioPlayer.onended = () => {
        if (recorder.state !== 'inactive') recorder.stop();
        clearInterval(interval);
        audioContext.close();
    };

    fileInput.value = '';
}

function clearSession() {
    if (state.isRecording) stopRecording();
    socket.emit('clear_session');
}

// Initialize UI
updateAutoInferenceUI();
showEmptyState();
