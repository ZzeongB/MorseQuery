// MorseQuery Simple - Widget Frontend
const socket = io();

// State
let state = {
    isRecording: false,
    mediaRecorder: null,
    audioStream: null,
    audioChunks: [],
    currentKeywords: [],
    currentKeywordIndex: 0,
    keywordHistory: [],
    currentKeyword: null,
    currentDescription: null,
};

// DOM elements
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const keywordsPanel = document.getElementById('keywordsPanel');
const keywordsList = document.getElementById('keywordsList');
const descriptionPanel = document.getElementById('descriptionPanel');
const selectedKeyword = document.getElementById('selectedKeyword');
const keywordDescription = document.getElementById('keywordDescription');
const groundingPanel = document.getElementById('groundingPanel');
const groundingKeyword = document.getElementById('groundingKeyword');
const groundingText = document.getElementById('groundingText');
const citationList = document.getElementById('citationList');
const historyPanel = document.getElementById('historyPanel');
const historyList = document.getElementById('historyList');
const micBtn = document.getElementById('micBtn');
const fileBtn = document.getElementById('fileBtn');
const clearBtn = document.getElementById('clearBtn');
const fileInput = document.getElementById('fileInput');
const audioPlayer = document.getElementById('audioPlayer');

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
    // Just update status briefly to show we're receiving audio
    if (data.is_complete) {
        setStatus('listening', 'Listening...');
    }
});

socket.on('keywords_extracted', (data) => {
    console.log('[Keywords]', data);
    state.currentKeywords = data.keywords || [];
    state.currentKeywordIndex = 0;
    state.keywordHistory = data.history || [];

    displayKeywords(state.currentKeywords);
    updateHistory(state.keywordHistory);
    hideDescription();
    hideGrounding();

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
    displayGrounding(data.keyword, data.text, data.citations);
    setStatus('connected', 'Detailed info loaded');
});

socket.on('keyword_history', (data) => {
    state.keywordHistory = data.history || [];
    updateHistory(state.keywordHistory);
});

socket.on('session_cleared', () => {
    state.currentKeywords = [];
    state.currentKeywordIndex = 0;
    state.keywordHistory = [];
    state.currentKeyword = null;
    state.currentDescription = null;

    hideKeywords();
    hideDescription();
    hideGrounding();
    updateHistory([]);

    setStatus('connected', 'Session cleared');
});

// Button events
micBtn.addEventListener('click', toggleMicrophone);
fileBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
clearBtn.addEventListener('click', clearSession);

// Spacebar handling
const DOUBLE_PRESS_THRESHOLD = 300;
const LONG_PRESS_THRESHOLD = 500;

let spacebarTimer = null;
let spacebarPressCount = 0;
let spacebarDownTime = 0;
let longPressAnimationFrame = null;
let longPressTriggered = false;

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();

        if (!e.repeat && spacebarDownTime === 0) {
            spacebarDownTime = Date.now();
            longPressTriggered = false;
            startLongPressTimer();
        }
    }
});

document.addEventListener('keyup', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        spacebarDownTime = 0;
        cancelLongPressTimer();

        if (longPressTriggered) {
            longPressTriggered = false;
            spacebarPressCount = 0;
            return;
        }

        spacebarPressCount++;

        if (spacebarPressCount === 2) {
            // Double press: next keyword / show description
            clearTimeout(spacebarTimer);
            spacebarTimer = null;
            spacebarPressCount = 0;
            handleDoublePress();
            return;
        }

        if (spacebarTimer) clearTimeout(spacebarTimer);
        spacebarTimer = setTimeout(() => {
            spacebarTimer = null;
            if (spacebarPressCount === 1) {
                handleSinglePress();
            }
            spacebarPressCount = 0;
        }, DOUBLE_PRESS_THRESHOLD);
    }
});

function startLongPressTimer() {
    const startTime = Date.now();

    function updateProgress() {
        const elapsed = Date.now() - startTime;
        const progress = Math.min(elapsed / LONG_PRESS_THRESHOLD, 1);

        showLongPressIndicator(progress);

        if (progress >= 1 && !longPressTriggered) {
            longPressTriggered = true;
            cancelLongPressTimer();
            handleLongPress();
        } else if (progress < 1) {
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
    console.log('[Action] Single press - extract keywords');
    setStatus('processing', 'Extracting keywords...');

    socket.emit('search_request', {
        client_timestamp: new Date().toISOString(),
    });
}

function handleDoublePress() {
    console.log('[Action] Double press - next keyword');

    if (state.currentKeywords.length === 0) {
        setStatus('error', 'No keywords. Press SPACE first.');
        return;
    }

    socket.emit('next_keyword', {});
}

function handleLongPress() {
    console.log('[Action] Long press - get grounding');

    const keyword = state.currentKeyword || (state.currentKeywords[0]?.keyword);
    const description = state.currentDescription || (state.currentKeywords[0]?.description);

    if (!keyword) {
        setStatus('error', 'No keyword selected');
        return;
    }

    setStatus('processing', 'Getting detailed info...');

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

function displayKeywords(keywords) {
    keywordsPanel.classList.add('visible');

    keywordsList.innerHTML = keywords.map((kw, i) => `
        <div class="keyword-item ${i === 0 ? 'active' : ''}" data-index="${i}">
            <span class="keyword-text">${kw.keyword}</span>
        </div>
    `).join('');

    // Click to select
    keywordsList.querySelectorAll('.keyword-item').forEach(item => {
        item.addEventListener('click', () => {
            const index = parseInt(item.dataset.index);
            socket.emit('select_keyword', { index });
        });
    });

    // Auto-select first keyword
    if (keywords.length > 0) {
        state.currentKeyword = keywords[0].keyword;
        state.currentDescription = keywords[0].description;
    }
}

function hideKeywords() {
    keywordsPanel.classList.remove('visible');
    keywordsList.innerHTML = '';
}

function highlightKeyword(index) {
    keywordsList.querySelectorAll('.keyword-item').forEach((item, i) => {
        item.classList.toggle('active', i === index);
    });
}

function displayDescription(keyword, description) {
    descriptionPanel.classList.add('visible');
    selectedKeyword.textContent = keyword;
    keywordDescription.textContent = description || 'No description available';
}

function hideDescription() {
    descriptionPanel.classList.remove('visible');
}

function displayGrounding(keyword, text, citations) {
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

function updateHistory(history) {
    if (history.length === 0) {
        historyPanel.classList.remove('visible');
        return;
    }

    historyPanel.classList.add('visible');
    historyList.innerHTML = history.map(h => `
        <div class="history-item">${h.keyword}</div>
    `).join('');
}

function showLongPressIndicator(progress) {
    const pct = Math.round(progress * 100);
    statusDot.style.background = `conic-gradient(#10a37f ${pct}%, #e5e7eb ${pct}%)`;
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
        setStatus('listening', 'Listening...');

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
    setStatus('connected', 'Stopped');
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

    // Capture and stream audio
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
    setStatus('listening', 'Playing file...');

    audioPlayer.onended = () => {
        if (recorder.state !== 'inactive') recorder.stop();
        clearInterval(interval);
        audioContext.close();
        setStatus('connected', 'Playback finished');
    };

    fileInput.value = '';
}

function clearSession() {
    if (state.isRecording) stopRecording();
    socket.emit('clear_session');
}
