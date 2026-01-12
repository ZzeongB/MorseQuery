// MorseQuery Frontend JavaScript
const socket = io();

// State management
let state = {
    isRecording: false,
    recognitionMode: 'whisper', // 'whisper' or 'google'
    searchMode: 'gpt', // 'instant' or 'tfidf'
    searchType: 'text', // 'text' or 'image'
    mediaRecorder: null,
    audioContext: null,
    audioStream: null,
    audioChunks: [],
    currentAudioFile: null,
    isProcessingAudio: false,
    lastWord: '' // Track the most recent word
};

// DOM elements
const micBtn = document.getElementById('micBtn');
const fileInput = document.getElementById('fileInput');
const whisperBtn = document.getElementById('whisperBtn');
const googleBtn = document.getElementById('googleBtn');
const instantSearchBtn = document.getElementById('instantSearchBtn');
const recentSearchBtn = document.getElementById('recentSearchBtn');
const tfidfSearchBtn = document.getElementById('tfidfSearchBtn');
const gptSearchBtn = document.getElementById('gptSearchBtn');
const textSearchBtn = document.getElementById('textSearchBtn');
const imageSearchBtn = document.getElementById('imageSearchBtn');
const bothSearchBtn = document.getElementById('bothSearchBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const transcriptionEl = document.getElementById('transcription');
const searchSection = document.getElementById('searchSection');
const searchKeywordEl = document.getElementById('searchKeyword');
const searchResultsEl = document.getElementById('searchResults');
const audioPlayerSection = document.getElementById('audioPlayerSection');
const audioPlayer = document.getElementById('audioPlayer');
const playBtn = document.getElementById('playBtn');
const stopAudioBtn = document.getElementById('stopAudioBtn');

// Socket event handlers
socket.on('connect', () => {
    updateStatus('Connected to server');
});

socket.on('connected', (data) => {
    console.log('Server response:', data);
});

socket.on('status', (data) => {
    console.log('[Status]', data.message);
    updateStatus(data.message);
});

socket.on('error', (data) => {
    console.error('[Error]', data.message);
    updateStatus('Error: ' + data.message, true);
    alert('Error: ' + data.message);
});

socket.on('transcription', (data) => {
    console.log('[Transcription]', data);

    // Handle real-time transcription updates (like transcribe_demo.py)
    if (data.source === 'whisper-realtime') {
        if (data.is_complete) {
            // New complete phrase - append it
            appendTranscription(data.text);
            console.log('[Real-time] Complete phrase:', data.text);
        } else {
            // Update current phrase - replace last line
            updateCurrentTranscription(data.text);
            console.log('[Real-time] Updating phrase:', data.text);
        }
    } else {
        // Standard transcription - just append
        appendTranscription(data.text);
    }

    // Track the last word for instant search
    const words = data.text.trim().split(/\s+/);
    if (words.length > 0) {
        state.lastWord = words[words.length - 1];
    }

    updateStatus(`Transcribed (${data.source}): ${data.text}`);
});

socket.on('search_keyword', (data) => {
    console.log('[Search Keyword]', data.keyword);
    updateStatus(`🔍 Searching for: "${data.keyword}"`);
});

socket.on('search_results', (data) => {
    displaySearchResults(data);
});

socket.on('session_cleared', () => {
    transcriptionEl.textContent = '';
    searchSection.style.display = 'none';
    updateStatus('Session cleared');
});

// Button event listeners
micBtn.addEventListener('click', toggleMicrophone);
fileInput.addEventListener('change', handleFileUpload);
whisperBtn.addEventListener('click', () => setRecognitionMode('whisper'));
googleBtn.addEventListener('click', () => setRecognitionMode('google'));
instantSearchBtn.addEventListener('click', () => setSearchMode('instant'));
recentSearchBtn.addEventListener('click', () => setSearchMode('recent'));
tfidfSearchBtn.addEventListener('click', () => setSearchMode('tfidf'));
gptSearchBtn.addEventListener('click', () => setSearchMode('gpt'));
textSearchBtn.addEventListener('click', () => setSearchType('text'));
imageSearchBtn.addEventListener('click', () => setSearchType('image'));
bothSearchBtn.addEventListener('click', () => setSearchType('both'));
clearBtn.addEventListener('click', clearSession);
playBtn.addEventListener('click', playAndTranscribe);
stopAudioBtn.addEventListener('click', stopAudioPlayback);

// Spacebar listener for search
document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();
        triggerSearch();
    }
});

// Functions
function updateStatus(message, isError = false) {
    statusEl.textContent = 'Status: ' + message;
    statusEl.style.color = isError ? '#dc3545' : '#667eea';
}

function appendTranscription(text) {
    if (transcriptionEl.textContent === 'Transcribed text will appear here...') {
        transcriptionEl.textContent = '';
    }
    transcriptionEl.textContent += (transcriptionEl.textContent ? ' ' : '') + text;
    transcriptionEl.scrollTop = transcriptionEl.scrollHeight;
}

function updateCurrentTranscription(text) {
    // Update the last phrase in real-time (like transcribe_demo.py)
    // This replaces the current incomplete phrase with updated text
    const lines = transcriptionEl.textContent.split('\n');
    if (lines.length > 0 && transcriptionEl.textContent !== 'Transcribed text will appear here...') {
        // Replace the last line with the new text
        lines[lines.length - 1] = text;
        transcriptionEl.textContent = lines.join('\n');
    } else {
        transcriptionEl.textContent = text;
    }
    transcriptionEl.scrollTop = transcriptionEl.scrollHeight;
}

function setRecognitionMode(mode) {
    state.recognitionMode = mode;

    whisperBtn.classList.toggle('active', mode === 'whisper');
    googleBtn.classList.toggle('active', mode === 'google');

    updateStatus(`Recognition mode: ${mode}`);

    if (mode === 'whisper') {
        socket.emit('start_whisper');
    } else if (mode === 'google') {
        socket.emit('start_google_streaming');
    }
}

function setSearchMode(mode) {
    state.searchMode = mode;

    instantSearchBtn.classList.toggle('active', mode === 'instant');
    recentSearchBtn.classList.toggle('active', mode === 'recent');
    tfidfSearchBtn.classList.toggle('active', mode === 'tfidf');
    gptSearchBtn.classList.toggle('active', mode === 'gpt');

    const modeNames = {
        instant: 'Instant Word (최신 단어)',
        recent: 'Recent 5s (최근 5초)',
        tfidf: 'Important Word (중요 단어)',
        gpt: 'GPT 4o mini'
    };

    updateStatus(`Search mode: ${modeNames[mode]}`);
}

function setSearchType(type) {
    state.searchType = type;

    textSearchBtn.classList.toggle('active', type === 'text');
    imageSearchBtn.classList.toggle('active', type === 'image');
    bothSearchBtn.classList.toggle('active', type === 'both');

    const typeNames = {
        text: 'Text Only',
        image: 'Image Only',
        both: 'Text+Image'
    };

    updateStatus(`Search type: ${typeNames[type]}`);
}

async function toggleMicrophone() {
    if (state.isRecording) {
        stopRecording();
    } else {
        await startRecording();
    }
}

async function startRecording() {
    if (!state.recognitionMode) {
        alert('Please select a recognition mode first (Whisper or Google STT)');
        return;
    }

    try {
        // Show transcription and info sections for microphone input
        document.querySelector('.transcription-section').style.display = 'block';
        document.querySelector('.info-box').style.display = 'block';

        // Request microphone access
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 48000
            }
        });

        state.audioStream = stream;
        state.audioContext = new AudioContext({ sampleRate: 48000 });

        // Create MediaRecorder
        const options = { mimeType: 'audio/webm;codecs=opus' };
        state.mediaRecorder = new MediaRecorder(stream, options);

        state.audioChunks = [];

        state.mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                state.audioChunks.push(event.data);
            }
        };

        state.mediaRecorder.onstop = () => {
            const audioBlob = new Blob(state.audioChunks, { type: 'audio/webm' });
            sendAudioToServer(audioBlob);
            state.audioChunks = [];
        };

        // Start recording and send chunks every 3 seconds
        state.mediaRecorder.start();
        state.recordingInterval = setInterval(() => {
            if (state.mediaRecorder && state.mediaRecorder.state === 'recording') {
                state.mediaRecorder.stop();
                state.mediaRecorder.start();
            }
        }, 3000);

        state.isRecording = true;
        micBtn.classList.add('recording');
        micBtn.innerHTML = '<span class="icon">⏹️</span> Stop';
        updateStatus('Recording... Press SPACE to search');

    } catch (error) {
        updateStatus('Microphone access denied: ' + error.message, true);
        console.error('Error accessing microphone:', error);
    }
}

function stopRecording() {
    if (state.mediaRecorder && state.mediaRecorder.state !== 'inactive') {
        state.mediaRecorder.stop();
    }

    if (state.recordingInterval) {
        clearInterval(state.recordingInterval);
    }

    if (state.audioStream) {
        state.audioStream.getTracks().forEach(track => track.stop());
    }

    if (state.audioContext) {
        state.audioContext.close();
    }

    state.isRecording = false;
    micBtn.classList.remove('recording');
    micBtn.innerHTML = '<span class="icon">🎤</span> Microphone';
    updateStatus('Recording stopped');
}

function handleFileUpload(event) {
    if (!state.recognitionMode) {
        alert('Please select a recognition mode first (Whisper or Google STT)');
        fileInput.value = '';
        return;
    }

    const file = event.target.files[0];
    if (!file) return;

    // Store the file
    state.currentAudioFile = file;

    // Create object URL for audio player
    const audioURL = URL.createObjectURL(file);
    audioPlayer.src = audioURL;

    // Hide transcription and info sections when playing uploaded file
    document.querySelector('.transcription-section').style.display = 'none';
    document.querySelector('.info-box').style.display = 'none';

    // Show audio player section
    audioPlayerSection.style.display = 'block';

    updateStatus(`Audio file loaded: ${file.name}. Click "Play & Transcribe" to start.`);

    // Reset file input
    fileInput.value = '';
}

function sendAudioToServer(audioBlob) {
    const reader = new FileReader();
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];

        const eventName = state.recognitionMode === 'whisper'
            ? 'audio_chunk_whisper'
            : 'audio_chunk_google';

        socket.emit(eventName, { audio: base64Audio, format: 'webm' });
    };
    reader.readAsDataURL(audioBlob);
}

function sendAudioToServerWithFormat(audioBlob, format) {
    const reader = new FileReader();
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];

        const eventName = state.recognitionMode === 'whisper'
            ? 'audio_chunk_whisper'
            : 'audio_chunk_google';

        console.log(`[Send] Sending audio to server via ${eventName}: format=${format}, size=${audioBlob.size} bytes, base64 length=${base64Audio.length}`);
        socket.emit(eventName, { audio: base64Audio, format: format });

        updateStatus(`Audio sent (${format}, ${Math.round(audioBlob.size / 1024)}KB). Processing...`);
    };
    reader.readAsDataURL(audioBlob);
}

async function processFileInRealTimeChunks(audioFile, fileExt) {
    const CHUNK_DURATION = 2; // seconds per chunk (like record_timeout in demo)

    console.log(`[ProcessFileInRealTimeChunks] Starting real-time processing`);
    updateStatus('Processing audio in real-time...');

    // Read file as ArrayBuffer
    const arrayBuffer = await audioFile.arrayBuffer();

    // Create AudioContext to decode audio
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    console.log(`[ProcessFileInRealTimeChunks] Audio loaded: ${audioBuffer.duration}s, ${audioBuffer.sampleRate}Hz`);

    const sampleRate = audioBuffer.sampleRate;
    const totalDuration = audioBuffer.duration;
    const numChunks = Math.ceil(totalDuration / CHUNK_DURATION);

    console.log(`[ProcessFileInRealTimeChunks] Splitting into ${numChunks} chunks of ${CHUNK_DURATION}s each`);

    // Process each chunk sequentially
    for (let i = 0; i < numChunks; i++) {
        const startTime = i * CHUNK_DURATION;
        const endTime = Math.min((i + 1) * CHUNK_DURATION, totalDuration);
        const startSample = Math.floor(startTime * sampleRate);
        const endSample = Math.floor(endTime * sampleRate);
        const chunkLength = endSample - startSample;

        console.log(`[Chunk ${i+1}/${numChunks}] Time: ${startTime.toFixed(2)}s - ${endTime.toFixed(2)}s`);

        // Create a new buffer for this chunk
        const chunkBuffer = audioContext.createBuffer(
            1, // mono
            chunkLength,
            sampleRate
        );

        // Copy audio data for this chunk
        const sourceData = audioBuffer.getChannelData(0);
        const chunkData = chunkBuffer.getChannelData(0);
        for (let j = 0; j < chunkLength; j++) {
            chunkData[j] = sourceData[startSample + j];
        }

        // Convert chunk to WAV blob
        const chunkBlob = await audioBufferToWav(chunkBuffer);

        // Send chunk to server
        const isFinal = (i === numChunks - 1);
        await sendAudioChunkRealtime(chunkBlob, 'wav', isFinal);

        updateStatus(`Processing chunk ${i+1}/${numChunks}...`);

        // Wait a bit between chunks to simulate real-time processing
        await new Promise(resolve => setTimeout(resolve, 500));
    }

    console.log('[ProcessFileInRealTimeChunks] All chunks sent');
    updateStatus('All chunks processed');

    audioContext.close();
}

function audioBufferToWav(audioBuffer) {
    // Convert AudioBuffer to WAV format
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const data = audioBuffer.getChannelData(0);
    const dataLength = data.length * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // fmt chunk size
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);

    // Write audio data
    const volume = 0.8;
    let offset = 44;
    for (let i = 0; i < data.length; i++) {
        const sample = Math.max(-1, Math.min(1, data[i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
}

function sendAudioChunkRealtime(audioBlob, format, isFinal) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64Audio = reader.result.split(',')[1];

            console.log(`[SendChunkRealtime] Sending chunk: format=${format}, size=${audioBlob.size} bytes, is_final=${isFinal}`);

            socket.emit('audio_chunk_realtime', {
                audio: base64Audio,
                format: format,
                is_final: isFinal,
                phrase_timeout: 3
            });

            resolve();
        };
        reader.onerror = reject;
        reader.readAsDataURL(audioBlob);
    });
}

async function captureAudioPlaybackRealtime() {
    console.log('[CapturePlayback] Setting up real-time audio capture from player');

    // Create AudioContext and connect audio player as source
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    const source = audioContext.createMediaElementSource(audioPlayer);

    // Create destination for capturing audio
    const destination = audioContext.createMediaStreamDestination();

    // Connect: audioPlayer -> destination (for capture) AND -> audioContext.destination (for playback)
    source.connect(destination);
    source.connect(audioContext.destination);

    // Create MediaRecorder to capture the stream (like microphone recording)
    const options = { mimeType: 'audio/webm;codecs=opus' };
    const mediaRecorder = new MediaRecorder(destination.stream, options);

    let audioChunks = [];

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
            audioChunks.push(event.data);
        }
    };

    mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        sendAudioToServer(audioBlob);
        audioChunks = [];
    };

    // Start recording from the audio player output
    mediaRecorder.start();

    // Send chunks every 2 seconds (like microphone recording)
    const recordingInterval = setInterval(() => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            mediaRecorder.start();
        }
    }, 2000);

    // Play the audio
    audioPlayer.play();
    updateStatus('Playing and transcribing in real-time...');

    // Clean up when playback ends
    audioPlayer.onended = () => {
        console.log('[CapturePlayback] Playback ended, cleaning up');

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        clearInterval(recordingInterval);

        audioContext.close();

        state.isProcessingAudio = false;
        playBtn.disabled = false;
        playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
        updateStatus('Audio playback and transcription finished');
    };

    // Handle pause/stop
    audioPlayer.onpause = () => {
        if (!audioPlayer.ended) {
            console.log('[CapturePlayback] Playback paused');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            clearInterval(recordingInterval);
        }
    };
}

async function playAndTranscribe() {
    if (!state.currentAudioFile) {
        updateStatus('No audio file loaded', true);
        return;
    }

    if (state.isProcessingAudio) {
        updateStatus('Already processing audio', true);
        return;
    }

    state.isProcessingAudio = true;
    playBtn.disabled = true;
    playBtn.innerHTML = '<span class="icon">⏳</span> Processing...';

    console.log(`[PlayAndTranscribe] Starting real-time capture of audio playback`);

    try {
        // Capture audio playback in real-time (like microphone)
        await captureAudioPlaybackRealtime();
    } catch (error) {
        console.error('[PlayAndTranscribe] Error:', error);
        updateStatus('Error processing file: ' + error.message, true);
        state.isProcessingAudio = false;
        playBtn.disabled = false;
        playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    }
}

function stopAudioPlayback() {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    state.isProcessingAudio = false;
    playBtn.disabled = false;
    playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    updateStatus('Audio playback stopped');
}

function triggerSearch() {
    if (!transcriptionEl.textContent || transcriptionEl.textContent === 'Transcribed text will appear here...') {
        updateStatus('No transcription available for search', true);
        return;
    }

    if (state.searchMode === 'instant') {
        // Instant search: use the last word
        if (!state.lastWord) {
            updateStatus('No word available for instant search', true);
            return;
        }

        console.log(`[Instant Search] Searching for last word: "${state.lastWord}"`);
        updateStatus(`⚡ Instant search: "${state.lastWord}"`);

        socket.emit('search_request', {
            mode: 'instant',
            keyword: state.lastWord,
            type: state.searchType
        });

    } else if (state.searchMode === 'recent') {
        // Recent search: important word from last 5 seconds
        console.log('[Recent Search] Requesting important keyword from last 5 seconds');
        updateStatus('⏱️ Finding important word from last 5 seconds...');

        socket.emit('search_request', {
            mode: 'recent',
            time_threshold: 5,
            type: state.searchType
        });

    } else if (state.searchMode === 'gpt') {
        // GPT search: use GPT to predict what user wants to look up
        console.log('[GPT Search] Requesting GPT prediction from last 3 seconds');
        updateStatus('🤖 GPT is predicting keyword...');

        socket.emit('search_request', {
            mode: 'gpt',
            time_threshold: 3,
            type: state.searchType
        });

    } else {
        // TF-IDF search: let server calculate important word
        console.log('[TF-IDF Search] Requesting important keyword from server');
        updateStatus('🎯 Calculating important keyword...');

        socket.emit('search_request', {
            mode: 'tfidf',
            type: state.searchType
        });
    }
}

function displaySearchResults(data) {
    let modeIcon, modeName;
    if (data.mode === 'instant') {
        modeIcon = '⚡';
        modeName = 'Instant';
    } else if (data.mode === 'recent') {
        modeIcon = '⏱️';
        modeName = 'Recent 5s';
    } else if (data.mode === 'gpt') {
        modeIcon = '🤖';
        modeName = 'GPT 4o mini';
    } else {
        modeIcon = '🎯';
        modeName = 'Important';
    }

    searchKeywordEl.textContent = `${modeIcon} ${data.keyword} (${modeName})`;
    searchResultsEl.innerHTML = '';

    // Handle 'both' type - display both text and image results
    if (data.type === 'both') {
        const textResults = data.results.text || [];
        const imageResults = data.results.image || [];

        if (textResults.length === 0 && imageResults.length === 0) {
            searchResultsEl.innerHTML = '<p>No results found.</p>';
            searchSection.style.display = 'block';
            return;
        }

        // Display text results section
        if (textResults.length > 0) {
            const textHeader = document.createElement('h3');
            textHeader.textContent = '📝 Text Results';
            textHeader.style.marginTop = '0';
            searchResultsEl.appendChild(textHeader);

            textResults.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'search-result-item';

                // If image is available, display it alongside the text
                if (result.image) {
                    resultDiv.innerHTML = `
                        <div style="display: flex; gap: 15px; align-items: start;">
                            <img src="${result.image}" alt="${result.title}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 8px; flex-shrink: 0;" onerror="this.style.display='none'">
                            <div style="flex: 1;">
                                <h4>${result.title}</h4>
                                <p>${result.snippet}</p>
                            </div>
                        </div>
                    `;
                } else {
                    resultDiv.innerHTML = `
                        <h4>${result.title}</h4>
                        <p>${result.snippet}</p>
                    `;
                }
                searchResultsEl.appendChild(resultDiv);
            });
        }

        // Display image results section
        if (imageResults.length > 0) {
            const imageHeader = document.createElement('h3');
            imageHeader.textContent = '🖼️ Image Results';
            imageHeader.style.marginTop = '20px';
            searchResultsEl.appendChild(imageHeader);

            imageResults.forEach(result => {
                const resultDiv = document.createElement('div');
                resultDiv.className = 'search-result-item image-result';
                resultDiv.innerHTML = `
                    <a href="${result.link}" target="_blank">
                        <img src="${result.thumbnail}" alt="${result.title}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22150%22 height=%22150%22%3E%3Crect fill=%22%23ddd%22 width=%22150%22 height=%22150%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3ENo Image%3C/text%3E%3C/svg%3E'">
                    </a>
                    <div class="image-result-info">
                        <h4>${result.title}</h4>
                        <p>${result.context}</p>
                    </div>
                `;
                searchResultsEl.appendChild(resultDiv);
            });
        }

        searchSection.style.display = 'block';
        updateStatus(`Found ${textResults.length} text + ${imageResults.length} image results for "${data.keyword}"`);
        return;
    }

    // Handle single type (text or image)
    if (data.results.length === 0) {
        searchResultsEl.innerHTML = '<p>No results found.</p>';
        searchSection.style.display = 'block';
        return;
    }

    if (data.type === 'image') {
        data.results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'search-result-item image-result';
            resultDiv.innerHTML = `
                <a href="${result.link}" target="_blank">
                    <img src="${result.thumbnail}" alt="${result.title}" onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22150%22 height=%22150%22%3E%3Crect fill=%22%23ddd%22 width=%22150%22 height=%22150%22/%3E%3Ctext x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22 dy=%22.3em%22%3ENo Image%3C/text%3E%3C/svg%3E'">
                </a>
                <div class="image-result-info">
                    <h4>${result.title}</h4>
                    <p>${result.context}</p>
                </div>
            `;
            searchResultsEl.appendChild(resultDiv);
        });
    } else {
        data.results.forEach(result => {
            const resultDiv = document.createElement('div');
            resultDiv.className = 'search-result-item';

            // If image is available, display it alongside the text
            if (result.image) {
                resultDiv.innerHTML = `
                    <div style="display: flex; gap: 15px; align-items: start;">
                        <img src="${result.image}" alt="${result.title}" style="width: 120px; height: 120px; object-fit: cover; border-radius: 8px; flex-shrink: 0;" onerror="this.style.display='none'">
                        <div style="flex: 1;">
                            <h4>${result.title}</h4>
                            <p>${result.snippet}</p>
                        </div>
                    </div>
                `;
            } else {
                resultDiv.innerHTML = `
                    <h4>${result.title}</h4>
                    <p>${result.snippet}</p>
                `;
            }
            searchResultsEl.appendChild(resultDiv);
        });
    }

    searchSection.style.display = 'block';
    updateStatus(`Found ${data.results.length} results for "${data.keyword}"`);
}

function clearSession() {
    if (state.isRecording) {
        stopRecording();
    }

    // Stop audio playback if playing
    if (state.isProcessingAudio) {
        stopAudioPlayback();
    }

    // Hide and reset audio player
    audioPlayerSection.style.display = 'none';
    audioPlayer.src = '';
    state.currentAudioFile = null;

    // Show transcription and info sections again
    document.querySelector('.transcription-section').style.display = 'block';
    document.querySelector('.info-box').style.display = 'block';

    socket.emit('clear_session');
}

// Initialize - Auto-load Whisper model
socket.emit('start_whisper');
updateStatus('Loading Whisper model...');
