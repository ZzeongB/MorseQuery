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
    lastWord: '', // Track the most recent word
    isVideoFile: false, // Track if current file is video
    srtLoaded: false, // Track if SRT file is loaded
    srtTimeUpdateInterval: null, // Interval for SRT time updates
    // Double-spacebar detection for GPT keyword navigation
    lastSpacebarTime: 0,
    hasGptKeywords: false, // Track if GPT keywords are available
    totalGptKeywords: 0,
    currentGptKeywordIndex: 0,
    // Options
    showSearchResults: true, // Toggle Google search results on/off
    showAllKeywords: false   // Show all GPT keywords at once instead of one by one
};

// DOM elements
const micBtn = document.getElementById('micBtn');
const fileInput = document.getElementById('fileInput');
const srtInput = document.getElementById('srtInput');
const whisperBtn = document.getElementById('whisperBtn');
const instantSearchBtn = document.getElementById('instantSearchBtn');
const recentSearchBtn = document.getElementById('recentSearchBtn');
const tfidfSearchBtn = document.getElementById('tfidfSearchBtn');
const gptSearchBtn = document.getElementById('gptSearchBtn');
const textSearchBtn = document.getElementById('textSearchBtn');
const imageSearchBtn = document.getElementById('imageSearchBtn');
const bothSearchBtn = document.getElementById('bothSearchBtn');
const toggleSearchResultsBtn = document.getElementById('toggleSearchResultsBtn');
const showAllKeywordsBtn = document.getElementById('showAllKeywordsBtn');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');
const transcriptionEl = document.getElementById('transcription');

// Layout containers
const videoLayout = document.getElementById('videoLayout');
const defaultLayout = document.getElementById('defaultLayout');

// Default layout elements
const searchSection = document.getElementById('searchSection');
const searchKeywordEl = document.getElementById('searchKeyword');
const searchResultsEl = document.getElementById('searchResults');
const audioPlayerSection = document.getElementById('audioPlayerSection');
const audioPlayer = document.getElementById('audioPlayer');
const playBtn = document.getElementById('playBtn');
const stopAudioBtn = document.getElementById('stopAudioBtn');

// Video layout elements
const videoPlayer = document.getElementById('videoPlayer');
const playVideoBtn = document.getElementById('playVideoBtn');
const stopVideoBtn = document.getElementById('stopVideoBtn');
const videoSearchSection = document.getElementById('videoSearchSection');
const videoSearchKeywordEl = document.getElementById('videoSearchKeyword');
const videoSearchResultsEl = document.getElementById('videoSearchResults');

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
    console.log('[Search Keyword]', data);

    // Update GPT keyword state for double-spacebar navigation
    if (data.mode === 'gpt' && data.total_keywords > 0) {
        state.hasGptKeywords = true;
        state.totalGptKeywords = data.total_keywords;
        state.currentGptKeywordIndex = data.current_index;
    }

    // Build status message with description if available
    let statusMsg = `🔍 Searching for: "${data.keyword}"`;
    if (data.mode === 'gpt' && data.total_keywords > 1) {
        statusMsg += ` (${data.current_index + 1}/${data.total_keywords})`;
    }
    if (data.description) {
        statusMsg += ` - ${data.description}`;
    }
    if (data.mode === 'gpt' && data.total_keywords > 1) {
        statusMsg += ' [Double-space for next]';
    }
    updateStatus(statusMsg);

    // Update keyword display with description
    displayKeywordWithDescription(data);
});

socket.on('search_results', (data) => {
    displaySearchResults(data);
});

socket.on('all_keywords', (data) => {
    console.log('[All Keywords]', data);
    displayAllKeywords(data);
});

socket.on('session_cleared', () => {
    transcriptionEl.textContent = '';
    searchSection.style.display = 'none';
    updateStatus('Session cleared');
});

socket.on('srt_loaded', (data) => {
    console.log('[SRT] Loaded:', data);
    state.srtLoaded = true;
    state.recognitionMode = null; // Disable Whisper when SRT is loaded

    // Update UI - deactivate Whisper button, activate SRT
    whisperBtn.classList.remove('active');
    srtInput.parentElement.classList.add('active');

    const autoMsg = data.auto ? ' (auto-detected)' : '';
    updateStatus(`SRT loaded: ${data.count} entries${autoMsg}. Click "Play & Transcribe" to start.`);
});

socket.on('srt_not_found', (data) => {
    console.log('[SRT] Not found for:', data.filename);
    if (!state.srtLoaded) {
        updateStatus(`No SRT found for ${data.filename}. Will use Whisper for transcription.`);
    }
});

// Button event listeners
micBtn.addEventListener('click', toggleMicrophone);
fileInput.addEventListener('change', handleFileUpload);
srtInput.addEventListener('change', handleSrtUpload);
whisperBtn.addEventListener('click', () => setRecognitionMode('whisper'));
instantSearchBtn.addEventListener('click', () => setSearchMode('instant'));
recentSearchBtn.addEventListener('click', () => setSearchMode('recent'));
tfidfSearchBtn.addEventListener('click', () => setSearchMode('tfidf'));
gptSearchBtn.addEventListener('click', () => setSearchMode('gpt'));
textSearchBtn.addEventListener('click', () => setSearchType('text'));
imageSearchBtn.addEventListener('click', () => setSearchType('image'));
bothSearchBtn.addEventListener('click', () => setSearchType('both'));
toggleSearchResultsBtn.addEventListener('click', toggleSearchResults);
showAllKeywordsBtn.addEventListener('click', toggleShowAllKeywords);
clearBtn.addEventListener('click', clearSession);
playBtn.addEventListener('click', playAndTranscribe);
stopAudioBtn.addEventListener('click', stopAudioPlayback);
playVideoBtn.addEventListener('click', playVideoAndTranscribe);
stopVideoBtn.addEventListener('click', stopVideoPlayback);

// Spacebar listener for search (with double-spacebar detection)
const DOUBLE_SPACEBAR_THRESHOLD = 400; // ms
let spacebarTimer = null;

document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
        e.preventDefault();

        // Check if double-spacebar detection should be active (GPT mode with multiple keywords)
        if (state.searchMode === 'gpt' &&
            state.hasGptKeywords &&
            state.totalGptKeywords > 1) {

            if (spacebarTimer) {
                // Second spacebar within threshold - cancel pending search and trigger next keyword
                clearTimeout(spacebarTimer);
                spacebarTimer = null;
                console.log('[Double Spacebar] Requesting next keyword');
                triggerNextKeyword();
            } else {
                // First spacebar - wait for potential second
                spacebarTimer = setTimeout(() => {
                    spacebarTimer = null;
                    triggerSearch();
                }, DOUBLE_SPACEBAR_THRESHOLD);
            }
        } else {
            // Not in GPT mode with multiple keywords - trigger search immediately
            triggerSearch();
        }
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

    // Clear SRT when Whisper is selected
    if (state.srtLoaded) {
        state.srtLoaded = false;
        srtInput.parentElement.classList.remove('active');
        socket.emit('clear_srt');
    }

    whisperBtn.classList.toggle('active', mode === 'whisper');

    updateStatus(`Recognition mode: ${mode}`);

    if (mode === 'whisper') {
        socket.emit('start_whisper');
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

function toggleSearchResults() {
    state.showSearchResults = !state.showSearchResults;
    toggleSearchResultsBtn.classList.toggle('active', state.showSearchResults);
    updateStatus(`Search results: ${state.showSearchResults ? 'ON' : 'OFF'}`);
}

function toggleShowAllKeywords() {
    state.showAllKeywords = !state.showAllKeywords;
    showAllKeywordsBtn.classList.toggle('active', state.showAllKeywords);
    updateStatus(`Show all keywords: ${state.showAllKeywords ? 'ON' : 'OFF'}`);
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
        // Use default layout for microphone recording
        videoLayout.style.display = 'none';
        defaultLayout.style.display = 'block';

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

function handleSrtUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
        const srtContent = e.target.result;
        socket.emit('load_srt', { content: srtContent });
        console.log('[SRT] Sent SRT content to server');
    };
    reader.readAsText(file);
    srtInput.value = '';
    updateStatus(`Loading SRT file: ${file.name}...`);
}

function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const fileName = file.name.toLowerCase();

    // For audio/video, need recognition mode unless SRT is already loaded
    if (!state.srtLoaded && !state.recognitionMode) {
        alert('Please select a recognition mode (Whisper/Google STT) or load an SRT file first');
        fileInput.value = '';
        return;
    }

    // Store the file
    state.currentAudioFile = file;

    // Check file extension to determine if it's a video file
    const isVideo = fileName.endsWith('.mp4') || fileName.endsWith('.mov') || fileName.endsWith('.avi') || fileName.endsWith('.webm');
    state.isVideoFile = isVideo;

    // Create object URL
    const fileURL = URL.createObjectURL(file);

    // Check for matching SRT file on server
    socket.emit('check_srt_for_media', { filename: file.name });

    if (isVideo) {
        // Video file: show video layout with left video + right search results
        videoLayout.style.display = 'block';
        defaultLayout.style.display = 'none';
        document.querySelector('.info-box').style.display = 'none';

        videoPlayer.src = fileURL;

        updateStatus(`Video file loaded: ${file.name}. Checking for SRT...`);
    } else {
        // Audio file: show default layout with transcription and audio player
        videoLayout.style.display = 'none';
        defaultLayout.style.display = 'block';
        document.querySelector('.transcription-section').style.display = 'none';
        document.querySelector('.info-box').style.display = 'none';

        audioPlayer.src = fileURL;
        audioPlayerSection.style.display = 'block';

        updateStatus(`Audio file loaded: ${file.name}. Checking for SRT...`);
    }

    // Reset file input
    fileInput.value = '';
}

function sendAudioToServer(audioBlob) {
    const reader = new FileReader();
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];
        socket.emit('audio_chunk_whisper', { audio: base64Audio, format: 'webm' });
    };
    reader.readAsDataURL(audioBlob);
}

function sendAudioToServerWithFormat(audioBlob, format) {
    const reader = new FileReader();
    reader.onloadend = () => {
        const base64Audio = reader.result.split(',')[1];

        console.log(`[Send] Sending audio to server: format=${format}, size=${audioBlob.size} bytes, base64 length=${base64Audio.length}`);
        socket.emit('audio_chunk_whisper', { audio: base64Audio, format: format });

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

    try {
        // If SRT is loaded, use SRT-based transcription synced with audio time
        if (state.srtLoaded) {
            console.log('[PlayAndTranscribe] Using SRT transcription (Whisper disabled)');
            await playAudioWithSrt();
        } else {
            console.log('[PlayAndTranscribe] Using Whisper transcription');
            await captureAudioPlaybackRealtime();
        }
    } catch (error) {
        console.error('[PlayAndTranscribe] Error:', error);
        updateStatus('Error processing file: ' + error.message, true);
        state.isProcessingAudio = false;
        playBtn.disabled = false;
        playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    }
}

async function playAudioWithSrt() {
    console.log('[PlayAudioWithSrt] Playing audio with SRT transcription');

    // Play the audio
    audioPlayer.play();
    updateStatus('Playing audio with SRT transcription...');

    // Send time updates to server every 200ms
    state.srtTimeUpdateInterval = setInterval(() => {
        if (!audioPlayer.paused && !audioPlayer.ended) {
            const currentTimeMs = Math.floor(audioPlayer.currentTime * 1000);
            socket.emit('srt_time_update', { time_ms: currentTimeMs });
        }
    }, 200);

    // Clean up when playback ends
    audioPlayer.onended = () => {
        console.log('[PlayAudioWithSrt] Playback ended');
        clearInterval(state.srtTimeUpdateInterval);
        state.srtTimeUpdateInterval = null;
        state.isProcessingAudio = false;
        playBtn.disabled = false;
        playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
        updateStatus('Audio playback finished');
    };

    // Handle pause
    audioPlayer.onpause = () => {
        if (!audioPlayer.ended) {
            console.log('[PlayAudioWithSrt] Playback paused');
            clearInterval(state.srtTimeUpdateInterval);
            state.srtTimeUpdateInterval = null;
        }
    };

    // Handle resume
    audioPlayer.onplay = () => {
        if (!state.srtTimeUpdateInterval) {
            state.srtTimeUpdateInterval = setInterval(() => {
                if (!audioPlayer.paused && !audioPlayer.ended) {
                    const currentTimeMs = Math.floor(audioPlayer.currentTime * 1000);
                    socket.emit('srt_time_update', { time_ms: currentTimeMs });
                }
            }, 200);
        }
    };
}

function stopAudioPlayback() {
    audioPlayer.pause();
    audioPlayer.currentTime = 0;
    if (state.srtTimeUpdateInterval) {
        clearInterval(state.srtTimeUpdateInterval);
        state.srtTimeUpdateInterval = null;
    }
    state.isProcessingAudio = false;
    playBtn.disabled = false;
    playBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    updateStatus('Audio playback stopped');
}

async function captureVideoPlaybackRealtime() {
    console.log('[CaptureVideoPlayback] Setting up real-time audio capture from video player');

    // Create AudioContext and connect video player as source (extracts audio track)
    const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    const source = audioContext.createMediaElementSource(videoPlayer);

    // Create destination for capturing audio
    const destination = audioContext.createMediaStreamDestination();

    // Connect: videoPlayer -> destination (for capture) AND -> audioContext.destination (for playback)
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

    // Start recording from the video player audio output
    mediaRecorder.start();

    // Send chunks every 2 seconds (like microphone recording)
    const recordingInterval = setInterval(() => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            mediaRecorder.start();
        }
    }, 2000);

    // Play the video
    videoPlayer.play();
    updateStatus('Playing video and transcribing audio in real-time...');

    // Clean up when playback ends
    videoPlayer.onended = () => {
        console.log('[CaptureVideoPlayback] Playback ended, cleaning up');

        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
        }
        clearInterval(recordingInterval);

        audioContext.close();

        state.isProcessingAudio = false;
        playVideoBtn.disabled = false;
        playVideoBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
        updateStatus('Video playback and transcription finished');
    };

    // Handle pause/stop
    videoPlayer.onpause = () => {
        if (!videoPlayer.ended) {
            console.log('[CaptureVideoPlayback] Playback paused');
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
            }
            clearInterval(recordingInterval);
        }
    };
}

async function playVideoAndTranscribe() {
    if (!state.currentAudioFile) {
        updateStatus('No video file loaded', true);
        return;
    }

    if (state.isProcessingAudio) {
        updateStatus('Already processing video', true);
        return;
    }

    state.isProcessingAudio = true;
    playVideoBtn.disabled = true;
    playVideoBtn.innerHTML = '<span class="icon">⏳</span> Processing...';

    try {
        // If SRT is loaded, use SRT-based transcription synced with video time
        if (state.srtLoaded) {
            console.log('[PlayVideoAndTranscribe] Using SRT transcription (Whisper disabled)');
            await playVideoWithSrt();
        } else {
            // Capture video audio playback in real-time (like microphone)
            console.log('[PlayVideoAndTranscribe] Using Whisper transcription');
            await captureVideoPlaybackRealtime();
        }
    } catch (error) {
        console.error('[PlayVideoAndTranscribe] Error:', error);
        updateStatus('Error processing video: ' + error.message, true);
        state.isProcessingAudio = false;
        playVideoBtn.disabled = false;
        playVideoBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    }
}

async function playVideoWithSrt() {
    console.log('[PlayVideoWithSrt] Playing video with SRT transcription');

    // Play the video
    videoPlayer.play();
    updateStatus('Playing video with SRT transcription...');

    // Send time updates to server every 200ms
    state.srtTimeUpdateInterval = setInterval(() => {
        if (!videoPlayer.paused && !videoPlayer.ended) {
            const currentTimeMs = Math.floor(videoPlayer.currentTime * 1000);
            socket.emit('srt_time_update', { time_ms: currentTimeMs });
        }
    }, 200);

    // Clean up when playback ends
    videoPlayer.onended = () => {
        console.log('[PlayVideoWithSrt] Playback ended');
        clearInterval(state.srtTimeUpdateInterval);
        state.srtTimeUpdateInterval = null;
        state.isProcessingAudio = false;
        playVideoBtn.disabled = false;
        playVideoBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
        updateStatus('Video playback finished');
    };

    // Handle pause
    videoPlayer.onpause = () => {
        if (!videoPlayer.ended) {
            console.log('[PlayVideoWithSrt] Playback paused');
            clearInterval(state.srtTimeUpdateInterval);
            state.srtTimeUpdateInterval = null;
        }
    };

    // Handle resume
    videoPlayer.onplay = () => {
        if (!state.srtTimeUpdateInterval) {
            state.srtTimeUpdateInterval = setInterval(() => {
                if (!videoPlayer.paused && !videoPlayer.ended) {
                    const currentTimeMs = Math.floor(videoPlayer.currentTime * 1000);
                    socket.emit('srt_time_update', { time_ms: currentTimeMs });
                }
            }, 200);
        }
    };
}

function stopVideoPlayback() {
    videoPlayer.pause();
    videoPlayer.currentTime = 0;
    if (state.srtTimeUpdateInterval) {
        clearInterval(state.srtTimeUpdateInterval);
        state.srtTimeUpdateInterval = null;
    }
    state.isProcessingAudio = false;
    playVideoBtn.disabled = false;
    playVideoBtn.innerHTML = '<span class="icon">▶️</span> Play & Transcribe';
    updateStatus('Video playback stopped');
}

function triggerSearch() {
    if (!transcriptionEl.textContent || transcriptionEl.textContent === 'Transcribed text will appear here...') {
        updateStatus('No transcription available for search', true);
        return;
    }

    // Log client-side timestamp when spacebar is pressed
    const spacebarPressTime = new Date().toISOString();
    console.log(`[TIMING] Spacebar pressed at (client): ${spacebarPressTime}`);

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
            type: state.searchType,
            client_timestamp: spacebarPressTime,
            skip_search: !state.showSearchResults
        });

    } else if (state.searchMode === 'recent') {
        // Recent search: important word from last 5 seconds
        console.log('[Recent Search] Requesting important keyword from last 5 seconds');
        updateStatus('⏱️ Finding important word from last 5 seconds...');

        socket.emit('search_request', {
            mode: 'recent',
            time_threshold: 5,
            type: state.searchType,
            client_timestamp: spacebarPressTime,
            skip_search: !state.showSearchResults
        });

    } else if (state.searchMode === 'gpt') {
        // GPT search: use GPT to predict what user wants to look up
        console.log('[GPT Search] Requesting GPT prediction from last 10 seconds');
        updateStatus('🤖 GPT is predicting keyword...');

        socket.emit('search_request', {
            mode: 'gpt',
            time_threshold: 10,
            type: state.searchType,
            client_timestamp: spacebarPressTime,
            show_all_keywords: state.showAllKeywords,
            skip_search: !state.showSearchResults
        });

    } else {
        // TF-IDF search: let server calculate important word
        console.log('[TF-IDF Search] Requesting important keyword from server');
        updateStatus('🎯 Calculating important keyword...');

        socket.emit('search_request', {
            mode: 'tfidf',
            type: state.searchType,
            client_timestamp: spacebarPressTime,
            skip_search: !state.showSearchResults
        });
    }
}

function triggerNextKeyword() {
    // Request next GPT keyword (for double-spacebar)
    console.log('[Next Keyword] Requesting next keyword');
    updateStatus('🔄 Loading next keyword...');

    socket.emit('next_keyword', {
        type: state.searchType
    });
}

function displayAllKeywords(data) {
    // Use video layout elements if video file, otherwise use default layout
    const keywordEl = state.isVideoFile ? videoSearchKeywordEl : searchKeywordEl;
    const resultsEl = state.isVideoFile ? videoSearchResultsEl : searchResultsEl;
    const sectionEl = state.isVideoFile ? videoSearchSection : searchSection;

    const keywords = data.keywords || [];

    keywordEl.innerHTML = `🤖 GPT Keywords (${keywords.length} total)`;

    resultsEl.innerHTML = '';

    keywords.forEach((item, index) => {
        const keywordDiv = document.createElement('div');
        keywordDiv.className = 'search-result-item';
        keywordDiv.style.cursor = 'pointer';
        keywordDiv.innerHTML = `
            <h4>${index + 1}. ${item.keyword}</h4>
            ${item.description ? `<p>${item.description}</p>` : ''}
        `;

        // Click to search this keyword
        keywordDiv.addEventListener('click', () => {
            console.log(`[All Keywords] Clicking keyword: ${item.keyword}`);
            socket.emit('search_single_keyword', {
                keyword: item.keyword,
                type: state.searchType
            });
        });

        resultsEl.appendChild(keywordDiv);
    });

    sectionEl.style.display = 'block';
    updateStatus(`Found ${keywords.length} GPT keywords. Click one to search.`);
}

function displayKeywordWithDescription(data) {
    // Use video layout elements if video file, otherwise use default layout
    const keywordEl = state.isVideoFile ? videoSearchKeywordEl : searchKeywordEl;
    const sectionEl = state.isVideoFile ? videoSearchSection : searchSection;

    let modeIcon, modeName;
    if (data.mode === 'instant') {
        modeIcon = '⚡';
        modeName = 'Instant';
    } else if (data.mode === 'recent') {
        modeIcon = '⏱️';
        modeName = 'Recent 5s';
    } else if (data.mode === 'gpt') {
        modeIcon = '🤖';
        modeName = 'GPT';
    } else {
        modeIcon = '🎯';
        modeName = 'Important';
    }

    // Build keyword display with index for GPT mode
    let displayText = `${modeIcon} ${data.keyword}`;
    if (data.mode === 'gpt' && data.total_keywords > 1) {
        displayText += ` (${data.current_index + 1}/${data.total_keywords})`;
    } else {
        displayText += ` (${modeName})`;
    }

    keywordEl.innerHTML = displayText;

    // Add description below keyword if available
    if (data.description) {
        keywordEl.innerHTML += `<div class="keyword-description">${data.description}</div>`;
    }

    // Add hint for double-spacebar if more keywords available
    if (data.mode === 'gpt' && data.total_keywords > 1) {
        keywordEl.innerHTML += `<div class="keyword-hint">Press spacebar twice quickly for next keyword</div>`;
    }

    sectionEl.style.display = 'block';
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

    // Use video layout elements if video file, otherwise use default layout
    const resultsEl = state.isVideoFile ? videoSearchResultsEl : searchResultsEl;
    const sectionEl = state.isVideoFile ? videoSearchSection : searchSection;

    // Don't overwrite keyword - it's already set by displayKeywordWithDescription with description
    resultsEl.innerHTML = '';

    // Handle 'both' type - display both text and image results
    if (data.type === 'both') {
        const textResults = data.results.text || [];
        const imageResults = data.results.image || [];

        if (textResults.length === 0 && imageResults.length === 0) {
            resultsEl.innerHTML = '<p>No results found.</p>';
            sectionEl.style.display = 'block';
            return;
        }

        // Display text results section
        if (textResults.length > 0) {
            const textHeader = document.createElement('h3');
            textHeader.textContent = '📝 Text Results';
            textHeader.style.marginTop = '0';
            resultsEl.appendChild(textHeader);

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
                resultsEl.appendChild(resultDiv);
            });
        }

        // Display image results section
        if (imageResults.length > 0) {
            const imageHeader = document.createElement('h3');
            imageHeader.textContent = '🖼️ Image Results';
            imageHeader.style.marginTop = '20px';
            resultsEl.appendChild(imageHeader);

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
                resultsEl.appendChild(resultDiv);
            });
        }

        sectionEl.style.display = 'block';
        updateStatus(`Found ${textResults.length} text + ${imageResults.length} image results for "${data.keyword}"`);
        return;
    }

    // Handle single type (text or image)
    if (data.results.length === 0) {
        resultsEl.innerHTML = '<p>No results found.</p>';
        sectionEl.style.display = 'block';
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
            resultsEl.appendChild(resultDiv);
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
            resultsEl.appendChild(resultDiv);
        });
    }

    sectionEl.style.display = 'block';
    updateStatus(`Found ${data.results.length} results for "${data.keyword}"`);
}

function clearSession() {
    if (state.isRecording) {
        stopRecording();
    }

    // Stop audio/video playback if playing
    if (state.isProcessingAudio) {
        if (state.isVideoFile) {
            stopVideoPlayback();
        } else {
            stopAudioPlayback();
        }
    }

    // Clear SRT interval if running
    if (state.srtTimeUpdateInterval) {
        clearInterval(state.srtTimeUpdateInterval);
        state.srtTimeUpdateInterval = null;
    }

    // Reset audio and video players
    audioPlayer.src = '';
    videoPlayer.src = '';

    // Reset layouts: hide video layout, show default layout
    videoLayout.style.display = 'none';
    defaultLayout.style.display = 'block';

    // Hide audio player section in default layout
    audioPlayerSection.style.display = 'none';

    // Show transcription and info sections in default layout
    document.querySelector('.transcription-section').style.display = 'block';
    document.querySelector('.info-box').style.display = 'block';

    // Hide search sections
    searchSection.style.display = 'none';

    // Reset state
    state.currentAudioFile = null;
    state.isVideoFile = false;
    state.srtLoaded = false;

    // Reset GPT keyword state
    state.hasGptKeywords = false;
    state.totalGptKeywords = 0;
    state.currentGptKeywordIndex = 0;
    state.lastSpacebarTime = 0;

    // Reset SRT button state
    srtInput.parentElement.classList.remove('active');

    socket.emit('clear_session');
}

// Initialize - Auto-load Whisper model
socket.emit('start_whisper');
updateStatus('Loading Whisper model...');
