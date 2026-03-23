// Socket.IO connection
const socket = io();

// State
let sessionId = null;
let isRunning = false;
let selectedAnswer = null;
let currentQuestionIndex = 0;
let sessionStartTime = null;
let nextQuestionCountdownInterval = null;
let questionTimerInterval = null;
let questionTimeRemaining = 0;
let currentStreamingPlayer = null;
let currentStreamId = null;

// Fixed quiz times (matching server)
const QUIZ_TIMES_MINUTES = [0.5, 5, 8];
const QUIZ_TIME_LIMIT_SEC = 60;

// DOM elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusIndicator = document.getElementById('statusIndicator');
const ancIndicator = document.getElementById('ancIndicator');
const quizCountdown = document.getElementById('quizCountdown');
const countdownTime = document.getElementById('countdownTime');
const quizContainer = document.getElementById('quizContainer');
const quizNumber = document.getElementById('quizNumber');
const quizQuestion = document.getElementById('quizQuestion');
const quizOptions = document.getElementById('quizOptions');
const quizTimer = document.getElementById('quizTimer');
const quizTimerTime = document.getElementById('quizTimerTime');
const quizListeningStatus = document.getElementById('quizListeningStatus');
const transcriptList = document.getElementById('transcriptList');
const summaryIndicator = document.getElementById('summaryIndicator');
const summaryText = document.getElementById('summaryText');

// Socket event handlers
socket.on('connect', () => {
    console.log('Connected to server');
    socket.emit('get_devices');
});

socket.on('connected', (data) => {
    sessionId = data.session_id;
    console.log('Session ID:', sessionId);
});

socket.on('devices', (data) => {
    populateDeviceSelect('summaryMic1', data.devices);
    populateDeviceSelect('summaryMic2', data.devices);
});

socket.on('status', (data) => {
    if (data.status === 'started') {
        isRunning = true;
        sessionStartTime = Date.now();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusIndicator.classList.add('running');
        statusIndicator.querySelector('.status-text').textContent = 'Running';

        // Start countdown to first question (2 minutes)
        startNextQuestionCountdown(QUIZ_TIMES_MINUTES[0] * 60);
    } else if (data.status === 'stopped') {
        isRunning = false;
        sessionStartTime = null;
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusIndicator.classList.remove('running');
        statusIndicator.querySelector('.status-text').textContent = 'Stopped';
        stopAllTimers();
        quizContainer.style.display = 'none';
        quizCountdown.style.display = 'none';
        summaryIndicator.style.display = 'none';
        summaryText.textContent = '';
    }
});

socket.on('quiz_question', (data) => {
    currentQuestionIndex = data.index;
    const timeLimit = data.time_limit_sec || QUIZ_TIME_LIMIT_SEC;
    showQuizQuestion(data);
    stopNextQuestionCountdown();
    startQuestionTimer(timeLimit);
});

socket.on('quiz_complete', () => {
    quizContainer.style.display = 'none';
    quizCountdown.style.display = 'none';
    stopAllTimers();
    console.log('All quiz questions completed');

    // Show completion message
    statusIndicator.querySelector('.status-text').textContent = 'Quiz Complete!';
});

socket.on('listening_start', (data) => {
    if (quizListeningStatus) {
        quizListeningStatus.textContent = '🎙️ Listening...';
        quizListeningStatus.classList.add('listening');
    }
    setAncIndicator(true);
});

socket.on('listening_end', (data) => {
    if (quizListeningStatus) {
        quizListeningStatus.textContent = '⏸️ Paused';
        quizListeningStatus.classList.remove('listening');
    }
});

socket.on('listening_status', (data) => {
    if (data.listening) {
        if (quizListeningStatus) {
            quizListeningStatus.textContent = '🎙️ Listening...';
            quizListeningStatus.classList.add('listening');
        }
        setAncIndicator(true);
    } else {
        if (quizListeningStatus) {
            quizListeningStatus.textContent = '⏸️ Paused';
            quizListeningStatus.classList.remove('listening');
        }
        setAncIndicator(false);
    }
});

socket.on('vad_transcript', (data) => {
    addTranscript(data);
});

socket.on('summary_closed', () => {
    console.log('Summary client closed');
});

socket.on('summary_error', (data) => {
    console.error('Summary error:', data.error);
});

socket.on('summary_text', (data) => {
    if (!summaryIndicator || !summaryText) return;
    summaryText.textContent = data.text || '';
    summaryIndicator.style.display = data.text ? 'flex' : 'none';
});

socket.on('streaming_tts_chunk', (data) => {
    if (!data || !data.audio) return;
    const streamId = String(data.stream_id || '').trim();
    if (!streamId) return;

    const audioData = atob(data.audio);
    const chunk = new Uint8Array(audioData.length);
    for (let i = 0; i < audioData.length; i++) {
        chunk[i] = audioData.charCodeAt(i);
    }

    if (!currentStreamingPlayer || currentStreamId !== streamId) {
        if (currentStreamingPlayer) {
            currentStreamingPlayer.stop();
        }
        currentStreamId = streamId;
        currentStreamingPlayer = new StreamingTtsPlayer(streamId, Number(data.sample_rate || 24000));
        currentStreamingPlayer.onEndCallback = () => {
            currentStreamingPlayer = null;
            currentStreamId = null;
        };
    }

    currentStreamingPlayer.pushChunk(chunk);
});

socket.on('streaming_tts_done', (data) => {
    const streamId = String((data && data.stream_id) || '').trim();
    if (!streamId || !currentStreamingPlayer || currentStreamId !== streamId) return;
    currentStreamingPlayer.finish();
});

// Functions
function populateDeviceSelect(selectId, devices) {
    const select = document.getElementById(selectId);
    select.innerHTML = '<option value="">Select device...</option>';

    devices.forEach(device => {
        const option = document.createElement('option');
        option.value = device.index;
        option.textContent = `${device.name} (${device.channels}ch)`;
        select.appendChild(option);
    });
}

function start() {
    const summaryMic1 = document.getElementById('summaryMic1').value;
    const summaryMic2 = document.getElementById('summaryMic2').value;
    const quizFile = document.getElementById('quizFile').value;
    const ttsVoiceId = document.getElementById('ttsVoiceId').value.trim();
    const noiseThreshold = parseInt(document.getElementById('noiseThreshold').value) || 0;

    const summaryMics = [];
    if (summaryMic1) summaryMics.push(parseInt(summaryMic1));
    if (summaryMic2) summaryMics.push(parseInt(summaryMic2));

    if (summaryMics.length === 0) {
        alert('Please select at least one microphone');
        return;
    }

    // Clear transcripts
    transcriptList.innerHTML = '';
    summaryIndicator.style.display = 'none';
    summaryText.textContent = '';

    socket.emit('start', {
        session_id: `quiz_${Date.now()}`,
        source: 'mic',
        summary_mics: summaryMics,
        quiz_file: quizFile,
        tts_voice_id: ttsVoiceId || null,
        noise_threshold: noiseThreshold
    });
}

function stop() {
    if (currentStreamingPlayer) {
        currentStreamingPlayer.stop();
        currentStreamingPlayer = null;
        currentStreamId = null;
    }
    socket.emit('stop');
}

function stopAllTimers() {
    stopNextQuestionCountdown();
    stopQuestionTimer();
}

// Countdown to next question
function startNextQuestionCountdown(seconds) {
    quizCountdown.style.display = 'block';
    let remaining = seconds;

    updateNextQuestionCountdownDisplay(remaining);

    nextQuestionCountdownInterval = setInterval(() => {
        remaining--;
        if (remaining <= 0) {
            stopNextQuestionCountdown();
        } else {
            updateNextQuestionCountdownDisplay(remaining);
        }
    }, 1000);
}

function stopNextQuestionCountdown() {
    if (nextQuestionCountdownInterval) {
        clearInterval(nextQuestionCountdownInterval);
        nextQuestionCountdownInterval = null;
    }
    quizCountdown.style.display = 'none';
}

function updateNextQuestionCountdownDisplay(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    countdownTime.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}

// Question timer (1 minute limit)
function startQuestionTimer(seconds) {
    questionTimeRemaining = seconds;
    updateQuestionTimerDisplay(questionTimeRemaining);

    if (quizTimer) {
        quizTimer.style.display = 'block';
        quizTimer.classList.remove('warning');
    }

    questionTimerInterval = setInterval(() => {
        questionTimeRemaining--;
        updateQuestionTimerDisplay(questionTimeRemaining);

        // Warning when 10 seconds left
        if (questionTimeRemaining <= 10 && quizTimer) {
            quizTimer.classList.add('warning');
        }

        if (questionTimeRemaining <= 0) {
            // Time's up - auto submit
            autoSubmitAnswer();
        }
    }, 1000);
}

function stopQuestionTimer() {
    if (questionTimerInterval) {
        clearInterval(questionTimerInterval);
        questionTimerInterval = null;
    }
    if (quizTimer) {
        quizTimer.style.display = 'none';
        quizTimer.classList.remove('warning');
    }
}

function updateQuestionTimerDisplay(seconds) {
    if (!quizTimerTime) return;
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    quizTimerTime.textContent = `${mins}:${secs.toString().padStart(2, '0')}`;
}

function showQuizQuestion(data) {
    quizContainer.style.display = 'block';

    const quizTimeMin = data.quiz_time_minutes || QUIZ_TIMES_MINUTES[data.index] || 0;
    quizNumber.textContent = `Question ${data.index + 1}/${data.total} (at ${quizTimeMin} min)`;

    const question = data.question;
    quizQuestion.textContent = question.text || question.question || question;

    // Render options if available
    quizOptions.innerHTML = '';
    selectedAnswer = null;

    const options = question.options || question.choices || [];
    if (options.length > 0) {
        options.forEach((option, i) => {
            const optionDiv = document.createElement('div');
            optionDiv.className = 'quiz-option';
            optionDiv.onclick = () => selectOption(i, optionDiv);

            const marker = String.fromCharCode(65 + i); // A, B, C, D...
            optionDiv.innerHTML = `
                <div class="quiz-option-label">
                    <span class="quiz-option-marker">${marker}</span>
                    <span class="quiz-option-text">${option.text || option}</span>
                </div>
            `;

            quizOptions.appendChild(optionDiv);
        });
    }

    // Update listening status
    if (quizListeningStatus) {
        quizListeningStatus.textContent = '🎙️ Listening...';
        quizListeningStatus.classList.add('listening');
    }
    setAncIndicator(true);
}

function selectOption(index, element) {
    // Remove selection from all options
    document.querySelectorAll('.quiz-option').forEach(opt => {
        opt.classList.remove('selected');
    });

    // Select this option
    element.classList.add('selected');
    selectedAnswer = index;
}

function autoSubmitAnswer() {
    stopQuestionTimer();

    socket.emit('quiz_answer', {
        question_index: currentQuestionIndex,
        answer: selectedAnswer,
        timeout: true
    });

    // Hide question
    quizContainer.style.display = 'none';
    setAncIndicator(false);

    // Calculate time until next question
    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < QUIZ_TIMES_MINUTES.length && sessionStartTime) {
        const elapsedSec = (Date.now() - sessionStartTime) / 1000;
        const nextQuestionTimeSec = QUIZ_TIMES_MINUTES[nextQuestionIndex] * 60;
        const remainingSec = Math.max(0, nextQuestionTimeSec - elapsedSec);

        if (remainingSec > 0) {
            startNextQuestionCountdown(Math.ceil(remainingSec));
        }
    }
}

function submitAnswer() {
    stopQuestionTimer();

    socket.emit('quiz_answer', {
        question_index: currentQuestionIndex,
        answer: selectedAnswer,
        timeout: false
    });

    quizContainer.style.display = 'none';
    setAncIndicator(false);

    const nextQuestionIndex = currentQuestionIndex + 1;
    if (nextQuestionIndex < QUIZ_TIMES_MINUTES.length && sessionStartTime) {
        const elapsedSec = (Date.now() - sessionStartTime) / 1000;
        const nextQuestionTimeSec = QUIZ_TIMES_MINUTES[nextQuestionIndex] * 60;
        const remainingSec = Math.max(0, nextQuestionTimeSec - elapsedSec);

        if (remainingSec > 0) {
            startNextQuestionCountdown(Math.ceil(remainingSec));
        }
    }
}

function addTranscript(data) {
    const entry = document.createElement('div');
    entry.className = `transcript-entry speaker-${data.speaker}`;

    const time = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <div class="transcript-speaker">Speaker ${data.speaker}</div>
        <div class="transcript-text">${data.text}</div>
        <div class="transcript-time">${time}</div>
    `;

    transcriptList.insertBefore(entry, transcriptList.firstChild);

    // Keep only last 50 entries
    while (transcriptList.children.length > 50) {
        transcriptList.removeChild(transcriptList.lastChild);
    }
}

function setAncIndicator(isAnc) {
    if (!ancIndicator) return;
    if (isAnc) {
        ancIndicator.classList.add('anc-on');
        ancIndicator.querySelector('.anc-text').textContent = 'ANC On';
    } else {
        ancIndicator.classList.remove('anc-on');
        ancIndicator.querySelector('.anc-text').textContent = 'Transparency';
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    socket.emit('get_devices');
});

class StreamingTtsPlayer {
    constructor(streamId, sampleRate) {
        this.streamId = streamId;
        this.sampleRate = sampleRate;
        this.audioContext = null;
        this.nextPlayTime = 0;
        this.isStopped = false;
        this.onEndCallback = null;
    }

    async start() {
        if (this.audioContext) return;
        const Ctx = window.AudioContext || window.webkitAudioContext;
        if (!Ctx) return;
        this.audioContext = new Ctx({ sampleRate: this.sampleRate });
        if (this.audioContext.state === 'suspended') {
            await this.audioContext.resume();
        }
        this.nextPlayTime = this.audioContext.currentTime + 0.05;
    }

    async pushChunk(pcmBytes) {
        if (this.isStopped) return;
        if (!this.audioContext) {
            await this.start();
        }
        if (!this.audioContext || this.audioContext.state === 'closed') return;

        const samples = pcmBytes.length / 2;
        const float32 = new Float32Array(samples);
        const view = new DataView(pcmBytes.buffer, pcmBytes.byteOffset, pcmBytes.byteLength);
        for (let i = 0; i < samples; i++) {
            float32[i] = view.getInt16(i * 2, true) / 32768.0;
        }

        const buffer = this.audioContext.createBuffer(1, samples, this.sampleRate);
        buffer.getChannelData(0).set(float32);

        const source = this.audioContext.createBufferSource();
        source.buffer = buffer;
        source.connect(this.audioContext.destination);

        const currentTime = this.audioContext.currentTime;
        if (this.nextPlayTime < currentTime) {
            this.nextPlayTime = currentTime + 0.02;
        }
        source.start(this.nextPlayTime);
        this.nextPlayTime += samples / this.sampleRate;
    }

    async finish() {
        if (this.isStopped) return;
        if (this.audioContext && this.nextPlayTime > this.audioContext.currentTime) {
            const remainingMs = (this.nextPlayTime - this.audioContext.currentTime) * 1000;
            await new Promise(resolve => setTimeout(resolve, remainingMs + 50));
        }
        this.stop(false);
        if (this.onEndCallback) {
            this.onEndCallback();
        }
    }

    stop(runCallback = true) {
        this.isStopped = true;
        if (this.audioContext && this.audioContext.state !== 'closed') {
            this.audioContext.close().catch(() => {});
        }
        this.audioContext = null;
        if (runCallback && this.onEndCallback) {
            this.onEndCallback();
        }
    }
}
