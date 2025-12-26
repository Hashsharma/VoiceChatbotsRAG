// Voice Assistant WebSocket Client
let recorder;
let isListening = false;
let ws = null;
let audioContext;
let analyser;
let silenceStart;
let mediaStream;
let isConnected = false;

// DOM Elements
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const resetBtn = document.getElementById('resetBtn');
const statusEl = document.getElementById('status');
const connectionStatusEl = document.getElementById('connectionStatus');
const conversationLog = document.getElementById('conversationLog');
const visualizer = document.getElementById('visualizer');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Connect to WebSocket when page loads
    connectWebSocket();
    
    // Set up event listeners
    startBtn.addEventListener('click', startRecording);
    stopBtn.addEventListener('click', stopRecording);
    resetBtn.addEventListener('click', resetConversation);
    
    // Add initial message
    addSystemMessage('Voice Assistant ready. Click "Start Listening" to begin.');
});

// Connect to WebSocket
function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/voice`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    
    ws = new WebSocket(wsUrl);
    ws.binaryType = 'arraybuffer'; // Important for receiving audio
    
    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        isConnected = true;
        connectionStatusEl.textContent = '● Connected';
        connectionStatusEl.className = 'connected';
        addSystemMessage('Connected to voice assistant');
        statusEl.textContent = 'Ready to listen';
    };
    
    ws.onmessage = async (event) => {
        console.log('📨 Received message:', event.data);
        
        try {
            // Try to parse as JSON
            const data = JSON.parse(event.data);
            
            if (data.message) {
                // Handle text messages
                addSystemMessage(data.message);
                statusEl.textContent = data.message;
                
                // If message indicates processing, show status
                if (data.message.includes('Processing') || data.message.includes('Thinking')) {
                    statusEl.textContent = data.message;
                }
            }
            
        } catch (e) {
            // Handle plain text messages
            if (typeof event.data === 'string') {
                addSystemMessage(event.data);
                statusEl.textContent = event.data;
            }
            // Handle binary audio data
            else if (event.data instanceof ArrayBuffer) {
                console.log('🎵 Received audio data:', event.data.byteLength, 'bytes');
                await playAudioResponse(event.data);
            }
        }
    };
    
    ws.onerror = (error) => {
        console.error('❌ WebSocket error:', error);
        isConnected = false;
        connectionStatusEl.textContent = '● Connection error';
        connectionStatusEl.className = 'disconnected';
        addSystemMessage('Connection error');
        statusEl.textContent = 'Connection error';
    };
    
    ws.onclose = () => {
        console.log('🔌 WebSocket disconnected');
        isConnected = false;
        connectionStatusEl.textContent = '● Disconnected';
        connectionStatusEl.className = 'disconnected';
        addSystemMessage('Disconnected from server');
        statusEl.textContent = 'Disconnected';
        
        // Try to reconnect after 5 seconds
        setTimeout(() => {
            if (!isConnected) {
                addSystemMessage('Attempting to reconnect...');
                connectWebSocket();
            }
        }, 5000);
    };
}

// Start recording audio
async function startRecording() {
    if (isListening || !isConnected) {
        if (!isConnected) {
            addSystemMessage('Not connected to server');
            connectWebSocket();
            return;
        }
        return;
    }
    
    try {
        // Request microphone access
        mediaStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                sampleRate: 16000,
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true
            }
        });
        
        // Create audio context
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: 16000
        });
        
        // Create media stream source
        const source = audioContext.createMediaStreamSource(mediaStream);
        
        // Initialize recorder
        recorder = new Recorder(source, {
            numChannels: 1,
            sampleRate: 16000
        });
        
        // Start recording
        recorder.record();
        isListening = true;
        
        // Update UI
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = 'Listening... Speak now';
        addSystemMessage('Started listening...');
        
        // Setup silence detection
        analyser = audioContext.createAnalyser();
        analyser.fftSize = 2048;
        source.connect(analyser);
        
        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);
        silenceStart = performance.now();
        
        // Function to check for silence
        function checkSilence() {
            if (!isListening) return;
            
            analyser.getByteTimeDomainData(dataArray);
            
            // Calculate RMS volume
            let sum = 0;
            for (let i = 0; i < bufferLength; i++) {
                const value = (dataArray[i] - 128) / 128;
                sum += value * value;
            }
            const rms = Math.sqrt(sum / bufferLength);
            
            // Check if silent (adjust threshold as needed)
            if (rms < 0.01) {
                // Check if silence has been long enough (3 seconds)
                if (performance.now() - silenceStart > 3000) {
                    console.log('Silence detected for 3 seconds, stopping recording');
                    stopRecording();
                    return;
                }
            } else {
                // Reset silence timer when sound is detected
                silenceStart = performance.now();
            }
            
            // Continue checking
            requestAnimationFrame(checkSilence);
        }
        
        // Start silence detection
        checkSilence();
        
        // Also process audio chunks in real-time
        const processor = audioContext.createScriptProcessor(4096, 1, 1);
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        processor.onaudioprocess = (e) => {
            if (!isListening || !isConnected || ws.readyState !== WebSocket.OPEN) return;
            
            const audioData = e.inputBuffer.getChannelData(0);
            
            // Convert float32 to int16 for transmission
            const pcmData = new Int16Array(audioData.length);
            for (let i = 0; i < audioData.length; i++) {
                pcmData[i] = Math.max(-32768, Math.min(32767, audioData[i] * 32768));
            }
            
            // Send audio chunk via WebSocket
            ws.send(pcmData.buffer);
        };
        
    } catch (error) {
        console.error('Error accessing microphone:', error);
        addSystemMessage('Error accessing microphone: ' + error.message);
        statusEl.textContent = 'Microphone error';
        isListening = false;
        startBtn.disabled = false;
        stopBtn.disabled = true;
    }
}

// Stop recording and send final audio
function stopRecording() {
    if (!isListening) return;
    
    isListening = false;
    
    // Update UI immediately
    statusEl.textContent = 'Processing...';
    addSystemMessage('Stopped listening, processing...');
    startBtn.disabled = false;
    stopBtn.disabled = true;
    
    // Stop the recorder
    if (recorder) {
        recorder.stop();
        
        // Export the complete recording
        recorder.exportWAV((blob) => {
            console.log('Final audio blob size:', blob.size, 'bytes');
            
            // Send final audio via WebSocket
            if (ws && ws.readyState === WebSocket.OPEN) {
                // Convert blob to array buffer
                blob.arrayBuffer().then(buffer => {
                    console.log('Sending final audio:', buffer.byteLength, 'bytes');
                    ws.send(buffer);
                });
            }
            
            // Clean up
            recorder.clear();
            
            // Stop media stream
            if (mediaStream) {
                mediaStream.getTracks().forEach(track => track.stop());
                mediaStream = null;
            }
            
            // Close audio context
            if (audioContext && audioContext.state !== 'closed') {
                audioContext.close();
                audioContext = null;
            }
        });
    }
}

// Play audio response
async function playAudioResponse(audioBuffer) {
    try {
        // Create blob from audio buffer
        const audioBlob = new Blob([audioBuffer], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(audioBlob);
        
        // Create audio element
        const audio = new Audio(audioUrl);
        
        // Update UI
        addSystemMessage('Playing response...');
        statusEl.textContent = 'Playing response...';
        
        // Play audio
        await audio.play();
        
        // When audio ends
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            addSystemMessage('Response finished');
            statusEl.textContent = 'Ready';
            
            // Auto-restart listening if enabled
            const autoRestart = document.getElementById('autoRestart');
            if (autoRestart && autoRestart.checked) {
                setTimeout(() => {
                    startRecording();
                }, 1000);
            }
        };
        
    } catch (error) {
        console.error('Error playing audio:', error);
        addSystemMessage('Error playing audio response');
        statusEl.textContent = 'Audio playback error';
    }
}

// Add message to conversation log
function addMessage(content, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const timestamp = new Date().toLocaleTimeString([], { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
    });
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-sender">${type === 'user' ? 'You' : type === 'assistant' ? 'Assistant' : 'System'}</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${content}</div>
    `;
    
    conversationLog.appendChild(messageDiv);
    conversationLog.scrollTop = conversationLog.scrollHeight;
}

function addSystemMessage(content) {
    addMessage(content, 'system');
}

function addUserMessage(content) {
    addMessage(content, 'user');
}

function addAssistantMessage(content) {
    addMessage(content, 'assistant');
}

// Reset conversation
function resetConversation() {
    conversationLog.innerHTML = '';
    addSystemMessage('Conversation reset');
    
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({
            type: 'control',
            command: 'reset'
        }));
    }
}

// Clean up when page closes
window.addEventListener('beforeunload', () => {
    if (ws) {
        ws.close();
    }
    
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
    }
});