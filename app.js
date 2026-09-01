// Import transformers.js from jsDelivr CDN
import { pipeline, env, TextStreamer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// Configuration
// We disable local models to force fetching from the Hugging Face Hub.
// Transformers.js automatically caches downloaded models using the Cache API.
env.allowLocalModels = false;

// State
let chatPipeline = null;
let whisperPipeline = null;
let currentGeneration = null;
let abortController = null;
let isGenerating = false;
let mediaRecorder = null;
let audioChunks = [];
let conversationHistory = [
    { role: 'system', content: 'You are a helpful, smart, and concise assistant.' }
];

// DOM Elements
const gpuStatus = document.getElementById('gpu-status');
const progressContainer = document.getElementById('progress-container');
const progressText = document.getElementById('progress-text');
const progressBarFg = document.getElementById('progress-bar-fg');
const chatContainer = document.getElementById('chat-container');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const micBtn = document.getElementById('mic-btn');
const clearBtn = document.getElementById('clear-btn');

/**
 * Checks if WebGPU is supported by the current browser.
 */
async function checkWebGPU() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch (e) {
        return false;
    }
}

/**
 * Initialize the application and load the models.
 * WebGPU is strongly preferred for the 1.5B model.
 */
async function init() {
    const hasWebGPU = await checkWebGPU();
    // Use WebGPU if available, otherwise fallback to WASM (CPU).
    const device = hasWebGPU ? 'webgpu' : 'wasm';
    
    gpuStatus.textContent = `Backend: ${device.toUpperCase()}`;
    gpuStatus.style.color = hasWebGPU ? '#4ade80' : '#facc15';

    progressContainer.classList.remove('hidden');
    
    const progressMap = new Map();
    
    // Unified progress callback for downloading model weights
    const progressCallback = (info) => {
        if (info.status === 'progress') {
            progressMap.set(info.file, { loaded: info.loaded, total: info.total });
            
            let totalLoaded = 0;
            let totalSize = 0;
            for (const progress of progressMap.values()) {
                totalLoaded += progress.loaded;
                totalSize += progress.total;
            }
            
            if (totalSize > 0) {
                const percent = (totalLoaded / totalSize) * 100;
                progressBarFg.style.width = `${percent}%`;
                progressText.textContent = `Downloading models... ${Math.round(percent)}%`;
            }
        }
    };

    try {
        progressText.textContent = "Loading Whisper (Speech-to-Text)...";
        // Initialize Whisper Pipeline
        whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-base', {
            device: device,
            progress_callback: progressCallback
        });

        progressText.textContent = "Loading DeepSeek-R1-Distill-Qwen-1.5B (Chat)...";
        // Initialize DeepSeek Chat Pipeline
        chatPipeline = await pipeline('text-generation', 'onnx-community/DeepSeek-R1-Distill-Qwen-1.5B-ONNX', {
            device: device,
            dtype: 'q4f16', // Use 4-bit quantization with fp16 to fit in memory
            progress_callback: progressCallback
        });

        // Enable UI inputs once models are fully loaded
        progressContainer.classList.add('hidden');
        chatInput.disabled = false;
        micBtn.disabled = false;
        sendBtn.disabled = false;
        chatInput.placeholder = "Type a message or use mic...";
        chatInput.focus();

    } catch (error) {
        console.error("Model loading failed:", error);
        progressText.textContent = "Error loading models. Check console.";
        progressText.style.color = "#ef4444";
    }
}

/**
 * Appends a message to the chat container.
 */
function addMessage(text, isAi = false) {
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', isAi ? 'ai' : 'user');
    
    const contentDiv = document.createElement('div');
    contentDiv.classList.add('message-content');
    contentDiv.textContent = text;
    
    msgDiv.appendChild(contentDiv);
    chatContainer.appendChild(msgDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return contentDiv;
}

/**
 * Handles sending a message to the model.
 */
async function handleSend() {
    const text = chatInput.value.trim();
    if (!text || isGenerating || !chatPipeline) return;

    chatInput.value = '';
    adjustTextareaHeight();
    
    addMessage(text, false);
    conversationHistory.push({ role: 'user', content: text });

    const aiContentDiv = addMessage("", true);
    
    isGenerating = true;
    sendBtn.classList.add('hidden');
    stopBtn.classList.remove('hidden');
    
    abortController = new AbortController();

    try {
        // TextStreamer for token-by-token streaming output
        const streamer = new TextStreamer(chatPipeline.tokenizer, {
            skip_prompt: true,
            skip_special_tokens: true,
            callback_function: (tokenText) => {
                aiContentDiv.textContent += tokenText;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        });

        // Run inference
        currentGeneration = chatPipeline(conversationHistory, {
            max_new_tokens: 512,
            temperature: 0.7,
            do_sample: true,
            streamer: streamer,
            // AbortSignal to stop generation early
            signal: abortController.signal 
        });

        const result = await currentGeneration;
        
        // Extract the final assistant response string
        let generatedText = result[0].generated_text;
        if (Array.isArray(generatedText)) {
            const lastMsg = generatedText[generatedText.length - 1];
            if (lastMsg && lastMsg.role === 'assistant') {
                generatedText = lastMsg.content;
            }
        }
        
        conversationHistory.push({ role: 'assistant', content: generatedText });
        
    } catch (error) {
        if (error.name === 'AbortError') {
            aiContentDiv.textContent += "\n[Generation stopped]";
            // Update history with partial response
            conversationHistory.push({ role: 'assistant', content: aiContentDiv.textContent });
        } else {
            console.error("Generation error:", error);
            aiContentDiv.textContent += "\n[Error generating response]";
        }
    } finally {
        isGenerating = false;
        currentGeneration = null;
        sendBtn.classList.remove('hidden');
        stopBtn.classList.add('hidden');
    }
}

/**
 * Handle audio recording and transcription via Whisper.
 */
async function toggleMic() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        micBtn.classList.remove('recording');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];

        mediaRecorder.ondataavailable = (event) => audioChunks.push(event.data);

        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
            stream.getTracks().forEach(track => track.stop());
            await transcribeAudio(audioBlob);
        };

        mediaRecorder.start();
        micBtn.classList.add('recording');
    } catch (err) {
        console.error("Mic access error:", err);
        alert("Microphone access is required for speech-to-text.");
    }
}

/**
 * Process the audio blob and run Whisper model.
 */
async function transcribeAudio(blob) {
    if (!whisperPipeline) return;
    
    chatInput.placeholder = "Transcribing audio...";
    chatInput.disabled = true;

    try {
        // Decode audio data to extract floating point representations
        const arrayBuffer = await blob.arrayBuffer();
        const audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        const float32Array = audioBuffer.getChannelData(0);

        // Run STT inference
        const result = await whisperPipeline(float32Array);
        
        // Append transcribed text to the input field
        const transcript = result.text.trim();
        chatInput.value += (chatInput.value ? " " : "") + transcript;
        adjustTextareaHeight();
        
    } catch (error) {
        console.error("Transcription error:", error);
        alert("Failed to transcribe audio.");
    } finally {
        chatInput.placeholder = "Type a message or use mic...";
        chatInput.disabled = false;
        chatInput.focus();
    }
}

// Event Listeners
sendBtn.addEventListener('click', handleSend);
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSend();
    }
});
stopBtn.addEventListener('click', () => {
    if (abortController) {
        abortController.abort();
    }
});
micBtn.addEventListener('click', toggleMic);
clearBtn.addEventListener('click', () => {
    // Keep only the system prompt in history
    conversationHistory = [conversationHistory[0]];
    
    // Clear UI but leave the initial greeting
    chatContainer.innerHTML = `
        <div class="message ai">
            <div class="message-content">Chat history cleared. How can I help you?</div>
        </div>
    `;
});

function adjustTextareaHeight() {
    chatInput.style.height = 'auto';
    chatInput.style.height = chatInput.scrollHeight + 'px';
}
chatInput.addEventListener('input', adjustTextareaHeight);

// Start app
init();
