import { AudioClassifierResult } from '@mediapipe/tasks-audio';
import template from '../templates/audio-classifier.html?raw';

let worker: Worker | undefined;
let isWorkerReady = false;
let audioContext: AudioContext | undefined;
let scriptProcessor: ScriptProcessorNode | undefined;
let mediaStreamSource: MediaStreamAudioSourceNode | undefined;
let stream: MediaStream | undefined;
let isRecording = false;

// Visualization
let canvasElement: HTMLCanvasElement;
let canvasCtx: CanvasRenderingContext2D;

// Options
let maxResults = 3;
let scoreThreshold = 0.02;
let currentDelegate: 'CPU' | 'GPU' = 'GPU'; // Default to GPU as requested
let runningMode: 'AUDIO_STREAM' | 'AUDIO_CLIPS' = 'AUDIO_STREAM';
// Visualization State
const WAVEFORM_HISTORY_SIZE = 8000;
let waveformBuffer = new Float32Array(WAVEFORM_HISTORY_SIZE);
let waveformSnapshots: Float32Array[] = [];
const MAX_SNAPSHOTS = 5;

// @ts-ignore
import AudioClassifierWorker from '../workers/audio-classifier.worker.ts?worker';

export async function setupAudioClassifier(container: HTMLElement) {
  container.innerHTML = template;

  // Update UI to match default
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  if (delegateSelect) delegateSelect.value = 'GPU';

  canvasElement = document.getElementById('waveform-canvas') as HTMLCanvasElement;
  canvasCtx = canvasElement.getContext('2d')!;

  // Initialize worker
  initWorker();

  // HTML Elements


  // Setup UI
  setupUI();

  // Initialize classifier
  await initializeClassifier();

  // Cleanup function will be returned or handled by router via export
}

function initWorker() {
  if (!worker) {
    worker = new AudioClassifierWorker();
    worker.onmessage = handleWorkerMessage;
    worker.onerror = (e) => {
      const msg = e instanceof ErrorEvent ? e.message : 'Unknown Worker Error';
      console.error(`Worker script error: ${msg}`, e);
      updateStatus(`Worker Error: ${msg}`);
    };
  }
}

async function initializeClassifier() {
  const statusMessage = document.getElementById('status-message');
  if (statusMessage) statusMessage.innerText = 'Initializing...';

  // Timeout to prevent infinite loading
  // Timeout to prevent infinite loading
  const timeoutId = setTimeout(() => {
    if (!isWorkerReady) {
      document.querySelector('.viewport')?.classList.remove('loading-model');
      updateStatus('Error: Initialization timed out. Check console.');
      console.error('Audio Classifier initialization timed out after 30s.');
    }
  }, 30000);

  document.querySelector('.viewport')?.classList.add('loading-model');
  isWorkerReady = false;

  // @ts-ignore
  const baseUrl = import.meta.env.BASE_URL;

  // We use the same model asset path for now (Yamnet)
  // URL: https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
  const modelPath = 'https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite';

  try {
    console.log(`[Main] Sending INIT to worker with model: ${modelPath} and wasmBase: ${baseUrl}`);
    worker?.postMessage({
      type: 'INIT',
      modelAssetPath: modelPath,
      delegate: currentDelegate,
      maxResults,
      scoreThreshold,
      runningMode: runningMode === 'AUDIO_STREAM' ? 'AUDIO_STREAM' : 'AUDIO_CLIPS',
      baseUrl
    });
    console.log('[Main] INIT message sent');
  } catch (e) {
    clearTimeout(timeoutId);
    console.error('Failed to post message to worker:', e);
    document.querySelector('.viewport')?.classList.remove('loading-model');
  }
}

function handleWorkerMessage(event: MessageEvent) {
  const { type } = event.data;

  switch (type) {
    case 'INIT_DONE':
      document.querySelector('.viewport')?.classList.remove('loading-model');
      isWorkerReady = true;
      updateStatus('Ready');
      const recBtn = document.getElementById('recordButton') as HTMLButtonElement;
      if (recBtn) recBtn.disabled = false;
      break;

    case 'CLASSIFY_RESULT':
      const { results, inferenceTime } = event.data;
      updateStatus(`Done in ${Math.round(inferenceTime)}ms`);
      updateInferenceTime(inferenceTime);
      displayClassificationResults(results);
      break;

    case 'ERROR':
    case 'CLASSIFY_ERROR':
      console.error('Worker error:', event.data.error);
      updateStatus(`Error: ${event.data.error}`);
      break;
  }
}

function setupUI() {
  // Tab switching
  const tabMic = document.getElementById('tab-microphone')!;
  const tabFile = document.getElementById('tab-file')!;
  const viewMic = document.getElementById('view-microphone')!;
  const viewFile = document.getElementById('view-file')!;

  const switchView = (mode: 'MIC' | 'FILE') => {
    if (mode === 'MIC') {
      tabMic.classList.add('active');
      tabFile.classList.remove('active');
      viewMic.classList.add('active');
      viewFile.classList.remove('active');
      runningMode = 'AUDIO_STREAM';
    } else {
      tabMic.classList.remove('active');
      tabFile.classList.add('active');
      viewMic.classList.remove('active');
      viewFile.classList.add('active');
      runningMode = 'AUDIO_CLIPS';
      stopRecording();
    }
    // Re-init? Or assume worker handles mode switch via SET_OPTIONS if needed?
    // AudioClassifier usually just takes data.
    // But options might have runningMode.
  };

  tabMic.addEventListener('click', () => switchView('MIC'));
  tabFile.addEventListener('click', () => switchView('FILE'));

  // Delegate
  const delegateSelect = document.getElementById('delegate-select') as HTMLSelectElement;
  if (delegateSelect) {
    delegateSelect.addEventListener('change', () => {
      currentDelegate = delegateSelect.value as 'CPU' | 'GPU';
      initializeClassifier(); // Re-init needed for delegate change
    });
  }

  // Record Button
  const recordButton = document.getElementById('recordButton') as HTMLButtonElement;
  recordButton.addEventListener('click', toggleRecording);

  // Audio File Upload
  const audioUpload = document.getElementById('audio-upload') as HTMLInputElement;
  const dropzone = document.querySelector('.upload-dropzone') as HTMLElement;

  if (dropzone) {
    dropzone.addEventListener('click', (e) => {
      // Only click if not clicking the player or button
      if ((e.target as HTMLElement).closest('audio') || (e.target as HTMLElement).closest('button')) return;
      audioUpload.click();
    });
  }

  audioUpload.addEventListener('change', handleFileUpload);

  // Sliders
  const maxResultsInput = document.getElementById('max-results') as HTMLInputElement;
  const maxResultsValue = document.getElementById('max-results-value')!;
  maxResultsInput.addEventListener('input', () => {
    maxResults = parseInt(maxResultsInput.value);
    maxResultsValue.innerText = maxResults.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', maxResults });
  });

  const scoreThresholdInput = document.getElementById('score-threshold') as HTMLInputElement;
  const scoreThresholdValue = document.getElementById('score-threshold-value')!;
  scoreThresholdInput.addEventListener('input', () => {
    scoreThreshold = parseFloat(scoreThresholdInput.value);
    scoreThresholdValue.innerText = scoreThreshold.toString();
    worker?.postMessage({ type: 'SET_OPTIONS', scoreThreshold });
  });
}

async function toggleRecording() {
  const recordButton = document.getElementById('recordButton') as HTMLButtonElement;
  if (isRecording) {
    stopRecording();
    recordButton.innerHTML = '<span class="material-icons">mic</span> Start Recording';
    recordButton.classList.remove('recording'); // Optional style
  } else {
    await startRecording();
    if (isRecording) {
      recordButton.innerHTML = '<span class="material-icons">stop</span> Stop Recording';
      recordButton.classList.add('recording');
    }
  }
}

async function startRecording() {
  try {
    // Enforce 16kHz for Yamnet
    audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });

    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaStreamSource = audioContext.createMediaStreamSource(stream);

    // Use ScriptProcessor for simplicity in getting raw data
    // Buffer size 4096 is common
    scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

    mediaStreamSource.connect(scriptProcessor);
    scriptProcessor.connect(audioContext.destination); // Mute loopback? Warning: feedback loop if speaker is near mic.
    // Actually, we usually shouldn't connect to destination if we don't want to hear ourselves.
    // But ScriptProcessor sometimes needs downstream connection to fire?
    // In Chrome, it needs to be connected to destination OR we can use AudioWorklet.
    // Safest is to connect to a GainNode with gain 0, then to destination.
    // But for now let's try not connecting to destination and see if onaudioprocess fires (it might not).
    // Standard says it must be connected.
    // Let's connect to destination but we should rely on system echo cancellation or browser handling if we mute it?
    // Actually, let's create a silent gain node.
    const gainNode = audioContext.createGain();
    gainNode.gain.value = 0;
    scriptProcessor.connect(gainNode);
    gainNode.connect(audioContext.destination);

    scriptProcessor.onaudioprocess = (e) => {
      const inputData = e.inputBuffer.getChannelData(0);
      // Visualization always runs
      visualizeWaveform(inputData);

      // Send to worker
      if (isWorkerReady) {
        worker?.postMessage({
          type: 'CLASSIFY',
          audioData: inputData, // Copied automatically? Or transferred?
          sampleRate: audioContext!.sampleRate,
          timestampMs: performance.now()
        });
      }
    };

    isRecording = true;
    updateStatus('Recording...');
  } catch (err) {
    console.error('Failed to start recording', err);
    updateStatus('Mic Error');
  }
}

function stopRecording() {
  if (scriptProcessor) {
    scriptProcessor.disconnect();
    scriptProcessor.onaudioprocess = null;
    scriptProcessor = undefined;
  }
  if (mediaStreamSource) {
    mediaStreamSource.disconnect();
    mediaStreamSource = undefined;
  }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = undefined;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = undefined;
  }
  isRecording = false;
  updateStatus('Ready');
}

function handleFileUpload(e: Event) {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  const player = document.getElementById('audio-player') as HTMLAudioElement;
  const previewContainer = document.getElementById('audio-preview-container')!;
  const dropzoneContent = document.querySelector('.dropzone-content') as HTMLElement;

  player.src = URL.createObjectURL(file);
  previewContainer.style.display = 'flex';
  dropzoneContent.style.display = 'none';
  // Set up classification button
  const runBtn = document.getElementById('run-file-classification') as HTMLButtonElement;
  runBtn.onclick = async () => {
    updateStatus('Processing file...');
    runBtn.disabled = true;
    try {
      await processAudioFile(file);
    } catch (err) {
      console.error(err);
      updateStatus('File Error');
    } finally {
      runBtn.disabled = false;
    }
  };
}

async function processAudioFile(file: File) {
  const arrayBuffer = await file.arrayBuffer();
  const tempCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
  const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);

  const inputData = audioBuffer.getChannelData(0);

  // Visualize whole file? Or just a chunk?
  visualizeWaveform(inputData.slice(0, 4096)); // Just a snippet for vis

  if (isWorkerReady) {
    worker?.postMessage({
      type: 'CLASSIFY',
      audioData: inputData,
      sampleRate: tempCtx.sampleRate,
      timestampMs: 0
    });
  }

  await tempCtx.close();
}

// Waveform state moved up
// const WAVEFORM_HISTORY_SIZE = 8000; // Keep it smaller for now
// let waveformBuffer = new Float32Array(WAVEFORM_HISTORY_SIZE);

function updateWaveformBuffer(newData: Float32Array) {
  // Copy logic
  const validData = newData.length > 0 ? newData : new Float32Array(newData.length); // Handle silence?

  // Simple shift
  // Create new buffer to avoid overlap issues if any
  const newBuffer = new Float32Array(WAVEFORM_HISTORY_SIZE);
  // Copy end of old buffer to start of new
  newBuffer.set(waveformBuffer.subarray(validData.length), 0);
  // Copy new data to end
  newBuffer.set(validData, WAVEFORM_HISTORY_SIZE - validData.length);

  waveformBuffer = newBuffer;
}

function visualizeWaveform(newData: Float32Array) {
  if (!canvasCtx || !canvasElement) return;

  // Ensure canvas dimensions match display
  if (canvasElement.width !== canvasElement.clientWidth) {
    canvasElement.width = canvasElement.clientWidth;
    canvasElement.height = canvasElement.clientHeight;
  }

  const width = canvasElement.width;
  const height = canvasElement.height;

  // Update buffer
  updateWaveformBuffer(newData);

  // Update snapshots for ghosting effect
  // We only push a snapshot every N frames or just every frame? Every frame is smoothest.
  waveformSnapshots.push(new Float32Array(waveformBuffer));
  if (waveformSnapshots.length > MAX_SNAPSHOTS) {
    waveformSnapshots.shift();
  }

  // Clear with light background
  canvasCtx.fillStyle = '#e0f7fa'; // Light Cyan
  canvasCtx.fillRect(0, 0, width, height);

  canvasCtx.lineWidth = 2;

  // Draw snapshots (ghosts)
  waveformSnapshots.forEach((snapshot, index) => {
    const isLast = index === waveformSnapshots.length - 1;
    const alpha = (index + 1) / (waveformSnapshots.length + 1);

    // Main line is dark teal, ghosts are lighter/faded
    // Reference: Teal/Cyan lines
    if (isLast) {
      canvasCtx.strokeStyle = `rgba(0, 96, 100, 1.0)`; // Cyan 900
      canvasCtx.lineWidth = 2.5;
    } else {
      canvasCtx.strokeStyle = `rgba(0, 151, 167, ${alpha * 0.5})`; // Cyan 700 with alpha
      canvasCtx.lineWidth = 2;
    }

    canvasCtx.beginPath();
    const sliceWidth = width / snapshot.length;
    let x = 0;

    // Decimation for performance if needed, but 8000 pts is fine for canvas usually
    const step = Math.ceil(snapshot.length / width);

    for (let i = 0; i < snapshot.length; i += step) {
      const v = snapshot[i];
      // Normalize? v is usually -1 to 1
      // Add some gain for visibility if needed
      const y = (v * height / 1.5) + (height / 2);

      if (i === 0) {
        canvasCtx.moveTo(x, y);
      } else {
        canvasCtx.lineTo(x, y);
      }
      x += sliceWidth * step;
    }
    canvasCtx.stroke();
  });
}

function displayClassificationResults(results: AudioClassifierResult[]) {
  if (!results || results.length === 0) return;

  const classifications = results[0].classifications[0].categories;
  // Sort by score
  classifications.sort((a, b) => b.score - a.score);

  const container = document.getElementById('classification-results');
  if (!container) return;

  container.innerHTML = '';

  const topResults = classifications.slice(0, maxResults);

  // Colors from reference (Blue, Orange, etc.)
  const barColors = ['#4285f4', '#fb8c00', '#4285f4', '#fb8c00', '#4285f4'];

  topResults.forEach((c, index) => {
    const row = document.createElement('div');
    row.className = 'result-row';
    row.style.display = 'flex';
    row.style.alignItems = 'center';
    row.style.marginBottom = '8px';
    row.style.fontFamily = "'Google Sans', sans-serif";

    const label = document.createElement('span');
    label.className = 'result-label';
    label.innerText = c.categoryName;
    label.style.width = '120px'; // Fixed width
    label.style.color = '#5f6368';
    label.style.fontWeight = '500';
    label.style.overflow = 'hidden';
    label.style.textOverflow = 'ellipsis';
    label.style.whiteSpace = 'nowrap';
    label.style.fontSize = '0.9rem';

    const barContainer = document.createElement('div');
    barContainer.className = 'bar-container';
    barContainer.style.flex = '1';
    barContainer.style.background = '#e8eaed'; // Light grey track
    barContainer.style.height = '24px'; // Thicker bars
    barContainer.style.borderRadius = '4px'; // Slightly rounded
    barContainer.style.margin = '0 12px';
    barContainer.style.overflow = 'hidden';
    barContainer.style.position = 'relative';

    const pct = Math.round(c.score * 100);
    const color = barColors[index % barColors.length];

    const bar = document.createElement('div');
    bar.style.width = `${pct}%`;
    bar.style.height = '100%';
    bar.style.backgroundColor = color;
    bar.style.borderRadius = '4px'; // Rounded
    bar.style.transition = 'width 0.1s linear';
    bar.style.display = 'flex';
    bar.style.alignItems = 'center';
    bar.style.justifyContent = 'flex-end';
    bar.style.paddingRight = '8px';
    bar.style.boxSizing = 'border-box';

    // Label inside bar if enough space
    const scoreLabel = document.createElement('span');
    scoreLabel.innerText = `${pct}%`;
    scoreLabel.style.color = 'white';
    scoreLabel.style.fontSize = '0.8rem';
    scoreLabel.style.fontWeight = 'bold';

    // If bar is too small, maybe show outside? For now reference shows inside
    if (pct > 10) {
      bar.appendChild(scoreLabel);
    }

    barContainer.appendChild(bar);

    // If bar is very small, we might want to put text outside, but reference shows 6% inside.
    if (pct <= 10) {
      // Just force it inside or let it clip?
      // Reference 6% is visible inside. 
      // We'll append it anyway.
      if (!bar.contains(scoreLabel)) bar.appendChild(scoreLabel);
    }

    row.appendChild(label);
    row.appendChild(barContainer);
    // row.appendChild(score); // Removing external score

    container.appendChild(row);
  });
}

function updateStatus(msg: string) {
  const el = document.getElementById('status-message');
  if (el) el.innerText = msg;
}

function updateInferenceTime(time: number) {
  const el = document.getElementById('inference-time');
  if (el) el.innerText = `Inference Time: ${time.toFixed(2)} ms`;
}

export function cleanupAudioClassifier() {
  stopRecording();
  if (worker) {
    worker.terminate(); // Terminate to stop any WASM loops etc?
    // Or post cleanup
    worker = undefined;
  }
}
